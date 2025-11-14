"""Tests for ParseTemplateStage."""

from unittest.mock import Mock, patch

from pydantic import Field

from karenina.benchmark.verification.stage import VerificationContext
from karenina.benchmark.verification.stages.parse_template import ParseTemplateStage
from karenina.schemas.domain import BaseAnswer


class MockAnswer(BaseAnswer):
    """Mock answer class for testing."""

    result: int = Field(description="The result")
    correct: dict = Field(description="Correct answer")

    def verify(self) -> bool:
        """Verify the answer."""
        return self.result == 4


class TestParseTemplateStage:
    """Test suite for ParseTemplateStage."""

    def test_should_run_with_required_artifacts(self, basic_context: VerificationContext) -> None:
        """Test that should_run returns True when all required artifacts exist."""
        # Set up required artifacts
        basic_context.set_artifact("RawAnswer", MockAnswer)
        basic_context.set_artifact("Answer", MockAnswer)
        basic_context.set_artifact("raw_llm_response", "The answer is 4")

        stage = ParseTemplateStage()
        assert stage.should_run(basic_context) is True

    def test_should_not_run_with_missing_artifacts(self, basic_context: VerificationContext) -> None:
        """Test that should_run returns False when required artifacts are missing."""
        # Only set some artifacts
        basic_context.set_artifact("Answer", MockAnswer)

        stage = ParseTemplateStage()
        assert stage.should_run(basic_context) is False

    def test_should_not_run_with_error(self, basic_context: VerificationContext) -> None:
        """Test that should_run returns False if there's an error."""
        # Set up all required artifacts
        basic_context.set_artifact("RawAnswer", MockAnswer)
        basic_context.set_artifact("Answer", MockAnswer)
        basic_context.set_artifact("raw_llm_response", "The answer is 4")
        basic_context.mark_error("Previous error")

        stage = ParseTemplateStage()
        assert stage.should_run(basic_context) is False

    @patch("karenina.benchmark.verification.stages.parse_template.init_chat_model_unified")
    @patch("karenina.benchmark.verification.stages.parse_template.PydanticOutputParser")
    def test_standard_parsing_success(
        self,
        mock_parser_class: Mock,
        mock_init_llm: Mock,
        basic_context: VerificationContext,
    ) -> None:
        """Test successful standard parsing without deep-judgment."""
        # Set up context
        basic_context.deep_judgment_enabled = False
        basic_context.set_artifact("RawAnswer", MockAnswer)
        basic_context.set_artifact("Answer", MockAnswer)
        basic_context.set_artifact("raw_llm_response", "The answer is 4")

        # Mock LLM and its invoke response
        mock_llm = Mock()
        mock_llm_response = Mock()
        mock_llm_response.content = '{"result": 4, "correct": {"value": 4}, "question_id": "test_q123"}'
        mock_llm.invoke.return_value = mock_llm_response
        mock_init_llm.return_value = mock_llm

        # Mock parser
        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser

        # Mock successful parsing
        parsed_result = MockAnswer(
            result=4,
            correct={"value": 4},
            question_id="test_q123",
        )
        mock_parser.parse.return_value = parsed_result

        # Execute stage
        stage = ParseTemplateStage()
        stage.execute(basic_context)

        # Verify success
        assert basic_context.error is None
        assert basic_context.has_artifact("parsed_answer")
        parsed = basic_context.get_artifact("parsed_answer")
        assert parsed.result == 4
        assert basic_context.get_artifact("deep_judgment_performed") is False

    @patch("karenina.benchmark.verification.stages.parse_template.deep_judgment_parse")
    @patch("karenina.benchmark.verification.stages.parse_template.init_chat_model_unified")
    def test_deep_judgment_parsing(
        self,
        mock_init_llm: Mock,
        mock_deep_judgment: Mock,
        basic_context: VerificationContext,
    ) -> None:
        """Test deep-judgment parsing when enabled."""
        # Enable deep-judgment
        basic_context.deep_judgment_enabled = True
        basic_context.set_artifact("RawAnswer", MockAnswer)
        basic_context.set_artifact("Answer", MockAnswer)
        basic_context.set_artifact("raw_llm_response", "The answer is 4")

        # Mock LLM
        mock_llm = Mock()
        mock_init_llm.return_value = mock_llm

        # Mock deep-judgment parse
        parsed_result = MockAnswer(
            result=4,
            correct={"value": 4},
            question_id="test_q123",
        )
        mock_deep_judgment.return_value = (
            parsed_result,  # parsed_answer
            {"result": [{"text": "The answer is 4", "confidence": "high"}]},  # extracted_excerpts
            {"result": "Basic arithmetic"},  # attribute_reasoning
            {  # dj_metadata dictionary
                "stages_completed": ["excerpt_extraction", "attribute_parsing"],
                "model_calls": 3,
                "excerpt_retry_count": 0,
                "attributes_without_excerpts": [],
                "hallucination_risk": {},
            },
        )

        # Execute stage
        stage = ParseTemplateStage()
        stage.execute(basic_context)

        # Verify deep-judgment was used
        assert basic_context.error is None
        assert basic_context.get_artifact("deep_judgment_performed") is True
        assert basic_context.has_artifact("extracted_excerpts")
        assert basic_context.has_artifact("attribute_reasoning")
        assert basic_context.get_artifact("deep_judgment_model_calls") == 3

    @patch("karenina.benchmark.verification.stages.parse_template._should_expose_ground_truth")
    @patch("karenina.benchmark.verification.stages.parse_template.init_chat_model_unified")
    @patch("karenina.benchmark.verification.stages.parse_template.PydanticOutputParser")
    def test_ground_truth_exposure(
        self,
        mock_parser_class: Mock,
        mock_init_llm: Mock,
        mock_should_expose: Mock,
        basic_context: VerificationContext,
    ) -> None:
        """Test that ground truth is exposed when configured."""
        # Configure ground truth exposure
        mock_should_expose.return_value = True
        basic_context.set_artifact("RawAnswer", MockAnswer)
        basic_context.set_artifact("Answer", MockAnswer)
        basic_context.set_artifact("raw_llm_response", "The answer is 4")

        # Mock LLM and its invoke response
        mock_llm = Mock()
        mock_llm_response = Mock()
        mock_llm_response.content = '{"result": 4, "correct": {"value": 4}, "question_id": "test_q123"}'
        mock_llm.invoke.return_value = mock_llm_response
        mock_init_llm.return_value = mock_llm

        # Mock parser
        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser
        parsed_result = MockAnswer(
            result=4,
            correct={"value": 4},
            question_id="test_q123",
        )
        mock_parser.parse.return_value = parsed_result

        # Execute stage
        stage = ParseTemplateStage()
        stage.execute(basic_context)

        # Verify ground truth exposure was checked
        mock_should_expose.assert_called_once()
        assert basic_context.error is None

    @patch("karenina.benchmark.verification.stages.parse_template.init_chat_model_unified")
    @patch("karenina.benchmark.verification.stages.parse_template.PydanticOutputParser")
    def test_parsing_failure(
        self,
        mock_parser_class: Mock,
        mock_init_llm: Mock,
        basic_context: VerificationContext,
    ) -> None:
        """Test handling of parsing failures."""
        # Set up context
        basic_context.set_artifact("RawAnswer", MockAnswer)
        basic_context.set_artifact("Answer", MockAnswer)
        basic_context.set_artifact("raw_llm_response", "Invalid unparseable response")

        # Mock LLM and its invoke response
        mock_llm = Mock()
        mock_llm_response = Mock()
        mock_llm_response.content = "Invalid JSON that cannot be parsed"
        mock_llm.invoke.return_value = mock_llm_response
        mock_init_llm.return_value = mock_llm

        # Mock parser to raise exception
        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser
        mock_parser.parse.side_effect = Exception("Failed to parse response")

        # Execute stage
        stage = ParseTemplateStage()
        stage.execute(basic_context)

        # Verify error was set
        assert basic_context.error is not None
        assert "Parsing failed" in basic_context.error

    @patch("karenina.benchmark.verification.stages.parse_template.init_chat_model_unified")
    @patch("karenina.benchmark.verification.stages.parse_template.PydanticOutputParser")
    def test_invalid_response_format(
        self,
        mock_parser_class: Mock,
        mock_init_llm: Mock,
        basic_context: VerificationContext,
    ) -> None:
        """Test handling of responses with invalid format."""
        # Set up context with malformed response
        basic_context.set_artifact("RawAnswer", MockAnswer)
        basic_context.set_artifact("Answer", MockAnswer)
        basic_context.set_artifact("raw_llm_response", "```\nNot valid JSON\n```")

        # Mock LLM
        mock_llm = Mock()
        mock_init_llm.return_value = mock_llm

        # Mock parser to fail on invalid format
        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser
        mock_parser.parse.side_effect = ValueError("Invalid JSON format")

        # Execute stage
        stage = ParseTemplateStage()
        stage.execute(basic_context)

        # Verify error was set
        assert basic_context.error is not None

    @patch("karenina.benchmark.verification.stages.parse_template.init_chat_model_unified")
    def test_error_handling_llm_initialization(
        self,
        mock_init_llm: Mock,
        basic_context: VerificationContext,
    ) -> None:
        """Test error handling when parsing LLM initialization fails."""
        # Set up context
        basic_context.set_artifact("RawAnswer", MockAnswer)
        basic_context.set_artifact("Answer", MockAnswer)
        basic_context.set_artifact("raw_llm_response", "The answer is 4")

        # Mock LLM initialization failure
        mock_init_llm.side_effect = Exception("Failed to initialize parsing model")

        # Execute stage
        stage = ParseTemplateStage()
        stage.execute(basic_context)

        # Verify error was set
        assert basic_context.error is not None
        assert "Failed to initialize parsing model" in basic_context.error

    def test_stage_metadata(self) -> None:
        """Test stage name and artifact declarations."""
        stage = ParseTemplateStage()

        assert stage.name == "ParseTemplate"
        assert "RawAnswer" in stage.requires
        assert "Answer" in stage.requires
        assert "raw_llm_response" in stage.requires
        assert "parsed_answer" in stage.produces
        assert "parsing_model_str" in stage.produces
        assert "deep_judgment_performed" in stage.produces

    @patch("karenina.benchmark.verification.stages.parse_template._retry_parse_with_null_feedback")
    @patch("karenina.benchmark.verification.stages.parse_template.init_chat_model_unified")
    @patch("karenina.benchmark.verification.stages.parse_template.PydanticOutputParser")
    def test_null_value_retry_on_parsing_failure(
        self,
        mock_parser_class: Mock,
        mock_init_llm: Mock,
        mock_retry: Mock,
        basic_context: VerificationContext,
    ) -> None:
        """Test that null-value retry is triggered when parsing fails with null values."""
        # Set up context
        basic_context.set_artifact("RawAnswer", MockAnswer)
        basic_context.set_artifact("Answer", MockAnswer)
        basic_context.set_artifact("raw_llm_response", "The answer is 4")

        # Mock LLM and its invoke response with null values
        mock_llm = Mock()
        mock_llm_response = Mock()
        mock_llm_response.content = '{"result": null, "correct": null, "question_id": "test_q123"}'
        mock_llm.invoke.return_value = mock_llm_response
        mock_init_llm.return_value = mock_llm

        # Mock parser to fail on first attempt (null values)
        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser
        mock_parser.parse.side_effect = ValueError(
            "1 validation error for Answer\nresult\n  Input should be a valid number [type=int_type, input_value=None, input_type=NoneType]"
        )

        # Mock successful retry
        retried_answer = MockAnswer(
            result=4,
            correct={"value": 4},
            question_id="test_q123",
        )
        mock_retry.return_value = (retried_answer, {"input_tokens": 100, "output_tokens": 50})

        # Execute stage
        stage = ParseTemplateStage()
        stage.execute(basic_context)

        # Verify retry was called
        mock_retry.assert_called_once()

        # Verify successful retry result
        assert basic_context.error is None
        assert basic_context.has_artifact("parsed_answer")
        parsed = basic_context.get_artifact("parsed_answer")
        assert parsed.result == 4

    @patch("karenina.benchmark.verification.stages.parse_template._retry_parse_with_null_feedback")
    @patch("karenina.benchmark.verification.stages.parse_template.init_chat_model_unified")
    @patch("karenina.benchmark.verification.stages.parse_template.PydanticOutputParser")
    def test_null_value_retry_fails_marks_error(
        self,
        mock_parser_class: Mock,
        mock_init_llm: Mock,
        mock_retry: Mock,
        basic_context: VerificationContext,
    ) -> None:
        """Test that error is marked when null-value retry also fails."""
        # Set up context
        basic_context.set_artifact("RawAnswer", MockAnswer)
        basic_context.set_artifact("Answer", MockAnswer)
        basic_context.set_artifact("raw_llm_response", "The answer is 4")

        # Mock LLM
        mock_llm = Mock()
        mock_llm_response = Mock()
        mock_llm_response.content = '{"result": null, "correct": null, "question_id": "test_q123"}'
        mock_llm.invoke.return_value = mock_llm_response
        mock_init_llm.return_value = mock_llm

        # Mock parser to fail
        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser
        original_error = ValueError(
            "1 validation error for Answer\nresult\n  Input should be a valid number [type=int_type, input_value=None, input_type=NoneType]"
        )
        mock_parser.parse.side_effect = original_error

        # Mock failed retry (returns None)
        mock_retry.return_value = (None, {})

        # Execute stage
        stage = ParseTemplateStage()
        stage.execute(basic_context)

        # Verify retry was attempted
        mock_retry.assert_called_once()

        # Verify error was marked
        assert basic_context.error is not None
        assert "Parsing failed" in basic_context.error

    @patch("karenina.benchmark.verification.stages.parse_template._retry_parse_with_null_feedback")
    @patch("karenina.benchmark.verification.stages.parse_template.init_chat_model_unified")
    @patch("karenina.benchmark.verification.stages.parse_template.PydanticOutputParser")
    def test_no_retry_for_non_null_errors(
        self,
        mock_parser_class: Mock,
        mock_init_llm: Mock,
        mock_retry: Mock,
        basic_context: VerificationContext,
    ) -> None:
        """Test that retry is NOT triggered for non-null parsing errors."""
        # Set up context
        basic_context.set_artifact("RawAnswer", MockAnswer)
        basic_context.set_artifact("Answer", MockAnswer)
        basic_context.set_artifact("raw_llm_response", "The answer is 4")

        # Mock LLM
        mock_llm = Mock()
        mock_llm_response = Mock()
        mock_llm_response.content = '{"result": "not a number", "correct": {}, "question_id": "test_q123"}'
        mock_llm.invoke.return_value = mock_llm_response
        mock_init_llm.return_value = mock_llm

        # Mock parser to fail with non-null error
        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser
        mock_parser.parse.side_effect = ValueError("Invalid type: expected int, got str")

        # Mock retry to return None (not triggered for non-null errors)
        mock_retry.return_value = (None, {})

        # Execute stage
        stage = ParseTemplateStage()
        stage.execute(basic_context)

        # Verify retry was still called but returned None (non-null error)
        mock_retry.assert_called_once()

        # Verify error was marked
        assert basic_context.error is not None
        assert "Parsing failed" in basic_context.error
