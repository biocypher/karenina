"""Tests for ParseTemplateStage."""

from unittest.mock import Mock, patch

from pydantic import Field

from karenina.benchmark.verification.evaluators.template_evaluator import ParseResult
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

    @patch("karenina.benchmark.verification.stages.parse_template.TemplateEvaluator")
    def test_standard_parsing_success(
        self,
        mock_evaluator_class: Mock,
        basic_context: VerificationContext,
    ) -> None:
        """Test successful standard parsing without deep-judgment."""
        # Set up context
        basic_context.deep_judgment_enabled = False
        basic_context.set_artifact("RawAnswer", MockAnswer)
        basic_context.set_artifact("Answer", MockAnswer)
        basic_context.set_artifact("raw_llm_response", "The answer is 4")

        # Mock evaluator
        mock_evaluator = Mock()
        mock_evaluator.model_str = "openai/gpt-4"
        mock_evaluator_class.return_value = mock_evaluator

        # Mock successful parsing result
        parsed_answer = MockAnswer(
            result=4,
            correct={"value": 4},
            question_id="test_q123",
        )
        mock_parse_result = ParseResult(
            parsed_answer=parsed_answer,
            success=True,
            deep_judgment_performed=False,
        )
        mock_evaluator.parse_response.return_value = mock_parse_result

        # Execute stage
        stage = ParseTemplateStage()
        stage.execute(basic_context)

        # Verify success
        assert basic_context.error is None
        assert basic_context.has_artifact("parsed_answer")
        parsed = basic_context.get_artifact("parsed_answer")
        assert parsed.result == 4
        assert basic_context.get_artifact("deep_judgment_performed") is False

    @patch("karenina.benchmark.verification.stages.parse_template.TemplateEvaluator")
    def test_deep_judgment_parsing(
        self,
        mock_evaluator_class: Mock,
        basic_context: VerificationContext,
    ) -> None:
        """Test deep-judgment parsing when enabled."""
        # Enable deep-judgment
        basic_context.deep_judgment_enabled = True
        basic_context.set_artifact("RawAnswer", MockAnswer)
        basic_context.set_artifact("Answer", MockAnswer)
        basic_context.set_artifact("raw_llm_response", "The answer is 4")

        # Mock evaluator
        mock_evaluator = Mock()
        mock_evaluator.model_str = "openai/gpt-4"
        mock_evaluator_class.return_value = mock_evaluator

        # Mock deep-judgment parse result
        parsed_answer = MockAnswer(
            result=4,
            correct={"value": 4},
            question_id="test_q123",
        )
        mock_parse_result = ParseResult(
            parsed_answer=parsed_answer,
            success=True,
            deep_judgment_performed=True,
            extracted_excerpts={"result": [{"text": "The answer is 4", "confidence": "high"}]},
            attribute_reasoning={"result": "Basic arithmetic"},
            deep_judgment_stages_completed=["excerpt_extraction", "attribute_parsing"],
            deep_judgment_model_calls=3,
            deep_judgment_excerpt_retry_count=0,
            attributes_without_excerpts=[],
        )
        mock_evaluator.parse_response.return_value = mock_parse_result

        # Execute stage
        stage = ParseTemplateStage()
        stage.execute(basic_context)

        # Verify deep-judgment was used
        assert basic_context.error is None
        assert basic_context.get_artifact("deep_judgment_performed") is True
        assert basic_context.has_artifact("extracted_excerpts")
        assert basic_context.has_artifact("attribute_reasoning")
        assert basic_context.get_artifact("deep_judgment_model_calls") == 3

    @patch("karenina.benchmark.verification.stages.parse_template.TemplateEvaluator")
    def test_ground_truth_exposure(
        self,
        mock_evaluator_class: Mock,
        basic_context: VerificationContext,
    ) -> None:
        """Test that ground truth is exposed when configured."""
        # Set up context
        basic_context.set_artifact("RawAnswer", MockAnswer)
        basic_context.set_artifact("Answer", MockAnswer)
        basic_context.set_artifact("raw_llm_response", "The answer is 4")

        # Mock evaluator
        mock_evaluator = Mock()
        mock_evaluator.model_str = "openai/gpt-4"
        mock_evaluator_class.return_value = mock_evaluator

        # Mock successful parsing result
        parsed_answer = MockAnswer(
            result=4,
            correct={"value": 4},
            question_id="test_q123",
        )
        mock_parse_result = ParseResult(
            parsed_answer=parsed_answer,
            success=True,
            deep_judgment_performed=False,
        )
        mock_evaluator.parse_response.return_value = mock_parse_result

        # Execute stage
        stage = ParseTemplateStage()
        stage.execute(basic_context)

        # Verify evaluator was called (ground truth exposure is internal to evaluator now)
        mock_evaluator.parse_response.assert_called_once()
        assert basic_context.error is None

    @patch("karenina.benchmark.verification.stages.parse_template.TemplateEvaluator")
    def test_parsing_failure(
        self,
        mock_evaluator_class: Mock,
        basic_context: VerificationContext,
    ) -> None:
        """Test handling of parsing failures."""
        # Set up context
        basic_context.set_artifact("RawAnswer", MockAnswer)
        basic_context.set_artifact("Answer", MockAnswer)
        basic_context.set_artifact("raw_llm_response", "Invalid unparseable response")

        # Mock evaluator
        mock_evaluator = Mock()
        mock_evaluator.model_str = "openai/gpt-4"
        mock_evaluator_class.return_value = mock_evaluator

        # Mock failed parsing result
        mock_parse_result = ParseResult(
            success=False,
            error="Failed to parse response",
        )
        mock_evaluator.parse_response.return_value = mock_parse_result

        # Execute stage
        stage = ParseTemplateStage()
        stage.execute(basic_context)

        # Verify error was set
        assert basic_context.error is not None
        assert "Failed to parse response" in basic_context.error

    @patch("karenina.benchmark.verification.stages.parse_template.TemplateEvaluator")
    def test_invalid_response_format(
        self,
        mock_evaluator_class: Mock,
        basic_context: VerificationContext,
    ) -> None:
        """Test handling of responses with invalid format."""
        # Set up context with malformed response
        basic_context.set_artifact("RawAnswer", MockAnswer)
        basic_context.set_artifact("Answer", MockAnswer)
        basic_context.set_artifact("raw_llm_response", "```\nNot valid JSON\n```")

        # Mock evaluator
        mock_evaluator = Mock()
        mock_evaluator.model_str = "openai/gpt-4"
        mock_evaluator_class.return_value = mock_evaluator

        # Mock failed parsing result
        mock_parse_result = ParseResult(
            success=False,
            error="Invalid JSON format",
        )
        mock_evaluator.parse_response.return_value = mock_parse_result

        # Execute stage
        stage = ParseTemplateStage()
        stage.execute(basic_context)

        # Verify error was set
        assert basic_context.error is not None

    @patch("karenina.benchmark.verification.stages.parse_template.TemplateEvaluator")
    def test_error_handling_llm_initialization(
        self,
        mock_evaluator_class: Mock,
        basic_context: VerificationContext,
    ) -> None:
        """Test error handling when TemplateEvaluator initialization fails."""
        # Set up context
        basic_context.set_artifact("RawAnswer", MockAnswer)
        basic_context.set_artifact("Answer", MockAnswer)
        basic_context.set_artifact("raw_llm_response", "The answer is 4")

        # Mock evaluator initialization failure
        mock_evaluator_class.side_effect = RuntimeError("Failed to initialize LLM for template evaluation")

        # Execute stage
        stage = ParseTemplateStage()
        stage.execute(basic_context)

        # Verify error was set
        assert basic_context.error is not None
        assert "Failed to create TemplateEvaluator" in basic_context.error

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
        assert "template_evaluator" in stage.produces  # New artifact

    @patch("karenina.benchmark.verification.stages.parse_template.TemplateEvaluator")
    def test_null_value_retry_on_parsing_failure(
        self,
        mock_evaluator_class: Mock,
        basic_context: VerificationContext,
    ) -> None:
        """Test that null-value retry is handled by evaluator when parsing fails with null values."""
        # Set up context
        basic_context.set_artifact("RawAnswer", MockAnswer)
        basic_context.set_artifact("Answer", MockAnswer)
        basic_context.set_artifact("raw_llm_response", "The answer is 4")

        # Mock evaluator
        mock_evaluator = Mock()
        mock_evaluator.model_str = "openai/gpt-4"
        mock_evaluator_class.return_value = mock_evaluator

        # Mock successful parsing result (after internal retry)
        parsed_answer = MockAnswer(
            result=4,
            correct={"value": 4},
            question_id="test_q123",
        )
        mock_parse_result = ParseResult(
            parsed_answer=parsed_answer,
            success=True,
            deep_judgment_performed=False,
            usage_metadata_list=[],  # Empty list - usage tracking not tested here
        )
        mock_evaluator.parse_response.return_value = mock_parse_result

        # Execute stage
        stage = ParseTemplateStage()
        stage.execute(basic_context)

        # Verify successful result (retry is internal to evaluator now)
        assert basic_context.error is None
        assert basic_context.has_artifact("parsed_answer")
        parsed = basic_context.get_artifact("parsed_answer")
        assert parsed.result == 4

    @patch("karenina.benchmark.verification.stages.parse_template.TemplateEvaluator")
    def test_null_value_retry_fails_marks_error(
        self,
        mock_evaluator_class: Mock,
        basic_context: VerificationContext,
    ) -> None:
        """Test that error is marked when null-value retry also fails."""
        # Set up context
        basic_context.set_artifact("RawAnswer", MockAnswer)
        basic_context.set_artifact("Answer", MockAnswer)
        basic_context.set_artifact("raw_llm_response", "The answer is 4")

        # Mock evaluator
        mock_evaluator = Mock()
        mock_evaluator.model_str = "openai/gpt-4"
        mock_evaluator_class.return_value = mock_evaluator

        # Mock failed parsing result (retry also failed)
        mock_parse_result = ParseResult(
            success=False,
            error="Parsing failed: 1 validation error for Answer - result field has null value",
        )
        mock_evaluator.parse_response.return_value = mock_parse_result

        # Execute stage
        stage = ParseTemplateStage()
        stage.execute(basic_context)

        # Verify error was marked
        assert basic_context.error is not None
        assert "Parsing failed" in basic_context.error

    @patch("karenina.benchmark.verification.stages.parse_template.TemplateEvaluator")
    def test_no_retry_for_non_null_errors(
        self,
        mock_evaluator_class: Mock,
        basic_context: VerificationContext,
    ) -> None:
        """Test that non-null parsing errors are handled by evaluator."""
        # Set up context
        basic_context.set_artifact("RawAnswer", MockAnswer)
        basic_context.set_artifact("Answer", MockAnswer)
        basic_context.set_artifact("raw_llm_response", "The answer is 4")

        # Mock evaluator
        mock_evaluator = Mock()
        mock_evaluator.model_str = "openai/gpt-4"
        mock_evaluator_class.return_value = mock_evaluator

        # Mock failed parsing result
        mock_parse_result = ParseResult(
            success=False,
            error="Parsing failed: Invalid type - expected int, got str",
        )
        mock_evaluator.parse_response.return_value = mock_parse_result

        # Execute stage
        stage = ParseTemplateStage()
        stage.execute(basic_context)

        # Verify error was marked
        assert basic_context.error is not None
        assert "Parsing failed" in basic_context.error
