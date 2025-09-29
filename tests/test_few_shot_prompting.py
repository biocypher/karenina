"""Tests for few-shot prompting functionality."""

from unittest.mock import Mock, patch

import pytest

from karenina.benchmark.models import ModelConfig, VerificationConfig
from karenina.benchmark.verification.runner import _construct_few_shot_prompt
from karenina.schemas.question_class import Question


class TestFewShotPromptConstruction:
    """Test the few-shot prompt construction logic."""

    def test_construct_few_shot_prompt_disabled(self) -> None:
        """Test that when few-shot is disabled, original question is returned."""
        question_text = "What is 2 + 2?"
        examples = [
            {"question": "What is 1 + 1?", "answer": "2"},
            {"question": "What is 3 + 3?", "answer": "6"},
        ]

        result = _construct_few_shot_prompt(question_text, examples, few_shot_enabled=False)

        assert result == question_text

    def test_construct_few_shot_prompt_no_examples(self) -> None:
        """Test that when no examples are provided, original question is returned."""
        question_text = "What is 2 + 2?"

        result = _construct_few_shot_prompt(question_text, None, few_shot_enabled=True)

        assert result == question_text

    def test_construct_few_shot_prompt_empty_examples(self) -> None:
        """Test that when empty examples list is provided, original question is returned."""
        question_text = "What is 2 + 2?"
        examples = []

        result = _construct_few_shot_prompt(question_text, examples, few_shot_enabled=True)

        assert result == question_text

    def test_construct_few_shot_prompt_single_example(self) -> None:
        """Test few-shot prompt construction with a single example."""
        question_text = "What is 2 + 2?"
        examples = [{"question": "What is 1 + 1?", "answer": "2"}]

        result = _construct_few_shot_prompt(question_text, examples, few_shot_enabled=True)

        expected = "Question: What is 1 + 1?\nAnswer: 2\n\nQuestion: What is 2 + 2?\nAnswer:"
        assert result == expected

    def test_construct_few_shot_prompt_multiple_examples(self) -> None:
        """Test few-shot prompt construction with multiple examples."""
        question_text = "What is 2 + 2?"
        examples = [
            {"question": "What is 1 + 1?", "answer": "2"},
            {"question": "What is 3 + 3?", "answer": "6"},
        ]

        result = _construct_few_shot_prompt(question_text, examples, few_shot_enabled=True)

        expected = (
            "Question: What is 1 + 1?\n"
            "Answer: 2\n\n"
            "Question: What is 3 + 3?\n"
            "Answer: 6\n\n"
            "Question: What is 2 + 2?\n"
            "Answer:"
        )
        assert result == expected

    def test_construct_few_shot_prompt_malformed_examples(self) -> None:
        """Test that malformed examples are skipped."""
        question_text = "What is 2 + 2?"
        examples = [
            {"question": "What is 1 + 1?", "answer": "2"},  # Valid
            {"query": "What is 3 + 3?", "response": "6"},  # Invalid keys
            {"question": "What is 4 + 4?", "answer": "8"},  # Valid
        ]

        result = _construct_few_shot_prompt(question_text, examples, few_shot_enabled=True)

        expected = (
            "Question: What is 1 + 1?\n"
            "Answer: 2\n\n"
            "Question: What is 4 + 4?\n"
            "Answer: 8\n\n"
            "Question: What is 2 + 2?\n"
            "Answer:"
        )
        assert result == expected


class TestQuestionSchemaWithFewShot:
    """Test Question schema with few-shot examples."""

    def test_question_with_few_shot_examples(self) -> None:
        """Test creating a Question with few-shot examples."""
        question = Question(
            question="What is the capital of France?",
            raw_answer="Paris",
            few_shot_examples=[
                {"question": "What is the capital of Germany?", "answer": "Berlin"},
                {"question": "What is the capital of Italy?", "answer": "Rome"},
            ],
        )

        assert question.question == "What is the capital of France?"
        assert question.raw_answer == "Paris"
        assert len(question.few_shot_examples) == 2
        assert question.few_shot_examples[0]["question"] == "What is the capital of Germany?"
        assert question.few_shot_examples[0]["answer"] == "Berlin"

    def test_question_without_few_shot_examples(self) -> None:
        """Test creating a Question without few-shot examples."""
        question = Question(
            question="What is the capital of France?",
            raw_answer="Paris",
        )

        assert question.question == "What is the capital of France?"
        assert question.raw_answer == "Paris"
        assert question.few_shot_examples is None

    def test_question_id_generation_with_few_shot(self) -> None:
        """Test that question ID generation is not affected by few-shot examples."""
        question_without_examples = Question(
            question="What is the capital of France?",
            raw_answer="Paris",
        )

        question_with_examples = Question(
            question="What is the capital of France?",
            raw_answer="Paris",
            few_shot_examples=[
                {"question": "What is the capital of Germany?", "answer": "Berlin"},
            ],
        )

        # ID should be the same since it's based only on question text
        assert question_without_examples.id == question_with_examples.id


class TestVerificationConfigWithFewShot:
    """Test VerificationConfig with few-shot settings."""

    def test_verification_config_default_few_shot_disabled(self) -> None:
        """Test that few-shot is disabled by default."""
        config = VerificationConfig(
            answering_models=[self._create_test_model("answering")],
            parsing_models=[self._create_test_model("parsing")],
        )

        assert config.is_few_shot_enabled() is False
        assert config.get_few_shot_config() is None

    def test_verification_config_with_few_shot_enabled(self) -> None:
        """Test creating VerificationConfig with few-shot enabled."""
        config = VerificationConfig(
            answering_models=[self._create_test_model("answering")],
            parsing_models=[self._create_test_model("parsing")],
            few_shot_enabled=True,
            few_shot_mode="k-shot",
            few_shot_k=5,
        )

        assert config.few_shot_enabled is True
        assert config.few_shot_mode == "k-shot"
        assert config.few_shot_k == 5

    def test_verification_config_few_shot_validation_k_shot_positive(self) -> None:
        """Test that k-shot validation requires positive k value."""
        with pytest.raises(ValueError, match="Global few-shot k value must be at least 1"):
            VerificationConfig(
                answering_models=[self._create_test_model("answering")],
                parsing_models=[self._create_test_model("parsing")],
                few_shot_enabled=True,
                few_shot_mode="k-shot",
                few_shot_k=0,
            )

    def test_verification_config_few_shot_validation_k_shot_negative(self) -> None:
        """Test that k-shot validation rejects negative k values."""
        with pytest.raises(ValueError, match="Global few-shot k value must be at least 1"):
            VerificationConfig(
                answering_models=[self._create_test_model("answering")],
                parsing_models=[self._create_test_model("parsing")],
                few_shot_enabled=True,
                few_shot_mode="k-shot",
                few_shot_k=-1,
            )

    def test_verification_config_few_shot_validation_all_mode_no_k_required(self) -> None:
        """Test that 'all' mode doesn't require specific k validation."""
        config = VerificationConfig(
            answering_models=[self._create_test_model("answering")],
            parsing_models=[self._create_test_model("parsing")],
            few_shot_enabled=True,
            few_shot_mode="all",
            few_shot_k=0,  # Should be ignored for 'all' mode
        )

        assert config.few_shot_enabled is True
        assert config.few_shot_mode == "all"
        # k value is still stored but not validated for 'all' mode

    def _create_test_model(self, model_type: str) -> ModelConfig:
        """Helper to create a test model configuration."""
        return ModelConfig(
            id=f"test-{model_type}",
            model_provider="test",
            model_name="test-model",
            temperature=0.1,
            interface="langchain",
            system_prompt=f"Test {model_type} prompt",
        )


class TestFewShotIntegrationInVerification:
    """Integration tests for few-shot prompting in verification."""

    @patch("karenina.benchmark.verification.runner.init_chat_model_unified")
    def test_few_shot_examples_passed_to_llm(self, mock_init_chat_model) -> None:
        """Test that few-shot examples are included in LLM messages when enabled."""
        from karenina.benchmark.verification.runner import run_single_model_verification

        # Mock the LLM
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "4"
        mock_llm.invoke.return_value = mock_response
        mock_init_chat_model.return_value = mock_llm

        # Create test models
        answering_model = ModelConfig(
            id="test-answering",
            model_provider="test",
            model_name="test-model",
            temperature=0.1,
            interface="langchain",
            system_prompt="Test answering prompt",
        )

        parsing_model = ModelConfig(
            id="test-parsing",
            model_provider="test",
            model_name="test-model",
            temperature=0.1,
            interface="langchain",
            system_prompt="Test parsing prompt",
        )

        # Test data
        question_id = "test-question-id"
        question_text = "What is 2 + 2?"
        template_code = """
class Answer(BaseModel):
    answer: int

    def model_post_init(self, __context):
        self.id = "test-question-id"
        self.correct = 4

    def verify(self) -> bool:
        return self.answer == self.correct
        """

        few_shot_examples = [
            {"question": "What is 1 + 1?", "answer": "2"},
            {"question": "What is 3 + 3?", "answer": "6"},
        ]

        # Mock the validation and injection functions
        with (
            patch("karenina.benchmark.verification.runner.validate_answer_template") as mock_validate,
            patch("karenina.benchmark.verification.runner.inject_question_id_into_answer_class") as mock_inject,
        ):
            # Configure mocks
            mock_answer_class = Mock()
            mock_answer_class.verify.return_value = True
            mock_answer_class.verify_regex.return_value = {"success": True, "results": {}, "details": {}}
            mock_answer_class.model_dump.return_value = {"answer": 4}
            mock_validate.return_value = (True, None, mock_answer_class)
            mock_inject.return_value = mock_answer_class

            # Also mock the parser creation and parsing
            with patch("karenina.benchmark.verification.runner.PydanticOutputParser") as mock_parser_class:
                mock_parser = Mock()
                mock_parser.parse.return_value = mock_answer_class
                mock_parser.get_format_instructions.return_value = "Format instructions"
                mock_parser_class.return_value = mock_parser

                # Run verification with few-shot enabled
                result = run_single_model_verification(
                    question_id=question_id,
                    question_text=question_text,
                    template_code=template_code,
                    answering_model=answering_model,
                    parsing_model=parsing_model,
                    few_shot_examples=few_shot_examples,
                    few_shot_enabled=True,
                )

        # Check that the LLM was called with the few-shot prompt
        assert mock_llm.invoke.call_count == 2  # Once for answering, once for parsing
        answering_call_args = mock_llm.invoke.call_args_list[0][0][0]  # First call, first arg (messages)

        # Check that the human message contains the few-shot examples
        human_message = answering_call_args[1]  # Second message should be the human message
        assert "Question: What is 1 + 1?" in human_message.content
        assert "Answer: 2" in human_message.content
        assert "Question: What is 3 + 3?" in human_message.content
        assert "Answer: 6" in human_message.content
        assert "Question: What is 2 + 2?" in human_message.content

        # Verify the result is successful
        assert result.success is True

    @patch("karenina.benchmark.verification.runner.init_chat_model_unified")
    def test_few_shot_disabled_uses_original_question(self, mock_init_chat_model) -> None:
        """Test that when few-shot is disabled, the original question text is used."""
        from karenina.benchmark.verification.runner import run_single_model_verification

        # Mock the LLM
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "4"
        mock_llm.invoke.return_value = mock_response
        mock_init_chat_model.return_value = mock_llm

        # Create test models
        answering_model = ModelConfig(
            id="test-answering",
            model_provider="test",
            model_name="test-model",
            temperature=0.1,
            interface="langchain",
            system_prompt="Test answering prompt",
        )

        parsing_model = ModelConfig(
            id="test-parsing",
            model_provider="test",
            model_name="test-model",
            temperature=0.1,
            interface="langchain",
            system_prompt="Test parsing prompt",
        )

        # Test data
        question_id = "test-question-id"
        question_text = "What is 2 + 2?"
        template_code = """
class Answer(BaseModel):
    answer: int

    def model_post_init(self, __context):
        self.id = "test-question-id"
        self.correct = 4

    def verify(self) -> bool:
        return self.answer == self.correct
        """

        few_shot_examples = [
            {"question": "What is 1 + 1?", "answer": "2"},
        ]

        # Mock the validation and injection functions
        with (
            patch("karenina.benchmark.verification.runner.validate_answer_template") as mock_validate,
            patch("karenina.benchmark.verification.runner.inject_question_id_into_answer_class") as mock_inject,
        ):
            # Configure mocks
            mock_answer_class = Mock()
            mock_answer_class.verify.return_value = True
            mock_answer_class.verify_regex.return_value = {"success": True, "results": {}, "details": {}}
            mock_answer_class.model_dump.return_value = {"answer": 4}
            mock_validate.return_value = (True, None, mock_answer_class)
            mock_inject.return_value = mock_answer_class

            # Also mock the parser creation and parsing
            with patch("karenina.benchmark.verification.runner.PydanticOutputParser") as mock_parser_class:
                mock_parser = Mock()
                mock_parser.parse.return_value = mock_answer_class
                mock_parser.get_format_instructions.return_value = "Format instructions"
                mock_parser_class.return_value = mock_parser

                # Run verification with few-shot disabled (default)
                result = run_single_model_verification(
                    question_id=question_id,
                    question_text=question_text,
                    template_code=template_code,
                    answering_model=answering_model,
                    parsing_model=parsing_model,
                    few_shot_examples=few_shot_examples,
                    few_shot_enabled=False,
                )

        # Check that the LLM was called with the original question text
        answering_call_args = mock_llm.invoke.call_args_list[0][0][0]  # First call, first arg (messages)
        human_message = answering_call_args[1]  # Second message should be the human message

        # Should only contain the original question, no examples
        assert human_message.content == question_text
        assert "What is 1 + 1?" not in human_message.content
        assert "Answer: 2" not in human_message.content

        # Verify the result is successful
        assert result.success is True
