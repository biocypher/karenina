"""
End-to-end tests for evaluation mode feature (Task 6.1).

Tests all three evaluation modes:
- template_only: Template verification only (default)
- template_and_rubric: Template verification + rubric evaluation
- rubric_only: Rubric evaluation only, skip template verification
"""

from unittest.mock import MagicMock, patch

import pytest

from karenina.benchmark.verification.runner import run_single_model_verification
from karenina.schemas import ModelConfig, Rubric, RubricTrait, VerificationConfig


@pytest.fixture
def answering_model() -> ModelConfig:
    """Create test answering model."""
    return ModelConfig(
        id="test-answering",
        model_provider="test",
        model_name="test-model",
        temperature=0.0,
        interface="langchain",
        system_prompt="You are a test assistant.",
    )


@pytest.fixture
def parsing_model() -> ModelConfig:
    """Create test parsing model."""
    return ModelConfig(
        id="test-parsing",
        model_provider="test",
        model_name="test-model",
        temperature=0.0,
        interface="langchain",
        system_prompt="You are a test parser.",
    )


@pytest.fixture
def test_rubric() -> Rubric:
    """Create test rubric."""
    return Rubric(
        traits=[
            RubricTrait(
                name="Clarity",
                description="Response clarity",
                kind="score",
                min_score=1,
                max_score=5,
            ),
            RubricTrait(
                name="Completeness",
                description="Response completeness",
                kind="score",
                min_score=1,
                max_score=5,
            ),
        ]
    )


class TestTemplateOnlyMode:
    """Test template_only evaluation mode (default behavior)."""

    @patch("karenina.benchmark.verification.stages.generate_answer.init_chat_model_unified")
    @patch("karenina.benchmark.verification.stages.parse_template.init_chat_model_unified")
    def test_template_only_performs_template_verification(
        self, mock_parse_init, mock_answer_init, answering_model, parsing_model
    ):
        """Test template_only mode performs template verification, skips rubric."""
        # Mock LLM responses
        mock_answer_llm = MagicMock()
        mock_answer_llm.invoke.return_value.content = "The answer is 42"
        mock_answer_init.return_value = mock_answer_llm

        mock_parse_llm = MagicMock()
        mock_parse_llm.invoke.return_value.content = '{"answer": "42", "expected": "42"}'
        mock_parse_init.return_value = mock_parse_llm

        # Define simple template (must inherit from BaseAnswer and have verify method)
        template_code = """
from karenina.schemas.domain import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    answer: str = Field(description="The answer", default="")
    expected: str = Field(description="Expected answer", default="42")

    def verify(self):
        return self.answer == self.expected
"""

        # Run verification with template_only mode
        result = run_single_model_verification(
            question_id="test_q1",
            question_text="What is the answer?",
            template_code=template_code,
            answering_model=answering_model,
            parsing_model=parsing_model,
            rubric=None,  # No rubric
            evaluation_mode="template_only",
            run_name="test_run",
            job_id="test_job",
        )

        # Verify template verification was performed
        assert result.template_verification_performed is True
        assert result.verify_result is not None
        assert result.completed_without_errors is True

        # Verify rubric evaluation was NOT performed
        assert result.rubric_evaluation_performed is False
        assert result.verify_rubric is None


class TestTemplateAndRubricMode:
    """Test template_and_rubric evaluation mode."""

    @patch("karenina.benchmark.verification.stages.rubric_evaluation.RubricEvaluator")
    @patch("karenina.benchmark.verification.stages.generate_answer.init_chat_model_unified")
    @patch("karenina.benchmark.verification.stages.parse_template.init_chat_model_unified")
    def test_template_and_rubric_performs_both(
        self, mock_parse_init, mock_answer_init, mock_rubric_class, answering_model, parsing_model, test_rubric
    ):
        """Test template_and_rubric mode performs both template verification and rubric evaluation."""
        # Mock LLM responses
        mock_answer_llm = MagicMock()
        mock_answer_llm.invoke.return_value.content = "Clear and complete answer"
        mock_answer_init.return_value = mock_answer_llm

        mock_parse_llm = MagicMock()
        mock_parse_llm.invoke.return_value.content = '{"answer": "Clear and complete", "expected": "clear"}'
        mock_parse_init.return_value = mock_parse_llm

        # Mock RubricEvaluator class
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate_rubric.return_value = ({"Clarity": 5, "Completeness": 4}, [])
        mock_rubric_class.return_value = mock_evaluator

        template_code = """
from karenina.schemas.domain import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    answer: str = Field(description="The answer", default="")
    expected: str = Field(description="Expected answer", default="clear")

    def verify(self):
        return "clear" in self.answer.lower()
"""

        # Run with template_and_rubric mode
        result = run_single_model_verification(
            question_id="test_q3",
            question_text="What is the answer?",
            template_code=template_code,
            answering_model=answering_model,
            parsing_model=parsing_model,
            rubric=test_rubric,
            evaluation_mode="template_and_rubric",
            run_name="test_run",
            job_id="test_job",
        )

        # Verify BOTH template verification and rubric evaluation performed
        assert result.template_verification_performed is True
        assert result.verify_result is not None
        assert result.completed_without_errors is True

        assert result.rubric_evaluation_performed is True
        assert result.verify_rubric is not None
        assert "Clarity" in result.verify_rubric
        assert "Completeness" in result.verify_rubric


class TestRubricOnlyMode:
    """Test rubric_only evaluation mode."""

    @patch("karenina.benchmark.verification.stages.rubric_evaluation.RubricEvaluator")
    @patch("karenina.benchmark.verification.stages.generate_answer.init_chat_model_unified")
    def test_rubric_only_skips_template_verification(
        self, mock_answer_init, mock_rubric_class, answering_model, parsing_model, test_rubric
    ):
        """Test rubric_only mode skips template verification, only evaluates rubric."""
        # Mock LLM response (only answering model called)
        mock_answer_llm = MagicMock()
        mock_answer_llm.invoke.return_value.content = "Clear and complete answer without template structure"
        mock_answer_init.return_value = mock_answer_llm

        # Mock RubricEvaluator class
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate_rubric.return_value = ({"Clarity": 5, "Completeness": 4}, [])
        mock_rubric_class.return_value = mock_evaluator

        # Note: Template code still required but won't be validated in rubric_only mode
        template_code = "# Template validation skipped in rubric_only mode"

        # Run with rubric_only mode
        result = run_single_model_verification(
            question_id="test_q4",
            question_text="What is the answer?",
            template_code=template_code,
            answering_model=answering_model,
            parsing_model=parsing_model,
            rubric=test_rubric,
            evaluation_mode="rubric_only",
            run_name="test_run",
            job_id="test_job",
        )

        # Verify template verification was NOT performed
        assert result.template_verification_performed is False
        assert result.verify_result is None  # No template verification result
        assert result.completed_without_errors is True

        # Verify rubric evaluation WAS performed
        assert result.rubric_evaluation_performed is True
        assert result.verify_rubric is not None
        assert "Clarity" in result.verify_rubric
        assert "Completeness" in result.verify_rubric

    @patch("karenina.benchmark.verification.stages.rubric_evaluation.RubricEvaluator")
    @patch("karenina.benchmark.verification.stages.generate_answer.init_chat_model_unified")
    def test_rubric_only_with_abstention_check(
        self, mock_answer_init, mock_rubric_class, answering_model, parsing_model, test_rubric
    ):
        """Test rubric_only mode with abstention detection enabled."""
        # Mock LLM response
        mock_answer_llm = MagicMock()
        mock_answer_llm.invoke.return_value.content = "I cannot answer that question."
        mock_answer_init.return_value = mock_answer_llm

        # Mock RubricEvaluator class
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate_rubric.return_value = {"Clarity": 1, "Completeness": 1}
        mock_rubric_class.return_value = mock_evaluator

        template_code = "# Skipped in rubric_only mode"

        # Run with rubric_only mode and abstention enabled
        with patch("karenina.benchmark.verification.stages.abstention_check.detect_abstention") as mock_abstention:
            mock_abstention.return_value = (True, True, "Model refused to answer")

            result = run_single_model_verification(
                question_id="test_q5",
                question_text="What is the answer?",
                template_code=template_code,
                answering_model=answering_model,
                parsing_model=parsing_model,
                rubric=test_rubric,
                evaluation_mode="rubric_only",
                abstention_enabled=True,
                run_name="test_run",
                job_id="test_job",
            )

        # Verify template verification was NOT performed
        assert result.template_verification_performed is False

        # In rubric_only mode with abstention, verify_result is set to False by abstention stage
        assert result.verify_result is False  # Abstention overrides to False

        # Verify rubric evaluation WAS performed
        assert result.rubric_evaluation_performed is True
        assert result.verify_rubric is not None

        # Verify abstention was detected and override applied
        assert result.abstention_detected is True
        assert result.abstention_check_performed is True
        assert result.abstention_override_applied is True


class TestEvaluationModeValidation:
    """Test that invalid evaluation mode configurations are rejected."""

    def test_rubric_only_without_rubric_raises_error(self, answering_model, parsing_model):
        """Test that rubric_only mode without rubric raises validation error."""
        with pytest.raises(ValueError, match="evaluation_mode='rubric_only' requires rubric_enabled=True"):
            VerificationConfig(
                answering_models=[answering_model],
                parsing_models=[parsing_model],
                evaluation_mode="rubric_only",
                rubric_enabled=False,  # Invalid: rubric_only requires rubric
            )

    def test_template_and_rubric_without_rubric_raises_error(self, answering_model, parsing_model):
        """Test that template_and_rubric mode without rubric raises validation error."""
        with pytest.raises(ValueError, match="evaluation_mode='template_and_rubric' requires rubric_enabled=True"):
            VerificationConfig(
                answering_models=[answering_model],
                parsing_models=[parsing_model],
                evaluation_mode="template_and_rubric",
                rubric_enabled=False,  # Invalid: template_and_rubric requires rubric
            )

    def test_template_only_with_rubric_raises_error(self, answering_model, parsing_model):
        """Test that template_only mode with rubric enabled raises validation error."""
        with pytest.raises(
            ValueError, match="evaluation_mode='template_only' is incompatible with rubric_enabled=True"
        ):
            VerificationConfig(
                answering_models=[answering_model],
                parsing_models=[parsing_model],
                evaluation_mode="template_only",
                rubric_enabled=True,  # Invalid: template_only incompatible with rubric
            )
