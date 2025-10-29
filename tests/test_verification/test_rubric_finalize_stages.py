"""Tests for RubricEvaluationStage and FinalizeResultStage."""

from unittest.mock import Mock, patch

from pydantic import Field

from karenina.benchmark.models import VerificationResult
from karenina.benchmark.verification.stage import VerificationContext
from karenina.benchmark.verification.stages.finalize_result import FinalizeResultStage
from karenina.benchmark.verification.stages.rubric_evaluation import RubricEvaluationStage
from karenina.schemas.answer_class import BaseAnswer
from karenina.schemas.rubric_class import MetricRubricTrait


class MockAnswer(BaseAnswer):
    """Mock answer class for testing."""

    result: int = Field(description="The result")
    correct: dict = Field(description="Correct answer")

    def verify(self) -> bool:
        """Verify the answer."""
        return self.result == 4


class TestRubricEvaluationStage:
    """Test suite for RubricEvaluationStage."""

    def test_should_run_with_rubric_and_traits(self, basic_context: VerificationContext, sample_rubric) -> None:
        """Test that stage runs when rubric has traits."""
        basic_context.rubric = sample_rubric
        basic_context.set_artifact("raw_llm_response", "The answer is 4")

        stage = RubricEvaluationStage()
        assert stage.should_run(basic_context) is True

    def test_should_not_run_without_rubric(self, basic_context: VerificationContext) -> None:
        """Test that stage skips when rubric is None."""
        basic_context.rubric = None

        stage = RubricEvaluationStage()
        assert stage.should_run(basic_context) is False

    def test_should_not_run_with_empty_rubric(self, basic_context: VerificationContext) -> None:
        """Test that stage skips when rubric has no traits."""
        from karenina.schemas.rubric_class import Rubric

        basic_context.rubric = Rubric(name="Empty Rubric", traits=[])

        stage = RubricEvaluationStage()
        assert stage.should_run(basic_context) is False

    @patch("karenina.benchmark.verification.stages.rubric_evaluation.RubricEvaluator")
    def test_rubric_evaluation_success(
        self,
        mock_evaluator_class: Mock,
        basic_context: VerificationContext,
        sample_rubric,
    ) -> None:
        """Test successful rubric evaluation."""
        basic_context.rubric = sample_rubric
        basic_context.set_artifact("raw_llm_response", "The answer is 4")

        # Mock RubricEvaluator
        mock_evaluator = Mock()
        mock_evaluator_class.return_value = mock_evaluator

        # Mock evaluation result
        mock_result = Mock()
        mock_result.overall_score = 8.5
        mock_result.trait_scores = {"Accuracy": 9, "Completeness": 8}
        mock_evaluator.evaluate_rubric.return_value = mock_result

        stage = RubricEvaluationStage()
        stage.execute(basic_context)

        # Verify evaluation was performed
        assert basic_context.error is None
        assert basic_context.has_artifact("rubric_result")

    @patch("karenina.benchmark.verification.stages.rubric_evaluation.RubricEvaluator")
    def test_metric_trait_evaluation(
        self,
        mock_evaluator_class: Mock,
        basic_context: VerificationContext,
    ) -> None:
        """Test evaluation of metric traits."""
        from karenina.schemas.rubric_class import Rubric

        # Create rubric with metric traits
        metric_trait = MetricRubricTrait(
            name="Token Count",
            description="Count of tokens in response",
            evaluation_mode="regex",
            metrics=["\\w+"],
        )
        basic_context.rubric = Rubric(metric_traits=[metric_trait])
        basic_context.set_artifact("raw_llm_response", "The answer is four")

        # Mock evaluator
        mock_evaluator = Mock()
        mock_evaluator_class.return_value = mock_evaluator
        mock_evaluator.evaluate_metric_traits.return_value = (
            {"Token Count": ["The", "answer", "is", "four"]},  # confusion lists
            {"Token Count": 4},  # metric results
        )

        stage = RubricEvaluationStage()
        stage.execute(basic_context)

        # Verify metric evaluation
        assert basic_context.has_artifact("metric_confusion_lists")
        assert basic_context.has_artifact("metric_results")
        assert basic_context.get_artifact("metric_results")["Token Count"] == 4

    @patch("karenina.benchmark.verification.stages.rubric_evaluation.RubricEvaluator")
    def test_rubric_evaluation_with_error(
        self,
        mock_evaluator_class: Mock,
        basic_context: VerificationContext,
        sample_rubric,
    ) -> None:
        """Test rubric evaluation continues even if main verification has error."""
        basic_context.rubric = sample_rubric
        basic_context.set_artifact("raw_llm_response", "The answer is 4")
        # Note: rubric evaluation runs independently of template verification errors

        # Mock evaluator
        mock_evaluator = Mock()
        mock_evaluator_class.return_value = mock_evaluator
        mock_result = Mock()
        mock_result.overall_score = 5.0
        mock_evaluator.evaluate_rubric.return_value = mock_result

        stage = RubricEvaluationStage()
        # Even though context might have errors elsewhere, rubric runs independently
        assert stage.should_run(basic_context) is True

    def test_stage_metadata(self) -> None:
        """Test stage name and artifact declarations."""
        stage = RubricEvaluationStage()

        assert stage.name == "RubricEvaluation"
        assert "raw_llm_response" in stage.requires
        assert "rubric_result" in stage.produces
        assert "metric_confusion_lists" in stage.produces
        assert "metric_results" in stage.produces


class TestFinalizeResultStage:
    """Test suite for FinalizeResultStage."""

    def test_should_run_always(self, basic_context: VerificationContext) -> None:
        """Test that finalize stage always runs."""
        stage = FinalizeResultStage()
        assert stage.should_run(basic_context) is True

    def test_finalize_with_all_fields(self, basic_context: VerificationContext) -> None:
        """Test finalization with complete successful verification."""
        # Set up all artifacts for successful run
        parsed = MockAnswer(result=4, correct={"value": 4}, question_id="test_q123")
        basic_context.set_artifact("parsed_answer", parsed)
        basic_context.set_artifact("raw_llm_response", "The answer is 4")
        basic_context.set_artifact("field_verification_result", True)
        basic_context.set_artifact("regex_match_result", True)
        basic_context.set_artifact("answering_model_str", "openai/gpt-4.1-mini")
        basic_context.set_artifact("parsing_model_str", "openai/gpt-4.1-mini")
        basic_context.set_artifact("deep_judgment_performed", False)
        basic_context.set_artifact("embedding_check_performed", False)
        basic_context.set_artifact("abstention_detected", False)

        stage = FinalizeResultStage()
        stage.execute(basic_context)
        result = basic_context.get_artifact("final_result")

        # Verify result is a VerificationResult
        assert isinstance(result, VerificationResult)
        assert result.success is True
        assert result.question_id == "test_q123"
        assert result.template_id == "test_t456"
        assert result.answering_model == "openai/gpt-4.1-mini"

    def test_finalize_with_errors(self, basic_context: VerificationContext) -> None:
        """Test finalization when errors occurred during verification."""
        # Set up error state
        basic_context.mark_error("Template validation failed")
        basic_context.set_artifact("raw_llm_response", "Some response")
        basic_context.set_artifact("answering_model_str", "openai/gpt-4.1-mini")
        basic_context.set_artifact("parsing_model_str", "openai/gpt-4.1-mini")

        stage = FinalizeResultStage()
        stage.execute(basic_context)
        result = basic_context.get_artifact("final_result")

        # Verify result reflects error
        assert isinstance(result, VerificationResult)
        assert result.success is False
        assert result.error_message == "Template validation failed"

    def test_finalize_with_partial_results(self, basic_context: VerificationContext) -> None:
        """Test finalization with partial verification results."""
        # Set up partial verification
        parsed = MockAnswer(result=5, correct={"value": 5}, question_id="test_q123")
        basic_context.set_artifact("parsed_answer", parsed)
        basic_context.set_artifact("raw_llm_response", "The answer is 5")
        basic_context.set_artifact("field_verification_result", False)
        basic_context.set_artifact("embedding_check_performed", True)
        basic_context.set_artifact("embedding_check_passed", False)
        basic_context.set_artifact("answering_model_str", "openai/gpt-4.1-mini")
        basic_context.set_artifact("parsing_model_str", "openai/gpt-4.1-mini")

        stage = FinalizeResultStage()
        stage.execute(basic_context)
        result = basic_context.get_artifact("final_result")

        # Verify result captures partial verification
        assert isinstance(result, VerificationResult)
        assert result.success is False  # Field verification failed
        assert result.parsed_answer is not None
        assert result.embedding_check_performed is True

    def test_finalize_with_rubric_results(self, basic_context: VerificationContext, sample_rubric) -> None:
        """Test finalization includes rubric evaluation results."""
        basic_context.rubric = sample_rubric

        # Set up basic verification
        parsed = MockAnswer(result=4, correct={"value": 4}, question_id="test_q123")
        basic_context.set_artifact("parsed_answer", parsed)
        basic_context.set_artifact("raw_llm_response", "The answer is 4")
        basic_context.set_artifact("field_verification_result", True)
        basic_context.set_artifact("answering_model_str", "openai/gpt-4.1-mini")
        basic_context.set_artifact("parsing_model_str", "openai/gpt-4.1-mini")

        # Set up rubric results
        mock_rubric_result = Mock()
        mock_rubric_result.overall_score = 8.5
        basic_context.set_artifact("rubric_result", mock_rubric_result)

        stage = FinalizeResultStage()
        stage.execute(basic_context)
        result = basic_context.get_artifact("final_result")

        # Verify rubric results are included
        assert isinstance(result, VerificationResult)

    def test_finalize_with_deep_judgment(self, basic_context: VerificationContext) -> None:
        """Test finalization includes deep-judgment metadata."""
        # Set up deep-judgment verification
        parsed = MockAnswer(result=4, correct={"value": 4}, question_id="test_q123")
        basic_context.set_artifact("parsed_answer", parsed)
        basic_context.set_artifact("raw_llm_response", "The answer is 4")
        basic_context.set_artifact("field_verification_result", True)
        basic_context.set_artifact("deep_judgment_performed", True)
        basic_context.set_artifact("extracted_excerpts", {"result": "4"})
        basic_context.set_artifact("deep_judgment_model_calls", 3)
        basic_context.set_artifact("answering_model_str", "openai/gpt-4.1-mini")
        basic_context.set_artifact("parsing_model_str", "openai/gpt-4.1-mini")

        stage = FinalizeResultStage()
        stage.execute(basic_context)
        result = basic_context.get_artifact("final_result")

        # Verify deep-judgment metadata is included
        assert isinstance(result, VerificationResult)
        assert result.deep_judgment_performed is True
        assert result.extracted_excerpts is not None

    def test_finalize_with_abstention(self, basic_context: VerificationContext) -> None:
        """Test finalization handles abstention detection."""
        # Set up abstention
        basic_context.set_artifact("raw_llm_response", "I cannot answer this")
        basic_context.set_artifact("abstention_detected", True)
        basic_context.set_artifact("abstention_reason", "Explicit refusal")
        basic_context.set_artifact("answering_model_str", "openai/gpt-4.1-mini")
        basic_context.set_artifact("parsing_model_str", "openai/gpt-4.1-mini")

        stage = FinalizeResultStage()
        stage.execute(basic_context)
        result = basic_context.get_artifact("final_result")

        # Verify abstention is captured
        assert isinstance(result, VerificationResult)
        assert result.abstention_detected is True
        assert result.abstention_reason == "Explicit refusal"

    def test_stage_metadata(self) -> None:
        """Test stage name and artifact declarations."""
        stage = FinalizeResultStage()

        assert stage.name == "FinalizeResult"
        assert "final_result" in stage.produces
