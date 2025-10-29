"""Tests for VerifyTemplateStage, EmbeddingCheckStage, AbstentionCheckStage, and DeepJudgmentAutoFailStage."""

from unittest.mock import Mock, patch

from pydantic import Field

from karenina.benchmark.verification.stage import VerificationContext
from karenina.benchmark.verification.stages.abstention_check import AbstentionCheckStage
from karenina.benchmark.verification.stages.deep_judgment_autofail import DeepJudgmentAutoFailStage
from karenina.benchmark.verification.stages.embedding_check import EmbeddingCheckStage
from karenina.benchmark.verification.stages.verify_template import VerifyTemplateStage
from karenina.schemas.answer_class import BaseAnswer


class MockAnswer(BaseAnswer):
    """Mock answer class for testing."""

    result: int = Field(description="The result")
    correct: dict = Field(description="Correct answer")

    def verify(self) -> bool:
        """Verify the answer."""
        return self.result == 4


class TestVerifyTemplateStage:
    """Test suite for VerifyTemplateStage."""

    def test_should_run_with_parsed_answer(self, basic_context: VerificationContext) -> None:
        """Test that should_run returns True when parsed_answer and raw_llm_response exist."""
        parsed = MockAnswer(result=4, correct={"value": 4}, question_id="test_q123")
        basic_context.set_artifact("parsed_answer", parsed)
        basic_context.set_artifact("raw_llm_response", "The answer is 4")

        stage = VerifyTemplateStage()
        assert stage.should_run(basic_context) is True

    def test_should_not_run_without_parsed_answer(self, basic_context: VerificationContext) -> None:
        """Test that should_run returns False without parsed_answer."""
        stage = VerifyTemplateStage()
        assert stage.should_run(basic_context) is False

    def test_field_verification_success(self, basic_context: VerificationContext) -> None:
        """Test successful field verification."""
        # Create parsed answer that will pass verify()
        parsed = MockAnswer(result=4, correct={"value": 4}, question_id="test_q123")
        basic_context.set_artifact("parsed_answer", parsed)
        basic_context.set_artifact("raw_llm_response", "The answer is 4")

        stage = VerifyTemplateStage()
        stage.execute(basic_context)

        # Should not have error
        assert basic_context.error is None
        # Should have verification result
        assert basic_context.has_artifact("field_verification_result")
        assert basic_context.get_artifact("field_verification_result") is True

    def test_field_verification_failure(self, basic_context: VerificationContext) -> None:
        """Test field verification failure."""
        # Create parsed answer that will fail verify()
        parsed = MockAnswer(result=5, correct={"value": 5}, question_id="test_q123")
        basic_context.set_artifact("parsed_answer", parsed)
        basic_context.set_artifact("raw_llm_response", "The answer is 5")

        stage = VerifyTemplateStage()
        stage.execute(basic_context)

        # Should not have critical error, just verification failure
        assert basic_context.error is None
        # Should have verification result = False
        assert basic_context.get_artifact("field_verification_result") is False

    def test_regex_validation_combined(self, basic_context: VerificationContext) -> None:
        """Test regex validation combined with field verification."""
        # Set up keywords for regex validation
        basic_context.keywords = ["special", "keyword"]

        parsed = MockAnswer(result=4, correct={"value": 4}, question_id="test_q123")
        basic_context.set_artifact("parsed_answer", parsed)
        basic_context.set_artifact("raw_llm_response", "The answer includes the special keyword")

        stage = VerifyTemplateStage()
        stage.execute(basic_context)

        # Should have both field and regex results
        assert basic_context.has_artifact("field_verification_result")
        assert basic_context.has_artifact("regex_verification_results")

    def test_stage_metadata(self) -> None:
        """Test stage name and artifact declarations."""
        stage = VerifyTemplateStage()

        assert stage.name == "VerifyTemplate"
        assert "parsed_answer" in stage.requires
        assert "field_verification_result" in stage.produces
        assert "regex_verification_results" in stage.produces


class TestEmbeddingCheckStage:
    """Test suite for EmbeddingCheckStage."""

    def test_should_run_after_field_verification_failure(self, basic_context: VerificationContext) -> None:
        """Test that embedding check runs when field verification fails."""
        parsed = MockAnswer(result=5, correct={"value": 5}, question_id="test_q123")
        basic_context.set_artifact("parsed_answer", parsed)
        basic_context.set_artifact("field_verification_result", False)
        basic_context.set_artifact("raw_llm_response", "The answer is 5")

        stage = EmbeddingCheckStage()
        assert stage.should_run(basic_context) is True

    def test_should_not_run_after_field_verification_success(self, basic_context: VerificationContext) -> None:
        """Test that embedding check skips when field verification succeeds."""
        basic_context.set_artifact("field_verification_result", True)

        stage = EmbeddingCheckStage()
        assert stage.should_run(basic_context) is False

    @patch("karenina.benchmark.verification.stages.embedding_check.perform_embedding_check")
    def test_embedding_override_success(
        self,
        mock_perform_check: Mock,
        basic_context: VerificationContext,
    ) -> None:
        """Test that embedding check can override field verification failure."""
        # Set up failed field verification
        parsed = MockAnswer(result=5, correct={"value": 5}, question_id="test_q123")
        basic_context.set_artifact("parsed_answer", parsed)
        basic_context.set_artifact("field_verification_result", False)
        basic_context.set_artifact("raw_llm_response", "The answer is basically 4")
        # Embedding check needs regex results to recalculate overall verification
        basic_context.set_artifact("regex_verification_results", {"success": True})

        # Mock perform_embedding_check to return success
        # Returns: (should_override, similarity_score, embedding_model, embedding_performed)
        mock_perform_check.return_value = (True, 0.92, "all-MiniLM-L6-v2", True)

        stage = EmbeddingCheckStage()
        stage.execute(basic_context)

        # Should have embedding check result
        assert basic_context.has_artifact("embedding_check_performed")
        assert basic_context.get_artifact("embedding_check_performed") is True
        assert basic_context.get_artifact("embedding_override_applied") is True

    @patch("karenina.benchmark.verification.stages.embedding_check.perform_embedding_check")
    def test_embedding_check_disabled(
        self,
        mock_perform_check: Mock,
        basic_context: VerificationContext,
    ) -> None:
        """Test that embedding check respects disabled configuration."""
        # Set up context
        parsed = MockAnswer(result=5, correct={"value": 5}, question_id="test_q123")
        basic_context.set_artifact("parsed_answer", parsed)
        basic_context.set_artifact("field_verification_result", False)

        # Mock perform_embedding_check to return disabled state
        mock_perform_check.return_value = (False, None, None, False)

        stage = EmbeddingCheckStage()
        stage.execute(basic_context)

        # Should not perform embedding check
        assert basic_context.get_artifact("embedding_check_performed") is False

    def test_stage_metadata(self) -> None:
        """Test stage name and artifact declarations."""
        stage = EmbeddingCheckStage()

        assert stage.name == "EmbeddingCheck"
        assert "embedding_check_performed" in stage.produces
        assert "embedding_override_applied" in stage.produces


class TestAbstentionCheckStage:
    """Test suite for AbstentionCheckStage."""

    def test_should_run_when_enabled(self, basic_context: VerificationContext) -> None:
        """Test that abstention check runs when enabled."""
        basic_context.abstention_enabled = True
        basic_context.set_artifact("raw_llm_response", "I cannot answer this question")

        stage = AbstentionCheckStage()
        assert stage.should_run(basic_context) is True

    def test_should_not_run_when_disabled(self, basic_context: VerificationContext) -> None:
        """Test that abstention check skips when disabled."""
        basic_context.abstention_enabled = False

        stage = AbstentionCheckStage()
        assert stage.should_run(basic_context) is False

    @patch("karenina.benchmark.verification.stages.abstention_check.detect_abstention")
    def test_abstention_override_detected(
        self,
        mock_detect: Mock,
        basic_context: VerificationContext,
    ) -> None:
        """Test that abstention check can override verification result."""
        basic_context.abstention_enabled = True
        basic_context.set_artifact("raw_llm_response", "I cannot answer this question")
        basic_context.set_artifact("field_verification_result", False)

        # Mock abstention detection
        # Returns: (abstention_detected, check_performed, reasoning)
        mock_detect.return_value = (True, True, "Explicit refusal detected")

        stage = AbstentionCheckStage()
        stage.execute(basic_context)

        # Should have abstention result
        assert basic_context.has_artifact("abstention_detected")
        assert basic_context.get_artifact("abstention_detected") is True
        assert basic_context.has_artifact("abstention_reasoning")

    @patch("karenina.benchmark.verification.stages.abstention_check.detect_abstention")
    def test_no_abstention_detected(
        self,
        mock_detect: Mock,
        basic_context: VerificationContext,
    ) -> None:
        """Test normal response without abstention."""
        basic_context.abstention_enabled = True
        basic_context.set_artifact("raw_llm_response", "The answer is 4")

        # Mock no abstention
        # Returns: (abstention_detected, check_performed, reasoning)
        mock_detect.return_value = (False, True, "")

        stage = AbstentionCheckStage()
        stage.execute(basic_context)

        # Should have abstention result = False
        assert basic_context.get_artifact("abstention_detected") is False

    def test_stage_metadata(self) -> None:
        """Test stage name and artifact declarations."""
        stage = AbstentionCheckStage()

        assert stage.name == "AbstentionCheck"
        assert "abstention_detected" in stage.produces
        assert "abstention_reasoning" in stage.produces


class TestDeepJudgmentAutoFailStage:
    """Test suite for DeepJudgmentAutoFailStage."""

    def test_should_run_with_deep_judgment_enabled(self, basic_context: VerificationContext) -> None:
        """Test that stage runs when deep-judgment is enabled AND has missing excerpts."""
        basic_context.deep_judgment_enabled = True
        basic_context.set_artifact("deep_judgment_performed", True)
        basic_context.set_artifact("attributes_without_excerpts", ["result"])  # Has missing excerpts

        stage = DeepJudgmentAutoFailStage()
        assert stage.should_run(basic_context) is True

    def test_should_not_run_with_deep_judgment_disabled(self, basic_context: VerificationContext) -> None:
        """Test that stage skips when deep-judgment is disabled."""
        basic_context.deep_judgment_enabled = False

        stage = DeepJudgmentAutoFailStage()
        assert stage.should_run(basic_context) is False

    def test_deep_judgment_autofail_missing_excerpts(self, basic_context: VerificationContext) -> None:
        """Test that missing excerpts trigger auto-fail."""
        basic_context.deep_judgment_enabled = True
        basic_context.set_artifact("deep_judgment_performed", True)
        basic_context.set_artifact("attributes_without_excerpts", ["result", "correct"])
        basic_context.set_artifact("field_verification_result", True)  # Initially passing

        stage = DeepJudgmentAutoFailStage()
        stage.execute(basic_context)

        # Should have modified field_verification_result to False
        assert basic_context.get_artifact("field_verification_result") is False

    def test_deep_judgment_no_autofail(self, basic_context: VerificationContext) -> None:
        """Test that complete excerpts don't trigger auto-fail."""
        basic_context.deep_judgment_enabled = True
        basic_context.set_artifact("deep_judgment_performed", True)
        basic_context.set_artifact("attributes_without_excerpts", [])  # Empty - no missing excerpts

        stage = DeepJudgmentAutoFailStage()
        # Should not run because attributes_without_excerpts is empty
        assert stage.should_run(basic_context) is False

    def test_stage_metadata(self) -> None:
        """Test stage name and artifact declarations."""
        stage = DeepJudgmentAutoFailStage()

        assert stage.name == "DeepJudgmentAutoFail"
        assert "deep_judgment_performed" in stage.requires
        assert "attributes_without_excerpts" in stage.requires
        assert stage.produces == []  # This stage modifies existing results, doesn't produce new artifacts
