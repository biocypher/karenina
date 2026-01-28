"""Unit tests for SufficiencyCheckStage.

Tests cover:
- Stage metadata (name, requires, produces)
- should_run() conditions
- execute() behavior with mocked evaluator
- Integration with VerificationContext

Note: These tests use mocking for the detect_sufficiency evaluator.
For fixture-backed tests with real LLM responses, see integration tests.
"""

from unittest.mock import patch

import pytest
from pydantic import Field

from karenina.benchmark.verification.stages.base import VerificationContext
from karenina.benchmark.verification.stages.sufficiency_check import (
    SufficiencyCheckStage,
)
from karenina.schemas.domain import BaseAnswer
from karenina.schemas.workflow import ModelConfig

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def minimal_model_config() -> ModelConfig:
    """Return a minimal ModelConfig for testing."""
    return ModelConfig(
        id="test-model",
        model_provider="anthropic",
        model_name="claude-haiku-4-5",
        temperature=0.0,
    )


@pytest.fixture
def sample_answer_class():
    """Return a sample Answer class for testing."""

    class Answer(BaseAnswer):
        capital: str = Field(description="The capital city")

        def verify(self) -> bool:
            return self.capital.lower() == "paris"

    return Answer


@pytest.fixture
def minimal_context(minimal_model_config: ModelConfig) -> VerificationContext:
    """Return a minimal VerificationContext for testing."""
    return VerificationContext(
        question_id="test-question-1",
        template_id="template-hash-123",
        question_text="What is the capital of France?",
        template_code="""
from pydantic import Field
from karenina.schemas.domain import BaseAnswer

class Answer(BaseAnswer):
    capital: str = Field(description="The capital city")

    def verify(self) -> bool:
        return self.capital.lower() == "paris"
""",
        answering_model=minimal_model_config,
        parsing_model=minimal_model_config,
    )


# =============================================================================
# Stage Metadata Tests
# =============================================================================


@pytest.mark.unit
class TestSufficiencyCheckStageMetadata:
    """Test SufficiencyCheckStage metadata properties."""

    def test_name(self) -> None:
        """Verify stage name is 'SufficiencyCheck'."""
        stage = SufficiencyCheckStage()
        assert stage.name == "SufficiencyCheck"

    def test_requires(self) -> None:
        """Verify stage requires raw_llm_response and Answer."""
        stage = SufficiencyCheckStage()
        requires = stage.requires
        assert "raw_llm_response" in requires
        assert "Answer" in requires

    def test_produces(self) -> None:
        """Verify stage produces sufficiency artifacts."""
        stage = SufficiencyCheckStage()
        produces = stage.produces
        assert "sufficiency_check_performed" in produces
        assert "sufficiency_detected" in produces
        assert "sufficiency_override_applied" in produces
        assert "sufficiency_reasoning" in produces


# =============================================================================
# should_run() Tests
# =============================================================================


@pytest.mark.unit
class TestSufficiencyCheckStageShouldRun:
    """Test should_run() conditions for SufficiencyCheckStage."""

    def test_should_run_when_enabled(self, minimal_context: VerificationContext) -> None:
        """Verify stage runs when sufficiency_enabled=True."""
        minimal_context.sufficiency_enabled = True

        stage = SufficiencyCheckStage()
        assert stage.should_run(minimal_context) is True

    def test_should_not_run_when_disabled(self, minimal_context: VerificationContext) -> None:
        """Verify stage skips when sufficiency_enabled=False."""
        minimal_context.sufficiency_enabled = False

        stage = SufficiencyCheckStage()
        assert stage.should_run(minimal_context) is False

    def test_should_not_run_on_error(self, minimal_context: VerificationContext) -> None:
        """Verify stage skips when context has error."""
        minimal_context.sufficiency_enabled = True
        minimal_context.mark_error("Previous stage failed")

        stage = SufficiencyCheckStage()
        assert stage.should_run(minimal_context) is False

    def test_should_not_run_on_recursion_limit(self, minimal_context: VerificationContext) -> None:
        """Verify stage skips when recursion limit was reached."""
        minimal_context.sufficiency_enabled = True
        minimal_context.set_artifact("recursion_limit_reached", True)

        stage = SufficiencyCheckStage()
        assert stage.should_run(minimal_context) is False

    def test_should_not_run_on_trace_validation_failed(self, minimal_context: VerificationContext) -> None:
        """Verify stage skips when trace validation failed."""
        minimal_context.sufficiency_enabled = True
        minimal_context.set_artifact("trace_validation_failed", True)

        stage = SufficiencyCheckStage()
        assert stage.should_run(minimal_context) is False

    def test_should_not_run_when_abstention_detected(self, minimal_context: VerificationContext) -> None:
        """Verify stage skips when abstention was already detected."""
        minimal_context.sufficiency_enabled = True
        minimal_context.set_artifact("abstention_detected", True)

        stage = SufficiencyCheckStage()
        assert stage.should_run(minimal_context) is False

    def test_should_run_with_abstention_false(self, minimal_context: VerificationContext) -> None:
        """Verify stage runs when abstention was checked but not detected."""
        minimal_context.sufficiency_enabled = True
        minimal_context.set_artifact("abstention_detected", False)

        stage = SufficiencyCheckStage()
        assert stage.should_run(minimal_context) is True


# =============================================================================
# execute() Tests
# =============================================================================


@pytest.mark.unit
class TestSufficiencyCheckStageExecute:
    """Test execute() behavior for SufficiencyCheckStage."""

    def test_execute_sufficient_response(
        self,
        minimal_context: VerificationContext,
        sample_answer_class,
    ) -> None:
        """Verify sufficient response sets correct artifacts."""
        minimal_context.sufficiency_enabled = True
        minimal_context.set_artifact("raw_llm_response", "The capital of France is Paris.")
        minimal_context.set_artifact("Answer", sample_answer_class)

        with patch("karenina.benchmark.verification.stages.sufficiency_check.detect_sufficiency") as mock_detect:
            mock_detect.return_value = (
                True,  # sufficient
                True,  # check_performed
                "Response contains the capital city.",  # reasoning
                {"claude-haiku-4-5": {"input_tokens": 50, "output_tokens": 50, "total_tokens": 100}},  # usage_metadata
            )

            stage = SufficiencyCheckStage()
            stage.execute(minimal_context)

            # Check artifacts were set correctly
            assert minimal_context.get_artifact("sufficiency_check_performed") is True
            assert minimal_context.get_artifact("sufficiency_detected") is True
            assert minimal_context.get_artifact("sufficiency_override_applied") is False
            assert minimal_context.get_artifact("sufficiency_reasoning") == "Response contains the capital city."

            # Check result fields were set
            assert minimal_context.get_result_field("sufficiency_check_performed") is True
            assert minimal_context.get_result_field("sufficiency_detected") is True
            assert minimal_context.get_result_field("sufficiency_override_applied") is False

    def test_execute_insufficient_response_sets_verify_result_false(
        self,
        minimal_context: VerificationContext,
        sample_answer_class,
    ) -> None:
        """Verify insufficient response sets verify_result=False."""
        minimal_context.sufficiency_enabled = True
        minimal_context.set_artifact("raw_llm_response", "I don't know.")
        minimal_context.set_artifact("Answer", sample_answer_class)

        with patch("karenina.benchmark.verification.stages.sufficiency_check.detect_sufficiency") as mock_detect:
            mock_detect.return_value = (
                False,  # sufficient=False (insufficient)
                True,  # check_performed
                "Response does not contain the required information.",  # reasoning
                {"claude-haiku-4-5": {"input_tokens": 50, "output_tokens": 50, "total_tokens": 100}},  # usage_metadata
            )

            stage = SufficiencyCheckStage()
            stage.execute(minimal_context)

            # Insufficient response should override verify_result to False
            assert minimal_context.get_artifact("sufficiency_detected") is False
            assert minimal_context.get_artifact("sufficiency_override_applied") is True
            assert minimal_context.get_artifact("verify_result") is False
            assert minimal_context.get_result_field("verify_result") is False

    def test_execute_check_failed_does_not_override(
        self,
        minimal_context: VerificationContext,
        sample_answer_class,
    ) -> None:
        """Verify failed check does not override verify_result."""
        minimal_context.sufficiency_enabled = True
        minimal_context.set_artifact("raw_llm_response", "Some response.")
        minimal_context.set_artifact("Answer", sample_answer_class)

        with patch("karenina.benchmark.verification.stages.sufficiency_check.detect_sufficiency") as mock_detect:
            mock_detect.return_value = (
                True,  # sufficient (default on failure)
                False,  # check_performed=False (failed)
                None,  # reasoning=None
                {},  # usage_metadata
            )

            stage = SufficiencyCheckStage()
            stage.execute(minimal_context)

            # Failed check should not override
            assert minimal_context.get_artifact("sufficiency_check_performed") is False
            assert minimal_context.get_artifact("sufficiency_override_applied") is False
            # verify_result should not be set by this stage when check failed
            assert minimal_context.get_artifact("verify_result") is None

    def test_execute_tracks_usage_metadata(
        self,
        minimal_context: VerificationContext,
        sample_answer_class,
    ) -> None:
        """Verify usage metadata is tracked."""
        minimal_context.sufficiency_enabled = True
        minimal_context.set_artifact("raw_llm_response", "Paris")
        minimal_context.set_artifact("Answer", sample_answer_class)

        with patch("karenina.benchmark.verification.stages.sufficiency_check.detect_sufficiency") as mock_detect:
            mock_detect.return_value = (
                True,
                True,
                "OK",
                {"claude-haiku-4-5": {"input_tokens": 50, "output_tokens": 20, "total_tokens": 70}},
            )

            stage = SufficiencyCheckStage()
            stage.execute(minimal_context)

            # Should have updated usage tracker
            usage_tracker = minimal_context.get_artifact("usage_tracker")
            assert usage_tracker is not None

    def test_execute_handles_schema_extraction_failure(
        self,
        minimal_context: VerificationContext,
    ) -> None:
        """Verify graceful handling when JSON schema extraction fails."""
        minimal_context.sufficiency_enabled = True
        minimal_context.set_artifact("raw_llm_response", "Some response.")

        # Set an Answer class that will fail schema extraction
        class BadAnswer:
            def model_json_schema(self):
                raise RuntimeError("Schema extraction failed")

        minimal_context.set_artifact("Answer", BadAnswer)

        stage = SufficiencyCheckStage()
        stage.execute(minimal_context)

        # Should mark as not performed
        assert minimal_context.get_artifact("sufficiency_check_performed") is False
        assert minimal_context.get_artifact("sufficiency_detected") is None
        assert minimal_context.get_artifact("sufficiency_override_applied") is False


@pytest.mark.unit
class TestSufficiencyCheckStageSemantics:
    """Test the semantic behavior of SufficiencyCheckStage.

    Key semantic: sufficient=False triggers verify_result=False override.
    This is the opposite of abstention where detected=True triggers override.
    """

    def test_semantic_sufficient_true_no_override(
        self,
        minimal_context: VerificationContext,
        sample_answer_class,
    ) -> None:
        """When sufficient=True, no override should happen."""
        minimal_context.sufficiency_enabled = True
        minimal_context.set_artifact("raw_llm_response", "Paris")
        minimal_context.set_artifact("Answer", sample_answer_class)

        with patch("karenina.benchmark.verification.stages.sufficiency_check.detect_sufficiency") as mock_detect:
            mock_detect.return_value = (True, True, "OK", {})

            stage = SufficiencyCheckStage()
            stage.execute(minimal_context)

            # No override when sufficient
            assert minimal_context.get_artifact("sufficiency_override_applied") is False

    def test_semantic_sufficient_false_triggers_override(
        self,
        minimal_context: VerificationContext,
        sample_answer_class,
    ) -> None:
        """When sufficient=False, verify_result should be overridden to False."""
        minimal_context.sufficiency_enabled = True
        minimal_context.set_artifact("raw_llm_response", "I don't know")
        minimal_context.set_artifact("Answer", sample_answer_class)

        with patch("karenina.benchmark.verification.stages.sufficiency_check.detect_sufficiency") as mock_detect:
            mock_detect.return_value = (False, True, "Insufficient", {})

            stage = SufficiencyCheckStage()
            stage.execute(minimal_context)

            # Override when insufficient
            assert minimal_context.get_artifact("sufficiency_override_applied") is True
            assert minimal_context.get_result_field("verify_result") is False
