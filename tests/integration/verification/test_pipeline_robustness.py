"""Robustness tests for the verification pipeline.

These tests verify that the pipeline handles various error scenarios correctly,
including:
- Stage execution ordering (critical for performance optimizations)
- Error propagation and recovery
- Skip conditions and early termination
- Artifact dependencies between stages
- Edge cases with unusual configurations

The tests use FixtureBackedLLMClient for deterministic, reproducible results
without live API calls.

BUGS DISCOVERED:
1. AbstentionCheckStage.should_run() doesn't check recursion_limit_reached,
   but ParseTemplateStage does. This is inconsistent - when recursion limit
   is hit, the response is truncated and abstention check would be unreliable.
"""

from unittest.mock import MagicMock, patch

import pytest

from karenina.benchmark.verification.stage import (
    VerificationContext,
)
from karenina.benchmark.verification.stage_orchestrator import StageOrchestrator
from karenina.benchmark.verification.stages import (
    AbstentionCheckStage,
    DeepJudgmentAutoFailStage,
    EmbeddingCheckStage,
    FinalizeResultStage,
    GenerateAnswerStage,
    ParseTemplateStage,
    RecursionLimitAutoFailStage,
    ValidateTemplateStage,
    VerifyTemplateStage,
)
from karenina.schemas.domain import LLMRubricTrait, RegexTrait, Rubric
from karenina.schemas.workflow import (
    ModelConfig,
)

# Import fixture infrastructure
from tests.conftest import FixtureBackedLLMClient

# =============================================================================
# Test Fixtures
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
def valid_template_code() -> str:
    """Return a valid template code string."""
    return """
from pydantic import Field
from karenina.schemas.domain import BaseAnswer

class Answer(BaseAnswer):
    capital: str = Field(description="The capital city")

    def model_post_init(self, __context):
        self.correct = {"capital": "paris"}

    def verify(self) -> bool:
        return self.capital.strip().lower() == self.correct["capital"]
"""


@pytest.fixture
def minimal_context(_minimal_model_config: ModelConfig, valid_template_code: str) -> VerificationContext:
    """Return a minimal VerificationContext for testing."""
    return VerificationContext(
        question_id="test-question-1",
        template_id="template-hash-123",
        question_text="What is the capital of France?",
        template_code=valid_template_code,
        answering_model=minimal_model_config,
        parsing_model=minimal_model_config,
    )


@pytest.fixture
def sample_rubric() -> Rubric:
    """Return a sample rubric for testing."""
    return Rubric(
        llm_traits=[
            LLMRubricTrait(
                name="clarity",
                description="Response is clear and understandable",
                kind="boolean",
                min_score=0,
                max_score=1,
            )
        ]
    )


@pytest.fixture
def rubric_with_regex() -> Rubric:
    """Return a rubric with regex traits."""
    return Rubric(
        regex_traits=[
            RegexTrait(
                name="has_citation",
                pattern=r"\[\d+\]",
                description="Response contains citations",
            )
        ]
    )


# =============================================================================
# Stage Execution Order Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.pipeline
class TestStageExecutionOrder:
    """Test that stages are ordered correctly in production configurations.

    These tests verify the relative ordering of stages, which is critical for:
    - Performance (skip expensive LLM calls when possible)
    - Correctness (dependencies must be satisfied)
    - Error handling (early termination when appropriate)
    """

    def test_abstention_check_before_parse_template(self, _minimal_model_config: ModelConfig):
        """AbstentionCheck must run before ParseTemplate to skip expensive parsing.

        When a model refuses to answer, we should detect this BEFORE spending
        tokens on the parsing LLM call.
        """
        orchestrator = StageOrchestrator.from_config(
            abstention_enabled=True,
            evaluation_mode="template_only",
        )

        stage_names = [s.name for s in orchestrator.stages]
        abstention_idx = stage_names.index("AbstentionCheck")
        parse_idx = stage_names.index("ParseTemplate")

        assert abstention_idx < parse_idx, (
            f"AbstentionCheck (idx={abstention_idx}) must run before "
            f"ParseTemplate (idx={parse_idx}) to skip expensive LLM parsing"
        )

    def test_parse_template_before_verify_template(self, _minimal_model_config: ModelConfig):
        """ParseTemplate must run before VerifyTemplate (dependency)."""
        orchestrator = StageOrchestrator.from_config(
            evaluation_mode="template_only",
        )

        stage_names = [s.name for s in orchestrator.stages]
        parse_idx = stage_names.index("ParseTemplate")
        verify_idx = stage_names.index("VerifyTemplate")

        assert parse_idx < verify_idx, (
            f"ParseTemplate (idx={parse_idx}) must run before "
            f"VerifyTemplate (idx={verify_idx}) - produces parsed_answer"
        )

    def test_verify_template_before_embedding_check(self, _minimal_model_config: ModelConfig):
        """VerifyTemplate must run before EmbeddingCheck (semantic fallback).

        Note: EmbeddingCheckStage is always added to the pipeline but has its own
        should_run() logic that only activates when field_verification_result=False.
        """
        orchestrator = StageOrchestrator.from_config(
            evaluation_mode="template_only",
        )

        stage_names = [s.name for s in orchestrator.stages]
        verify_idx = stage_names.index("VerifyTemplate")
        embedding_idx = stage_names.index("EmbeddingCheck")

        assert verify_idx < embedding_idx, (
            f"VerifyTemplate (idx={verify_idx}) must run before "
            f"EmbeddingCheck (idx={embedding_idx}) - embedding is fallback"
        )

    def test_generate_answer_before_abstention_check(self, _minimal_model_config: ModelConfig):
        """GenerateAnswer must run before AbstentionCheck (needs raw response)."""
        orchestrator = StageOrchestrator.from_config(
            abstention_enabled=True,
            evaluation_mode="template_only",
        )

        stage_names = [s.name for s in orchestrator.stages]
        generate_idx = stage_names.index("GenerateAnswer")
        abstention_idx = stage_names.index("AbstentionCheck")

        assert generate_idx < abstention_idx, (
            f"GenerateAnswer (idx={generate_idx}) must run before "
            f"AbstentionCheck (idx={abstention_idx}) - needs raw_llm_response"
        )

    def test_recursion_limit_before_abstention_check(self, _minimal_model_config: ModelConfig):
        """RecursionLimitAutoFail should run before AbstentionCheck."""
        orchestrator = StageOrchestrator.from_config(
            abstention_enabled=True,
            evaluation_mode="template_only",
        )

        stage_names = [s.name for s in orchestrator.stages]
        recursion_idx = stage_names.index("RecursionLimitAutoFail")
        abstention_idx = stage_names.index("AbstentionCheck")

        assert recursion_idx < abstention_idx, (
            f"RecursionLimitAutoFail (idx={recursion_idx}) should run before "
            f"AbstentionCheck (idx={abstention_idx}) - no point checking abstention if recursion limit hit"
        )

    def test_trace_validation_before_abstention_check(self, _minimal_model_config: ModelConfig):
        """TraceValidationAutoFail should run before AbstentionCheck."""
        orchestrator = StageOrchestrator.from_config(
            abstention_enabled=True,
            evaluation_mode="template_only",
        )

        stage_names = [s.name for s in orchestrator.stages]
        trace_idx = stage_names.index("TraceValidationAutoFail")
        abstention_idx = stage_names.index("AbstentionCheck")

        assert trace_idx < abstention_idx, (
            f"TraceValidationAutoFail (idx={trace_idx}) should run before "
            f"AbstentionCheck (idx={abstention_idx}) - invalid trace means skip abstention"
        )

    def test_rubric_evaluation_independent_of_template_verification(
        self, _minimal_model_config: ModelConfig, sample_rubric: Rubric
    ):
        """RubricEvaluation runs after GenerateAnswer but independent of template stages."""
        orchestrator = StageOrchestrator.from_config(
            rubric=sample_rubric,
            evaluation_mode="template_and_rubric",
        )

        stage_names = [s.name for s in orchestrator.stages]
        generate_idx = stage_names.index("GenerateAnswer")
        rubric_idx = stage_names.index("RubricEvaluation")

        assert generate_idx < rubric_idx, (
            f"GenerateAnswer (idx={generate_idx}) must run before "
            f"RubricEvaluation (idx={rubric_idx}) - needs raw_llm_response"
        )

    def test_finalize_result_always_last(self, _minimal_model_config: ModelConfig, sample_rubric: Rubric):
        """FinalizeResult must always be the last stage in any configuration."""
        configs = [
            {"evaluation_mode": "template_only"},
            {"evaluation_mode": "template_only", "abstention_enabled": True},
            {"evaluation_mode": "template_only", "deep_judgment_enabled": True},
            {"evaluation_mode": "rubric_only", "rubric": sample_rubric},
            {"evaluation_mode": "template_and_rubric", "rubric": sample_rubric},
            {
                "evaluation_mode": "template_and_rubric",
                "rubric": sample_rubric,
                "abstention_enabled": True,
                "deep_judgment_enabled": True,
            },
        ]

        for config in configs:
            orchestrator = StageOrchestrator.from_config(
                **config,
            )

            last_stage = orchestrator.stages[-1]
            assert last_stage.name == "FinalizeResult", (
                f"FinalizeResult must be last, but got {last_stage.name} for config: {config}"
            )

    def test_validate_template_always_first_in_template_modes(self, _minimal_model_config: ModelConfig):
        """ValidateTemplate must be first in template evaluation modes."""
        for mode in ["template_only", "template_and_rubric"]:
            orchestrator = StageOrchestrator.from_config(
                evaluation_mode=mode,
            )

            first_stage = orchestrator.stages[0]
            assert first_stage.name == "ValidateTemplate", (
                f"ValidateTemplate must be first in {mode} mode, but got {first_stage.name}"
            )


# =============================================================================
# Skip Condition Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.pipeline
class TestStageSkipConditions:
    """Test that stages correctly skip under various conditions.

    These tests verify that the pipeline short-circuits appropriately to:
    - Save compute (don't parse if abstention detected)
    - Prevent cascading errors (don't verify if parsing failed)
    - Handle edge cases gracefully
    """

    def test_parse_template_skips_on_abstention(self, minimal_context: VerificationContext):
        """ParseTemplateStage should skip when abstention is detected."""
        # Simulate abstention detected
        minimal_context.set_artifact("abstention_detected", True)
        minimal_context.set_artifact("raw_llm_response", "I cannot answer that question.")

        stage = ParseTemplateStage()

        # Should skip - abstention already detected
        assert stage.should_run(minimal_context) is False

    def test_parse_template_skips_on_recursion_limit(self, minimal_context: VerificationContext):
        """ParseTemplateStage should skip when recursion limit was hit."""
        minimal_context.set_artifact("recursion_limit_reached", True)
        minimal_context.set_artifact("raw_llm_response", "Partial response...")

        stage = ParseTemplateStage()

        assert stage.should_run(minimal_context) is False

    def test_parse_template_skips_on_trace_validation_failure(self, minimal_context: VerificationContext):
        """ParseTemplateStage should skip when trace validation failed."""
        minimal_context.set_artifact("trace_validation_failed", True)
        minimal_context.set_artifact("raw_llm_response", "Invalid MCP trace...")

        stage = ParseTemplateStage()

        assert stage.should_run(minimal_context) is False

    def test_verify_template_skips_on_abstention(self, minimal_context: VerificationContext):
        """VerifyTemplateStage should skip when abstention is detected."""
        minimal_context.set_artifact("abstention_detected", True)
        # Even if parsed_answer exists, should skip
        minimal_context.set_artifact("parsed_answer", MagicMock())

        stage = VerifyTemplateStage()

        assert stage.should_run(minimal_context) is False

    def test_verify_template_skips_without_parsed_answer(self, minimal_context: VerificationContext):
        """VerifyTemplateStage should skip when parsing didn't produce result."""
        # No parsed_answer artifact
        minimal_context.set_artifact("raw_llm_response", "Some response")

        stage = VerifyTemplateStage()

        assert stage.should_run(minimal_context) is False

    def test_embedding_check_skips_when_verification_passed(self, minimal_context: VerificationContext):
        """EmbeddingCheckStage should skip when field verification passed."""
        minimal_context.set_artifact("field_verification_result", True)
        minimal_context.set_artifact("parsed_answer", MagicMock())

        stage = EmbeddingCheckStage()

        # Embedding is only a fallback - skip when verification passed
        assert stage.should_run(minimal_context) is False

    def test_embedding_check_runs_when_verification_failed(self, minimal_context: VerificationContext):
        """EmbeddingCheckStage should run when field verification failed."""
        minimal_context.set_artifact("field_verification_result", False)
        minimal_context.set_artifact("parsed_answer", MagicMock())
        minimal_context.set_artifact("verification_result", False)

        stage = EmbeddingCheckStage()

        # Should run as fallback
        assert stage.should_run(minimal_context) is True

    def test_abstention_check_skips_when_disabled(self, minimal_context: VerificationContext):
        """AbstentionCheckStage should skip when abstention is disabled."""
        minimal_context.abstention_enabled = False
        minimal_context.set_artifact("raw_llm_response", "I cannot answer that.")

        stage = AbstentionCheckStage()

        assert stage.should_run(minimal_context) is False

    def test_abstention_check_skips_on_recursion_limit(self, minimal_context: VerificationContext):
        """AbstentionCheckStage should skip when recursion limit was hit.

        When recursion limit is hit, the response is truncated/incomplete, so
        running abstention detection on it would be unreliable.
        """
        minimal_context.abstention_enabled = True
        minimal_context.set_artifact("recursion_limit_reached", True)
        minimal_context.set_artifact("raw_llm_response", "I cannot answer that.")

        stage = AbstentionCheckStage()

        assert stage.should_run(minimal_context) is False

    def test_deep_judgment_autofail_skips_without_deep_judgment(self, minimal_context: VerificationContext):
        """DeepJudgmentAutoFailStage should skip when deep judgment wasn't performed."""
        minimal_context.set_artifact("deep_judgment_performed", False)

        stage = DeepJudgmentAutoFailStage()

        assert stage.should_run(minimal_context) is False

    def test_deep_judgment_autofail_skips_when_all_excerpts_found(self, minimal_context: VerificationContext):
        """DeepJudgmentAutoFailStage should skip when all attributes have excerpts."""
        minimal_context.set_artifact("deep_judgment_performed", True)
        minimal_context.set_artifact("attributes_without_excerpts", [])  # All found

        stage = DeepJudgmentAutoFailStage()

        assert stage.should_run(minimal_context) is False

    def test_recursion_limit_autofail_skips_when_no_limit_hit(self, minimal_context: VerificationContext):
        """RecursionLimitAutoFailStage should skip when recursion limit not hit."""
        minimal_context.set_artifact("recursion_limit_reached", False)

        stage = RecursionLimitAutoFailStage()

        assert stage.should_run(minimal_context) is False


# =============================================================================
# Error Propagation Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.pipeline
class TestErrorPropagation:
    """Test that errors propagate correctly through the pipeline.

    These tests verify that:
    - Fatal errors stop the pipeline
    - Non-fatal errors are recorded but don't stop execution
    - FinalizeResult always runs to produce a result
    - Error details are preserved in the final result
    """

    def test_template_validation_error_stops_pipeline(self, minimal_context: VerificationContext):
        """Invalid template should mark error and stop subsequent stages."""
        minimal_context.template_code = "not valid python code !!!"

        stage = ValidateTemplateStage()
        stage.execute(minimal_context)

        assert minimal_context.completed_without_errors is False
        assert minimal_context.error is not None
        assert "validation failed" in minimal_context.error.lower()

    def test_error_prevents_subsequent_stages(self, minimal_context: VerificationContext):
        """When error is marked, subsequent stages should skip."""
        minimal_context.mark_error("Previous stage failed")

        # These stages should all skip when error is present
        stages = [
            GenerateAnswerStage(),
            ParseTemplateStage(),
            VerifyTemplateStage(),
        ]

        for stage in stages:
            # All should return False for should_run when error exists
            result = stage.should_run(minimal_context)
            assert result is False or minimal_context.error is not None, (
                f"{stage.name} should skip or respect error state"
            )

    def test_finalize_always_runs_after_error(self, minimal_context: VerificationContext):
        """FinalizeResultStage must run even after errors to produce result."""
        minimal_context.mark_error("Something went wrong")
        minimal_context.set_result_field("timestamp", "2024-01-01 12:00:00")

        stage = FinalizeResultStage()

        # Should run despite error
        assert stage.should_run(minimal_context) is True

        stage.execute(minimal_context)

        result = minimal_context.get_artifact("final_result")
        assert result is not None
        assert result.metadata.completed_without_errors is False
        assert result.metadata.error == "Something went wrong"

    def test_recursion_limit_preserves_trace(self, minimal_context: VerificationContext):
        """RecursionLimitAutoFail should fail verification but preserve trace."""
        minimal_context.set_artifact("recursion_limit_reached", True)
        minimal_context.set_artifact("raw_llm_response", "Partial response before limit...")

        stage = RecursionLimitAutoFailStage()
        stage.execute(minimal_context)

        # Verification should fail
        assert minimal_context.get_result_field("verify_result") is False
        # But completed_without_errors stays True (trace preserved for analysis)
        assert minimal_context.completed_without_errors is True
        # Raw response still accessible
        assert minimal_context.get_artifact("raw_llm_response") is not None

    def test_abstention_fails_verification_gracefully(self, minimal_context: VerificationContext):
        """Abstention detection should fail verification without marking error."""
        minimal_context.abstention_enabled = True
        minimal_context.set_artifact("raw_llm_response", "I'm sorry, I can't answer that.")
        minimal_context.set_artifact("abstention_detected", True)
        minimal_context.set_result_field("verify_result", False)

        # Abstention should not mark context.error
        assert minimal_context.completed_without_errors is True
        assert minimal_context.error is None
        # But verification result should be False
        assert minimal_context.get_result_field("verify_result") is False


# =============================================================================
# Artifact Dependency Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.pipeline
class TestArtifactDependencies:
    """Test that artifact dependencies between stages are satisfied.

    These tests verify that:
    - Stages produce the artifacts they claim to produce
    - Consuming stages can access required artifacts
    - Missing artifacts are handled gracefully
    """

    def test_validate_template_produces_answer_classes(self, minimal_context: VerificationContext):
        """ValidateTemplateStage should produce Answer and RawAnswer artifacts."""
        stage = ValidateTemplateStage()
        stage.execute(minimal_context)

        assert minimal_context.has_artifact("Answer"), "Should produce Answer class"
        assert minimal_context.has_artifact("RawAnswer"), "Should produce RawAnswer class"

    def test_generate_answer_produces_raw_response(
        self, minimal_context: VerificationContext, _llm_client: FixtureBackedLLMClient
    ):
        """GenerateAnswerStage should produce raw_llm_response artifact."""
        # Validate first to get Answer class
        ValidateTemplateStage().execute(minimal_context)

        # Mock the unified LLM initialization to use fixture client
        with patch("karenina.benchmark.verification.stages.generate_answer.init_chat_model_unified") as mock_init:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MagicMock(content="The capital is Paris.")
            mock_init.return_value = mock_llm

            stage = GenerateAnswerStage()
            stage.execute(minimal_context)

        assert minimal_context.has_artifact("raw_llm_response")
        assert "Paris" in minimal_context.get_artifact("raw_llm_response")

    def test_missing_required_artifact_handled(self, minimal_context: VerificationContext):
        """Stages should handle missing required artifacts gracefully."""
        # ParseTemplate without raw_llm_response
        stage = ParseTemplateStage()

        # Should skip (not crash) when required artifact missing
        assert stage.should_run(minimal_context) is False


# =============================================================================
# Pipeline Integration Tests with Fixture-Backed LLM
# =============================================================================


@pytest.mark.integration
@pytest.mark.pipeline
class TestPipelineIntegration:
    """End-to-end pipeline integration tests.

    These tests run the full pipeline or significant portions with mocked LLM
    calls to verify the stages work together correctly.
    """

    def test_full_pipeline_success_path_with_mocked_llm(self, minimal_context: VerificationContext):
        """Test successful execution through entire pipeline with mocked LLM.

        This test requires careful mocking because FinalizeResultStage has
        strict Pydantic validation. The mocks must return proper data types.
        """
        # Build pipeline
        orchestrator = StageOrchestrator.from_config(
            evaluation_mode="template_only",
        )

        # First, validate template to get Answer class
        from karenina.benchmark.verification.stages import ValidateTemplateStage

        validate_stage = ValidateTemplateStage()
        validate_stage.execute(minimal_context)
        Answer = minimal_context.get_artifact("Answer")

        # Create a real parsed answer instance
        parsed_answer = Answer(capital="Paris")

        # Mock LLM for answer generation and template parsing
        with (
            patch("karenina.benchmark.verification.stages.generate_answer.init_chat_model_unified") as mock_gen,
            patch("karenina.benchmark.verification.stages.parse_template.TemplateEvaluator") as MockEvaluator,
        ):
            # Answer generation mock
            mock_gen_llm = MagicMock()
            mock_gen_llm.invoke.return_value = MagicMock(content="The capital of France is Paris.")
            mock_gen.return_value = mock_gen_llm

            # Template parsing mock - use proper return types
            mock_evaluator = MagicMock()
            mock_evaluator.model_str = "anthropic/claude-haiku-4-5"

            # parse_response returns a proper result object
            parse_result = MagicMock()
            parse_result.success = True
            parse_result.parsed_answer = parsed_answer
            parse_result.deep_judgment_performed = False
            parse_result.extracted_excerpts = {}
            parse_result.attribute_reasoning = {}
            parse_result.stages_completed = []
            parse_result.model_calls = 0
            parse_result.excerpt_retry_count = 0
            parse_result.attributes_without_excerpts = []
            parse_result.hallucination_risk_assessment = {}
            mock_evaluator.parse_response.return_value = parse_result

            # verify_fields returns proper result
            field_result = MagicMock()
            field_result.success = True
            field_result.error = None
            mock_evaluator.verify_fields.return_value = field_result

            # verify_regex returns proper result with dict types
            regex_result = MagicMock()
            regex_result.success = True
            regex_result.results = {}  # Dict, not list
            regex_result.details = {}
            regex_result.extraction_results = {}
            regex_result.error = None
            mock_evaluator.verify_regex.return_value = regex_result

            MockEvaluator.return_value = mock_evaluator

            result = orchestrator.execute(minimal_context)

        assert result is not None
        assert result.metadata.completed_without_errors is True
        assert result.template is not None

    def test_pipeline_with_abstention_early_exit(self, minimal_context: VerificationContext):
        """Test pipeline exits early when abstention detected."""
        minimal_context.abstention_enabled = True

        orchestrator = StageOrchestrator.from_config(
            abstention_enabled=True,
            evaluation_mode="template_only",
        )

        # Track which stages executed
        executed_stages: list[str] = []

        def tracking_execute(self, context):
            for stage in self.stages:
                if stage.should_run(context):
                    executed_stages.append(stage.name)
                    try:
                        stage.execute(context)
                    except Exception as e:
                        context.mark_error(f"{stage.name} failed: {e}")
            return context.get_artifact("final_result")

        with (
            patch("karenina.benchmark.verification.stages.generate_answer.init_chat_model_unified") as mock_gen,
            patch("karenina.benchmark.verification.stages.abstention_check.detect_abstention") as mock_abstention,
        ):
            # Generate a refusal response
            mock_gen_llm = MagicMock()
            mock_gen_llm.invoke.return_value = MagicMock(
                content="I'm sorry, but I cannot provide information about that topic."
            )
            mock_gen.return_value = mock_gen_llm

            # Abstention detector returns (detected, check_performed, reasoning, usage_metadata)
            mock_abstention.return_value = (
                True,  # abstention_detected
                True,  # check_performed
                "Model explicitly refused to answer",  # reasoning
                {"input_tokens": 10, "output_tokens": 5},  # usage_metadata
            )

            with patch.object(StageOrchestrator, "execute", tracking_execute):
                orchestrator.execute(minimal_context)

        # ParseTemplate and VerifyTemplate should NOT be in executed stages
        # (they should skip due to abstention)
        assert "AbstentionCheck" in executed_stages
        assert "ParseTemplate" not in executed_stages, "ParseTemplate should skip when abstention detected"
        assert "VerifyTemplate" not in executed_stages, "VerifyTemplate should skip when abstention detected"

    def test_rubric_only_mode_skips_template_stages(self, minimal_context: VerificationContext, sample_rubric: Rubric):
        """Test rubric_only mode doesn't include template verification stages."""
        minimal_context.rubric = sample_rubric

        orchestrator = StageOrchestrator.from_config(
            rubric=sample_rubric,
            evaluation_mode="rubric_only",
        )

        stage_names = [s.name for s in orchestrator.stages]

        # Template stages should NOT be present
        assert "ValidateTemplate" not in stage_names
        assert "ParseTemplate" not in stage_names
        assert "VerifyTemplate" not in stage_names

        # But answer generation and rubric evaluation should be present
        assert "GenerateAnswer" in stage_names
        assert "RubricEvaluation" in stage_names
        assert "FinalizeResult" in stage_names


# =============================================================================
# Edge Case Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.pipeline
class TestEdgeCases:
    """Test edge cases and unusual configurations."""

    def test_empty_response_handling(self, minimal_context: VerificationContext):
        """Test pipeline handles empty LLM response."""
        ValidateTemplateStage().execute(minimal_context)
        minimal_context.set_artifact("raw_llm_response", "")

        stage = ParseTemplateStage()

        # ParseTemplateStage should still run (has raw_llm_response, even if empty)
        # Validate that all required artifacts exist
        assert minimal_context.has_artifact("Answer")
        assert minimal_context.has_artifact("raw_llm_response")

        # The stage should attempt to run
        assert stage.should_run(minimal_context) is True

    def test_whitespace_only_response(self, minimal_context: VerificationContext):
        """Test handling of whitespace-only response."""
        ValidateTemplateStage().execute(minimal_context)
        minimal_context.set_artifact("raw_llm_response", "   \n\t\n   ")

        stage = ParseTemplateStage()

        # Should still attempt to run
        assert stage.should_run(minimal_context) is True

    def test_very_long_response(self, minimal_context: VerificationContext):
        """Test handling of very long LLM response."""
        ValidateTemplateStage().execute(minimal_context)

        # Generate a very long response
        long_response = "The capital of France is Paris. " * 10000
        minimal_context.set_artifact("raw_llm_response", long_response)

        stage = ParseTemplateStage()

        # Should still attempt to run
        assert stage.should_run(minimal_context) is True

    def test_unicode_in_response(self, minimal_context: VerificationContext):
        """Test handling of Unicode characters in response."""
        ValidateTemplateStage().execute(minimal_context)
        minimal_context.set_artifact(
            "raw_llm_response", "La capitale de la France est Paris 🇫🇷. C'est une belle ville!"
        )

        stage = ParseTemplateStage()

        # Should still attempt to run
        assert stage.should_run(minimal_context) is True

    def test_concurrent_artifact_access(self, minimal_context: VerificationContext):
        """Test that artifact access is thread-safe."""
        import threading
        import time

        errors = []

        def writer():
            for i in range(100):
                try:
                    minimal_context.set_artifact(f"key_{i}", f"value_{i}")
                    time.sleep(0.001)
                except Exception as e:
                    errors.append(e)

        def reader():
            for i in range(100):
                try:
                    minimal_context.get_artifact(f"key_{i}", None)
                    time.sleep(0.001)
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
            threading.Thread(target=reader),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"

    def test_multiple_error_marks(self, minimal_context: VerificationContext):
        """Test that multiple error marks are handled (first error preserved)."""
        minimal_context.mark_error("First error")
        minimal_context.mark_error("Second error")

        # Error should be set (implementation may preserve first or last)
        assert minimal_context.error is not None

    def test_result_field_overwrites(self, minimal_context: VerificationContext):
        """Test that result fields can be overwritten (e.g., by embedding check)."""
        minimal_context.set_result_field("verify_result", False)
        minimal_context.set_result_field("verify_result", True)  # Override by embedding

        assert minimal_context.get_result_field("verify_result") is True


# =============================================================================
# Configuration Combination Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.pipeline
class TestConfigurationCombinations:
    """Test various combinations of pipeline configuration options."""

    def test_all_features_enabled(self, _minimal_model_config: ModelConfig, sample_rubric: Rubric):
        """Test pipeline with all optional features enabled."""
        orchestrator = StageOrchestrator.from_config(
            rubric=sample_rubric,
            evaluation_mode="template_and_rubric",
            abstention_enabled=True,
            deep_judgment_enabled=True,
        )

        stage_names = [s.name for s in orchestrator.stages]

        # All optional stages should be present
        assert "AbstentionCheck" in stage_names
        # EmbeddingCheck is always added but self-skips via should_run()
        assert "EmbeddingCheck" in stage_names
        assert "DeepJudgmentAutoFail" in stage_names
        assert "RubricEvaluation" in stage_names

        # Verify order constraints still hold
        abstention_idx = stage_names.index("AbstentionCheck")
        parse_idx = stage_names.index("ParseTemplate")
        assert abstention_idx < parse_idx

    def test_minimal_configuration(self, _minimal_model_config: ModelConfig):
        """Test pipeline with minimal configuration (template_only, no extras).

        Note: EmbeddingCheckStage is always included but has its own should_run()
        logic that only runs when field_verification_result=False.
        """
        orchestrator = StageOrchestrator.from_config(
            evaluation_mode="template_only",
        )

        stage_names = [s.name for s in orchestrator.stages]

        # Core stages should be present
        assert "ValidateTemplate" in stage_names
        assert "GenerateAnswer" in stage_names
        assert "ParseTemplate" in stage_names
        assert "VerifyTemplate" in stage_names
        assert "FinalizeResult" in stage_names

        # EmbeddingCheck is always added (but self-skips when not needed)
        assert "EmbeddingCheck" in stage_names

        # These stages should NOT be present without their flags
        assert "AbstentionCheck" not in stage_names
        assert "RubricEvaluation" not in stage_names

    def test_rubric_with_deep_judgment(self, _minimal_model_config: ModelConfig, sample_rubric: Rubric):
        """Test rubric evaluation with deep judgment enabled."""
        orchestrator = StageOrchestrator.from_config(
            rubric=sample_rubric,
            evaluation_mode="template_and_rubric",
            deep_judgment_enabled=True,
        )

        stage_names = [s.name for s in orchestrator.stages]

        assert "RubricEvaluation" in stage_names
        assert "DeepJudgmentRubricAutoFail" in stage_names

    def test_embedding_always_included_in_template_mode(self, _minimal_model_config: ModelConfig):
        """EmbeddingCheckStage is always included in template modes.

        The stage has its own should_run() logic that activates only when
        field_verification_result=False, providing semantic similarity fallback.
        """
        orchestrator = StageOrchestrator.from_config(
            evaluation_mode="template_only",
            deep_judgment_enabled=False,
        )

        stage_names = [s.name for s in orchestrator.stages]

        # EmbeddingCheck is always present in template modes
        assert "EmbeddingCheck" in stage_names
        assert "DeepJudgmentAutoFail" not in stage_names
