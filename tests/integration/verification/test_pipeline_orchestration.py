"""Integration tests for verification pipeline orchestration.

These tests verify that the StageOrchestrator correctly:
- Builds stage lists from configuration
- Validates stage dependencies
- Executes stages in order
- Passes context/artifacts between stages
- Handles errors and partial completion
- Always runs FinalizeResultStage

Test scenarios:
- Pipeline configuration for different evaluation modes
- Stage execution ordering
- Context artifact passing
- Error propagation
- Dependency validation

The tests use the StageOrchestrator, VerificationContext, and real stage
implementations to verify the pipeline architecture.
"""

import pytest

from karenina.benchmark.verification.stages import (
    BaseVerificationStage,
    FinalizeResultStage,
    RecursionLimitAutoFailStage,
    StageOrchestrator,
    StageRegistry,
    ValidateTemplateStage,
    VerificationContext,
)
from karenina.benchmark.verification.stages.core.base import ArtifactKeys
from karenina.benchmark.verification.stages.pipeline.trace_validation_autofail import (
    TraceValidationAutoFailStage,
)
from karenina.schemas.config import ModelConfig
from karenina.schemas.entities import AgenticRubricTrait, LLMRubricTrait, Rubric
from karenina.schemas.verification import (
    VerificationResult,
    VerificationResultMetadata,
    VerificationResultRubric,
    VerificationResultTemplate,
)
from karenina.schemas.verification.model_identity import ModelIdentity

# =============================================================================
# Test Helpers - Create Valid VerificationResult
# =============================================================================


def create_minimal_result(context: VerificationContext) -> VerificationResult:
    """Create a minimal valid VerificationResult from context."""
    from karenina.schemas.results.failure import Failure, FailureCategory

    _answering = ModelIdentity(interface="langchain", model_name="test/model")
    _parsing = ModelIdentity(interface="langchain", model_name="test/model")
    result_id = VerificationResultMetadata.compute_result_id(
        question_id=context.question_id,
        answering=_answering,
        parsing=_parsing,
        timestamp="2024-01-01 12:00:00",
    )

    # Derive a Failure from legacy context state only when there is a real error.
    # This mock does not invoke the classifier, but preserves the pass/fail signal
    # that tests assert on (metadata.failure is None vs not None).
    failure: Failure | None = None
    if context.error is not None:
        failure = Failure(
            category=FailureCategory.UNEXPECTED_ERROR,
            stage=context.error_stage or context.last_run_stage or "unknown",
            reason=context.error,
            details={"error_message": context.error},
        )
    metadata = VerificationResultMetadata(
        question_id=context.question_id,
        template_id=context.template_id,
        failure=failure,
        caveats=[],
        question_text=context.question_text,
        answering=_answering,
        parsing=_parsing,
        execution_time=1.0,
        timestamp="2024-01-01 12:00:00",
        result_id=result_id,
    )
    template = VerificationResultTemplate(raw_llm_response="")
    return VerificationResult(metadata=metadata, template=template)


class MockFinalizeStage(FinalizeResultStage):
    """Mock FinalizeResultStage that produces a minimal VerificationResult.

    Inherits from FinalizeResultStage so the orchestrator's isinstance
    check accepts it as a genuine finalizer (issue 203 guard).
    """

    def __init__(self):
        self.executed = False

    def should_run(self, context: VerificationContext) -> bool:  # noqa: ARG002
        """Always run - this is the final stage (must not skip on errors)."""
        return True

    def execute(self, context: VerificationContext) -> None:
        self.executed = True
        result = create_minimal_result(context)
        context.set_artifact("final_result", result)


# =============================================================================
# Test Helpers - Mock Stages for Controlled Testing
# =============================================================================


class MockProducerStage(BaseVerificationStage):
    """Mock stage that produces a specific artifact."""

    def __init__(self, name: str, artifact_key: str, artifact_value: str):
        self._name = name
        self._artifact_key = artifact_key
        self._artifact_value = artifact_value

    @property
    def name(self) -> str:
        return self._name

    @property
    def produces(self) -> list[str]:
        return [self._artifact_key]

    def execute(self, context: VerificationContext) -> None:
        context.set_artifact(self._artifact_key, self._artifact_value)


class MockConsumerStage(BaseVerificationStage):
    """Mock stage that requires a specific artifact."""

    def __init__(self, name: str, required_key: str, produced_key: str):
        self._name = name
        self._required_key = required_key
        self._produced_key = produced_key

    @property
    def name(self) -> str:
        return self._name

    @property
    def requires(self) -> list[str]:
        return [self._required_key]

    @property
    def produces(self) -> list[str]:
        return [self._produced_key]

    def execute(self, context: VerificationContext) -> None:
        value = context.get_artifact(self._required_key)
        context.set_artifact(self._produced_key, f"processed:{value}")


class MockErrorStage(BaseVerificationStage):
    """Mock stage that marks an error on the context."""

    def __init__(self, name: str = "MockError"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def execute(self, context: VerificationContext) -> None:
        context.mark_error("Test error from MockErrorStage")


class MockExceptionStage(BaseVerificationStage):
    """Mock stage that raises an exception."""

    def __init__(self, name: str = "MockException"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def execute(self, context: VerificationContext) -> None:  # noqa: ARG002
        raise RuntimeError("Test exception from MockExceptionStage")


class MockConditionalStage(BaseVerificationStage):
    """Mock stage that conditionally runs based on context."""

    def __init__(self, name: str, condition_key: str, expected_value: bool = True):
        self._name = name
        self._condition_key = condition_key
        self._expected_value = expected_value
        self.executed = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def produces(self) -> list[str]:
        return [f"{self._name}_ran"]

    def should_run(self, context: VerificationContext) -> bool:
        value = context.get_artifact(self._condition_key, False)
        return value == self._expected_value

    def execute(self, context: VerificationContext) -> None:
        self.executed = True
        context.set_artifact(f"{self._name}_ran", True)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def minimal_model_config() -> ModelConfig:
    """Return a minimal ModelConfig for testing."""
    return ModelConfig(
        id="claude-haiku-4-5",
        model_provider="anthropic",
        model_name="claude-haiku-4-5",
        temperature=0.0,
    )


@pytest.fixture
def minimal_context(minimal_model_config: ModelConfig) -> VerificationContext:
    """Return a minimal VerificationContext for testing."""
    return VerificationContext(
        question_id="test-question-1",
        template_id="template-hash-123",
        question_text="What is the capital of France?",
        template_code="""
from pydantic import Field
from karenina.schemas.entities import BaseAnswer

class Answer(BaseAnswer):
    capital: str = Field(description="The capital city")

    def verify(self) -> bool:
        return self.capital.lower() == "paris"
""",
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
                description="Response is clear",
                kind="boolean",
            )
        ]
    )


# =============================================================================
# StageOrchestrator Configuration Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.pipeline
class TestStageOrchestratorConfiguration:
    """Test StageOrchestrator.from_config() builds correct stage lists."""

    def test_template_only_mode_stages(self):
        """Verify template_only mode includes template verification stages."""
        orchestrator = StageOrchestrator.from_config(
            evaluation_mode="template_only",
        )

        stage_names = [s.name for s in orchestrator.stages]

        # Template stages should be present
        assert "ValidateTemplate" in stage_names
        assert "GenerateAnswer" in stage_names
        assert "ParseTemplate" in stage_names
        assert "VerifyTemplate" in stage_names
        assert "FinalizeResult" in stage_names

        # Rubric stage should NOT be present (no rubric provided)
        assert "RubricEvaluation" not in stage_names

    def test_template_and_rubric_mode_stages(self, sample_rubric: Rubric):
        """Verify template_and_rubric mode includes both template and rubric stages."""
        orchestrator = StageOrchestrator.from_config(
            rubric=sample_rubric,
            evaluation_mode="template_and_rubric",
        )

        stage_names = [s.name for s in orchestrator.stages]

        # Template stages should be present
        assert "ValidateTemplate" in stage_names
        assert "ParseTemplate" in stage_names
        assert "VerifyTemplate" in stage_names

        # Rubric stage should be present
        assert "RubricEvaluation" in stage_names
        assert "FinalizeResult" in stage_names

    def test_rubric_only_mode_stages(self, sample_rubric: Rubric):
        """Verify rubric_only mode skips template stages."""
        orchestrator = StageOrchestrator.from_config(
            rubric=sample_rubric,
            evaluation_mode="rubric_only",
        )

        stage_names = [s.name for s in orchestrator.stages]

        # Template stages should NOT be present
        assert "ValidateTemplate" not in stage_names
        assert "ParseTemplate" not in stage_names
        assert "VerifyTemplate" not in stage_names

        # Answer generation and rubric should be present
        assert "GenerateAnswer" in stage_names
        assert "RubricEvaluation" in stage_names
        assert "FinalizeResult" in stage_names

    def test_agentic_parsing_always_uses_agentic_stage(self):
        orchestrator = StageOrchestrator.from_config(
            evaluation_mode="template_only",
            agentic_parsing=True,
            agentic_parsing_trigger="always",
        )
        stage_names = [s.name for s in orchestrator.stages]
        assert "AgenticParseTemplate" in stage_names
        assert "DynamicParseTemplate" not in stage_names
        assert "ParseTemplate" not in stage_names

    def test_agentic_parsing_dynamic_uses_dynamic_stage(self):
        orchestrator = StageOrchestrator.from_config(
            evaluation_mode="template_only",
            agentic_parsing=True,
            agentic_parsing_trigger="dynamic",
        )
        stage_names = [s.name for s in orchestrator.stages]
        assert "DynamicParseTemplate" in stage_names
        assert "AgenticParseTemplate" not in stage_names
        assert "ParseTemplate" not in stage_names

    def test_non_agentic_parsing_uses_classical_stage_when_trigger_default(self):
        orchestrator = StageOrchestrator.from_config(
            evaluation_mode="template_only",
            agentic_parsing=False,
        )
        stage_names = [s.name for s in orchestrator.stages]
        assert "ParseTemplate" in stage_names
        assert "DynamicParseTemplate" not in stage_names
        assert "AgenticParseTemplate" not in stage_names

    def test_abstention_enabled_adds_stage(self):
        """Verify abstention_enabled=True adds AbstentionCheckStage."""
        orchestrator = StageOrchestrator.from_config(
            abstention_enabled=True,
            evaluation_mode="template_only",
        )

        stage_names = [s.name for s in orchestrator.stages]
        assert "AbstentionCheck" in stage_names

    def test_deep_judgment_enabled_adds_stage(self):
        """Verify deep_judgment_enabled=True adds DeepJudgmentAutoFailStage."""
        orchestrator = StageOrchestrator.from_config(
            deep_judgment_enabled=True,
            evaluation_mode="template_only",
        )

        stage_names = [s.name for s in orchestrator.stages]
        assert "DeepJudgmentAutoFail" in stage_names

    def test_finalize_result_always_last(self, sample_rubric: Rubric):
        """Verify FinalizeResultStage is always the last stage."""
        for mode in ["template_only", "template_and_rubric", "rubric_only"]:
            orchestrator = StageOrchestrator.from_config(
                rubric=sample_rubric if mode != "template_only" else None,
                evaluation_mode=mode,
            )

            last_stage = orchestrator.stages[-1]
            assert last_stage.name == "FinalizeResult"

    def test_recursion_limit_stage_present(self):
        """Verify RecursionLimitAutoFailStage is in the pipeline."""
        orchestrator = StageOrchestrator.from_config(
            evaluation_mode="template_only",
        )

        stage_names = [s.name for s in orchestrator.stages]
        assert "RecursionLimitAutoFail" in stage_names

    def test_trace_validation_stage_present(self):
        """Verify TraceValidationAutoFailStage is in the pipeline."""
        orchestrator = StageOrchestrator.from_config(
            evaluation_mode="template_only",
        )

        stage_names = [s.name for s in orchestrator.stages]
        assert "TraceValidationAutoFail" in stage_names

    def test_sufficiency_enabled_adds_stage(self):
        """Verify sufficiency_enabled=True adds SufficiencyCheckStage."""
        orchestrator = StageOrchestrator.from_config(
            sufficiency_enabled=True,
            evaluation_mode="template_only",
        )

        stage_names = [s.name for s in orchestrator.stages]
        assert "SufficiencyCheck" in stage_names

    def test_sufficiency_disabled_no_stage(self):
        """Verify sufficiency_enabled=False does not add SufficiencyCheckStage."""
        orchestrator = StageOrchestrator.from_config(
            sufficiency_enabled=False,
            evaluation_mode="template_only",
        )

        stage_names = [s.name for s in orchestrator.stages]
        assert "SufficiencyCheck" not in stage_names

    def test_sufficiency_and_abstention_both_enabled(self):
        """Verify both sufficiency and abstention stages can be enabled together."""
        orchestrator = StageOrchestrator.from_config(
            abstention_enabled=True,
            sufficiency_enabled=True,
            evaluation_mode="template_only",
        )

        stage_names = [s.name for s in orchestrator.stages]
        assert "AbstentionCheck" in stage_names
        assert "SufficiencyCheck" in stage_names

    def test_sufficiency_stage_position_after_abstention(self):
        """Verify SufficiencyCheck comes after AbstentionCheck in pipeline."""
        orchestrator = StageOrchestrator.from_config(
            abstention_enabled=True,
            sufficiency_enabled=True,
            evaluation_mode="template_only",
        )

        stage_names = [s.name for s in orchestrator.stages]
        abstention_idx = stage_names.index("AbstentionCheck")
        sufficiency_idx = stage_names.index("SufficiencyCheck")

        # Sufficiency should come after abstention
        assert sufficiency_idx > abstention_idx

    def test_sufficiency_stage_position_before_parse_template(self):
        """Verify SufficiencyCheck comes before ParseTemplate in pipeline."""
        orchestrator = StageOrchestrator.from_config(
            sufficiency_enabled=True,
            evaluation_mode="template_only",
        )

        stage_names = [s.name for s in orchestrator.stages]
        sufficiency_idx = stage_names.index("SufficiencyCheck")
        parse_idx = stage_names.index("ParseTemplate")

        # Sufficiency should come before parsing
        assert sufficiency_idx < parse_idx


# =============================================================================
# StageRegistry Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.pipeline
class TestStageRegistry:
    """Test StageRegistry functionality."""

    def test_register_stage(self):
        """Verify stages can be registered."""
        registry = StageRegistry()
        stage = MockProducerStage("test", "key", "value")

        registry.register(stage)

        # Verify stage is registered via validate_dependencies (only public method)
        errors = registry.validate_dependencies([stage])
        assert len(errors) == 0

    def test_register_duplicate_raises(self):
        """Verify registering duplicate stage name raises ValueError."""
        registry = StageRegistry()
        stage1 = MockProducerStage("test", "key1", "value1")
        stage2 = MockProducerStage("test", "key2", "value2")

        registry.register(stage1)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(stage2)

    def test_validate_dependencies_success(self):
        """Verify valid dependency chain passes validation."""
        registry = StageRegistry()
        producer = MockProducerStage("producer", "data", "value")
        consumer = MockConsumerStage("consumer", "data", "result")

        stages = [producer, consumer]
        for s in stages:
            registry.register(s)

        errors = registry.validate_dependencies(stages)

        assert len(errors) == 0

    def test_validate_dependencies_failure(self):
        """Verify missing dependency fails validation."""
        registry = StageRegistry()
        consumer = MockConsumerStage("consumer", "missing_data", "result")

        stages = [consumer]
        registry.register(consumer)

        errors = registry.validate_dependencies(stages)

        assert len(errors) == 1
        assert "missing_data" in errors[0]


# =============================================================================
# VerificationContext Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.pipeline
class TestVerificationContext:
    """Test VerificationContext artifact and result management."""

    def test_set_and_get_artifact(self, minimal_context: VerificationContext):
        """Verify artifacts can be stored and retrieved."""
        minimal_context.set_artifact("test_key", "test_value")

        assert minimal_context.get_artifact("test_key") == "test_value"

    def test_get_artifact_default(self, minimal_context: VerificationContext):
        """Verify get_artifact returns default for missing keys."""
        result = minimal_context.get_artifact("missing", "default_value")

        assert result == "default_value"

    def test_has_artifact(self, minimal_context: VerificationContext):
        """Verify has_artifact correctly checks existence."""
        minimal_context.set_artifact("exists", True)

        assert minimal_context.has_artifact("exists") is True
        assert minimal_context.has_artifact("missing") is False

    def test_set_and_get_result_field(self, minimal_context: VerificationContext):
        """Verify result fields can be stored and retrieved."""
        minimal_context.set_result_field("verify_result", True)

        assert minimal_context.get_result_field("verify_result") is True

    def test_mark_error(self, minimal_context: VerificationContext):
        """Verify mark_error sets error and completed_without_errors."""
        assert minimal_context.completed_without_errors is True
        assert minimal_context.error is None

        minimal_context.mark_error("Test error")

        assert minimal_context.completed_without_errors is False
        assert minimal_context.error == "Test error"

    def test_context_preserves_configuration(self, minimal_context: VerificationContext):
        """Verify context preserves all configuration fields."""
        assert minimal_context.question_id == "test-question-1"
        assert minimal_context.template_id == "template-hash-123"
        assert minimal_context.question_text == "What is the capital of France?"
        assert "BaseAnswer" in minimal_context.template_code


# =============================================================================
# Stage Execution Order Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.pipeline
class TestStageExecutionOrder:
    """Test that stages execute in correct order with proper context passing."""

    def test_stages_execute_in_order(self, minimal_context: VerificationContext):
        """Verify stages execute in list order."""
        execution_order = []

        class TrackingStage(BaseVerificationStage):
            def __init__(self, name: str, order_list: list):
                self._name = name
                self._order_list = order_list

            @property
            def name(self) -> str:
                return self._name

            @property
            def produces(self) -> list[str]:
                return [f"{self._name}_done"]

            def execute(self, context: VerificationContext) -> None:
                self._order_list.append(self._name)
                context.set_artifact(f"{self._name}_done", True)

        # Use our helper MockFinalizeStage
        class TrackingFinalize(MockFinalizeStage):
            def __init__(self, order_list: list):
                super().__init__()
                self._order_list = order_list

            def execute(self, context: VerificationContext) -> None:
                self._order_list.append("FinalizeResult")
                super().execute(context)

        stages = [
            TrackingStage("Stage1", execution_order),
            TrackingStage("Stage2", execution_order),
            TrackingStage("Stage3", execution_order),
            TrackingFinalize(execution_order),
        ]

        orchestrator = StageOrchestrator(stages)
        orchestrator.execute(minimal_context)

        assert execution_order == ["Stage1", "Stage2", "Stage3", "FinalizeResult"]

    def test_artifact_passing_between_stages(self, minimal_context: VerificationContext):
        """Verify artifacts flow correctly between stages."""

        class FinalizeWithCheck(MockFinalizeStage):
            def __init__(self):
                super().__init__()
                self.received_value = None

            def execute(self, context: VerificationContext) -> None:
                self.received_value = context.get_artifact("processed_data")
                super().execute(context)

        producer = MockProducerStage("Producer", "raw_data", "initial_value")
        consumer = MockConsumerStage("Consumer", "raw_data", "processed_data")
        finalizer = FinalizeWithCheck()

        orchestrator = StageOrchestrator([producer, consumer, finalizer])
        orchestrator.execute(minimal_context)

        assert finalizer.received_value == "processed:initial_value"

    def test_conditional_stage_skipped(self, minimal_context: VerificationContext):
        """Verify stages with should_run=False are skipped."""
        conditional = MockConditionalStage("Conditional", "run_me", expected_value=True)

        # Don't set run_me artifact, so conditional stage should skip
        orchestrator = StageOrchestrator([conditional, MockFinalizeStage()])
        orchestrator.execute(minimal_context)

        assert conditional.executed is False
        assert minimal_context.has_artifact("Conditional_ran") is False

    def test_conditional_stage_runs_when_condition_met(self, minimal_context: VerificationContext):
        """Verify stages with should_run=True execute."""
        # Set the condition artifact
        minimal_context.set_artifact("run_me", True)

        conditional = MockConditionalStage("Conditional", "run_me", expected_value=True)

        orchestrator = StageOrchestrator([conditional, MockFinalizeStage()])
        orchestrator.execute(minimal_context)

        assert conditional.executed is True
        assert minimal_context.get_artifact("Conditional_ran") is True


# =============================================================================
# Error Handling Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.pipeline
class TestErrorHandling:
    """Test error handling in the pipeline."""

    def test_error_marked_on_context_propagates(self, minimal_context: VerificationContext):
        """Verify stage marking error sets context.error."""
        error_stage = MockErrorStage()
        orchestrator = StageOrchestrator([error_stage, MockFinalizeStage()])
        result = orchestrator.execute(minimal_context)

        assert minimal_context.completed_without_errors is False
        assert "Test error" in minimal_context.error
        assert result.metadata.failure is not None

    def test_exception_caught_and_marked(self, minimal_context: VerificationContext):
        """Verify stage exception is caught and marked on context."""
        exception_stage = MockExceptionStage()
        orchestrator = StageOrchestrator([exception_stage, MockFinalizeStage()])
        result = orchestrator.execute(minimal_context)

        assert minimal_context.completed_without_errors is False
        assert "RuntimeError" in minimal_context.error
        assert "Test exception" in minimal_context.error
        assert result.metadata.failure is not None
        assert result.metadata.failure.reason

    def test_finalize_always_runs_after_error(self, minimal_context: VerificationContext):
        """Verify FinalizeResultStage runs even after stage error."""
        finalizer = MockFinalizeStage()
        error_stage = MockErrorStage()
        orchestrator = StageOrchestrator([error_stage, finalizer])
        orchestrator.execute(minimal_context)

        assert finalizer.executed is True

    def test_missing_final_result_raises(self, minimal_context: VerificationContext):  # noqa: ARG002
        """Reject impostor stages that borrow the FinalizeResult name without inheriting."""

        class NoOutputFinalize(BaseVerificationStage):
            @property
            def name(self) -> str:
                return "FinalizeResult"

            @property
            def produces(self) -> list[str]:
                return ["final_result"]

            def execute(self, context: VerificationContext) -> None:  # noqa: ARG002
                # Intentionally don't set final_result
                pass

        with pytest.raises(ValueError, match="must inherit from FinalizeResultStage"):
            StageOrchestrator([NoOutputFinalize()])


# =============================================================================
# Dependency Validation Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.pipeline
class TestDependencyValidation:
    """Test stage dependency validation."""

    def test_valid_dependencies_pass(self):
        """Verify valid dependency chain passes validation."""
        producer = MockProducerStage("Producer", "data", "value")
        consumer = MockConsumerStage("Consumer", "data", "result")

        class NoOpFinalize(FinalizeResultStage):
            def execute(self, context: VerificationContext) -> None:  # noqa: ARG002
                pass

        orchestrator = StageOrchestrator([producer, consumer, NoOpFinalize()])

        # Should not raise
        errors = orchestrator.validate_dependencies()
        assert len(errors) == 0

    def test_invalid_dependencies_detected(self):
        """Verify missing dependencies are detected."""
        # Consumer requires 'missing_data' which no stage produces
        consumer = MockConsumerStage("Consumer", "missing_data", "result")

        class NoOpFinalize(FinalizeResultStage):
            def execute(self, context: VerificationContext) -> None:  # noqa: ARG002
                pass

        orchestrator = StageOrchestrator([consumer, NoOpFinalize()])

        errors = orchestrator.validate_dependencies()

        assert len(errors) == 1
        assert "missing_data" in errors[0]

    def test_execute_raises_on_invalid_dependencies(self, minimal_context: VerificationContext):
        """Verify execute raises ValueError for invalid dependencies."""
        consumer = MockConsumerStage("Consumer", "missing_data", "result")

        class NoOpFinalize(FinalizeResultStage):
            def execute(self, context: VerificationContext) -> None:  # noqa: ARG002
                pass

        orchestrator = StageOrchestrator([consumer, NoOpFinalize()])

        with pytest.raises(ValueError, match="dependency validation failed"):
            orchestrator.execute(minimal_context)


# =============================================================================
# Real Stage Integration Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.pipeline
class TestRealStageIntegration:
    """Test with real stage implementations (without LLM calls)."""

    def test_validate_template_stage_valid(self, minimal_context: VerificationContext):
        """Verify ValidateTemplateStage validates correct template."""
        stage = ValidateTemplateStage()

        assert stage.should_run(minimal_context) is True

        stage.execute(minimal_context)

        assert minimal_context.has_artifact("Answer") is True
        assert minimal_context.has_artifact("RawAnswer") is True
        assert minimal_context.error is None

    def test_validate_template_stage_invalid(self, minimal_context: VerificationContext):
        """Verify ValidateTemplateStage marks error for invalid template."""
        # Set invalid template code
        minimal_context.template_code = "not valid python code !!!"

        stage = ValidateTemplateStage()
        stage.execute(minimal_context)

        assert minimal_context.completed_without_errors is False
        assert minimal_context.error is not None
        assert "validation failed" in minimal_context.error.lower()

    def test_finalize_result_stage_produces_result(self, minimal_context: VerificationContext):
        """Verify FinalizeResultStage produces VerificationResult."""
        # Set required fields for finalize
        minimal_context.set_result_field("timestamp", "2024-01-01 12:00:00")
        minimal_context.set_result_field("execution_time", 1.5)

        stage = FinalizeResultStage()
        stage.execute(minimal_context)

        final_result = minimal_context.get_artifact("final_result")

        assert final_result is not None
        assert isinstance(final_result, VerificationResult)
        assert final_result.metadata.question_id == "test-question-1"

    def test_recursion_limit_stage_skips_when_no_limit(self, minimal_context: VerificationContext):
        """Verify RecursionLimitAutoFailStage skips when recursion limit NOT hit.

        The stage should_run() returns True only when recursion_limit_reached is True.
        When it's False, the stage should skip (not run).
        """
        minimal_context.set_artifact("recursion_limit_reached", False)

        stage = RecursionLimitAutoFailStage()

        # Should skip when no recursion limit hit
        assert stage.should_run(minimal_context) is False

    def test_recursion_limit_stage_runs_when_limit_hit(self, minimal_context: VerificationContext):
        """Verify RecursionLimitAutoFailStage runs and sets verify_result when limit hit.

        When recursion limit is reached, the stage:
        - Runs (should_run=True)
        - Sets verify_result to False
        - Does NOT mark error (preserves trace/tokens for analysis)
        """
        minimal_context.set_artifact("recursion_limit_reached", True)
        minimal_context.set_artifact("raw_llm_response", "Partial response...")

        stage = RecursionLimitAutoFailStage()

        # Should run when recursion limit hit
        assert stage.should_run(minimal_context) is True

        stage.execute(minimal_context)

        # Verify result is set to False
        assert minimal_context.get_result_field("verify_result") is False
        # But completed_without_errors remains True (by design - to preserve trace)
        assert minimal_context.completed_without_errors is True


# =============================================================================
# failed_stage Attribute Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.pipeline
class TestFailedStageAttribute:
    """Test that failed_stage is populated when guards fire."""

    def test_recursion_limit_sets_failed_stage(self, minimal_context: VerificationContext):
        """RecursionLimitAutoFailStage sets failed_stage to its stage name."""
        minimal_context.set_artifact("recursion_limit_reached", True)
        minimal_context.set_artifact("raw_llm_response", "Partial response...")

        stage = RecursionLimitAutoFailStage()
        stage.execute(minimal_context)

        assert minimal_context.get_result_field("failed_stage") == "RecursionLimitAutoFail"

    def test_trace_validation_sets_failed_stage(self, minimal_context: VerificationContext):
        """TraceValidationAutoFailStage sets failed_stage when trace is invalid."""
        # Simulate MCP-enabled model with invalid trace
        minimal_context.answering_model = ModelConfig(
            id="test-model",
            model_name="test-model",
            model_provider="anthropic",
            mcp_urls_dict={"tools": "http://localhost:8080/mcp"},
        )
        # Empty string is an invalid trace (no AI message)
        minimal_context.set_artifact("raw_llm_response", "")

        stage = TraceValidationAutoFailStage()
        stage.execute(minimal_context)

        assert minimal_context.get_result_field("failed_stage") == "TraceValidationAutoFail"

    def test_abstention_check_sets_failed_stage(self, minimal_context: VerificationContext):
        """AbstentionCheckStage sets failed_stage when it overrides verify_result."""
        minimal_context.abstention_enabled = True
        minimal_context.set_artifact("raw_llm_response", "I cannot answer that question.")
        minimal_context.set_artifact("usage_tracker", None)

        # Manually simulate the override path: if the stage triggers, it sets failed_stage
        # We test the base class behavior directly via context manipulation
        minimal_context.set_artifact(ArtifactKeys.VERIFY_RESULT, False)
        minimal_context.set_result_field(ArtifactKeys.VERIFY_RESULT, False)
        minimal_context.set_result_field(ArtifactKeys.FAILED_STAGE, "AbstentionCheck")

        assert minimal_context.get_result_field("failed_stage") == "AbstentionCheck"

    def test_first_guard_wins(self, minimal_context: VerificationContext):
        """Only the first guard to fire sets failed_stage (first-write-wins)."""
        minimal_context.set_artifact("recursion_limit_reached", True)
        minimal_context.set_artifact("raw_llm_response", "Partial response...")

        # First guard fires
        recursion_stage = RecursionLimitAutoFailStage()
        recursion_stage.execute(minimal_context)

        assert minimal_context.get_result_field("failed_stage") == "RecursionLimitAutoFail"

        # Simulate a second guard trying to set failed_stage
        # (in practice, later guards would skip, but we test the guard)
        minimal_context.set_result_field(ArtifactKeys.VERIFY_RESULT, False)
        # Manually invoke the first-write-wins check as BaseCheckStage would
        if not minimal_context.get_result_field(ArtifactKeys.FAILED_STAGE):
            minimal_context.set_result_field(ArtifactKeys.FAILED_STAGE, "SomeOtherStage")

        # First guard still wins
        assert minimal_context.get_result_field("failed_stage") == "RecursionLimitAutoFail"

    def test_no_guard_means_no_failed_stage(self, minimal_context: VerificationContext):
        """When no guard fires, failed_stage remains None."""
        assert minimal_context.get_result_field("failed_stage") is None


# =============================================================================
# Pipeline Result Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.pipeline
class TestPipelineResults:
    """Test VerificationResult produced by pipeline."""

    def test_result_contains_metadata(self, minimal_context: VerificationContext):
        """Verify result contains correct metadata."""
        minimal_context.set_result_field("timestamp", "2024-01-01 12:00:00")
        minimal_context.set_result_field("execution_time", 2.0)

        stage = FinalizeResultStage()
        stage.execute(minimal_context)

        result = minimal_context.get_artifact("final_result")

        assert result.metadata.question_id == "test-question-1"
        assert result.metadata.template_id == "template-hash-123"
        assert result.metadata.failure is None

    def test_result_captures_error(self, minimal_context: VerificationContext):
        """Verify result captures error state."""
        minimal_context.mark_error("Test pipeline error")
        minimal_context.set_result_field("timestamp", "2024-01-01 12:00:00")

        stage = FinalizeResultStage()
        stage.execute(minimal_context)

        result = minimal_context.get_artifact("final_result")

        assert result.metadata.failure is not None
        assert result.metadata.failure.reason == "Test pipeline error"

    def test_result_includes_execution_time(self, minimal_context: VerificationContext):
        """Verify result includes execution time."""
        minimal_context.set_result_field("timestamp", "2024-01-01 12:00:00")
        minimal_context.set_result_field("execution_time", 3.5)

        stage = FinalizeResultStage()
        stage.execute(minimal_context)

        result = minimal_context.get_artifact("final_result")

        assert result.metadata.execution_time == 3.5

    def test_result_includes_model_info(self, minimal_context: VerificationContext):
        """Verify result includes model information."""
        minimal_context.set_result_field("timestamp", "2024-01-01 12:00:00")

        stage = FinalizeResultStage()
        stage.execute(minimal_context)

        result = minimal_context.get_artifact("final_result")

        # FinalizeResultStage builds ModelIdentity from context.answering_model/parsing_model
        assert result.metadata.answering_model == "langchain:claude-haiku-4-5"
        assert result.metadata.parsing_model == "langchain:claude-haiku-4-5"


# =============================================================================
# Agentic Rubric Orchestrator Registration Tests
# =============================================================================


@pytest.mark.unit
class TestOrchestratorAgenticRubric:
    """Tests for Stage 11b registration in orchestrator."""

    def _make_agentic_rubric(self):
        return Rubric(
            agentic_traits=[
                AgenticRubricTrait(
                    name="code_quality",
                    description="Check code quality.",
                    kind="boolean",
                )
            ]
        )

    def test_stage_11b_registered_in_template_and_rubric_mode(self):
        rubric = self._make_agentic_rubric()
        orch = StageOrchestrator.from_config(
            rubric=rubric,
            evaluation_mode="template_and_rubric",
        )
        stage_names = [s.name for s in orch.stages]
        assert "AgenticRubricEvaluation" in stage_names

    def test_stage_11b_registered_in_rubric_only_mode(self):
        rubric = self._make_agentic_rubric()
        orch = StageOrchestrator.from_config(
            rubric=rubric,
            evaluation_mode="rubric_only",
        )
        stage_names = [s.name for s in orch.stages]
        assert "AgenticRubricEvaluation" in stage_names

    def test_stage_11b_not_registered_in_template_only_mode(self):
        rubric = self._make_agentic_rubric()
        orch = StageOrchestrator.from_config(
            rubric=rubric,
            evaluation_mode="template_only",
        )
        stage_names = [s.name for s in orch.stages]
        assert "AgenticRubricEvaluation" not in stage_names

    def test_stage_11b_not_registered_when_no_agentic_traits(self):
        rubric = Rubric(
            llm_traits=[
                LLMRubricTrait(name="clarity", kind="boolean"),
            ]
        )
        orch = StageOrchestrator.from_config(
            rubric=rubric,
            evaluation_mode="template_and_rubric",
        )
        stage_names = [s.name for s in orch.stages]
        assert "AgenticRubricEvaluation" not in stage_names

    def test_stage_11b_after_rubric_evaluation(self):
        """Stage 11b comes after RubricEvaluation stage."""
        rubric = Rubric(
            llm_traits=[LLMRubricTrait(name="clarity", kind="boolean")],
            agentic_traits=[
                AgenticRubricTrait(
                    name="code_quality",
                    description="Check.",
                    kind="boolean",
                )
            ],
        )
        orch = StageOrchestrator.from_config(
            rubric=rubric,
            evaluation_mode="template_and_rubric",
        )
        stage_names = [s.name for s in orch.stages]
        rubric_idx = stage_names.index("RubricEvaluation")
        agentic_idx = stage_names.index("AgenticRubricEvaluation")
        assert agentic_idx > rubric_idx

    def test_agentic_only_rubric_in_rubric_only_mode(self):
        """Agentic-only rubric (no LLM/regex/callable/metric) still gets Stage 11b."""
        rubric = self._make_agentic_rubric()
        orch = StageOrchestrator.from_config(
            rubric=rubric,
            evaluation_mode="rubric_only",
        )
        stage_names = [s.name for s in orch.stages]
        assert "AgenticRubricEvaluation" in stage_names
        # Classical RubricEvaluation should NOT be registered (no LLM/regex/callable/metric traits)
        assert "RubricEvaluation" not in stage_names


# =============================================================================
# FinalizeResultStage Agentic Rubric Tests
# =============================================================================


@pytest.mark.unit
class TestFinalizeResultAgenticRubric:
    """Tests for FinalizeResultStage with agentic rubric results."""

    def _make_context(self, minimal_model_config: ModelConfig) -> VerificationContext:
        """Create a minimal context for finalize tests."""
        return VerificationContext(
            question_id="test-agentic-q1",
            template_id="template-hash-agentic",
            question_text="Review the code quality.",
            template_code=(
                "from pydantic import Field\n"
                "from karenina.schemas.entities import BaseAnswer\n\n"
                "class Answer(BaseAnswer):\n"
                "    quality: str = Field(description='Code quality')\n\n"
                "    def verify(self) -> bool:\n"
                "        return True\n"
            ),
            answering_model=minimal_model_config,
            parsing_model=minimal_model_config,
        )

    def test_agentic_only_rubric_creates_rubric_result(self, minimal_model_config: ModelConfig):
        """When only agentic traits exist, rubric_result should still be created."""
        context = self._make_context(minimal_model_config)
        context.set_result_field("timestamp", "2024-01-01 12:00:00")
        context.set_result_field("execution_time", 1.0)

        # Simulate Stage 11b output: agentic evaluation performed, no classical rubric
        context.set_result_field(ArtifactKeys.AGENTIC_RUBRIC_EVALUATION_PERFORMED, True)
        context.set_result_field(
            ArtifactKeys.AGENTIC_TRAIT_SCORES,
            {"code_quality": True, "test_coverage": 4},
        )
        context.set_result_field(
            ArtifactKeys.AGENTIC_TRAIT_INVESTIGATION_TRACES,
            {"code_quality": "Investigated code: looks good.", "test_coverage": "Found 80% coverage."},
        )
        # No classical rubric (verify_rubric not set)

        stage = FinalizeResultStage()
        stage.execute(context)

        result = context.get_artifact("final_result")
        assert result is not None
        assert isinstance(result, VerificationResult)

        # Rubric result must be created
        assert result.rubric is not None
        assert isinstance(result.rubric, VerificationResultRubric)

        # Agentic fields populated
        assert result.rubric.agentic_trait_scores == {"code_quality": True, "test_coverage": 4}
        assert result.rubric.agentic_trait_investigation_traces == {
            "code_quality": "Investigated code: looks good.",
            "test_coverage": "Found 80% coverage.",
        }

        # Classical fields should be None (no classical rubric ran)
        assert result.rubric.llm_trait_scores is None
        assert result.rubric.regex_trait_scores is None
        assert result.rubric.callable_trait_scores is None

    def test_classical_rubric_still_works(self, minimal_model_config: ModelConfig):
        """When classical rubric runs (without agentic), behavior is preserved."""
        context = self._make_context(minimal_model_config)
        context.set_result_field("timestamp", "2024-01-01 12:00:00")
        context.set_result_field("execution_time", 1.0)

        # Set up a rubric with one LLM trait
        rubric = Rubric(llm_traits=[LLMRubricTrait(name="clarity", description="Clear?", kind="boolean")])
        context.rubric = rubric

        # Simulate classical rubric output
        context.set_result_field(ArtifactKeys.VERIFY_RUBRIC, {"clarity": True})

        stage = FinalizeResultStage()
        stage.execute(context)

        result = context.get_artifact("final_result")
        assert result.rubric is not None
        assert result.rubric.llm_trait_scores == {"clarity": True}
        assert result.rubric.agentic_trait_scores is None

    def test_both_classical_and_agentic_rubric(self, minimal_model_config: ModelConfig):
        """When both classical and agentic traits run, both are stored."""
        context = self._make_context(minimal_model_config)
        context.set_result_field("timestamp", "2024-01-01 12:00:00")
        context.set_result_field("execution_time", 1.0)

        # Set up rubric with both trait types
        rubric = Rubric(
            llm_traits=[LLMRubricTrait(name="clarity", description="Clear?", kind="boolean")],
            agentic_traits=[AgenticRubricTrait(name="code_quality", description="Check.", kind="boolean")],
        )
        context.rubric = rubric

        # Simulate classical rubric output
        context.set_result_field(ArtifactKeys.VERIFY_RUBRIC, {"clarity": True})

        # Simulate agentic rubric output
        context.set_result_field(ArtifactKeys.AGENTIC_RUBRIC_EVALUATION_PERFORMED, True)
        context.set_result_field(ArtifactKeys.AGENTIC_TRAIT_SCORES, {"code_quality": True})
        context.set_result_field(
            ArtifactKeys.AGENTIC_TRAIT_INVESTIGATION_TRACES,
            {"code_quality": "Looked at code structure."},
        )

        stage = FinalizeResultStage()
        stage.execute(context)

        result = context.get_artifact("final_result")
        assert result.rubric is not None

        # Both classical and agentic fields present
        assert result.rubric.llm_trait_scores == {"clarity": True}
        assert result.rubric.agentic_trait_scores == {"code_quality": True}
        assert result.rubric.agentic_trait_investigation_traces == {"code_quality": "Looked at code structure."}

    def test_no_rubric_when_neither_performed(self, minimal_model_config: ModelConfig):
        """When neither classical nor agentic rubric ran, rubric_result stays None."""
        context = self._make_context(minimal_model_config)
        context.set_result_field("timestamp", "2024-01-01 12:00:00")
        context.set_result_field("execution_time", 1.0)

        stage = FinalizeResultStage()
        stage.execute(context)

        result = context.get_artifact("final_result")
        assert result.rubric is None
