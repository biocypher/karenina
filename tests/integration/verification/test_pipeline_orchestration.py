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
from karenina.schemas.domain import LLMRubricTrait, Rubric
from karenina.schemas.workflow import (
    ModelConfig,
    VerificationResult,
    VerificationResultMetadata,
    VerificationResultTemplate,
)

# =============================================================================
# Test Helpers - Create Valid VerificationResult
# =============================================================================


def create_minimal_result(context: VerificationContext) -> VerificationResult:
    """Create a minimal valid VerificationResult from context."""
    result_id = VerificationResultMetadata.compute_result_id(
        question_id=context.question_id,
        answering_model="test/model",
        parsing_model="test/model",
        timestamp="2024-01-01 12:00:00",
    )

    metadata = VerificationResultMetadata(
        question_id=context.question_id,
        template_id=context.template_id,
        completed_without_errors=context.completed_without_errors,
        error=context.error,
        question_text=context.question_text,
        answering_model="test/model",
        parsing_model="test/model",
        execution_time=1.0,
        timestamp="2024-01-01 12:00:00",
        result_id=result_id,
    )
    template = VerificationResultTemplate(raw_llm_response="")
    return VerificationResult(metadata=metadata, template=template)


class MockFinalizeStage(BaseVerificationStage):
    """Mock FinalizeResultStage that produces valid VerificationResult."""

    def __init__(self):
        self.executed = False

    @property
    def name(self) -> str:
        return "FinalizeResult"

    @property
    def produces(self) -> list[str]:
        return ["final_result"]

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
        id="test-model",
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
from karenina.schemas.domain import BaseAnswer

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
        assert result.metadata.completed_without_errors is False

    def test_exception_caught_and_marked(self, minimal_context: VerificationContext):
        """Verify stage exception is caught and marked on context."""
        exception_stage = MockExceptionStage()
        orchestrator = StageOrchestrator([exception_stage, MockFinalizeStage()])
        result = orchestrator.execute(minimal_context)

        assert minimal_context.completed_without_errors is False
        assert "RuntimeError" in minimal_context.error
        assert "Test exception" in minimal_context.error
        assert result.metadata.error is not None

    def test_finalize_always_runs_after_error(self, minimal_context: VerificationContext):
        """Verify FinalizeResultStage runs even after stage error."""
        finalizer = MockFinalizeStage()
        error_stage = MockErrorStage()
        orchestrator = StageOrchestrator([error_stage, finalizer])
        orchestrator.execute(minimal_context)

        assert finalizer.executed is True

    def test_missing_final_result_raises(self, minimal_context: VerificationContext):
        """Verify missing final_result artifact raises RuntimeError."""

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

        orchestrator = StageOrchestrator([NoOutputFinalize()])

        with pytest.raises(RuntimeError, match="did not produce a final_result"):
            orchestrator.execute(minimal_context)


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

        class NoOpFinalize(BaseVerificationStage):
            @property
            def name(self) -> str:
                return "FinalizeResult"

            @property
            def produces(self) -> list[str]:
                return ["final_result"]

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

        class NoOpFinalize(BaseVerificationStage):
            @property
            def name(self) -> str:
                return "FinalizeResult"

            @property
            def produces(self) -> list[str]:
                return ["final_result"]

            def execute(self, context: VerificationContext) -> None:  # noqa: ARG002
                pass

        orchestrator = StageOrchestrator([consumer, NoOpFinalize()])

        errors = orchestrator.validate_dependencies()

        assert len(errors) == 1
        assert "missing_data" in errors[0]

    def test_execute_raises_on_invalid_dependencies(self, minimal_context: VerificationContext):
        """Verify execute raises ValueError for invalid dependencies."""
        consumer = MockConsumerStage("Consumer", "missing_data", "result")

        class NoOpFinalize(BaseVerificationStage):
            @property
            def name(self) -> str:
                return "FinalizeResult"

            @property
            def produces(self) -> list[str]:
                return ["final_result"]

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
        assert result.metadata.completed_without_errors is True

    def test_result_captures_error(self, minimal_context: VerificationContext):
        """Verify result captures error state."""
        minimal_context.mark_error("Test pipeline error")
        minimal_context.set_result_field("timestamp", "2024-01-01 12:00:00")

        stage = FinalizeResultStage()
        stage.execute(minimal_context)

        result = minimal_context.get_artifact("final_result")

        assert result.metadata.completed_without_errors is False
        assert result.metadata.error == "Test pipeline error"

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
        minimal_context.set_artifact("answering_model_str", "anthropic/claude-haiku-4-5")
        minimal_context.set_artifact("parsing_model_str", "anthropic/claude-haiku-4-5")
        minimal_context.set_result_field("timestamp", "2024-01-01 12:00:00")

        stage = FinalizeResultStage()
        stage.execute(minimal_context)

        result = minimal_context.get_artifact("final_result")

        assert result.metadata.answering_model == "anthropic/claude-haiku-4-5"
        assert result.metadata.parsing_model == "anthropic/claude-haiku-4-5"
