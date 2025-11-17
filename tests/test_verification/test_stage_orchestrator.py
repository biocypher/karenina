"""Tests for StageOrchestrator.

Tests the orchestration of verification pipeline stages, including:
- Stage selection based on configuration
- Stage ordering and dependencies
- Error handling and recovery
- Conditional stage execution
"""

from unittest.mock import Mock

import pytest

from karenina.benchmark.verification.stage import VerificationContext
from karenina.benchmark.verification.stage_orchestrator import StageOrchestrator
from karenina.benchmark.verification.stages import (
    AbstentionCheckStage,
    DeepJudgmentAutoFailStage,
    EmbeddingCheckStage,
    FinalizeResultStage,
    GenerateAnswerStage,
    ParseTemplateStage,
    RubricEvaluationStage,
    ValidateTemplateStage,
    VerifyTemplateStage,
)
from karenina.schemas import ModelConfig, VerificationResult
from karenina.schemas.domain import LLMRubricTrait, Rubric


class TestStageOrchestratorConfiguration:
    """Test stage selection based on configuration."""

    def test_template_only_mode_default(self, answering_model: ModelConfig, parsing_model: ModelConfig) -> None:
        """Test default configuration (template-only, no special features)."""

        orchestrator = StageOrchestrator.from_config(
            answering_model=answering_model,
            parsing_model=parsing_model,
        )

        # Verify stage composition
        stage_types = [type(stage) for stage in orchestrator.stages]
        expected_stages = [
            ValidateTemplateStage,
            GenerateAnswerStage,
            ParseTemplateStage,
            VerifyTemplateStage,
            EmbeddingCheckStage,  # Always included (has should_run logic)
            FinalizeResultStage,
        ]

        assert stage_types == expected_stages

    def test_template_with_rubric_mode(self, answering_model: ModelConfig, parsing_model: ModelConfig) -> None:
        """Test template + rubric mode (requires evaluation_mode='template_and_rubric')."""
        rubric = Rubric(
            llm_traits=[
                LLMRubricTrait(name="Clarity", description="Response clarity", kind="score", min_score=1, max_score=5)
            ]
        )

        orchestrator = StageOrchestrator.from_config(
            answering_model=answering_model,
            parsing_model=parsing_model,
            rubric=rubric,
            evaluation_mode="template_and_rubric",
        )

        # Verify rubric evaluation stage is included
        stage_types = [type(stage) for stage in orchestrator.stages]
        assert RubricEvaluationStage in stage_types

        # Verify order (rubric before finalize)
        rubric_idx = stage_types.index(RubricEvaluationStage)
        finalize_idx = stage_types.index(FinalizeResultStage)
        assert rubric_idx < finalize_idx

    def test_abstention_enabled(self, answering_model: ModelConfig, parsing_model: ModelConfig) -> None:
        """Test abstention detection enabled."""
        orchestrator = StageOrchestrator.from_config(
            answering_model=answering_model,
            parsing_model=parsing_model,
            abstention_enabled=True,
        )

        # Verify abstention stage is included
        stage_types = [type(stage) for stage in orchestrator.stages]
        assert AbstentionCheckStage in stage_types

    def test_deep_judgment_enabled(self, answering_model: ModelConfig, parsing_model: ModelConfig) -> None:
        """Test deep-judgment auto-fail enabled."""
        orchestrator = StageOrchestrator.from_config(
            answering_model=answering_model,
            parsing_model=parsing_model,
            deep_judgment_enabled=True,
        )

        # Verify deep-judgment stage is included
        stage_types = [type(stage) for stage in orchestrator.stages]
        assert DeepJudgmentAutoFailStage in stage_types

    def test_all_features_enabled(self, answering_model: ModelConfig, parsing_model: ModelConfig) -> None:
        """Test all features enabled together (requires evaluation_mode='template_and_rubric')."""
        rubric = Rubric(
            llm_traits=[
                LLMRubricTrait(name="Clarity", description="Response clarity", kind="score", min_score=1, max_score=5)
            ]
        )

        orchestrator = StageOrchestrator.from_config(
            answering_model=answering_model,
            parsing_model=parsing_model,
            rubric=rubric,
            abstention_enabled=True,
            deep_judgment_enabled=True,
            evaluation_mode="template_and_rubric",
        )

        # Verify all optional stages are included
        stage_types = [type(stage) for stage in orchestrator.stages]
        assert RubricEvaluationStage in stage_types
        assert AbstentionCheckStage in stage_types
        assert DeepJudgmentAutoFailStage in stage_types

    def test_empty_rubric_not_included(self, answering_model: ModelConfig, parsing_model: ModelConfig) -> None:
        """Test that rubric stage is not included if rubric has no traits."""
        empty_rubric = Rubric(llm_traits=[], manual_traits=[], metric_traits=[])

        orchestrator = StageOrchestrator.from_config(
            answering_model=answering_model,
            parsing_model=parsing_model,
            rubric=empty_rubric,
            evaluation_mode="template_and_rubric",
        )

        # Verify rubric stage is NOT included
        stage_types = [type(stage) for stage in orchestrator.stages]
        assert RubricEvaluationStage not in stage_types


class TestEvaluationModeStageSelection:
    """Test stage selection based on evaluation_mode."""

    def test_template_only_mode_excludes_rubric(self, answering_model: ModelConfig, parsing_model: ModelConfig) -> None:
        """Test that template_only mode excludes rubric stage even if rubric provided."""
        rubric = Rubric(
            llm_traits=[
                LLMRubricTrait(name="Clarity", description="Response clarity", kind="score", min_score=1, max_score=5)
            ]
        )

        orchestrator = StageOrchestrator.from_config(
            answering_model=answering_model,
            parsing_model=parsing_model,
            rubric=rubric,
            evaluation_mode="template_only",  # Explicit template_only
        )

        # Verify rubric stage is NOT included in template_only mode
        stage_types = [type(stage) for stage in orchestrator.stages]
        assert RubricEvaluationStage not in stage_types

        # Verify template stages ARE included
        assert ValidateTemplateStage in stage_types
        assert ParseTemplateStage in stage_types
        assert VerifyTemplateStage in stage_types

    def test_template_and_rubric_mode_includes_both(
        self, answering_model: ModelConfig, parsing_model: ModelConfig
    ) -> None:
        """Test that template_and_rubric mode includes both template and rubric stages."""
        rubric = Rubric(
            llm_traits=[
                LLMRubricTrait(name="Clarity", description="Response clarity", kind="score", min_score=1, max_score=5)
            ]
        )

        orchestrator = StageOrchestrator.from_config(
            answering_model=answering_model,
            parsing_model=parsing_model,
            rubric=rubric,
            evaluation_mode="template_and_rubric",
        )

        stage_types = [type(stage) for stage in orchestrator.stages]

        # Verify template stages are included
        assert ValidateTemplateStage in stage_types
        assert ParseTemplateStage in stage_types
        assert VerifyTemplateStage in stage_types

        # Verify rubric stage is included
        assert RubricEvaluationStage in stage_types

    def test_rubric_only_mode_skips_template_stages(
        self, answering_model: ModelConfig, parsing_model: ModelConfig
    ) -> None:
        """Test that rubric_only mode skips template verification stages."""
        rubric = Rubric(
            llm_traits=[
                LLMRubricTrait(name="Clarity", description="Response clarity", kind="score", min_score=1, max_score=5)
            ]
        )

        orchestrator = StageOrchestrator.from_config(
            answering_model=answering_model,
            parsing_model=parsing_model,
            rubric=rubric,
            evaluation_mode="rubric_only",
        )

        stage_types = [type(stage) for stage in orchestrator.stages]

        # Verify template stages are NOT included
        assert ValidateTemplateStage not in stage_types
        assert ParseTemplateStage not in stage_types
        assert VerifyTemplateStage not in stage_types
        assert EmbeddingCheckStage not in stage_types
        assert DeepJudgmentAutoFailStage not in stage_types

        # Verify minimal rubric-only stages are included
        assert GenerateAnswerStage in stage_types
        assert RubricEvaluationStage in stage_types
        assert FinalizeResultStage in stage_types

    def test_rubric_only_mode_with_abstention(self, answering_model: ModelConfig, parsing_model: ModelConfig) -> None:
        """Test that rubric_only mode can include abstention check."""
        rubric = Rubric(
            llm_traits=[
                LLMRubricTrait(name="Clarity", description="Response clarity", kind="score", min_score=1, max_score=5)
            ]
        )

        orchestrator = StageOrchestrator.from_config(
            answering_model=answering_model,
            parsing_model=parsing_model,
            rubric=rubric,
            abstention_enabled=True,
            evaluation_mode="rubric_only",
        )

        stage_types = [type(stage) for stage in orchestrator.stages]

        # Verify template stages are NOT included
        assert ValidateTemplateStage not in stage_types
        assert ParseTemplateStage not in stage_types

        # Verify rubric-only stages with abstention
        assert GenerateAnswerStage in stage_types
        assert AbstentionCheckStage in stage_types
        assert RubricEvaluationStage in stage_types
        assert FinalizeResultStage in stage_types

    def test_rubric_only_mode_stage_order(self, answering_model: ModelConfig, parsing_model: ModelConfig) -> None:
        """Test that rubric_only mode stages are in correct order."""
        rubric = Rubric(
            llm_traits=[
                LLMRubricTrait(name="Clarity", description="Response clarity", kind="score", min_score=1, max_score=5)
            ]
        )

        orchestrator = StageOrchestrator.from_config(
            answering_model=answering_model,
            parsing_model=parsing_model,
            rubric=rubric,
            abstention_enabled=True,
            evaluation_mode="rubric_only",
        )

        stage_types = [type(stage) for stage in orchestrator.stages]

        # Verify order: Generate → Abstention → Rubric → Finalize
        generate_idx = stage_types.index(GenerateAnswerStage)
        abstention_idx = stage_types.index(AbstentionCheckStage)
        rubric_idx = stage_types.index(RubricEvaluationStage)
        finalize_idx = stage_types.index(FinalizeResultStage)

        assert generate_idx < abstention_idx < rubric_idx < finalize_idx


class TestStageOrdering:
    """Test that stages are executed in correct order."""

    def test_core_stage_order(self, answering_model: ModelConfig, parsing_model: ModelConfig) -> None:
        """Test that core stages are in correct order."""
        orchestrator = StageOrchestrator.from_config(
            answering_model=answering_model,
            parsing_model=parsing_model,
        )

        stage_types = [type(stage) for stage in orchestrator.stages]

        # Verify order of core stages
        validate_idx = stage_types.index(ValidateTemplateStage)
        generate_idx = stage_types.index(GenerateAnswerStage)
        parse_idx = stage_types.index(ParseTemplateStage)
        verify_idx = stage_types.index(VerifyTemplateStage)
        finalize_idx = stage_types.index(FinalizeResultStage)

        # Validate < Generate < Parse < Verify < Finalize
        assert validate_idx < generate_idx
        assert generate_idx < parse_idx
        assert parse_idx < verify_idx
        assert verify_idx < finalize_idx

    def test_finalize_always_last(self, answering_model: ModelConfig, parsing_model: ModelConfig) -> None:
        """Test that FinalizeResultStage is always last."""
        rubric = Rubric(
            llm_traits=[
                LLMRubricTrait(name="Clarity", description="Response clarity", kind="score", min_score=1, max_score=5)
            ]
        )

        orchestrator = StageOrchestrator.from_config(
            answering_model=answering_model,
            parsing_model=parsing_model,
            rubric=rubric,
            abstention_enabled=True,
            deep_judgment_enabled=True,
        )

        # Verify FinalizeResultStage is last
        assert isinstance(orchestrator.stages[-1], FinalizeResultStage)

    def test_embedding_after_verify(self, answering_model: ModelConfig, parsing_model: ModelConfig) -> None:
        """Test that EmbeddingCheckStage comes after VerifyTemplateStage."""
        orchestrator = StageOrchestrator.from_config(
            answering_model=answering_model,
            parsing_model=parsing_model,
        )

        stage_types = [type(stage) for stage in orchestrator.stages]
        verify_idx = stage_types.index(VerifyTemplateStage)
        embedding_idx = stage_types.index(EmbeddingCheckStage)

        assert verify_idx < embedding_idx


class TestStageExecution:
    """Test stage execution logic."""

    def test_successful_execution_creates_result(self, basic_context: VerificationContext) -> None:
        """Test that successful pipeline execution creates a VerificationResult."""
        # Create minimal orchestrator (just validate and finalize for this test)
        from karenina.benchmark.verification.stages.finalize_result import FinalizeResultStage
        from karenina.benchmark.verification.stages.validate_template import ValidateTemplateStage

        orchestrator = StageOrchestrator(stages=[ValidateTemplateStage(), FinalizeResultStage()])

        # Set up minimal context with required artifacts
        basic_context.set_artifact("answering_model_str", "openai/gpt-4o-mini")
        basic_context.set_artifact("parsing_model_str", "openai/gpt-4o-mini")

        # Execute pipeline
        result = orchestrator.execute(basic_context)

        # Verify result was created
        assert isinstance(result, VerificationResult)
        assert result.question_id == basic_context.question_id
        assert result.metadata.template_id == basic_context.metadata.template_id

    def test_error_in_stage_continues_to_finalize(self, basic_context: VerificationContext) -> None:
        """Test that errors in stages don't prevent FinalizeResultStage from running."""
        # Create mock stage that raises an exception
        mock_stage = Mock()
        mock_stage.name = "FailingStage"
        mock_stage.requires = []
        mock_stage.produces = ["nothing"]
        mock_stage.should_run.return_value = True
        mock_stage.execute.side_effect = RuntimeError("Stage failed!")

        finalize_stage = FinalizeResultStage()
        orchestrator = StageOrchestrator(stages=[mock_stage, finalize_stage])

        # Set up minimal context
        basic_context.set_artifact("answering_model_str", "openai/gpt-4o-mini")
        basic_context.set_artifact("parsing_model_str", "openai/gpt-4o-mini")

        # Execute pipeline
        result = orchestrator.execute(basic_context)

        # Verify result was still created despite error
        assert isinstance(result, VerificationResult)
        assert result.completed_without_errors is False
        assert "FailingStage raised exception" in result.error

    def test_should_run_false_skips_stage(self, basic_context: VerificationContext) -> None:
        """Test that stages with should_run=False are skipped."""
        # Create mock stage that should not run
        mock_stage = Mock()
        mock_stage.name = "SkippedStage"
        mock_stage.requires = []
        mock_stage.produces = ["nothing"]
        mock_stage.should_run.return_value = False

        finalize_stage = FinalizeResultStage()
        orchestrator = StageOrchestrator(stages=[mock_stage, finalize_stage])

        # Set up minimal context
        basic_context.set_artifact("answering_model_str", "openai/gpt-4o-mini")
        basic_context.set_artifact("parsing_model_str", "openai/gpt-4o-mini")

        # Execute pipeline
        orchestrator.execute(basic_context)

        # Verify stage was not executed
        mock_stage.should_run.assert_called_once_with(basic_context)
        mock_stage.execute.assert_not_called()

    def test_execution_time_tracked(self, basic_context: VerificationContext) -> None:
        """Test that execution time is tracked."""
        orchestrator = StageOrchestrator(stages=[FinalizeResultStage()])

        # Set up minimal context
        basic_context.set_artifact("answering_model_str", "openai/gpt-4o-mini")
        basic_context.set_artifact("parsing_model_str", "openai/gpt-4o-mini")

        # Execute pipeline
        result = orchestrator.execute(basic_context)

        # Verify timing was tracked
        assert result.metadata.execution_time >= 0.0
        assert result.timestamp != ""


class TestDependencyValidation:
    """Test stage dependency validation."""

    def test_valid_dependencies(self, answering_model: ModelConfig, parsing_model: ModelConfig) -> None:
        """Test that valid stage dependencies pass validation."""
        orchestrator = StageOrchestrator.from_config(
            answering_model=answering_model,
            parsing_model=parsing_model,
        )

        # Validate dependencies
        errors = orchestrator.validate_dependencies()
        assert errors == []

    def test_missing_finalize_raises_error(self, basic_context: VerificationContext) -> None:
        """Test that missing FinalizeResultStage causes error."""
        # Create orchestrator without FinalizeResultStage
        orchestrator = StageOrchestrator(stages=[ValidateTemplateStage()])

        # Set up minimal context
        basic_context.set_artifact("answering_model_str", "openai/gpt-4o-mini")
        basic_context.set_artifact("parsing_model_str", "openai/gpt-4o-mini")

        # Execute should raise error because no result produced
        with pytest.raises(RuntimeError, match="FinalizeResultStage did not produce a final_result"):
            orchestrator.execute(basic_context)


class TestErrorRecovery:
    """Test error recovery and resilience."""

    def test_context_error_allows_finalize(self, basic_context: VerificationContext) -> None:
        """Test that context.mark_error() doesn't prevent FinalizeResultStage."""
        # Create mock stage that marks error
        mock_stage = Mock()
        mock_stage.name = "ErrorMarkingStage"
        mock_stage.requires = []
        mock_stage.produces = []
        mock_stage.should_run.return_value = True

        def mark_error(context: VerificationContext) -> None:
            context.mark_error("Something went wrong")

        mock_stage.execute.side_effect = mark_error

        finalize_stage = FinalizeResultStage()
        orchestrator = StageOrchestrator(stages=[mock_stage, finalize_stage])

        # Set up minimal context
        basic_context.set_artifact("answering_model_str", "openai/gpt-4o-mini")
        basic_context.set_artifact("parsing_model_str", "openai/gpt-4o-mini")

        # Execute pipeline
        result = orchestrator.execute(basic_context)

        # Verify result was created with error
        assert isinstance(result, VerificationResult)
        assert result.completed_without_errors is False
        assert result.error == "Something went wrong"

    def test_multiple_errors_captured(self, basic_context: VerificationContext) -> None:
        """Test that last error overwrites previous errors (mark_error behavior)."""
        # Create two mock stages that fail
        mock_stage1 = Mock()
        mock_stage1.name = "FailingStage1"
        mock_stage1.requires = []
        mock_stage1.produces = []
        mock_stage1.should_run.return_value = True
        mock_stage1.execute.side_effect = RuntimeError("First error")

        mock_stage2 = Mock()
        mock_stage2.name = "FailingStage2"
        mock_stage2.requires = []
        mock_stage2.produces = []
        mock_stage2.should_run.return_value = True
        mock_stage2.execute.side_effect = RuntimeError("Second error")

        finalize_stage = FinalizeResultStage()
        orchestrator = StageOrchestrator(stages=[mock_stage1, mock_stage2, finalize_stage])

        # Set up minimal context
        basic_context.set_artifact("answering_model_str", "openai/gpt-4o-mini")
        basic_context.set_artifact("parsing_model_str", "openai/gpt-4o-mini")

        # Execute pipeline
        result = orchestrator.execute(basic_context)

        # Verify result contains an error (last one overwrites first)
        assert isinstance(result, VerificationResult)
        assert result.completed_without_errors is False
        # Note: mark_error() overwrites the previous error, so we get the last error
        assert "FailingStage2 raised exception" in result.error


class TestOrchestratorRepr:
    """Test string representation."""

    def test_repr(self, answering_model: ModelConfig, parsing_model: ModelConfig) -> None:
        """Test __repr__ method."""
        orchestrator = StageOrchestrator.from_config(
            answering_model=answering_model,
            parsing_model=parsing_model,
        )

        repr_str = repr(orchestrator)
        assert "StageOrchestrator" in repr_str
        assert "ValidateTemplate" in repr_str
        assert "FinalizeResult" in repr_str
