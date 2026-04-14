"""Tests for AgenticRubricEvaluationStage."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from karenina.benchmark.verification.stages.core.base import (
    ArtifactKeys,
    VerificationContext,
)
from karenina.schemas.config.models import ModelConfig
from karenina.schemas.entities.rubric import AgenticRubricTrait, Rubric


def _make_trait(name="code_quality", kind="boolean"):
    """Build a minimal AgenticRubricTrait for testing."""
    return AgenticRubricTrait(
        name=name,
        description="Evaluate code quality.",
        kind=kind,
        higher_is_better=True,
    )


def _make_context(
    agentic_traits=None,
    raw_llm_response="def foo(): pass",
    workspace_path=None,
    error=None,
):
    """Build a minimal VerificationContext for testing.

    Uses the claude_agent_sdk interface so that AdapterRegistry validation
    passes for AgenticRubricTrait model_override checks.
    """
    rubric = Rubric(agentic_traits=agentic_traits or [])
    parsing_model = ModelConfig(
        id="test",
        model_name="test-model",
        model_provider="anthropic",
        interface="claude_agent_sdk",
    )
    ctx = VerificationContext(
        question_id="q1",
        template_id="t1",
        question_text="Write a function.",
        template_code="class Answer: pass",
        answering_model=parsing_model,
        parsing_model=parsing_model,
        rubric=rubric,
    )
    ctx.set_artifact(ArtifactKeys.RAW_LLM_RESPONSE, raw_llm_response)
    if workspace_path:
        ctx.workspace_path = Path(workspace_path)
    if error:
        ctx.mark_error(error)
    return ctx


@pytest.mark.unit
class TestAgenticRubricEvaluationShouldRun:
    """Tests for AgenticRubricEvaluationStage.should_run()."""

    def test_should_run_false_when_no_agentic_traits(self):
        from karenina.benchmark.verification.stages.pipeline.agentic_rubric_evaluation import (
            AgenticRubricEvaluationStage,
        )

        stage = AgenticRubricEvaluationStage()
        ctx = _make_context(agentic_traits=[])
        assert stage.should_run(ctx) is False

    def test_should_run_true_when_agentic_traits_present(self):
        from karenina.benchmark.verification.stages.pipeline.agentic_rubric_evaluation import (
            AgenticRubricEvaluationStage,
        )

        stage = AgenticRubricEvaluationStage()
        ctx = _make_context(agentic_traits=[_make_trait()])
        assert stage.should_run(ctx) is True

    def test_should_run_false_when_error_set(self):
        from karenina.benchmark.verification.stages.pipeline.agentic_rubric_evaluation import (
            AgenticRubricEvaluationStage,
        )

        stage = AgenticRubricEvaluationStage()
        ctx = _make_context(agentic_traits=[_make_trait()], error="prior failure")
        assert stage.should_run(ctx) is False

    def test_should_run_false_when_no_rubric(self):
        from karenina.benchmark.verification.stages.pipeline.agentic_rubric_evaluation import (
            AgenticRubricEvaluationStage,
        )

        stage = AgenticRubricEvaluationStage()
        ctx = _make_context(agentic_traits=[_make_trait()])
        ctx.rubric = None
        assert stage.should_run(ctx) is False


@pytest.mark.unit
class TestAgenticRubricEvaluationProperties:
    """Tests for stage name, requires, and produces properties."""

    def test_name(self):
        from karenina.benchmark.verification.stages.pipeline.agentic_rubric_evaluation import (
            AgenticRubricEvaluationStage,
        )

        stage = AgenticRubricEvaluationStage()
        assert stage.name == "AgenticRubricEvaluation"

    def test_requires(self):
        from karenina.benchmark.verification.stages.pipeline.agentic_rubric_evaluation import (
            AgenticRubricEvaluationStage,
        )

        stage = AgenticRubricEvaluationStage()
        assert ArtifactKeys.RAW_LLM_RESPONSE in stage.requires

    def test_produces(self):
        from karenina.benchmark.verification.stages.pipeline.agentic_rubric_evaluation import (
            AgenticRubricEvaluationStage,
        )

        stage = AgenticRubricEvaluationStage()
        assert ArtifactKeys.AGENTIC_RUBRIC_EVALUATION_PERFORMED in stage.produces
        assert ArtifactKeys.AGENTIC_TRAIT_SCORES in stage.produces
        assert ArtifactKeys.AGENTIC_TRAIT_INVESTIGATION_TRACES in stage.produces


@pytest.mark.unit
class TestAgenticRubricEvaluationExecute:
    """Tests for AgenticRubricEvaluationStage.execute()."""

    def test_execute_individual_strategy(self, monkeypatch):
        from karenina.benchmark.verification.stages.pipeline import (
            agentic_rubric_evaluation as mod,
        )

        fake_evaluator = MagicMock()
        fake_evaluator.evaluate_trait.return_value = (True, "investigation trace")

        monkeypatch.setattr(
            mod,
            "AgenticTraitEvaluator",
            lambda model_config, **kwargs: fake_evaluator,  # noqa: ARG005
        )

        ctx = _make_context(agentic_traits=[_make_trait()])
        stage = mod.AgenticRubricEvaluationStage()
        stage.execute(ctx)

        scores = ctx.get_artifact(ArtifactKeys.AGENTIC_TRAIT_SCORES)
        assert scores == {"code_quality": True}

        traces = ctx.get_artifact(ArtifactKeys.AGENTIC_TRAIT_INVESTIGATION_TRACES)
        assert "code_quality" in traces
        assert traces["code_quality"] == "investigation trace"

        performed = ctx.get_artifact(ArtifactKeys.AGENTIC_RUBRIC_EVALUATION_PERFORMED)
        assert performed is True

    def test_execute_failed_trait_stores_none(self, monkeypatch):
        from karenina.benchmark.verification.stages.pipeline import (
            agentic_rubric_evaluation as mod,
        )

        fake_evaluator = MagicMock()
        fake_evaluator.evaluate_trait.return_value = (None, None)

        monkeypatch.setattr(
            mod,
            "AgenticTraitEvaluator",
            lambda model_config, **kwargs: fake_evaluator,  # noqa: ARG005
        )

        ctx = _make_context(agentic_traits=[_make_trait()])
        stage = mod.AgenticRubricEvaluationStage()
        stage.execute(ctx)

        scores = ctx.get_artifact(ArtifactKeys.AGENTIC_TRAIT_SCORES)
        assert scores == {"code_quality": None}

        # No pipeline error should be set
        assert ctx.error is None

    def test_execute_multiple_traits(self, monkeypatch):
        from karenina.benchmark.verification.stages.pipeline import (
            agentic_rubric_evaluation as mod,
        )

        call_count = {"n": 0}

        def fake_evaluate_trait(trait, question_text, raw_llm_response, workspace_path):
            call_count["n"] += 1
            if trait.name == "code_quality":
                return True, "trace_cq"
            return False, "trace_safety"

        fake_evaluator = MagicMock()
        fake_evaluator.evaluate_trait.side_effect = fake_evaluate_trait

        monkeypatch.setattr(
            mod,
            "AgenticTraitEvaluator",
            lambda model_config, **kwargs: fake_evaluator,  # noqa: ARG005
        )

        traits = [
            _make_trait(name="code_quality"),
            _make_trait(name="safety"),
        ]
        ctx = _make_context(agentic_traits=traits)
        stage = mod.AgenticRubricEvaluationStage()
        stage.execute(ctx)

        scores = ctx.get_artifact(ArtifactKeys.AGENTIC_TRAIT_SCORES)
        assert scores["code_quality"] is True
        assert scores["safety"] is False

        traces = ctx.get_artifact(ArtifactKeys.AGENTIC_TRAIT_INVESTIGATION_TRACES)
        assert traces["code_quality"] == "trace_cq"
        assert traces["safety"] == "trace_safety"

    def test_execute_stores_results_in_result_builder(self, monkeypatch):
        from karenina.benchmark.verification.stages.pipeline import (
            agentic_rubric_evaluation as mod,
        )

        fake_evaluator = MagicMock()
        fake_evaluator.evaluate_trait.return_value = (True, "investigation trace")

        monkeypatch.setattr(
            mod,
            "AgenticTraitEvaluator",
            lambda model_config, **kwargs: fake_evaluator,  # noqa: ARG005
        )

        ctx = _make_context(agentic_traits=[_make_trait()])
        stage = mod.AgenticRubricEvaluationStage()
        stage.execute(ctx)

        # Result builder should also have the scores
        assert ctx.get_result_field(ArtifactKeys.AGENTIC_RUBRIC_EVALUATION_PERFORMED) is True
        assert ctx.get_result_field(ArtifactKeys.AGENTIC_TRAIT_SCORES) == {"code_quality": True}
        assert ctx.get_result_field(ArtifactKeys.AGENTIC_TRAIT_INVESTIGATION_TRACES) is not None

    def test_execute_raises_when_model_lacks_agent_support(self):
        """ValueError raised if resolved model lacks deep_agent tier."""
        from karenina.benchmark.verification.stages.pipeline import (
            agentic_rubric_evaluation as mod,
        )

        no_agent_model = ModelConfig(
            id="test",
            model_name="test-model",
            model_provider="anthropic",
            interface="langchain",
        )

        trait = _make_trait()
        ctx = _make_context(agentic_traits=[trait])
        ctx.parsing_model = no_agent_model

        stage = mod.AgenticRubricEvaluationStage()
        with pytest.raises(ValueError, match="agent_tier='deep_agent'"):
            stage.execute(ctx)

    def test_execute_with_workspace_path(self, monkeypatch):
        from karenina.benchmark.verification.stages.pipeline import (
            agentic_rubric_evaluation as mod,
        )

        fake_evaluator = MagicMock()
        fake_evaluator.evaluate_trait.return_value = (True, "investigation trace")

        monkeypatch.setattr(
            mod,
            "AgenticTraitEvaluator",
            lambda model_config, **kwargs: fake_evaluator,  # noqa: ARG005
        )

        ctx = _make_context(
            agentic_traits=[_make_trait()],
            workspace_path="/tmp/test_workspace",
        )
        stage = mod.AgenticRubricEvaluationStage()
        stage.execute(ctx)

        # Verify workspace_path was passed to evaluate_trait
        call_kwargs = fake_evaluator.evaluate_trait.call_args
        assert call_kwargs[1]["workspace_path"] == Path("/tmp/test_workspace")


# ------------------------------------------------------------------
# Template kind flattening
# ------------------------------------------------------------------


class _StageTestFindings(BaseModel):
    count: int = Field(description="Count")
    items: list[str] = Field(description="Items")


@pytest.mark.unit
class TestTemplateKindFlattening:
    """Tests for dot-notation flattening of template kind scores."""

    def test_template_trait_scores_are_dot_expanded(self, monkeypatch):
        """Template kind scores should be flattened with dot notation."""
        from karenina.benchmark.verification.stages.pipeline import (
            agentic_rubric_evaluation as mod,
        )

        fake_evaluator = MagicMock()
        fake_evaluator.evaluate_trait.return_value = (
            {"count": 3, "items": ["pip", "run"]},
            "investigation trace",
        )

        monkeypatch.setattr(
            mod,
            "AgenticTraitEvaluator",
            lambda model_config, **kwargs: fake_evaluator,  # noqa: ARG005
        )

        trait = AgenticRubricTrait(
            name="tool_usage",
            description="Evaluate tool usage.",
            kind=_StageTestFindings,
            higher_is_better=None,
        )
        ctx = _make_context(agentic_traits=[trait])
        stage = mod.AgenticRubricEvaluationStage()
        stage.execute(ctx)

        scores = ctx.get_artifact(ArtifactKeys.AGENTIC_TRAIT_SCORES)
        # Must be flattened into dot-notation keys
        assert "tool_usage.count" in scores
        assert "tool_usage.items" in scores
        assert scores["tool_usage.count"] == 3
        assert scores["tool_usage.items"] == ["pip", "run"]
        # The base key should NOT be present
        assert "tool_usage" not in scores

    def test_non_template_trait_scores_stored_flat(self, monkeypatch):
        """Non-template kind scores remain stored under the trait name."""
        from karenina.benchmark.verification.stages.pipeline import (
            agentic_rubric_evaluation as mod,
        )

        fake_evaluator = MagicMock()
        fake_evaluator.evaluate_trait.return_value = (True, "trace")

        monkeypatch.setattr(
            mod,
            "AgenticTraitEvaluator",
            lambda model_config, **kwargs: fake_evaluator,  # noqa: ARG005
        )

        ctx = _make_context(agentic_traits=[_make_trait(name="quality", kind="boolean")])
        stage = mod.AgenticRubricEvaluationStage()
        stage.execute(ctx)

        scores = ctx.get_artifact(ArtifactKeys.AGENTIC_TRAIT_SCORES)
        assert scores == {"quality": True}


# ------------------------------------------------------------------
# Trace file writing
# ------------------------------------------------------------------


@pytest.mark.unit
class TestWriteTraceFile:
    """Tests for AgenticRubricEvaluationStage._write_trace_file()."""

    def test_writes_trace_to_workspace(self, tmp_path):
        from karenina.benchmark.verification.stages.pipeline.agentic_rubric_evaluation import (
            AgenticRubricEvaluationStage,
        )

        trace_path = AgenticRubricEvaluationStage._write_trace_file(
            workspace_path=tmp_path,
            trace="Hello trace content",
            question_id="q1",
            scenario_turn=None,
        )
        assert trace_path.exists()
        assert trace_path.read_text(encoding="utf-8") == "Hello trace content"
        assert trace_path.parent.name == "traces"
        assert "q1_trace.txt" in trace_path.name

    def test_includes_turn_in_filename(self, tmp_path):
        from karenina.benchmark.verification.stages.pipeline.agentic_rubric_evaluation import (
            AgenticRubricEvaluationStage,
        )

        trace_path = AgenticRubricEvaluationStage._write_trace_file(
            workspace_path=tmp_path,
            trace="content",
            question_id="q1",
            scenario_turn=2,
        )
        assert "q1_turn2_trace.txt" in trace_path.name

    def test_falls_back_to_tempdir_when_no_workspace(self):
        from karenina.benchmark.verification.stages.pipeline.agentic_rubric_evaluation import (
            AgenticRubricEvaluationStage,
        )

        trace_path = AgenticRubricEvaluationStage._write_trace_file(
            workspace_path=None,
            trace="content",
            question_id="q1",
            scenario_turn=None,
        )
        assert trace_path.exists()
        assert trace_path.read_text(encoding="utf-8") == "content"
        trace_path.unlink()

    def test_sanitizes_question_id(self, tmp_path):
        from karenina.benchmark.verification.stages.pipeline.agentic_rubric_evaluation import (
            AgenticRubricEvaluationStage,
        )

        trace_path = AgenticRubricEvaluationStage._write_trace_file(
            workspace_path=tmp_path,
            trace="content",
            question_id="q/1:bad<chars>",
            scenario_turn=None,
        )
        assert "/" not in trace_path.name
        assert ":" not in trace_path.name


@pytest.mark.unit
class TestValidateAgentSupport:
    """Tests for _validate_agent_support raising on non-deep_agent interfaces."""

    def test_raises_when_parsing_model_lacks_deep_agent(self):
        """ValueError raised when parsing model interface is not deep_agent."""
        from karenina.benchmark.verification.stages.pipeline.agentic_rubric_evaluation import (
            AgenticRubricEvaluationStage,
        )

        trait = _make_trait("my_trait")
        rubric = Rubric(agentic_traits=[trait])
        parsing_model = ModelConfig(
            id="test",
            model_name="gpt-4",
            model_provider="openai",
            interface="langchain",
        )
        ctx = VerificationContext(
            question_id="q1",
            template_id="t1",
            question_text="Question",
            template_code="class Answer: pass",
            answering_model=parsing_model,
            parsing_model=parsing_model,
            rubric=rubric,
        )

        with pytest.raises(ValueError, match="agent_tier='deep_agent'"):
            AgenticRubricEvaluationStage._validate_agent_support([trait], ctx)

    def test_passes_when_parsing_model_is_deep_agent(self):
        """No error when parsing model supports deep_agent."""
        from karenina.benchmark.verification.stages.pipeline.agentic_rubric_evaluation import (
            AgenticRubricEvaluationStage,
        )

        trait = _make_trait("my_trait")
        ctx = _make_context(agentic_traits=[trait])

        # Should not raise
        AgenticRubricEvaluationStage._validate_agent_support([trait], ctx)

    def test_model_override_with_non_deep_agent_rejected_at_construction(self):
        """AgenticRubricTrait rejects non-deep_agent model_override at construction."""
        from pydantic import ValidationError as PydanticValidationError

        override = ModelConfig(
            id="bad",
            model_name="gpt-4",
            model_provider="openai",
            interface="langchain",
        )
        with pytest.raises(PydanticValidationError, match="agent_tier"):
            AgenticRubricTrait(
                name="my_trait",
                description="Test trait",
                kind="boolean",
                higher_is_better=True,
                model_override=override,
            )
