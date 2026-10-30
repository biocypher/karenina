"""Tests for AgenticRubricEvaluationStage."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from karenina.benchmark.verification.evaluators.rubric.agentic_trait import AgenticTraitEvaluation
from karenina.benchmark.verification.stages.core.base import (
    ArtifactKeys,
    VerificationContext,
)
from karenina.schemas.config.models import ModelConfig
from karenina.schemas.entities.rubric import AgenticRubricTrait, Rubric
from karenina.schemas.results.caveat import Caveat


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
        assert ArtifactKeys.AGENTIC_TRAIT_EXTRACTION_METADATA in stage.produces


@pytest.mark.unit
class TestAgenticRubricEvaluationExecute:
    """Tests for AgenticRubricEvaluationStage.execute()."""

    def test_execute_individual_strategy(self, monkeypatch):
        from karenina.benchmark.verification.stages.pipeline import (
            agentic_rubric_evaluation as mod,
        )

        fake_evaluator = MagicMock()
        fake_evaluator.evaluate_trait.return_value = AgenticTraitEvaluation(score=True, trace="investigation trace")

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

        metadata = ctx.get_artifact(ArtifactKeys.AGENTIC_TRAIT_EXTRACTION_METADATA)
        assert metadata == {}

        performed = ctx.get_artifact(ArtifactKeys.AGENTIC_RUBRIC_EVALUATION_PERFORMED)
        assert performed is True

    def test_execute_individual_strategy_stores_extraction_metadata(self, monkeypatch):
        from karenina.benchmark.verification.stages.pipeline import (
            agentic_rubric_evaluation as mod,
        )

        fake_evaluator = MagicMock()
        fake_evaluator.evaluate_trait.return_value = AgenticTraitEvaluation(score=True, trace="investigation trace")
        fake_evaluator.last_extraction_metadata = {
            "method": "local_json",
            "local_json_error": None,
            "parser_error": None,
        }

        monkeypatch.setattr(
            mod,
            "AgenticTraitEvaluator",
            lambda model_config, **kwargs: fake_evaluator,  # noqa: ARG005
        )

        ctx = _make_context(agentic_traits=[_make_trait()])
        stage = mod.AgenticRubricEvaluationStage()
        stage.execute(ctx)

        metadata = ctx.get_artifact(ArtifactKeys.AGENTIC_TRAIT_EXTRACTION_METADATA)
        assert metadata == {
            "code_quality": {
                "method": "local_json",
                "local_json_error": None,
                "parser_error": None,
            }
        }

    def test_execute_failed_trait_stores_none(self, monkeypatch):
        from karenina.benchmark.verification.stages.pipeline import (
            agentic_rubric_evaluation as mod,
        )

        fake_evaluator = MagicMock()
        fake_evaluator.evaluate_trait.return_value = AgenticTraitEvaluation(score=None, trace=None)

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
                return AgenticTraitEvaluation(score=True, trace="trace_cq")
            return AgenticTraitEvaluation(score=False, trace="trace_safety")

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
        fake_evaluator.evaluate_trait.return_value = AgenticTraitEvaluation(score=True, trace="investigation trace")

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
        fake_evaluator.evaluate_trait.return_value = AgenticTraitEvaluation(score=True, trace="investigation trace")

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
        fake_evaluator.evaluate_trait.return_value = AgenticTraitEvaluation(
            score={"count": 3, "items": ["pip", "run"]},
            trace="investigation trace",
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
        fake_evaluator.evaluate_trait.return_value = AgenticTraitEvaluation(score=True, trace="trace")

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


# ------------------------------------------------------------------
# Shared strategy
# ------------------------------------------------------------------


def _make_shared_context(agentic_traits, mcp_urls_dict=None):
    """Build a context that runs the shared strategy with an optional MCP judge."""
    ctx = _make_context(agentic_traits=agentic_traits)
    ctx.parsing_model = ModelConfig(
        id="judge",
        model_name="judge-model",
        model_provider="anthropic",
        interface="claude_agent_sdk",
        mcp_urls_dict=mcp_urls_dict,
    )
    ctx.agentic_rubric_strategy = "shared"
    return ctx


def _make_shared_agent_result(raw_trace="Shared investigation."):
    from karenina.ports import AgentResult, Message
    from karenina.ports.usage import UsageMetadata

    return AgentResult(
        final_response="Done.",
        raw_trace=raw_trace,
        trace_messages=[Message.assistant("Done.")],
        usage=UsageMetadata(input_tokens=40, output_tokens=10, total_tokens=50),
        turns=2,
        limit_reached=False,
    )


@pytest.mark.unit
class TestSharedStrategy:
    """Tests for the shared investigation strategy."""

    def test_shared_agent_receives_judge_mcp_servers(self, monkeypatch):
        import karenina.adapters as adapters_pkg
        from karenina.benchmark.verification.stages.pipeline import (
            agentic_rubric_evaluation as mod,
        )

        fake_agent = MagicMock()
        fake_agent.run.return_value = _make_shared_agent_result()
        monkeypatch.setattr(adapters_pkg, "get_agent", lambda _model: fake_agent)

        fake_evaluator = MagicMock()
        fake_evaluator.run_extraction.return_value = True
        fake_evaluator.last_extraction_metadata = None
        monkeypatch.setattr(
            mod,
            "AgenticTraitEvaluator",
            lambda model_config, **kwargs: fake_evaluator,  # noqa: ARG005
        )

        ctx = _make_shared_context(
            [_make_trait(name="a"), _make_trait(name="b")],
            mcp_urls_dict={"citations": "http://localhost:8765/mcp"},
        )
        mod.AgenticRubricEvaluationStage().execute(ctx)

        fake_agent.run.assert_called_once()
        assert fake_agent.run.call_args.kwargs["mcp_servers"] == {
            "citations": {"type": "http", "url": "http://localhost:8765/mcp"},
        }
        assert ctx.get_artifact(ArtifactKeys.AGENTIC_TRAIT_SCORES) == {"a": True, "b": True}
        traces = ctx.get_artifact(ArtifactKeys.AGENTIC_TRAIT_INVESTIGATION_TRACES)
        assert traces == {"a": "Shared investigation.", "b": "Shared investigation."}


# ------------------------------------------------------------------
# Usage tracking
# ------------------------------------------------------------------


@pytest.mark.unit
class TestJudgeUsageTracking:
    """The judge's usage lands in the usage tracker under the agentic rubric stage."""

    def test_individual_strategy_tracks_investigation_and_extraction(self, monkeypatch):
        from karenina.benchmark.verification.stages.pipeline import (
            agentic_rubric_evaluation as mod,
        )
        from karenina.benchmark.verification.utils.trace_usage_tracker import UsageTracker
        from karenina.ports.usage import UsageMetadata

        fake_evaluator = MagicMock()
        fake_evaluator.evaluate_trait.return_value = AgenticTraitEvaluation(
            score=True,
            trace="trace",
            investigation_usage=UsageMetadata(
                input_tokens=100,
                output_tokens=40,
                total_tokens=140,
                cache_read_tokens=30,
                reasoning_tokens=12,
                web_search_queries=3,
            ),
            extraction_usage=UsageMetadata(input_tokens=20, output_tokens=5, total_tokens=25),
        )
        monkeypatch.setattr(
            mod,
            "AgenticTraitEvaluator",
            lambda model_config, **kwargs: fake_evaluator,  # noqa: ARG005
        )

        ctx = _make_context(agentic_traits=[_make_trait()])
        prior = UsageTracker()
        prior.track_call("answer_generation", "answerer", {"answerer": {"input_tokens": 7, "total_tokens": 7}})
        ctx.set_artifact(ArtifactKeys.USAGE_TRACKER, prior)

        mod.AgenticRubricEvaluationStage().execute(ctx)

        tracker = ctx.get_artifact(ArtifactKeys.USAGE_TRACKER)
        assert tracker is prior
        summary = tracker.get_total_summary()
        stage = summary[mod.AGENTIC_RUBRIC_USAGE_STAGE]
        assert stage["input_tokens"] == 120
        assert stage["output_tokens"] == 45
        assert stage["total_tokens"] == 165
        assert stage["web_search_queries"] == 3
        assert stage["reasoning_tokens"] == 12
        assert summary["answer_generation"]["input_tokens"] == 7
        assert summary["total"]["input_tokens"] == 127
        assert summary["total"]["web_search_queries"] == 3

    def test_missing_usage_is_not_tracked(self, monkeypatch):
        from karenina.benchmark.verification.stages.pipeline import (
            agentic_rubric_evaluation as mod,
        )

        fake_evaluator = MagicMock()
        fake_evaluator.evaluate_trait.return_value = AgenticTraitEvaluation(score=True, trace="trace")
        monkeypatch.setattr(
            mod,
            "AgenticTraitEvaluator",
            lambda model_config, **kwargs: fake_evaluator,  # noqa: ARG005
        )

        ctx = _make_context(agentic_traits=[_make_trait()])
        mod.AgenticRubricEvaluationStage().execute(ctx)

        tracker = ctx.get_artifact(ArtifactKeys.USAGE_TRACKER)
        assert tracker.get_stage_summary(mod.AGENTIC_RUBRIC_USAGE_STAGE) is None

    def test_shared_strategy_tracks_shared_run_and_extractions(self, monkeypatch):
        import karenina.adapters as adapters_pkg
        from karenina.benchmark.verification.stages.pipeline import (
            agentic_rubric_evaluation as mod,
        )
        from karenina.ports.usage import UsageMetadata

        fake_agent = MagicMock()
        fake_agent.run.return_value = _make_shared_agent_result()
        monkeypatch.setattr(adapters_pkg, "get_agent", lambda _model: fake_agent)

        fake_evaluator = MagicMock()
        fake_evaluator.run_extraction.return_value = True
        fake_evaluator.last_extraction_metadata = None
        fake_evaluator.last_extraction_usage = UsageMetadata(input_tokens=5, output_tokens=1, total_tokens=6)
        monkeypatch.setattr(
            mod,
            "AgenticTraitEvaluator",
            lambda model_config, **kwargs: fake_evaluator,  # noqa: ARG005
        )

        ctx = _make_shared_context([_make_trait(name="a"), _make_trait(name="b")])
        mod.AgenticRubricEvaluationStage().execute(ctx)

        stage = ctx.get_artifact(ArtifactKeys.USAGE_TRACKER).get_stage_summary(mod.AGENTIC_RUBRIC_USAGE_STAGE)
        # One shared run (40 in) plus one extraction per trait (5 in each).
        assert stage["input_tokens"] == 50
        assert stage["total_tokens"] == 62


# ------------------------------------------------------------------
# Investigation messages
# ------------------------------------------------------------------


@pytest.mark.unit
class TestInvestigationMessages:
    """The stage stores each trait's serialised investigation messages."""

    def test_individual_strategy_serialises_messages_like_template_trace(self, monkeypatch):
        from karenina.benchmark.verification.stages.pipeline import (
            agentic_rubric_evaluation as mod,
        )
        from karenina.ports import Message, Role, ToolUseContent

        judge_messages = [
            Message(role=Role.ASSISTANT, content=[ToolUseContent(id="c1", name="europepmc_get", input={"id": "1"})]),
            Message.tool_result("c1", "Title: Example"),
            Message.assistant("Supported."),
        ]
        fake_evaluator = MagicMock()
        fake_evaluator.evaluate_trait.return_value = AgenticTraitEvaluation(
            score=True,
            trace="trace",
            messages=judge_messages,
        )
        monkeypatch.setattr(
            mod,
            "AgenticTraitEvaluator",
            lambda model_config, **kwargs: fake_evaluator,  # noqa: ARG005
        )

        ctx = _make_context(agentic_traits=[_make_trait()])
        mod.AgenticRubricEvaluationStage().execute(ctx)

        messages = ctx.get_artifact(ArtifactKeys.AGENTIC_TRAIT_INVESTIGATION_MESSAGES)
        stored = messages["code_quality"]
        assert [m["block_index"] for m in stored] == [0, 1, 2]
        assert stored[0]["tool_calls"] == [{"id": "c1", "name": "europepmc_get", "input": {"id": "1"}}]
        assert stored[1]["role"] == "tool"
        assert stored[1]["tool_result"] == {"tool_use_id": "c1", "is_error": False}
        assert stored[2] == {"role": "assistant", "content": "Supported.", "block_index": 2}
        assert ctx.get_result_field(ArtifactKeys.AGENTIC_TRAIT_INVESTIGATION_MESSAGES) == messages
        assert ArtifactKeys.AGENTIC_TRAIT_INVESTIGATION_MESSAGES in mod.AgenticRubricEvaluationStage().produces

    def test_missing_messages_stored_as_none(self, monkeypatch):
        from karenina.benchmark.verification.stages.pipeline import (
            agentic_rubric_evaluation as mod,
        )

        fake_evaluator = MagicMock()
        fake_evaluator.evaluate_trait.return_value = AgenticTraitEvaluation(score=None, trace=None, messages=None)
        monkeypatch.setattr(
            mod,
            "AgenticTraitEvaluator",
            lambda model_config, **kwargs: fake_evaluator,  # noqa: ARG005
        )

        ctx = _make_context(agentic_traits=[_make_trait()])
        mod.AgenticRubricEvaluationStage().execute(ctx)

        assert ctx.get_artifact(ArtifactKeys.AGENTIC_TRAIT_INVESTIGATION_MESSAGES) == {"code_quality": None}

    def test_shared_strategy_stores_shared_messages_per_trait(self, monkeypatch):
        import karenina.adapters as adapters_pkg
        from karenina.benchmark.verification.stages.pipeline import (
            agentic_rubric_evaluation as mod,
        )

        fake_agent = MagicMock()
        fake_agent.run.return_value = _make_shared_agent_result()
        monkeypatch.setattr(adapters_pkg, "get_agent", lambda _model: fake_agent)

        fake_evaluator = MagicMock()
        fake_evaluator.run_extraction.return_value = True
        fake_evaluator.last_extraction_metadata = None
        monkeypatch.setattr(
            mod,
            "AgenticTraitEvaluator",
            lambda model_config, **kwargs: fake_evaluator,  # noqa: ARG005
        )

        ctx = _make_shared_context([_make_trait(name="a"), _make_trait(name="b")])
        mod.AgenticRubricEvaluationStage().execute(ctx)

        messages = ctx.get_artifact(ArtifactKeys.AGENTIC_TRAIT_INVESTIGATION_MESSAGES)
        expected = [{"role": "assistant", "content": "Done.", "block_index": 0}]
        assert messages == {"a": expected, "b": expected}


# ------------------------------------------------------------------
# Failed investigations
# ------------------------------------------------------------------


def _failing_evaluator(failing_trait: str, error: Exception) -> MagicMock:
    """Fake evaluator whose investigation raises for one trait and scores the rest."""

    def evaluate_trait(trait, question_text, raw_llm_response, workspace_path):  # noqa: ARG001
        if trait.name == failing_trait:
            return AgenticTraitEvaluation(
                score=None,
                trace=f"[investigation failed: {type(error).__name__}: {error}]",
                investigation_error=error,
            )
        return AgenticTraitEvaluation(score=True, trace="fine", messages=[])

    fake = MagicMock()
    fake.evaluate_trait.side_effect = evaluate_trait
    return fake


@pytest.mark.unit
class TestFailedInvestigation:
    """A failed investigation leaves a record and marks an infrastructure failure."""

    def test_failure_record_and_stage_error(self, monkeypatch):
        from karenina.benchmark.verification.stages.pipeline import (
            agentic_rubric_evaluation as mod,
        )

        fake_evaluator = _failing_evaluator("citations", RuntimeError("judge crashed"))
        monkeypatch.setattr(
            mod,
            "AgenticTraitEvaluator",
            lambda model_config, **kwargs: fake_evaluator,  # noqa: ARG005
        )

        ctx = _make_context(agentic_traits=[_make_trait(name="citations"), _make_trait(name="clarity")])
        mod.AgenticRubricEvaluationStage().execute(ctx)

        traces = ctx.get_artifact(ArtifactKeys.AGENTIC_TRAIT_INVESTIGATION_TRACES)
        assert traces["citations"] == "[investigation failed: RuntimeError: judge crashed]"
        assert ctx.get_artifact(ArtifactKeys.AGENTIC_TRAIT_INVESTIGATION_MESSAGES)["citations"] is None
        # The other trait is still evaluated and stored.
        assert ctx.get_artifact(ArtifactKeys.AGENTIC_TRAIT_SCORES) == {"citations": None, "clarity": True}

        assert ctx.error is not None
        assert "'citations' (RuntimeError: judge crashed)" in ctx.error
        assert ctx.error_stage == "AgenticRubricEvaluation"
        assert ctx.get_result_field(ArtifactKeys.AGENTIC_INVESTIGATION_FAILED) is True

    def test_error_category_comes_from_error_registry(self, monkeypatch):
        from karenina.benchmark.verification.stages.pipeline import (
            agentic_rubric_evaluation as mod,
        )
        from karenina.utils.errors import ErrorCategory

        fake_evaluator = _failing_evaluator("citations", ConnectionError("MCP server unreachable"))
        monkeypatch.setattr(
            mod,
            "AgenticTraitEvaluator",
            lambda model_config, **kwargs: fake_evaluator,  # noqa: ARG005
        )

        ctx = _make_context(agentic_traits=[_make_trait(name="citations")])
        mod.AgenticRubricEvaluationStage().execute(ctx)

        assert ctx.error_category == ErrorCategory.CONNECTION

    def test_extraction_failure_is_not_an_infrastructure_failure(self, monkeypatch):
        from karenina.benchmark.verification.stages.pipeline import (
            agentic_rubric_evaluation as mod,
        )

        fake_evaluator = MagicMock()
        fake_evaluator.evaluate_trait.return_value = AgenticTraitEvaluation(score=None, trace="kept trace", messages=[])
        monkeypatch.setattr(
            mod,
            "AgenticTraitEvaluator",
            lambda model_config, **kwargs: fake_evaluator,  # noqa: ARG005
        )

        ctx = _make_context(agentic_traits=[_make_trait()])
        mod.AgenticRubricEvaluationStage().execute(ctx)

        assert ctx.get_artifact(ArtifactKeys.AGENTIC_TRAIT_INVESTIGATION_TRACES) == {"code_quality": "kept trace"}
        assert ctx.error is None
        assert ctx.get_result_field(ArtifactKeys.AGENTIC_INVESTIGATION_FAILED) is None

    def test_finalized_result_is_a_system_failure_at_the_stage(self, monkeypatch):
        """End to end through finalize: the row carries a Failure, not a judgment."""
        from karenina.benchmark.verification.stages.pipeline import (
            agentic_rubric_evaluation as mod,
        )
        from karenina.benchmark.verification.stages.pipeline.finalize_result import FinalizeResultStage
        from karenina.schemas.results.failure import FailureCategory, FailureGroup

        fake_evaluator = _failing_evaluator("citations", RuntimeError("judge crashed"))
        monkeypatch.setattr(
            mod,
            "AgenticTraitEvaluator",
            lambda model_config, **kwargs: fake_evaluator,  # noqa: ARG005
        )

        ctx = _make_context(agentic_traits=[_make_trait(name="citations")])
        ctx.set_result_field("timestamp", "2026-10-30 12:00:00")
        ctx.set_result_field("execution_time", 1.0)
        mod.AgenticRubricEvaluationStage().execute(ctx)
        FinalizeResultStage().execute(ctx)

        result = ctx.get_artifact("final_result")
        failure = result.metadata.failure
        assert failure is not None
        assert failure.category == FailureCategory.UNEXPECTED_ERROR
        assert failure.group == FailureGroup.SYSTEM
        assert failure.stage == "AgenticRubricEvaluation"
        assert Caveat.AGENTIC_INVESTIGATION_FAILED in result.metadata.caveats
        assert result.rubric.agentic_trait_investigation_traces == {
            "citations": "[investigation failed: RuntimeError: judge crashed]",
        }
        assert result.rubric.agentic_trait_investigation_messages == {}

    @pytest.mark.parametrize(
        ("prior_fields", "expected_category"),
        [
            (
                {ArtifactKeys.TEMPLATE_VERIFICATION_PERFORMED: True, ArtifactKeys.VERIFY_RESULT: False},
                "content",
            ),
            ({ArtifactKeys.ABSTENTION_DETECTED: True}, "abstention"),
            ({ArtifactKeys.SUFFICIENCY_DETECTED: True}, "sufficiency"),
        ],
    )
    def test_caveat_marks_failure_when_an_earlier_rule_sets_the_category(
        self, monkeypatch, prior_fields, expected_category
    ):
        """Content, abstention and sufficiency outrank the stage error, the caveat does not depend on them."""
        from karenina.benchmark.verification.failure_classifier import classify_failure, collect_caveats
        from karenina.benchmark.verification.stages.pipeline import (
            agentic_rubric_evaluation as mod,
        )

        fake_evaluator = _failing_evaluator("citations", RuntimeError("judge crashed"))
        monkeypatch.setattr(
            mod,
            "AgenticTraitEvaluator",
            lambda model_config, **kwargs: fake_evaluator,  # noqa: ARG005
        )

        ctx = _make_context(agentic_traits=[_make_trait(name="citations")])
        for key, value in prior_fields.items():
            ctx.set_result_field(key, value)
        mod.AgenticRubricEvaluationStage().execute(ctx)

        failure = classify_failure(ctx)
        assert failure is not None
        assert failure.category.value == expected_category
        assert Caveat.AGENTIC_INVESTIGATION_FAILED in collect_caveats(ctx)
