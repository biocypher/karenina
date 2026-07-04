"""Integration tests for the replay short-circuit in GenerateAnswerStage."""

from __future__ import annotations

import pytest

from karenina.benchmark.verification.stages.core.base import (
    ArtifactKeys,
    VerificationContext,
)
from karenina.benchmark.verification.stages.pipeline.generate_answer import (
    GenerateAnswerStage,
)
from karenina.replay import ReplayEntry, ReplayKey, ReplayStore
from karenina.schemas.config import ModelConfig
from karenina.schemas.verification.model_identity import ModelIdentity


def _answering_display(model: ModelConfig) -> str:
    """Canonical answering-model display string for use as a ReplayKey field."""
    return ModelIdentity.from_model_config(model, role="answering").display_string


def _make_context(*, replay_store=None, scenario=False, question_id="q"):
    ans = ModelConfig(id="gpt-5", model_name="gpt-5", model_provider="openai")
    parse = ModelConfig(id="p", model_name="p", model_provider="anthropic")

    context = VerificationContext(
        question_id=question_id,
        template_id="t",
        question_text="hi",
        template_code="class Answer(BaseAnswer):\n    pass\n",
        answering_model=ans,
        parsing_model=parse,
    )
    context.replay_store = replay_store
    if scenario:
        context.scenario_id = "s"
        context.scenario_node = "n"
        context.scenario_node_visit_index = 0
    return context


@pytest.fixture
def stage():
    return GenerateAnswerStage()


@pytest.mark.integration
class TestGenerateAnswerReplayHit:
    def test_qa_hit_short_circuits_before_adapter(self, stage, monkeypatch):
        def _boom(*_args, **_kwargs):
            raise AssertionError("get_agent / get_llm must not be called on a replay hit")

        monkeypatch.setattr(
            "karenina.benchmark.verification.stages.pipeline.generate_answer.get_agent",
            _boom,
        )
        monkeypatch.setattr(
            "karenina.benchmark.verification.stages.pipeline.generate_answer.get_llm",
            _boom,
        )

        ans = ModelConfig(id="gpt-5", model_name="gpt-5", model_provider="openai")
        store = ReplayStore()
        store.register(
            ReplayKey(question_id="q", answering_model_id=_answering_display(ans)),
            ReplayEntry(raw_trace="canned answer"),
        )
        context = _make_context(replay_store=store)
        stage.execute(context)

        assert context.get_artifact(ArtifactKeys.RAW_LLM_RESPONSE) == "canned answer"
        # Issue 198: replay must also write raw_llm_response and
        # recursion_limit_reached to result fields, because finalize_result
        # reads them from result fields rather than artifacts.
        assert context.get_result_field(ArtifactKeys.RAW_LLM_RESPONSE) == "canned answer"
        assert context.get_result_field(ArtifactKeys.RECURSION_LIMIT_REACHED) is False
        assert context.get_artifact(ArtifactKeys.REPLAY_ENTRY) is not None
        assert context.completed_without_errors is True

    def test_hit_populates_model_str_suffix(self, stage, monkeypatch):
        def _boom(*_args, **_kwargs):
            raise AssertionError("get_agent / get_llm must not be called on a replay hit")

        monkeypatch.setattr(
            "karenina.benchmark.verification.stages.pipeline.generate_answer.get_agent",
            _boom,
        )
        monkeypatch.setattr(
            "karenina.benchmark.verification.stages.pipeline.generate_answer.get_llm",
            _boom,
        )

        ans = ModelConfig(id="gpt-5", model_name="gpt-5", model_provider="openai")
        store = ReplayStore()
        store.register(
            ReplayKey(question_id="q", answering_model_id=_answering_display(ans)),
            ReplayEntry(raw_trace="trace"),
        )
        context = _make_context(replay_store=store)
        stage.execute(context)
        model_str = context.get_artifact(ArtifactKeys.ANSWERING_MODEL_STR)
        assert model_str is not None
        assert "(replay)" in model_str

    def test_hit_rehydrates_usage_metadata_and_agent_metrics(self, stage, monkeypatch):
        def _boom(*_args, **_kwargs):
            raise AssertionError("get_agent / get_llm must not be called on a replay hit")

        monkeypatch.setattr(
            "karenina.benchmark.verification.stages.pipeline.generate_answer.get_agent",
            _boom,
        )
        monkeypatch.setattr(
            "karenina.benchmark.verification.stages.pipeline.generate_answer.get_llm",
            _boom,
        )

        ans = ModelConfig(id="gpt-5", model_name="gpt-5", model_provider="openai")
        usage = {
            "answer_generation": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "model": "gpt-5",
            },
            "parsing": {
                "input_tokens": 7,
                "output_tokens": 3,
                "total_tokens": 10,
                "model": "parser",
            },
            "total": {"input_tokens": 17, "output_tokens": 8, "total_tokens": 25},
        }
        metrics = {"iterations": 2, "tool_calls": 1, "limit_reached": False}
        store = ReplayStore()
        store.register(
            ReplayKey(question_id="q", answering_model_id=_answering_display(ans)),
            ReplayEntry(raw_trace="trace", usage_metadata=usage, agent_metrics=metrics),
        )
        context = _make_context(replay_store=store)

        stage.execute(context)

        tracker = context.get_artifact(ArtifactKeys.USAGE_TRACKER)
        assert tracker is not None
        assert tracker.get_total_summary()["total"] == usage["total"]
        assert tracker.get_total_summary()["answer_generation"]["model"] == "gpt-5"
        assert tracker.get_total_summary()["parsing"]["model"] == "parser"
        assert tracker.get_agent_metrics() == metrics

    def test_strict_miss_marks_error(self, stage, monkeypatch):
        def _boom(*_args, **_kwargs):
            raise AssertionError("get_agent / get_llm must not be called on a replay hit")

        monkeypatch.setattr(
            "karenina.benchmark.verification.stages.pipeline.generate_answer.get_agent",
            _boom,
        )
        monkeypatch.setattr(
            "karenina.benchmark.verification.stages.pipeline.generate_answer.get_llm",
            _boom,
        )

        store = ReplayStore(miss_policy="strict")
        context = _make_context(replay_store=store)
        stage.execute(context)

        assert context.completed_without_errors is False
        assert "replay" in (context.error or "").lower()


@pytest.mark.unit
class TestTryReplayHitReplicateThreading:
    def test_context_replicate_reaches_lookup(self):
        """A context with replicate=2 resolves to the replicate=2 entry
        when the store contains separate entries for replicates 1, 2,
        and 3 under the same (question, model)."""
        from types import SimpleNamespace

        from karenina.benchmark.verification.stages.pipeline.generate_answer import (
            _try_replay_hit,
        )
        from karenina.replay import ReplayEntry, ReplayKey, ReplayStore
        from karenina.schemas.verification.model_identity import ModelIdentity

        model_config = ModelConfig(id="gpt-5", model_name="gpt-5", model_provider="openai")

        display = ModelIdentity.from_model_config(model_config, role="answering").display_string

        store = ReplayStore()
        for rep in (1, 2, 3):
            store.register(
                ReplayKey(
                    question_id="q1",
                    answering_model_id=display,
                    replicate=rep,
                ),
                ReplayEntry(raw_trace=f"raw-rep-{rep}"),
            )

        context = SimpleNamespace(
            question_id="q1",
            answering_model=model_config,
            scenario_id=None,
            scenario_node=None,
            scenario_node_visit_index=None,
            replicate=2,
            replay_store=store,
        )

        hit = _try_replay_hit(context)
        assert hit is not None
        assert hit.raw_trace == "raw-rep-2"

    def test_context_without_replicate_attr_still_works(self):
        """Backwards-compat with test fakes that predate the replicate
        field: a SimpleNamespace context without a .replicate attribute
        should lookup with replicate=None and resolve via the wildcard
        rung.
        """
        from types import SimpleNamespace

        from karenina.benchmark.verification.stages.pipeline.generate_answer import (
            _try_replay_hit,
        )
        from karenina.replay import ReplayEntry, ReplayKey, ReplayStore

        model_config = ModelConfig(id="gpt-5", model_name="gpt-5", model_provider="openai")

        store = ReplayStore()
        store.register(
            ReplayKey(question_id="q1", answering_model_id=None),
            ReplayEntry(raw_trace="legacy"),
        )

        context = SimpleNamespace(
            question_id="q1",
            answering_model=model_config,
            scenario_id=None,
            scenario_node=None,
            scenario_node_visit_index=None,
            replay_store=store,
            # deliberately no `replicate` attribute
        )

        hit = _try_replay_hit(context)
        assert hit is not None
        assert hit.raw_trace == "legacy"


@pytest.mark.unit
class TestBenchmarkSingleReplicateWarningRemoved:
    def test_warning_string_removed_from_source(self):
        """Regression: after R1, the 'Replay is single-replicate'
        warning must no longer exist in benchmark.py. The store now
        supports per-replicate entries natively, so the warning is
        obsolete.
        """
        import inspect

        from karenina.benchmark import benchmark as benchmark_mod

        src = inspect.getsource(benchmark_mod)
        assert "Replay is single-replicate" not in src
