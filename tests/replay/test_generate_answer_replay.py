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
