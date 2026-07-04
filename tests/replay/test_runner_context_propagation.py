"""Tests that run_single_model_verification forwards replay_store onto
VerificationContext."""

from __future__ import annotations

import pytest

from karenina.benchmark.verification.stages.core.base import VerificationContext
from karenina.replay import ReplayStore


@pytest.mark.unit
def test_run_single_model_verification_accepts_replay_kwargs(monkeypatch):
    captured: dict = {}

    class _FakeOrchestrator:
        @classmethod
        def from_config(cls, **kwargs):  # noqa: ARG003
            return cls()

        def execute(self, context: VerificationContext):
            captured["replay_store"] = context.replay_store
            captured["replay_parse_on_hydration_mismatch"] = context.replay_parse_on_hydration_mismatch
            captured["scenario_node_visit_index"] = context.scenario_node_visit_index

            from karenina.schemas.verification.result import VerificationResult

            return VerificationResult.model_construct()

    monkeypatch.setattr(
        "karenina.benchmark.verification.runner.StageOrchestrator",
        _FakeOrchestrator,
    )

    from karenina.benchmark.verification.runner import run_single_model_verification
    from karenina.schemas.config import ModelConfig

    store = ReplayStore()
    ans = ModelConfig(id="a", model_name="m", model_provider="anthropic")
    parse = ModelConfig(id="p", model_name="m", model_provider="anthropic")

    run_single_model_verification(
        question_id="q",
        question_text="hi",
        template_code="",
        answering_model=ans,
        parsing_model=parse,
        replay_store=store,
        replay_parse_on_hydration_mismatch="strict",
    )

    assert captured["replay_store"] is store
    assert captured["replay_parse_on_hydration_mismatch"] == "strict"
    assert captured["scenario_node_visit_index"] is None
