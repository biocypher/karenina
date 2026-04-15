"""Tests that ScenarioManager._run_turn propagates replay fields to the
per-turn VerificationContext."""

from __future__ import annotations

import pytest

from karenina.replay import ReplayStore


@pytest.mark.unit
def test_scenario_manager_copies_replay_fields(monkeypatch):
    captured: dict = {}

    from karenina.benchmark.verification.stages.core.base import VerificationContext

    class _FakeOrchestrator:
        @classmethod
        def from_config(cls, **kwargs):  # noqa: ARG003
            return cls()

        def execute(self, context: VerificationContext):
            captured.setdefault("contexts", []).append(
                {
                    "replay_store": context.replay_store,
                    "replay_parse_on_hydration_mismatch": context.replay_parse_on_hydration_mismatch,
                    "scenario_node_visit_index": context.scenario_node_visit_index,
                    "scenario_node": context.scenario_node,
                }
            )

            from karenina.schemas.verification.result import VerificationResult

            metadata_stub = type(
                "Md",
                (),
                {
                    "failure": None,
                    "result_id": "rid",
                },
            )()
            template_stub = type("Tpl", (), {"verify_result": True})()

            return VerificationResult.model_construct(
                metadata=metadata_stub,
                template=template_stub,
                rubric=None,
            )

    monkeypatch.setattr(
        "karenina.scenario.manager.StageOrchestrator",
        _FakeOrchestrator,
    )

    from karenina.scenario.manager import ScenarioManager
    from karenina.schemas.config import ModelConfig
    from karenina.schemas.entities.question import Question
    from karenina.schemas.scenario.definition import ScenarioDefinition
    from karenina.schemas.scenario.types import END, ScenarioEdge, ScenarioNode
    from karenina.schemas.verification.config import VerificationConfig

    question = Question(
        question="hi",
        raw_answer="hi",
        keywords=[],
        answer_template="class Answer(BaseAnswer):\n    pass\n",
    )
    node = ScenarioNode(node_id="n1", question=question)
    defn = ScenarioDefinition(
        name="test-scenario",
        entry_node="n1",
        nodes={"n1": node},
        edges=[ScenarioEdge(source="n1", target=END)],
        outcome_criteria=[],
    )

    store = ReplayStore()
    config = VerificationConfig(
        answering_models=[ModelConfig(id="a", model_name="m", model_provider="anthropic")],
        parsing_models=[ModelConfig(id="p", model_name="m", model_provider="anthropic")],
        replay_store=store,
        replay_parse_on_hydration_mismatch="strict",
    )

    ans = config.answering_models[0]
    parse = config.parsing_models[0]

    manager = ScenarioManager()
    manager.run(defn, config, ans, parse)

    assert captured["contexts"], "orchestrator was never called"
    ctx = captured["contexts"][0]
    assert ctx["replay_store"] is store
    assert ctx["replay_parse_on_hydration_mismatch"] == "strict"
    assert ctx["scenario_node_visit_index"] == 0
