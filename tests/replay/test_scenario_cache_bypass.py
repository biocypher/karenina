"""Test that ScenarioManager skips AnswerTraceCache reservation when a
replay hit is known for the upcoming node."""

from __future__ import annotations

import pytest

from karenina.replay import ReplayEntry, ReplayKey, ReplayStore


@pytest.mark.unit
def test_scenario_cache_reservation_skipped_on_replay_hit(monkeypatch):
    from karenina.scenario.manager import ScenarioManager
    from karenina.schemas.config import ModelConfig
    from karenina.schemas.entities.question import Question
    from karenina.schemas.scenario.definition import ScenarioDefinition
    from karenina.schemas.scenario.types import END, ScenarioEdge, ScenarioNode
    from karenina.schemas.verification.config import VerificationConfig
    from karenina.utils.answer_cache import AnswerTraceCache
    from karenina.utils.checkpoint import generate_question_id

    # Build a scenario with one node.
    question = Question(
        question="What is X?",
        raw_answer="X",
        keywords=[],
        answer_template="class Answer(BaseAnswer):\n    pass\n",
    )
    node = ScenarioNode(node_id="n1", question=question)
    defn = ScenarioDefinition(
        name="s",
        entry_node="n1",
        nodes={"n1": node},
        edges=[ScenarioEdge(source="n1", target=END)],
        outcome_criteria=[],
    )

    store = ReplayStore()
    store.register(
        ReplayKey(
            question_id=generate_question_id("What is X?"),
            scenario_id="s",
            scenario_node="n1",
        ),
        ReplayEntry(raw_trace="canned"),
    )

    ans = ModelConfig(id="gpt-5", model_name="m", model_provider="openai")
    parse = ModelConfig(id="p", model_name="m", model_provider="anthropic")
    config = VerificationConfig(
        answering_models=[ans],
        parsing_models=[parse],
        replay_store=store,
    )

    cache = AnswerTraceCache()
    reserved: list[str] = []
    original_get_or_reserve = cache.get_or_reserve

    def _tracked(key: str):
        reserved.append(key)
        return original_get_or_reserve(key)

    monkeypatch.setattr(cache, "get_or_reserve", _tracked)

    # Orchestrator stub: pretend the pipeline succeeded
    class _FakeOrchestrator:
        @classmethod
        def from_config(cls, **_kwargs):
            return cls()

        def execute(self, _context):
            from karenina.schemas.verification.result import VerificationResult

            return VerificationResult.model_construct(
                metadata=type(
                    "Md",
                    (),
                    {
                        "completed_without_errors": True,
                        "result_id": "rid",
                        "error": None,
                        "error_category": None,
                    },
                )(),
                template=type("Tpl", (), {"verify_result": True})(),
                rubric=None,
            )

    monkeypatch.setattr(
        "karenina.scenario.manager.StageOrchestrator",
        _FakeOrchestrator,
    )

    manager = ScenarioManager()
    manager.run(defn, config, ans, parse, answer_cache=cache)

    assert reserved == [], "get_or_reserve must not be called on a replay-hit turn"
