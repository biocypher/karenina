"""Tests for scenario checkpoint serialization."""

import json

import pytest

from karenina.scenario.builder import Scenario
from karenina.schemas.scenario.definition import ScenarioDefinition
from karenina.schemas.scenario.types import END, ScenarioOutcomeCriterion


def _make_question(text="What?"):
    from karenina.schemas.entities import Question

    return Question(question=text, raw_answer="Y", answer_template="class Answer: pass")


def _build_scenario():
    s = Scenario("pipeline")
    s.add_node(
        "ask",
        question=_make_question("Q1"),
        state_update="lambda acc, p: {**acc, 'x': p.get('x')}",
    )
    s.add_node("check", question=_make_question("Q2"))
    s.add_edge("ask", "check", when={"verify_result": True})
    s.add_edge("ask", END)
    s.add_edge("check", END)
    s.set_entry("ask")
    s.add_outcome_criterion(
        ScenarioOutcomeCriterion(
            name="fast",
            description="Done in 2 turns?",
            evaluate=lambda r: r.turn_count <= 2,
        )
    )
    return s.validate()


@pytest.mark.unit
class TestScenarioCheckpointSerialization:
    def test_definition_to_dict(self):
        defn = _build_scenario()
        data = defn.model_dump()
        assert data["name"] == "pipeline"
        assert "ask" in data["nodes"]
        assert len(data["edges"]) == 3
        assert data["entry_node"] == "ask"

    def test_definition_roundtrip(self):
        defn = _build_scenario()
        data = defn.model_dump()
        json_str = json.dumps(data, default=str)
        restored_data = json.loads(json_str)
        restored = ScenarioDefinition.model_validate(restored_data)
        assert restored.name == defn.name
        assert restored.entry_node == defn.entry_node
        assert len(restored.nodes) == len(defn.nodes)
        assert len(restored.edges) == len(defn.edges)

    def test_callable_sources_preserved(self):
        defn = _build_scenario()
        data = defn.model_dump()
        ask_node = data["nodes"]["ask"]
        assert ask_node["state_update_source"] is not None
        assert "lambda" in ask_node["state_update_source"]
        assert data["outcome_criteria"][0]["evaluate_source"] is not None

    def test_callables_not_in_serialized_form(self):
        defn = _build_scenario()
        data = defn.model_dump()
        ask_node = data["nodes"]["ask"]
        assert "state_update" not in ask_node
