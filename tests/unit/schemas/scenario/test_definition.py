"""Tests for ScenarioDefinition frozen schema."""

import pytest
from pydantic import ValidationError

from karenina.schemas.primitives import BooleanMatch
from karenina.schemas.scenario.definition import ScenarioDefinition
from karenina.schemas.scenario.types import (
    END,
    ScenarioEdge,
    ScenarioNode,
    ScenarioOutcomeCriterion,
    StateCheck,
)


def _make_question(text="What?"):
    from karenina.schemas.entities import Question

    return Question(question=text, raw_answer="Y", answer_template="class Answer: pass")


@pytest.mark.unit
class TestScenarioDefinition:
    def test_minimal_definition(self):
        q = _make_question()
        defn = ScenarioDefinition(
            name="test",
            nodes={"ask": ScenarioNode(node_id="ask", question=q)},
            edges=[],
            entry_node="ask",
            outcome_criteria=[],
        )
        assert defn.name == "test"
        assert defn.entry_node == "ask"

    def test_frozen(self):
        q = _make_question()
        defn = ScenarioDefinition(
            name="test",
            nodes={"ask": ScenarioNode(node_id="ask", question=q)},
            edges=[],
            entry_node="ask",
            outcome_criteria=[],
        )
        with pytest.raises(ValidationError):
            defn.name = "changed"

    def test_serialization_roundtrip(self):
        q = _make_question()
        cond = StateCheck(field="verify_result", expected=True, verify_with=BooleanMatch())
        defn = ScenarioDefinition(
            name="pipeline",
            description="A test scenario",
            nodes={
                "ask": ScenarioNode(node_id="ask", question=q),
                "done": ScenarioNode(node_id="done", question=q),
            },
            edges=[
                ScenarioEdge(source="ask", target="done", condition=cond),
                ScenarioEdge(source="ask", target=END),
            ],
            entry_node="ask",
            outcome_criteria=[
                ScenarioOutcomeCriterion(
                    name="fast",
                    description="Done quickly?",
                    evaluate_source="lambda r: r.turn_count <= 3",
                ),
            ],
        )
        data = defn.model_dump()
        restored = ScenarioDefinition.model_validate(data)
        assert restored.name == "pipeline"
        assert len(restored.edges) == 2
        assert len(restored.outcome_criteria) == 1

    def test_default_metadata_empty(self):
        q = _make_question()
        defn = ScenarioDefinition(
            name="test",
            nodes={"ask": ScenarioNode(node_id="ask", question=q)},
            edges=[],
            entry_node="ask",
        )
        assert defn.metadata == {}

    def test_default_description_empty(self):
        q = _make_question()
        defn = ScenarioDefinition(
            name="test",
            nodes={"ask": ScenarioNode(node_id="ask", question=q)},
            edges=[],
            entry_node="ask",
        )
        assert defn.description == ""

    def test_default_outcome_criteria_empty(self):
        q = _make_question()
        defn = ScenarioDefinition(
            name="test",
            nodes={"ask": ScenarioNode(node_id="ask", question=q)},
            edges=[],
            entry_node="ask",
        )
        assert defn.outcome_criteria == []
