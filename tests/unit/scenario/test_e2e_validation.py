"""End-to-end test: build scenario, validate, serialize, deserialize."""

import json

import pytest

from karenina.scenario import (
    END,
    ModelOverride,
    Scenario,
    ScenarioDefinition,
    ScenarioOutcomeCriterion,
)
from karenina.scenario.sugar import all_of, last_turn, status_is
from karenina.schemas.config import ModelConfig
from karenina.schemas.entities import Question

TEMPLATE = """
from karenina.schemas.entities.answer import BaseAnswer

class Answer(BaseAnswer):
    value: str = ""
    def ground_truth(self):
        self.correct = {"value": "Y"}
    def verify(self) -> bool:
        return self.value.strip().upper() == self.correct["value"]
"""


@pytest.mark.unit
class TestEndToEndScenario:
    def test_build_validate_serialize_roundtrip(self):
        """Full lifecycle: build -> validate -> serialize -> deserialize."""
        q1 = Question(question="What is X?", raw_answer="Y", answer_template=TEMPLATE)
        q2 = Question(question="Are you sure?", raw_answer="Y", answer_template=TEMPLATE)

        # Build
        s = Scenario("e2e_test", description="End-to-end test scenario")
        s.add_node(
            "ask",
            question=q1,
            state_update="lambda acc, p: {**acc, 'answer': p.get('value')}",
        )
        s.add_node(
            "confirm",
            question=q2,
            model_override=ModelOverride(
                answering_model=ModelConfig(id="gpt4o", model_name="gpt-4o", model_provider="openai"),
            ),
        )
        s.add_edge("ask", "confirm", when={"verify_result": True})
        s.add_edge("ask", END)  # fallback
        s.add_edge("confirm", END)
        s.set_entry("ask")
        s.add_outcome_criterion(
            ScenarioOutcomeCriterion(
                name="completed_both",
                description="Did both stages execute?",
                evaluate=lambda r: r.turn_count == 2,
            )
        )

        # Validate
        defn = s.validate()
        assert isinstance(defn, ScenarioDefinition)
        assert defn.name == "e2e_test"
        assert len(defn.nodes) == 2
        assert len(defn.edges) == 3
        assert len(defn.outcome_criteria) == 1

        # Serialize
        data = defn.model_dump()
        json_str = json.dumps(data, default=str)
        assert '"e2e_test"' in json_str

        # Deserialize
        restored_data = json.loads(json_str)
        restored = ScenarioDefinition.model_validate(restored_data)
        assert restored.name == defn.name
        assert restored.entry_node == "ask"
        assert len(restored.outcome_criteria) == 1
        assert restored.outcome_criteria[0].evaluate_source is not None

    def test_retry_loop_pattern(self):
        """Validate the retry-loop pattern from the spec."""
        q = Question(
            question="What drug targets BCL2?",
            raw_answer="Venetoclax",
            answer_template=TEMPLATE,
        )

        s = Scenario("retry_test")
        s.add_node("ask", question=q)
        s.add_node("retry", question=q)

        s.add_edge("ask", END, when={"verify_result": True})
        s.add_edge("ask", "retry")  # fallback

        s.add_edge("retry", END, when={"verify_result": True})
        s.add_edge("retry", END, when={"node_visits.retry": {"gte": 3}})
        s.add_edge("retry", "retry")  # fallback: retry again

        s.set_entry("ask")
        defn = s.validate()
        assert defn.entry_node == "ask"
        assert len(defn.edges) == 5

    def test_declarative_outcome_with_sugar(self):
        """Test that declarative outcome criteria with sugar functions work."""
        q = Question(question="What is X?", raw_answer="Y", answer_template=TEMPLATE)

        s = Scenario("sugar_test")
        s.add_node("ask", question=q)
        s.add_edge("ask", END)
        s.set_entry("ask")

        # Add declarative outcome using sugar
        s.add_outcome(
            "correct_last",
            all_of(last_turn(verify_result=True), status_is("completed")),
            description="Last turn correct and scenario completed",
        )

        defn = s.validate()
        assert len(defn.outcome_criteria) == 1
        assert defn.outcome_criteria[0].check is not None

        # Verify serialization roundtrip preserves the check
        data = defn.model_dump()
        json_str = json.dumps(data, default=str)
        restored = ScenarioDefinition.model_validate(json.loads(json_str))
        assert restored.outcome_criteria[0].check is not None
