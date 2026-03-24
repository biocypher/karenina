"""Tests for the Scenario builder class."""

from __future__ import annotations

import pytest

from karenina.scenario.builder import Scenario
from karenina.schemas.entities.question import Question
from karenina.schemas.primitives import (
    BooleanMatch,
    ExactMatch,
    NumericExact,
    NumericRange,
)
from karenina.schemas.scenario.checks import ResultCheck, TurnCheck
from karenina.schemas.scenario.definition import ScenarioDefinition
from karenina.schemas.scenario.types import (
    END,
    ModelOverride,
    ScenarioOutcomeCriterion,
    StateCheck,
)


def _make_question(text: str = "What drug targets BCL2?") -> Question:
    return Question(question=text, raw_answer="venetoclax")


# ---------------------------------------------------------------------------
# Node management
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAddNode:
    def test_add_node(self) -> None:
        s = Scenario("test")
        q = _make_question()
        s.add_node("q1", question=q)
        assert "q1" in s._nodes

    def test_add_node_duplicate_raises(self) -> None:
        s = Scenario("test")
        q = _make_question()
        s.add_node("q1", question=q)
        with pytest.raises(ValueError, match="already exists"):
            s.add_node("q1", question=q)

    def test_add_node_deep_copies_question(self) -> None:
        s = Scenario("test")
        q = _make_question()
        s.add_node("q1", question=q)
        q.raw_answer = "something else"
        assert s._nodes["q1"].question.raw_answer == "venetoclax"

    def test_add_node_with_model_override(self) -> None:
        from karenina.schemas.config import ModelConfig

        s = Scenario("test")
        override = ModelOverride(answering_model=ModelConfig(id="gpt-4", model_provider="openai", model_name="gpt-4"))
        s.add_node("q1", question=_make_question(), model_override=override)
        assert s._nodes["q1"].model_override is not None
        assert s._nodes["q1"].model_override.answering_model.id == "gpt-4"

    def test_add_node_with_state_update_lambda(self) -> None:
        s = Scenario("test")
        s.add_node(
            "q1",
            question=_make_question(),
            state_update=lambda acc, _p: {**acc, "seen": True},
        )
        node = s._nodes["q1"]
        assert node.state_update is not None
        assert node.state_update_source is not None
        assert "lambda" in node.state_update_source

    def test_add_node_with_state_update_string(self) -> None:
        s = Scenario("test")
        s.add_node(
            "q1",
            question=_make_question(),
            state_update="lambda acc, p: {**acc, 'seen': True}",
        )
        node = s._nodes["q1"]
        assert node.state_update is not None
        assert callable(node.state_update)
        assert node.state_update_source == "lambda acc, p: {**acc, 'seen': True}"

    def test_add_node_with_metadata_kwargs(self) -> None:
        s = Scenario("test")
        s.add_node("q1", question=_make_question(), difficulty="hard", domain="pharma")
        assert s._nodes["q1"].metadata == {"difficulty": "hard", "domain": "pharma"}


# ---------------------------------------------------------------------------
# Edge management
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAddEdge:
    def _scenario_with_nodes(self) -> Scenario:
        s = Scenario("test")
        s.add_node("q1", question=_make_question("Q1"))
        s.add_node("q2", question=_make_question("Q2"))
        s.add_node("retry", question=_make_question("Retry"))
        return s

    def test_add_edge_unconditional(self) -> None:
        s = self._scenario_with_nodes()
        s.add_edge("q1", "q2")
        assert len(s._edges) == 1
        edge = s._edges[0]
        assert edge.source == "q1"
        assert edge.target == "q2"
        assert edge.condition is None

    def test_add_edge_with_dict_shorthand(self) -> None:
        s = self._scenario_with_nodes()
        s.add_edge("q1", "q2", when={"verify_result": True})
        edge = s._edges[0]
        assert isinstance(edge.condition, StateCheck)
        assert edge.condition.field == "verify_result"
        assert edge.condition.expected is True
        assert isinstance(edge.condition.verify_with, BooleanMatch)

    def test_add_edge_with_operator_shorthand(self) -> None:
        s = self._scenario_with_nodes()
        s.add_edge("q1", "retry", when={"node_visits.retry": {"gte": 3}})
        edge = s._edges[0]
        assert isinstance(edge.condition, StateCheck)
        assert edge.condition.field == "node_visits.retry"
        assert isinstance(edge.condition.verify_with, NumericRange)
        assert edge.condition.verify_with.min == 3

    def test_add_edge_with_string_exact_match(self) -> None:
        s = self._scenario_with_nodes()
        s.add_edge("q1", "q2", when={"parsed.drug": "venetoclax"})
        edge = s._edges[0]
        assert isinstance(edge.condition, StateCheck)
        assert edge.condition.field == "parsed.drug"
        assert edge.condition.expected == "venetoclax"
        assert isinstance(edge.condition.verify_with, ExactMatch)

    def test_add_edge_with_numeric_exact(self) -> None:
        s = self._scenario_with_nodes()
        s.add_edge("q1", "q2", when={"turn": 5})
        edge = s._edges[0]
        assert isinstance(edge.condition, StateCheck)
        assert edge.condition.field == "turn"
        assert edge.condition.expected == 5
        assert isinstance(edge.condition.verify_with, NumericExact)

    def test_add_edge_to_end(self) -> None:
        s = self._scenario_with_nodes()
        s.add_edge("q1", END)
        edge = s._edges[0]
        assert edge.target == END

    def test_add_edge_with_callable(self) -> None:
        s = self._scenario_with_nodes()
        s.add_edge("q1", "q2", when=lambda state: state.get("verify_result"))
        edge = s._edges[0]
        assert edge.condition_callable is not None
        assert edge.condition_source is not None

    def test_add_edge_with_string_callable(self) -> None:
        s = self._scenario_with_nodes()
        s.add_edge("q1", "q2", when="lambda state: state.get('verify_result')")
        edge = s._edges[0]
        assert edge.condition_callable is not None
        assert callable(edge.condition_callable)
        assert edge.condition_source == "lambda state: state.get('verify_result')"

    def test_add_edge_with_state_check(self) -> None:
        s = self._scenario_with_nodes()
        check = StateCheck(field="verify_result", expected=True, verify_with=BooleanMatch())
        s.add_edge("q1", "q2", when=check)
        edge = s._edges[0]
        assert isinstance(edge.condition, StateCheck)
        assert edge.condition.field == "verify_result"

    def test_add_edge_with_list_of_dicts(self) -> None:
        s = self._scenario_with_nodes()
        s.add_edge(
            "q1",
            "q2",
            when=[
                {"verify_result": True},
                {"parsed.drug": "venetoclax"},
            ],
        )
        edge = s._edges[0]
        assert isinstance(edge.condition, list)
        assert len(edge.condition) == 2
        assert all(isinstance(c, StateCheck) for c in edge.condition)


# ---------------------------------------------------------------------------
# Entry node
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSetEntry:
    def test_set_entry(self) -> None:
        s = Scenario("test")
        s.add_node("q1", question=_make_question())
        s.set_entry("q1")
        assert s._entry_node == "q1"

    def test_set_entry_nonexistent_raises(self) -> None:
        s = Scenario("test")
        with pytest.raises(ValueError, match="not a known node"):
            s.set_entry("nonexistent")


# ---------------------------------------------------------------------------
# Outcome criteria
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOutcomeCriteria:
    def test_add_outcome_criterion(self) -> None:
        s = Scenario("test")
        criterion = ScenarioOutcomeCriterion(
            name="passes",
            description="Last turn passes",
            check=TurnCheck(
                scope={"type": "last_turn"},
                field="verify_result",
                expected=True,
                verify_with=BooleanMatch(),
            ),
        )
        s.add_outcome_criterion(criterion)
        assert len(s._outcome_criteria) == 1
        assert s._outcome_criteria[0].name == "passes"

    def test_add_outcome(self) -> None:
        s = Scenario("test")
        check = ResultCheck(field="status", expected="completed", verify_with=ExactMatch())
        s.add_outcome("completed", check, description="Scenario completed normally")
        assert len(s._outcome_criteria) == 1
        assert s._outcome_criteria[0].name == "completed"
        assert s._outcome_criteria[0].description == "Scenario completed normally"
        assert s._outcome_criteria[0].check is not None


# ---------------------------------------------------------------------------
# Validate (build)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestScenarioValidate:
    def _valid_scenario(self) -> Scenario:
        s = Scenario("simple", description="A simple two-node scenario")
        s.add_node("q1", question=_make_question("Q1"))
        s.add_node("q2", question=_make_question("Q2"))
        s.add_edge("q1", "q2")
        s.add_edge("q2", END)
        s.set_entry("q1")
        return s

    def test_valid_scenario(self) -> None:
        s = self._valid_scenario()
        defn = s.validate()
        assert isinstance(defn, ScenarioDefinition)
        assert defn.name == "simple"
        assert defn.entry_node == "q1"
        assert len(defn.nodes) == 2
        assert len(defn.edges) == 2

    def test_no_entry_raises(self) -> None:
        s = Scenario("test")
        s.add_node("q1", question=_make_question())
        with pytest.raises(ValueError, match="entry"):
            s.validate()

    def test_entry_not_in_nodes(self) -> None:
        s = Scenario("test")
        s.add_node("q1", question=_make_question())
        s._entry_node = "nonexistent"
        with pytest.raises(ValueError, match="(?i)entry.*not.*node"):
            s.validate()

    def test_edge_source_not_in_nodes(self) -> None:
        s = Scenario("test")
        s.add_node("q1", question=_make_question())
        s.set_entry("q1")
        # Manually inject a bad edge to bypass add_edge validation
        from karenina.schemas.scenario.types import ScenarioEdge

        s._edges.append(ScenarioEdge(source="ghost", target="q1"))
        with pytest.raises(ValueError, match="ghost"):
            s.validate()

    def test_orphan_node(self) -> None:
        s = Scenario("test")
        s.add_node("q1", question=_make_question("Q1"))
        s.add_node("q2", question=_make_question("Q2"))
        s.add_node("orphan", question=_make_question("Orphan"))
        s.add_edge("q1", "q2")
        s.add_edge("q2", END)
        s.set_entry("q1")
        with pytest.raises(ValueError, match="orphan"):
            s.validate()

    def test_conditional_without_fallback(self) -> None:
        s = Scenario("test")
        s.add_node("q1", question=_make_question("Q1"))
        s.add_node("q2", question=_make_question("Q2"))
        s.set_entry("q1")
        # Only a conditional edge, no unconditional fallback
        s.add_edge("q1", "q2", when={"verify_result": True})
        with pytest.raises(ValueError, match="fallback"):
            s.validate()

    def test_implicit_terminal(self) -> None:
        """A node with no outbound edges is a valid implicit terminal."""
        s = Scenario("test")
        s.add_node("q1", question=_make_question("Q1"))
        s.add_node("q2", question=_make_question("Q2"))
        s.add_edge("q1", "q2")
        s.set_entry("q1")
        # q2 has no outbound edges: that is valid (implicit terminal)
        defn = s.validate()
        assert isinstance(defn, ScenarioDefinition)
