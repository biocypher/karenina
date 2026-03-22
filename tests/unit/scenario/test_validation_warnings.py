"""Tests for scenario graph validation warnings."""

import warnings

import pytest

from karenina.schemas.entities import Question
from karenina.schemas.scenario.types import END, ScenarioEdge, ScenarioNode


def _make_question(text: str = "What?") -> Question:
    return Question(question=text, raw_answer="Y", answer_template="class Answer: pass")


def _make_node(node_id: str, text: str = "What?") -> ScenarioNode:
    return ScenarioNode(node_id=node_id, question=_make_question(text))


@pytest.mark.unit
class TestMultipleUnconditionalEdgesWarning:
    """Issue 105: warn when a node has multiple unconditional edges."""

    def test_warns_on_multiple_unconditional_edges_from_same_source(self):
        """Two unconditional edges from same source triggers UserWarning."""
        from karenina.scenario.validation import validate_scenario_graph

        nodes = {
            "ask": _make_node("ask", "Q1?"),
            "path_a": _make_node("path_a", "Q2?"),
            "path_b": _make_node("path_b", "Q3?"),
        }
        edges = [
            ScenarioEdge(source="ask", target="path_a"),  # unconditional
            ScenarioEdge(source="ask", target="path_b"),  # unconditional (duplicate)
            ScenarioEdge(source="path_a", target=END),
            ScenarioEdge(source="path_b", target=END),
        ]

        with pytest.warns(UserWarning, match="ask"):
            validate_scenario_graph(nodes, edges, entry_node="ask")

    def test_no_warning_on_single_unconditional_edge(self):
        """Single unconditional edge does not trigger warning."""
        from karenina.scenario.validation import validate_scenario_graph

        nodes = {
            "ask": _make_node("ask", "Q1?"),
            "confirm": _make_node("confirm", "Q2?"),
        }
        edges = [
            ScenarioEdge(source="ask", target="confirm"),  # single unconditional
            ScenarioEdge(source="confirm", target=END),
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            validate_scenario_graph(nodes, edges, entry_node="ask")
