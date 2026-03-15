"""Tests for edge condition evaluation and resolution."""

from __future__ import annotations

import pytest

from karenina.scenario.edge_resolution import (
    evaluate_state_check,
    resolve_next_node,
)
from karenina.schemas.primitives import (
    BooleanMatch,
    ExactMatch,
    NumericRange,
)
from karenina.schemas.scenario.state import ScenarioState
from karenina.schemas.scenario.types import END, ScenarioEdge, StateCheck


def _make_state(**overrides: object) -> ScenarioState:
    defaults: dict[str, object] = {
        "turn": 0,
        "current_node": "a",
        "verify_result": None,
        "parsed": {},
        "node_visits": {},
        "history": [],
        "accumulated": {},
        "node_results": {},
    }
    defaults.update(overrides)
    return ScenarioState(**defaults)


# ---------- evaluate_state_check ----------


@pytest.mark.unit
class TestEvaluateStateCheck:
    def test_boolean_match_true(self) -> None:
        c = StateCheck(field="verify_result", expected=True, verify_with=BooleanMatch())
        state = _make_state(verify_result=True)
        assert evaluate_state_check(c, state) is True

    def test_boolean_match_false(self) -> None:
        c = StateCheck(field="verify_result", expected=True, verify_with=BooleanMatch())
        state = _make_state(verify_result=False)
        assert evaluate_state_check(c, state) is False

    def test_numeric_range_gte(self) -> None:
        c = StateCheck(field="node_visits.retry", verify_with=NumericRange(min=3))
        state = _make_state(node_visits={"retry": 3})
        assert evaluate_state_check(c, state) is True

    def test_numeric_range_below(self) -> None:
        c = StateCheck(field="node_visits.retry", verify_with=NumericRange(min=3))
        state = _make_state(node_visits={"retry": 2})
        assert evaluate_state_check(c, state) is False

    def test_numeric_range_gt(self) -> None:
        c = StateCheck(field="turn", verify_with=NumericRange(min=2, exclusive_min=True))
        assert evaluate_state_check(c, _make_state(turn=3)) is True
        assert evaluate_state_check(c, _make_state(turn=2)) is False

    def test_exact_match_string(self) -> None:
        c = StateCheck(
            field="accumulated.drug",
            expected="venetoclax",
            verify_with=ExactMatch(),
        )
        state = _make_state(accumulated={"drug": "venetoclax"})
        assert evaluate_state_check(c, state) is True

    def test_dot_path_parsed(self) -> None:
        c = StateCheck(field="parsed.category", expected="A", verify_with=ExactMatch())
        state = _make_state(parsed={"category": "A"})
        assert evaluate_state_check(c, state) is True

    def test_dot_path_node_results_parsed(self) -> None:
        c = StateCheck(
            field="node_results.ask.parsed.drug_name",
            expected="venetoclax",
            verify_with=ExactMatch(),
        )
        state = _make_state(
            node_results={
                "ask": {
                    "verify_result": True,
                    "parsed": {"drug_name": "venetoclax"},
                    "rubric": {},
                },
            }
        )
        assert evaluate_state_check(c, state) is True

    def test_dot_path_node_results_rubric(self) -> None:
        c = StateCheck(
            field="node_results.ask.rubric.safety",
            expected=True,
            verify_with=BooleanMatch(),
        )
        state = _make_state(
            node_results={
                "ask": {
                    "verify_result": True,
                    "parsed": {},
                    "rubric": {"safety": True},
                },
            }
        )
        assert evaluate_state_check(c, state) is True

    def test_dot_path_node_results_verify_result(self) -> None:
        c = StateCheck(
            field="node_results.ask.verify_result",
            expected=True,
            verify_with=BooleanMatch(),
        )
        state = _make_state(
            node_results={
                "ask": {"verify_result": True, "parsed": {}, "rubric": {}},
            }
        )
        assert evaluate_state_check(c, state) is True

    def test_missing_path_returns_none(self) -> None:
        c = StateCheck(field="accumulated.missing", expected=None, verify_with=BooleanMatch())
        state = _make_state(accumulated={})
        # BooleanMatch: check(None, None) -> bool(None) == bool(None) -> True
        assert evaluate_state_check(c, state) is True

    def test_missing_node_visits_returns_zero(self) -> None:
        c = StateCheck(field="node_visits.unvisited", verify_with=NumericRange(max=0))
        state = _make_state(node_visits={})
        assert evaluate_state_check(c, state) is True

    def test_current_node_path(self) -> None:
        c = StateCheck(field="current_node", expected="ask", verify_with=ExactMatch())
        state = _make_state(current_node="ask")
        assert evaluate_state_check(c, state) is True

    def test_and_conditions(self) -> None:
        checks = [
            StateCheck(field="verify_result", expected=True, verify_with=BooleanMatch()),
            StateCheck(field="turn", verify_with=NumericRange(max=5, exclusive_max=True)),
        ]
        state = _make_state(verify_result=True, turn=3)
        assert all(evaluate_state_check(c, state) for c in checks)

    def test_unknown_root_returns_none(self) -> None:
        c = StateCheck(field="nonexistent", expected=None, verify_with=BooleanMatch())
        state = _make_state()
        # bool(None) == bool(None) -> True
        assert evaluate_state_check(c, state) is True


# ---------- resolve_next_node ----------


@pytest.mark.unit
class TestResolveNextNode:
    def test_unconditional_is_fallback(self) -> None:
        edges = [
            ScenarioEdge(source="a", target="b"),
            ScenarioEdge(
                source="a",
                target="c",
                condition=StateCheck(field="verify_result", expected=True, verify_with=BooleanMatch()),
            ),
        ]
        state = _make_state(verify_result=True)
        assert resolve_next_node(edges, state) == "c"

    def test_unconditional_fallback_taken(self) -> None:
        edges = [
            ScenarioEdge(
                source="a",
                target="c",
                condition=StateCheck(field="verify_result", expected=True, verify_with=BooleanMatch()),
            ),
            ScenarioEdge(source="a", target="b"),
        ]
        state = _make_state(verify_result=False)
        assert resolve_next_node(edges, state) == "b"

    def test_no_edges_returns_none(self) -> None:
        assert resolve_next_node([], _make_state()) is None

    def test_callable_condition(self) -> None:
        edges = [
            ScenarioEdge(source="a", target="b", condition_callable=lambda s: s.turn > 2),
            ScenarioEdge(source="a", target="c"),
        ]
        state = _make_state(turn=5)
        assert resolve_next_node(edges, state) == "b"

    def test_callable_condition_not_matched(self) -> None:
        edges = [
            ScenarioEdge(source="a", target="b", condition_callable=lambda s: s.turn > 10),
            ScenarioEdge(source="a", target="c"),
        ]
        state = _make_state(turn=5)
        assert resolve_next_node(edges, state) == "c"

    def test_callable_exception_treated_as_no_match(self) -> None:
        def bad_callable(s: ScenarioState) -> bool:
            raise ValueError("boom")

        edges = [
            ScenarioEdge(source="a", target="b", condition_callable=bad_callable),
            ScenarioEdge(source="a", target="c"),
        ]
        assert resolve_next_node(edges, _make_state()) == "c"

    def test_first_match_wins(self) -> None:
        edges = [
            ScenarioEdge(
                source="a",
                target="b",
                condition=StateCheck(field="verify_result", expected=True, verify_with=BooleanMatch()),
            ),
            ScenarioEdge(
                source="a",
                target="c",
                condition=StateCheck(field="verify_result", expected=True, verify_with=BooleanMatch()),
            ),
            ScenarioEdge(source="a", target="d"),
        ]
        state = _make_state(verify_result=True)
        assert resolve_next_node(edges, state) == "b"

    def test_end_target_returned(self) -> None:
        edges = [ScenarioEdge(source="a", target=END)]
        assert resolve_next_node(edges, _make_state()) == END

    def test_list_condition_all_must_pass(self) -> None:
        edges = [
            ScenarioEdge(
                source="a",
                target="b",
                condition=[
                    StateCheck(field="verify_result", expected=True, verify_with=BooleanMatch()),
                    StateCheck(field="turn", verify_with=NumericRange(min=2)),
                ],
            ),
            ScenarioEdge(source="a", target="c"),
        ]
        # Both conditions met
        assert resolve_next_node(edges, _make_state(verify_result=True, turn=3)) == "b"
        # Only one condition met
        assert resolve_next_node(edges, _make_state(verify_result=True, turn=1)) == "c"

    def test_only_unconditional_edges(self) -> None:
        edges = [
            ScenarioEdge(source="a", target="b"),
            ScenarioEdge(source="a", target="c"),
        ]
        # First unconditional wins as fallback
        assert resolve_next_node(edges, _make_state()) == "b"

    def test_no_match_no_fallback_returns_none(self) -> None:
        edges = [
            ScenarioEdge(
                source="a",
                target="b",
                condition=StateCheck(field="verify_result", expected=True, verify_with=BooleanMatch()),
            ),
        ]
        state = _make_state(verify_result=False)
        assert resolve_next_node(edges, state) is None
