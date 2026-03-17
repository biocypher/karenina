"""Tests for scenario builder sugar functions."""

from __future__ import annotations

import pytest

from karenina.scenario.sugar import (
    all_of,
    all_turns,
    any_of,
    any_turn,
    at_least_n,
    count_turns,
    cross_turn,
    first_match_index,
    first_turn,
    first_turn_scope,
    last_turn,
    last_turn_scope,
    status_is,
    turn_at,
    turn_count_eq,
    turn_count_gte,
)
from karenina.schemas.primitives import (
    BooleanMatch,
    ExactMatch,
    NumericExact,
    NumericRange,
)
from karenina.schemas.primitives.composition import AllOf, AnyOf, AtLeastN
from karenina.schemas.primitives.scope import (
    AllTurns,
    AnyTurn,
    FirstTurn,
    LastTurn,
    TurnAt,
)
from karenina.schemas.scenario.checks import (
    CountTurns,
    CrossTurnCheck,
    FirstMatchIndex,
    ResultCheck,
    TurnCheck,
)

# ---------------------------------------------------------------------------
# Turn sugar
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLastTurn:
    def test_single_kwarg(self) -> None:
        result = last_turn(verify_result=True)
        assert isinstance(result, TurnCheck)
        assert isinstance(result.scope, LastTurn)
        assert result.field == "verify_result"
        assert result.expected is True
        assert isinstance(result.verify_with, BooleanMatch)

    def test_multiple_kwargs(self) -> None:
        result = last_turn(verify_result=True, node_id="q1")
        assert isinstance(result, AllOf)
        assert len(result.conditions) == 2
        for cond in result.conditions:
            assert isinstance(cond, TurnCheck)
            assert isinstance(cond.scope, LastTurn)

    def test_string_value(self) -> None:
        result = last_turn(node_id="q1")
        assert isinstance(result, TurnCheck)
        assert result.field == "node_id"
        assert result.expected == "q1"
        assert isinstance(result.verify_with, ExactMatch)

    def test_numeric_value(self) -> None:
        result = last_turn(turn=5)
        assert isinstance(result, TurnCheck)
        assert isinstance(result.verify_with, NumericExact)


@pytest.mark.unit
class TestFirstTurn:
    def test_single_kwarg(self) -> None:
        result = first_turn(verify_result=True)
        assert isinstance(result, TurnCheck)
        assert isinstance(result.scope, FirstTurn)
        assert result.field == "verify_result"

    def test_multiple_kwargs(self) -> None:
        result = first_turn(verify_result=False, node_id="q1")
        assert isinstance(result, AllOf)
        assert len(result.conditions) == 2


@pytest.mark.unit
class TestAnyTurn:
    def test_basic(self) -> None:
        result = any_turn(verify_result=True)
        assert isinstance(result, TurnCheck)
        assert isinstance(result.scope, AnyTurn)
        assert result.scope.node_id is None

    def test_with_node(self) -> None:
        result = any_turn(node="retry", verify_result=False)
        assert isinstance(result, TurnCheck)
        assert isinstance(result.scope, AnyTurn)
        assert result.scope.node_id == "retry"


@pytest.mark.unit
class TestAllTurns:
    def test_basic(self) -> None:
        result = all_turns(verify_result=True)
        assert isinstance(result, TurnCheck)
        assert isinstance(result.scope, AllTurns)
        assert result.scope.node_id is None

    def test_with_node(self) -> None:
        result = all_turns(node="q1", verify_result=True)
        assert isinstance(result, TurnCheck)
        assert isinstance(result.scope, AllTurns)
        assert result.scope.node_id == "q1"


# ---------------------------------------------------------------------------
# Result sugar
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResultSugar:
    def test_status_is(self) -> None:
        result = status_is("completed")
        assert isinstance(result, ResultCheck)
        assert result.field == "status"
        assert result.expected == "completed"
        assert isinstance(result.verify_with, ExactMatch)

    def test_turn_count_gte(self) -> None:
        result = turn_count_gte(3)
        assert isinstance(result, ResultCheck)
        assert result.field == "turn_count"
        assert isinstance(result.verify_with, NumericRange)
        assert result.verify_with.min == 3

    def test_turn_count_eq(self) -> None:
        result = turn_count_eq(5)
        assert isinstance(result, ResultCheck)
        assert result.field == "turn_count"
        assert result.expected == 5
        assert isinstance(result.verify_with, NumericExact)


# ---------------------------------------------------------------------------
# Aggregation sugar
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAggregationSugar:
    def test_count_turns(self) -> None:
        result = count_turns(node="retry", verify_result=False)
        assert isinstance(result, CountTurns)
        assert result.node_id == "retry"
        assert result.verify_result is False

    def test_count_turns_defaults(self) -> None:
        result = count_turns()
        assert isinstance(result, CountTurns)
        assert result.node_id is None
        assert result.verify_result is None

    def test_first_match_index(self) -> None:
        result = first_match_index(node="q2", verify_result=True)
        assert isinstance(result, FirstMatchIndex)
        assert result.node_id == "q2"
        assert result.verify_result is True


# ---------------------------------------------------------------------------
# Cross-turn sugar
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCrossTurn:
    def test_cross_turn(self) -> None:
        result = cross_turn(
            source=FirstTurn(),
            source_field="parsed.drug",
            target=LastTurn(),
            target_field="parsed.drug",
            comparison="eq",
        )
        assert isinstance(result, CrossTurnCheck)
        assert isinstance(result.source_turn, FirstTurn)
        assert isinstance(result.target_turn, LastTurn)
        assert result.source_field == "parsed.drug"
        assert result.target_field == "parsed.drug"
        assert result.comparison == "eq"

    def test_cross_turn_with_normalize(self) -> None:
        result = cross_turn(
            source=FirstTurn(),
            source_field="raw_response",
            target=LastTurn(),
            target_field="raw_response",
            comparison="contains",
            normalize=["lowercase"],
        )
        assert result.normalize == ["lowercase"]


# ---------------------------------------------------------------------------
# Scope helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestScopeHelpers:
    def test_first_turn_scope(self) -> None:
        result = first_turn_scope()
        assert isinstance(result, FirstTurn)

    def test_last_turn_scope(self) -> None:
        result = last_turn_scope()
        assert isinstance(result, LastTurn)

    def test_turn_at(self) -> None:
        result = turn_at(2)
        assert isinstance(result, TurnAt)
        assert result.index == 2


# ---------------------------------------------------------------------------
# Composition helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCompositionHelpers:
    def test_all_of(self) -> None:
        a = last_turn(verify_result=True)
        b = status_is("completed")
        result = all_of(a, b)
        assert isinstance(result, AllOf)
        assert len(result.conditions) == 2

    def test_any_of(self) -> None:
        a = last_turn(verify_result=True)
        b = status_is("completed")
        result = any_of(a, b)
        assert isinstance(result, AnyOf)
        assert len(result.conditions) == 2

    def test_at_least_n(self) -> None:
        a = last_turn(verify_result=True)
        b = status_is("completed")
        c = turn_count_gte(2)
        result = at_least_n(2, a, b, c)
        assert isinstance(result, AtLeastN)
        assert result.n == 2
        assert len(result.conditions) == 3
