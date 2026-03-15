"""Tests for scenario check node types."""

from __future__ import annotations

import pytest

from karenina.schemas.primitives import BooleanMatch, ExactMatch, NumericRange
from karenina.schemas.primitives.scope import AnyTurn, LastTurn, TurnAt
from karenina.schemas.scenario.checks import (
    CountTurns,
    CrossTurnCheck,
    FirstMatchIndex,
    ResultCheck,
    TurnCheck,
)


@pytest.mark.unit
class TestTurnCheck:
    def test_basic_turn_check(self):
        tc = TurnCheck(
            scope=LastTurn(),
            field="verify_result",
            expected=True,
            verify_with=BooleanMatch(),
        )
        assert tc.type == "turn_check"
        assert isinstance(tc.scope, LastTurn)

    def test_turn_check_with_any_turn(self):
        tc = TurnCheck(
            scope=AnyTurn(node_id="ask"),
            field="parsed.drug",
            expected="venetoclax",
            verify_with=ExactMatch(),
        )
        assert tc.scope.node_id == "ask"

    def test_serialization_roundtrip(self):
        tc = TurnCheck(
            scope=TurnAt(index=0),
            field="verify_result",
            expected=True,
            verify_with=BooleanMatch(),
        )
        data = tc.model_dump()
        restored = TurnCheck.model_validate(data)
        assert restored.scope.index == 0


@pytest.mark.unit
class TestResultCheck:
    def test_status_check(self):
        rc = ResultCheck(
            field="status",
            expected="completed",
            verify_with=ExactMatch(),
        )
        assert rc.type == "result_check"
        assert rc.field == "status"

    def test_turn_count_check(self):
        rc = ResultCheck(
            field="turn_count",
            verify_with=NumericRange(min=1, max=5),
        )
        assert rc.field == "turn_count"


@pytest.mark.unit
class TestCrossTurnCheck:
    def test_basic_cross_turn(self):
        ct = CrossTurnCheck(
            source_turn=TurnAt(index=0),
            source_field="parsed.drug",
            target_turn=LastTurn(),
            target_field="parsed.drug",
            comparison="eq",
        )
        assert ct.type == "cross_turn_check"
        assert ct.comparison == "eq"

    def test_with_normalizers(self):
        ct = CrossTurnCheck(
            source_turn=TurnAt(index=0),
            source_field="raw_response",
            target_turn=LastTurn(),
            target_field="raw_response",
            comparison="contains",
            normalize=["lowercase"],
        )
        assert len(ct.normalize) == 1


@pytest.mark.unit
class TestCountTurns:
    def test_no_filter(self):
        c = CountTurns()
        assert c.type == "count_turns"
        assert c.node_id is None
        assert c.verify_result is None

    def test_with_node_filter(self):
        c = CountTurns(node_id="retry", verify_result=True)
        assert c.node_id == "retry"


@pytest.mark.unit
class TestFirstMatchIndex:
    def test_basic(self):
        f = FirstMatchIndex(node_id="challenge", verify_result=False)
        assert f.type == "first_match_index"
        assert f.node_id == "challenge"
