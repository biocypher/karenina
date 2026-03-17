"""Tests for turn scope selectors."""

from __future__ import annotations

import pytest

from karenina.schemas.primitives.scope import (
    AllTurns,
    AnyTurn,
    FirstTurn,
    LastTurn,
    TurnAt,
    TurnScope,
)


@pytest.mark.unit
class TestScopeSelectors:
    def test_last_turn_type(self):
        s = LastTurn()
        assert s.type == "last_turn"
        assert isinstance(s, TurnScope)

    def test_first_turn_type(self):
        s = FirstTurn()
        assert s.type == "first_turn"

    def test_turn_at_index(self):
        s = TurnAt(index=2)
        assert s.index == 2
        assert s.type == "turn_at"

    def test_turn_at_negative_index(self):
        s = TurnAt(index=-1)
        assert s.index == -1

    def test_any_turn_no_filter(self):
        s = AnyTurn()
        assert s.node_id is None
        assert s.type == "any_turn"

    def test_any_turn_single_node(self):
        s = AnyTurn(node_id="ask")
        assert s.node_id == "ask"

    def test_any_turn_multiple_nodes(self):
        s = AnyTurn(node_id=["ask", "challenge"])
        assert s.node_id == ["ask", "challenge"]

    def test_all_turns_with_filter(self):
        s = AllTurns(node_id="probe")
        assert s.node_id == "probe"
        assert s.type == "all_turns"

    def test_serialization_roundtrip(self):
        s = AnyTurn(node_id=["a", "b"])
        data = s.model_dump()
        restored = AnyTurn.model_validate(data)
        assert restored.node_id == ["a", "b"]
