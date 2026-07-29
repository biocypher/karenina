"""Behavior tests for turn-scope selectors.

``TurnScope`` subclasses are pure data carriers — they hold a discriminator
literal (``type``) plus optional ``index`` / ``node_id``. The realistic
regression signal is the *contract* downstream code depends on:

1. The discriminator literals are stable (``scenario/checkpoint.py`` maps
   ``"LastTurn"`` to ``"last_turn"`` etc., and ``TurnCheck.scope`` is a
   ``Field(discriminator="type")`` union, so renaming any literal silently
   breaks checkpoint loading and JSON deserialization).
2. Each scope round-trips through JSON and re-dispatches to the *same*
   concrete subclass via the discriminated union on ``TurnCheck.scope``.
3. ``TurnAt`` preserves negative indices verbatim (the evaluator relies on
   this for last-relative indexing).

We do NOT assert ``s.type == "last_turn"`` style constants in isolation —
those would just re-state the literal in the source file. We exercise the
discriminator end-to-end through ``TurnCheck``, which is how scopes are
actually parsed at runtime.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from karenina.schemas.primitives import ExactMatch
from karenina.schemas.primitives.scope import (
    AllTurns,
    AnyTurn,
    FirstTurn,
    LastTurn,
    TurnAt,
    TurnScope,
)
from karenina.schemas.scenario.checks import TurnCheck


def _check_with(scope: TurnScope) -> TurnCheck:
    """Wrap a scope in a TurnCheck so it goes through the discriminated union."""
    return TurnCheck(
        scope=scope,
        field="verify_result",
        expected=True,
        verify_with=ExactMatch(),
    )


_ALL_SCOPES = [
    LastTurn(),
    FirstTurn(),
    TurnAt(index=2),
    TurnAt(index=-1),
    AnyTurn(),
    AnyTurn(node_id="ask"),
    AnyTurn(node_id=["ask", "challenge"]),
    AllTurns(node_id="probe"),
]


@pytest.mark.unit
class TestScopeDiscriminatorRouting:
    """Every scope subclass round-trips through TurnCheck and re-dispatches."""

    @pytest.mark.parametrize("scope", _ALL_SCOPES)
    def test_roundtrip_dispatches_to_same_concrete_type(self, scope: TurnScope) -> None:
        original = _check_with(scope)
        restored = TurnCheck.model_validate_json(original.model_dump_json())
        assert isinstance(restored.scope, type(scope))
        # The discriminator value survives the round-trip exactly.
        assert restored.scope.type == scope.type

    def test_turn_at_negative_index_survives_roundtrip(self) -> None:
        """Negative indices must not be normalized away — the evaluator uses them."""
        check = _check_with(TurnAt(index=-1))
        restored = TurnCheck.model_validate_json(check.model_dump_json())
        assert isinstance(restored.scope, TurnAt)
        assert restored.scope.index == -1

    def test_any_turn_node_id_list_survives_roundtrip(self) -> None:
        check = _check_with(AnyTurn(node_id=["ask", "challenge"]))
        restored = TurnCheck.model_validate_json(check.model_dump_json())
        assert isinstance(restored.scope, AnyTurn)
        assert restored.scope.node_id == ["ask", "challenge"]

    def test_unknown_discriminator_rejected(self) -> None:
        """A bogus 'type' value must not silently coerce to a sibling scope."""
        bad_json = (
            '{"scope": {"type": "middle_turn"}, '
            '"field": "verify_result", "expected": true, '
            '"verify_with": {"type": "ExactMatch"}}'
        )
        with pytest.raises(ValidationError):
            TurnCheck.model_validate_json(bad_json)


def test_scope_subclasses_share_common_base() -> None:
    """All selectors are TurnScope instances (matters for type-narrowing in evaluator code)."""
    for scope in _ALL_SCOPES:
        assert isinstance(scope, TurnScope)
