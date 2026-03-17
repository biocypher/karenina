"""Turn scope selectors for scenario evaluation.

Scope selectors specify which turn(s) from execution history to inspect.
They carry no evaluation logic; the evaluator interprets them.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class TurnScope(BaseModel):
    """Base for all turn selection strategies."""

    pass


class LastTurn(TurnScope):
    """Select the last turn in the execution history."""

    type: Literal["last_turn"] = "last_turn"


class FirstTurn(TurnScope):
    """Select the first turn in the execution history."""

    type: Literal["first_turn"] = "first_turn"


class TurnAt(TurnScope):
    """Select a specific turn by index. Supports negative indexing."""

    type: Literal["turn_at"] = "turn_at"
    index: int


class AnyTurn(TurnScope):
    """Quantifier: true if any matching turn satisfies the check.

    When node_id is provided, only turns at that node (or nodes) are
    considered. When node_id is None, all turns are considered.
    Both AnyTurn and AllTurns return False if no turns match the
    filter (no vacuous truth).
    """

    type: Literal["any_turn"] = "any_turn"
    node_id: str | list[str] | None = None


class AllTurns(TurnScope):
    """Quantifier: true if all matching turns satisfy the check.

    Filtering follows the same rules as AnyTurn.
    """

    type: Literal["all_turns"] = "all_turns"
    node_id: str | list[str] | None = None
