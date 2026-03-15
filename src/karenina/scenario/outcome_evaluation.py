"""Outcome criterion evaluation for completed scenario executions."""

from __future__ import annotations

import logging
from typing import Any

from karenina.schemas.primitives.composition import evaluate_composition
from karenina.schemas.primitives.normalizers import apply_normalizers
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
from karenina.schemas.scenario.state import ScenarioExecutionResult, TurnRecord

logger = logging.getLogger(__name__)


def evaluate_outcome(
    node: Any,
    result: ScenarioExecutionResult,
) -> bool | int | float:
    """Evaluate an outcome node against a completed execution result.

    Boolean check nodes (TurnCheck, ResultCheck, CrossTurnCheck) compose
    via AllOf/AnyOf/AtLeastN. Aggregation types (CountTurns, FirstMatchIndex)
    return int.
    """
    if isinstance(node, CountTurns):
        return _evaluate_count_turns(node, result)
    if isinstance(node, FirstMatchIndex):
        return _evaluate_first_match_index(node, result)

    # Boolean check nodes + composition
    return evaluate_composition(node, lambda leaf: _evaluate_check_leaf(leaf, result))


def _evaluate_check_leaf(leaf: Any, result: ScenarioExecutionResult) -> bool:
    """Evaluate a single boolean check leaf node."""
    if isinstance(leaf, TurnCheck):
        return _evaluate_turn_check(leaf, result)
    if isinstance(leaf, ResultCheck):
        return _evaluate_result_check(leaf, result)
    if isinstance(leaf, CrossTurnCheck):
        return _evaluate_cross_turn_check(leaf, result)
    logger.warning("Unknown check leaf type: %s", type(leaf).__name__)
    return False


# ---------- TurnCheck ----------


def _evaluate_turn_check(check: TurnCheck, result: ScenarioExecutionResult) -> bool:
    """Evaluate a TurnCheck by resolving its scope to turn(s) and checking the field."""
    scope = check.scope
    history = result.history

    if isinstance(scope, LastTurn):
        if not history:
            return False
        return _check_turn_field(check, history[-1])

    if isinstance(scope, FirstTurn):
        if not history:
            return False
        return _check_turn_field(check, history[0])

    if isinstance(scope, TurnAt):
        try:
            turn = history[scope.index]
        except IndexError:
            return False
        return _check_turn_field(check, turn)

    if isinstance(scope, AnyTurn):
        filtered = _filter_turns_by_node(history, scope.node_id)
        if not filtered:
            return False
        return any(_check_turn_field(check, t) for t in filtered)

    if isinstance(scope, AllTurns):
        filtered = _filter_turns_by_node(history, scope.node_id)
        if not filtered:
            return False
        return all(_check_turn_field(check, t) for t in filtered)

    logger.warning("Unknown scope type: %s", type(scope).__name__)
    return False


def _check_turn_field(check: TurnCheck, turn: TurnRecord) -> bool:
    """Extract a field from a TurnRecord and run the verification primitive."""
    value = _resolve_turn_field(check.field, turn)
    return check.verify_with.check(value, check.expected)


def _resolve_turn_field(field: str, turn: TurnRecord) -> Any:
    """Resolve a field path against a TurnRecord.

    Supported fields:
        "node_id": turn.node_id
        "verify_result": turn.verify_result
        "raw_response": turn.raw_response
        "question_text": turn.question_text
        "parsed.<x>": turn.parsed_fields[x]

    Returns None for missing keys.
    """
    parts = field.split(".", 1)
    root = parts[0]

    if root == "parsed" and len(parts) == 2:
        return turn.parsed_fields.get(parts[1])

    return getattr(turn, root, None)


def _filter_turns_by_node(
    history: list[TurnRecord],
    node_id: str | list[str] | None,
) -> list[TurnRecord]:
    """Filter turns by node_id. Returns all turns if node_id is None."""
    if node_id is None:
        return history
    if isinstance(node_id, str):
        return [t for t in history if t.node_id == node_id]
    return [t for t in history if t.node_id in node_id]


# ---------- ResultCheck ----------


def _evaluate_result_check(
    check: ResultCheck,
    result: ScenarioExecutionResult,
) -> bool:
    """Evaluate a ResultCheck against execution-level fields."""
    value = _resolve_result_field(check.field, result)
    return check.verify_with.check(value, check.expected)


def _resolve_result_field(field: str, result: ScenarioExecutionResult) -> Any:
    """Resolve a field name against ScenarioExecutionResult.

    Supported fields: "status", "turn_count", "path", "scenario_id".
    Returns None for unrecognized fields.
    """
    return getattr(result, field, None)


# ---------- CrossTurnCheck ----------


_COMPARISON_OPS: dict[str, Any] = {
    "eq": lambda t, s: t == s,
    "neq": lambda t, s: t != s,
    "gt": lambda t, s: t > s,
    "gte": lambda t, s: t >= s,
    "lt": lambda t, s: t < s,
    "lte": lambda t, s: t <= s,
    "contains": lambda t, s: str(s) in str(t),
}


def _evaluate_cross_turn_check(
    check: CrossTurnCheck,
    result: ScenarioExecutionResult,
) -> bool:
    """Compare values between two different turns.

    Semantics: target_value <comparison> source_value.
    "contains" means target contains source. "gt" means target > source.
    """
    source_turn = _resolve_single_turn(check.source_turn, result.history)
    target_turn = _resolve_single_turn(check.target_turn, result.history)

    if source_turn is None or target_turn is None:
        return False

    source_val = _resolve_turn_field(check.source_field, source_turn)
    target_val = _resolve_turn_field(check.target_field, target_turn)

    if check.normalize:
        source_val = apply_normalizers(check.normalize, str(source_val))
        target_val = apply_normalizers(check.normalize, str(target_val))

    op = _COMPARISON_OPS.get(check.comparison)
    if op is None:
        logger.warning("Unknown comparison operator: %s", check.comparison)
        return False

    try:
        return bool(op(target_val, source_val))
    except (TypeError, ValueError):
        logger.warning(
            "CrossTurnCheck comparison %s failed for %r vs %r",
            check.comparison,
            target_val,
            source_val,
            exc_info=True,
        )
        return False


def _resolve_single_turn(
    scope: Any,
    history: list[TurnRecord],
) -> TurnRecord | None:
    """Resolve a scope selector to a single TurnRecord.

    Returns None if the scope cannot be resolved (empty history,
    out of range index).
    """
    if not history:
        return None
    if isinstance(scope, LastTurn):
        return history[-1]
    if isinstance(scope, FirstTurn):
        return history[0]
    if isinstance(scope, TurnAt):
        try:
            return history[scope.index]
        except IndexError:
            return None
    # AnyTurn/AllTurns are not valid for cross-turn single resolution
    logger.warning(
        "Scope type %s cannot resolve to a single turn for CrossTurnCheck",
        type(scope).__name__,
    )
    return None


# ---------- CountTurns ----------


def _evaluate_count_turns(
    check: CountTurns,
    result: ScenarioExecutionResult,
) -> int:
    """Count turns matching optional node_id and verify_result filters."""
    return sum(1 for t in result.history if _turn_matches_filter(t, check.node_id, check.verify_result))


def _turn_matches_filter(
    turn: TurnRecord,
    node_id: str | list[str] | None,
    verify_result: bool | None,
) -> bool:
    """Check whether a turn matches the given filters."""
    if node_id is not None:
        if isinstance(node_id, str):
            if turn.node_id != node_id:
                return False
        elif turn.node_id not in node_id:
            return False
    return not (verify_result is not None and turn.verify_result != verify_result)


# ---------- FirstMatchIndex ----------


def _evaluate_first_match_index(
    check: FirstMatchIndex,
    result: ScenarioExecutionResult,
) -> int:
    """Return the index of the first turn matching filters, or -1 if none."""
    for i, turn in enumerate(result.history):
        if _turn_matches_filter(turn, check.node_id, check.verify_result):
            return i
    return -1
