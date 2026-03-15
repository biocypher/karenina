"""Builder sugar functions for declarative scenario outcome criteria.

Convenience wrappers that produce TurnCheck, ResultCheck, CrossTurnCheck,
CountTurns, FirstMatchIndex, and composition nodes with minimal boilerplate.
"""

from __future__ import annotations

from typing import Any, Literal

from karenina.schemas.primitives import (
    BooleanMatch,
    ExactMatch,
    NumericExact,
    NumericRange,
)
from karenina.schemas.primitives.composition import AllOf, AnyOf, AtLeastN
from karenina.schemas.primitives.normalizers import Normalizer
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
    ScopeUnion,
    TurnCheck,
)


def _infer_primitive(value: Any) -> Any:
    """Infer the verification primitive from the value type.

    Args:
        value: The expected value.

    Returns:
        An appropriate VerificationPrimitive instance.
    """
    if isinstance(value, bool):
        return BooleanMatch()
    if isinstance(value, str):
        return ExactMatch()
    if isinstance(value, int | float):
        return NumericExact()
    return ExactMatch()


def _make_turn_checks(scope: ScopeUnion, **fields: Any) -> TurnCheck | AllOf:
    """Build TurnCheck(s) from keyword fields.

    Single kwarg produces a bare TurnCheck. Multiple kwargs produce
    an AllOf wrapping one TurnCheck per field.
    """
    checks = []
    for field, value in fields.items():
        checks.append(
            TurnCheck(
                scope=scope,
                field=field,
                expected=value,
                verify_with=_infer_primitive(value),
            )
        )
    if len(checks) == 1:
        return checks[0]
    return AllOf(conditions=checks)


# ---------------------------------------------------------------------------
# Turn sugar
# ---------------------------------------------------------------------------


def last_turn(**fields: Any) -> TurnCheck | AllOf:
    """TurnCheck on the last turn. Multiple kwargs produce AllOf of TurnChecks."""
    return _make_turn_checks(LastTurn(), **fields)


def first_turn(**fields: Any) -> TurnCheck | AllOf:
    """TurnCheck on the first turn. Multiple kwargs produce AllOf of TurnChecks."""
    return _make_turn_checks(FirstTurn(), **fields)


def any_turn(*, node: str | None = None, **fields: Any) -> TurnCheck | AllOf:
    """TurnCheck quantified over any matching turn.

    Args:
        node: Optional node_id filter.
        **fields: Field name to expected value mappings.
    """
    return _make_turn_checks(AnyTurn(node_id=node), **fields)


def all_turns(*, node: str | None = None, **fields: Any) -> TurnCheck | AllOf:
    """TurnCheck quantified over all matching turns.

    Args:
        node: Optional node_id filter.
        **fields: Field name to expected value mappings.
    """
    return _make_turn_checks(AllTurns(node_id=node), **fields)


# ---------------------------------------------------------------------------
# Result sugar
# ---------------------------------------------------------------------------


def status_is(expected: str) -> ResultCheck:
    """ResultCheck for execution status."""
    return ResultCheck(field="status", expected=expected, verify_with=ExactMatch())


def turn_count_gte(n: int) -> ResultCheck:
    """ResultCheck: turn_count >= n."""
    return ResultCheck(field="turn_count", verify_with=NumericRange(min=n))


def turn_count_eq(n: int) -> ResultCheck:
    """ResultCheck: turn_count == n."""
    return ResultCheck(field="turn_count", expected=n, verify_with=NumericExact())


# ---------------------------------------------------------------------------
# Aggregation sugar
# ---------------------------------------------------------------------------


def count_turns(*, node: str | None = None, verify_result: bool | None = None) -> CountTurns:
    """CountTurns aggregation.

    Args:
        node: Optional node_id filter.
        verify_result: Optional verify_result filter.
    """
    return CountTurns(node_id=node, verify_result=verify_result)


def first_match_index(*, node: str | None = None, verify_result: bool | None = None) -> FirstMatchIndex:
    """FirstMatchIndex aggregation.

    Args:
        node: Optional node_id filter.
        verify_result: Optional verify_result filter.
    """
    return FirstMatchIndex(node_id=node, verify_result=verify_result)


# ---------------------------------------------------------------------------
# Cross-turn sugar
# ---------------------------------------------------------------------------


def cross_turn(
    *,
    source: ScopeUnion,
    source_field: str,
    target: ScopeUnion,
    target_field: str,
    comparison: Literal["eq", "neq", "contains", "gt", "gte", "lt", "lte"],
    normalize: list[Normalizer] | None = None,
) -> CrossTurnCheck:
    """CrossTurnCheck between two turns.

    Args:
        source: Scope selector for the source turn.
        source_field: Dot-path field on the source turn.
        target: Scope selector for the target turn.
        target_field: Dot-path field on the target turn.
        comparison: Comparison operator (eq, neq, contains, gt, gte, lt, lte).
        normalize: Optional list of normalizers to apply before comparison.
    """
    return CrossTurnCheck(
        source_turn=source,
        source_field=source_field,
        target_turn=target,
        target_field=target_field,
        comparison=comparison,
        normalize=normalize or [],
    )


# ---------------------------------------------------------------------------
# Scope helpers
# ---------------------------------------------------------------------------


def first_turn_scope() -> FirstTurn:
    """Create a FirstTurn scope selector."""
    return FirstTurn()


def last_turn_scope() -> LastTurn:
    """Create a LastTurn scope selector."""
    return LastTurn()


def turn_at(index: int) -> TurnAt:
    """Create a TurnAt scope selector.

    Args:
        index: Turn index (supports negative indexing).
    """
    return TurnAt(index=index)


# ---------------------------------------------------------------------------
# Composition helpers
# ---------------------------------------------------------------------------


def all_of(*checks: Any) -> AllOf:
    """Combine checks with AllOf (all must pass)."""
    return AllOf(conditions=list(checks))


def any_of(*checks: Any) -> AnyOf:
    """Combine checks with AnyOf (at least one must pass)."""
    return AnyOf(conditions=list(checks))


def at_least_n(n: int, *checks: Any) -> AtLeastN:
    """Combine checks with AtLeastN (n or more must pass).

    Args:
        n: Minimum number of checks that must pass.
        *checks: Check nodes to evaluate.
    """
    return AtLeastN(n=n, conditions=list(checks))
