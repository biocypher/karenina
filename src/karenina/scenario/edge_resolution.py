"""Edge condition evaluation and next-node resolution."""

from __future__ import annotations

import logging
from typing import Any

from karenina.schemas.scenario.state import ScenarioState
from karenina.schemas.scenario.types import ScenarioEdge, StateCheck

logger = logging.getLogger(__name__)


def resolve_next_node(edges: list[ScenarioEdge], state: ScenarioState) -> str | None:
    """Resolve the next node given outbound edges and current state.

    Conditional edges are evaluated in definition order; first match wins.
    Unconditional edges serve as fallback regardless of position.
    Returns None if no edges match (implicit terminal).
    """
    if not edges:
        return None

    fallback_target: str | None = None

    for edge in edges:
        is_unconditional = edge.condition is None and edge.condition_callable is None
        if is_unconditional:
            if fallback_target is None:
                fallback_target = edge.target
            continue

        if _edge_matches(edge, state):
            return edge.target

    return fallback_target


def _edge_matches(edge: ScenarioEdge, state: ScenarioState) -> bool:
    """Check whether an edge's condition matches the current state."""
    if edge.condition_callable is not None:
        try:
            return bool(edge.condition_callable(state))
        except Exception:
            logger.warning(
                "Callable condition on edge %s->%s raised",
                edge.source,
                edge.target,
                exc_info=True,
            )
            return False

    condition = edge.condition
    if condition is None:
        return True

    if isinstance(condition, list):
        return all(evaluate_state_check(c, state) for c in condition)

    return evaluate_state_check(condition, state)


def evaluate_state_check(check: StateCheck, state: ScenarioState) -> bool:
    """Evaluate a single StateCheck against state using its comparison primitive.

    Resolves the dot-path field from state, then delegates to
    ``check.verify_with.check(extracted=resolved_value, expected=check.expected)``.
    """
    resolved = _resolve_dot_path(check.field, state)
    return check.verify_with.check(resolved, check.expected)


def _resolve_dot_path(path: str, state: ScenarioState) -> Any:
    """Resolve a dot-path against ScenarioState.

    Supported paths:
        "verify_result": state.verify_result
        "turn": state.turn
        "current_node": state.current_node
        "parsed.<field>": state.parsed[field]
        "accumulated.<field>": state.accumulated[field]
        "node_visits.<node>": state.node_visits[node] (0 if missing)
        "node_results.<node>": state.node_results[node]
        "node_results.<node>.verify_result": state.node_results[node]["verify_result"]
        "node_results.<node>.parsed.<field>": state.node_results[node]["parsed"][field]
        "node_results.<node>.rubric.<trait>": state.node_results[node]["rubric"][trait]

    Returns None for missing keys (except node_visits, which returns 0).
    """
    parts = path.split(".", 1)
    root = parts[0]

    if root == "verify_result":
        return state.verify_result
    if root == "turn":
        return state.turn
    if root == "current_node":
        return state.current_node

    if len(parts) == 2:
        sub_key = parts[1]
        if root == "parsed":
            return state.parsed.get(sub_key)
        if root == "accumulated":
            return state.accumulated.get(sub_key)
        if root == "node_visits":
            return state.node_visits.get(sub_key, 0)
        if root == "node_results":
            return _resolve_node_results_path(sub_key, state)

    return None


def _resolve_node_results_path(sub_key: str, state: ScenarioState) -> Any:
    """Resolve a path within node_results.

    Args:
        sub_key: The path after "node_results.", e.g. "ask.parsed.drug_name".
        state: The current scenario state.

    Returns:
        The resolved value, or None if the path does not exist.
    """
    sub_parts = sub_key.split(".", 1)
    node_key = sub_parts[0]
    node_data = state.node_results.get(node_key, {})

    if len(sub_parts) == 1:
        return node_data

    remaining = sub_parts[1]
    remaining_parts = remaining.split(".", 1)
    value = node_data.get(remaining_parts[0])

    if len(remaining_parts) == 2 and isinstance(value, dict):
        return value.get(remaining_parts[1])

    return value
