"""Multi-turn scenario benchmarks for karenina."""

from karenina.scenario.builder import Scenario
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
from karenina.schemas.scenario.definition import ScenarioDefinition
from karenina.schemas.scenario.types import (
    END,
    ModelOverride,
    ScenarioOutcomeCriterion,
)

__all__ = [
    # Builder
    "Scenario",
    # Frozen definition
    "ScenarioDefinition",
    # Types
    "END",
    "ModelOverride",
    "ScenarioOutcomeCriterion",
    # Sugar functions
    "all_of",
    "all_turns",
    "any_of",
    "any_turn",
    "at_least_n",
    "count_turns",
    "cross_turn",
    "first_match_index",
    "first_turn",
    "first_turn_scope",
    "last_turn",
    "last_turn_scope",
    "status_is",
    "turn_at",
    "turn_count_eq",
    "turn_count_gte",
]
