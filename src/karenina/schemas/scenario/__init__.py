"""Scenario schema types for multi-turn benchmarks.

This module contains Pydantic models for defining multi-turn scenario
benchmarks: graph nodes, edges, conditions, outcome criteria, and
declarative check nodes.
"""

from .checks import (
    CountTurns,
    CrossTurnCheck,
    FirstMatchIndex,
    OutcomeCheckNode,
    OutcomeNode,
    ResultCheck,
    TurnCheck,
)
from .definition import ScenarioDefinition
from .state import ScenarioExecutionResult, ScenarioState, TurnRecord
from .types import (
    END,
    EdgeCondition,
    ModelOverride,
    ScenarioEdge,
    ScenarioNode,
    ScenarioOutcomeCriterion,
    StateCheck,
    ToolFilter,
    ToolFilterEntry,
)

# Resolve OutcomeNode forward reference in ScenarioOutcomeCriterion.check
# (types.py cannot import from checks.py without creating a circular import)
ScenarioOutcomeCriterion.model_rebuild(
    force=True,
    _types_namespace={"OutcomeNode": OutcomeNode},
)

__all__ = [
    # Sentinel
    "END",
    # Types
    "StateCheck",
    "EdgeCondition",
    "ModelOverride",
    "ToolFilterEntry",
    "ToolFilter",
    "ScenarioEdge",
    "ScenarioNode",
    "ScenarioOutcomeCriterion",
    # Check nodes
    "TurnCheck",
    "ResultCheck",
    "CrossTurnCheck",
    "CountTurns",
    "FirstMatchIndex",
    # Discriminated unions
    "OutcomeCheckNode",
    "OutcomeNode",
    # Definition
    "ScenarioDefinition",
    # State dataclasses
    "ScenarioExecutionResult",
    "ScenarioState",
    "TurnRecord",
]
