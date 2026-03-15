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
]
