"""Frozen Pydantic schema for validated scenario definitions."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .types import ScenarioEdge, ScenarioNode, ScenarioOutcomeCriterion


class ScenarioDefinition(BaseModel):
    """Frozen, validated scenario definition for serialization and execution.

    Produced by Scenario.build(). Stored in JSON-LD checkpoints.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    description: str = ""
    nodes: dict[str, ScenarioNode]
    edges: list[ScenarioEdge]
    entry_node: str
    outcome_criteria: list[ScenarioOutcomeCriterion] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
