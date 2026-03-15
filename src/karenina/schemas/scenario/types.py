"""Pydantic schema types for multi-turn scenario benchmarks."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from karenina.schemas.config import ModelConfig
from karenina.schemas.entities import Question
from karenina.schemas.primitives import VerificationPrimitive

if TYPE_CHECKING:
    from karenina.schemas.scenario.checks import OutcomeNode

# Sentinel for scenario termination
END: str = "__end__"


class StateCheck(BaseModel):
    """Check a ScenarioState field using a comparison primitive.

    The field uses dot-path syntax to resolve against state attributes:
    "verify_result", "parsed.<field>", "accumulated.<field>",
    "node_visits.<node>", "node_results.<node>.verify_result",
    "node_results.<node>.parsed.<field>",
    "node_results.<node>.rubric.<trait>", "turn".
    """

    type: Literal["state_check"] = "state_check"
    field: str
    expected: Any = None
    verify_with: VerificationPrimitive


# Type alias for edge conditions
EdgeCondition = StateCheck | list[StateCheck]


class ModelOverride(BaseModel):
    """Per-node override for answering and/or parsing models."""

    answering_model: ModelConfig | None = None
    parsing_model: ModelConfig | None = None


class ToolFilterEntry(BaseModel):
    """A single tool to remove from the answering model's tool set."""

    server: str
    tool: str | None = None


class ToolFilter(BaseModel):
    """Per-node filter removing MCP tools from the answering model."""

    remove: list[ToolFilterEntry]


class ScenarioEdge(BaseModel):
    """A conditional transition between scenario nodes."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: str
    target: str  # node_id or END
    condition: EdgeCondition | None = None

    # Callable condition (excluded from serialization)
    condition_callable: Callable[..., Any] | None = Field(default=None, exclude=True)
    condition_source: str | None = None


class ScenarioNode(BaseModel):
    """A unique execution point in a scenario graph."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    node_id: str
    question: Question
    model_override: ModelOverride | None = None
    tool_filter: ToolFilter | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Callable field (excluded from serialization)
    state_update: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None = Field(default=None, exclude=True)
    state_update_source: str | None = None


class ScenarioOutcomeCriterion(BaseModel):
    """Evaluates the entire scenario execution after all turns complete.

    Primary path: declarative ``check`` field (fully serializable).
    Escape hatch: ``evaluate`` callable for complex cross-turn logic.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str

    # Primary: declarative check (fully serializable via Pydantic).
    # OutcomeNode is a forward reference resolved by model_rebuild() in __init__.py.
    check: OutcomeNode | None = None

    # Escape hatch: callable (excluded from serialization)
    evaluate: Callable[..., Any] | None = Field(default=None, exclude=True)
    evaluate_source: str | None = None
