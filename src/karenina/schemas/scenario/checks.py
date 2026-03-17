"""Check node types for declarative scenario outcome evaluation.

Check nodes compose via AllOf/AnyOf/AtLeastN from karenina.schemas.primitives.
Boolean check nodes participate in composition trees. Aggregation types
(CountTurns, FirstMatchIndex) are standalone.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from karenina.schemas.primitives import (
    AllOf,
    AnyOf,
    AtLeastN,
    VerificationPrimitive,
)
from karenina.schemas.primitives.normalizers import Normalizer
from karenina.schemas.primitives.scope import (
    AllTurns,
    AnyTurn,
    FirstTurn,
    LastTurn,
    TurnAt,
)

# Discriminated union for scope fields so deserialization preserves the concrete type
ScopeUnion = Annotated[
    LastTurn | FirstTurn | TurnAt | AnyTurn | AllTurns,
    Field(discriminator="type"),
]


class TurnCheck(BaseModel):
    """Check a field on selected turn(s) from execution history."""

    type: Literal["turn_check"] = "turn_check"
    scope: ScopeUnion
    field: str  # "node_id", "verify_result", "raw_response", "parsed.<x>"
    expected: Any = None
    verify_with: VerificationPrimitive


class ResultCheck(BaseModel):
    """Check an execution-level field (status, turn_count, path, scenario_id)."""

    type: Literal["result_check"] = "result_check"
    field: str  # "status", "turn_count", "path", "scenario_id"
    expected: Any = None
    verify_with: VerificationPrimitive


class CrossTurnCheck(BaseModel):
    """Compare values between two different turns.

    Semantics: target_value <comparison> source_value.
    "contains" means target contains source. "gt" means target > source.
    """

    type: Literal["cross_turn_check"] = "cross_turn_check"
    source_turn: ScopeUnion
    source_field: str
    target_turn: ScopeUnion
    target_field: str
    comparison: Literal["eq", "neq", "contains", "gt", "gte", "lt", "lte"]
    normalize: list[Normalizer] = Field(default_factory=list)


class CountTurns(BaseModel):
    """Count turns matching optional filters. Returns int."""

    type: Literal["count_turns"] = "count_turns"
    node_id: str | list[str] | None = None
    verify_result: bool | None = None


class FirstMatchIndex(BaseModel):
    """Index of first turn matching filters. Returns int (-1 if no match)."""

    type: Literal["first_match_index"] = "first_match_index"
    node_id: str | list[str] | None = None
    verify_result: bool | None = None


# Discriminated unions
OutcomeCheckNode = Annotated[
    TurnCheck | ResultCheck | CrossTurnCheck | AllOf | AnyOf | AtLeastN,
    Field(discriminator="type"),
]
OutcomeNode = OutcomeCheckNode | CountTurns | FirstMatchIndex
