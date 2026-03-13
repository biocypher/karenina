"""Composition strategy nodes for combining field verification results.

The composition tree defines how individual field pass/fail results
combine into the template-level verify() result. Common patterns:

- AllOf: all fields must pass (default when no strategy specified)
- AnyOf: at least one field must pass
- AtLeastN: N or more fields must pass
- FieldCheck: leaf node referencing a single field's result
"""

from __future__ import annotations

import logging
from typing import Annotated, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FieldCheck(BaseModel):
    """Leaf node: references a single field's pass/fail result."""

    type: Literal["field_check"] = "field_check"
    field: str


class AllOf(BaseModel):
    """All child conditions must pass."""

    type: Literal["all_of"] = "all_of"
    conditions: list[StrategyNode] = []


class AnyOf(BaseModel):
    """At least one child condition must pass."""

    type: Literal["any_of"] = "any_of"
    conditions: list[StrategyNode] = []


class AtLeastN(BaseModel):
    """N or more child conditions must pass."""

    type: Literal["at_least_n"] = "at_least_n"
    n: int
    conditions: list[StrategyNode] = []


StrategyNode = Annotated[
    FieldCheck | AllOf | AnyOf | AtLeastN,
    Field(discriminator="type"),
]

# Update forward references for recursive models
AllOf.model_rebuild()
AnyOf.model_rebuild()
AtLeastN.model_rebuild()


def evaluate_strategy(node: StrategyNode, field_results: dict[str, bool]) -> bool:
    """Evaluate a composition strategy tree against field results.

    Args:
        node: The root node of the strategy tree.
        field_results: Mapping of field names to their pass/fail results.

    Returns:
        True if the strategy passes.

    Raises:
        KeyError: If a FieldCheck references a field not in field_results.
    """
    if isinstance(node, FieldCheck):
        return field_results[node.field]
    if isinstance(node, AllOf):
        if not node.conditions:
            return True
        return all(evaluate_strategy(c, field_results) for c in node.conditions)
    if isinstance(node, AnyOf):
        if not node.conditions:
            return False
        return any(evaluate_strategy(c, field_results) for c in node.conditions)
    if isinstance(node, AtLeastN):
        passing = sum(1 for c in node.conditions if evaluate_strategy(c, field_results))
        return passing >= node.n

    raise TypeError(f"Unknown strategy node type: {type(node)}")
