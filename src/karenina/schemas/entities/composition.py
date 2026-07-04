"""Composition strategy nodes for combining field verification results.

The composition tree defines how individual field pass/fail results
combine into the template-level verify() result. Common patterns:

- AllOf: all fields must pass (default when no strategy specified)
- AnyOf: at least one field must pass
- AtLeastN: N or more fields must pass
- FieldCheck: leaf node referencing a single field's result

Generic composition logic (evaluate_composition) lives in
``schemas.primitives.composition``. This module adds the
template-domain leaf (FieldCheck), the discriminated StrategyNode union,
and the ``evaluate_strategy()`` wrapper.
"""

from __future__ import annotations

import logging
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from karenina.schemas.primitives.composition import AllOf as _GenericAllOf
from karenina.schemas.primitives.composition import AnyOf as _GenericAnyOf
from karenina.schemas.primitives.composition import AtLeastN as _GenericAtLeastN
from karenina.schemas.primitives.composition import evaluate_composition

logger = logging.getLogger(__name__)


class FieldCheck(BaseModel):
    """Leaf node: references a single field's pass/fail result."""

    type: Literal["field_check"] = "field_check"
    field: str


class AllOf(_GenericAllOf):
    """All child conditions must pass (template domain).

    Overrides conditions with the discriminated StrategyNode union
    so nested trees deserialize correctly.
    """

    conditions: list[StrategyNode] = []


class AnyOf(_GenericAnyOf):
    """At least one child condition must pass (template domain).

    Overrides conditions with the discriminated StrategyNode union
    so nested trees deserialize correctly.
    """

    conditions: list[StrategyNode] = []


class AtLeastN(_GenericAtLeastN):
    """N or more child conditions must pass (template domain).

    Overrides conditions with the discriminated StrategyNode union
    so nested trees deserialize correctly.
    """

    conditions: list[StrategyNode] = []


StrategyNode = Annotated[
    FieldCheck | AllOf | AnyOf | AtLeastN,
    Field(discriminator="type"),
]

# Update forward references for recursive models
AllOf.model_rebuild()
AnyOf.model_rebuild()
AtLeastN.model_rebuild()


def evaluate_strategy(node: StrategyNode, field_results: dict[str, bool | None]) -> bool:
    """Evaluate a composition strategy tree against field results.

    Thin wrapper around ``evaluate_composition()`` that supplies the
    FieldCheck leaf evaluator for the template domain.

    The leaf evaluator returns True only when ``field_results[leaf.field]``
    is exactly ``True``. ``None`` (null extraction) is treated as
    not-satisfied at composition time, equivalent to soft-False, so
    AllOf / AnyOf / AtLeastN behave conservatively when a sub-field is
    unanswered. Distinguishing None from False is preserved in
    ``field_results`` for downstream consumers (granular scoring,
    serialized result rows).

    Args:
        node: The root node of the strategy tree.
        field_results: Mapping of field names to their tri-valued result
            (True / False / None).

    Returns:
        True if the strategy passes.

    Raises:
        KeyError: If a FieldCheck references a field not in field_results.
    """

    def _field_check_evaluator(leaf: FieldCheck) -> bool:
        return field_results[leaf.field] is True

    return evaluate_composition(node, _field_check_evaluator)
