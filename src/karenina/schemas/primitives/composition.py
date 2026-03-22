"""Generic composition nodes for combining boolean results.

AllOf, AnyOf, and AtLeastN form a recursive tree. The generic
``evaluate_composition()`` walks the tree, delegating leaf evaluation
to a caller-provided callback. Domain-specific wrappers (e.g.,
``evaluate_strategy`` in entities/composition) supply the leaf logic.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AllOf(BaseModel):
    """All child conditions must pass."""

    type: Literal["all_of"] = "all_of"
    conditions: list[Any] = []


class AnyOf(BaseModel):
    """At least one child condition must pass."""

    type: Literal["any_of"] = "any_of"
    conditions: list[Any] = []


class AtLeastN(BaseModel):
    """N or more child conditions must pass."""

    type: Literal["at_least_n"] = "at_least_n"
    n: int = Field(ge=0)
    conditions: list[Any] = []


def evaluate_composition(
    node: Any,
    leaf_evaluator: Callable[[Any], bool],
) -> bool:
    """Walk a composition tree, delegating leaves to the caller.

    Args:
        node: A composition node (AllOf, AnyOf, AtLeastN) or a leaf.
        leaf_evaluator: Callable that receives a leaf node and returns bool.

    Returns:
        True if the composition passes.
    """
    if isinstance(node, AllOf):
        if not node.conditions:
            return True
        return all(evaluate_composition(c, leaf_evaluator) for c in node.conditions)
    if isinstance(node, AnyOf):
        if not node.conditions:
            return False
        return any(evaluate_composition(c, leaf_evaluator) for c in node.conditions)
    if isinstance(node, AtLeastN):
        passing = sum(1 for c in node.conditions if evaluate_composition(c, leaf_evaluator))
        return passing >= node.n

    # Leaf node: delegate to caller
    return leaf_evaluator(node)
