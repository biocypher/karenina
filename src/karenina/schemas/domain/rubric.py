"""Rubric data models for qualitative evaluation traits.

DEPRECATED: This module is deprecated. Import from ``schemas.entities.rubric`` instead.

This module re-exports from ``schemas.entities.rubric`` for backward compatibility.
"""

import warnings

from ..entities.rubric import (
    METRIC_REQUIREMENTS,
    VALID_METRICS,
    VALID_METRICS_FULL_MATRIX,
    VALID_METRICS_TP_ONLY,
    CallableTrait,
    LLMRubricTrait,
    MetricRubricTrait,
    RegexTrait,
    Rubric,
    RubricEvaluation,
    TraitKind,
    merge_rubrics,
)

__all__ = [
    "CallableTrait",
    "LLMRubricTrait",
    "METRIC_REQUIREMENTS",
    "MetricRubricTrait",
    "RegexTrait",
    "Rubric",
    "RubricEvaluation",
    "TraitKind",
    "VALID_METRICS",
    "VALID_METRICS_FULL_MATRIX",
    "VALID_METRICS_TP_ONLY",
    "merge_rubrics",
]


def __getattr__(name: str) -> object:
    """Emit deprecation warning when accessing this module."""
    if name in __all__:
        warnings.warn(
            f"Importing {name} from 'karenina.schemas.domain.rubric' is deprecated. "
            f"Use 'karenina.schemas.entities.rubric' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
