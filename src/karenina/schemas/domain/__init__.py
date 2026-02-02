"""Domain models for core business entities.

DEPRECATED: This module is deprecated. Import from `schemas.entities` instead.

This module re-exports from `schemas.entities` for backward compatibility.
"""

import warnings

# Re-export from entities for backward compatibility
from ..entities import (
    BaseAnswer,
    CallableTrait,
    LLMRubricTrait,
    MetricRubricTrait,
    Question,
    RegexTrait,
    Rubric,
    RubricEvaluation,
    TraitKind,
    capture_answer_source,
    merge_rubrics,
)

__all__ = [
    "BaseAnswer",
    "capture_answer_source",
    "Question",
    "Rubric",
    "LLMRubricTrait",
    "RegexTrait",
    "CallableTrait",
    "MetricRubricTrait",
    "RubricEvaluation",
    "TraitKind",
    "merge_rubrics",
]


def __getattr__(name: str) -> object:
    """Emit deprecation warning when accessing this module."""
    if name in __all__:
        warnings.warn(
            f"Importing {name} from 'karenina.schemas.domain' is deprecated. Use 'karenina.schemas.entities' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
