"""Core business entity models.

This module contains the fundamental entities used throughout Karenina:
- BaseAnswer: Base class for answer templates
- Question: Benchmark question definition
- Rubric: Rubric traits for qualitative evaluation
"""

from .answer import BaseAnswer, capture_answer_source
from .question import Question
from .rubric import (
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
