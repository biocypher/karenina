"""Core business entity models.

This module contains the fundamental entities used throughout Karenina:
- BaseAnswer: Base class for answer templates
- Question: Benchmark question definition
- Rubric: Rubric traits for qualitative evaluation
- VerifiedField: Declarative field verification for answer templates
- Primitives: Verification primitives (ExactMatch, BooleanMatch, etc.)
- Composition: Strategy nodes for combining field results (AllOf, AnyOf, etc.)
"""

from .answer import BaseAnswer, capture_answer_source
from .composition import AllOf, AnyOf, AtLeastN, FieldCheck, evaluate_strategy
from .normalizers import Normalizer, SynonymMap, apply_normalizer, apply_normalizers
from .primitives import (
    BooleanMatch,
    ContainsAll,
    ContainsAny,
    DateMatch,
    DateRange,
    DateTolerance,
    ExactMatch,
    LiteralMatch,
    NumericExact,
    NumericRange,
    NumericTolerance,
    OrderedMatch,
    RegexMatch,
    SemanticMatch,
    SetContainment,
    TraceContains,
    TraceLength,
    TracePrimitive,
    TraceRegex,
    VerificationPrimitive,
)
from .question import Question, QuestionRegistryEntry
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
from .verified_field import VerificationMeta, VerifiedField

__all__ = [
    # Answer templates
    "BaseAnswer",
    "capture_answer_source",
    # VerifiedField system
    "VerifiedField",
    "VerificationMeta",
    # Primitives
    "VerificationPrimitive",
    "TracePrimitive",
    "BooleanMatch",
    "ExactMatch",
    "ContainsAny",
    "ContainsAll",
    "RegexMatch",
    "SemanticMatch",
    "NumericExact",
    "NumericTolerance",
    "NumericRange",
    "SetContainment",
    "OrderedMatch",
    "LiteralMatch",
    "DateMatch",
    "DateTolerance",
    "DateRange",
    "TraceRegex",
    "TraceContains",
    "TraceLength",
    # Composition
    "AllOf",
    "AnyOf",
    "AtLeastN",
    "FieldCheck",
    "evaluate_strategy",
    # Normalizers
    "Normalizer",
    "SynonymMap",
    "apply_normalizer",
    "apply_normalizers",
    # Questions
    "Question",
    "QuestionRegistryEntry",
    # Rubrics
    "Rubric",
    "LLMRubricTrait",
    "RegexTrait",
    "CallableTrait",
    "MetricRubricTrait",
    "RubricEvaluation",
    "TraitKind",
    "merge_rubrics",
]
