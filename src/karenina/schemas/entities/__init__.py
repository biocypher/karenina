"""Core business entity models.

This module contains the fundamental entities used throughout Karenina:
- BaseAnswer: Base class for answer templates
- Question: Benchmark question definition
- Rubric: Rubric traits for qualitative evaluation
- VerifiedField: Declarative field verification for answer templates
- Primitives: Verification primitives (ExactMatch, BooleanMatch, etc.)
- Composition: Strategy nodes for combining field results (AllOf, AnyOf, etc.)
"""

from karenina.schemas.primitives import (
    BooleanMatch,
    ContainsAll,
    ContainsAny,
    DateMatch,
    DateRange,
    DateTolerance,
    ExactMatch,
    LiteralMatch,
    Normalizer,
    NumericExact,
    NumericMaximum,
    NumericMinimum,
    NumericRange,
    NumericTolerance,
    OrderedMatch,
    RegexMatch,
    SemanticMatch,
    SetContainment,
    SynonymMap,
    TraceContains,
    TraceLength,
    TracePrimitive,
    TraceRegex,
    VerificationPrimitive,
    apply_normalizer,
    apply_normalizers,
)

from .answer import BaseAnswer, capture_answer_source
from .composition import AllOf, AnyOf, AtLeastN, FieldCheck, evaluate_strategy
from .conditional import ConditionalGroundTruth, GroundTruthCase
from .question import Question, QuestionRegistryEntry
from .rubric import (
    AgenticRubricTrait,
    CallableRubricTrait,
    DynamicRubric,
    LLMRubricTrait,
    MetricRubricTrait,
    RegexRubricTrait,
    Rubric,
    RubricEvaluation,
    TraitKind,
    merge_dynamic_rubrics,
    merge_rubrics,
)
from .template_spec import TemplateFieldSpec, TemplateSpec, VerifyStrategySpec
from .verified_field import VerificationMeta, VerifiedField

__all__ = [
    # Answer templates
    "BaseAnswer",
    "capture_answer_source",
    # VerifiedField system
    "VerifiedField",
    "VerificationMeta",
    # Conditional ground truth
    "ConditionalGroundTruth",
    "GroundTruthCase",
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
    "NumericMaximum",
    "NumericMinimum",
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
    # TemplateSpec (GUI interchange)
    "TemplateFieldSpec",
    "TemplateSpec",
    "VerifyStrategySpec",
    # Questions
    "Question",
    "QuestionRegistryEntry",
    # Rubrics
    "Rubric",
    "DynamicRubric",
    "LLMRubricTrait",
    "RegexRubricTrait",
    "CallableRubricTrait",
    "AgenticRubricTrait",
    "MetricRubricTrait",
    "RubricEvaluation",
    "TraitKind",
    "merge_rubrics",
    "merge_dynamic_rubrics",
]
