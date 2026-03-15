"""Verification primitives for VerifiedField-based answer templates.

Primitives define how extracted values are compared against ground truth.
Two categories:
- Parsed primitives: operate on judge-extracted values (field included in parsing schema)
- Trace primitives: operate on raw LLM response (field excluded from parsing schema)
"""

from karenina.schemas.primitives.comparisons import (
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
    VerificationPrimitive,
)
from karenina.schemas.primitives.normalizers import (
    Normalizer,
    SynonymMap,
    apply_normalizer,
    apply_normalizers,
)
from karenina.schemas.primitives.trace import (
    TraceContains,
    TraceLength,
    TracePrimitive,
    TraceRegex,
)

__all__ = [
    # Base classes
    "VerificationPrimitive",
    "TracePrimitive",
    # Parsed primitives
    "BooleanMatch",
    "ContainsAll",
    "ContainsAny",
    "DateMatch",
    "DateRange",
    "DateTolerance",
    "ExactMatch",
    "LiteralMatch",
    "NumericExact",
    "NumericRange",
    "NumericTolerance",
    "OrderedMatch",
    "RegexMatch",
    "SemanticMatch",
    "SetContainment",
    # Trace primitives
    "TraceContains",
    "TraceLength",
    "TraceRegex",
    # Normalizers
    "Normalizer",
    "SynonymMap",
    "apply_normalizer",
    "apply_normalizers",
]
