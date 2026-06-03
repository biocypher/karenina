"""Verification primitives for VerifiedField-based answer templates.

Primitives define how extracted values are compared against ground truth.
Four categories:
- Parsed primitives: operate on judge-extracted values (field included in parsing schema)
- Trace primitives: operate on raw LLM response (field excluded from parsing schema)
- Composition: generic boolean composition nodes (AllOf, AnyOf, AtLeastN)
- Scope: turn scope selectors for scenario evaluation
"""

from __future__ import annotations

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
    NumericGraded,
    NumericMaximum,
    NumericMinimum,
    NumericRange,
    NumericTolerance,
    OrderedMatch,
    RegexMatch,
    SemanticMatch,
    SetContainment,
    VerificationPrimitive,
)
from karenina.schemas.primitives.composition import (
    AllOf,
    AnyOf,
    AtLeastN,
    evaluate_composition,
)
from karenina.schemas.primitives.normalizers import (
    Normalizer,
    SynonymMap,
    apply_normalizer,
    apply_normalizers,
)
from karenina.schemas.primitives.scope import (
    AllTurns,
    AnyTurn,
    FirstTurn,
    LastTurn,
    TurnAt,
    TurnScope,
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
    "NumericGraded",
    "NumericMaximum",
    "NumericMinimum",
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
    # Composition
    "AllOf",
    "AnyOf",
    "AtLeastN",
    "evaluate_composition",
    # Scope selectors
    "TurnScope",
    "LastTurn",
    "FirstTurn",
    "TurnAt",
    "AnyTurn",
    "AllTurns",
    # Normalizers
    "Normalizer",
    "SynonymMap",
    "apply_normalizer",
    "apply_normalizers",
]
