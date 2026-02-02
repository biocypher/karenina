"""Pydantic models for structured rubric evaluation outputs.

DEPRECATED: Import from `karenina.schemas.outputs` instead.
"""

# Re-export from new location for backward compatibility
from ..outputs.rubric import (
    BatchLiteralClassifications,
    BatchRubricScores,
    ConfusionMatrixOutput,
    HallucinationRiskOutput,
    SingleBooleanScore,
    SingleLiteralClassification,
    SingleNumericScore,
    TraitExcerpt,
    TraitExcerptsOutput,
)

__all__ = [
    "BatchRubricScores",
    "SingleBooleanScore",
    "SingleNumericScore",
    "ConfusionMatrixOutput",
    "TraitExcerpt",
    "TraitExcerptsOutput",
    "HallucinationRiskOutput",
    "SingleLiteralClassification",
    "BatchLiteralClassifications",
]
