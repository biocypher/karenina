"""Structured output models for LLM responses.

This module contains Pydantic models that define the expected structure
of LLM outputs during rubric evaluation:
- Rubric scoring outputs (boolean, numeric, literal)
- Confusion matrix outputs for metric traits
- Deep judgment excerpt outputs
"""

from .rubric import (
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
    # Rubric scoring outputs
    "BatchRubricScores",
    "SingleBooleanScore",
    "SingleNumericScore",
    "ConfusionMatrixOutput",
    # Deep Judgment Rubric outputs
    "TraitExcerpt",
    "TraitExcerptsOutput",
    "HallucinationRiskOutput",
    # Literal trait classification outputs
    "SingleLiteralClassification",
    "BatchLiteralClassifications",
]
