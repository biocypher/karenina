"""Pydantic models for structured rubric evaluation outputs.

These models are used with LangChain's structured output features
(ProviderStrategy and ToolStrategy) to ensure reliable parsing of
LLM responses during rubric evaluation.
"""

from typing import Literal

from pydantic import BaseModel, Field


class BatchRubricScores(BaseModel):
    """Structured output for batch trait evaluation.

    Used when evaluating multiple rubric traits in a single LLM call.
    The LLM returns all trait scores in one response.
    """

    scores: dict[str, int | bool] = Field(
        description="Dictionary mapping trait names to scores (int for score traits, bool for binary)"
    )


class SingleBooleanScore(BaseModel):
    """Structured output for boolean trait evaluation.

    Used for single-trait evaluation where the trait is binary (pass/fail).
    """

    result: bool = Field(description="True if the criteria is met, False otherwise")


class SingleNumericScore(BaseModel):
    """Structured output for numeric trait evaluation.

    Used for single-trait evaluation where the trait is a score
    within a defined range (e.g., 1-5).
    """

    score: int = Field(description="Numeric score within the valid range")


class ConfusionMatrixOutput(BaseModel):
    """Structured output for metric trait evaluation.

    Used for confusion-matrix based evaluation where the LLM
    categorizes content into true positives, false negatives,
    false positives, and true negatives.
    """

    tp: list[str] = Field(
        default_factory=list,
        description="True positives - content that SHOULD be present AND IS present (excerpts from answer)",
    )
    fn: list[str] = Field(
        default_factory=list,
        description="False negatives - content that SHOULD be present BUT IS NOT (reference instruction content)",
    )
    fp: list[str] = Field(
        default_factory=list,
        description="False positives - content that should NOT be present BUT IS (excerpts from answer)",
    )
    tn: list[str] = Field(
        default_factory=list,
        description="True negatives - content that should NOT be present AND IS correctly absent (reference instruction content)",
    )


# ========== Deep Judgment Rubric Models ==========


class TraitExcerpt(BaseModel):
    """Single excerpt for trait evaluation in Deep Judgment Rubric."""

    text: str = Field(description="Exact verbatim quote from the answer")
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence level: high=strong evidence, medium=moderate evidence, low=weak/ambiguous evidence"
    )


class TraitExcerptsOutput(BaseModel):
    """Structured output for excerpt extraction in Deep Judgment Rubric."""

    excerpts: list[TraitExcerpt] = Field(
        default_factory=list,
        description="List of verbatim excerpts demonstrating the trait",
    )


class HallucinationRiskOutput(BaseModel):
    """Structured output for hallucination risk assessment in Deep Judgment Rubric."""

    risk: Literal["none", "low", "medium", "high"] = Field(
        description="Hallucination risk level: none=strong external support, low=some support, medium=weak/ambiguous, high=no support or contradicted"
    )
    justification: str = Field(description="Brief explanation for the risk assessment")
