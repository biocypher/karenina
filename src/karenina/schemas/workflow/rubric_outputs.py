"""Pydantic models for structured rubric evaluation outputs.

These models are used with LangChain's structured output features
(ProviderStrategy and ToolStrategy) to ensure reliable parsing of
LLM responses during rubric evaluation.
"""

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
