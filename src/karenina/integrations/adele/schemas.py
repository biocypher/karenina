"""
Pydantic schemas for ADeLe (Annotated Demand Levels) question classification.

ADeLe rubrics from Zhou et al. (2025), arXiv:2503.06378.
https://kinds-of-intelligence-cfi.github.io/ADELE/
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QuestionClassificationResult(BaseModel):
    """Result of classifying a single question against ADeLe dimensions."""

    question_id: str | None = Field(
        default=None,
        description="Unique identifier for the question",
    )
    question_text: str = Field(
        description="The question text that was classified",
    )
    scores: dict[str, int] = Field(
        default_factory=dict,
        description=("Mapping from trait name to integer score (0-5). -1 indicates classification error."),
    )
    labels: dict[str, str] = Field(
        default_factory=dict,
        description=("Mapping from trait name to class label (none, very_low, low, intermediate, high, very_high)"),
    )
    model: str = Field(
        default="unknown",
        description="The model used for classification",
    )
    classified_at: str = Field(
        default="",
        description="ISO timestamp of when classification was performed",
    )
    usage_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Token usage and other metadata from the LLM call",
    )

    def to_checkpoint_metadata(self) -> dict[str, Any]:
        """
        Convert to format suitable for storing in checkpoint custom_metadata.

        Returns:
            Dictionary with adele_classification key containing scores, labels,
            timestamp, and model.
        """
        return {
            "adele_classification": {
                "scores": self.scores,
                "labels": self.labels,
                "classified_at": self.classified_at,
                "model": self.model,
            }
        }

    @classmethod
    def from_checkpoint_metadata(
        cls,
        metadata: dict[str, Any],
        question_id: str | None = None,
        question_text: str = "",
    ) -> QuestionClassificationResult | None:
        """
        Create from checkpoint custom_metadata format.

        Args:
            metadata: The custom_metadata dict from a question
            question_id: Optional question ID
            question_text: Optional question text

        Returns:
            QuestionClassificationResult if adele_classification exists, else None
        """
        adele_data = metadata.get("adele_classification")
        if adele_data is None:
            return None

        return cls(
            question_id=question_id,
            question_text=question_text,
            scores=adele_data.get("scores", {}),
            labels=adele_data.get("labels", {}),
            model=adele_data.get("model", "unknown"),
            classified_at=adele_data.get("classified_at", ""),
            usage_metadata={},  # Not stored in checkpoint
        )

    def get_summary(self) -> dict[str, str]:
        """
        Get a summary of classifications as trait -> "label (score)" pairs.

        Returns:
            Dictionary mapping trait names to "label (score)" strings.
        """
        summary: dict[str, str] = {}
        for trait_name, score in self.scores.items():
            label = self.labels.get(trait_name, "unknown")
            if score == -1:
                summary[trait_name] = f"error: {label}"
            else:
                summary[trait_name] = f"{label} ({score})"
        return summary


class AdeleTraitInfo(BaseModel):
    """Information about an ADeLe trait for API responses."""

    name: str = Field(description="Snake_case trait name")
    code: str = Field(description="Original ADeLe code (e.g., 'AS', 'AT')")
    description: str | None = Field(
        default=None,
        description="Trait description/header from the rubric",
    )
    classes: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping from class name to class description",
    )
    class_names: list[str] = Field(
        default_factory=list,
        description="Ordered list of class names (from level 0 to 5)",
    )
