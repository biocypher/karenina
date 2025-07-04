"""
Rubric data models for qualitative evaluation traits.
"""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


TraitKind = Literal["boolean", "score"]


class RubricTrait(BaseModel):
    """
    Single, atomic evaluation trait for qualitative assessment.

    A trait can be either boolean (true/false) or score-based (1-5 scale).
    """

    name: str = Field(..., min_length=1, description="Human readable identifier for the trait")
    description: str | None = Field(None, description="Detailed description shown to user/LLM")
    kind: TraitKind = Field(..., description="Type of trait: 'boolean' or 'score'")
    min_score: int | None = Field(1, description="Lower bound for score traits (default: 1)")
    max_score: int | None = Field(5, description="Upper bound for score traits (default: 5)")

    model_config = ConfigDict(extra="forbid")

    def validate_score(self, value: int | bool) -> bool:
        """Validate that a given score is valid for this trait."""
        if self.kind == "boolean":
            return isinstance(value, bool)
        else:  # self.kind == "score"
            if not isinstance(value, int):
                return False
            min_val = self.min_score or 1
            max_val = self.max_score or 5
            return min_val <= value <= max_val


class Rubric(BaseModel):
    """
    Collection of evaluation traits applied to all question-answer pairs.

    A rubric defines the qualitative criteria used to evaluate LLM responses
    beyond basic correctness checking.
    """

    traits: list[RubricTrait] = Field(default_factory=list, description="List of evaluation traits")

    model_config = ConfigDict(extra="forbid")

    def get_trait_names(self) -> list[str]:
        """Get list of trait names in this rubric."""
        return [trait.name for trait in self.traits]

    def validate_evaluation(self, evaluation: dict[str, int | bool]) -> bool:
        """Validate that an evaluation result matches this rubric structure."""
        trait_names = set(self.get_trait_names())
        eval_names = set(evaluation.keys())

        # Check that all trait names are present
        if trait_names != eval_names:
            return False

        # Check that each score is valid for its trait
        trait_map = {trait.name: trait for trait in self.traits}
        for name, value in evaluation.items():
            if not trait_map[name].validate_score(value):
                return False

        return True


class RubricEvaluation(BaseModel):
    """
    Result of applying a rubric to a specific question-answer pair.
    """

    trait_scores: dict[str, int | bool] = Field(..., description="Scores for each trait")

    model_config = ConfigDict(extra="forbid")


def merge_rubrics(global_rubric: "Rubric | None", question_rubric: "Rubric | None") -> "Rubric | None":
    """
    Merge global and question-specific rubrics.

    Args:
        global_rubric: The global rubric (applied to all questions)
        question_rubric: Question-specific rubric (overrides/adds to global)

    Returns:
        Merged rubric with global traits + question-specific traits, or None if both are None

    Raises:
        ValueError: If there are trait name conflicts between global and question rubrics
    """
    if not global_rubric and not question_rubric:
        return None

    if not global_rubric:
        return question_rubric

    if not question_rubric:
        return global_rubric

    # Check for trait name conflicts
    global_trait_names = {trait.name for trait in global_rubric.traits}
    question_trait_names = {trait.name for trait in question_rubric.traits}
    conflicts = global_trait_names.intersection(question_trait_names)

    if conflicts:
        raise ValueError(f"Trait name conflicts between global and question rubrics: {conflicts}")

    # Merge traits
    merged_traits = list(global_rubric.traits) + list(question_rubric.traits)

    return Rubric(traits=merged_traits)
