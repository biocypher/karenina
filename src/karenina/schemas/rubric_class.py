"""
Rubric data models for qualitative evaluation traits.
"""

import re
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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


class ManualRubricTrait(BaseModel):
    """
    Manual evaluation trait that uses callable functions for validation.

    This trait type allows for deterministic, non-LLM evaluation using:
    - Regex patterns for simple text matching
    - Custom callable functions for complex logic

    The trait always returns a boolean result.
    """

    name: str = Field(..., min_length=1, description="Human readable identifier for the trait")
    description: str | None = Field(None, description="Detailed description of what this trait evaluates")
    pattern: str | None = Field(
        None, description="Regex pattern to match against text (mutually exclusive with callable)"
    )
    callable_name: str | None = Field(
        None, description="Name of registered callable function (mutually exclusive with pattern)"
    )
    case_sensitive: bool = Field(True, description="Whether pattern matching should be case sensitive")
    invert_result: bool = Field(False, description="Whether to invert the boolean result (for negative matching)")

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_mutually_exclusive(self) -> "ManualRubricTrait":
        """Ensure only one of pattern or callable_name is specified."""
        pattern = self.pattern
        callable_name = self.callable_name

        if pattern and callable_name:
            raise ValueError("Only one of 'pattern' or 'callable_name' can be specified, not both")

        if not pattern and not callable_name:
            raise ValueError("Either 'pattern' or 'callable_name' must be specified")

        return self

    @field_validator("pattern")
    @classmethod
    def validate_regex_pattern(cls, v: str | None) -> str | None:
        """Validate that pattern is a valid regex."""
        if v is not None:
            try:
                re.compile(v)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {e}") from e
        return v

    def evaluate(self, text: str, callable_registry: dict[str, Callable[[str], bool]] | None = None) -> bool:
        """
        Evaluate the trait against the provided text.

        Args:
            text: The text to evaluate
            callable_registry: Registry of available callable functions

        Returns:
            Boolean evaluation result

        Raises:
            ValueError: If callable_name is specified but not found in registry
            RuntimeError: If evaluation fails
        """
        try:
            if self.pattern:
                flags = 0 if self.case_sensitive else re.IGNORECASE
                match = re.search(self.pattern, text, flags)
                result = match is not None
            elif self.callable_name:
                if not callable_registry or self.callable_name not in callable_registry:
                    raise ValueError(f"Callable '{self.callable_name}' not found in registry")

                callable_func = callable_registry[self.callable_name]
                result = callable_func(text)
                if not isinstance(result, bool):
                    raise ValueError(f"Callable '{self.callable_name}' must return boolean, got {type(result)}")
            else:
                raise ValueError("Neither pattern nor callable_name is specified")

            return not result if self.invert_result else result

        except Exception as e:
            raise RuntimeError(f"Failed to evaluate manual trait '{self.name}': {e}") from e


# Valid metric names that can be computed
# With TP (what should be extracted) and TN (what should not be extracted) instructions:
# - TP: Excerpts in model output matching TP instructions (correct extractions)
# - FP: Excerpts in model output matching TN instructions (incorrect extractions)
# - Precision = TP / (TP + FP)
VALID_METRICS = {"precision"}

# Metric computation requirements (which instruction buckets are needed)
METRIC_REQUIREMENTS = {
    "precision": {"tp", "tn"},  # TP identifies correct extractions, TN identifies incorrect ones (FP)
}


class MetricRubricTrait(BaseModel):
    """
    Metric evaluation trait using extraction-based precision analysis.

    This trait type evaluates free-text answers by having an LLM identify excerpts
    in the model output that match two instruction categories:
    - TP instructions: What the model SHOULD extract (correct extractions)
    - TN instructions: What the model SHOULD NOT extract (incorrect extractions become FP)

    From these categories, the system computes:
    - TP count: Number of excerpts matching TP instructions (correct)
    - FP count: Number of excerpts matching TN instructions (incorrect)
    - Precision: TP / (TP + FP)

    The trait returns both the excerpt lists and computed precision value.
    """

    name: str = Field(..., min_length=1, description="Human readable identifier for the trait")
    description: str | None = Field(None, description="Detailed description of what this trait evaluates")
    metrics: list[str] = Field(
        ..., min_length=1, description="List of metrics to compute (currently only 'precision' is supported)"
    )
    tp_instructions: list[str] = Field(
        default_factory=list,
        description="Instructions for identifying correct extractions (what should be in the answer)",
    )
    tn_instructions: list[str] = Field(
        default_factory=list,
        description="Instructions for identifying incorrect extractions (what should not be in the answer)",
    )
    repeated_extraction: bool = Field(
        True, description="Whether to deduplicate repeated excerpts (case-insensitive exact match)"
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("metrics")
    @classmethod
    def validate_metric_names(cls, v: list[str]) -> list[str]:
        """Validate that all requested metrics are valid."""
        if not v:
            raise ValueError("At least one metric must be specified")

        invalid_metrics = set(v) - VALID_METRICS
        if invalid_metrics:
            raise ValueError(f"Invalid metric names: {invalid_metrics}. Valid metrics are: {VALID_METRICS}")

        return v

    @model_validator(mode="after")
    def validate_metric_computability(self) -> "MetricRubricTrait":
        """Validate that requested metrics can be computed from provided instruction buckets."""
        # Check that both TP and TN instructions are provided (required for precision)
        if not self.tp_instructions:
            raise ValueError("TP instructions must be provided (to identify correct extractions)")

        if not self.tn_instructions:
            raise ValueError("TN instructions must be provided (to identify incorrect extractions as FP)")

        # Determine which buckets have instructions
        available_buckets = {"tp", "tn"}

        # Check each requested metric can be computed
        uncomputable_metrics = []
        for metric in self.metrics:
            required_buckets = METRIC_REQUIREMENTS[metric]
            if not required_buckets.issubset(available_buckets):
                missing = required_buckets - available_buckets
                uncomputable_metrics.append(f"{metric} (needs {missing})")

        if uncomputable_metrics:
            raise ValueError(
                f"Cannot compute the following metrics with provided instruction buckets: "
                f"{', '.join(uncomputable_metrics)}. "
                f"Available buckets: {available_buckets}"
            )

        return self

    def get_required_buckets(self) -> set[str]:
        """Get the set of instruction buckets required for computing all requested metrics."""
        # Currently only precision is supported, which requires both TP and TN
        return {"tp", "tn"}


class Rubric(BaseModel):
    """
    Collection of evaluation traits applied to all question-answer pairs.

    A rubric defines the qualitative criteria used to evaluate LLM responses
    beyond basic correctness checking. Supports LLM-based, manual, and metric traits.
    """

    traits: list[RubricTrait] = Field(default_factory=list, description="List of LLM-based evaluation traits")
    manual_traits: list[ManualRubricTrait] = Field(default_factory=list, description="List of manual evaluation traits")
    metric_traits: list[MetricRubricTrait] = Field(
        default_factory=list, description="List of metric-based evaluation traits (confusion-matrix analysis)"
    )

    model_config = ConfigDict(extra="forbid")

    def get_trait_names(self) -> list[str]:
        """Get list of all trait names in this rubric (LLM, manual, and metric)."""
        llm_names = [trait.name for trait in self.traits]
        manual_names = [trait.name for trait in self.manual_traits]
        metric_names = [trait.name for trait in self.metric_traits]
        return llm_names + manual_names + metric_names

    def get_llm_trait_names(self) -> list[str]:
        """Get list of LLM trait names only."""
        return [trait.name for trait in self.traits]

    def get_manual_trait_names(self) -> list[str]:
        """Get list of manual trait names only."""
        return [trait.name for trait in self.manual_traits]

    def get_metric_trait_names(self) -> list[str]:
        """Get list of metric trait names only."""
        return [trait.name for trait in self.metric_traits]

    def validate_evaluation(self, evaluation: dict[str, int | bool]) -> bool:
        """
        Validate that an evaluation result matches this rubric structure.

        Note: This validates LLM and manual trait scores only. Metric traits
        are stored separately in VerificationResult fields (metric_trait_confusion_lists
        and metric_trait_metrics) and don't participate in this validation.
        """
        # Get trait names excluding metric traits (they're validated separately)
        llm_names = set(self.get_llm_trait_names())
        manual_names = set(self.get_manual_trait_names())
        expected_names = llm_names | manual_names

        eval_names = set(evaluation.keys())

        # Check that all expected trait names are present
        if expected_names != eval_names:
            return False

        # Check that each score is valid for its trait
        llm_trait_map = {trait.name: trait for trait in self.traits}
        manual_trait_map = {trait.name: trait for trait in self.manual_traits}

        for name, value in evaluation.items():
            if name in llm_trait_map:
                if not llm_trait_map[name].validate_score(value):
                    return False
            elif name in manual_trait_map:
                # Manual traits always return boolean
                if not isinstance(value, bool):
                    return False
            else:
                # Unknown trait name
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

    # Check for trait name conflicts (across all trait types)
    global_all_names = set(global_rubric.get_trait_names())
    question_all_names = set(question_rubric.get_trait_names())
    conflicts = global_all_names.intersection(question_all_names)

    if conflicts:
        raise ValueError(f"Trait name conflicts between global and question rubrics: {conflicts}")

    # Merge all trait types separately
    merged_traits = list(global_rubric.traits) + list(question_rubric.traits)
    merged_manual_traits = list(global_rubric.manual_traits) + list(question_rubric.manual_traits)
    merged_metric_traits = list(global_rubric.metric_traits) + list(question_rubric.metric_traits)

    return Rubric(traits=merged_traits, manual_traits=merged_manual_traits, metric_traits=merged_metric_traits)
