"""
Rubric data models for qualitative evaluation traits.
"""

import base64
import re
import warnings
from collections.abc import Callable
from typing import Any, Literal

import cloudpickle
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator

TraitKind = Literal["boolean", "score", "literal"]


class LLMRubricTrait(BaseModel):
    """
    LLM-evaluated trait for qualitative assessment.

    A trait can be:
    - boolean (true/false): Binary pass/fail assessment
    - score (1-5 scale): Numeric rating within a range
    - literal (categorical): Classification into predefined classes

    For kind="literal":
    - The `classes` field is REQUIRED
    - `min_score` is automatically set to 0 (first class index)
    - `max_score` is automatically set to len(classes)-1 (last class index)
    - Returns int index (0, 1, 2...) based on class order
    - `higher_is_better` controls ordering interpretation

    Deep Judgment Mode (optional):
        When enabled, provides evidence-based evaluation with:
        - Optional excerpt extraction from answer text
        - Retry mechanism with validation feedback
        - Reasoning generation explaining the score
        - Optional search-enhanced hallucination detection
    """

    name: str = Field(..., min_length=1, description="Human readable identifier for the trait")
    description: str | None = Field(None, description="Detailed description shown to user/LLM")
    kind: TraitKind = Field(..., description="Type of trait: 'boolean', 'score', or 'literal'")
    min_score: int | None = Field(1, description="Lower bound for score traits (default: 1). Auto-derived for literal.")
    max_score: int | None = Field(5, description="Upper bound for score traits (default: 5). Auto-derived for literal.")

    # Literal-specific field (required when kind="literal")
    classes: dict[str, str] | None = Field(
        None,
        description="Class name → description mapping. Required when kind='literal'. "
        "Order determines indices (0, 1, 2...). Must have 2-20 classes.",
    )

    # Deep Judgment fields
    deep_judgment_enabled: bool = Field(
        False,
        description="Enable deep judgment evaluation for this trait (multi-stage with reasoning)",
    )
    deep_judgment_excerpt_enabled: bool = Field(
        True,
        description="Extract verbatim excerpts from answer as evidence (only if deep_judgment_enabled=True)",
    )
    deep_judgment_max_excerpts: int | None = Field(
        None,
        description="Maximum number of excerpts to extract (overrides global default if set)",
    )
    deep_judgment_fuzzy_match_threshold: float | None = Field(
        None,
        description="Fuzzy matching threshold for excerpt validation 0.0-1.0 (overrides global default if set)",
    )
    deep_judgment_excerpt_retry_attempts: int | None = Field(
        None,
        description="Number of retry attempts for excerpt extraction (overrides global default if set)",
    )
    deep_judgment_search_enabled: bool = Field(
        False,
        description="Enable search-enhanced hallucination detection for excerpts (only if excerpt_enabled=True)",
    )

    # Directionality field
    higher_is_better: bool = Field(
        ...,
        description="Whether higher values indicate better performance. "
        "For boolean: True means True is good. "
        "For score: True means higher scores are better. "
        "For literal: True means higher indices (later classes) are better.",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("classes")
    @classmethod
    def validate_classes(cls, v: dict[str, str] | None) -> dict[str, str] | None:
        """Validate class definitions when present."""
        if v is None:
            return None
        if len(v) < 2:
            raise ValueError("Literal trait must have at least 2 classes")
        if len(v) > 20:
            raise ValueError("Literal trait cannot have more than 20 classes")

        seen_names: set[str] = set()
        for class_name, class_desc in v.items():
            if not class_name.strip():
                raise ValueError("Class names cannot be empty")
            if not class_desc.strip():
                raise ValueError(f"Description for class '{class_name}' cannot be empty")
            lower_name = class_name.lower()
            if lower_name in seen_names:
                raise ValueError(f"Duplicate class name (case-insensitive): '{class_name}'")
            seen_names.add(lower_name)
        return v

    @model_validator(mode="before")
    @classmethod
    def set_legacy_defaults(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Set default for higher_is_better when loading legacy data."""
        if isinstance(values, dict) and ("higher_is_better" not in values or values.get("higher_is_better") is None):
            values["higher_is_better"] = True
        return values

    @model_validator(mode="after")
    def validate_kind_fields(self) -> "LLMRubricTrait":
        """Validate and set kind-specific fields."""
        if self.kind == "literal":
            if self.classes is None:
                raise ValueError("classes field is required when kind='literal'")
            # Automatically derive min_score and max_score from classes
            object.__setattr__(self, "min_score", 0)
            object.__setattr__(self, "max_score", len(self.classes) - 1)
        return self

    def get_class_names(self) -> list[str]:
        """Get list of valid class names (preserves dict order). Only for kind='literal'."""
        if self.kind != "literal" or self.classes is None:
            return []
        return list(self.classes.keys())

    def get_class_index(self, class_name: str) -> int:
        """Get numeric index for a class name. Returns -1 if invalid. Only for kind='literal'."""
        class_names = self.get_class_names()
        try:
            return class_names.index(class_name)
        except ValueError:
            return -1

    def validate_score(self, value: int | bool) -> bool:
        """Validate that a given score is valid for this trait."""
        if self.kind == "boolean":
            return isinstance(value, bool)
        else:  # kind == "score" or kind == "literal"
            # Both use min_score/max_score (literal derives them from classes)
            # Reject boolean values explicitly (bool is a subclass of int in Python)
            if isinstance(value, bool):
                return False
            if not isinstance(value, int):
                return False
            # Use explicit None checks to allow min_score=0
            min_val = self.min_score if self.min_score is not None else 0
            max_val = self.max_score if self.max_score is not None else 5
            # For literal, also allow -1 as error state
            if self.kind == "literal" and value == -1:
                return True
            return min_val <= value <= max_val


class RegexTrait(BaseModel):
    """
    Regex-based evaluation trait for deterministic pattern matching.

    This trait type uses regular expressions to perform simple text matching
    against answers. It always returns a boolean result.

    Examples:
        - Email format validation: r"\\S+@\\S+"
        - Keyword presence: r"\\bmachine learning\\b"
        - URL detection: r"https?://[^\\s]+"
    """

    name: str = Field(..., min_length=1, description="Human readable identifier for the trait")
    description: str | None = Field(None, description="Detailed description of what this trait evaluates")
    pattern: str = Field(..., description="Regex pattern to match against text")
    case_sensitive: bool = Field(True, description="Whether pattern matching should be case sensitive")
    invert_result: bool = Field(False, description="Whether to invert the boolean result (for negative matching)")

    # Directionality field
    higher_is_better: bool = Field(
        ...,
        description="Whether a regex match indicates a positive outcome. True: match = good. False: match = bad.",
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def set_legacy_defaults(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Set default for higher_is_better when loading legacy data."""
        if isinstance(values, dict) and ("higher_is_better" not in values or values.get("higher_is_better") is None):
            values["higher_is_better"] = True
        return values

    @field_validator("pattern")
    @classmethod
    def validate_regex_pattern(cls, v: str) -> str:
        """Validate that pattern is a valid regex."""
        try:
            re.compile(v)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}") from e
        return v

    def evaluate(self, text: str) -> bool:
        """
        Evaluate the trait against the provided text.

        Args:
            text: The text to evaluate

        Returns:
            Boolean evaluation result

        Raises:
            RuntimeError: If evaluation fails
        """
        try:
            flags = 0 if self.case_sensitive else re.IGNORECASE
            match = re.search(self.pattern, text, flags)
            result = match is not None
            return not result if self.invert_result else result
        except Exception as e:
            raise RuntimeError(f"Failed to evaluate regex trait '{self.name}': {e}") from e


class CallableTrait(BaseModel):
    """
    Callable-based evaluation trait using custom Python functions.

    This trait type serializes and stores custom Python functions using cloudpickle,
    enabling complex, stateful, or domain-specific validation logic that cannot be
    expressed as simple regex patterns.

    **SECURITY WARNING**: Deserializing callable code can execute arbitrary Python code.
    Only load CallableTrait instances from trusted sources. CallableTrait cannot be
    created via the web API for security reasons.

    The trait can return either boolean (pass/fail) or numeric score results, matching
    LLMRubricTrait behavior.

    Examples:
        Boolean:
        - Word count validation: lambda text: len(text.split()) >= 50
        - Custom domain logic: checking medical terminology consistency

        Score:
        - Readability score: lambda text: calculate_flesch_kincaid(text)
        - Custom metric: lambda text: compute_domain_score(text)
    """

    name: str = Field(..., min_length=1, description="Human readable identifier for the trait")
    description: str | None = Field(None, description="Detailed description of what this trait evaluates")
    kind: TraitKind = Field(..., description="Type of evaluation: 'boolean' for pass/fail, 'score' for numeric")
    callable_code: bytes = Field(..., description="Serialized callable function (cloudpickle)")
    min_score: int | None = Field(None, description="Minimum score value (required if kind='score')")
    max_score: int | None = Field(None, description="Maximum score value (required if kind='score')")
    invert_result: bool = Field(False, description="Whether to invert the boolean result (only for kind='boolean')")

    # Directionality field
    higher_is_better: bool = Field(
        ...,
        description="Whether higher return values indicate better performance. "
        "True: high value = good. False: high value = bad.",
    )

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    @model_validator(mode="before")
    @classmethod
    def set_legacy_defaults(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Set default for higher_is_better when loading legacy data."""
        if isinstance(values, dict) and ("higher_is_better" not in values or values.get("higher_is_better") is None):
            values["higher_is_better"] = True
        return values

    @field_serializer("callable_code")
    def serialize_callable_code(self, value: bytes, _info: Any) -> str:
        """Serialize callable_code bytes to base64 string for JSON export."""
        return base64.b64encode(value).decode("ascii")

    @field_validator("callable_code", mode="before")
    @classmethod
    def validate_callable_code(cls, value: bytes | str) -> bytes:
        """Convert base64 string to bytes if needed."""
        if isinstance(value, bytes):
            return value
        if isinstance(value, str):
            return base64.b64decode(value)
        raise ValueError(f"callable_code must be bytes or base64 string, got {type(value)}")

    @classmethod
    def from_callable(
        cls,
        name: str,
        func: Callable[[str], bool | int],
        kind: TraitKind,
        description: str | None = None,
        min_score: int | None = None,
        max_score: int | None = None,
        invert_result: bool = False,
        higher_is_better: bool = True,
    ) -> "CallableTrait":
        """
        Create a CallableTrait from a callable function.

        Args:
            name: Trait name
            func: Function that takes a string (the verification trace/answer text) and returns bool or int
            kind: Type of evaluation - 'boolean' or 'score'
            description: Optional trait description
            min_score: Minimum score (required if kind='score')
            max_score: Maximum score (required if kind='score')
            invert_result: Whether to invert boolean result (only for kind='boolean')
            higher_is_better: Whether higher return values indicate better performance

        Returns:
            CallableTrait instance with serialized function

        Raises:
            ValueError: If function signature is invalid or score parameters are missing
        """
        # Validate function signature
        import inspect

        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        if len(params) != 1:
            raise ValueError(f"Callable must have exactly one parameter, got {len(params)}")

        # Validate score parameters
        if kind == "score":
            if min_score is None or max_score is None:
                raise ValueError("min_score and max_score are required when kind='score'")
            if min_score >= max_score:
                raise ValueError(f"min_score ({min_score}) must be less than max_score ({max_score})")
        else:  # kind == "boolean"
            if min_score is not None or max_score is not None:
                raise ValueError("min_score and max_score should not be set when kind='boolean'")

        # Serialize the function
        callable_code = cloudpickle.dumps(func)

        return cls(
            name=name,
            description=description,
            kind=kind,
            callable_code=callable_code,
            min_score=min_score,
            max_score=max_score,
            invert_result=invert_result,
            higher_is_better=higher_is_better,
        )

    def deserialize_callable(self) -> Callable[[str], bool | int]:
        """
        Deserialize the callable function from stored bytes.

        **SECURITY WARNING**: This executes code that was serialized and may contain
        arbitrary Python code. Only deserialize callables from trusted sources.

        Returns:
            The deserialized callable function

        Raises:
            RuntimeError: If deserialization fails
        """
        try:
            warnings.warn(
                f"Deserializing callable for trait '{self.name}'. "
                "This executes stored code. Only load from trusted sources.",
                category=UserWarning,
                stacklevel=2,
            )
            callable_func: Callable[[str], bool | int] = cloudpickle.loads(self.callable_code)
            return callable_func
        except Exception as e:
            raise RuntimeError(f"Failed to deserialize callable for trait '{self.name}': {e}") from e

    def evaluate(self, text: str) -> bool | int:
        """
        Evaluate the trait against the provided text.

        Args:
            text: The text to evaluate (verification trace or answer text)

        Returns:
            Boolean result for kind='boolean', numeric score for kind='score'

        Raises:
            RuntimeError: If evaluation fails
            ValueError: If return type doesn't match kind or score is out of range
        """
        try:
            func = self.deserialize_callable()
            result = func(text)

            if self.kind == "boolean":
                if not isinstance(result, bool):
                    raise ValueError(f"Callable with kind='boolean' must return bool, got {type(result)}")
                return not result if self.invert_result else result
            else:  # kind == "score"
                if not isinstance(result, int | float):
                    raise ValueError(f"Callable with kind='score' must return int or float, got {type(result)}")

                # Convert to int if float
                score = int(result) if isinstance(result, float) else result

                # Validate score range
                if self.min_score is not None and score < self.min_score:
                    raise ValueError(f"Score {score} is below minimum {self.min_score} for trait '{self.name}'")
                if self.max_score is not None and score > self.max_score:
                    raise ValueError(f"Score {score} is above maximum {self.max_score} for trait '{self.name}'")

                return score
        except Exception as e:
            raise RuntimeError(f"Failed to evaluate callable trait '{self.name}': {e}") from e


# Valid metric names for each evaluation mode
# TP-only mode: Only TP instructions provided, FP = anything else not in TP list
VALID_METRICS_TP_ONLY = {"precision", "recall", "f1"}

# Full matrix mode: Both TP and TN instructions provided
VALID_METRICS_FULL_MATRIX = {"precision", "recall", "specificity", "accuracy", "f1"}

# All valid metrics (union of both modes)
VALID_METRICS = VALID_METRICS_TP_ONLY | VALID_METRICS_FULL_MATRIX

# Metric computation requirements (which confusion matrix values are needed)
METRIC_REQUIREMENTS = {
    "precision": {"tp", "fp"},
    "recall": {"tp", "fn"},
    "specificity": {"tn", "fp"},
    "accuracy": {"tp", "tn", "fp", "fn"},
    "f1": {"tp", "fp", "fn"},
}


class MetricRubricTrait(BaseModel):
    """
    Metric evaluation trait using instruction-level confusion matrix analysis.

    Two evaluation modes are supported:

    1. TP-only mode (evaluation_mode="tp_only"):
       - User defines: TP instructions (what should be present)
       - System evaluates:
         * TP: Instructions found in answer
         * FN: Instructions missing from answer
         * FP: Extra content in answer not matching TP instructions
         * TN: Cannot be computed (no explicit negative set)
       - Available metrics: precision, recall, f1

    2. Full matrix mode (evaluation_mode="full_matrix"):
       - User defines: TP instructions (should be present) + TN instructions (should NOT be present)
       - System evaluates:
         * TP: TP instructions found in answer
         * FN: TP instructions missing from answer
         * TN: TN instructions correctly absent
         * FP: TN instructions incorrectly present
       - Available metrics: precision, recall, specificity, accuracy, f1

    The trait returns confusion matrix counts/lists and computed metric values.
    """

    name: str = Field(..., min_length=1, description="Human readable identifier for the trait")
    description: str | None = Field(None, description="Detailed description of what this trait evaluates")
    evaluation_mode: Literal["tp_only", "full_matrix"] = Field(
        "tp_only", description="Evaluation mode: tp_only (only TP defined) or full_matrix (TP+TN defined)"
    )
    metrics: list[str] = Field(
        ...,
        min_length=1,
        description="List of metrics to compute (mode-dependent: see VALID_METRICS_TP_ONLY and VALID_METRICS_FULL_MATRIX)",
    )
    tp_instructions: list[str] = Field(
        default_factory=list,
        description="Instructions for what should be present in the answer",
    )
    tn_instructions: list[str] = Field(
        default_factory=list,
        description="Instructions for what should NOT be present in the answer (required in full_matrix mode, ignored in tp_only mode)",
    )
    repeated_extraction: bool = Field(
        True, description="Whether to deduplicate repeated excerpts/instructions (case-insensitive exact match)"
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
        """Validate that requested metrics are compatible with the evaluation mode and provided instructions."""
        # Validate TP instructions are always provided
        if not self.tp_instructions:
            raise ValueError("TP instructions must be provided (define what should be present in the answer)")

        # Mode-specific validation
        if self.evaluation_mode == "tp_only":
            # TP-only mode: TN instructions should be empty, validate metrics
            valid_metrics_for_mode = VALID_METRICS_TP_ONLY
            invalid_for_mode = set(self.metrics) - valid_metrics_for_mode
            if invalid_for_mode:
                raise ValueError(
                    f"Metrics {invalid_for_mode} are not available in tp_only mode. "
                    f"Available metrics: {valid_metrics_for_mode}. "
                    f"Use full_matrix mode for specificity and accuracy."
                )
            # In tp_only mode, we can compute: TP, FN, FP (but not TN)
            available_buckets = {"tp", "fn", "fp"}

        elif self.evaluation_mode == "full_matrix":
            # Full matrix mode: Both TP and TN instructions required
            if not self.tn_instructions:
                raise ValueError(
                    "TN instructions must be provided in full_matrix mode "
                    "(define what should NOT be present in the answer)"
                )
            valid_metrics_for_mode = VALID_METRICS_FULL_MATRIX
            invalid_for_mode = set(self.metrics) - valid_metrics_for_mode
            if invalid_for_mode:
                raise ValueError(
                    f"Metrics {invalid_for_mode} are not valid. Available metrics: {valid_metrics_for_mode}"
                )
            # In full_matrix mode, we can compute all four: TP, FN, TN, FP
            available_buckets = {"tp", "fn", "tn", "fp"}

        # Check each requested metric can be computed
        uncomputable_metrics = []
        for metric in self.metrics:
            required_buckets = METRIC_REQUIREMENTS[metric]
            if not required_buckets.issubset(available_buckets):
                missing = required_buckets - available_buckets
                uncomputable_metrics.append(f"{metric} (needs {missing})")

        if uncomputable_metrics:
            raise ValueError(
                f"Cannot compute the following metrics with current mode ({self.evaluation_mode}): "
                f"{', '.join(uncomputable_metrics)}. "
                f"Available buckets: {available_buckets}"
            )

        return self

    def get_required_buckets(self) -> set[str]:
        """Get the set of confusion matrix buckets that will be computed for this trait."""
        if self.evaluation_mode == "tp_only":
            return {"tp", "fn", "fp"}  # TN cannot be computed
        else:  # full_matrix
            return {"tp", "fn", "tn", "fp"}  # All four can be computed


class Rubric(BaseModel):
    """
    Collection of evaluation traits applied to all question-answer pairs.

    A rubric defines the qualitative criteria used to evaluate LLM responses
    beyond basic correctness checking. Supports LLM-based, regex, callable, and metric traits.
    """

    llm_traits: list[LLMRubricTrait] = Field(default_factory=list, description="List of LLM-based evaluation traits")
    regex_traits: list[RegexTrait] = Field(default_factory=list, description="List of regex-based evaluation traits")
    callable_traits: list[CallableTrait] = Field(
        default_factory=list, description="List of callable function-based evaluation traits"
    )
    metric_traits: list[MetricRubricTrait] = Field(
        default_factory=list, description="List of metric-based evaluation traits (confusion-matrix analysis)"
    )

    model_config = ConfigDict(extra="forbid")

    def get_trait_names(self) -> list[str]:
        """Get list of all trait names in this rubric (LLM, regex, callable, and metric)."""
        llm_names = [trait.name for trait in self.llm_traits]
        regex_names = [trait.name for trait in self.regex_traits]
        callable_names = [trait.name for trait in self.callable_traits]
        metric_names = [trait.name for trait in self.metric_traits]
        return llm_names + regex_names + callable_names + metric_names

    def get_llm_trait_names(self) -> list[str]:
        """Get list of LLM trait names only."""
        return [trait.name for trait in self.llm_traits]

    def get_regex_trait_names(self) -> list[str]:
        """Get list of regex trait names only."""
        return [trait.name for trait in self.regex_traits]

    def get_callable_trait_names(self) -> list[str]:
        """Get list of callable trait names only."""
        return [trait.name for trait in self.callable_traits]

    def get_metric_trait_names(self) -> list[str]:
        """Get list of metric trait names only."""
        return [trait.name for trait in self.metric_traits]

    def get_trait_max_scores(self) -> dict[str, int]:
        """Get max_score for all score-based traits (LLM and callable).

        Returns:
            Dict mapping trait name to max_score for traits with kind='score' or 'literal'.
            Boolean traits and metric traits are not included.
            For literal traits, max_score is len(classes)-1.
        """
        max_scores: dict[str, int] = {}

        for llm_trait in self.llm_traits:
            if llm_trait.kind in ("score", "literal") and llm_trait.max_score is not None:
                max_scores[llm_trait.name] = llm_trait.max_score

        for callable_trait in self.callable_traits:
            if callable_trait.kind == "score" and callable_trait.max_score is not None:
                max_scores[callable_trait.name] = callable_trait.max_score

        return max_scores

    def get_trait_directionalities(self) -> dict[str, bool]:
        """Get higher_is_better for LLM, regex, and callable traits.

        Note: MetricRubricTraits are excluded as metrics (precision/recall/F1)
        are inherently 'higher is better'.

        Returns:
            Dict mapping trait name to higher_is_better value.
        """
        directionalities: dict[str, bool] = {}

        llm_trait: LLMRubricTrait
        for llm_trait in self.llm_traits:
            directionalities[llm_trait.name] = llm_trait.higher_is_better

        regex_trait: RegexTrait
        for regex_trait in self.regex_traits:
            directionalities[regex_trait.name] = regex_trait.higher_is_better

        callable_trait: CallableTrait
        for callable_trait in self.callable_traits:
            directionalities[callable_trait.name] = callable_trait.higher_is_better

        # MetricRubricTraits always have higher_is_better=True (implicit)
        return directionalities

    def validate_evaluation(self, evaluation: dict[str, int | bool]) -> bool:
        """
        Validate that an evaluation result matches this rubric structure.

        Note: This validates LLM, regex, and callable trait scores only. Metric traits
        are stored separately in VerificationResult fields (metric_trait_confusion_lists
        and metric_trait_metrics) and don't participate in this validation.
        """
        # Get trait names excluding metric traits (they're validated separately)
        llm_names = set(self.get_llm_trait_names())
        regex_names = set(self.get_regex_trait_names())
        callable_names = set(self.get_callable_trait_names())
        expected_names = llm_names | regex_names | callable_names

        eval_names = set(evaluation.keys())

        # Check that all expected trait names are present
        if expected_names != eval_names:
            return False

        # Check that each score is valid for its trait
        llm_trait_map = {trait.name: trait for trait in self.llm_traits}
        regex_trait_map = {trait.name: trait for trait in self.regex_traits}
        callable_trait_map = {trait.name: trait for trait in self.callable_traits}

        for name, value in evaluation.items():
            if name in llm_trait_map:
                if not llm_trait_map[name].validate_score(value):
                    return False
            elif name in regex_trait_map or name in callable_trait_map:
                # Regex and callable traits always return boolean
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
    merged_traits = list(global_rubric.llm_traits) + list(question_rubric.llm_traits)
    merged_regex_traits = list(global_rubric.regex_traits) + list(question_rubric.regex_traits)
    merged_callable_traits = list(global_rubric.callable_traits) + list(question_rubric.callable_traits)
    merged_metric_traits = list(global_rubric.metric_traits) + list(question_rubric.metric_traits)

    return Rubric(
        llm_traits=merged_traits,
        regex_traits=merged_regex_traits,
        callable_traits=merged_callable_traits,
        metric_traits=merged_metric_traits,
    )
