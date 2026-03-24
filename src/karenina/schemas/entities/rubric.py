"""
Rubric data models for qualitative evaluation traits.
"""

import base64
import logging
import re
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import cloudpickle
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator

from karenina.schemas.entities._schema_reconstruction import _reconstruct_model_from_schema
from karenina.schemas.entities._template_validation import _validate_template_fields

if TYPE_CHECKING:
    from karenina.schemas.config.models import ModelConfig

logger = logging.getLogger(__name__)

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
    summary: str | None = Field(None, description="Short concept label for dynamic rubric presence check")
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


class RegexRubricTrait(BaseModel):
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
    summary: str | None = Field(None, description="Short concept label for dynamic rubric presence check")
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


class CallableRubricTrait(BaseModel):
    """
    Callable-based evaluation trait using custom Python functions.

    This trait type serializes and stores custom Python functions using cloudpickle,
    enabling complex, stateful, or domain-specific validation logic that cannot be
    expressed as simple regex patterns.

    **SECURITY WARNING**: Deserializing callable code can execute arbitrary Python code.
    Only load CallableRubricTrait instances from trusted sources. CallableRubricTrait cannot be
    created via the web API for security reasons.

    Supported kinds:
    - boolean: callable returns bool (pass/fail)
    - score: callable returns int or float (numeric rating within min/max range)
    - literal: callable returns str (class label from predefined classes dict)

    For kind="literal":
    - The ``classes`` field is REQUIRED (dict mapping class name to description)
    - ``min_score`` is automatically set to 0 (first class index)
    - ``max_score`` is automatically set to len(classes)-1 (last class index)
    - The callable must return a string that matches one of the class names
    - evaluate() returns the int index of the matched class

    Examples:
        Boolean:
        - Word count validation: lambda text: len(text.split()) >= 50
        - Custom domain logic: checking medical terminology consistency

        Score (int or float):
        - Readability score: lambda text: calculate_flesch_kincaid(text)
        - Custom metric: lambda text: compute_domain_score(text)

        Literal:
        - Tone classifier: lambda text: "formal" if "therefore" in text else "casual"
    """

    name: str = Field(..., min_length=1, description="Human readable identifier for the trait")
    description: str | None = Field(None, description="Detailed description of what this trait evaluates")
    summary: str | None = Field(None, description="Short concept label for dynamic rubric presence check")
    kind: TraitKind = Field(..., description="Type of evaluation: 'boolean', 'score', or 'literal'")
    callable_code: bytes = Field(..., description="Serialized callable function (cloudpickle)")
    min_score: int | None = Field(
        None, description="Minimum score value (required if kind='score', auto-derived for 'literal')"
    )
    max_score: int | None = Field(
        None, description="Maximum score value (required if kind='score', auto-derived for 'literal')"
    )
    invert_result: bool = Field(False, description="Whether to invert the boolean result (only for kind='boolean')")

    # Literal-specific field (required when kind="literal")
    classes: dict[str, str] | None = Field(
        None,
        description="Class name to description mapping. Required when kind='literal'. "
        "Order determines indices (0, 1, 2...). Must have 2-20 classes.",
    )

    # Directionality field
    higher_is_better: bool = Field(
        ...,
        description="Whether higher return values indicate better performance. "
        "True: high value = good. False: high value = bad.",
    )

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

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
        return v

    @model_validator(mode="before")
    @classmethod
    def set_legacy_defaults(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Set defaults for higher_is_better and validate literal kind requires classes."""
        if isinstance(values, dict):
            if "higher_is_better" not in values or values.get("higher_is_better") is None:
                values["higher_is_better"] = True

            # Literal kind requires classes
            if values.get("kind") == "literal":
                if not values.get("classes"):
                    raise ValueError("classes field is required when kind='literal'")
                # Auto-derive min/max score from classes
                num_classes = len(values["classes"])
                if values.get("min_score") is None:
                    values["min_score"] = 0
                if values.get("max_score") is None:
                    values["max_score"] = num_classes - 1

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
        func: Callable[[str], bool | int | float | str],
        kind: TraitKind,
        description: str | None = None,
        summary: str | None = None,
        min_score: int | None = None,
        max_score: int | None = None,
        invert_result: bool = False,
        higher_is_better: bool = True,
        classes: dict[str, str] | None = None,
    ) -> "CallableRubricTrait":
        """
        Create a CallableRubricTrait from a callable function.

        Args:
            name: Trait name.
            func: Function that takes a string and returns bool, int, float, or str
                depending on kind.
            kind: Type of evaluation: 'boolean', 'score', or 'literal'.
            description: Optional trait description.
            summary: Short concept label for dynamic rubric presence check.
            min_score: Minimum score (required if kind='score', auto-derived for 'literal').
            max_score: Maximum score (required if kind='score', auto-derived for 'literal').
            invert_result: Whether to invert boolean result (only for kind='boolean').
            higher_is_better: Whether higher return values indicate better performance.
            classes: Class name to description mapping (required if kind='literal').

        Returns:
            CallableRubricTrait instance with serialized function.

        Raises:
            ValueError: If function signature is invalid or required parameters are missing.
        """
        import inspect

        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        if len(params) != 1:
            raise ValueError(f"Callable must have exactly one parameter, got {len(params)}")

        if kind == "score":
            if min_score is None or max_score is None:
                raise ValueError("min_score and max_score are required when kind='score'")
            if min_score >= max_score:
                raise ValueError(f"min_score ({min_score}) must be less than max_score ({max_score})")
        elif kind == "literal":
            if not classes:
                raise ValueError("classes field is required when kind='literal'")
        elif kind == "boolean":
            if min_score is not None or max_score is not None:
                raise ValueError("min_score and max_score should not be set when kind='boolean'")

        callable_code = cloudpickle.dumps(func)

        return cls(
            name=name,
            description=description,
            summary=summary,
            kind=kind,
            callable_code=callable_code,
            min_score=min_score,
            max_score=max_score,
            invert_result=invert_result,
            higher_is_better=higher_is_better,
            classes=classes,
        )

    def deserialize_callable(self) -> Callable[[str], bool | int | float | str]:
        """
        Deserialize the callable function from stored bytes.

        **SECURITY WARNING**: This executes code that was serialized and may contain
        arbitrary Python code. Only deserialize callables from trusted sources.

        Returns:
            The deserialized callable function.

        Raises:
            RuntimeError: If deserialization fails.
        """
        try:
            warnings.warn(
                f"Deserializing callable for trait '{self.name}'. "
                "This executes stored code. Only load from trusted sources.",
                category=UserWarning,
                stacklevel=2,
            )
            callable_func: Callable[[str], bool | int | float | str] = cloudpickle.loads(self.callable_code)
            return callable_func
        except Exception as e:
            raise RuntimeError(f"Failed to deserialize callable for trait '{self.name}': {e}") from e

    def evaluate(self, text: str) -> bool | int | float:
        """
        Evaluate the trait against the provided text.

        Args:
            text: The text to evaluate (verification trace or answer text).

        Returns:
            For kind='boolean': bool result (possibly inverted).
            For kind='score': int or float score within [min_score, max_score].
            For kind='literal': int class index (0-based, matching classes key order).

        Raises:
            RuntimeError: If evaluation fails.
            ValueError: If return type does not match kind or value is out of range.
        """
        try:
            func = self.deserialize_callable()
            result = func(text)

            if self.kind == "boolean":
                if not isinstance(result, bool):
                    raise ValueError(f"Callable with kind='boolean' must return bool, got {type(result)}")
                return not result if self.invert_result else result

            elif self.kind == "literal":
                if not isinstance(result, str):
                    raise ValueError(f"Callable with kind='literal' must return str, got {type(result)}")
                if self.classes is None:
                    raise ValueError(f"Trait '{self.name}' has kind='literal' but no classes defined")
                class_names = list(self.classes.keys())
                if result not in class_names:
                    raise ValueError(
                        f"'{result}' is not a valid class for trait '{self.name}'. Valid classes: {class_names}"
                    )
                return class_names.index(result)

            else:  # kind == "score"
                if not isinstance(result, int | float):
                    raise ValueError(f"Callable with kind='score' must return int or float, got {type(result)}")

                score: int | float = result

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
    summary: str | None = Field(None, description="Short concept label for dynamic rubric presence check")
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


class AgenticRubricTrait(BaseModel):
    """Rubric trait evaluated by an agent with tools.

    Unlike LLMRubricTrait (single LLM call), this trait launches an agent
    that can investigate the response and workspace using tools before
    producing a score. Supports boolean, score, literal, and template kinds.

    When ``kind`` is a ``BaseModel`` subclass (template kind), the agent
    produces structured output matching that schema instead of a scalar
    score. Template kinds require ``higher_is_better=None`` because
    directionality is not meaningful for structured results.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    summary: str | None = Field(None, description="Short concept label for dynamic rubric presence check")
    kind: Literal["boolean", "score", "literal"] | type[BaseModel]
    higher_is_better: bool | None = Field(
        ...,
        description="Whether higher values indicate better performance. "
        "For boolean: True means True is good. "
        "For score: True means higher scores are better. "
        "For literal: True means higher indices (later classes) are better. "
        "Must be None for template kind.",
    )
    min_score: int | None = Field(1, description="Lower bound for score traits (default: 1). Auto-derived for literal.")
    max_score: int | None = Field(5, description="Upper bound for score traits (default: 5). Auto-derived for literal.")
    classes: dict[str, str] | None = None
    context_mode: Literal["workspace_only", "trace_and_workspace", "trace_only"] = "trace_and_workspace"
    materialize_trace: bool = Field(
        False,
        description=(
            "Write the agent trace to a file in the workspace instead of "
            "including it in the prompt. The agent receives the file path "
            "and can use grep/search tools on it."
        ),
    )
    persist_trace: bool = Field(
        False,
        description=(
            "When True, the materialized trace file is kept after evaluation. "
            "When False (default), cleaned up after evaluation."
        ),
    )
    max_turns: int = Field(15, gt=0)
    timeout_seconds: int = Field(120, gt=0)
    model_override: "ModelConfig | None" = None

    @field_validator("kind", mode="before")
    @classmethod
    def validate_kind(cls, v: Any) -> Any:
        """Accept string literals, BaseModel subclasses, or serialized template dicts."""
        if isinstance(v, str):
            return v
        if isinstance(v, type) and issubclass(v, BaseModel):
            _validate_template_fields(v)
            return v
        if isinstance(v, dict) and v.get("type") == "template":
            schema = v.get("schema")
            if schema is None:
                raise ValueError("Template kind dict must include a 'schema' key")
            return _reconstruct_model_from_schema(schema)
        raise ValueError(f"kind must be a string literal, BaseModel subclass, or template dict, got {type(v)}")

    @field_serializer("kind")
    def serialize_kind(self, value: Any, _info: Any) -> Any:
        """Serialize BaseModel subclass to a template dict with JSON Schema."""
        if isinstance(value, type) and issubclass(value, BaseModel):
            return {"type": "template", "schema": value.model_json_schema()}
        return value

    @model_validator(mode="before")
    @classmethod
    def set_legacy_defaults(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Set default for higher_is_better when loading legacy data.

        Skips the legacy default (True) when kind is a template, because
        template kinds require higher_is_better=None.
        """
        if not isinstance(values, dict):
            return values
        kind = values.get("kind")
        # Template kind: do not inject legacy default
        if not isinstance(kind, str):
            return values
        if "higher_is_better" not in values or values.get("higher_is_better") is None:
            values["higher_is_better"] = True
        return values

    @field_validator("higher_is_better", mode="after")
    @classmethod
    def validate_higher_is_better(cls, v: bool | None, info: Any) -> bool | None:
        """Enforce higher_is_better=None for template kind."""
        kind = info.data.get("kind")
        if isinstance(kind, type) and issubclass(kind, BaseModel) and v is not None:
            raise ValueError("higher_is_better must be None for template kind")
        return v

    @model_validator(mode="after")
    def validate_kind_fields(self) -> "AgenticRubricTrait":
        """Validate kind-specific field constraints."""
        if self.materialize_trace and self.context_mode == "workspace_only":
            raise ValueError(
                "materialize_trace=True requires a trace, but context_mode='workspace_only' "
                "excludes the trace. Use 'trace_only' or 'trace_and_workspace'."
            )
        if not isinstance(self.kind, str):
            return self
        if self.kind == "literal":
            if not self.classes:
                raise ValueError("classes field is required for literal kind")
            # Automatically derive min_score and max_score from classes
            object.__setattr__(self, "min_score", 0)
            object.__setattr__(self, "max_score", len(self.classes) - 1)
        return self

    @model_validator(mode="after")
    def validate_model_override_supports_agents(self) -> "AgenticRubricTrait":
        """Validate that model_override supports agent creation (if set)."""
        if self.model_override is not None:
            from karenina.adapters.registry import AdapterRegistry

            spec = AdapterRegistry.get_spec(self.model_override.interface)
            if spec is None or spec.agent_tier != "deep_agent":
                tier = spec.agent_tier if spec else "unknown"
                raise ValueError(
                    f"model_override interface '{self.model_override.interface}' "
                    f"has agent_tier='{tier}'; agentic traits require "
                    f"agent_tier='deep_agent'."
                )
        return self

    def validate_score(self, value: int | bool) -> bool:
        """Validate that a given score is valid for this trait."""
        if self.is_template_kind:
            return True
        if self.kind == "boolean":
            return isinstance(value, bool)
        else:
            if isinstance(value, bool):
                return False
            if not isinstance(value, int):
                return False
            min_val = self.min_score if self.min_score is not None else 0
            max_val = self.max_score if self.max_score is not None else 5
            if self.kind == "literal" and value == -1:
                return True
            return min_val <= value <= max_val

    @property
    def is_template_kind(self) -> bool:
        """Return True if kind is a BaseModel subclass (template kind)."""
        return isinstance(self.kind, type) and issubclass(self.kind, BaseModel)


class Rubric(BaseModel):
    """
    Collection of evaluation traits applied to all question-answer pairs.

    A rubric defines the qualitative criteria used to evaluate LLM responses
    beyond basic correctness checking. Supports LLM-based, regex, callable, and metric traits.
    """

    llm_traits: list[LLMRubricTrait] = Field(default_factory=list, description="List of LLM-based evaluation traits")
    regex_traits: list[RegexRubricTrait] = Field(
        default_factory=list, description="List of regex-based evaluation traits"
    )
    callable_traits: list[CallableRubricTrait] = Field(
        default_factory=list, description="List of callable function-based evaluation traits"
    )
    metric_traits: list[MetricRubricTrait] = Field(
        default_factory=list, description="List of metric-based evaluation traits (confusion-matrix analysis)"
    )
    agentic_traits: list[AgenticRubricTrait] = Field(
        default_factory=list,
        description="List of agent-investigated evaluation traits",
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_trait_names(self) -> "Rubric":
        """Reject duplicate trait names (within and across types) and dots in agentic names.

        Each trait type list must have unique names. Cross-type name overlaps
        are also rejected because downstream consumers (DataFrames, result
        dicts) use trait names as keys without type prefixes.

        Dots in agentic trait names are rejected because template kind traits
        produce dot-expanded keys (``trait.field``). A trait named ``"foo.bar"``
        would be ambiguous.
        """
        type_lists: list[tuple[str, list[Any]]] = [
            ("llm", self.llm_traits),
            ("regex", self.regex_traits),
            ("callable", self.callable_traits),
            ("metric", self.metric_traits),
            ("agentic", self.agentic_traits),
        ]
        for type_label, traits in type_lists:
            seen: set[str] = set()
            for trait in traits:
                if trait.name in seen:
                    raise ValueError(
                        f"Duplicate {type_label} trait name '{trait.name}' "
                        f"within the same rubric. Trait names must be unique "
                        f"per type."
                    )
                seen.add(trait.name)

        # Cross-type uniqueness check
        all_names = self.get_trait_names()
        seen_all: set[str] = set()
        for name in all_names:
            if name in seen_all:
                raise ValueError(
                    f"Duplicate trait name '{name}' across different trait types. "
                    f"Trait names must be unique across all types within a rubric."
                )
            seen_all.add(name)

        for trait in self.agentic_traits:
            if "." in trait.name:
                raise ValueError(
                    f"Agentic trait name '{trait.name}' contains '.', "
                    f"which would collide with dot-notation keys from "
                    f"template-kind traits."
                )
        return self

    def get_trait_names(self) -> list[str]:
        """Get list of all trait names in this rubric (LLM, regex, callable, metric, and agentic)."""
        llm_names = [trait.name for trait in self.llm_traits]
        regex_names = [trait.name for trait in self.regex_traits]
        callable_names = [trait.name for trait in self.callable_traits]
        metric_names = [trait.name for trait in self.metric_traits]
        agentic_names = [trait.name for trait in self.agentic_traits]
        return llm_names + regex_names + callable_names + metric_names + agentic_names

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

    def get_agentic_trait_names(self) -> list[str]:
        """Get list of agentic trait names only."""
        return [trait.name for trait in self.agentic_traits]

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

        for agentic_trait in self.agentic_traits:
            if agentic_trait.kind in ("score", "literal") and agentic_trait.max_score is not None:
                max_scores[agentic_trait.name] = agentic_trait.max_score

        return max_scores

    def get_trait_directionalities(self) -> dict[str, bool | None]:
        """Get higher_is_better for LLM, regex, callable, and agentic traits.

        Note: MetricRubricTraits are excluded as metrics (precision/recall/F1)
        are inherently 'higher is better'.

        Returns:
            Dict mapping trait name to higher_is_better value. Template kind
            agentic traits map to None because directionality is not meaningful
            for structured results.
        """
        directionalities: dict[str, bool | None] = {}

        llm_trait: LLMRubricTrait
        for llm_trait in self.llm_traits:
            directionalities[llm_trait.name] = llm_trait.higher_is_better

        regex_trait: RegexRubricTrait
        for regex_trait in self.regex_traits:
            directionalities[regex_trait.name] = regex_trait.higher_is_better

        callable_trait: CallableRubricTrait
        for callable_trait in self.callable_traits:
            directionalities[callable_trait.name] = callable_trait.higher_is_better

        for agentic_trait in self.agentic_traits:
            directionalities[agentic_trait.name] = agentic_trait.higher_is_better

        # MetricRubricTraits always have higher_is_better=True (implicit)
        return directionalities

    def validate_evaluation(self, evaluation: dict[str, int | bool]) -> bool:
        """
        Validate that an evaluation result matches this rubric structure.

        Note: This validates LLM, regex, callable, and agentic trait scores. Metric traits
        are stored separately in VerificationResult fields (metric_trait_confusion_lists
        and metric_trait_metrics) and don't participate in this validation.

        Template kind agentic traits produce dot-expanded keys (e.g.
        ``"trait_name.field_name"``), so the expected names set and per-key
        validation logic account for this notation.
        """
        # Get trait names excluding metric traits (they're validated separately)
        llm_names = set(self.get_llm_trait_names())
        regex_names = set(self.get_regex_trait_names())
        callable_names = set(self.get_callable_trait_names())

        # For agentic traits, expand template kinds to dot-notation keys
        agentic_expected: set[str] = set()
        for trait in self.agentic_traits:
            if trait.is_template_kind:
                assert isinstance(trait.kind, type)  # narrows for mypy
                for field_name in trait.kind.model_fields:
                    agentic_expected.add(f"{trait.name}.{field_name}")
            else:
                agentic_expected.add(trait.name)

        expected_names = llm_names | regex_names | callable_names | agentic_expected

        eval_names = set(evaluation.keys())

        # Check that all expected trait names are present
        if expected_names != eval_names:
            return False

        # Check that each score is valid for its trait
        llm_trait_map = {trait.name: trait for trait in self.llm_traits}
        regex_trait_map = {trait.name: trait for trait in self.regex_traits}
        callable_trait_map = {trait.name: trait for trait in self.callable_traits}
        agentic_trait_map = {trait.name: trait for trait in self.agentic_traits}

        for key, value in evaluation.items():
            trait_name = key.split(".")[0] if "." in key else key
            if trait_name in llm_trait_map:
                if not llm_trait_map[trait_name].validate_score(value):
                    return False
            elif trait_name in agentic_trait_map:
                trait = agentic_trait_map[trait_name]
                if trait.is_template_kind:
                    continue  # Template fields not individually validated
                if not trait.validate_score(value):
                    return False
            elif trait_name in regex_trait_map or trait_name in callable_trait_map:
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


def _iter_traits(
    rubric: "Rubric",
) -> "list[LLMRubricTrait | RegexRubricTrait | CallableRubricTrait | MetricRubricTrait | AgenticRubricTrait]":
    """Return all traits from a rubric in a flat list.

    Args:
        rubric: The rubric to iterate over.

    Returns:
        All traits across llm, regex, callable, metric, and agentic types.
    """
    return (
        list(rubric.llm_traits)
        + list(rubric.regex_traits)
        + list(rubric.callable_traits)
        + list(rubric.metric_traits)
        + list(rubric.agentic_traits)
    )


def merge_rubrics(
    global_rubric: "Rubric | None",
    question_rubric: "Rubric | None",
) -> "tuple[Rubric | None, dict[str, str] | None]":
    """Merge global and question-specific rubrics.

    Same-type trait name collisions (e.g., both rubrics have an LLM trait
    named "safety") raise ``ValueError``. Cross-type collisions (e.g., global
    regex trait "quality" + question LLM trait "quality") are rejected by the
    ``Rubric`` constructor's cross-type uniqueness validation.

    Args:
        global_rubric: The global rubric (applied to all questions).
        question_rubric: Question-specific rubric (adds to global).

    Returns:
        A tuple of (merged_rubric, provenance) where provenance maps each
        trait name to its source: "global" or "question_specific". Both
        elements are None when both inputs are None.

    Raises:
        ValueError: If a trait name appears in both rubrics within the same
            trait type.
    """
    if not global_rubric and not question_rubric:
        return None, None

    if not global_rubric:
        assert question_rubric is not None
        q_provenance: dict[str, str] = {t.name: "question_specific" for t in _iter_traits(question_rubric)}
        return question_rubric, q_provenance

    if not question_rubric:
        g_provenance: dict[str, str] = {t.name: "global" for t in _iter_traits(global_rubric)}
        return global_rubric, g_provenance

    # Check per-type name collisions
    type_pairs: list[tuple[str, list[Any], list[Any]]] = [
        ("llm", global_rubric.llm_traits, question_rubric.llm_traits),
        ("regex", global_rubric.regex_traits, question_rubric.regex_traits),
        ("callable", global_rubric.callable_traits, question_rubric.callable_traits),
        ("metric", global_rubric.metric_traits, question_rubric.metric_traits),
        ("agentic", global_rubric.agentic_traits, question_rubric.agentic_traits),
    ]
    all_conflicts: list[str] = []
    for type_label, g_traits, q_traits in type_pairs:
        g_names = {t.name for t in g_traits}
        q_names = {t.name for t in q_traits}
        overlap = g_names & q_names
        if overlap:
            all_conflicts.extend(f"{type_label}:{name}" for name in sorted(overlap))

    if all_conflicts:
        raise ValueError(f"Same-type trait name conflicts between global and question rubrics: {all_conflicts}")

    merged = Rubric(
        llm_traits=list(global_rubric.llm_traits) + list(question_rubric.llm_traits),
        regex_traits=list(global_rubric.regex_traits) + list(question_rubric.regex_traits),
        callable_traits=list(global_rubric.callable_traits) + list(question_rubric.callable_traits),
        metric_traits=list(global_rubric.metric_traits) + list(question_rubric.metric_traits),
        agentic_traits=list(global_rubric.agentic_traits) + list(question_rubric.agentic_traits),
    )
    provenance: dict[str, str] = {}
    for t in _iter_traits(global_rubric):
        provenance[t.name] = "global"
    for t in _iter_traits(question_rubric):
        provenance[t.name] = "question_specific"
    return merged, provenance


# Type alias for the union of all trait types stored in DynamicRubric
_AnyTrait = LLMRubricTrait | RegexRubricTrait | CallableRubricTrait | MetricRubricTrait | AgenticRubricTrait


class DynamicRubric(BaseModel):
    """Rubric whose traits are conditionally evaluated based on concept presence.

    Unlike a regular Rubric (evaluated unconditionally), a DynamicRubric gates
    each trait on whether its concept is detected in the response. Every trait
    must carry either a ``summary`` or ``description`` so that the presence
    check prompt can describe the concept to the judge LLM.
    """

    model_config = ConfigDict(extra="forbid")

    llm_traits: list[LLMRubricTrait] = Field(default_factory=list)
    regex_traits: list[RegexRubricTrait] = Field(default_factory=list)
    callable_traits: list[CallableRubricTrait] = Field(default_factory=list)
    metric_traits: list[MetricRubricTrait] = Field(default_factory=list)
    agentic_traits: list[AgenticRubricTrait] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_trait_names(self) -> "DynamicRubric":
        """Reject duplicate trait names within and across types.

        Mirrors ``Rubric.validate_trait_names``. Both same-type and
        cross-type duplicates are rejected.
        """
        type_lists: list[tuple[str, list[Any]]] = [
            ("llm", self.llm_traits),
            ("regex", self.regex_traits),
            ("callable", self.callable_traits),
            ("metric", self.metric_traits),
            ("agentic", self.agentic_traits),
        ]
        for type_label, traits in type_lists:
            seen: set[str] = set()
            for trait in traits:
                if trait.name in seen:
                    raise ValueError(
                        f"Duplicate {type_label} trait name '{trait.name}' "
                        f"within the same dynamic rubric. Trait names must be "
                        f"unique per type."
                    )
                seen.add(trait.name)

        # Cross-type uniqueness check
        all_names = self.get_trait_names()
        seen_all: set[str] = set()
        for name in all_names:
            if name in seen_all:
                raise ValueError(
                    f"Duplicate trait name '{name}' across different trait types. "
                    f"Trait names must be unique across all types within a dynamic rubric."
                )
            seen_all.add(name)
        return self

    @model_validator(mode="after")
    def validate_concept_text(self) -> "DynamicRubric":
        """Ensure every trait has text usable for concept presence checking.

        Each trait must have at least one of ``summary`` or ``description``.
        If ``summary`` is None but ``description`` exists, a warning is logged
        because ``summary`` is the preferred short label for the presence check
        prompt. If both are None, the trait cannot participate in presence
        checking and a ``ValueError`` is raised.
        """
        for trait in self._all_traits():
            has_summary = getattr(trait, "summary", None) is not None
            has_description = getattr(trait, "description", None) is not None

            if not has_summary and not has_description:
                raise ValueError(
                    f"Dynamic rubric trait '{trait.name}' has neither summary nor description. "
                    "At least one is required for concept presence checking."
                )
            if not has_summary and has_description:
                logger.warning(
                    "Dynamic rubric trait '%s' has no summary; falling back to description "
                    "for concept presence text. Consider adding a short summary.",
                    trait.name,
                )
        return self

    def _all_traits(self) -> list[_AnyTrait]:
        """Return a flat list of all traits across every type."""
        result: list[_AnyTrait] = []
        result.extend(self.llm_traits)
        result.extend(self.regex_traits)
        result.extend(self.callable_traits)
        result.extend(self.metric_traits)
        result.extend(self.agentic_traits)
        return result

    def get_trait_names(self) -> list[str]:
        """Return names of all traits in type order: llm, regex, callable, metric, agentic."""
        return [trait.name for trait in self._all_traits()]

    def is_empty(self) -> bool:
        """Return True if this dynamic rubric contains no traits."""
        return len(self._all_traits()) == 0

    def resolve_concept_text(self, trait: _AnyTrait) -> str:
        """Return the text to use for concept presence checking.

        Prefers ``summary`` when set; falls back to ``description``.

        Args:
            trait: A trait instance from this dynamic rubric.

        Returns:
            The concept text string (summary or description).
        """
        summary = getattr(trait, "summary", None)
        if summary is not None:
            return str(summary)
        description = getattr(trait, "description", None)
        if description is not None:
            return str(description)
        # Should not happen if validation passed, but guard defensively
        return trait.name


def merge_dynamic_rubrics(
    global_dynamic: "DynamicRubric | None",
    question_dynamic: "DynamicRubric | None",
) -> "DynamicRubric | None":
    """Merge global and question-specific dynamic rubrics.

    Mirrors :func:`merge_rubrics` for the dynamic rubric variant. Same-type
    name collisions are rejected; cross-type overlaps are allowed.

    Args:
        global_dynamic: The global dynamic rubric (applied to all questions).
        question_dynamic: Question-specific dynamic rubric.

    Returns:
        Merged DynamicRubric with traits from both sources, or None if both are None.

    Raises:
        ValueError: If a trait name appears in both rubrics within the same
            trait type.
    """
    if not global_dynamic and not question_dynamic:
        return None

    if not global_dynamic:
        return question_dynamic

    if not question_dynamic:
        return global_dynamic

    type_pairs: list[tuple[str, list[Any], list[Any]]] = [
        ("llm", global_dynamic.llm_traits, question_dynamic.llm_traits),
        ("regex", global_dynamic.regex_traits, question_dynamic.regex_traits),
        ("callable", global_dynamic.callable_traits, question_dynamic.callable_traits),
        ("metric", global_dynamic.metric_traits, question_dynamic.metric_traits),
        ("agentic", global_dynamic.agentic_traits, question_dynamic.agentic_traits),
    ]
    all_conflicts: list[str] = []
    for type_label, g_traits, q_traits in type_pairs:
        g_names = {t.name for t in g_traits}
        q_names = {t.name for t in q_traits}
        overlap = g_names & q_names
        if overlap:
            all_conflicts.extend(f"{type_label}:{name}" for name in sorted(overlap))

    if all_conflicts:
        raise ValueError(f"Same-type trait name conflicts between global and question dynamic rubrics: {all_conflicts}")

    return DynamicRubric(
        llm_traits=list(global_dynamic.llm_traits) + list(question_dynamic.llm_traits),
        regex_traits=list(global_dynamic.regex_traits) + list(question_dynamic.regex_traits),
        callable_traits=list(global_dynamic.callable_traits) + list(question_dynamic.callable_traits),
        metric_traits=list(global_dynamic.metric_traits) + list(question_dynamic.metric_traits),
        agentic_traits=list(global_dynamic.agentic_traits) + list(question_dynamic.agentic_traits),
    )
