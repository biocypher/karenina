"""Result dataclasses for template evaluation.

This module provides dataclasses for capturing results from template parsing
and verification operations.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParseResult:
    """Result of template parsing operation."""

    parsed_answer: Any | None = None
    """Parsed Pydantic object (Answer instance)."""

    success: bool = False
    """Whether parsing succeeded."""

    error: str | None = None
    """Error message if parsing failed."""

    # Deep judgment metadata
    deep_judgment_performed: bool = False
    """Whether deep judgment was used for parsing."""

    extracted_excerpts: dict[str, list[dict[str, Any]]] | None = None
    """Excerpts extracted per attribute (if deep judgment enabled)."""

    attribute_reasoning: dict[str, str] | None = None
    """Reasoning traces per attribute (if deep judgment enabled)."""

    deep_judgment_stages_completed: list[str] | None = None
    """List of completed deep judgment stages."""

    deep_judgment_model_calls: int = 0
    """Number of LLM calls made during deep judgment."""

    deep_judgment_excerpt_retry_count: int = 0
    """Number of excerpt extraction retries."""

    attributes_without_excerpts: list[str] | None = None
    """Attributes that failed excerpt extraction."""

    hallucination_risk_assessment: dict[str, Any] | None = None
    """Hallucination risk per attribute (if search enabled)."""

    usage_metadata_list: list[dict[str, Any]] = field(default_factory=list)
    """Usage metadata from LLM calls."""


@dataclass
class FieldVerificationResult:
    """Result of field verification."""

    success: bool = False
    """Whether all fields verified successfully."""

    error: str | None = None
    """Error message if verification failed."""


@dataclass
class RegexVerificationResult:
    """Result of regex verification."""

    success: bool = False
    """Whether all regex patterns matched."""

    results: dict[str, bool] = field(default_factory=dict)
    """Per-field regex results."""

    details: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Detailed regex match information per field."""

    extraction_results: dict[str, list[str]] = field(default_factory=dict)
    """Actual regex matches extracted per field."""

    error: str | None = None
    """Error message if verification failed."""
