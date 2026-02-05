"""Helper functions for LLM-as-judge evaluation patterns.

This module provides shared utilities for abstention detection, sufficiency checking,
and other LLM-based evaluation tasks that follow similar patterns:
1. Invoke LLM with structured output schema
2. Extract typed result from response
3. Fall back to JSON parsing if structured output fails

These helpers reduce code duplication between trace_abstention_checker.py
and trace_sufficiency_checker.py.
"""

import json
import logging
from typing import Any, TypeVar

from pydantic import BaseModel

from karenina.ports import LLMResponse
from karenina.utils.json_extraction import strip_markdown_fences

logger = logging.getLogger(__name__)

# TypeVar for generic result types (AbstentionResult, SufficiencyResult, etc.)
T = TypeVar("T", bound=BaseModel)


def extract_judge_result(
    response: LLMResponse,
    result_type: type[T],
    primary_field: str,
) -> T | None:
    """Extract a typed result from an LLMResponse.

    Checks if the raw response is already the expected type, or if it has
    the expected field and can be converted.

    Args:
        response: The LLMResponse from structured output invocation
        result_type: The Pydantic model class to extract (e.g., AbstentionResult)
        primary_field: The main field to check for (e.g., "abstention_detected")

    Returns:
        The extracted result, or None if extraction failed

    Example:
        >>> result = extract_judge_result(response, AbstentionResult, "abstention_detected")
        >>> if result is not None:
        ...     print(result.abstention_detected, result.reasoning)
    """
    # Best case: raw is already the correct type
    if isinstance(response.raw, result_type):
        return response.raw

    # Fallback: raw has the expected field, construct the result
    if hasattr(response.raw, primary_field):
        return result_type(
            **{
                primary_field: getattr(response.raw, primary_field),
                "reasoning": getattr(response.raw, "reasoning", "No reasoning provided"),
            }
        )

    return None


def fallback_json_parse(
    content: str,
    usage_metadata: dict[str, Any],
    primary_field: str,
    default_value: bool,
    log_prefix: str,
) -> tuple[bool, bool, str | None, dict[str, Any]]:
    """Fallback to manual JSON parsing when structured output fails.

    Parses the content as JSON and extracts the primary field and reasoning.
    Uses strip_markdown_fences to handle ```json wrapped responses.

    Args:
        content: The raw content string from LLM response
        usage_metadata: Token usage metadata dict to pass through
        primary_field: The field to extract (e.g., "abstention_detected", "sufficient")
        default_value: Default value if field is missing or parsing fails
        log_prefix: Prefix for debug logging (e.g., "Abstention check (fallback)")

    Returns:
        Tuple of (field_value, check_performed, reasoning, usage_metadata):
        - field_value: The extracted boolean value or default
        - check_performed: True if parsing succeeded, False otherwise
        - reasoning: The extracted reasoning or None on failure
        - usage_metadata: The passed-through usage metadata

    Example:
        >>> value, performed, reasoning, metadata = fallback_json_parse(
        ...     content='{"abstention_detected": true, "reasoning": "..."}',
        ...     usage_metadata={},
        ...     primary_field="abstention_detected",
        ...     default_value=False,
        ...     log_prefix="Abstention check (fallback)",
        ... )
    """
    try:
        cleaned_response = strip_markdown_fences(content)
        if cleaned_response is None:
            return default_value, False, None, usage_metadata

        result = json.loads(cleaned_response)
        field_value = result.get(primary_field, default_value)
        reasoning = result.get("reasoning", "No reasoning provided")
        logger.debug(f"{log_prefix}: {field_value}")
        return field_value, True, reasoning, usage_metadata

    except json.JSONDecodeError:
        return default_value, False, None, usage_metadata
