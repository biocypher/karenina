"""JSON extraction utilities for LLM responses.

This module provides functions for extracting JSON content from LLM responses
that may be wrapped in markdown code blocks or mixed with other text.

Module Organization:
    - JSON Extraction: extract_json_from_response for pulling JSON from mixed text
    - Error Detection: is_invalid_json_error for classifying parsing failures
"""

import re

__all__ = ["extract_json_from_response", "is_invalid_json_error"]


def extract_json_from_response(text: str) -> str:
    """Extract JSON from a response that may be wrapped in markdown or mixed with text.

    Attempts multiple extraction strategies in order of preference:
    1. Direct JSON (starts with { or [)
    2. Markdown code blocks (```json ... ``` or ``` ... ```)
    3. JSON-like content search (finds { ... } pattern)

    Args:
        text: Raw response text that may contain JSON.

    Returns:
        Extracted JSON string, stripped of surrounding whitespace.

    Raises:
        ValueError: If no valid JSON can be extracted from the response.

    Example:
        >>> extract_json_from_response('{"key": "value"}')
        '{"key": "value"}'
        >>> extract_json_from_response('```json\\n{"key": "value"}\\n```')
        '{"key": "value"}'
        >>> extract_json_from_response('Here is the result: {"key": "value"}')
        '{"key": "value"}'
    """
    # Try direct parsing first
    text = text.strip()
    if text.startswith("{") or text.startswith("["):
        return text

    # Try to extract from markdown code blocks
    # Match ```json ... ``` or ``` ... ```
    code_block_pattern = r"```(?:json)?\s*\n?([\s\S]*?)\n?```"
    matches: list[str] = re.findall(code_block_pattern, text)
    if matches:
        for match in matches:
            extracted = match.strip()
            if extracted.startswith("{") or extracted.startswith("["):
                return extracted

    # Last resort: find JSON-like content
    json_pattern = r"\{[\s\S]*\}"
    json_match = re.search(json_pattern, text)
    if json_match:
        return json_match.group()

    raise ValueError(f"Could not extract JSON from response: {text[:200]}...")


def is_invalid_json_error(error: Exception) -> bool:
    """Check if an error is related to invalid JSON output.

    Used by parser adapters to detect JSON-format errors and trigger
    appropriate retry strategies (e.g., format feedback).

    Args:
        error: The exception from a parsing attempt.

    Returns:
        True if this is an invalid JSON error.

    Example:
        >>> try:
        ...     json.loads("not json")
        ... except json.JSONDecodeError as e:
        ...     assert is_invalid_json_error(e) is True
    """
    error_str = str(error).lower()
    error_type = type(error).__name__

    json_error_patterns = [
        "invalid json",
        "json decode",
        "jsondecodeerror",
        "expecting value",
        "expecting property name",
        "unterminated string",
        "extra data",
        "invalid control character",
        "invalid \\escape",
        "invalid literal",
        "no json object could be decoded",
        "output_parsing_failure",
    ]

    if any(pattern in error_str for pattern in json_error_patterns):
        return True

    return error_type in ["JSONDecodeError", "OutputParserException"]
