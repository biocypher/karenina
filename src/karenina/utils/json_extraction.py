"""JSON extraction and parsing utilities for LLM responses.

This module provides functions for extracting JSON from LLM responses that
may contain markdown fences, reasoning text, or other non-JSON content.

Functions:
    strip_markdown_fences: Remove markdown code fences and extract JSON from text
    extract_json_from_text: Extract JSON objects from mixed text content
    extract_balanced_braces: Extract balanced brace expressions from text
    extract_json_from_response: Alias for extract_json_from_text (backwards compat)
    is_invalid_json_error: Check if an error is related to invalid JSON output
"""

import json
import logging
import re

logger = logging.getLogger(__name__)

__all__ = [
    "strip_markdown_fences",
    "extract_json_from_text",
    "extract_balanced_braces",
    "extract_json_from_response",
    "is_invalid_json_error",
]


def strip_markdown_fences(text: str | None) -> str | None:
    """Remove markdown code fences from text and extract JSON from mixed content.

    Handles multiple extraction strategies in order:
    1. Triple backtick fences with optional language tags (```json ... ```)
    2. JSON objects embedded in reasoning/explanation text
    3. Partial fences (only opening or only closing)

    The function is designed to handle cases where LLMs output reasoning text
    before/after the actual JSON response, such as:
    "Let me analyze this... the answer is { \"field\": \"value\" }"

    Args:
        text: Raw text potentially containing markdown fences or mixed content
              (can be None or non-string)

    Returns:
        Extracted JSON string, text with markdown fences removed,
        or original value if not a string

    Example:
        >>> strip_markdown_fences("```json\\n{...}\\n```")
        "{...}"
        >>> strip_markdown_fences("The answer is {\"field\": \"value\"}")
        '{"field": "value"}'
    """
    # Handle non-string inputs
    if not isinstance(text, str):
        return text

    # Strategy 1: Pattern matches: ```optional_language\nCONTENT\n```
    pattern = r"```(?:\w+)?\s*\n?(.*?)\n?```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Strategy 2: Try to extract JSON object from mixed text
    # This handles cases where LLM outputs reasoning + JSON
    extracted_json = extract_json_from_text(text)
    if extracted_json:
        return extracted_json

    # Strategy 3: Handle partial fences - try to remove opening fence
    text = re.sub(r"^```(?:\w+)?\s*\n?", "", text)
    # Handle closing fence
    text = re.sub(r"\n?```$", "", text)

    return text.strip()


def extract_json_from_text(text: str) -> str | None:
    """Extract a JSON object from text that may contain reasoning/explanation.

    Tries multiple strategies to find valid JSON:
    1. Find last JSON object (LLMs often reason first, then output JSON)
    2. Find first JSON object (fallback)
    3. Handle nested braces properly

    Args:
        text: Text that may contain JSON mixed with other content

    Returns:
        Extracted JSON string if found and valid, None otherwise

    Example:
        >>> extract_json_from_text('The answer is {"field": "value"} as shown.')
        '{"field": "value"}'
        >>> extract_json_from_text('Processing... Output: {"a": 1, "b": {"c": 2}}')
        '{"a": 1, "b": {"c": 2}}'
    """
    # Find all potential JSON object boundaries
    # We need to handle nested braces, so we can't use simple regex
    json_candidates: list[str] = []

    # Find all positions where JSON objects might start
    i = 0
    while i < len(text):
        if text[i] == "{":
            # Try to find matching closing brace
            json_str = extract_balanced_braces(text, i)
            if json_str:
                json_candidates.append(json_str)
                i += len(json_str)
            else:
                i += 1
        else:
            i += 1

    if not json_candidates:
        return None

    # Try candidates from last to first (LLMs often output JSON at the end)
    for candidate in reversed(json_candidates):
        try:
            # Validate it's actually valid JSON
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue

    return None


def extract_balanced_braces(text: str, start: int) -> str | None:
    """Extract a balanced brace expression from text starting at given position.

    Properly handles:
    - Nested braces: {"a": {"b": 1}}
    - Strings containing braces: {"text": "has { and }"}
    - Escaped quotes in strings: {"text": "say \\"hello\\""}

    Args:
        text: The full text
        start: Position of opening brace

    Returns:
        The balanced brace expression if found, None otherwise
    """
    if start >= len(text) or text[start] != "{":
        return None

    depth = 0
    in_string = False
    escape_next = False
    i = start

    while i < len(text):
        char = text[i]

        if escape_next:
            escape_next = False
            i += 1
            continue

        if char == "\\":
            escape_next = True
            i += 1
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            i += 1
            continue

        if not in_string:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

        i += 1

    return None  # Unbalanced braces


def extract_json_from_response(text: str) -> str:
    """Extract JSON from a response that may be wrapped in markdown or mixed with text.

    This function wraps extract_json_from_text with error handling for backwards
    compatibility. Prefer using extract_json_from_text directly for new code.

    Attempts multiple extraction strategies in order of preference:
    1. Direct JSON (starts with { or [)
    2. Markdown code blocks (```json ... ``` or ``` ... ```)
    3. JSON-like content search with balanced brace parsing

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
    text = text.strip()

    # Try direct parsing first: if it starts with { or [, validate before returning
    if text.startswith("{") or text.startswith("["):
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass  # Invalid JSON, fall through to other extraction strategies

    # Try to extract from markdown code blocks
    code_block_pattern = r"```(?:json)?\s*\n?([\s\S]*?)\n?```"
    matches: list[str] = re.findall(code_block_pattern, text)
    if matches:
        for match in matches:
            extracted = match.strip()
            if extracted.startswith("{") or extracted.startswith("["):
                return extracted

    # Use balanced brace extraction for mixed text
    result = extract_json_from_text(text)
    if result:
        return result

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
