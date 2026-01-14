"""Shared utility functions for verification operations.

This module consolidates commonly used utility functions that were previously
duplicated across multiple files in the verification system.

Functions:
    strip_markdown_fences: Remove markdown code fences and extract JSON from text
    extract_json_from_text: Extract JSON objects from mixed text content
    extract_balanced_braces: Extract balanced brace expressions from text
    is_retryable_error: Check if an exception is a transient/retryable error
    is_openai_endpoint_llm: Check if LLM is a custom OpenAI-compatible endpoint
    parse_tool_output: Parse search tool output into SearchResultItem list
"""

import json
import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ....schemas import SearchResultItem

logger = logging.getLogger(__name__)

__all__ = [
    "strip_markdown_fences",
    "extract_json_from_text",
    "extract_balanced_braces",
    "is_retryable_error",
    "is_openai_endpoint_llm",
    "parse_tool_output",
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


def is_retryable_error(exception: Exception) -> bool:
    """Check if an exception is retryable (transient error).

    Used by abstention and sufficiency checkers to determine whether
    to retry LLM calls after failures.

    Args:
        exception: The exception to check

    Returns:
        True if the error is transient and should be retried, False otherwise

    Example:
        >>> is_retryable_error(ConnectionError("timeout"))
        True
        >>> is_retryable_error(ValueError("invalid input"))
        False
    """
    exception_str = str(exception).lower()
    exception_type = type(exception).__name__

    # Connection-related errors (check error message content)
    if any(
        keyword in exception_str
        for keyword in [
            "connection",
            "timeout",
            "timed out",
            "rate limit",
            "429",
            "503",
            "502",
            "500",
            "network",
            "temporary failure",
        ]
    ):
        return True

    # Common retryable exception types (check exception class name)
    retryable_types = [
        "ConnectionError",
        "TimeoutError",
        "HTTPError",
        "ReadTimeout",
        "ConnectTimeout",
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
    ]

    return exception_type in retryable_types


def is_openai_endpoint_llm(llm: Any) -> bool:
    """Check if the LLM is a ChatOpenAIEndpoint (custom OpenAI-compatible endpoint).

    These endpoints often don't support native structured output (json_schema method)
    and can hang indefinitely when attempting to use it. This function helps callers
    decide whether to skip structured output attempts.

    Detection methods:
    1. Class name is "ChatOpenAIEndpoint"
    2. Class hierarchy includes ChatOpenAIEndpoint
    3. Module path suggests interface wrapper with ChatOpenAI
    4. Has custom base_url not pointing to api.openai.com

    Args:
        llm: LangChain chat model instance

    Returns:
        True if the LLM appears to be an OpenAI-compatible endpoint, False otherwise

    Example:
        >>> if is_openai_endpoint_llm(llm):
        ...     # Skip json_schema method, use fallback parsing
        ...     pass
    """
    # Check by class name to avoid circular imports
    llm_class_name = type(llm).__name__
    # Also check the module path for more robust detection
    llm_module = type(llm).__module__

    is_endpoint = (
        llm_class_name == "ChatOpenAIEndpoint"
        or "ChatOpenAIEndpoint" in str(type(llm).__mro__)
        or (llm_module and "interface" in llm_module and llm_class_name == "ChatOpenAI")
    )

    # Also check if it has a custom base_url that's not OpenAI's
    if hasattr(llm, "openai_api_base") and llm.openai_api_base:
        base_url = str(llm.openai_api_base)
        if base_url and not base_url.startswith("https://api.openai.com"):
            is_endpoint = True

    return bool(is_endpoint)


def parse_tool_output(raw_result: Any) -> list["SearchResultItem"]:
    """Parse raw tool output into list of SearchResultItem objects.

    This function handles multiple output formats:
    1. List of SearchResultItem objects (already structured)
    2. List of dicts with title/content/url keys
    3. JSON string containing list of dicts
    4. Plain string (creates single item with generic title)

    Args:
        raw_result: Raw output from search tool

    Returns:
        List of SearchResultItem objects

    Note:
        Returns empty list on failure rather than raising.
    """
    # Import here to avoid circular imports
    from ....schemas import SearchResultItem

    # Case 1: Already a list of SearchResultItem
    if isinstance(raw_result, list) and all(isinstance(item, SearchResultItem) for item in raw_result):
        return raw_result

    # Case 2: List of dicts
    if isinstance(raw_result, list) and all(isinstance(item, dict) for item in raw_result):
        items = []
        for item_dict in raw_result:
            try:
                # Handle optional title and url fields
                title = item_dict.get("title") or None  # Convert empty string to None
                content = item_dict.get("content", "No content")
                url = item_dict.get("url") or None  # Convert empty string to None

                # Skip items with no content
                if not content or content == "No content":
                    logger.warning("Skipping search result with no content")
                    continue

                item = SearchResultItem(
                    title=title,
                    content=content,
                    url=url,
                )
                items.append(item)
            except Exception as e:
                logger.warning(f"Failed to parse dict to SearchResultItem: {e}")
                continue
        return items

    # Case 3: JSON string
    if isinstance(raw_result, str):
        try:
            # Try to parse as JSON
            parsed = json.loads(raw_result)
            if isinstance(parsed, list):
                return parse_tool_output(parsed)  # Recursive call with parsed list
        except json.JSONDecodeError:
            # Not JSON - treat as plain text
            pass

        # Plain text fallback - create single generic item (no title, no URL)
        logger.info("Search tool returned plain text, wrapping in SearchResultItem")
        return [
            SearchResultItem(
                title=None,  # Will use truncated content in GUI
                content=raw_result.strip(),
                url=None,
            )
        ]

    # Case 4: Unknown format
    logger.warning(f"Unknown search result format: {type(raw_result)}")
    return []
