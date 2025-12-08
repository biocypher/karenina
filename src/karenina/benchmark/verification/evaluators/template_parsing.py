"""Template-specific parsing utilities.

This module provides parsing utilities for the TemplateEvaluator, consolidated
from utils/parsing.py for better encapsulation with the evaluator pattern.

These functions handle:
- Native structured output invocation
- Multi-strategy JSON parsing with fallbacks
- Markdown fence stripping
- JSON extraction from mixed text
"""

import json
import logging
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def invoke_with_structured_output_for_template(
    llm: Any,
    messages: list[Any],  # BaseMessage or dict
    answer_class: type[T],
) -> tuple[T | None, dict[str, Any], bool]:
    """
    Invoke LLM with structured output for template parsing.

    Uses a two-strategy approach:
    1. json_schema method (native structured output) - for providers that support it
    2. Returns None if structured output fails (caller should use fallback parsing)

    Unlike rubric parsing which always returns a result, this returns None on failure
    to allow the caller to fall back to enhanced PydanticOutputParser flow with retries.

    Args:
        llm: LangChain chat model
        messages: List of messages to send
        answer_class: Pydantic model (user-defined Answer class) for structured output

    Returns:
        Tuple of (parsed_result_or_none, usage_metadata, used_structured_output)
        - parsed_result_or_none: Parsed Answer instance, or None if structured output failed
        - usage_metadata: Token usage from the LLM call
        - used_structured_output: True if native structured output succeeded
    """
    from langchain_core.callbacks import get_usage_metadata_callback

    usage_metadata: dict[str, Any] = {}

    # Try json_schema method (native structured output)
    try:
        structured_llm = llm.with_structured_output(answer_class, method="json_schema")

        with get_usage_metadata_callback() as cb:
            result = structured_llm.invoke(messages)

        usage_metadata = dict(cb.usage_metadata) if cb.usage_metadata else {}

        if isinstance(result, answer_class):
            logger.debug(f"json_schema method succeeded for template {answer_class.__name__}")
            return result, usage_metadata, True

    except Exception as e:
        logger.debug(f"json_schema method failed for template: {e}")

    # Return None to signal caller should use fallback parsing
    return None, usage_metadata, False


def parse_template_response(response: str | Any, answer_class: type[T]) -> T:
    """
    Parse raw LLM response into Answer class with multiple fallback strategies.

    Strategy order:
    1. Already a model instance (pass through)
    2. Direct JSON parsing
    3. JSON extraction from mixed text (markdown fences, etc.)
    4. JSON repair with jsonrepair

    Args:
        response: Raw string response from LLM (or already parsed object)
        answer_class: Pydantic model (user-defined Answer class) to validate against

    Returns:
        Validated Pydantic model instance

    Raises:
        ValueError: If all parsing strategies fail
    """
    # Already a model instance
    if isinstance(response, answer_class):
        return response

    # Convert to string if needed
    if not isinstance(response, str):
        response = str(response)

    # Strategy 1: Direct JSON parsing
    try:
        data = json.loads(response.strip())
        return answer_class.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as e:
        logger.debug(f"Direct JSON parse failed for template: {e}")

    # Strategy 2: Extract JSON from mixed text (handles markdown fences)
    cleaned = _strip_markdown_fences(response)
    if cleaned and cleaned != response:
        # Fences were stripped, try parsing the cleaned text
        try:
            data = json.loads(cleaned)
            return answer_class.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.debug(f"Cleaned text parse failed for template: {e}")

    # Try extracting JSON object from text
    json_str = _extract_json_from_text(response)
    if json_str:
        try:
            data = json.loads(json_str)
            return answer_class.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.debug(f"Extracted JSON parse failed for template: {e}")

    # Strategy 3: JSON repair for malformed JSON
    try:
        from json_repair import repair_json

        repaired = repair_json(response)
        data = json.loads(repaired)
        logger.debug(f"JSON repair succeeded for template {answer_class.__name__}")
        return answer_class.model_validate(data)
    except ImportError:
        logger.warning("json-repair not installed, skipping repair strategy")
    except Exception as e:
        logger.debug(f"JSON repair failed for template: {e}")

    # Strategy 4: Try repair on cleaned text
    if cleaned and cleaned != response:
        try:
            from json_repair import repair_json

            repaired = repair_json(cleaned)
            data = json.loads(repaired)
            logger.debug(f"JSON repair on cleaned text succeeded for template {answer_class.__name__}")
            return answer_class.model_validate(data)
        except Exception as e:
            logger.debug(f"JSON repair on cleaned text failed for template: {e}")

    # All strategies failed
    preview = response[:200] if len(response) > 200 else response
    raise ValueError(f"Could not parse response into {answer_class.__name__}: {preview}")


def _strip_markdown_fences(text: str | None) -> str | None:
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
        >>> _strip_markdown_fences("```json\\n{...}\\n```")
        "{...}"
        >>> _strip_markdown_fences("The answer is {\"field\": \"value\"}")
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
    extracted_json = _extract_json_from_text(text)
    if extracted_json:
        return extracted_json

    # Strategy 3: Handle partial fences - try to remove opening fence
    text = re.sub(r"^```(?:\w+)?\s*\n?", "", text)
    # Handle closing fence
    text = re.sub(r"\n?```$", "", text)

    return text.strip()


def _extract_json_from_text(text: str) -> str | None:
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
        >>> _extract_json_from_text('The answer is {"field": "value"} as shown.')
        '{"field": "value"}'
        >>> _extract_json_from_text('Processing... Output: {"a": 1, "b": {"c": 2}}')
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
            json_str = _extract_balanced_braces(text, i)
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


def _extract_balanced_braces(text: str, start: int) -> str | None:
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
