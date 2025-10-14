"""Shared utility functions for parsing operations."""

import logging
import re
from typing import Any

from ...schemas.answer_class import BaseAnswer

logger = logging.getLogger(__name__)


def _extract_attribute_names_from_class(answer_class: type[BaseAnswer]) -> list[str]:
    """Extract attribute names from Answer class, excluding configuration fields.

    This function extracts all field names from a Pydantic BaseAnswer subclass,
    excluding special configuration fields that are not part of the actual answer content.

    Args:
        answer_class: A Pydantic BaseAnswer subclass with model_fields

    Returns:
        List of attribute names (strings) excluding: 'id', 'correct', 'regex'

    Excluded fields:
        - 'id': Question ID (metadata, not an answer field)
        - 'correct': Ground truth indicator (not extracted from LLM response)
        - 'regex': Regex validation configuration (not an answer field)

    Example:
        >>> class MyAnswer(BaseAnswer):
        ...     id: str
        ...     correct: str
        ...     drug_target: str
        ...     mechanism: str
        ...     confidence: str
        >>> _extract_attribute_names_from_class(MyAnswer)
        ['drug_target', 'mechanism', 'confidence']
    """
    # Get model fields from Pydantic v2
    if hasattr(answer_class, "model_fields"):
        model_fields = answer_class.model_fields
        field_names = list(model_fields.keys())
    else:
        # Fallback for older Pydantic versions
        fields = answer_class.__fields__
        field_names = list(fields.keys())  # type: ignore[attr-defined]

    # Exclude configuration fields that aren't part of the answer content
    # 'id': Question ID (metadata)
    # 'correct': Ground truth field (not extracted from LLM response)
    # 'regex': Regex validation configuration (not an answer field)
    return [name for name in field_names if name not in ("id", "correct", "regex")]


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from text.

    Handles both triple backtick fences with optional language tags:
    - ```json ... ```
    - ``` ... ```
    - ```\n ... \n```

    Args:
        text: Raw text potentially containing markdown fences

    Returns:
        Text with markdown fences removed

    Example:
        >>> _strip_markdown_fences("```json\\n{...}\\n```")
        "{...}"
    """
    # Pattern matches: ```optional_language\nCONTENT\n```
    pattern = r"```(?:\w+)?\s*\n?(.*?)\n?```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _extract_attribute_descriptions(json_schema: str, attribute_names: list[str]) -> dict[str, str]:
    """Extract attribute descriptions from JSON schema format instructions.

    Parses the JSON schema string to extract description fields for each attribute.
    This is used to provide guidance to the LLM about what evidence to look for
    without requiring it to follow the full schema format.

    Args:
        json_schema: JSON schema string from PydanticOutputParser.get_format_instructions()
        attribute_names: List of attribute names to extract descriptions for

    Returns:
        Dictionary mapping attribute names to their descriptions

    Example:
        >>> schema = '{"properties": {"drug_target": {"description": "The target protein", ...}}}'
        >>> _extract_attribute_descriptions(schema, ["drug_target"])
        {"drug_target": "The target protein"}
    """
    attribute_descriptions = {}
    for attr in attribute_names:
        # Extract description from JSON schema using regex
        # Pattern matches: "attr_name": {..., "description": "text", ...}
        pattern = rf'"{attr}":\s*{{[^}}]*"description":\s*"([^"]*)"'
        match = re.search(pattern, json_schema)
        if match:
            attribute_descriptions[attr] = match.group(1)
        else:
            # Fallback if description not found
            attribute_descriptions[attr] = f"Evidence for {attr}"
    return attribute_descriptions


def _invoke_llm_with_retry(llm: Any, messages: list[Any], max_retries: int = 3) -> str:
    """Invoke LLM with retry logic for transient failures.

    This helper provides basic retry functionality for LLM invocations,
    primarily for handling transient network errors or rate limiting.

    Args:
        llm: Initialized LLM instance
        messages: List of LangChain messages to send
        max_retries: Maximum number of retry attempts (default: 3)

    Returns:
        Raw text response from LLM

    Raises:
        Exception: If all retries are exhausted

    Note:
        This is separate from the excerpt validation retry logic in deep_judgment_parse,
        which handles specific validation failures with custom error feedback.
    """
    for attempt in range(max_retries):
        try:
            response = llm.invoke(messages)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.warning(f"LLM invocation attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt == max_retries - 1:
                raise
            # Simple exponential backoff could be added here
            continue
    raise RuntimeError("Unreachable code")
