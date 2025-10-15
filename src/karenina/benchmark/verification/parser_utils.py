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


def format_excerpts_for_reasoning(excerpts: dict[str, list[dict[str, Any]]]) -> str:
    """Format excerpts in a human-readable way for the reasoning stage.

    Formats excerpts with clear hierarchical structure, including search results when present.
    This improves LLM comprehension compared to raw JSON dumps.

    Args:
        excerpts: Dictionary mapping attribute names to lists of excerpt objects.
                  Each excerpt object may contain:
                  - text: The excerpt text (displayed in full)
                  - confidence: Confidence level (none/low/medium/high)
                  - similarity_score: Fuzzy match score (0.0-1.0)
                  - search_results: Optional external search validation (displayed in full)
                  - explanation: Optional explanation when no excerpt found

    Returns:
        Human-readable formatted string suitable for LLM reasoning stage

    Example:
        >>> excerpts = {
        ...     "drug_target": [
        ...         {
        ...             "text": "targets BCL-2",
        ...             "confidence": "high",
        ...             "similarity_score": 0.95,
        ...             "search_results": "External validation confirms BCL-2 targeting"
        ...         },
        ...     ],
        ... }
        >>> print(format_excerpts_for_reasoning(excerpts))
        drug_target:
          Excerpt 1:
            Text: "targets BCL-2"
            Confidence: high
            Similarity: 0.950
            Search Results:
              External validation confirms BCL-2 targeting
    """
    if not excerpts:
        return "(No excerpts extracted)"

    lines = []
    for attr, excerpt_list in excerpts.items():
        lines.append(f"{attr}:")

        if not excerpt_list:
            lines.append("  (No excerpts found for this attribute)")
            continue

        for i, excerpt_obj in enumerate(excerpt_list, 1):
            lines.append(f"  Excerpt {i}:")

            # Handle missing excerpt with explanation
            if not excerpt_obj.get("text") and excerpt_obj.get("explanation"):
                lines.append(f"    Explanation: {excerpt_obj['explanation']}")
                continue

            # Regular excerpt with text
            text = excerpt_obj.get("text", "")
            if text:
                lines.append(f'    Text: "{text}"')

            # Add confidence and similarity
            confidence = excerpt_obj.get("confidence", "unknown")
            similarity = excerpt_obj.get("similarity_score", 0.0)
            lines.append(f"    Confidence: {confidence}")
            lines.append(f"    Similarity: {similarity:.3f}")

            # Add search results if present
            if "search_results" in excerpt_obj:
                search_results = excerpt_obj["search_results"]
                lines.append("    Search Results:")
                search_lines = search_results.split("\n")
                for search_line in search_lines:
                    lines.append(f"      {search_line}")

        lines.append("")  # Blank line between attributes

    return "\n".join(lines)


def format_reasoning_for_parsing(reasoning: dict[str, str]) -> str:
    """Format reasoning traces in a human-readable way for the parsing stage.

    Formats reasoning traces with clear structure, making it easier for the LLM
    to understand how each attribute should be valued based on the reasoning.

    Args:
        reasoning: Dictionary mapping attribute names to reasoning text strings.
                   Each reasoning text explains how excerpts inform the attribute value.

    Returns:
        Human-readable formatted string suitable for LLM parsing stage

    Example:
        >>> reasoning = {
        ...     "drug_target": "The excerpt 'targets BCL-2' clearly indicates BCL-2 as the target",
        ...     "mechanism": "Based on the evidence, the mechanism involves apoptosis inhibition"
        ... }
        >>> print(format_reasoning_for_parsing(reasoning))
        drug_target:
          The excerpt 'targets BCL-2' clearly indicates BCL-2 as the target
        <BLANKLINE>
        mechanism:
          Based on the evidence, the mechanism involves apoptosis inhibition
    """
    if not reasoning:
        return "(No reasoning traces generated)"

    lines = []
    for attr, reasoning_text in reasoning.items():
        lines.append(f"{attr}:")
        # Indent the reasoning text for readability
        reasoning_lines = reasoning_text.split("\n")
        for reasoning_line in reasoning_lines:
            lines.append(f"  {reasoning_line}")
        lines.append("")  # Blank line between attributes

    return "\n".join(lines)
