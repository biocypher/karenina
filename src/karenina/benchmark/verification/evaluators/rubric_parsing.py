"""Robust parsing utilities for rubric evaluation using LangChain strategies.

This module provides a unified interface for invoking LLMs with structured output,
using a multi-strategy fallback approach:
1. ProviderStrategy (native structured output) - automatic for supported models
2. ToolStrategy (tool calling) - fallback with error handling
3. Manual parsing with jsonrepair - last resort

The module also supports adapter-based parsing via ParserPort for consistent
LLM backend abstraction. The adapter factory returns the appropriate implementation
(LangChainParserAdapter or ClaudeSDKParserAdapter) based on the model interface.
"""

import json
import logging
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ValidationError

from ..utils.llm_detection import is_openai_endpoint_llm as _is_openai_endpoint_llm

if TYPE_CHECKING:
    from ....ports import ParserPort

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _normalize_response_data(data: Any, model_class: type[T]) -> Any:
    """
    Normalize response data to match expected model structure.

    Handles cases where LLMs return unwrapped data (e.g., {"trait": true})
    when wrapped data is expected (e.g., {"scores": {"trait": true}}).

    Args:
        data: Parsed JSON data
        model_class: Target Pydantic model class

    Returns:
        Normalized data that matches the model structure
    """
    # Import here to avoid circular imports
    from ....schemas.workflow.rubric_outputs import BatchRubricScores

    # Handle BatchRubricScores: if data doesn't have "scores" key but is a dict,
    # wrap it in {"scores": data}
    if model_class is BatchRubricScores and isinstance(data, dict) and "scores" not in data:
        # Check if the data looks like scores (dict with non-model keys)
        # by checking if any key is not a known field name
        known_fields = {"scores"}
        if not any(k in known_fields for k in data):
            data = {"scores": data}

    return data


def invoke_with_structured_output(
    llm: Any,
    messages: list[Any],  # BaseMessage or dict
    model_class: type[T],
    *,
    parser: "ParserPort | None" = None,
    prompt_text: str | None = None,
) -> tuple[T, dict[str, Any]]:
    """
    Invoke LLM with structured output using a multi-strategy approach.

    The function supports two paths:
    - **Adapter path** (when parser provided): Uses ParserPort.parse_to_pydantic()
    - **LangChain path** (default): Uses llm.with_structured_output() method

    LangChain path strategy order:
    1. json_schema method (native structured output) - for providers that support it
       EXCEPT for models with dynamic dict fields (like BatchRubricScores)
       EXCEPT for OpenAI-compatible endpoints (can hang indefinitely)
    2. Manual parsing with json-repair - robust fallback for any response

    Args:
        llm: LangChain chat model (used for LangChain path)
        messages: List of messages to send (used for LangChain path)
        model_class: Pydantic model for structured output
        parser: Optional ParserPort adapter for adapter-based parsing
        prompt_text: Full prompt text to parse (required when parser is provided).
                     This should be the combined system+user prompt that instructs
                     the LLM how to evaluate and what schema to output.

    Returns:
        Tuple of (parsed_result, usage_metadata)

    Raises:
        ValueError: If all parsing strategies fail
    """
    usage_metadata: dict[str, Any] = {}

    # Adapter path: use ParserPort when provided
    if parser is not None:
        if prompt_text is None:
            logger.warning("ParserPort provided but prompt_text is None, falling back to LangChain path")
        else:
            try:
                result = parser.parse_to_pydantic(prompt_text, model_class)
                if isinstance(result, model_class):
                    logger.debug(f"ParserPort succeeded for rubric {model_class.__name__}")
                    # Note: ParserPort implementations track usage internally
                    # We don't have access to usage metadata through the sync API
                    return result, usage_metadata
            except Exception as e:
                logger.debug(f"ParserPort parsing failed for rubric: {e}")
                # Fall through to LangChain path for fallback

    # LangChain path
    from langchain_core.callbacks import get_usage_metadata_callback

    # Import here to avoid circular imports
    from ....schemas.workflow.rubric_outputs import BatchRubricScores

    # Skip json_schema method for models with dynamic dict fields
    # The json_schema method generates strict schemas with additionalProperties: false
    # which prevents dynamic keys like trait names in BatchRubricScores.scores
    skip_json_schema = model_class is BatchRubricScores

    # Also skip json_schema for OpenAI-compatible endpoints - they often don't support it
    # and can hang indefinitely when attempting to use json_schema method
    if _is_openai_endpoint_llm(llm):
        logger.debug(
            f"Skipping json_schema method for {type(llm).__name__} - "
            "OpenAI-compatible endpoints may not support json_schema method"
        )
        skip_json_schema = True

    # Strategy 1: Try json_schema method (native structured output)
    if not skip_json_schema:
        try:
            structured_llm = llm.with_structured_output(model_class, method="json_schema")

            with get_usage_metadata_callback() as cb:
                result = structured_llm.invoke(messages)

            usage_metadata = dict(cb.usage_metadata) if cb.usage_metadata else {}

            if isinstance(result, model_class):
                logger.debug(f"json_schema method succeeded for {model_class.__name__}")
                return result, usage_metadata

        except Exception as e:
            logger.debug(f"json_schema method failed: {e}")
    else:
        logger.debug(
            f"Skipping json_schema method for {model_class.__name__} (has dynamic dict fields or is OpenAI endpoint)"
        )

    # Strategy 2: Fall back to manual invoke + parsing with json-repair
    logger.debug(f"Falling back to manual parsing for {model_class.__name__}")

    with get_usage_metadata_callback() as cb:
        response = llm.invoke(messages)

    usage_metadata = dict(cb.usage_metadata) if cb.usage_metadata else {}
    raw_response = response.content if hasattr(response, "content") else str(response)

    # Try manual parsing (includes json-repair)
    parsed = parse_raw_response(raw_response, model_class)
    return parsed, usage_metadata


def parse_raw_response(response: str | Any, model_class: type[T]) -> T:
    """
    Parse raw LLM response with multiple fallback strategies.

    Strategy order:
    1. Already a model instance (pass through)
    2. Direct JSON parsing (with wrapper normalization)
    3. JSON extraction from mixed text (markdown fences, etc.)
    4. JSON repair with jsonrepair

    Args:
        response: Raw string response from LLM (or already parsed object)
        model_class: Pydantic model to validate against

    Returns:
        Validated Pydantic model instance

    Raises:
        ValueError: If all parsing strategies fail
    """
    # Already a model instance
    if isinstance(response, model_class):
        return response

    # Convert to string if needed
    if not isinstance(response, str):
        response = str(response)

    # Strategy 1: Direct JSON parsing with wrapper normalization
    try:
        data = json.loads(response.strip())
        data = _normalize_response_data(data, model_class)
        return model_class.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as e:
        logger.debug(f"Direct JSON parse failed: {e}")

    # Strategy 2: Extract JSON from mixed text (handles markdown fences)
    from ..utils.json_helpers import extract_json_from_text, strip_markdown_fences

    cleaned = strip_markdown_fences(response)
    if cleaned and cleaned != response:
        # Fences were stripped, try parsing the cleaned text
        try:
            data = json.loads(cleaned)
            data = _normalize_response_data(data, model_class)
            return model_class.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.debug(f"Cleaned text parse failed: {e}")

    # Try extracting JSON object from text
    json_str = extract_json_from_text(response)
    if json_str:
        try:
            data = json.loads(json_str)
            data = _normalize_response_data(data, model_class)
            return model_class.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.debug(f"Extracted JSON parse failed: {e}")

    # Strategy 3: JSON repair for malformed JSON
    try:
        from json_repair import repair_json

        repaired = repair_json(response)
        data = json.loads(repaired)
        data = _normalize_response_data(data, model_class)
        logger.debug(f"JSON repair succeeded for {model_class.__name__}")
        return model_class.model_validate(data)
    except ImportError:
        logger.warning("json-repair not installed, skipping repair strategy")
    except Exception as e:
        logger.debug(f"JSON repair failed: {e}")

    # Strategy 4: Try repair on cleaned text
    if cleaned and cleaned != response:
        try:
            from json_repair import repair_json

            repaired = repair_json(cleaned)
            data = json.loads(repaired)
            data = _normalize_response_data(data, model_class)
            logger.debug(f"JSON repair on cleaned text succeeded for {model_class.__name__}")
            return model_class.model_validate(data)
        except Exception as e:
            logger.debug(f"JSON repair on cleaned text failed: {e}")

    # All strategies failed
    preview = response[:200] if len(response) > 200 else response
    raise ValueError(f"Could not parse response into {model_class.__name__}: {preview}")


def parse_boolean_from_text(text: str) -> bool | None:
    """
    Extract a boolean value from text using flexible matching.

    Accepts:
    - JSON booleans: true, false
    - English words: yes, no
    - Numeric strings: 1, 0

    Args:
        text: Raw text potentially containing a boolean value

    Returns:
        Boolean value if found, None otherwise
    """
    text_lower = text.lower().strip()

    # Direct matches
    if text_lower in ("true", "yes", "1"):
        return True
    if text_lower in ("false", "no", "0"):
        return False

    # Search for keywords in text
    true_keywords = ["true", "yes", "correct", "pass", "passed"]
    false_keywords = ["false", "no", "incorrect", "fail", "failed"]

    # Check for true keywords
    for keyword in true_keywords:
        if keyword in text_lower:
            return True

    # Check for false keywords
    for keyword in false_keywords:
        if keyword in text_lower:
            return False

    return None


def parse_score_from_text(text: str, min_score: int, max_score: int) -> int | None:
    """
    Extract an integer score from text with clamping to valid range.

    Args:
        text: Raw text potentially containing a numeric score
        min_score: Minimum valid score (inclusive)
        max_score: Maximum valid score (inclusive)

    Returns:
        Clamped integer score if found, None otherwise
    """
    import re

    # Find all integers in text
    numbers = re.findall(r"-?\d+", text)

    if not numbers:
        return None

    # Take the first number found
    score = int(numbers[0])

    # Clamp to valid range
    return max(min_score, min(max_score, score))
