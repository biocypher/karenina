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
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from ..utils.shared import extract_json_from_text as _extract_json_from_text
from ..utils.shared import strip_markdown_fences as _strip_markdown_fences

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _is_openai_endpoint_llm(llm: Any) -> bool:
    """Check if the LLM is a ChatOpenAIEndpoint (custom OpenAI-compatible endpoint).

    These endpoints often don't support native structured output (json_schema method)
    and can hang indefinitely when attempting to use it.
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

    IMPORTANT: For OpenAI-compatible endpoints (ChatOpenAIEndpoint), we skip the
    structured output attempt entirely because many such endpoints don't support
    OpenAI's native structured output and can hang indefinitely.

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

    # Skip structured output for OpenAI-compatible endpoints - they often don't support it
    # and can hang indefinitely when attempting to use json_schema method
    if _is_openai_endpoint_llm(llm):
        logger.debug(
            f"Skipping structured output for {type(llm).__name__} - "
            "OpenAI-compatible endpoints may not support json_schema method"
        )
        return None, usage_metadata, False

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
