"""Abstention detection for identifying when models refuse to answer questions."""

import json
import logging
from typing import Any

from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ....adapters import get_llm
from ....ports import LLMResponse, Message
from ....ports.usage import UsageMetadata
from ....schemas.workflow import ModelConfig
from ..utils.error_helpers import is_retryable_error
from ..utils.json_helpers import strip_markdown_fences as _strip_markdown_fences
from ..utils.prompts import ABSTENTION_DETECTION_SYS, ABSTENTION_DETECTION_USER

# Set up logger
logger = logging.getLogger(__name__)


class AbstentionResult(BaseModel):
    """Result of abstention detection by LLM judge."""

    reasoning: str = Field(description="Brief explanation of why this was classified as abstention or genuine attempt")
    abstention_detected: bool = Field(
        description="True if the model refused to answer or abstained, False if genuine answer attempt"
    )


def _convert_usage_to_dict(usage: UsageMetadata) -> dict[str, Any]:
    """Convert UsageMetadata dataclass to dict for backward compatibility."""
    result: dict[str, Any] = {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
    }
    if usage.cost_usd is not None:
        result["cost_usd"] = usage.cost_usd
    if usage.cache_read_tokens is not None:
        result["cache_read_input_tokens"] = usage.cache_read_tokens
    if usage.cache_creation_tokens is not None:
        result["cache_creation_input_tokens"] = usage.cache_creation_tokens
    if usage.model is not None:
        result["model"] = usage.model
    return result


def _extract_result(response: LLMResponse) -> AbstentionResult | None:
    """Extract AbstentionResult from LLMResponse."""
    if isinstance(response.raw, AbstentionResult):
        return response.raw
    if hasattr(response.raw, "abstention_detected"):
        return AbstentionResult(
            abstention_detected=response.raw.abstention_detected,
            reasoning=getattr(response.raw, "reasoning", "No reasoning provided"),
        )
    return None


def _fallback_parse(content: str, usage_metadata: dict[str, Any]) -> tuple[bool, bool, str | None, dict[str, Any]]:
    """Fallback to manual JSON parsing when structured output fails."""
    try:
        cleaned_response = _strip_markdown_fences(content)
        if cleaned_response is None:
            return False, False, None, usage_metadata
        result = json.loads(cleaned_response)
        abstention_detected = result.get("abstention_detected", False)
        reasoning = result.get("reasoning", "No reasoning provided")
        logger.debug(f"Abstention check (fallback): {abstention_detected}")
        return abstention_detected, True, reasoning, usage_metadata
    except json.JSONDecodeError:
        return False, False, None, usage_metadata


def detect_abstention(
    raw_llm_response: str,
    parsing_model: ModelConfig,
    question_text: str,
) -> tuple[bool, bool, str | None, dict[str, Any]]:
    """
    Detect if the model refused to answer or abstained from answering.

    This function uses an LLM to analyze the response and determine if it contains
    patterns indicating refusal, abstention, or evasion. Uses retry logic for
    transient errors (connection issues, rate limits, etc.).

    Args:
        raw_llm_response: The raw response text from the answering model
        parsing_model: Configuration for the model to use for abstention detection
        question_text: The original question that was asked

    Returns:
        Tuple of (abstention_detected, check_performed, reasoning, usage_metadata):
        - abstention_detected: True if model refused/abstained, False if genuine attempt
        - check_performed: True if check completed successfully, False if check failed
        - reasoning: The LLM's explanation for its determination (None if check failed)
        - usage_metadata: Token usage metadata from the LLM invocation

    Examples:
        >>> config = ModelConfig(id="parser", model_provider="openai", ...)
        >>> detected, performed, reasoning, metadata = detect_abstention("I cannot answer this", config, "What is X?")
        >>> print(detected, performed, reasoning)
        True, True, "Response contains explicit refusal pattern"
    """

    def _log_retry(retry_state: Any) -> None:
        """Log retry attempt with error details."""
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        logger.warning(f"Retrying abstention detection (attempt {retry_state.attempt_number}/3) after error: {exc}")

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),  # Try 3 times
        wait=wait_exponential(multiplier=1, min=2, max=10),  # Exponential backoff: 2s, 4s, 8s
        reraise=True,
        before_sleep=_log_retry,
    )
    def _detect_with_retry() -> tuple[bool, bool, str | None, dict[str, Any]]:
        """Inner function with retry logic."""
        usage_metadata: dict[str, Any] = {}

        try:
            # Create config copy with temperature=0 for consistent detection
            # Note: LangChain adapter respects temperature; Claude SDK adapter ignores it
            detection_config = parsing_model.model_copy(update={"temperature": 0.0})

            # Get LLM via adapter factory
            llm = get_llm(detection_config)

            # Configure for structured output
            structured_llm = llm.with_structured_output(AbstentionResult)

            # Build messages using unified Message class
            user_prompt = ABSTENTION_DETECTION_USER.format(question=question_text, response=raw_llm_response)
            messages = [
                Message.system(ABSTENTION_DETECTION_SYS),
                Message.user(user_prompt),
            ]

            # Invoke with structured output
            response: LLMResponse = structured_llm.invoke(messages)
            usage_metadata = _convert_usage_to_dict(response.usage)

            # Extract result from structured output or fall back to manual parsing
            result = _extract_result(response)
            if result is not None:
                logger.debug(f"Abstention check: {result.abstention_detected} - Reasoning: {result.reasoning}")
                return result.abstention_detected, True, result.reasoning, usage_metadata

            # Fallback: manual JSON parsing from content
            return _fallback_parse(response.content, usage_metadata)

        except json.JSONDecodeError as e:
            # JSON parsing failed - log and treat as check failure
            logger.warning(f"Failed to parse abstention detection response as JSON: {e}")
            return False, False, None, usage_metadata

        except Exception as e:
            # Check if this is a retryable error
            if is_retryable_error(e):
                logger.info(f"Detected retryable error in abstention check: {type(e).__name__}: {e}")
                raise  # Re-raise to trigger retry
            else:
                # Non-retryable error - log and treat as check failure
                logger.warning(f"Abstention detection failed with non-retryable error: {e}")
                return False, False, None, usage_metadata

    try:
        return _detect_with_retry()
    except Exception as e:
        # All retries exhausted or unhandled error
        logger.error(f"Abstention detection failed after all retries: {e}")
        return False, False, None, {}
