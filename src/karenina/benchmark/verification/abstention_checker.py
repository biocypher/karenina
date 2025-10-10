"""Abstention detection for identifying when models refuse to answer questions."""

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ...llm.interface import init_chat_model_unified
from ...prompts.abstention_detection import ABSTENTION_DETECTION_SYS, ABSTENTION_DETECTION_USER
from ..models import ModelConfig

# Set up logger
logger = logging.getLogger(__name__)


def is_retryable_error(exception: Exception) -> bool:
    """Check if an exception is retryable (transient error)."""
    exception_str = str(exception).lower()
    exception_type = type(exception).__name__

    # Connection-related errors
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

    # Common retryable exception types
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


def _strip_markdown_fences(text: str) -> str:
    """
    Remove markdown JSON code fences from LLM response text.

    Args:
        text: Raw text response from LLM that may contain markdown fences

    Returns:
        Cleaned text with markdown fences removed
    """
    if not isinstance(text, str):
        return text

    # Strip leading and trailing markdown JSON fences
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]  # Remove ```json
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]  # Remove ```

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]  # Remove trailing ```

    return cleaned.strip()


def detect_abstention(
    raw_llm_response: str,
    parsing_model: ModelConfig,
    question_text: str,
) -> tuple[bool, bool]:
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
        Tuple of (abstention_detected, check_performed):
        - abstention_detected: True if model refused/abstained, False if genuine attempt
        - check_performed: True if check completed successfully, False if check failed

    Examples:
        >>> config = ModelConfig(id="parser", model_provider="openai", ...)
        >>> detected, performed = detect_abstention("I cannot answer this", config, "What is X?")
        >>> print(detected, performed)
        True, True
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
    def _detect_with_retry() -> tuple[bool, bool]:
        """Inner function with retry logic."""
        try:
            # Initialize the parsing model for abstention detection
            llm = init_chat_model_unified(
                model=parsing_model.model_name,
                provider=parsing_model.model_provider,
                temperature=0.0,  # Use temperature 0 for consistent detection
                interface=parsing_model.interface,
            )

            # Construct the prompt
            user_prompt = ABSTENTION_DETECTION_USER.format(question=question_text, response=raw_llm_response)

            messages = [
                SystemMessage(content=ABSTENTION_DETECTION_SYS),
                HumanMessage(content=user_prompt),
            ]

            # Invoke the LLM
            response = llm.invoke(messages)
            raw_response = response.content if hasattr(response, "content") else str(response)

            # Parse the JSON response
            cleaned_response = _strip_markdown_fences(raw_response)
            result = json.loads(cleaned_response)

            # Extract the abstention determination
            abstention_detected = result.get("abstention_detected", False)

            # Log the reasoning for debugging
            reasoning = result.get("reasoning", "No reasoning provided")
            logger.debug(f"Abstention check result: {abstention_detected} - Reasoning: {reasoning}")

            return abstention_detected, True

        except json.JSONDecodeError as e:
            # JSON parsing failed - log and treat as check failure
            logger.warning(f"Failed to parse abstention detection response as JSON: {e}")
            return False, False

        except Exception as e:
            # Check if this is a retryable error
            if is_retryable_error(e):
                logger.info(f"Detected retryable error in abstention check: {type(e).__name__}: {e}")
                raise  # Re-raise to trigger retry
            else:
                # Non-retryable error - log and treat as check failure
                logger.warning(f"Abstention detection failed with non-retryable error: {e}")
                return False, False

    try:
        return _detect_with_retry()
    except Exception as e:
        # All retries exhausted or unhandled error
        logger.error(f"Abstention detection failed after all retries: {e}")
        return False, False
