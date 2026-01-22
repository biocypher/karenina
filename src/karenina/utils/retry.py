"""Retry utilities for LLM calls and other transient-error-prone operations.

This module provides pre-configured retry decorators and factory functions
for creating retry logic with exponential backoff.
"""

import logging
from collections.abc import Callable
from typing import Any

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .errors import is_retryable_error

__all__ = [
    "log_retry",
    "create_transient_retry",
    "TRANSIENT_RETRY",
]

logger = logging.getLogger(__name__)


def log_retry(retry_state: Any, *, context: str = "LLM call") -> None:
    """Log retry attempt with error details.

    Args:
        retry_state: Tenacity retry state object containing attempt info.
        context: Description of what operation is being retried.

    Example:
        >>> from functools import partial
        >>> before_sleep = partial(log_retry, context="abstention check")
    """
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    attempt = retry_state.attempt_number
    max_attempts = 3  # Default max attempts
    logger.warning(f"Retrying {context} (attempt {attempt}/{max_attempts}) after error: {exc}")


def create_transient_retry(
    max_attempts: int = 3,
    min_wait: int = 2,
    max_wait: int = 10,
    *,
    context: str = "LLM call",
) -> Callable[..., Any]:
    """Create a tenacity retry decorator for transient errors.

    Creates a retry decorator configured for exponential backoff that only
    retries on transient errors (connection errors, timeouts, rate limits,
    5xx errors, overloaded services).

    Args:
        max_attempts: Maximum number of retry attempts. Default: 3.
        min_wait: Minimum wait time in seconds between retries. Default: 2.
        max_wait: Maximum wait time in seconds between retries. Default: 10.
        context: Description of operation for logging. Default: "LLM call".

    Returns:
        A tenacity retry decorator configured with the specified parameters.

    Example:
        >>> @create_transient_retry(max_attempts=5, context="API call")
        ... async def call_api():
        ...     pass
    """

    def _log_retry(retry_state: Any) -> None:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        logger.warning(f"Retrying {context} (attempt {retry_state.attempt_number}/{max_attempts}) after error: {exc}")

    return retry(
        retry=retry_if_exception(is_retryable_error),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        reraise=True,
        before_sleep=_log_retry,
    )


# Pre-configured retry decorator with default settings
# Use this for standard LLM operations with transient error handling
TRANSIENT_RETRY = create_transient_retry()
