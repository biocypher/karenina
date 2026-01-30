"""Error detection utilities for LLM operations.

This module provides functions for detecting transient errors that should
trigger retry logic in LLM invocations and other operations.
"""

__all__ = ["is_retryable_error"]


def is_retryable_error(exception: BaseException) -> bool:
    """Check if an exception is retryable (transient error).

    Detects transient errors that typically resolve on retry:
    - Connection errors (network issues, DNS failures)
    - Timeouts (read, connect, general)
    - Rate limits (429 errors)
    - Server errors (5xx status codes)
    - Overloaded services

    Args:
        exception: The exception to check.

    Returns:
        True if the error is transient and should be retried, False otherwise.

    Example:
        >>> is_retryable_error(ConnectionError("connection reset"))
        True
        >>> is_retryable_error(ValueError("invalid input"))
        False
        >>> is_retryable_error(Exception("rate limit exceeded"))
        True
    """
    exception_str = str(exception).lower()
    exception_type = type(exception).__name__

    # Connection-related errors (check error message content)
    transient_keywords = [
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
        "overloaded",
    ]

    if any(keyword in exception_str for keyword in transient_keywords):
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
        "OverloadedError",
        "InternalServerError",
    ]

    return exception_type in retryable_types
