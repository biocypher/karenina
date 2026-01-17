"""Error handling utilities for verification operations.

This module provides functions for detecting and handling transient errors
during LLM invocations and other operations.

Functions:
    is_retryable_error: Check if an exception is a transient/retryable error
"""

__all__ = [
    "is_retryable_error",
]


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
