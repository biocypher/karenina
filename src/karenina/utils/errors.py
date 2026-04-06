"""Error detection and classification utilities for LLM operations.

This module provides:
- ErrorCategory: enum classifying errors into connection, timeout, rate_limit,
  server_error, or permanent categories.
- ErrorRegistry: extensible classifier that maps exceptions to categories using
  built-in rules (type names, message substrings) and user-registered patterns.
- is_retryable_error: backward-compatible function wrapping ErrorRegistry.
"""

from __future__ import annotations

import enum
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "ErrorCategory",
    "ErrorRegistry",
    "is_retryable_error",
]


class ErrorCategory(enum.Enum):
    """Classification of an error for retry decisions.

    Each category maps to a distinct retry budget in RetryPolicy.
    """

    CONNECTION = "connection"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    SERVER_ERROR = "server_error"
    PERMANENT = "permanent"

    def is_retryable(self) -> bool:
        """Return True if errors in this category should be retried."""
        return self is not ErrorCategory.PERMANENT


# Built-in type name rules: maps exception class names to categories.
_BUILTIN_TYPE_RULES: dict[str, ErrorCategory] = {
    "ConnectionError": ErrorCategory.CONNECTION,
    "APIConnectionError": ErrorCategory.CONNECTION,
    "TimeoutError": ErrorCategory.TIMEOUT,
    "ReadTimeout": ErrorCategory.TIMEOUT,
    "ConnectTimeout": ErrorCategory.TIMEOUT,
    "APITimeoutError": ErrorCategory.TIMEOUT,
    "StreamingTimeoutError": ErrorCategory.TIMEOUT,
    "RateLimitError": ErrorCategory.RATE_LIMIT,
    "OverloadedError": ErrorCategory.RATE_LIMIT,
    "InternalServerError": ErrorCategory.SERVER_ERROR,
    "HTTPError": ErrorCategory.SERVER_ERROR,
}

# Built-in message substring rules: maps lowercased substrings to categories.
# Order matters: first match wins. Checked after all type-based rules.
_BUILTIN_MESSAGE_RULES: list[tuple[str, ErrorCategory]] = [
    ("connection", ErrorCategory.CONNECTION),
    ("network", ErrorCategory.CONNECTION),
    ("dns", ErrorCategory.CONNECTION),
    ("timeout", ErrorCategory.TIMEOUT),
    ("timed out", ErrorCategory.TIMEOUT),
    ("rate limit", ErrorCategory.RATE_LIMIT),
    ("429", ErrorCategory.RATE_LIMIT),
    ("overloaded", ErrorCategory.RATE_LIMIT),
    ("500", ErrorCategory.SERVER_ERROR),
    ("502", ErrorCategory.SERVER_ERROR),
    ("503", ErrorCategory.SERVER_ERROR),
    ("event loop", ErrorCategory.CONNECTION),
    ("portal", ErrorCategory.CONNECTION),
    ("temporary failure", ErrorCategory.CONNECTION),
]


class ErrorRegistry:
    """Extensible error classifier that maps exceptions to ErrorCategory.

    Classification checks in priority order:
    1. User-registered exception types (isinstance for classes, type name for strings)
    2. User-registered message substring patterns
    3. Built-in type name rules
    4. Built-in message substring rules
    5. Default: PERMANENT

    Example:
        >>> registry = ErrorRegistry()
        >>> registry.classify(ConnectionError("reset"))
        <ErrorCategory.CONNECTION: 'connection'>
        >>> registry.register(ValueError, ErrorCategory.RATE_LIMIT)
        >>> registry.classify(ValueError("custom"))
        <ErrorCategory.RATE_LIMIT: 'rate_limit'>
    """

    def __init__(self) -> None:
        self._custom_classes: list[tuple[type, ErrorCategory]] = []
        self._custom_type_names: list[tuple[str, ErrorCategory]] = []
        self._custom_message_patterns: list[tuple[str, ErrorCategory]] = []

    def register(self, pattern: str | type, category: ErrorCategory) -> None:
        """Register an exception class or type name string for classification.

        Args:
            pattern: An exception class (for isinstance checks) or a type name
                string (for exact type name matching).
            category: The ErrorCategory to assign when matched.
        """
        if isinstance(pattern, type):
            self._custom_classes.append((pattern, category))
        else:
            self._custom_type_names.append((pattern, category))

    def register_pattern(
        self,
        pattern: str,
        category: ErrorCategory,
        *,
        match_type: str = "message_substring",
    ) -> None:
        """Register a pattern for classification with explicit match type.

        Args:
            pattern: The string to match against.
            category: The ErrorCategory to assign when matched.
            match_type: Either "message_substring" (match against lowercased
                exception message) or "type_name" (match against exception
                class name).
        """
        if match_type == "type_name":
            self._custom_type_names.append((pattern, category))
        elif match_type == "message_substring":
            self._custom_message_patterns.append((pattern, category))
        else:
            msg = "match_type must be 'type_name' or 'message_substring', got %r"
            raise ValueError(msg % match_type)

    def classify(self, exception: BaseException) -> ErrorCategory:
        """Classify an exception into an ErrorCategory.

        Checks user-registered patterns first (types, then messages), then
        built-in rules (types, then messages). Returns PERMANENT if no rule
        matches.

        Special case: StreamingTimeoutError with no partial content is
        classified as RATE_LIMIT (server never started responding, likely
        queued due to congestion), not TIMEOUT. This gives it more retries
        with longer backoff, which is appropriate for overloaded servers.

        Args:
            exception: The exception to classify.

        Returns:
            The matched ErrorCategory.
        """
        exc_type = type(exception)
        exc_type_name = exc_type.__name__
        exc_message = str(exception).lower()

        # 1. User-registered exception classes (isinstance)
        for cls, category in self._custom_classes:
            if isinstance(exception, cls):
                return category

        # 2. User-registered type name strings
        for name, category in self._custom_type_names:
            if exc_type_name == name:
                return category

        # 3. User-registered message substring patterns
        for pattern, category in self._custom_message_patterns:
            if pattern.lower() in exc_message:
                return category

        # 4. StreamingTimeoutError: zero content = congestion (RATE_LIMIT),
        #    partial content = genuine slow response (TIMEOUT).
        #    Placed after user patterns so custom registrations take precedence.
        if (
            exc_type_name == "StreamingTimeoutError"
            and hasattr(exception, "partial_content")
            and not exception.partial_content
        ):
            return ErrorCategory.RATE_LIMIT

        # 5. Built-in type name rules (check full MRO for built-in types)
        for ancestor in exc_type.__mro__:
            ancestor_name = ancestor.__name__
            if ancestor_name in _BUILTIN_TYPE_RULES:
                return _BUILTIN_TYPE_RULES[ancestor_name]

        # 5. Built-in message substring rules
        for keyword, category in _BUILTIN_MESSAGE_RULES:
            if keyword in exc_message:
                return category

        # 6. No match: permanent
        return ErrorCategory.PERMANENT


def is_retryable_error(exception: BaseException) -> bool:
    """Check if an exception is retryable (transient error).

    This is a backward-compatible wrapper around ErrorRegistry. It will be
    removed in a future version; use ErrorRegistry.classify() directly.

    Args:
        exception: The exception to check.

    Returns:
        True if the error is transient and should be retried, False otherwise.

    Example:
        >>> is_retryable_error(ConnectionError("connection reset"))
        True
        >>> is_retryable_error(ValueError("invalid input"))
        False
    """
    return ErrorRegistry().classify(exception).is_retryable()
