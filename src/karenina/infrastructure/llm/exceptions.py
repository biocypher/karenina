"""LLM-related exceptions.

This module provides a centralized location for LLM infrastructure exceptions.

Note: Manual trace exceptions (ManualTraceError, ManualTraceNotFoundError) have been
moved to karenina.adapters.manual as part of the manual trace consolidation.
"""

__all__ = [
    "LLMError",
    "LLMNotAvailableError",
    "SessionError",
]


class LLMError(Exception):
    """Base exception for LLM-related errors."""

    pass


class LLMNotAvailableError(LLMError):
    """Raised when LangChain is not available."""

    pass


class SessionError(LLMError):
    """Raised when there's an error with session management."""

    pass
