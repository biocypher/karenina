"""LLM-related exceptions.

This module provides a centralized location for all LLM-related exceptions,
avoiding circular imports between interface.py, manual_traces.py, and manual_llm.py.
"""

__all__ = [
    "LLMError",
    "LLMNotAvailableError",
    "SessionError",
    "ManualTraceError",
    "ManualTraceNotFoundError",
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


class ManualTraceError(LLMError):
    """Raised when there's an error with manual trace operations."""

    pass


class ManualTraceNotFoundError(LLMError):
    """Raised when a manual trace is not found for a question."""

    pass
