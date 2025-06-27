"""LLM interface and session management for Karenina.

This package provides a unified interface for calling language models,
managing conversation sessions, and handling LLM-related operations
across the Karenina framework.
"""

from .interface import (
    ChatRequest,
    ChatResponse,
    ChatSession,
    LLMError,
    LLMNotAvailableError,
    SessionError,
    call_model,
    clear_all_sessions,
    delete_session,
    get_session,
    list_sessions,
)

__all__ = [
    "ChatRequest",
    "ChatResponse",
    "ChatSession",
    "LLMError",
    "LLMNotAvailableError",
    "SessionError",
    "call_model",
    "clear_all_sessions",
    "delete_session",
    "get_session",
    "list_sessions",
]
