"""LLM interface and session management for Karenina.

This package provides a unified interface for calling language models,
managing conversation sessions, and handling LLM-related operations
across the Karenina framework.
"""

from .exceptions import LLMError, LLMNotAvailableError, SessionError
from .interface import (
    ChatRequest,
    ChatResponse,
    ChatSession,
    call_model,
    clear_all_sessions,
    delete_session,
    get_session,
    list_sessions,
)
from .parallel_invoker import ParallelLLMInvoker, read_async_config

__all__ = [
    "ChatRequest",
    "ChatResponse",
    "ChatSession",
    "LLMError",
    "LLMNotAvailableError",
    "ParallelLLMInvoker",
    "SessionError",
    "call_model",
    "clear_all_sessions",
    "delete_session",
    "get_session",
    "list_sessions",
    "read_async_config",
]
