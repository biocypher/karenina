"""Error wrapping for the Codex SDK adapter.

Maps openai-codex exceptions onto karenina's port error hierarchy. All SDK
exception detection uses type-name strings so this module imports and works
without the openai-codex package installed.

SDK exception mapping:
    FileNotFoundError (binary spawn)      -> AdapterUnavailableError
    TransportClosedError                  -> AgentExecutionError
    ServerBusyError / RetryLimitExceeded  -> AgentExecutionError (transient)
    JSON-RPC errors (ParseError, Invalid* ,
    MethodNotFoundError, InternalRpcError,
    CodexRpcError, JsonRpcError)          -> AgentResponseError
    TimeoutError / asyncio.TimeoutError   -> AgentTimeoutError
    CodexError / anything else            -> AgentExecutionError

Codex has no max_turns equivalent, so the limit_reached flag is best
effort: it is set only when the error message mentions a turn, recursion,
or context limit.
"""

from __future__ import annotations

import asyncio
import logging

from karenina.ports.errors import (
    AdapterUnavailableError,
    AgentExecutionError,
    AgentResponseError,
    AgentTimeoutError,
    PortError,
)

logger = logging.getLogger(__name__)

_TRANSIENT_ERROR_TYPES = frozenset({"ServerBusyError", "RetryLimitExceededError"})
_JSONRPC_ERROR_TYPES = frozenset(
    {
        "JsonRpcError",
        "CodexRpcError",
        "ParseError",
        "InvalidRequestError",
        "MethodNotFoundError",
        "InvalidParamsError",
        "InternalRpcError",
    }
)
_LIMIT_HINTS = ("recursion", "turn limit", "max_turns", "context window", "context limit", "token limit")


def _is_limit_message(message: str) -> bool:
    lowered = message.lower()
    return any(hint in lowered for hint in _LIMIT_HINTS)


def wrap_codex_error(e: Exception) -> tuple[PortError, bool]:
    """Wrap a Codex SDK exception into a karenina port error.

    Args:
        e: The exception raised while driving the codex app-server.

    Returns:
        Tuple of (mapped_error, limit_reached). The limit_reached flag is
        True when the message indicates a turn, recursion, or context
        limit. Codex exposes no structured limit signal, so this is a
        best-effort message check.

    Example:
        >>> try:
        ...     result = await thread.run(prompt)
        ... except Exception as e:
        ...     mapped, was_limit = wrap_codex_error(e)
        ...     raise mapped from e
    """
    exc_type_name = type(e).__name__
    message = str(e)

    if isinstance(e, FileNotFoundError):
        return (
            AdapterUnavailableError(
                message=(
                    f"Codex CLI binary could not be launched: {e}. "
                    "Install the SDK with its bundled binary: pip install openai-codex"
                ),
                reason="codex binary not found",
                fallback_interface="langchain",
            ),
            False,
        )

    if isinstance(e, TimeoutError | asyncio.TimeoutError):
        return AgentTimeoutError(f"Codex agent execution timed out: {e}"), False

    if exc_type_name in _TRANSIENT_ERROR_TYPES:
        return (
            AgentExecutionError(
                message=f"Codex model provider is overloaded (transient, retryable): {e}",
                stderr=message or None,
            ),
            False,
        )

    if exc_type_name == "TransportClosedError":
        return (
            AgentExecutionError(
                message=f"Codex app-server transport closed unexpectedly: {e}",
                stderr=message or None,
            ),
            False,
        )

    if exc_type_name in _JSONRPC_ERROR_TYPES:
        return (
            AgentResponseError(message=f"Codex app-server JSON-RPC error ({exc_type_name}): {e}"),
            False,
        )

    if _is_limit_message(message):
        return (
            AgentExecutionError(message=f"Codex agent hit a limit: {e}", limit_reached=True),
            True,
        )

    return (
        AgentExecutionError(
            message=f"Codex SDK error ({exc_type_name}): {e}",
            stderr=message or None,
        ),
        False,
    )
