"""Error mapping for LangChain Deep Agents adapter.

Maps Deep Agents and LangGraph exceptions to karenina's exception hierarchy.
"""

from __future__ import annotations

import asyncio
import logging

from karenina.ports import AgentExecutionError, AgentResponseError, AgentTimeoutError

logger = logging.getLogger(__name__)


def wrap_deep_agents_error(error: Exception) -> tuple[Exception, bool]:
    """Map a Deep Agents / LangGraph exception to a karenina exception.

    Args:
        error: The original exception from Deep Agents or LangGraph.

    Returns:
        Tuple of (mapped_exception, limit_reached). The limit_reached flag
        is True when the error indicates the agent hit a recursion or turn limit.
    """
    error_str = str(error).lower()

    # Check for recursion / turn limit errors
    if "recursion" in error_str or "limit" in error_str or "max_turns" in error_str:
        return (
            AgentExecutionError(f"Agent hit turn limit: {error}"),
            True,
        )

    # Timeout errors
    if isinstance(error, TimeoutError | asyncio.TimeoutError):
        return AgentTimeoutError(f"Agent execution timed out: {error}"), False

    # Output parsing errors
    if ("parse" in error_str and "output" in error_str) or ("output" in error_str and "format" in error_str):
        return AgentResponseError(f"Failed to parse agent response: {error}"), False

    # GraphRecursionError from LangGraph
    try:
        from langgraph.errors import GraphRecursionError

        if isinstance(error, GraphRecursionError):
            return (
                AgentExecutionError(f"Agent hit recursion limit: {error}"),
                True,
            )
    except ImportError:
        pass

    # Default: general execution error
    return AgentExecutionError(f"Agent execution failed: {error}"), False
