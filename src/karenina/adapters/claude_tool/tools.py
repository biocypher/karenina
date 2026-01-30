"""Tool schema conversion utilities for the claude_tool adapter.

This module provides utilities for converting tools between karenina's Tool format
and the Anthropic SDK's @beta_async_tool decorator format. It handles both:
1. Static Tool definitions from karenina's ports
2. Dynamic MCP tools from connected MCP servers
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp import ClientSession

    from karenina.ports.agent import Tool

logger = logging.getLogger(__name__)


def create_mcp_tool_function(
    session: Any,
    mcp_tool: Any,
    server_name: str,
) -> Callable[..., Coroutine[Any, Any, str]]:
    """Create an async tool function that wraps an MCP tool.

    Creates a function that can be decorated with @beta_async_tool and
    calls the MCP session to execute the actual tool.

    Args:
        session: The MCP ClientSession for this server.
        mcp_tool: The MCP tool definition (has name, description, inputSchema).
        server_name: Name of the MCP server (used for logging).

    Returns:
        An async function that executes the MCP tool.
    """

    async def tool_fn(**kwargs: Any) -> str:
        """Execute the MCP tool with the given arguments."""
        logger.debug(f"Calling MCP tool '{mcp_tool.name}' on server '{server_name}' with args: {kwargs}")

        try:
            result = await session.call_tool(mcp_tool.name, kwargs)

            # Extract text content from result
            text_parts: list[str] = []
            for content in result.content:
                if hasattr(content, "text"):
                    text_parts.append(content.text)

            response = "\n".join(text_parts)
            logger.debug(f"MCP tool '{mcp_tool.name}' returned: {response[:200]}...")
            return response

        except Exception as e:
            error_msg = f"Error calling MCP tool '{mcp_tool.name}': {e}"
            logger.error(error_msg)
            return error_msg

    return tool_fn


def wrap_mcp_tool(
    session: ClientSession,
    mcp_tool: Any,
    server_name: str,
) -> Any:
    """Wrap an MCP tool with the @beta_async_tool decorator.

    Creates a callable that the Anthropic SDK's tool_runner can use.

    Args:
        session: The MCP ClientSession for this server.
        mcp_tool: The MCP tool definition from session.list_tools().
        server_name: Name of the MCP server (used for namespacing).

    Returns:
        A decorated async function ready for use with tool_runner.
    """
    from anthropic import beta_async_tool

    # Create the tool function
    tool_fn = create_mcp_tool_function(session, mcp_tool, server_name)

    # Apply the beta_async_tool decorator
    decorated = beta_async_tool(
        name=mcp_tool.name,
        description=mcp_tool.description or f"Tool from {server_name}",
        input_schema=mcp_tool.inputSchema,
    )(tool_fn)

    return decorated


def wrap_static_tool(tool: Tool) -> Any:
    """Wrap a static Tool definition with @beta_async_tool decorator.

    Creates a callable for tools that don't need MCP session execution.
    Note: These tools won't actually execute anything - they're placeholders
    for when the caller provides their own execution logic.

    Args:
        tool: A Tool definition from karenina's ports.

    Returns:
        A decorated async function ready for use with tool_runner.

    Note:
        Static tools without execution logic will return an error message
        indicating they need to be implemented. For actual tool execution,
        use MCP tools or provide a custom executor.
    """
    from anthropic import beta_async_tool

    async def tool_fn(**kwargs: Any) -> str:
        """Placeholder tool function."""
        # Static tools without MCP backing don't have execution logic
        return f"Tool '{tool.name}' was called with arguments: {kwargs}"

    decorated = beta_async_tool(
        name=tool.name,
        description=tool.description,
        input_schema=tool.input_schema,
    )(tool_fn)

    return decorated


def wrap_tool_with_executor(
    tool: Tool,
    executor: Callable[..., Coroutine[Any, Any, str]],
) -> Any:
    """Wrap a Tool definition with a custom executor.

    Args:
        tool: A Tool definition from karenina's ports.
        executor: An async function that executes the tool.

    Returns:
        A decorated async function ready for use with tool_runner.
    """
    from anthropic import beta_async_tool

    decorated = beta_async_tool(
        name=tool.name,
        description=tool.description,
        input_schema=tool.input_schema,
    )(executor)

    return decorated


def apply_cache_control_to_tool(wrapped_tool: Any) -> Any:
    """Add cache_control to a wrapped tool for Anthropic prompt caching.

    When applied to the last tool in a tools list, this enables caching of
    all tool definitions. Anthropic caches the entire prefix up to and
    including the tool with cache_control.

    The wrapped tool must have been created with @beta_async_tool decorator.
    This function patches the tool's to_dict() method to include cache_control.

    Args:
        wrapped_tool: A tool wrapped with @beta_async_tool decorator.

    Returns:
        The same tool with cache_control added to its dict representation.

    Note:
        Caching requires minimum 1024 tokens in tool definitions for most
        Claude models (4096 for Opus 4.5 and Haiku 4.5).
    """
    original_to_dict = getattr(wrapped_tool, "to_dict", None)

    if original_to_dict is None:
        logger.warning("Tool does not have to_dict method, cannot apply cache_control")
        return wrapped_tool

    def patched_to_dict() -> dict[str, Any]:
        """Return tool dict with cache_control added."""
        params: dict[str, Any] = original_to_dict()
        params["cache_control"] = {"type": "ephemeral"}
        return params

    wrapped_tool.to_dict = patched_to_dict
    return wrapped_tool
