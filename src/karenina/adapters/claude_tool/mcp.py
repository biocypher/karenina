"""MCP client session management for the claude_tool adapter.

This module provides utilities for connecting to MCP servers using HTTP/SSE transport
with session-per-run semantics. Each agent run creates fresh MCP sessions that are
cleaned up when the run completes.

The implementation uses the MCP Python SDK's streamable HTTP client for connecting
to remote MCP servers.
"""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp import ClientSession

    from karenina.ports.agent import MCPServerConfig

logger = logging.getLogger(__name__)


async def connect_mcp_session(
    exit_stack: AsyncExitStack,
    config: MCPServerConfig,
) -> ClientSession:
    """Connect to an MCP server via HTTP/SSE transport.

    Creates a new MCP client session and registers it with the exit stack
    for automatic cleanup when the stack closes.

    Args:
        exit_stack: AsyncExitStack for managing session lifecycle.
        config: MCP server configuration. Must include 'url' for HTTP/SSE transport.
            Optional 'headers' dict for authentication.

    Returns:
        Initialized ClientSession ready for tool calls.

    Raises:
        ValueError: If config doesn't include required 'url' field.
        ImportError: If mcp package is not installed.

    Example:
        >>> async with AsyncExitStack() as stack:
        ...     config = {"url": "https://mcp.example.com/mcp", "type": "http"}
        ...     session = await connect_mcp_session(stack, config)
        ...     tools = await session.list_tools()
    """
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    # Extract URL from config
    # Cast to dict for TypedDict access
    config_dict: dict[str, object] = dict(config)
    url = config_dict.get("url")
    if not url or not isinstance(url, str):
        raise ValueError("MCP server config must include 'url' for HTTP/SSE transport")

    headers_raw = config_dict.get("headers", {})
    headers: dict[str, str] = headers_raw if isinstance(headers_raw, dict) else {}

    logger.debug(f"Connecting to MCP server at {url}")

    # Create HTTP transport and enter it into the exit stack
    http_transport = await exit_stack.enter_async_context(streamablehttp_client(url, headers=headers))

    read_stream, write_stream, _ = http_transport

    # Create and initialize session
    session = await exit_stack.enter_async_context(ClientSession(read_stream, write_stream))

    await session.initialize()

    logger.debug(f"Successfully connected to MCP server at {url}")

    return session


async def connect_all_mcp_servers(
    exit_stack: AsyncExitStack,
    mcp_servers: dict[str, MCPServerConfig],
) -> dict[str, ClientSession]:
    """Connect to all configured MCP servers.

    Args:
        exit_stack: AsyncExitStack for managing session lifecycles.
        mcp_servers: Dict mapping server names to their configurations.

    Returns:
        Dict mapping server names to initialized ClientSession objects.

    Raises:
        ValueError: If any server config is invalid.
        ImportError: If mcp package is not installed.
    """
    sessions: dict[str, ClientSession] = {}

    for name, config in mcp_servers.items():
        try:
            session = await connect_mcp_session(exit_stack, config)
            sessions[name] = session
            logger.info(f"Connected to MCP server '{name}'")
        except Exception as e:
            logger.error(f"Failed to connect to MCP server '{name}': {e}")
            raise

    return sessions


async def get_all_mcp_tools(sessions: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    """Get all tools from connected MCP sessions.

    Args:
        sessions: Dict mapping server names to ClientSession objects.

    Returns:
        List of tuples (server_name, session, mcp_tool) for each tool.
    """
    all_tools: list[tuple[str, Any, Any]] = []

    for server_name, session in sessions.items():
        try:
            tools_response = await session.list_tools()
            mcp_tools = tools_response.tools

            for tool in mcp_tools:
                all_tools.append((server_name, session, tool))

            logger.debug(f"MCP server '{server_name}' provides {len(mcp_tools)} tools: {[t.name for t in mcp_tools]}")

        except Exception as e:
            logger.error(f"Failed to list tools from MCP server '{server_name}': {e}")
            raise

    return all_tools
