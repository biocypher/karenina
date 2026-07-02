"""MCP tool description utilities using the core mcp package.

This module provides utilities for fetching tool descriptions from MCP servers
using the core `mcp` package (not langchain-mcp-adapters).

Used primarily by GEPA optimization to get and modify tool descriptions.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from contextlib import AsyncExitStack
from typing import Any

from karenina.exceptions import McpTimeoutError

logger = logging.getLogger(__name__)


def apply_tool_description_overrides(
    tools: list[Any],
    overrides: dict[str, str],
) -> list[Any]:
    """Apply description overrides to tools.

    Used by GEPA optimization to test different tool descriptions.

    Args:
        tools: List of tool objects (LangChain Tool or MCP Tool)
        overrides: Dict mapping tool names to new descriptions

    Returns:
        Modified tools list with updated descriptions
    """
    for tool in tools:
        tool_name = getattr(tool, "name", None)
        if tool_name and tool_name in overrides:
            tool.description = overrides[tool_name]
    return tools


async def afetch_tool_descriptions(
    mcp_urls_dict: dict[str, str],
    tool_filter: list[str] | None = None,
) -> dict[str, str]:
    """Fetch tool descriptions from MCP servers using the core mcp package.

    This function connects to MCP servers and retrieves the descriptions
    for all available tools. Useful for getting seed descriptions for
    GEPA optimization.

    Args:
        mcp_urls_dict: Dictionary mapping server names to MCP server URLs
        tool_filter: Optional list of tool names to include

    Returns:
        Dict mapping tool names to their descriptions

    Example:
        >>> descriptions = await fetch_tool_descriptions(
        ...     {"biocontext": "https://mcp.biocontext.ai/mcp/"}
        ... )
        >>> print(descriptions)
        {'search_proteins': 'Search for protein information...', ...}
    """
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    from karenina.adapters._timeouts import MCP_CONNECT_TIMEOUT

    descriptions: dict[str, str] = {}

    async with AsyncExitStack() as stack:
        for server_name, url in mcp_urls_dict.items():
            try:
                # Connect to MCP server. The inner per-server bound is
                # derived from the same source as the outer dispatch bound
                # in fetch_tool_descriptions and fires first for a single
                # server. With multiple servers the per-server bounds sum,
                # so the outer dispatch bound may fire first instead.
                http_transport = await asyncio.wait_for(
                    stack.enter_async_context(streamablehttp_client(url)),
                    timeout=MCP_CONNECT_TIMEOUT,
                )
                read_stream, write_stream, _ = http_transport

                session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
                await session.initialize()

                # List tools from this server
                tools_response = await session.list_tools()
                mcp_tools = tools_response.tools

                # Extract descriptions
                for tool in mcp_tools:
                    tool_name = tool.name
                    tool_desc = tool.description or ""

                    # Apply filter if provided
                    if tool_filter is not None and tool_name not in tool_filter:
                        continue

                    descriptions[tool_name] = tool_desc

                logger.debug(f"Fetched {len(mcp_tools)} tools from MCP server '{server_name}'")

            except TimeoutError as e:
                raise McpTimeoutError(
                    f"MCP server '{server_name}' connection timed out after {MCP_CONNECT_TIMEOUT:.0f} seconds.",
                    server_name=server_name,
                    timeout_seconds=int(MCP_CONNECT_TIMEOUT),
                ) from e
            except Exception as e:
                logger.error(f"Failed to fetch tools from MCP server '{server_name}': {e}")
                raise

    logger.debug(f"Fetched descriptions for {len(descriptions)} total tools")
    return descriptions


def fetch_tool_descriptions(
    mcp_urls_dict: dict[str, str],
    tool_filter: list[str] | None = None,
) -> dict[str, str]:
    """Synchronous wrapper for afetch_tool_descriptions.

    Uses the same async handling pattern as other sync wrappers in karenina,
    supporting both the shared BlockingPortal (when available from parallel
    verification) and fallback to asyncio.run().

    Args:
        mcp_urls_dict: Dictionary mapping server names to MCP server URLs
        tool_filter: Optional list of tool names to include

    Returns:
        Dict mapping tool names to their descriptions
    """
    from karenina.adapters._timeouts import MCP_TOOL_FETCH_FLOOR, compute_sync_wrapper_timeout

    # No model config is available here, so the dispatch bound is the
    # historical floor for MCP tool description fetches. The inner
    # per-server connect bound (MCP_CONNECT_TIMEOUT) is derived from the
    # same source and fires first for a single server. With multiple
    # servers the per-server bounds sum, so this dispatch bound may fire
    # first instead.
    fetch_timeout = compute_sync_wrapper_timeout(None, floor=MCP_TOOL_FETCH_FLOOR)

    # Try to use the shared portal if available (from parallel verification)
    try:
        from karenina.benchmark.verification.executor import get_async_portal

        portal = get_async_portal()
        if portal is not None:
            # Bound the portal dispatch so a stalled coroutine inside the
            # portal's event loop cannot wedge the worker thread forever
            # (mirrors the langchain parser's bounded portal dispatch).
            portal_future = portal.start_task_soon(afetch_tool_descriptions, mcp_urls_dict, tool_filter)
            try:
                return portal_future.result(timeout=fetch_timeout)
            except concurrent.futures.TimeoutError as e:
                portal_future.cancel()
                raise McpTimeoutError(
                    f"Fetch tool descriptions timed out after {fetch_timeout:.0f} seconds",
                    timeout_seconds=int(fetch_timeout),
                ) from e
    except ImportError:
        pass

    # Check if we're already in an async context
    try:
        asyncio.get_running_loop()

        # Use ThreadPoolExecutor to avoid nested event loop issues
        def run_in_thread() -> dict[str, str]:
            return asyncio.run(afetch_tool_descriptions(mcp_urls_dict, tool_filter))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            try:
                return future.result(timeout=fetch_timeout)
            except TimeoutError as e:
                raise McpTimeoutError(
                    f"Fetch tool descriptions timed out after {fetch_timeout:.0f} seconds",
                    timeout_seconds=int(fetch_timeout),
                ) from e
    except RuntimeError:
        pass

    # Create new event loop and run
    return asyncio.run(afetch_tool_descriptions(mcp_urls_dict, tool_filter))
