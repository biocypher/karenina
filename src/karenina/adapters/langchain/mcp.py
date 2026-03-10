"""LangChain-specific MCP utilities using langchain-mcp-adapters.

This module provides MCP integration specifically for LangChain agents,
creating LangChain Tool objects that can be used with LangGraph agents.

For adapter-agnostic MCP utilities (session management, tool descriptions),
see karenina.utils.mcp.

Example:
    >>> mcp_urls = {"biocontext": "https://mcp.biocontext.ai/mcp/"}
    >>> client, tools = await acreate_mcp_client_and_tools(mcp_urls)
    >>> # tools are LangChain Tool objects ready for agent use
    >>> print(f"Loaded {len(tools)} tools")
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from contextlib import AsyncExitStack
from typing import Any

from karenina.exceptions import McpClientError, McpTimeoutError
from karenina.ports.agent import MCPHttpServerConfig
from karenina.utils.mcp.tools import apply_tool_description_overrides

logger = logging.getLogger(__name__)

# Re-export for backward compatibility
__all__ = [
    "acreate_mcp_client_and_tools",
    "acreate_persistent_mcp_tools",
    "create_mcp_client_and_tools",
    "cleanup_mcp_client",
    "apply_tool_description_overrides",
]


async def acreate_mcp_client_and_tools(
    mcp_urls_dict: dict[str, str],
    tool_filter: list[str] | None = None,
    tool_description_overrides: dict[str, str] | None = None,
) -> tuple[Any, list[Any]]:
    """Create an MCP client and fetch LangChain-compatible tools.

    This function uses langchain-mcp-adapters to create LangChain Tool objects
    that can be used directly with LangGraph agents.

    Args:
        mcp_urls_dict: Dictionary mapping tool names to MCP server URLs.
            Keys are tool names, values are server URLs.
        tool_filter: Optional list of tool names to include. If provided,
            only tools with names in this list will be returned.
        tool_description_overrides: Optional dict mapping tool names to custom
            descriptions. Used by GEPA optimization to test different
            tool descriptions.

    Returns:
        Tuple of (client, tools) where:
        - client: MultiServerMCPClient instance
        - tools: List of LangChain-compatible Tool objects

    Raises:
        ImportError: If langchain-mcp-adapters is not installed
        Exception: If MCP client creation or tool fetching fails

    Example:
        >>> mcp_urls = {"biocontext": "https://mcp.biocontext.ai/mcp/"}
        >>> client, tools = await acreate_mcp_client_and_tools(mcp_urls)
        >>> print(f"Loaded {len(tools)} tools")

        >>> # Filter to specific tools
        >>> client, tools = await create_mcp_client_and_tools(
        ...     mcp_urls,
        ...     tool_filter=["search_proteins", "get_interactions"]
        ... )
    """
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError as e:
        raise ImportError(
            "langchain-mcp-adapters package is required for MCP support. Install it with: uv add langchain-mcp-adapters"
        ) from e

    # Configure servers with streamable_http transport
    server_config = {}
    for tool_name, url in mcp_urls_dict.items():
        server_config[tool_name] = {"transport": "streamable_http", "url": url}

    try:
        # Create client and fetch tools with timeout
        client = MultiServerMCPClient(server_config)  # type: ignore[arg-type]

        # Add timeout to prevent hanging
        tools = await asyncio.wait_for(client.get_tools(), timeout=30.0)

        # Filter tools if tool_filter is provided
        if tool_filter is not None:
            allowed_tools = set(tool_filter)
            filtered_tools = []
            for tool in tools:
                current_tool_name: str | None = getattr(tool, "name", None)
                if current_tool_name and current_tool_name in allowed_tools:
                    filtered_tools.append(tool)
            tools = filtered_tools

        # Apply tool description overrides if provided (for GEPA optimization)
        if tool_description_overrides:
            tools = apply_tool_description_overrides(tools, tool_description_overrides)

        return client, tools

    except TimeoutError as e:
        raise McpTimeoutError(
            "MCP server connection timed out after 30 seconds. Check server availability and network connection.",
            timeout_seconds=30,
        ) from e
    except Exception as e:
        raise McpClientError(f"Failed to create MCP client or fetch tools: {e}") from e


async def acreate_persistent_mcp_tools(
    mcp_urls_dict: dict[str, str],
    exit_stack: AsyncExitStack,
    tool_filter: list[str] | None = None,
    tool_description_overrides: dict[str, str] | None = None,
) -> list[Any]:
    """Create LangChain tools bound to persistent MCP sessions.

    Unlike acreate_mcp_client_and_tools which creates tools that open a new
    MCP session per tool call, this function creates persistent sessions that
    stay alive for the duration of the exit_stack. This eliminates the per-call
    connection overhead (4 HTTP round-trips per tool call).

    Args:
        mcp_urls_dict: Dictionary mapping server names to MCP server URLs.
        exit_stack: AsyncExitStack that manages session lifecycles. Sessions
            remain open until the exit stack closes.
        tool_filter: Optional list of tool names to include.
        tool_description_overrides: Optional dict mapping tool names to custom
            descriptions.

    Returns:
        List of LangChain-compatible Tool objects bound to persistent sessions.

    Raises:
        ImportError: If required packages are not installed.
        McpTimeoutError: If connecting to MCP servers times out.
        McpClientError: If connection or tool loading fails.
    """
    try:
        from langchain_mcp_adapters.tools import load_mcp_tools
    except ImportError as e:
        raise ImportError(
            "langchain-mcp-adapters package is required for MCP support. Install it with: uv add langchain-mcp-adapters"
        ) from e

    from karenina.utils.mcp.client import connect_mcp_session

    all_tools: list[Any] = []

    try:
        for server_name, url in mcp_urls_dict.items():
            config: MCPHttpServerConfig = {"type": "http", "url": url}

            # Create persistent session (registered with exit_stack for cleanup)
            session = await asyncio.wait_for(
                connect_mcp_session(exit_stack, config),
                timeout=30.0,
            )

            # Load tools bound to this persistent session
            server_tools = await asyncio.wait_for(
                load_mcp_tools(session, server_name=server_name),
                timeout=30.0,
            )
            all_tools.extend(server_tools)

            logger.debug(f"Loaded {len(server_tools)} tools from '{server_name}' with persistent session")

    except TimeoutError as e:
        raise McpTimeoutError(
            "MCP server connection timed out after 30 seconds. Check server availability and network connection.",
            timeout_seconds=30,
        ) from e
    except Exception as e:
        raise McpClientError(f"Failed to create persistent MCP session or fetch tools: {e}") from e

    # Filter tools if tool_filter is provided
    if tool_filter is not None:
        allowed_tools = set(tool_filter)
        all_tools = [tool for tool in all_tools if getattr(tool, "name", None) in allowed_tools]

    # Apply tool description overrides if provided
    if tool_description_overrides:
        all_tools = apply_tool_description_overrides(all_tools, tool_description_overrides)

    logger.info(f"Created {len(all_tools)} persistent MCP tools across {len(mcp_urls_dict)} servers")

    return all_tools


def create_mcp_client_and_tools(
    mcp_urls_dict: dict[str, str],
    tool_filter: list[str] | None = None,
    tool_description_overrides: dict[str, str] | None = None,
) -> tuple[Any, list[Any]]:
    """Synchronous wrapper for creating MCP client and fetching tools.

    This function runs the async MCP client creation using the shared portal
    if available (from parallel verification), otherwise falls back to
    asyncio.run().

    Args:
        mcp_urls_dict: Dictionary mapping tool names to MCP server URLs
        tool_filter: Optional list of tool names to include
        tool_description_overrides: Optional dict mapping tool names to custom
            descriptions. Used by GEPA optimization.

    Returns:
        Tuple of (client, tools) as in create_mcp_client_and_tools

    Raises:
        ImportError: If langchain-mcp-adapters is not installed.
        McpTimeoutError: If MCP client creation times out.
        McpClientError: If MCP client creation or tool fetching fails.
    """
    from typing import cast

    # Try to use the shared portal if available (from parallel verification)
    try:
        from karenina.benchmark.verification.executor import get_async_portal

        portal = get_async_portal()
        if portal is not None:
            portal_result = portal.call(
                acreate_mcp_client_and_tools, mcp_urls_dict, tool_filter, tool_description_overrides
            )
            return cast(tuple[Any, list[Any]], portal_result)
    except ImportError:
        pass

    # Check if we're already in an async context
    try:
        asyncio.get_running_loop()

        # Use ThreadPoolExecutor to avoid nested event loop issues
        def run_in_thread() -> tuple[Any, list[Any]]:
            return asyncio.run(acreate_mcp_client_and_tools(mcp_urls_dict, tool_filter, tool_description_overrides))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            try:
                return future.result(timeout=45)
            except TimeoutError as e:
                raise McpTimeoutError(
                    "MCP client creation timed out after 45 seconds",
                    timeout_seconds=45,
                ) from e
    except RuntimeError:
        # No event loop running, safe to create new one
        pass

    # Create new event loop and run the async function
    return asyncio.run(acreate_mcp_client_and_tools(mcp_urls_dict, tool_filter, tool_description_overrides))


def cleanup_mcp_client(client: Any) -> None:
    """Clean up MCP client connections gracefully.

    This function attempts to close MCP client connections that might keep
    background threads or event loops alive after usage is complete.

    Args:
        client: MCP client instance (MultiServerMCPClient or similar)

    Example:
        >>> client, tools = create_mcp_client_and_tools(mcp_urls)
        >>> # ... use client ...
        >>> cleanup_mcp_client(client)
    """
    if client is None:
        return

    try:
        # Try synchronous close first
        if hasattr(client, "close"):
            client.close()
            logger.debug("MCP client closed (sync)")
            return

        # Try async close
        if hasattr(client, "aclose"):
            try:
                # Try to use the shared portal if available
                from karenina.benchmark.verification.executor import get_async_portal

                portal = get_async_portal()
                if portal is not None:
                    portal.call(client.aclose)
                    logger.debug("MCP client closed (via portal)")
                    return
            except (ImportError, Exception):
                pass

            try:
                # Check if we're in an async context
                loop = asyncio.get_running_loop()
                # We're in an async context - schedule close
                loop.create_task(client.aclose())
                logger.debug("MCP client close scheduled (async)")
            except RuntimeError:
                # No event loop running - run async close in new loop
                asyncio.run(client.aclose())
                logger.debug("MCP client closed (async new loop)")
            return

        # Try __exit__ if it's a context manager
        if hasattr(client, "__exit__"):
            client.__exit__(None, None, None)
            logger.debug("MCP client exited (context manager)")
            return

    except Exception as e:
        logger.warning(f"Failed to cleanup MCP client: {e}")

    # If no cleanup method found, log a warning
    logger.debug("No cleanup method found for MCP client")
