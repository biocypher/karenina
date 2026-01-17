"""MCP (Model Context Protocol) utilities for Karenina.

This module provides utilities for integrating MCP servers with Karenina's
LLM interface, including agent response harmonization and MCP client management.

Message harmonization and extraction functions are implemented in mcp_messages.py
and re-exported here for backward compatibility.
"""

import asyncio
import logging
from typing import Any

# Re-export message harmonization functions for backward compatibility
from karenina.infrastructure.llm.mcp_messages import (
    _extract_agent_trace,
    _format_message_for_trace,
    _is_summary_message,
    extract_final_ai_message,
    extract_final_ai_message_from_response,
    harmonize_agent_response,
)

logger = logging.getLogger(__name__)

# Suppress unused import warnings - these are re-exports for backward compatibility
__all__ = [
    "harmonize_agent_response",
    "_format_message_for_trace",
    "_is_summary_message",
    "_extract_agent_trace",
    "extract_final_ai_message_from_response",
    "extract_final_ai_message",
    "create_mcp_client_and_tools",
    "sync_create_mcp_client_and_tools",
    "fetch_tool_descriptions",
    "sync_fetch_tool_descriptions",
    "cleanup_mcp_client",
    "_apply_tool_description_overrides",
]


def _apply_tool_description_overrides(
    tools: list[Any],
    overrides: dict[str, str],
) -> list[Any]:
    """Apply description overrides to MCP tools.

    Used by GEPA optimization to test different tool descriptions.

    Args:
        tools: List of LangChain tool objects
        overrides: Dict mapping tool names to new descriptions

    Returns:
        Modified tools list with updated descriptions
    """
    for tool in tools:
        tool_name = getattr(tool, "name", None)
        if tool_name and tool_name in overrides:
            tool.description = overrides[tool_name]
    return tools


async def create_mcp_client_and_tools(
    mcp_urls_dict: dict[str, str],
    tool_filter: list[str] | None = None,
    tool_description_overrides: dict[str, str] | None = None,
) -> tuple[Any, list[Any]]:
    """
    Create an MCP client and fetch tools from the specified servers.

    Args:
        mcp_urls_dict: Dictionary mapping tool names to MCP server URLs.
                      Keys are tool names, values are server URLs.
        tool_filter: Optional list of tool names to include. If provided,
                    only tools with names in this list will be returned.
                    Converted to set internally for efficiency.
        tool_description_overrides: Optional dict mapping tool names to custom
                    descriptions. Used by GEPA optimization to test different
                    tool descriptions.

    Returns:
        Tuple of (client, tools) where:
        - client: MultiServerMCPClient instance
        - tools: List of LangChain-compatible tools fetched from servers,
                 optionally filtered by tool_filter

    Raises:
        ImportError: If langchain-mcp-adapters is not installed
        Exception: If MCP client creation or tool fetching fails

    Examples:
        >>> mcp_urls = {"biocontext": "https://mcp.biocontext.ai/mcp/"}
        >>> client, tools = await create_mcp_client_and_tools(mcp_urls)
        >>> print(f"Loaded {len(tools)} tools")

        >>> # Filter to specific tools
        >>> client, tools = await create_mcp_client_and_tools(mcp_urls, ["search_proteins", "get_interactions"])
        >>> print(f"Loaded {len(tools)} filtered tools")

        >>> # Override tool descriptions (for GEPA optimization)
        >>> overrides = {"search_proteins": "Search for protein information by name or ID."}
        >>> client, tools = await create_mcp_client_and_tools(mcp_urls, tool_description_overrides=overrides)
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
                # Check if tool has a name attribute and if it's in the allowed set
                current_tool_name: str | None = getattr(tool, "name", None)
                if current_tool_name and current_tool_name in allowed_tools:
                    filtered_tools.append(tool)
            tools = filtered_tools

        # Apply tool description overrides if provided (for GEPA optimization)
        if tool_description_overrides:
            tools = _apply_tool_description_overrides(tools, tool_description_overrides)

        return client, tools

    except TimeoutError as e:
        raise Exception(
            "MCP server connection timed out after 30 seconds. Check server availability and network connection."
        ) from e
    except Exception as e:
        raise Exception(f"Failed to create MCP client or fetch tools: {e}") from e


def sync_create_mcp_client_and_tools(
    mcp_urls_dict: dict[str, str],
    tool_filter: list[str] | None = None,
    tool_description_overrides: dict[str, str] | None = None,
) -> tuple[Any, list[Any]]:
    """
    Synchronous wrapper for creating MCP client and fetching tools.

    This function runs the async MCP client creation using the shared portal
    if available (from parallel verification), otherwise falls back to
    asyncio.run().

    Args:
        mcp_urls_dict: Dictionary mapping tool names to MCP server URLs
        tool_filter: Optional list of tool names to include. If provided,
                    only tools with names in this list will be returned.
        tool_description_overrides: Optional dict mapping tool names to custom
                    descriptions. Used by GEPA optimization.

    Returns:
        Tuple of (client, tools) as in create_mcp_client_and_tools

    Raises:
        Same exceptions as create_mcp_client_and_tools
    """
    import threading

    # Try to use the shared portal if available (from parallel verification)
    try:
        from typing import cast

        from karenina.benchmark.verification.batch_runner import get_async_portal

        portal = get_async_portal()
        if portal is not None:
            # Use the shared BlockingPortal for proper event loop management
            portal_result = portal.call(
                create_mcp_client_and_tools, mcp_urls_dict, tool_filter, tool_description_overrides
            )
            return cast(tuple[Any, list[Any]], portal_result)
    except ImportError:
        # batch_runner not available, fall back to asyncio.run()
        pass

    # Check if we're already in an async context
    try:
        current_loop = asyncio.get_running_loop()
        if current_loop:
            # We're in an async context, need to run in a separate thread
            def run_in_thread() -> tuple[Any, list[Any]]:
                return asyncio.run(create_mcp_client_and_tools(mcp_urls_dict, tool_filter, tool_description_overrides))

            result: list[tuple[Any, list[Any]] | None] = [None]
            exception: list[Exception | None] = [None]

            def thread_target() -> None:
                try:
                    result[0] = run_in_thread()
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=thread_target)
            thread.start()
            thread.join(timeout=45.0)  # 45 second total timeout

            if thread.is_alive():
                raise Exception("MCP client creation timed out after 45 seconds")

            if exception[0]:
                raise exception[0]

            if result[0] is None:
                raise Exception("MCP client creation failed without exception")

            return result[0]
    except RuntimeError:
        # No event loop running, safe to create new one
        pass

    # Create new event loop and run the async function
    return asyncio.run(create_mcp_client_and_tools(mcp_urls_dict, tool_filter, tool_description_overrides))


async def fetch_tool_descriptions(
    mcp_urls_dict: dict[str, str],
    tool_filter: list[str] | None = None,
) -> dict[str, str]:
    """
    Fetch tool descriptions from MCP servers without creating a full agent.

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
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError as e:
        raise ImportError(
            "langchain-mcp-adapters is required for MCP support. Install with: uv add langchain-mcp-adapters"
        ) from e

    # Configure servers
    server_config = {}
    for server_name, url in mcp_urls_dict.items():
        server_config[server_name] = {"transport": "streamable_http", "url": url}

    client = MultiServerMCPClient(server_config)  # type: ignore[arg-type]

    try:
        tools = await asyncio.wait_for(client.get_tools(), timeout=30.0)

        # Apply filter if provided
        if tool_filter is not None:
            allowed = set(tool_filter)
            tools = [t for t in tools if getattr(t, "name", None) in allowed]

        # Extract descriptions
        descriptions: dict[str, str] = {}
        for tool in tools:
            name = getattr(tool, "name", None)
            desc = getattr(tool, "description", "")
            if name:
                descriptions[name] = desc

        logger.debug(f"Fetched descriptions for {len(descriptions)} tools")
        return descriptions

    except TimeoutError as e:
        raise Exception("MCP server connection timed out after 30 seconds.") from e
    finally:
        # Clean up client
        if hasattr(client, "aclose"):
            import contextlib

            with contextlib.suppress(Exception):
                await client.aclose()


def sync_fetch_tool_descriptions(
    mcp_urls_dict: dict[str, str],
    tool_filter: list[str] | None = None,
) -> dict[str, str]:
    """
    Synchronous wrapper for fetch_tool_descriptions.

    Uses the same async handling pattern as sync_create_mcp_client_and_tools.

    Args:
        mcp_urls_dict: Dictionary mapping server names to MCP server URLs
        tool_filter: Optional list of tool names to include

    Returns:
        Dict mapping tool names to their descriptions
    """
    import threading

    # Try to use the shared portal if available
    try:
        from karenina.benchmark.verification.batch_runner import get_async_portal

        portal = get_async_portal()
        if portal is not None:
            return portal.call(fetch_tool_descriptions, mcp_urls_dict, tool_filter)
    except ImportError:
        pass

    # Check if we're already in an async context
    try:
        current_loop = asyncio.get_running_loop()
        if current_loop:
            # Run in a separate thread
            result: list[dict[str, str] | None] = [None]
            exception: list[Exception | None] = [None]

            def thread_target() -> None:
                try:
                    result[0] = asyncio.run(fetch_tool_descriptions(mcp_urls_dict, tool_filter))
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=thread_target)
            thread.start()
            thread.join(timeout=45.0)

            if thread.is_alive():
                raise Exception("Fetch tool descriptions timed out after 45 seconds")

            if exception[0]:
                raise exception[0]

            return result[0] or {}
    except RuntimeError:
        pass

    # Create new event loop and run
    return asyncio.run(fetch_tool_descriptions(mcp_urls_dict, tool_filter))


def cleanup_mcp_client(client: Any) -> None:
    """
    Clean up MCP client connections gracefully.

    This function attempts to close MCP client connections that might keep
    background threads or event loops alive after usage is complete.

    Args:
        client: MCP client instance (MultiServerMCPClient or similar)

    Examples:
        >>> client, tools = sync_create_mcp_client_and_tools(mcp_urls)
        >>> # ... use client ...
        >>> cleanup_mcp_client(client)
    """
    import asyncio
    import logging

    logger = logging.getLogger(__name__)

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
                from karenina.benchmark.verification.batch_runner import get_async_portal

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
