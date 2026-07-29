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
import json
import logging
from contextlib import AsyncExitStack
from typing import Any, cast

from langchain_core.tools import BaseTool
from pydantic import Field

from karenina.exceptions import McpClientError, McpTimeoutError
from karenina.ports.agent import MCPHttpServerConfig
from karenina.utils.mcp.tools import apply_tool_description_overrides

logger = logging.getLogger(__name__)

# Re-export for backward compatibility
__all__ = [
    "acleanup_mcp_client",
    "acreate_mcp_client_and_tools",
    "acreate_persistent_mcp_tools",
    "create_mcp_client_and_tools",
    "cleanup_mcp_client",
    "apply_tool_description_overrides",
]

# Wall-clock bound for awaited MCP client teardown from sync contexts.
MCP_CLEANUP_TIMEOUT = 30.0


def _schema_dict(args_schema: Any) -> dict[str, Any] | None:
    if args_schema is None:
        return None
    if isinstance(args_schema, dict):
        return args_schema
    if hasattr(args_schema, "model_json_schema"):
        schema = args_schema.model_json_schema()
        return cast(dict[str, Any], schema) if isinstance(schema, dict) else None
    if hasattr(args_schema, "schema"):
        schema = args_schema.schema()
        return cast(dict[str, Any], schema) if isinstance(schema, dict) else None
    return None


def _resolve_schema_ref(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    ref = schema.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return schema

    current: Any = root
    for part in ref.removeprefix("#/").split("/"):
        if not isinstance(current, dict):
            return schema
        current = current.get(part)
    return current if isinstance(current, dict) else schema


def _schema_accepts_type(schema: dict[str, Any], root: dict[str, Any], expected: str) -> bool:
    schema = _resolve_schema_ref(schema, root)
    schema_type = schema.get("type")
    if schema_type == expected or (isinstance(schema_type, list) and expected in schema_type):
        return True

    for key in ("anyOf", "oneOf", "allOf"):
        variants = schema.get(key)
        if isinstance(variants, list) and any(
            isinstance(item, dict) and _schema_accepts_type(item, root, expected) for item in variants
        ):
            return True
    return False


def _expected_json_container(
    schema: dict[str, Any], root: dict[str, Any]
) -> type[list[Any]] | type[dict[str, Any]] | None:
    if _schema_accepts_type(schema, root, "array"):
        return list
    if _schema_accepts_type(schema, root, "object"):
        return dict
    return None


def _coerce_json_string_tool_args(
    tool_input: str | dict[str, Any], args_schema: Any, tool_name: str
) -> str | dict[str, Any]:
    """Decode JSON-looking strings only where the tool schema expects containers."""
    if not isinstance(tool_input, dict):
        return tool_input

    schema = _schema_dict(args_schema)
    if schema is None:
        return tool_input

    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return tool_input

    coerced = dict(tool_input)
    for name, value in tool_input.items():
        if not isinstance(value, str):
            continue

        field_schema = properties.get(name)
        if not isinstance(field_schema, dict):
            continue

        expected_type = _expected_json_container(field_schema, schema)
        if expected_type is None:
            continue

        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, expected_type):
            coerced[name] = parsed
            logger.debug(
                "Coerced JSON string argument %s.%s to %s before tool validation",
                tool_name,
                name,
                expected_type.__name__,
            )

    return coerced


class _JsonStringCoercingTool(BaseTool):
    wrapped_tool: Any = Field(exclude=True)

    def _parse_input(
        self,
        tool_input: str | dict[str, Any],
        tool_call_id: str | None,
    ) -> str | dict[str, Any]:
        coerced = _coerce_json_string_tool_args(tool_input, self.args_schema, self.name)
        return super()._parse_input(coerced, tool_call_id)

    def _tool_input_from_call(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | dict[str, Any]:
        kwargs.pop("run_manager", None)
        if args and not kwargs:
            return cast(str | dict[str, Any], args[0])
        if args:
            return {"args": list(args), **kwargs}
        return kwargs

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        return self.wrapped_tool.invoke(self._tool_input_from_call(args, kwargs))

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        return await self.wrapped_tool.ainvoke(self._tool_input_from_call(args, kwargs))


def _wrap_json_string_args_tool(tool: Any) -> Any:
    if isinstance(tool, _JsonStringCoercingTool) or not isinstance(tool, BaseTool):
        return tool
    return _JsonStringCoercingTool(
        wrapped_tool=tool,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        return_direct=tool.return_direct,
        tags=tool.tags,
        metadata=tool.metadata,
        handle_tool_error=tool.handle_tool_error,
        handle_validation_error=tool.handle_validation_error,
        # The wrapped tool's public invoke/ainvoke returns final message
        # content, not the internal content/artifact tuple.
        response_format="content",
    )


def _wrap_json_string_args_tools(tools: list[Any]) -> list[Any]:
    return [_wrap_json_string_args_tool(tool) for tool in tools]


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
        client = MultiServerMCPClient(cast(Any, server_config))

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

        tools = _wrap_json_string_args_tools(tools)

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

    all_tools = _wrap_json_string_args_tools(all_tools)

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


async def acleanup_mcp_client(client: Any) -> None:
    """Clean up MCP client connections, awaiting async teardown.

    Prefer this over cleanup_mcp_client whenever you are already inside an
    async context: it awaits client.aclose() directly, so teardown is
    deterministic and resources are released before the function returns.

    Args:
        client: MCP client instance (MultiServerMCPClient or similar)
    """
    if client is None:
        return

    if hasattr(client, "aclose"):
        try:
            await client.aclose()
            logger.debug("MCP client closed (awaited aclose)")
        except Exception:
            logger.warning("MCP client aclose() failed during cleanup", exc_info=True)
        return

    if hasattr(client, "close"):
        try:
            client.close()
            logger.debug("MCP client closed (sync close)")
        except Exception:
            logger.warning("MCP client close() failed during cleanup", exc_info=True)
        return

    if hasattr(client, "__exit__"):
        try:
            client.__exit__(None, None, None)
            logger.debug("MCP client exited (context manager)")
        except Exception:
            logger.warning("MCP client __exit__ failed during cleanup", exc_info=True)
        return

    logger.debug("No cleanup method found for MCP client")


def _close_async_client(client: Any, loop: asyncio.AbstractEventLoop | None) -> None:
    """Run client.aclose() to completion from a synchronous context.

    Dispatch order: the own-running-loop guard comes first (the close
    cannot block there without deadlocking, so a warning directs callers
    to acleanup_mcp_client). Then the explicitly provided loop (the
    client's home loop) via run_coroutine_threadsafe, then the shared
    portal via a bounded start_task_soon, then asyncio.run(). Every
    dispatch is awaited with a wall-clock bound. The coroutine is never
    created and dropped.
    """
    # Own-running-loop guard first: any other branch would block this
    # thread's loop (or raise from inside it), so this case must short
    # circuit before portal or loop dispatch is even considered.
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        logger.warning(
            "cleanup_mcp_client called from a running event loop. The close cannot "
            "block here without deadlocking, so the client was NOT closed. "
            "Use 'await acleanup_mcp_client(client)' from async contexts instead."
        )
        return

    # Prefer the client's home loop when the caller provided it: the
    # client's resources live on that loop, so closing there is the most
    # correct dispatch.
    if loop is not None and loop.is_running():
        future = asyncio.run_coroutine_threadsafe(client.aclose(), loop)
        future.result(timeout=MCP_CLEANUP_TIMEOUT)
        logger.debug("MCP client closed (awaited via run_coroutine_threadsafe)")
        return

    portal = None
    try:
        from karenina.benchmark.verification.executor import get_async_portal

        portal = get_async_portal()
    except ImportError:
        logger.debug("Verification executor unavailable, skipping portal-based MCP cleanup")

    if portal is not None:
        # Bound the portal dispatch so a stalled close cannot wedge the
        # calling thread (mirrors fetch_tool_descriptions in
        # utils/mcp/tools.py).
        portal_future = portal.start_task_soon(client.aclose)
        try:
            portal_future.result(timeout=MCP_CLEANUP_TIMEOUT)
        except concurrent.futures.TimeoutError:
            portal_future.cancel()
            raise
        logger.debug("MCP client closed (awaited via portal)")
        return

    asyncio.run(client.aclose())
    logger.debug("MCP client closed (awaited via asyncio.run)")


def cleanup_mcp_client(client: Any, loop: asyncio.AbstractEventLoop | None = None) -> None:
    """Clean up MCP client connections gracefully from a synchronous context.

    Async teardown is always awaited to completion with a wall-clock
    bound (via run_coroutine_threadsafe against the provided loop, the
    shared portal, or asyncio.run), never scheduled and dropped. If this
    function is called on a running event loop's own thread it cannot
    block on the close, so it logs a warning and returns without
    closing. Call acleanup_mcp_client from async contexts.

    Args:
        client: MCP client instance (MultiServerMCPClient or similar)
        loop: Optional event loop the client belongs to. When the loop is
            running in another thread, the close is submitted to it with
            run_coroutine_threadsafe and awaited with a timeout.

    Example:
        >>> client, tools = create_mcp_client_and_tools(mcp_urls)
        >>> # ... use client ...
        >>> cleanup_mcp_client(client)
    """
    if client is None:
        return

    try:
        if hasattr(client, "close"):
            client.close()
            logger.debug("MCP client closed (sync)")
            return

        if hasattr(client, "aclose"):
            _close_async_client(client, loop)
            return

        if hasattr(client, "__exit__"):
            client.__exit__(None, None, None)
            logger.debug("MCP client exited (context manager)")
            return

    except Exception:
        logger.warning("Failed to cleanup MCP client", exc_info=True)
        return

    logger.debug("No cleanup method found for MCP client")
