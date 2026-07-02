"""Search tool abstraction for deep-judgment excerpt validation.

This module provides a flexible abstraction layer for search tools used in
search-enhanced deep-judgment. The abstraction supports:

1. Built-in search tools (e.g., Tavily) via string-based factory
2. Custom langchain tools passed directly as callables
3. MCP tool integration with native async support

The design prioritizes extensibility and allows users to:
- Use default Tavily search without configuration
- Inject custom langchain tools at runtime
- Replace with MCP-provided search tools in the future

Search Tool Interface:
    All search tools follow a structured contract:
    - Input: str or list[str] (query or queries)
    - Output: list[SearchResultItem] or list[list[SearchResultItem]]

    Each SearchResultItem has:
    - title: str (result title)
    - content: str (result content/snippet)
    - url: str (source URL)

Example Usage:
    # Built-in tool (string-based)
    search_tool = create_search_tool("tavily")

    # Search single query
    results = search_tool("What is the capital of France?")
    print(results)  # list[SearchResultItem]
    for item in results:
        print(f"{item.title}: {item.content}")

    # Search multiple queries
    results = search_tool(["Query 1", "Query 2", "Query 3"])
    print(results)  # list[list[SearchResultItem]]

    # Custom langchain tool (function-based)
    def my_search(query: str) -> str:
        return json.dumps([{"title": "Result", "content": "...", "url": "..."}])

    search_tool = create_search_tool(my_search)
    result = search_tool("test query")
"""

import asyncio
import concurrent.futures
import contextlib
import inspect
import logging
from collections.abc import Callable
from typing import Any

from karenina.schemas import SearchResultItem

from .search_helpers import parse_tool_output as _parse_tool_output

logger = logging.getLogger(__name__)

# Type alias for search tool callable
SearchToolCallable = Callable[[str | list[str]], list[SearchResultItem] | list[list[SearchResultItem]]]


def create_search_tool(
    tool: str | Any,
    **kwargs: Any,
) -> SearchToolCallable:
    """Factory function to create search tools.

    This factory supports two modes:

    1. String-based (built-in tools):
       create_search_tool("tavily") -> Tavily search function

    2. Langchain tool injection:
       create_search_tool(my_langchain_tool) -> Wrapped search function

    The dual-mode design allows:
    - Simple usage with built-in tools (just pass "tavily")
    - Advanced usage with custom langchain tools (pass tool instance)
    - Future MCP integration (pass MCP tool as langchain tool)

    All search tools follow the same interface:
    - Input: str or list[str]
    - Output: str or list[str]

    Args:
        tool: Either a string name ("tavily") or a langchain tool instance/callable
        **kwargs: Additional arguments passed to built-in tool constructors

    Returns:
        Search function that takes str|list[str] and returns str|list[str]

    Raises:
        ValueError: If tool name is unknown or tool instance is invalid

    Examples:
        # Built-in Tavily tool
        search = create_search_tool("tavily")
        result = search("Python programming")

        # Custom langchain tool
        from langchain.tools import Tool
        custom_tool = Tool(name="my_search", func=lambda q: f"Results for: {q}")
        search = create_search_tool(custom_tool)
        result = search("test query")

        # Batch search
        results = search(["query1", "query2", "query3"])
    """
    # Case 1: String-based built-in tool
    if isinstance(tool, str):
        tool_name = tool.lower()

        if tool_name == "tavily":
            # Import here to avoid dependency issues if Tavily not installed
            try:
                from .search_tavily import create_tavily_search_tool

                logger.info("Creating Tavily search tool")
                return create_tavily_search_tool(**kwargs)  # type: ignore[no-any-return]
            except ImportError as e:
                raise ValueError(
                    f"Tavily search tool requires 'langchain-community' and 'tavily-python'. "
                    f"Install with: pip install langchain-community tavily-python\n"
                    f"Original error: {e}"
                ) from e
        else:
            raise ValueError(
                f"Unknown built-in search tool: '{tool_name}'. "
                f"Supported tools: 'tavily'. "
                f"Or pass a langchain tool instance directly."
            )

    # Case 2: Langchain tool instance or callable
    else:
        logger.info(f"Creating wrapper for custom tool: {type(tool)}")
        return _wrap_langchain_tool(tool)


def _wrap_langchain_tool(langchain_tool: Any) -> SearchToolCallable:
    """Wrap a langchain tool to conform to search tool interface.

    Supports both sync and async tools (including MCP tools via langchain-mcp-adapters).

    Args:
        langchain_tool: Any langchain tool that implements invoke(), run(), arun(), or is callable

    Returns:
        Search function that takes str|list[str] and returns list[SearchResultItem]|list[list[SearchResultItem]]

    Raises:
        ValueError: If tool doesn't have required methods
    """
    # Determine which method to use and whether it's async
    # Priority: ainvoke > arun > invoke > run > callable
    call_method = None
    is_async = False

    if hasattr(langchain_tool, "ainvoke"):
        call_method = "ainvoke"
        is_async = True
    elif hasattr(langchain_tool, "arun"):
        call_method = "arun"
        is_async = True
    elif hasattr(langchain_tool, "invoke"):
        call_method = "invoke"
    elif hasattr(langchain_tool, "run"):
        call_method = "run"
    elif callable(langchain_tool):
        call_method = "call"
        # Check if the callable itself is async
        is_async = inspect.iscoroutinefunction(langchain_tool)
    else:
        raise ValueError(
            f"Langchain tool must have 'invoke', 'run', 'ainvoke', 'arun' method, or be callable. "
            f"Got: {type(langchain_tool)}"
        )

    logger.info(f"Wrapped langchain tool with method: {call_method} (async={is_async})")

    # If tool is async, create executor for running in separate thread
    executor = None
    if is_async:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=5, thread_name_prefix="search_tool")
        logger.info("Created thread pool executor for async tool")

    def search_function(query: str | list[str]) -> list[SearchResultItem] | list[list[SearchResultItem]]:
        """Execute search using wrapped langchain tool.

        Args:
            query: Single query string or list of query strings

        Returns:
            Single list of SearchResultItem (if query is str) or list of lists (if query is list)
        """
        # Handle batch queries
        if isinstance(query, list):
            results = []
            for q in query:
                try:
                    result = _invoke_tool(langchain_tool, call_method, q, is_async, executor)
                    parsed = _parse_tool_output(result)
                    results.append(parsed)
                except Exception as e:
                    logger.warning(f"Search failed for query '{q[:50]}...': {e}")
                    results.append([])
            return results

        # Handle single query
        else:
            try:
                result = _invoke_tool(langchain_tool, call_method, query, is_async, executor)
                return _parse_tool_output(result)
            except Exception as e:
                logger.warning(f"Search failed for query '{query[:50]}...': {e}")
                return []

    return search_function


def _invoke_tool(
    tool: Any,
    method: str,
    query: str,
    is_async: bool = False,
    executor: concurrent.futures.ThreadPoolExecutor | None = None,
) -> Any:
    """Invoke a langchain tool with the appropriate method.

    Handles both sync and async tools. For async tools, runs them in a separate
    thread with a new event loop to avoid conflicts with existing loops.

    Args:
        tool: The langchain tool instance
        method: Method to use ("invoke", "run", "ainvoke", "arun", or "call")
        query: Query string
        is_async: Whether the tool method is async
        executor: Thread pool executor for async tools (required if is_async=True)

    Returns:
        Raw search result (any type - will be parsed by _parse_tool_output)
    """
    if not is_async:
        # Synchronous invocation
        if method == "invoke":
            return tool.invoke(query)
        elif method == "run":
            return tool.run(query)
        else:  # callable
            return tool(query)
    else:
        from karenina.adapters._timeouts import SEARCH_PROVIDER_FLOOR, compute_sync_wrapper_timeout
        from karenina.benchmark.verification.async_lifecycle import get_async_portal

        # No model config is available here, so the bound is the historical
        # search floor.
        dispatch_timeout = compute_sync_wrapper_timeout(None, floor=SEARCH_PROVIDER_FLOOR)

        if method == "ainvoke":
            async_fn = tool.ainvoke
        elif method == "arun":
            async_fn = tool.arun
        else:  # async callable
            async_fn = tool

        # Loop affinity: prefer the shared BlockingPortal when one is active
        # for this thread, so the async tool runs on the same event loop as
        # the adapters (httpx clients opened on the portal loop stay usable).
        # Guard against nesting: if a loop is already running in this thread,
        # blocking on the portal future could deadlock (we might BE on the
        # portal's loop), so fall through to the thread fallback instead.
        portal = get_async_portal()
        if portal is not None:
            on_running_loop = True
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                on_running_loop = False
            if not on_running_loop:
                portal_future = portal.start_task_soon(async_fn, query)
                try:
                    return portal_future.result(timeout=dispatch_timeout)
                except concurrent.futures.TimeoutError:
                    # Cancel the abandoned coroutine so it cannot linger on
                    # the portal's loop, then surface the timeout to the
                    # caller (search_function logs and returns []).
                    portal_future.cancel()
                    raise

        # Asynchronous invocation fallback: run in a separate thread with a
        # new event loop (no portal active, or already on a running loop).
        if executor is None:
            raise ValueError("Executor required for async tool invocation")

        def run_async_in_thread() -> Any:
            """Run async tool in a new event loop within a thread.

            Note: We avoid setting the event loop as the thread's loop to prevent
            pollution when thread pool threads are reused. The loop is run directly
            and properly cleaned up to avoid stale references.
            """
            # Create new event loop for this thread but don't set it as the thread's loop
            # to avoid pollution when thread pool threads are reused
            loop = asyncio.new_event_loop()
            try:
                if method == "ainvoke":
                    return loop.run_until_complete(tool.ainvoke(query))
                elif method == "arun":
                    return loop.run_until_complete(tool.arun(query))
                else:  # async callable
                    return loop.run_until_complete(tool(query))
            finally:
                # Cancel any pending tasks to avoid warnings
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    # Give cancelled tasks a chance to clean up
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.close()
                # Clear any stale event loop reference from the thread
                # RuntimeError can occur if loop is already closed or thread has no loop set
                with contextlib.suppress(RuntimeError):
                    asyncio.set_event_loop(None)

        # Submit to thread pool and wait for result, bounded by the same
        # historical search floor as the portal dispatch above.
        future = executor.submit(run_async_in_thread)
        return future.result(timeout=dispatch_timeout)
