"""Search tool abstraction for deep-judgment excerpt validation.

This module provides a flexible abstraction layer for search tools used in
search-enhanced deep-judgment. The abstraction supports:

1. Built-in search tools (e.g., Tavily) via string-based factory
2. Custom langchain tools passed directly as callables
3. Future MCP tool integration (by passing MCP tools as langchain tools)

The design prioritizes extensibility and allows users to:
- Use default Tavily search without configuration
- Inject custom langchain tools at runtime
- Replace with MCP-provided search tools in the future

Search Tool Interface:
    All search tools follow a simple contract:
    - Input: str or list[str] (query or queries)
    - Output: str or list[str] (result or results)

Example Usage:
    # Built-in tool (string-based)
    search_tool = create_search_tool("tavily")

    # Search single query
    result = search_tool("What is the capital of France?")
    print(result)  # String with search results

    # Search multiple queries
    results = search_tool(["Query 1", "Query 2", "Query 3"])
    print(results)  # List of result strings

    # Custom langchain tool (function-based)
    def my_search(query: str) -> str:
        return f"Results for: {query}"

    search_tool = create_search_tool(my_search)
    result = search_tool("test query")
"""

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# Type alias for search tool callable
SearchToolCallable = Callable[[str | list[str]], str | list[str]]


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
                from .search_tools_tavily import create_tavily_search_tool

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

    Args:
        langchain_tool: Any langchain tool that implements invoke(), run(), or is callable

    Returns:
        Search function that takes str|list[str] and returns str|list[str]

    Raises:
        ValueError: If tool doesn't have required methods
    """
    # Determine which method to use (invoke for newer tools, run for legacy, or callable)
    if hasattr(langchain_tool, "invoke"):
        call_method = "invoke"
    elif hasattr(langchain_tool, "run"):
        call_method = "run"
    elif callable(langchain_tool):
        call_method = "call"
    else:
        raise ValueError(
            f"Langchain tool must have 'invoke', 'run' method, or be callable. Got: {type(langchain_tool)}"
        )

    logger.info(f"Wrapped langchain tool with method: {call_method}")

    def search_function(query: str | list[str]) -> str | list[str]:
        """Execute search using wrapped langchain tool.

        Args:
            query: Single query string or list of query strings

        Returns:
            Single result string (if query is str) or list of results (if query is list)
        """
        # Handle batch queries
        if isinstance(query, list):
            results = []
            for q in query:
                try:
                    result = _invoke_tool(langchain_tool, call_method, q)
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Search failed for query '{q[:50]}...': {e}")
                    results.append(f"Search failed: {str(e)}")
            return results

        # Handle single query
        else:
            try:
                return _invoke_tool(langchain_tool, call_method, query)
            except Exception as e:
                logger.warning(f"Search failed for query '{query[:50]}...': {e}")
                return f"Search failed: {str(e)}"

    return search_function


def _invoke_tool(tool: Any, method: str, query: str) -> str:
    """Invoke a langchain tool with the appropriate method.

    Args:
        tool: The langchain tool instance
        method: Method to use ("invoke", "run", or "call")
        query: Query string

    Returns:
        Search result as string
    """
    if method == "invoke":
        raw_result = tool.invoke(query)
    elif method == "run":
        raw_result = tool.run(query)
    else:  # callable
        raw_result = tool(query)

    # Parse result to string
    if isinstance(raw_result, str):
        return raw_result.strip()
    else:
        # Handle structured data (convert to string)
        return str(raw_result)
