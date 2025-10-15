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

Example Usage:
    # Built-in tool (string-based)
    search_tool = create_search_tool("tavily")

    # Custom langchain tool (function-based)
    from langchain.tools import Tool
    custom_tool = Tool(name="custom_search", func=my_search_function, ...)
    search_tool = create_search_tool(custom_tool)

    # Use the tool
    result = search_tool.search("What is the capital of France?")
    print(result.results_summary)
"""

import logging
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchResult:
    """Result from a search tool query.

    Attributes:
        query: The original search query (excerpt text)
        results_summary: Human-readable summary of top search results
        raw_results: Optional raw results from the search API (for debugging)
    """

    query: str
    results_summary: str
    raw_results: Any | None = None


class SearchTool(Protocol):
    """Protocol for search tools used in deep-judgment excerpt validation.

    This protocol defines the interface that all search tools must implement.
    Search tools can be:
    - Built-in implementations (e.g., TavilySearchTool)
    - Langchain tools wrapped in this interface
    - MCP tools wrapped in this interface (future)

    The protocol ensures consistent behavior across different search backends
    while allowing flexible implementation strategies.
    """

    def search(self, query: str) -> SearchResult:
        """Execute a search query and return structured results.

        Args:
            query: The search query (typically an excerpt from LLM response)

        Returns:
            SearchResult containing summary and confidence score

        Raises:
            Exception: Implementation-specific errors (should be caught by caller)
        """
        ...


class LangChainSearchToolAdapter:
    """Adapter that wraps a langchain tool to conform to SearchTool protocol.

    This adapter allows any langchain tool (including MCP tools) to be used
    as a search tool in deep-judgment. The langchain tool should:
    - Accept a string query as input
    - Return a string result

    The adapter handles:
    - Invoking the langchain tool
    - Parsing results into SearchResult format
    - Calculating confidence scores based on result quality
    - Error handling and fallbacks
    """

    def __init__(self, langchain_tool: Any):
        """Initialize adapter with a langchain tool.

        Args:
            langchain_tool: Any langchain tool that implements invoke() or run()
                Expected to have one of: invoke(str) -> str, run(str) -> str
        """
        self.langchain_tool = langchain_tool

        # Determine which method to use (invoke for newer tools, run for legacy)
        if hasattr(langchain_tool, "invoke"):
            self._call_method = "invoke"
        elif hasattr(langchain_tool, "run"):
            self._call_method = "run"
        elif callable(langchain_tool):
            self._call_method = "call"
        else:
            raise ValueError(
                f"Langchain tool must have 'invoke', 'run' method, or be callable. Got: {type(langchain_tool)}"
            )

        logger.info(f"Initialized LangChainSearchToolAdapter with tool: {getattr(langchain_tool, 'name', 'unknown')}")

    def search(self, query: str) -> SearchResult:
        """Execute search using the wrapped langchain tool.

        Args:
            query: Search query string

        Returns:
            SearchResult with summary and confidence score
        """
        try:
            # Invoke the langchain tool
            if self._call_method == "invoke":
                raw_result = self.langchain_tool.invoke(query)
            elif self._call_method == "run":
                raw_result = self.langchain_tool.run(query)
            else:  # callable
                raw_result = self.langchain_tool(query)

            # Parse result (langchain tools typically return strings)
            results_summary = raw_result.strip() if isinstance(raw_result, str) else str(raw_result)

            logger.info(f"Search completed for query: '{query[:50]}...'")

            return SearchResult(
                query=query,
                results_summary=results_summary,
                raw_results=raw_result,
            )

        except Exception as e:
            logger.warning(f"Search failed for query '{query[:50]}...': {e}")
            # Return empty result on failure
            return SearchResult(
                query=query,
                results_summary=f"Search failed: {str(e)}",
                raw_results=None,
            )


def create_search_tool(
    tool: str | Any,
    **kwargs: Any,
) -> SearchTool:
    """Factory function to create search tools.

    This factory supports two modes:

    1. String-based (built-in tools):
       create_search_tool("tavily") -> TavilySearchTool instance

    2. Langchain tool injection:
       create_search_tool(my_langchain_tool) -> LangChainSearchToolAdapter

    The dual-mode design allows:
    - Simple usage with built-in tools (just pass "tavily")
    - Advanced usage with custom langchain tools (pass tool instance)
    - Future MCP integration (pass MCP tool as langchain tool)

    Args:
        tool: Either a string name ("tavily") or a langchain tool instance
        **kwargs: Additional arguments passed to built-in tool constructors

    Returns:
        SearchTool instance conforming to the protocol

    Raises:
        ValueError: If tool name is unknown or tool instance is invalid

    Examples:
        # Built-in Tavily tool
        search_tool = create_search_tool("tavily")

        # Custom langchain tool
        from langchain.tools import Tool
        custom_tool = Tool(name="my_search", func=lambda q: f"Results for: {q}")
        search_tool = create_search_tool(custom_tool)

        # Future: MCP tool (treated as langchain tool)
        mcp_search_tool = load_mcp_tool("search")
        search_tool = create_search_tool(mcp_search_tool)
    """
    # Case 1: String-based built-in tool
    if isinstance(tool, str):
        tool_name = tool.lower()

        if tool_name == "tavily":
            # Import here to avoid dependency issues if Tavily not installed
            try:
                from .search_tools_tavily import TavilySearchTool

                logger.info("Creating TavilySearchTool instance")
                return TavilySearchTool(**kwargs)
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

    # Case 2: Langchain tool instance
    else:
        # Wrap the langchain tool in our adapter
        logger.info(f"Creating LangChainSearchToolAdapter for tool: {type(tool)}")
        return LangChainSearchToolAdapter(tool)
