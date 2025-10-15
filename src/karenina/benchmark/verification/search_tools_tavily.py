"""Tavily search tool implementation for deep-judgment excerpt validation.

This module provides a built-in implementation using Tavily's search API.
Tavily is designed for AI applications and provides high-quality, structured
search results.

API Key Setup:
    Set the TAVILY_API_KEY environment variable before using this tool:
    export TAVILY_API_KEY="tvly-..."

    Get an API key at: https://tavily.com

Dependencies:
    - langchain-community (for TavilySearchAPIWrapper)
    - tavily-python (underlying Tavily client)

    Install with: pip install langchain-community tavily-python
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def create_tavily_search_tool(
    api_key: str | None = None,
    max_results: int = 3,
    search_depth: str = "basic",
    **kwargs: Any,
) -> Any:
    """Create a Tavily search tool function.

    Args:
        api_key: Tavily API key. If None, reads from TAVILY_API_KEY env var
        max_results: Maximum number of search results to retrieve (default: 3)
        search_depth: Search depth ("basic" or "advanced", default: "basic")
        **kwargs: Additional arguments passed to TavilySearchAPIWrapper

    Returns:
        Search function that takes str|list[str] and returns str|list[str]

    Raises:
        ValueError: If API key is not provided and not in environment
        ImportError: If langchain-community or tavily-python not installed
    """
    # Get API key from argument or environment
    tavily_api_key = api_key or os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError(
            "Tavily API key required. Set TAVILY_API_KEY environment variable "
            "or pass api_key argument. Get a key at: https://tavily.com"
        )

    # Initialize Tavily tool
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults

        tavily_tool = TavilySearchResults(
            api_wrapper=_create_tavily_wrapper(tavily_api_key, **kwargs),
            max_results=max_results,
        )

        logger.info(f"Initialized Tavily search tool (max_results={max_results}, depth={search_depth})")

    except ImportError as e:
        raise ImportError(
            "Tavily search requires 'langchain-community' and 'tavily-python'. "
            "Install with: pip install langchain-community tavily-python"
        ) from e

    def tavily_search(query: str | list[str]) -> str | list[str]:
        """Execute Tavily search for one or more queries.

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
                    result = _search_single(tavily_tool, q, max_results)
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Tavily search failed for query '{q[:50]}...': {e}")
                    results.append(f"Tavily search failed: {str(e)}")
            return results

        # Handle single query
        else:
            try:
                return _search_single(tavily_tool, query, max_results)
            except Exception as e:
                logger.warning(f"Tavily search failed for query '{query[:50]}...': {e}")
                return f"Tavily search failed: {str(e)}"

    return tavily_search


def _create_tavily_wrapper(tavily_api_key: str, **kwargs: Any) -> Any:
    """Create Tavily API wrapper with configuration.

    Args:
        tavily_api_key: Tavily API key
        **kwargs: Additional arguments for TavilySearchAPIWrapper

    Returns:
        Configured TavilySearchAPIWrapper instance
    """
    from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

    return TavilySearchAPIWrapper(
        tavily_api_key=tavily_api_key,
        **kwargs,
    )


def _search_single(tavily_tool: Any, query: str, max_results: int) -> str:
    """Execute single Tavily search and format results.

    Args:
        tavily_tool: TavilySearchResults instance
        query: Search query string
        max_results: Maximum number of results to return

    Returns:
        Formatted search results as string
    """
    logger.debug(f"Tavily search for: '{query[:100]}...'")
    results = tavily_tool.invoke(query)

    # Parse results (Tavily returns list of dicts or string)
    if isinstance(results, list):
        # Structured results: [{"url": ..., "content": ...}, ...]
        results_summary = _format_structured_results(results, max_results)
    elif isinstance(results, str):
        # String results (fallback)
        results_summary = results.strip()
    else:
        # Unexpected format
        results_summary = str(results)

    logger.info(f"Tavily search completed: {len(results_summary)} chars")
    return results_summary


def _format_structured_results(results: list[dict[str, Any]], max_results: int) -> str:
    """Format structured Tavily results into summary string.

    Args:
        results: List of result dicts from Tavily API
        max_results: Maximum number of results to include

    Returns:
        Human-readable summary of top results
    """
    if not results:
        return "No search results found."

    # Take top N results (already limited by max_results)
    top_results = results[:max_results]

    summary_parts = []
    for i, result in enumerate(top_results, 1):
        title = result.get("title", "Untitled")
        content = result.get("content", "No content available")
        url = result.get("url", "")

        # Include full content (no truncation)
        summary_parts.append(f"[{i}] {title}\n    {content}\n    Source: {url}")

    return "\n\n".join(summary_parts)
