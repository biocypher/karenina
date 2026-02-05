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

from pydantic import SecretStr

from karenina.schemas import SearchResultItem

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
    # After check, tavily_api_key is guaranteed to be str
    assert isinstance(tavily_api_key, str), "API key must be a string"

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

    def tavily_search(query: str | list[str]) -> list[SearchResultItem] | list[list[SearchResultItem]]:
        """Execute Tavily search for one or more queries.

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
                    result = _search_single(tavily_tool, q, max_results)
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Tavily search failed for query '{q[:50]}...': {e}")
                    # Return empty list on failure
                    results.append([])
            return results

        # Handle single query
        else:
            try:
                return _search_single(tavily_tool, query, max_results)
            except Exception as e:
                logger.warning(f"Tavily search failed for query '{query[:50]}...': {e}")
                # Return empty list on failure
                return []

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
        tavily_api_key=SecretStr(tavily_api_key),
        **kwargs,
    )


def _search_single(tavily_tool: Any, query: str, max_results: int) -> list[SearchResultItem]:
    """Execute single Tavily search and format results.

    Args:
        tavily_tool: TavilySearchResults instance
        query: Search query string
        max_results: Maximum number of results to return

    Returns:
        List of SearchResultItem objects
    """
    logger.debug(f"Tavily search for: '{query[:100]}...'")
    results = tavily_tool.invoke(query)

    # Parse results (Tavily returns list of dicts)
    if isinstance(results, list):
        # Structured results: [{"url": ..., "content": ...}, ...]
        structured_results = _format_structured_results(results, max_results)
    else:
        # Unexpected format - return empty list
        logger.warning(f"Unexpected Tavily result format: {type(results)}")
        structured_results = []

    logger.info(f"Tavily search completed: {len(structured_results)} results")
    return structured_results


def _format_structured_results(results: list[dict[str, Any]], max_results: int) -> list[SearchResultItem]:
    """Format structured Tavily results into SearchResultItem objects.

    Args:
        results: List of result dicts from Tavily API
        max_results: Maximum number of results to include

    Returns:
        List of SearchResultItem objects
    """
    if not results:
        return []

    # Take top N results (already limited by max_results)
    top_results = results[:max_results]

    structured_items = []
    for result in top_results:
        # Handle optional title and url fields
        title = result.get("title") or None  # Convert empty string to None
        content = result.get("content", "No content available")
        url = result.get("url") or None  # Convert empty string to None

        # Skip if content is empty or missing
        if not content or content == "No content available":
            logger.warning("Skipping Tavily result with no content")
            continue

        # Create SearchResultItem (title and url are optional)
        try:
            item = SearchResultItem(title=title, content=content, url=url)
            structured_items.append(item)
        except Exception as e:
            logger.warning(f"Failed to create SearchResultItem: {e}")
            continue

    return structured_items
