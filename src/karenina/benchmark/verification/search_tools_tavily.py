"""Tavily search tool implementation for deep-judgment excerpt validation.

This module provides a built-in implementation of the SearchTool protocol
using Tavily's search API. Tavily is designed for AI applications and
provides high-quality, structured search results.

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

from .search_tools import SearchResult

logger = logging.getLogger(__name__)


class TavilySearchTool:
    """Tavily-based search tool for excerpt validation.

    This implementation uses Tavily's search API to validate whether excerpts
    extracted from LLM responses are backed by external evidence. Tavily is
    optimized for AI use cases and provides:
    - High-quality search results
    - Structured result format
    - Fast response times
    - Reasonable rate limits

    The tool handles:
    - API key management from environment
    - Result summarization (top 3 results)
    - Confidence scoring based on result quality
    - Error handling and fallback
    - Rate limiting considerations
    """

    def __init__(
        self,
        api_key: str | None = None,
        max_results: int = 3,
        search_depth: str = "basic",
        **kwargs: Any,
    ):
        """Initialize Tavily search tool.

        Args:
            api_key: Tavily API key. If None, reads from TAVILY_API_KEY env var
            max_results: Maximum number of search results to retrieve (default: 3)
            search_depth: Search depth ("basic" or "advanced", default: "basic")
            **kwargs: Additional arguments passed to TavilySearchAPIWrapper

        Raises:
            ValueError: If API key is not provided and not in environment
        """
        # Get API key from argument or environment
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Tavily API key required. Set TAVILY_API_KEY environment variable "
                "or pass api_key argument. Get a key at: https://tavily.com"
            )

        self.max_results = max_results
        self.search_depth = search_depth

        # Initialize Tavily wrapper
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults

            self.tavily_tool = TavilySearchResults(
                api_wrapper=self._create_tavily_wrapper(**kwargs),
                max_results=max_results,
            )

            logger.info(f"Initialized TavilySearchTool (max_results={max_results}, depth={search_depth})")

        except ImportError as e:
            raise ImportError(
                "Tavily search requires 'langchain-community' and 'tavily-python'. "
                "Install with: pip install langchain-community tavily-python"
            ) from e

    def _create_tavily_wrapper(self, **kwargs: Any) -> Any:
        """Create Tavily API wrapper with configuration.

        Args:
            **kwargs: Additional arguments for TavilySearchAPIWrapper

        Returns:
            Configured TavilySearchAPIWrapper instance
        """
        from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

        return TavilySearchAPIWrapper(
            tavily_api_key=self.api_key,
            **kwargs,
        )

    def search(self, query: str) -> SearchResult:
        """Execute search using Tavily API.

        Args:
            query: Search query (typically an excerpt to validate)

        Returns:
            SearchResult with summary and confidence score
        """
        try:
            # Invoke Tavily search
            logger.debug(f"Tavily search for: '{query[:100]}...'")
            results = self.tavily_tool.invoke(query)

            # Parse results (Tavily returns list of dicts or string)
            if isinstance(results, list):
                # Structured results: [{"url": ..., "content": ...}, ...]
                results_summary = self._format_structured_results(results)
                raw_results = results
            elif isinstance(results, str):
                # String results (fallback)
                results_summary = results.strip()
                raw_results = results
            else:
                # Unexpected format
                results_summary = str(results)
                raw_results = results

            logger.info(f"Tavily search completed: {len(results_summary)} chars")

            return SearchResult(
                query=query,
                results_summary=results_summary,
                raw_results=raw_results,
            )

        except Exception as e:
            logger.warning(f"Tavily search failed for query '{query[:50]}...': {e}")
            # Return empty result on failure
            return SearchResult(
                query=query,
                results_summary=f"Tavily search failed: {str(e)}",
                raw_results=None,
            )

    def _format_structured_results(self, results: list[dict[str, Any]]) -> str:
        """Format structured Tavily results into summary string.

        Args:
            results: List of result dicts from Tavily API

        Returns:
            Human-readable summary of top results
        """
        if not results:
            return "No search results found."

        # Take top N results (already limited by max_results)
        top_results = results[: self.max_results]

        summary_parts = []
        for i, result in enumerate(top_results, 1):
            title = result.get("title", "Untitled")
            content = result.get("content", "No content available")
            url = result.get("url", "")

            # Include full content (no truncation)
            summary_parts.append(f"[{i}] {title}\n    {content}\n    Source: {url}")

        return "\n\n".join(summary_parts)
