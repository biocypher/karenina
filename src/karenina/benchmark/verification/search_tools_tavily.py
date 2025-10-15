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
                confidence_score = self._calculate_confidence_from_structured(results)
                raw_results = results
            elif isinstance(results, str):
                # String results (fallback)
                results_summary = results.strip()
                confidence_score = self._calculate_confidence_from_string(results_summary)
                raw_results = results
            else:
                # Unexpected format
                results_summary = str(results)
                confidence_score = 0.5
                raw_results = results

            logger.info(f"Tavily search completed: {len(results_summary)} chars (confidence: {confidence_score:.2f})")

            return SearchResult(
                query=query,
                results_summary=results_summary,
                confidence_score=confidence_score,
                raw_results=raw_results,
            )

        except Exception as e:
            logger.warning(f"Tavily search failed for query '{query[:50]}...': {e}")
            # Return empty result on failure
            return SearchResult(
                query=query,
                results_summary=f"Tavily search failed: {str(e)}",
                confidence_score=0.0,
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

            # Truncate content to reasonable length
            content_preview = content[:200] + "..." if len(content) > 200 else content

            summary_parts.append(f"[{i}] {title}\n    {content_preview}\n    Source: {url}")

        return "\n\n".join(summary_parts)

    def _calculate_confidence_from_structured(self, results: list[dict[str, Any]]) -> float:
        """Calculate confidence score from structured Tavily results.

        Args:
            results: List of result dicts from Tavily API

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not results:
            return 0.0

        # Base confidence on number of results
        num_results = len(results)
        result_count_score = min(num_results / self.max_results, 1.0)

        # Bonus for result quality indicators
        quality_bonus = 0.0

        for result in results:
            # Check for quality indicators
            content = result.get("content", "")
            title = result.get("title", "")

            # Bonus for substantial content
            if len(content) > 100:
                quality_bonus += 0.1

            # Bonus for having title
            if title and len(title) > 5:
                quality_bonus += 0.05

            # Bonus for URL (source credibility)
            if result.get("url"):
                quality_bonus += 0.05

        # Average quality bonus across results
        avg_quality_bonus = quality_bonus / max(num_results, 1)
        avg_quality_bonus = min(avg_quality_bonus, 0.3)  # Cap at +0.3

        final_score = min(result_count_score + avg_quality_bonus, 1.0)
        return final_score

    def _calculate_confidence_from_string(self, results_summary: str) -> float:
        """Calculate confidence score from string result.

        Args:
            results_summary: String summary of search results

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not results_summary or len(results_summary) < 10:
            return 0.0

        # Check for failure indicators
        failure_keywords = ["no results", "not found", "error", "failed"]
        if any(keyword in results_summary.lower() for keyword in failure_keywords):
            return 0.2

        # Base confidence on length
        length_score = min(len(results_summary) / 500, 1.0)

        # Bonus for quality indicators
        quality_bonus = 0.0
        quality_keywords = ["according to", "source", "published", "http"]
        matches = sum(1 for keyword in quality_keywords if keyword in results_summary.lower())
        quality_bonus = min(matches * 0.1, 0.3)

        final_score = min(length_score + quality_bonus, 1.0)
        return final_score
