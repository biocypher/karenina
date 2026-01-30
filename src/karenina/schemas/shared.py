"""Search result schemas for deep-judgment excerpt validation.

This module defines Pydantic models for structured search results returned by
search tools. The structured format enables:

1. Tool-agnostic parsing (works with Tavily, MCP tools, custom tools)
2. Type-safe data handling in both backend and frontend
3. Easy extensibility (add fields like score, published_date, etc.)
4. Clear interface requirements for search tool implementations

The main model is SearchResultItem, which represents a single search result
with title, content, and source URL.
"""

from pydantic import BaseModel, Field


class SearchResultItem(BaseModel):
    """A single search result from a search tool.

    This is the standard format that all search tools should return. Each item
    represents one search result (e.g., one webpage, article, or document).

    Fields are designed to be robust to edge cases:
    - title: Optional. If missing, frontend will display truncated content
    - content: Required. The main text of the result
    - url: Optional. If missing, frontend will not display a clickable link

    Attributes:
        title: The title or heading of the search result (optional)
        content: The main text content or snippet from the result (required)
        url: The source URL where the content was found (optional)

    Examples:
        >>> # Full result with all fields
        >>> result1 = SearchResultItem(
        ...     title="BCL-2 Protein Overview",
        ...     content="BCL-2 is an anti-apoptotic protein that regulates cell death...",
        ...     url="https://example.com/bcl2"
        ... )

        >>> # Result with no title (will use truncated content in GUI)
        >>> result2 = SearchResultItem(
        ...     content="BCL-2 is an anti-apoptotic protein that regulates cell death...",
        ...     url="https://example.com/bcl2"
        ... )

        >>> # Result with no URL (will not show clickable link)
        >>> result3 = SearchResultItem(
        ...     title="Local Knowledge Base Entry",
        ...     content="Information from internal database..."
        ... )
    """

    title: str | None = Field(
        None,
        description="Title or heading of the search result (None if unavailable)",
    )

    content: str = Field(
        ...,
        description="Main text content or snippet from the search result",
        min_length=1,
    )

    url: str | None = Field(
        None,
        description="Source URL where the content was found (None if unavailable)",
    )

    def __str__(self) -> str:
        """Human-readable string representation."""
        title_display = self.title or self.content[:50] + "..."
        url_display = self.url or "No URL"
        return f"[{title_display}] {self.content[:100]}... (Source: {url_display})"

    class Config:
        """Pydantic model configuration."""

        # Enable JSON schema generation
        json_schema_extra = {
            "example": {
                "title": "Example Article Title",
                "content": "This is the main content of the search result...",
                "url": "https://example.com/article",
            }
        }
