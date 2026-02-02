"""Search result parsing utilities for verification operations.

This module provides functions for parsing raw search tool outputs into
structured SearchResultItem objects.

Functions:
    parse_tool_output: Parse search tool output into SearchResultItem list
"""

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ....schemas import SearchResultItem

logger = logging.getLogger(__name__)

__all__ = [
    "parse_tool_output",
]


def parse_tool_output(raw_result: Any) -> list["SearchResultItem"]:
    """Parse raw tool output into list of SearchResultItem objects.

    This function handles multiple output formats:
    1. List of SearchResultItem objects (already structured)
    2. List of dicts with title/content/url keys
    3. JSON string containing list of dicts
    4. Plain string (creates single item with generic title)

    Args:
        raw_result: Raw output from search tool

    Returns:
        List of SearchResultItem objects

    Note:
        Returns empty list on failure rather than raising.
    """
    # Import here to avoid circular imports
    from ....schemas import SearchResultItem

    # Case 1: Already a list of SearchResultItem
    if isinstance(raw_result, list) and all(isinstance(item, SearchResultItem) for item in raw_result):
        return raw_result

    # Case 2: List of dicts
    if isinstance(raw_result, list) and all(isinstance(item, dict) for item in raw_result):
        items = []
        for item_dict in raw_result:
            try:
                # Handle optional title and url fields
                title = item_dict.get("title") or None  # Convert empty string to None
                content = item_dict.get("content", "No content")
                url = item_dict.get("url") or None  # Convert empty string to None

                # Skip items with no content
                if not content or content == "No content":
                    logger.warning("Skipping search result with no content")
                    continue

                item = SearchResultItem(
                    title=title,
                    content=content,
                    url=url,
                )
                items.append(item)
            except Exception as e:
                logger.warning(f"Failed to parse dict to SearchResultItem: {e}")
                continue
        return items

    # Case 3: JSON string
    if isinstance(raw_result, str):
        try:
            # Try to parse as JSON
            parsed = json.loads(raw_result)
            if isinstance(parsed, list):
                return parse_tool_output(parsed)  # Recursive call with parsed list
        except json.JSONDecodeError:
            # Not JSON - treat as plain text
            pass

        # Plain text fallback - create single generic item (no title, no URL)
        logger.info("Search tool returned plain text, wrapping in SearchResultItem")
        return [
            SearchResultItem(
                title=None,  # Will use truncated content in GUI
                content=raw_result.strip(),
                url=None,
            )
        ]

    # Case 4: Unknown format
    logger.warning(f"Unknown search result format: {type(raw_result)}")
    return []
