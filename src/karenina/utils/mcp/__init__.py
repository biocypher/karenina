"""Shared MCP utilities for Karenina.

This module provides adapter-agnostic MCP utilities using the core `mcp` package.
For adapter-specific MCP functionality:
- LangChain tools: karenina.adapters.langchain.mcp
- Claude SDK tools: karenina.adapters.claude_tool.mcp

The multi-server helpers (connect_all_mcp_servers, get_all_mcp_tools) that
used to live here were unused duplicates of the live copies in
karenina.adapters.claude_tool.mcp and were removed. Use those instead.
"""

from .client import connect_mcp_session
from .tools import (
    afetch_tool_descriptions,
    apply_tool_description_overrides,
    fetch_tool_descriptions,
)

__all__ = [
    # Client session management
    "connect_mcp_session",
    # Tool description utilities
    "afetch_tool_descriptions",
    "fetch_tool_descriptions",
    "apply_tool_description_overrides",
]
