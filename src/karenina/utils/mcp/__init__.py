"""Shared MCP utilities for Karenina.

This module provides adapter-agnostic MCP utilities using the core `mcp` package.
For adapter-specific MCP functionality:
- LangChain tools: karenina.adapters.langchain.mcp
- Claude SDK tools: karenina.adapters.claude_tool.mcp
"""

from .client import (
    connect_all_mcp_servers,
    connect_mcp_session,
    get_all_mcp_tools,
)
from .tools import (
    afetch_tool_descriptions,
    apply_tool_description_overrides,
    fetch_tool_descriptions,
    sync_fetch_tool_descriptions,
)

__all__ = [
    # Client session management
    "connect_mcp_session",
    "connect_all_mcp_servers",
    "get_all_mcp_tools",
    # Tool description utilities
    "afetch_tool_descriptions",
    "fetch_tool_descriptions",
    "sync_fetch_tool_descriptions",  # Deprecated: use fetch_tool_descriptions
    "apply_tool_description_overrides",
]
