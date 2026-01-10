"""Mock MCP server for testing tool description optimization.

This module provides mock classes that simulate MCP server behavior,
allowing tests to verify tool description injection without requiring
real MCP server connections.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MockTool:
    """Mock MCP tool for testing.

    Simulates a LangChain-compatible tool object with name and description.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


class MockMCPClient:
    """Mock MCP client that returns configurable tools.

    Use this to test MCP tool fetching and description override behavior
    without connecting to real MCP servers.

    Example:
        >>> client = MockMCPClient([
        ...     MockTool("search", "Search description"),
        ...     MockTool("query", "Query description"),
        ... ])
        >>> tools = await client.get_tools()
        >>> assert len(tools) == 2
    """

    def __init__(self, tools: list[MockTool] | None = None):
        """Initialize with optional custom tools.

        Args:
            tools: List of MockTool objects. If None, uses default test tools.
        """
        self._tools = tools or [
            MockTool(
                name="search_proteins",
                description="Search for protein information by name or ID.",
            ),
            MockTool(
                name="get_interactions",
                description="Get protein-protein interactions from database.",
            ),
            MockTool(
                name="analyze_pathway",
                description="Analyze biological pathways and gene networks.",
            ),
        ]

    async def get_tools(self) -> list[MockTool]:
        """Return mock tools (simulates async network fetch).

        Returns:
            Copy of the tools list to prevent mutation affecting future calls.
        """
        await asyncio.sleep(0.001)  # Simulate minimal network latency
        # Return copies to ensure tests see fresh state
        return [
            MockTool(
                name=t.name,
                description=t.description,
                parameters=t.parameters.copy(),
            )
            for t in self._tools
        ]

    async def aclose(self) -> None:
        """Simulate async client cleanup."""
        pass


def create_mock_mcp_client_and_tools(
    mcp_urls_dict: dict[str, str],
    tool_filter: list[str] | None = None,
    tool_description_overrides: dict[str, str] | None = None,
    custom_tools: list[MockTool] | None = None,
) -> tuple[MockMCPClient, list[MockTool]]:
    """Create mock MCP client and tools for testing.

    This function mirrors the signature of sync_create_mcp_client_and_tools
    to make it easy to patch in tests.

    Args:
        mcp_urls_dict: Dictionary mapping server names to URLs (ignored in mock).
        tool_filter: Optional list of tool names to include.
        tool_description_overrides: Optional dict of tool descriptions to override.
        custom_tools: Optional list of MockTool objects to use instead of defaults.

    Returns:
        Tuple of (MockMCPClient, list[MockTool]) with overrides applied.

    Example:
        >>> client, tools = create_mock_mcp_client_and_tools(
        ...     {"test": "http://mock"},
        ...     tool_description_overrides={"search_proteins": "INJECTED"}
        ... )
        >>> search_tool = next(t for t in tools if t.name == "search_proteins")
        >>> assert search_tool.description == "INJECTED"
    """
    client = MockMCPClient(custom_tools)
    tools = asyncio.run(client.get_tools())

    # Apply filter if provided
    if tool_filter is not None:
        allowed = set(tool_filter)
        tools = [t for t in tools if t.name in allowed]

    # Apply description overrides
    if tool_description_overrides:
        for tool in tools:
            if tool.name in tool_description_overrides:
                tool.description = tool_description_overrides[tool.name]

    return client, tools
