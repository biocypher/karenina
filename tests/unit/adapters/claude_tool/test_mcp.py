"""Tests for MCP client session management in claude_tool adapter.

Tests connect_mcp_session, connect_all_mcp_servers, and get_all_mcp_tools.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestConnectMcpSession:
    """Tests for connect_mcp_session function."""

    @pytest.mark.asyncio
    async def test_raises_value_error_without_url(self) -> None:
        """Test raises ValueError when config doesn't have url."""
        from karenina.adapters.claude_tool.mcp import connect_mcp_session

        async with AsyncExitStack() as stack:
            config: dict[str, Any] = {"type": "http"}  # Missing url

            with pytest.raises(ValueError, match="must include 'url'"):
                await connect_mcp_session(stack, config)

    @pytest.mark.asyncio
    async def test_raises_value_error_with_non_string_url(self) -> None:
        """Test raises ValueError when url is not a string."""
        from karenina.adapters.claude_tool.mcp import connect_mcp_session

        async with AsyncExitStack() as stack:
            config: dict[str, Any] = {"type": "http", "url": 12345}

            with pytest.raises(ValueError, match="must include 'url'"):
                await connect_mcp_session(stack, config)

    @pytest.mark.asyncio
    async def test_connects_with_valid_config(self) -> None:
        """Test successful connection with valid config."""
        from karenina.adapters.claude_tool.mcp import connect_mcp_session

        # Create mock transport and session
        mock_read = AsyncMock()
        mock_write = AsyncMock()
        mock_session = AsyncMock()

        with (
            patch("mcp.client.streamable_http.streamablehttp_client") as mock_client,
            patch("mcp.ClientSession") as mock_session_class,
        ):
            # Configure mock client context manager
            mock_transport_cm = AsyncMock()
            mock_transport_cm.__aenter__.return_value = (mock_read, mock_write, None)
            mock_transport_cm.__aexit__.return_value = None
            mock_client.return_value = mock_transport_cm

            # Configure mock session context manager
            mock_session_cm = AsyncMock()
            mock_session_cm.__aenter__.return_value = mock_session
            mock_session_cm.__aexit__.return_value = None
            mock_session_class.return_value = mock_session_cm

            async with AsyncExitStack() as stack:
                config: dict[str, Any] = {
                    "type": "http",
                    "url": "https://mcp.example.com/mcp",
                }

                result = await connect_mcp_session(stack, config)
                assert result == mock_session
                mock_client.assert_called_once_with(
                    "https://mcp.example.com/mcp",
                    headers={},
                )
                mock_session.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_passes_headers_to_transport(self) -> None:
        """Test that headers are passed to the transport."""
        from karenina.adapters.claude_tool.mcp import connect_mcp_session

        mock_session = AsyncMock()

        with (
            patch("mcp.client.streamable_http.streamablehttp_client") as mock_client,
            patch("mcp.ClientSession") as mock_session_class,
        ):
            mock_transport_cm = AsyncMock()
            mock_transport_cm.__aenter__.return_value = (AsyncMock(), AsyncMock(), None)
            mock_transport_cm.__aexit__.return_value = None
            mock_client.return_value = mock_transport_cm

            mock_session_cm = AsyncMock()
            mock_session_cm.__aenter__.return_value = mock_session
            mock_session_cm.__aexit__.return_value = None
            mock_session_class.return_value = mock_session_cm

            async with AsyncExitStack() as stack:
                config: dict[str, Any] = {
                    "type": "http",
                    "url": "https://mcp.example.com/mcp",
                    "headers": {"Authorization": "Bearer token123"},
                }

                await connect_mcp_session(stack, config)
                mock_client.assert_called_once_with(
                    "https://mcp.example.com/mcp",
                    headers={"Authorization": "Bearer token123"},
                )


class TestConnectAllMcpServers:
    """Tests for connect_all_mcp_servers function."""

    @pytest.mark.asyncio
    async def test_connects_multiple_servers(self) -> None:
        """Test connecting to multiple MCP servers."""
        from karenina.adapters.claude_tool.mcp import connect_all_mcp_servers

        mock_session1 = AsyncMock()
        mock_session2 = AsyncMock()

        with patch("karenina.adapters.claude_tool.mcp.connect_mcp_session") as mock_connect:
            mock_connect.side_effect = [mock_session1, mock_session2]

            async with AsyncExitStack() as stack:
                servers: dict[str, Any] = {
                    "server1": {"url": "https://server1.com/mcp"},
                    "server2": {"url": "https://server2.com/mcp"},
                }

                result = await connect_all_mcp_servers(stack, servers)
                assert len(result) == 2
                assert result["server1"] == mock_session1
                assert result["server2"] == mock_session2
                assert mock_connect.call_count == 2

    @pytest.mark.asyncio
    async def test_handles_empty_servers(self) -> None:
        """Test handling empty servers dict."""
        from karenina.adapters.claude_tool.mcp import connect_all_mcp_servers

        async with AsyncExitStack() as stack:
            result = await connect_all_mcp_servers(stack, {})

            assert result == {}

    @pytest.mark.asyncio
    async def test_raises_on_connection_failure(self) -> None:
        """Test raises exception when server connection fails."""
        from karenina.adapters.claude_tool.mcp import connect_all_mcp_servers

        with patch("karenina.adapters.claude_tool.mcp.connect_mcp_session") as mock_connect:
            mock_connect.side_effect = ValueError("Connection failed")

            async with AsyncExitStack() as stack:
                servers: dict[str, Any] = {
                    "failing_server": {"url": "https://failing.com/mcp"},
                }

                with pytest.raises(ValueError, match="Connection failed"):
                    await connect_all_mcp_servers(stack, servers)


class TestGetAllMcpTools:
    """Tests for get_all_mcp_tools function."""

    @pytest.mark.asyncio
    async def test_gets_tools_from_single_server(self) -> None:
        """Test getting tools from a single MCP server."""
        from karenina.adapters.claude_tool.mcp import get_all_mcp_tools

        mock_tool = MagicMock()
        mock_tool.name = "search"

        mock_session = AsyncMock()
        mock_tools_response = MagicMock()
        mock_tools_response.tools = [mock_tool]
        mock_session.list_tools.return_value = mock_tools_response

        sessions = {"server1": mock_session}

        result = await get_all_mcp_tools(sessions)

        assert len(result) == 1
        server_name, session, tool = result[0]
        assert server_name == "server1"
        assert session == mock_session
        assert tool == mock_tool

    @pytest.mark.asyncio
    async def test_gets_tools_from_multiple_servers(self) -> None:
        """Test getting tools from multiple MCP servers."""
        from karenina.adapters.claude_tool.mcp import get_all_mcp_tools

        mock_tool1 = MagicMock()
        mock_tool1.name = "search"
        mock_tool2 = MagicMock()
        mock_tool2.name = "query"

        mock_session1 = AsyncMock()
        mock_tools_response1 = MagicMock()
        mock_tools_response1.tools = [mock_tool1]
        mock_session1.list_tools.return_value = mock_tools_response1

        mock_session2 = AsyncMock()
        mock_tools_response2 = MagicMock()
        mock_tools_response2.tools = [mock_tool2]
        mock_session2.list_tools.return_value = mock_tools_response2

        sessions = {"server1": mock_session1, "server2": mock_session2}

        result = await get_all_mcp_tools(sessions)

        assert len(result) == 2
        tool_names = [t[2].name for t in result]
        assert "search" in tool_names
        assert "query" in tool_names

    @pytest.mark.asyncio
    async def test_handles_server_with_multiple_tools(self) -> None:
        """Test handling server that provides multiple tools."""
        from karenina.adapters.claude_tool.mcp import get_all_mcp_tools

        mock_tool1 = MagicMock()
        mock_tool1.name = "tool1"
        mock_tool2 = MagicMock()
        mock_tool2.name = "tool2"
        mock_tool3 = MagicMock()
        mock_tool3.name = "tool3"

        mock_session = AsyncMock()
        mock_tools_response = MagicMock()
        mock_tools_response.tools = [mock_tool1, mock_tool2, mock_tool3]
        mock_session.list_tools.return_value = mock_tools_response

        sessions = {"multi_tool_server": mock_session}

        result = await get_all_mcp_tools(sessions)

        assert len(result) == 3
        for server_name, _, _ in result:
            assert server_name == "multi_tool_server"

    @pytest.mark.asyncio
    async def test_handles_empty_sessions(self) -> None:
        """Test handling empty sessions dict."""
        from karenina.adapters.claude_tool.mcp import get_all_mcp_tools

        result = await get_all_mcp_tools({})

        assert result == []

    @pytest.mark.asyncio
    async def test_raises_on_list_tools_failure(self) -> None:
        """Test raises exception when list_tools fails."""
        from karenina.adapters.claude_tool.mcp import get_all_mcp_tools

        mock_session = AsyncMock()
        mock_session.list_tools.side_effect = Exception("Failed to list tools")

        sessions = {"failing_server": mock_session}

        with pytest.raises(Exception, match="Failed to list tools"):
            await get_all_mcp_tools(sessions)
