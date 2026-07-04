"""Tests for MCP config conversion."""

from __future__ import annotations

from contextlib import AsyncExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.unit
class TestConvertMcpToToolsSessionLifetime:
    """Test that convert_mcp_to_tools keeps sessions alive via exit_stack."""

    @pytest.mark.asyncio
    async def test_sessions_remain_open_after_return(self):
        """Tools returned by convert_mcp_to_tools should have live sessions."""
        from karenina.adapters.langchain_deep_agents.mcp import convert_mcp_to_tools

        mock_tool = MagicMock(name="mock_tool")
        mock_client = AsyncMock()
        mock_client.get_tools = AsyncMock(return_value=[mock_tool])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "langchain_mcp_adapters.client.MultiServerMCPClient",
            return_value=mock_client,
        ):
            async with AsyncExitStack() as exit_stack:
                tools = await convert_mcp_to_tools(
                    {"test": {"type": "http", "url": "http://localhost:8080"}},
                    exit_stack,
                )
                assert len(tools) == 1
                # Client should NOT have been closed yet (exit_stack still open)
                mock_client.__aexit__.assert_not_called()

            # After exit_stack closes, client should be cleaned up
            mock_client.__aexit__.assert_called_once()


@pytest.mark.unit
class TestMCPConfigConversion:
    def test_stdio_config_produces_valid_params(self):
        from karenina.adapters.langchain_deep_agents.mcp import build_mcp_server_params

        config = {
            "filesystem": {
                "type": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            }
        }
        result = build_mcp_server_params(config)
        assert "filesystem" in result
        assert result["filesystem"]["command"] == "npx"

    def test_http_config_produces_valid_params(self):
        from karenina.adapters.langchain_deep_agents.mcp import build_mcp_server_params

        config = {
            "api": {
                "type": "http",
                "url": "https://mcp.example.com/api",
            }
        }
        result = build_mcp_server_params(config)
        assert "api" in result
        assert result["api"]["url"] == "https://mcp.example.com/api"

    def test_empty_config_returns_empty(self):
        from karenina.adapters.langchain_deep_agents.mcp import build_mcp_server_params

        assert build_mcp_server_params({}) == {}
        assert build_mcp_server_params(None) == {}
