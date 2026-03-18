"""Tests for MCP config conversion."""

from __future__ import annotations

import pytest


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
