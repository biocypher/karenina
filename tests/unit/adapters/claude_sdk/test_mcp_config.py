"""Tests for MCP configuration conversion and validation.

Tests convert_mcp_config, validate_mcp_config, and convert_and_validate_mcp_config.
"""

from __future__ import annotations

from typing import Any

from karenina.adapters.claude_agent_sdk import convert_mcp_config
from karenina.adapters.claude_agent_sdk.mcp import (
    convert_and_validate_mcp_config,
    validate_mcp_config,
)


class TestConvertMcpConfig:
    """Tests for convert_mcp_config function."""

    def test_http_url_conversion(self) -> None:
        """Test HTTP URL -> type='http' (NOT 'sse')."""
        config = convert_mcp_config({"api": "https://mcp.example.com/mcp/"})

        assert "api" in config
        api_config = config["api"]
        assert api_config.get("type") == "http"
        assert api_config.get("url") == "https://mcp.example.com/mcp/"

    def test_https_url_conversion(self) -> None:
        """Test HTTPS URL conversion."""
        config = convert_mcp_config({"secure": "https://secure.example.com/api"})

        secure_config = config["secure"]
        assert secure_config.get("type") == "http"
        assert secure_config.get("url") == "https://secure.example.com/api"

    def test_http_not_sse(self) -> None:
        """CRITICAL: Verify streamable_http maps to type='http' NOT 'sse'."""
        config = convert_mcp_config({"biocontext": "https://mcp.biocontext.ai/mcp/"})

        # This is the critical assertion from the PRD
        biocontext_config = config["biocontext"]
        assert biocontext_config.get("type") == "http"
        assert biocontext_config.get("type") != "sse"

    def test_http_url_with_auth_headers(self) -> None:
        """Test HTTP URL with auth headers."""
        auth_headers = {"Authorization": "Bearer token123"}
        config = convert_mcp_config(
            {"api": "https://mcp.example.com/"},
            auth_headers=auth_headers,
        )

        api_config = config["api"]
        assert api_config.get("type") == "http"
        assert api_config.get("url") == "https://mcp.example.com/"
        assert api_config.get("headers") == {"Authorization": "Bearer token123"}

    def test_command_with_args(self) -> None:
        """Test command with arguments is properly split."""
        config = convert_mcp_config({"github": "npx -y @modelcontextprotocol/server-github"})

        github_config = config["github"]
        assert "command" in github_config
        assert github_config.get("command") == "npx"
        assert github_config.get("args") == ["-y", "@modelcontextprotocol/server-github"]

    def test_simple_command_path(self) -> None:
        """Test simple command/path without args."""
        config = convert_mcp_config({"local": "/usr/local/bin/mcp-server"})

        local_config = config["local"]
        assert local_config.get("command") == "/usr/local/bin/mcp-server"
        assert "args" not in local_config

    def test_mixed_config(self) -> None:
        """Test mixed config with HTTP and stdio servers."""
        config = convert_mcp_config(
            {
                "biocontext": "https://mcp.biocontext.ai/mcp/",
                "github": "npx -y @mcp/server-github",
                "local": "/path/to/server",
            }
        )

        # HTTP URL
        biocontext_config = config["biocontext"]
        assert biocontext_config.get("type") == "http"
        assert biocontext_config.get("url") == "https://mcp.biocontext.ai/mcp/"

        # Command with args
        github_config = config["github"]
        assert github_config.get("command") == "npx"
        assert github_config.get("args") == ["-y", "@mcp/server-github"]

        # Simple path
        local_config = config["local"]
        assert local_config.get("command") == "/path/to/server"


class TestValidateMcpConfig:
    """Tests for validate_mcp_config function."""

    def test_valid_http_config(self) -> None:
        """Test validation of valid HTTP config."""
        config: dict[str, Any] = {"api": {"type": "http", "url": "https://example.com/mcp/"}}
        errors = validate_mcp_config(config)

        assert errors == []

    def test_valid_stdio_config(self) -> None:
        """Test validation of valid stdio config."""
        config: dict[str, Any] = {"local": {"command": "/path/to/server", "args": ["-v"]}}
        errors = validate_mcp_config(config)

        assert errors == []

    def test_http_config_missing_url(self) -> None:
        """Test validation fails when HTTP config missing url."""
        config: dict[str, Any] = {"bad": {"type": "http"}}
        errors = validate_mcp_config(config)

        assert len(errors) == 1
        assert "missing 'url' field" in errors[0]

    def test_stdio_config_missing_command(self) -> None:
        """Test validation catches missing command field."""
        config: dict[str, Any] = {"bad": {"type": "stdio"}}  # This has type but not command
        errors = validate_mcp_config(config)

        # Should have error about unknown type (since it's not http/sse/sdk)
        assert len(errors) >= 1

    def test_invalid_url_scheme(self) -> None:
        """Test validation catches invalid URL scheme."""
        config: dict[str, Any] = {"bad": {"type": "http", "url": "ftp://example.com"}}
        errors = validate_mcp_config(config)

        assert len(errors) == 1
        assert "http://" in errors[0] or "https://" in errors[0]

    def test_unknown_type(self) -> None:
        """Test validation catches unknown type."""
        config: dict[str, Any] = {"bad": {"type": "unknown"}}
        errors = validate_mcp_config(config)

        assert len(errors) == 1
        assert "Unknown type" in errors[0]

    def test_invalid_headers_type(self) -> None:
        """Test validation catches invalid headers type."""
        config: dict[str, Any] = {"bad": {"type": "http", "url": "https://example.com", "headers": "not-a-dict"}}
        errors = validate_mcp_config(config)

        assert len(errors) == 1
        assert "headers" in errors[0]

    def test_invalid_args_type(self) -> None:
        """Test validation catches invalid args type."""
        config: dict[str, Any] = {"bad": {"command": "/path/to/server", "args": "not-a-list"}}
        errors = validate_mcp_config(config)

        assert len(errors) == 1
        assert "args" in errors[0]


class TestConvertAndValidateMcpConfig:
    """Tests for convert_and_validate_mcp_config function."""

    def test_valid_conversion(self) -> None:
        """Test valid conversion passes validation."""
        config = convert_and_validate_mcp_config({"api": "https://mcp.example.com/"})

        api_config = config["api"]
        assert api_config.get("type") == "http"
        assert api_config.get("url") == "https://mcp.example.com/"

    def test_conversion_always_produces_valid_config(self) -> None:
        """Test that convert_mcp_config always produces valid configs."""
        # All possible input types should produce valid SDK configs
        test_configs = {
            "http_url": "https://example.com/mcp/",
            "http_insecure": "http://localhost:8080/mcp/",
            "cmd_with_args": "npx -y @mcp/server",
            "simple_cmd": "/usr/bin/mcp-server",
        }

        # Should not raise any errors
        config = convert_and_validate_mcp_config(test_configs)

        # All should be present
        assert len(config) == 4
