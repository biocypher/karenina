"""MCP configuration converter for Claude Agent SDK.

Converts karenina's MCP config format to Claude Agent SDK format.

Karenina uses a simple dict mapping server names to URLs or commands:
    {"biocontext": "https://mcp.biocontext.ai/mcp/"}

The Claude Agent SDK uses a more structured format with explicit types:
    {
        "biocontext": {
            "type": "http",
            "url": "https://mcp.biocontext.ai/mcp/",
            "headers": {...}
        }
    }

IMPORTANT: Karenina's `streamable_http` transport maps to SDK's `type="http"`,
NOT `"sse"`. Both are URL-based but use different protocols.

Transport Mapping:
    | Karenina | SDK Type | Notes |
    |----------|----------|-------|
    | HTTP URL | type="http" | Streamable HTTP transport |
    | stdio | command + args | Standard subprocess |

Example:
    >>> from karenina.adapters.claude_agent_sdk.mcp import convert_mcp_config
    >>>
    >>> # Convert karenina MCP config
    >>> karenina_config = {
    ...     "biocontext": "https://mcp.biocontext.ai/mcp/",
    ...     "github": "npx -y @modelcontextprotocol/server-github",
    ...     "local": "/path/to/mcp-server"
    ... }
    >>> sdk_config = convert_mcp_config(karenina_config)
    >>> print(sdk_config)
    {
        "biocontext": {"type": "http", "url": "https://mcp.biocontext.ai/mcp/"},
        "github": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"]},
        "local": {"command": "/path/to/mcp-server"}
    }
"""

from __future__ import annotations

from typing import Any, TypedDict

from karenina.exceptions import McpError


class McpHttpServerConfig(TypedDict, total=False):
    """SDK configuration for HTTP/Streamable HTTP MCP servers."""

    type: str  # Must be "http"
    url: str
    headers: dict[str, str]


class McpStdioServerConfig(TypedDict, total=False):
    """SDK configuration for stdio MCP servers."""

    command: str
    args: list[str]
    env: dict[str, str]


# Union type for SDK MCP server configs
McpServerConfig = McpHttpServerConfig | McpStdioServerConfig | dict[str, Any]


def convert_mcp_config(
    mcp_urls_dict: dict[str, str],
    auth_headers: dict[str, str] | None = None,
) -> dict[str, McpServerConfig]:
    """Convert karenina MCP URLs to Claude Agent SDK format.

    Karenina uses a simplified format where servers are specified as either:
    - HTTP URLs (e.g., "https://mcp.example.com/mcp/")
    - Commands with args (e.g., "npx -y @modelcontextprotocol/server-github")
    - Simple paths/commands (e.g., "/path/to/mcp-server")

    The Claude Agent SDK requires explicit type configuration:
    - HTTP URLs -> {"type": "http", "url": ..., "headers": ...}
    - Commands -> {"command": ..., "args": [...]}

    IMPORTANT: Karenina's streamable_http transport maps to SDK's type="http",
    NOT "sse". This is a critical distinction.

    Args:
        mcp_urls_dict: Dictionary mapping server names to URLs or commands.
            Keys are server names (e.g., "biocontext").
            Values are either:
            - HTTP URLs: "https://mcp.example.com/mcp/"
            - Commands with args: "npx -y @mcp/server"
            - Simple commands/paths: "/path/to/server"
        auth_headers: Optional headers to include with HTTP requests.
            Applied to all HTTP servers in the config.

    Returns:
        Dictionary mapping server names to SDK-formatted configs.
        Each config is one of:
        - McpHttpServerConfig: {"type": "http", "url": ..., "headers": ...}
        - McpStdioServerConfig: {"command": ..., "args": [...]}

    Examples:
        >>> # HTTP URL
        >>> convert_mcp_config({"api": "https://mcp.example.com/mcp/"})
        {"api": {"type": "http", "url": "https://mcp.example.com/mcp/"}}

        >>> # HTTP URL with auth
        >>> convert_mcp_config(
        ...     {"api": "https://mcp.example.com/mcp/"},
        ...     auth_headers={"Authorization": "Bearer token123"}
        ... )
        {"api": {"type": "http", "url": "https://mcp.example.com/mcp/", "headers": {"Authorization": "Bearer token123"}}}

        >>> # Command with args
        >>> convert_mcp_config({"github": "npx -y @mcp/server-github"})
        {"github": {"command": "npx", "args": ["-y", "@mcp/server-github"]}}

        >>> # Simple path
        >>> convert_mcp_config({"local": "/usr/local/bin/mcp-server"})
        {"local": {"command": "/usr/local/bin/mcp-server"}}
    """
    result: dict[str, McpServerConfig] = {}

    for name, url_or_command in mcp_urls_dict.items():
        if url_or_command.startswith(("http://", "https://")):
            # HTTP URLs -> Streamable HTTP (type="http", NOT "sse")
            config: McpHttpServerConfig = {"type": "http", "url": url_or_command}
            if auth_headers:
                config["headers"] = auth_headers
            result[name] = config
        elif " " in url_or_command:
            # Command with args - split on whitespace
            parts = url_or_command.split()
            result[name] = {"command": parts[0], "args": parts[1:]}
        else:
            # Path or simple command
            result[name] = {"command": url_or_command}

    return result


class McpConfigValidationError(McpError):
    """Raised when MCP configuration is invalid."""

    def __init__(self, message: str, server_name: str | None = None) -> None:
        super().__init__(message)
        self.server_name = server_name


def validate_mcp_config(config: dict[str, McpServerConfig]) -> list[str]:
    """Validate an SDK MCP configuration for correctness.

    Checks that each server config has the required fields based on its type:
    - HTTP configs: Must have "type" == "http" and "url"
    - stdio configs: Must have "command"

    Args:
        config: SDK-formatted MCP server configuration dict.

    Returns:
        List of validation error messages. Empty list if config is valid.

    Examples:
        >>> errors = validate_mcp_config({
        ...     "api": {"type": "http", "url": "https://example.com/mcp/"},
        ...     "local": {"command": "/path/to/server"}
        ... })
        >>> print(errors)
        []

        >>> errors = validate_mcp_config({
        ...     "bad": {"type": "http"}  # Missing url
        ... })
        >>> print(errors)
        ["Server 'bad': HTTP config missing 'url' field"]
    """
    errors: list[str] = []

    for name, server_config in config.items():
        if not isinstance(server_config, dict):
            errors.append(f"Server '{name}': Config must be a dict, got {type(server_config).__name__}")
            continue

        # Determine config type using .get() to avoid TypedDict key errors
        if "type" in server_config:
            config_type = server_config.get("type")
            if config_type == "http":
                # Validate HTTP config
                url = server_config.get("url")
                if not url:
                    errors.append(f"Server '{name}': HTTP config missing 'url' field")
                elif not isinstance(url, str):
                    errors.append(f"Server '{name}': 'url' must be a string")
                elif not url.startswith(("http://", "https://")):
                    errors.append(f"Server '{name}': URL must start with http:// or https://")

                # Validate headers if present
                if "headers" in server_config:
                    headers = server_config.get("headers")
                    if not isinstance(headers, dict):
                        errors.append(f"Server '{name}': 'headers' must be a dict")
                    elif not all(isinstance(k, str) and isinstance(v, str) for k, v in headers.items()):
                        errors.append(f"Server '{name}': 'headers' must be dict[str, str]")

            elif config_type == "sse":
                # SSE is valid but we warn that karenina uses HTTP
                if not server_config.get("url"):
                    errors.append(f"Server '{name}': SSE config missing 'url' field")

            elif config_type == "sdk":
                # SDK type is for in-process servers, typically from create_sdk_mcp_server()
                # No validation needed - handled by SDK
                pass

            else:
                errors.append(f"Server '{name}': Unknown type '{config_type}'. Expected 'http', 'sse', or 'sdk'")

        elif "command" in server_config:
            # Validate stdio config
            command = server_config.get("command")
            if not isinstance(command, str):
                errors.append(f"Server '{name}': 'command' must be a string")

            if "args" in server_config:
                args = server_config.get("args")
                if not isinstance(args, list):
                    errors.append(f"Server '{name}': 'args' must be a list")
                elif not all(isinstance(arg, str) for arg in args):
                    errors.append(f"Server '{name}': 'args' must be list[str]")

            if "env" in server_config:
                env = server_config.get("env")
                if not isinstance(env, dict):
                    errors.append(f"Server '{name}': 'env' must be a dict")
                elif not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
                    errors.append(f"Server '{name}': 'env' must be dict[str, str]")

        else:
            errors.append(f"Server '{name}': Config must have either 'type' (for HTTP/SSE) or 'command' (for stdio)")

    return errors


def convert_and_validate_mcp_config(
    mcp_urls_dict: dict[str, str],
    auth_headers: dict[str, str] | None = None,
) -> dict[str, McpServerConfig]:
    """Convert and validate karenina MCP config to SDK format.

    Combines convert_mcp_config() and validate_mcp_config() for convenience.
    Raises McpConfigValidationError if validation fails.

    Args:
        mcp_urls_dict: Dictionary mapping server names to URLs or commands.
        auth_headers: Optional headers to include with HTTP requests.

    Returns:
        Validated SDK-formatted MCP server configuration.

    Raises:
        McpConfigValidationError: If the converted config is invalid.

    Examples:
        >>> config = convert_and_validate_mcp_config({
        ...     "api": "https://mcp.example.com/mcp/"
        ... })
        >>> # Returns validated config
    """
    config = convert_mcp_config(mcp_urls_dict, auth_headers)
    errors = validate_mcp_config(config)

    if errors:
        error_msg = "MCP configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise McpConfigValidationError(error_msg)

    return config
