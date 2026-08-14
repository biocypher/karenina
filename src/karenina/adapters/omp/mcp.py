"""Translate Karenina MCP configuration to ACP v1 server definitions."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from karenina.ports import AgentExecutionError, MCPServerConfig


def convert_mcp_servers(mcp_servers: dict[str, MCPServerConfig] | None) -> list[Any]:
    """Convert Karenina MCP dictionaries to official ACP schema models."""
    if not mcp_servers:
        return []

    from acp.schema import EnvVariable, HttpHeader, HttpMcpServer, McpServerStdio, SseMcpServer

    converted: list[Any] = []
    for name, server in mcp_servers.items():
        server_type = server.get("type", "stdio")
        if server_type == "stdio":
            command = str(server.get("command", ""))
            if not command:
                raise AgentExecutionError(f"MCP stdio server {name!r} is missing command")
            executable = command if Path(command).is_absolute() else shutil.which(command)
            if executable is None:
                raise AgentExecutionError(f"MCP stdio server executable not found: {command}")
            env_config = server.get("env")
            env = [
                EnvVariable(name=str(key), value=str(value))
                for key, value in (env_config.items() if isinstance(env_config, dict) else [])
            ]
            args_config = server.get("args")
            args = [str(arg) for arg in args_config] if isinstance(args_config, list) else []
            converted.append(
                McpServerStdio(
                    name=name,
                    command=str(Path(executable).resolve()),
                    args=args,
                    env=env,
                )
            )
            continue

        url = str(server.get("url", ""))
        if not url:
            raise AgentExecutionError(f"MCP {server_type} server {name!r} is missing url")
        headers_config = server.get("headers")
        headers = [
            HttpHeader(name=str(key), value=str(value))
            for key, value in (headers_config.items() if isinstance(headers_config, dict) else [])
        ]
        if server_type == "http":
            converted.append(HttpMcpServer(type="http", name=name, url=url, headers=headers))
        elif server_type == "sse":
            converted.append(SseMcpServer(type="sse", name=name, url=url, headers=headers))
        else:
            raise AgentExecutionError(f"Unsupported MCP transport for {name!r}: {server_type!r}")
    return converted
