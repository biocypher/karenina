"""MCP server configuration conversion for Deep Agents adapter.

Converts karenina's MCPServerConfig (TypedDict union) to parameters
compatible with langchain-mcp-adapters' MultiServerMCPClient.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_mcp_server_params(
    mcp_servers: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Convert MCPServerConfig dict to langchain-mcp-adapters parameters.

    Discriminates stdio vs HTTP/SSE transports using the 'type' field
    from MCPServerConfig TypedDicts.

    Args:
        mcp_servers: Dict mapping server names to MCPServerConfig.

    Returns:
        Dict of server parameters for MultiServerMCPClient.
    """
    if not mcp_servers:
        return {}

    server_params: dict[str, dict[str, Any]] = {}

    for name, config in mcp_servers.items():
        if not isinstance(config, dict):
            logger.warning("Skipping non-dict MCP config for server %s", name)
            continue

        transport_type = config.get("type", "stdio")

        if transport_type == "stdio":
            params: dict[str, Any] = {
                "transport": "stdio",
                "command": config["command"],
                "args": config.get("args", []),
            }
            env = config.get("env")
            if env:
                params["env"] = env
            server_params[name] = params

        elif transport_type in ("http", "sse"):
            # Map karenina's "http" type to the library's "sse" transport
            params = {
                "transport": "sse" if transport_type == "http" else transport_type,
                "url": config["url"],
            }
            headers = config.get("headers")
            if headers:
                params["headers"] = headers
            server_params[name] = params

        else:
            logger.warning(
                "Unknown MCP transport type '%s' for server %s",
                transport_type,
                name,
            )

    return server_params


async def convert_mcp_to_tools(
    mcp_servers: dict[str, Any] | None,
) -> list[Any]:
    """Convert MCP server configs to LangChain tools via langchain-mcp-adapters.

    Creates a MultiServerMCPClient, connects to all servers, and returns
    their tools as LangChain BaseTool instances.

    Args:
        mcp_servers: Dict mapping server names to MCPServerConfig.

    Returns:
        List of LangChain BaseTool instances from all MCP servers.
    """
    server_params = build_mcp_server_params(mcp_servers)
    if not server_params:
        return []

    from langchain_mcp_adapters.client import MultiServerMCPClient

    async with MultiServerMCPClient(server_params) as client:  # type: ignore[arg-type,misc]
        return await client.get_tools()
