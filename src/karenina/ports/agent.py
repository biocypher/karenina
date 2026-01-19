"""Agent Port interface for multi-turn agent execution with tools/MCP.

This module defines the AgentPort Protocol for agentic LLM interactions
that involve tool use, MCP servers, and multi-turn conversations. Use this
for answer generation and other tasks requiring agent loops.

For simple LLM calls without tools, use LLMPort instead.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict


@dataclass(frozen=True)
class Tool:
    """Definition of a tool that can be used by an agent.

    Tools are typically exposed via MCP servers, but this dataclass
    can represent any callable tool definition.

    Attributes:
        name: Unique identifier for the tool.
        description: Human-readable description of what the tool does.
        input_schema: JSON Schema defining the tool's input parameters.

    Example:
        >>> tool = Tool(
        ...     name="search_docs",
        ...     description="Search documentation for a query",
        ...     input_schema={
        ...         "type": "object",
        ...         "properties": {
        ...             "query": {"type": "string", "description": "Search query"},
        ...             "limit": {"type": "integer", "default": 10}
        ...         },
        ...         "required": ["query"]
        ...     }
        ... )
    """

    name: str
    description: str
    input_schema: dict[str, Any]


class MCPStdioServerConfig(TypedDict, total=False):
    """Configuration for stdio-based MCP servers.

    Stdio servers run as local processes that communicate via stdin/stdout.
    Use this for local MCP servers like filesystem or git tools.

    Attributes:
        type: Transport type - "stdio" or omitted (default).
        command: The command to run (e.g., "npx", "python").
        args: Command arguments (e.g., ["-y", "@modelcontextprotocol/server-github"]).
        env: Environment variables for the server process.

    Example:
        >>> config: MCPStdioServerConfig = {
        ...     "command": "npx",
        ...     "args": ["-y", "@modelcontextprotocol/server-github"],
        ...     "env": {"GITHUB_TOKEN": "ghp_xxx"}
        ... }
    """

    type: Literal["stdio"]
    command: str
    args: list[str]
    env: dict[str, str]


class MCPHttpServerConfig(TypedDict, total=False):
    """Configuration for HTTP/SSE-based MCP servers.

    HTTP servers communicate over network connections using either
    streaming HTTP (streamable_http) or Server-Sent Events (SSE).

    Attributes:
        type: Transport type - "http" for streamable HTTP, "sse" for SSE.
        url: Server URL endpoint.
        headers: HTTP headers (e.g., Authorization).

    Example:
        >>> config: MCPHttpServerConfig = {
        ...     "type": "http",
        ...     "url": "https://api.example.com/mcp",
        ...     "headers": {"Authorization": "Bearer token123"}
        ... }
    """

    type: Literal["http", "sse"]
    url: str
    headers: dict[str, str]


# Union type for MCP server configuration
# Supports both stdio (local process) and HTTP/SSE (remote) transports
MCPServerConfig = MCPStdioServerConfig | MCPHttpServerConfig


@dataclass
class AgentConfig:
    """Configuration for agent execution.

    Controls agent behavior including turn limits, prompts, and timeouts.

    Attributes:
        max_turns: Maximum number of conversation turns before stopping.
            Maps to recursion_limit in verification config. Default: 25.
        system_prompt: Optional system prompt to prepend to the conversation.
            If None, uses the adapter's default system prompt.
        timeout: Optional timeout in seconds for the entire agent run.
            If None, no timeout is applied.
        extra: Additional adapter-specific configuration options.
            Keys and values depend on the specific adapter implementation.

    Example:
        >>> config = AgentConfig(
        ...     max_turns=10,
        ...     system_prompt="You are a helpful assistant.",
        ...     timeout=60.0
        ... )

        >>> # With adapter-specific options
        >>> config = AgentConfig(
        ...     max_turns=25,
        ...     extra={
        ...         "permission_mode": "bypassPermissions",
        ...         "max_budget_usd": 1.0
        ...     }
        ... )
    """

    max_turns: int = 25
    system_prompt: str | None = None
    timeout: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)
