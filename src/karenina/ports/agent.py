"""Agent Port interface for multi-turn agent execution with tools/MCP.

This module defines the AgentPort Protocol for agentic LLM interactions
that involve tool use, MCP servers, and multi-turn conversations. Use this
for answer generation and other tasks requiring agent loops.

For simple LLM calls without tools, use LLMPort instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypedDict

if TYPE_CHECKING:
    from karenina.ports.messages import Message
    from karenina.ports.usage import UsageMetadata


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


@dataclass
class AgentResult:
    """Result from an agent execution.

    Contains both the final response and trace information in dual formats
    (raw string and structured messages) for backward compatibility and
    type-safe access.

    Attributes:
        final_response: The final text response from the agent. Extracted from
            the last assistant message in the conversation.
        raw_trace: Legacy string format of the conversation trace with
            delimiters (e.g., "--- AI Message ---"). Used for database storage,
            regex-based highlighting, and backward compatibility.
        trace_messages: Structured list of Message objects representing the
            full conversation trace. Enables type-safe access and the new
            structured trace display in the frontend.
        usage: Token and cost usage metadata for the entire agent run.
        turns: Number of conversation turns (agent iterations) completed.
        limit_reached: True if the agent stopped due to hitting max_turns
            or recursion limit, rather than completing naturally.
        session_id: Optional session identifier for checkpointing and resuming
            agent conversations. Adapter-specific (e.g., Claude SDK session ID).
        actual_model: The actual model that generated the response, which may
            differ from the requested model due to fallback or routing. Extracted
            from SDK AssistantMessage.model field.

    Example:
        >>> result = AgentResult(
        ...     final_response="The answer is 42.",
        ...     raw_trace="--- AI Message ---\\nThe answer is 42.",
        ...     trace_messages=[
        ...         Message.assistant("The answer is 42.")
        ...     ],
        ...     usage=UsageMetadata(input_tokens=10, output_tokens=5),
        ...     turns=1,
        ...     limit_reached=False
        ... )

    Note:
        Both `raw_trace` and `trace_messages` represent the same conversation
        but in different formats. During the migration period, both are produced
        by adapters to support existing and new frontend components.
    """

    final_response: str
    raw_trace: str
    trace_messages: list[Message]
    usage: UsageMetadata
    turns: int
    limit_reached: bool
    session_id: str | None = None
    actual_model: str | None = None
