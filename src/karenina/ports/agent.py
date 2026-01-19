"""Agent Port interface for multi-turn agent execution with tools/MCP.

This module defines the AgentPort Protocol for agentic LLM interactions
that involve tool use, MCP servers, and multi-turn conversations. Use this
for answer generation and other tasks requiring agent loops.

For simple LLM calls without tools, use LLMPort instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypedDict, runtime_checkable

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


@runtime_checkable
class AgentPort(Protocol):
    """Protocol for multi-turn agent execution with tools and MCP servers.

    AgentPort defines the interface for agentic LLM interactions that involve
    tool use, MCP servers, and multi-turn conversations. This is the primary
    interface for answer generation and other tasks requiring agent loops.

    The protocol is async-first (Claude Agent SDK has no sync API), with a
    sync wrapper for convenience.

    Implementations:
        - LangChainAgentAdapter: Uses LangGraph and langchain-mcp-adapters
        - ClaudeSDKAgentAdapter: Uses native Claude Agent SDK with MCP

    Example:
        >>> # Using an AgentPort implementation
        >>> agent: AgentPort = get_agent(model_config)
        >>> result = await agent.run(
        ...     messages=[Message.user("What files are in the repo?")],
        ...     mcp_servers={
        ...         "filesystem": {
        ...             "command": "npx",
        ...             "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"]
        ...         }
        ...     },
        ...     config=AgentConfig(max_turns=10)
        ... )
        >>> print(result.final_response)

    Notes:
        - MCP servers are passed to `run()` rather than at construction time
          to allow per-run server configuration
        - The `config` parameter controls execution behavior (max_turns, timeout, etc.)
        - Both `raw_trace` and `trace_messages` in AgentResult provide the same
          conversation data in different formats for backward compatibility
    """

    async def run(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        mcp_servers: dict[str, MCPServerConfig] | None = None,
        config: AgentConfig | None = None,
    ) -> AgentResult:
        """Execute an agent loop with optional tools and MCP servers.

        This is the primary async API for agent execution. The agent processes
        the input messages, potentially making tool calls and MCP requests,
        until it produces a final response or hits a limit.

        Args:
            messages: Initial conversation messages. Typically starts with a
                user message, but can include system prompts or prior context.
            tools: Optional list of Tool definitions the agent can invoke.
                These are standalone tools, not from MCP servers.
            mcp_servers: Optional dict mapping server names to MCP server
                configurations. Supports both stdio and HTTP/SSE transports.
            config: Optional AgentConfig controlling execution parameters
                like max_turns, timeout, and adapter-specific options.

        Returns:
            AgentResult containing the final response, trace (both formats),
            usage metadata, and execution metadata.

        Raises:
            AgentExecutionError: If the agent fails during execution.
            AgentTimeoutError: If the execution exceeds the timeout.
            AgentResponseError: If the response is malformed or invalid.
        """
        ...

    def run_sync(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        mcp_servers: dict[str, MCPServerConfig] | None = None,
        config: AgentConfig | None = None,
    ) -> AgentResult:
        """Synchronous wrapper for run().

        Convenience method that wraps the async `run()` method for use in
        synchronous contexts. Uses asyncio.run() internally.

        Args:
            messages: Initial conversation messages.
            tools: Optional list of Tool definitions.
            mcp_servers: Optional MCP server configurations.
            config: Optional AgentConfig for execution parameters.

        Returns:
            AgentResult from the agent execution.

        Raises:
            Same exceptions as run().

        Note:
            This method creates a new event loop. Do not call from within
            an existing async context - use `run()` directly instead.
        """
        ...
