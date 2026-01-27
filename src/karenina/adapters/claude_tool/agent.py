"""Claude Tool Agent adapter implementing the AgentPort interface.

This module provides the ClaudeToolAgentAdapter class that uses the Anthropic Python SDK's
client.beta.messages.tool_runner() for multi-turn agent execution with tool use and MCP
server support.

Key features:
- Uses tool_runner for automatic multi-turn agent loops
- Supports MCP servers via HTTP/SSE transport with session-per-run semantics
- Implements prompt caching for efficiency
- Produces dual trace output (raw_trace string and trace_messages list)
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from contextlib import AsyncExitStack
from typing import Any

from dotenv import load_dotenv

from karenina.ports import (
    AgentConfig,
    AgentExecutionError,
    AgentPort,
    AgentResponseError,
    AgentResult,
    AgentTimeoutError,
    MCPServerConfig,
    Message,
    Role,
    Tool,
    UsageMetadata,
)
from karenina.ports.messages import TextContent
from karenina.schemas.workflow.models import ModelConfig

from .mcp import connect_all_mcp_servers, get_all_mcp_tools
from .messages import (
    build_system_with_cache,
    convert_from_anthropic_message,
    convert_to_anthropic,
    extract_system_prompt,
)
from .tools import apply_cache_control_to_tool, wrap_mcp_tool, wrap_static_tool
from .usage import aggregate_usage, extract_usage_from_response

# Load environment variables from .env file (for ANTHROPIC_API_KEY)
load_dotenv()

logger = logging.getLogger(__name__)


class ClaudeToolAgentAdapter:
    """Agent adapter using Anthropic SDK's tool_runner for multi-turn execution.

    This adapter implements the AgentPort Protocol using client.beta.messages.tool_runner()
    for agent loops with tool use and MCP server support.

    The adapter handles:
    - Message conversion from unified Message to Anthropic SDK format
    - MCP server connection via HTTP/SSE transport
    - Tool wrapping with @beta_async_tool decorator
    - Automatic multi-turn agent loop execution
    - Dual trace output (raw_trace string and trace_messages list)
    - Usage metadata aggregation across turns

    Example:
        >>> from karenina.schemas.workflow.models import ModelConfig
        >>> config = ModelConfig(
        ...     id="claude-haiku",
        ...     model_name="claude-haiku-4-5",
        ...     model_provider="anthropic",
        ...     interface="claude_tool"
        ... )
        >>> adapter = ClaudeToolAgentAdapter(config)
        >>> result = await adapter.run(
        ...     messages=[Message.user("What files are in /tmp?")],
        ...     mcp_servers={
        ...         "open-targets": {
        ...             "type": "http",
        ...             "url": "https://mcp.platform.opentargets.org/mcp",
        ...         }
        ...     }
        ... )
        >>> print(result.final_response)
    """

    def __init__(self, model_config: ModelConfig) -> None:
        """Initialize the Claude Tool Agent adapter.

        Args:
            model_config: Configuration specifying model, provider, and interface.
        """
        self._config = model_config

        # Initialize client lazily
        self._async_client: Any = None

    def _get_async_client(self) -> Any:
        """Get or create the async Anthropic client."""
        if self._async_client is None:
            from anthropic import AsyncAnthropic

            self._async_client = AsyncAnthropic()
        return self._async_client

    def _build_raw_trace(self, trace_messages: list[Message]) -> str:
        """Build raw_trace string from collected messages.

        Produces a format compatible with existing infrastructure that uses
        delimiters like "--- AI Message ---".

        Args:
            trace_messages: List of unified Message objects.

        Returns:
            Formatted trace string.
        """
        parts: list[str] = []

        for msg in trace_messages:
            if msg.role == Role.ASSISTANT:
                parts.append("--- AI Message ---")
                # Add text content
                for block in msg.content:
                    if isinstance(block, TextContent):
                        parts.append(block.text)
                    elif hasattr(block, "name"):
                        # Tool use
                        parts.append(f"[Tool: {block.name}]")
            elif msg.role == Role.USER:
                # Skip user messages in trace (original question is known)
                pass
            elif msg.role == Role.TOOL:
                # Tool results
                for block in msg.content:
                    if hasattr(block, "content"):
                        preview = str(block.content)[:200]
                        parts.append(f"[Tool Result: {preview}...]")

        return "\n".join(parts)

    def _extract_final_response(self, trace_messages: list[Message]) -> str:
        """Extract final text response from trace messages.

        Args:
            trace_messages: List of unified Message objects.

        Returns:
            The final text response from the last assistant message.
        """
        # Find the last assistant message with text content
        for msg in reversed(trace_messages):
            if msg.role == Role.ASSISTANT:
                text = msg.text
                if text:
                    return text

        return "[No final response extracted]"

    async def run(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        mcp_servers: dict[str, MCPServerConfig] | None = None,
        config: AgentConfig | None = None,
    ) -> AgentResult:
        """Execute an agent loop with optional tools and MCP servers.

        Uses client.beta.messages.tool_runner() for automatic multi-turn execution.

        Args:
            messages: Initial conversation messages.
            tools: Optional list of Tool definitions the agent can invoke.
            mcp_servers: Optional dict of MCP server configurations (HTTP/SSE).
            config: Optional AgentConfig for execution parameters.

        Returns:
            AgentResult with final response, traces, usage, and metadata.

        Raises:
            AgentExecutionError: If the agent fails during execution.
            AgentTimeoutError: If execution exceeds the timeout.
            AgentResponseError: If the response is malformed or invalid.
        """
        config = config or AgentConfig()
        exit_stack = AsyncExitStack()

        try:
            # Build tool list
            all_tools: list[Any] = []

            # Wrap static tools (if any)
            if tools:
                for tool in tools:
                    all_tools.append(wrap_static_tool(tool))

            # Connect to MCP servers and wrap their tools
            if mcp_servers:
                sessions = await connect_all_mcp_servers(exit_stack, mcp_servers)
                mcp_tool_list = await get_all_mcp_tools(sessions)

                for server_name, session, mcp_tool in mcp_tool_list:
                    wrapped = wrap_mcp_tool(session, mcp_tool, server_name)
                    all_tools.append(wrapped)
                    logger.debug(f"Added MCP tool '{mcp_tool.name}' from server '{server_name}'")

            # Apply cache_control to the last tool for Anthropic prompt caching
            # This caches all tool definitions. Enabled by default, disable with
            # config.extra["cache_tools"] = False
            cache_tools = config.extra.get("cache_tools", True)
            if cache_tools and all_tools:
                all_tools[-1] = apply_cache_control_to_tool(all_tools[-1])
                logger.debug("Applied cache_control to last tool for prompt caching")

            # Execute agent loop
            result = await self._execute_agent_loop(
                messages=messages,
                tools=all_tools,
                config=config,
            )

            return result

        except TimeoutError as e:
            raise AgentTimeoutError(f"Agent execution timed out after {config.timeout}s") from e
        except Exception as e:
            if isinstance(e, AgentExecutionError | AgentTimeoutError | AgentResponseError):
                raise
            raise AgentExecutionError(f"Agent execution failed: {e}") from e
        finally:
            await exit_stack.aclose()

    async def _execute_agent_loop(
        self,
        messages: list[Message],
        tools: list[Any],
        config: AgentConfig,
    ) -> AgentResult:
        """Execute the agent loop using tool_runner.

        Args:
            messages: Input messages.
            tools: List of wrapped tool functions.
            config: Agent configuration.

        Returns:
            AgentResult with execution results.
        """
        client = self._get_async_client()

        # Convert messages to Anthropic format
        anthropic_messages = convert_to_anthropic(messages)
        system_prompt = extract_system_prompt(messages)

        # Use config system prompt if not in messages
        if not system_prompt and config.system_prompt:
            system_prompt = config.system_prompt
        elif not system_prompt and self._config.system_prompt:
            system_prompt = self._config.system_prompt

        # Build kwargs for tool_runner
        if not self._config.model_name:
            raise AgentExecutionError("model_name is required in ModelConfig")

        kwargs: dict[str, Any] = {
            "model": self._config.model_name,
            "max_tokens": self._config.max_tokens,
            "messages": anthropic_messages,
        }

        # Add tools if any
        if tools:
            kwargs["tools"] = tools

        # Add system with caching if present
        if system_prompt:
            cached_system = build_system_with_cache(system_prompt)
            if cached_system:
                kwargs["system"] = cached_system

        # Add temperature if specified
        if self._config.temperature is not None:
            kwargs["temperature"] = self._config.temperature

        # Collect messages and usage during the loop
        collected_responses: list[Any] = []
        trace_messages: list[Message] = []
        total_usage = UsageMetadata(model=self._config.model_name)
        turns = 0
        limit_reached = False

        async def execute_loop() -> None:
            nonlocal collected_responses, trace_messages, total_usage, turns, limit_reached

            # Create the tool_runner
            runner = client.beta.messages.tool_runner(**kwargs)

            async for response in runner:
                turns += 1
                collected_responses.append(response)

                # Convert response to unified Message
                unified_msg = convert_from_anthropic_message(response)
                trace_messages.append(unified_msg)

                # Aggregate usage
                response_usage = extract_usage_from_response(response, model=self._config.model_name)
                total_usage = aggregate_usage(total_usage, response_usage)

                # Check turn limit
                if turns >= config.max_turns:
                    limit_reached = True
                    logger.warning(f"Agent hit turn limit ({config.max_turns})")
                    break

        # Execute with optional timeout
        if config.timeout:
            await asyncio.wait_for(execute_loop(), timeout=config.timeout)
        else:
            await execute_loop()

        if not trace_messages:
            raise AgentResponseError("No messages received from tool_runner")

        # Build outputs
        raw_trace = self._build_raw_trace(trace_messages)
        if limit_reached:
            raw_trace += "\n\n[Note: Turn limit reached - partial response shown]"

        final_response = self._extract_final_response(trace_messages)

        # Extract actual model from last response if available
        actual_model = self._config.model_name
        if collected_responses:
            last_resp = collected_responses[-1]
            if hasattr(last_resp, "model") and last_resp.model:
                actual_model = last_resp.model

        return AgentResult(
            final_response=final_response,
            raw_trace=raw_trace,
            trace_messages=trace_messages,
            usage=total_usage,
            turns=turns,
            limit_reached=limit_reached,
            session_id=None,  # tool_runner doesn't provide session IDs
            actual_model=actual_model,
        )

    def run_sync(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        mcp_servers: dict[str, MCPServerConfig] | None = None,
        config: AgentConfig | None = None,
    ) -> AgentResult:
        """Synchronous wrapper for run().

        Args:
            messages: Initial conversation messages.
            tools: Optional list of Tool definitions.
            mcp_servers: Optional MCP server configurations.
            config: Optional AgentConfig for execution parameters.

        Returns:
            AgentResult from the agent execution.
        """
        from karenina.benchmark.verification.executor import get_async_portal

        portal = get_async_portal()

        if portal is not None:
            return portal.call(self.run, messages, tools, mcp_servers, config)

        try:
            asyncio.get_running_loop()

            def run_in_thread() -> AgentResult:
                return asyncio.run(self.run(messages, tools, mcp_servers, config))

            timeout = config.timeout if config and config.timeout else 600
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=timeout)

        except RuntimeError:
            return asyncio.run(self.run(messages, tools, mcp_servers, config))

    async def aclose(self) -> None:
        """Close underlying HTTP client resources.

        This method should be called when the adapter is no longer needed
        to properly close httpx connection pools and prevent resource leaks.
        Safe to call multiple times.
        """
        if self._async_client is not None:
            await self._async_client.close()
            self._async_client = None


# Protocol verification
def _verify_protocol_compliance() -> None:
    """Verify ClaudeToolAgentAdapter implements AgentPort protocol."""
    adapter_instance: AgentPort = None  # type: ignore[assignment]
    _ = adapter_instance
