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

from anthropic import Omit

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
from karenina.ports.capabilities import PortCapabilities
from karenina.schemas.config import ModelConfig

from .mcp import connect_all_mcp_servers, get_all_mcp_tools
from .messages import (
    build_system_with_cache,
    convert_from_anthropic_message,
    convert_to_anthropic,
    extract_system_prompt,
)
from .tools import ToolResultCollector, apply_cache_control_to_tool, wrap_mcp_tool, wrap_static_tool
from .trace import claude_tool_messages_to_raw_trace
from .usage import aggregate_usage, extract_usage_from_response

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
        >>> from karenina.schemas.config import ModelConfig
        >>> config = ModelConfig(
        ...     id="claude-haiku",
        ...     model_name="claude-haiku-4-5",
        ...     model_provider="anthropic",
        ...     interface="claude_tool"
        ... )
        >>> adapter = ClaudeToolAgentAdapter(config)
        >>> result = await adapter.arun(
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

    @property
    def capabilities(self) -> PortCapabilities:
        """Declare adapter capabilities.

        Returns:
            PortCapabilities with system_prompt=True.
        """
        return PortCapabilities(
            supports_system_prompt=True,
            supports_file_tools=True,
            supports_code_execution=True,
        )

    def _get_async_client(self) -> Any:
        """Get or create the async Anthropic client."""
        if self._async_client is None:
            from anthropic import AsyncAnthropic

            # Build kwargs for Anthropic client (api_key, base_url from config)
            kwargs: dict[str, Any] = {}
            if self._config.anthropic_api_key:
                kwargs["api_key"] = self._config.anthropic_api_key.get_secret_value()
            if self._config.anthropic_base_url:
                kwargs["base_url"] = self._config.anthropic_base_url

            self._async_client = AsyncAnthropic(**kwargs)
        return self._async_client

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

    async def arun(
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
            # Build tool list with result collector for trace capture
            all_tools: list[Any] = []
            collector = ToolResultCollector()

            # Wrap static tools (if any)
            if tools:
                for tool in tools:
                    all_tools.append(wrap_static_tool(tool, collector=collector))

            # Connect to MCP servers and wrap their tools
            if mcp_servers:
                sessions = await connect_all_mcp_servers(exit_stack, mcp_servers)
                mcp_tool_list = await get_all_mcp_tools(sessions)

                # Apply tool filter from model config or tools parameter
                tool_filter_names: set[str] | None = None
                if self._config.mcp_tool_filter:
                    tool_filter_names = set(self._config.mcp_tool_filter)
                    logger.info("Restricting Claude Tool agent to MCP tools: %s", self._config.mcp_tool_filter)

                for server_name, session, mcp_tool in mcp_tool_list:
                    if tool_filter_names and mcp_tool.name not in tool_filter_names:
                        logger.debug("Skipping MCP tool '%s' (not in tool filter)", mcp_tool.name)
                        continue
                    wrapped = wrap_mcp_tool(session, mcp_tool, server_name, collector=collector)
                    all_tools.append(wrapped)
                    logger.debug("Added MCP tool '%s' from server '%s'", mcp_tool.name, server_name)

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
                collector=collector,
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

    async def _single_turn_create(
        self,
        client: Any,
        kwargs: dict[str, Any],
    ) -> AgentResult:
        """Complete a single-turn request via messages.create().

        Used when no tools are provided. The tool_runner API requires a
        "tools" kwarg, so this method avoids it entirely by calling the
        standard messages.create() endpoint for a single completion.

        Args:
            client: The async Anthropic client.
            kwargs: Keyword arguments for messages.create() (model,
                max_tokens, messages, etc.).

        Returns:
            AgentResult with the single response.
        """
        logger.debug("No tools provided, using single-turn messages.create()")
        response = await client.messages.create(**kwargs)

        unified_msg = convert_from_anthropic_message(response)
        result_messages = [unified_msg]
        raw_trace = claude_tool_messages_to_raw_trace(result_messages)
        final_response = self._extract_final_response(result_messages)
        usage = extract_usage_from_response(response, model=self._config.model_name)

        actual_model = self._config.model_name
        if hasattr(response, "model") and response.model:
            actual_model = response.model

        return AgentResult(
            final_response=final_response,
            raw_trace=raw_trace,
            trace_messages=result_messages,
            usage=usage,
            turns=1,
            limit_reached=False,
            session_id=None,
            actual_model=actual_model,
        )

    async def _execute_agent_loop(
        self,
        messages: list[Message],
        tools: list[Any],
        config: AgentConfig,
        collector: ToolResultCollector | None = None,
    ) -> AgentResult:
        """Execute the agent loop using tool_runner.

        Args:
            messages: Input messages.
            tools: List of wrapped tool functions.
            config: Agent configuration.
            collector: Optional collector capturing tool results for trace.

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
            # Workaround for anthropic SDK 0.77.0 bug: tool_runner -> parse()
            # reassigns betas=[] when not given, then is_given([]) returns True,
            # producing an empty "anthropic-beta: " header that the API rejects.
            # Using Omit() removes the header during _merge_mappings.
            "extra_headers": {"anthropic-beta": Omit()},
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

        # No tools: single-turn completion without tool_runner.
        # tool_runner requires the "tools" kwarg, so when no tools are
        # provided we fall back to a plain messages.create() call.
        if not tools:
            if config.timeout:
                return await asyncio.wait_for(
                    self._single_turn_create(client, kwargs),
                    timeout=config.timeout,
                )
            return await self._single_turn_create(client, kwargs)

        # Collect messages and usage during the loop
        collected_responses: list[Any] = []
        trace_messages: list[Message] = []
        total_usage = UsageMetadata(model=self._config.model_name)
        turns = 0
        limit_reached = False

        # Track tool_use blocks from previous assistant message to inject
        # tool result messages after the tool_runner executes them
        prev_tool_uses: list[Any] = []

        async def execute_loop() -> None:
            nonlocal collected_responses, trace_messages, total_usage, turns, limit_reached
            nonlocal prev_tool_uses

            # Create the tool_runner
            runner = client.beta.messages.tool_runner(**kwargs)

            async for response in runner:
                turns += 1
                collected_responses.append(response)

                # Inject tool result messages from previous turn's tool_use blocks.
                # By the time tool_runner yields the next response, it has already
                # executed the tools from the previous response.
                if prev_tool_uses and collector is not None:
                    collected_results = collector.drain()
                    # Match by order: tool_runner executes tools in the order
                    # they appear in the assistant message
                    for i, tool_use in enumerate(prev_tool_uses):
                        if i < len(collected_results):
                            record = collected_results[i]
                            tool_msg = Message.tool_result(
                                tool_use_id=tool_use.id,
                                content=record.content,
                                is_error=record.is_error,
                            )
                        else:
                            # Fallback if collector missed a result
                            tool_msg = Message.tool_result(
                                tool_use_id=tool_use.id,
                                content="[Tool result not captured]",
                                is_error=False,
                            )
                        trace_messages.append(tool_msg)
                    prev_tool_uses = []

                # Convert response to unified Message
                unified_msg = convert_from_anthropic_message(response)
                trace_messages.append(unified_msg)

                # Track tool_use blocks for next iteration
                prev_tool_uses = unified_msg.tool_calls

                # Aggregate usage
                response_usage = extract_usage_from_response(response, model=self._config.model_name)
                total_usage = aggregate_usage(total_usage, response_usage)

                # Check turn limit
                if turns >= config.max_turns:
                    limit_reached = True
                    logger.warning("Agent hit turn limit (%d)", config.max_turns)
                    break

        # Execute with optional timeout
        timeout_reached = False
        if config.timeout:
            try:
                await asyncio.wait_for(execute_loop(), timeout=config.timeout)
            except TimeoutError:
                timeout_reached = True
                logger.warning(
                    "Agent timed out after %ss (%d turns, %d trace messages accumulated)",
                    config.timeout,
                    turns,
                    len(trace_messages),
                )
        else:
            await execute_loop()

        if not trace_messages:
            if timeout_reached:
                raise AgentTimeoutError(f"Agent execution timed out after {config.timeout}s with no messages")
            raise AgentResponseError("No messages received from tool_runner")

        # Build outputs
        raw_trace = claude_tool_messages_to_raw_trace(trace_messages)
        if limit_reached:
            raw_trace += "\n\n[Note: Turn limit reached - partial response shown]"
        if timeout_reached:
            raw_trace += "\n\n[Note: Agent timed out - partial response shown]"

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
            session_id=None,
            actual_model=actual_model,
            timeout_reached=timeout_reached,
        )

    def run(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        mcp_servers: dict[str, MCPServerConfig] | None = None,
        config: AgentConfig | None = None,
    ) -> AgentResult:
        """Synchronous wrapper for arun().

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
            return portal.call(self.arun, messages, tools, mcp_servers, config)

        try:
            asyncio.get_running_loop()

            def run_in_thread() -> AgentResult:
                return asyncio.run(self.arun(messages, tools, mcp_servers, config))

            timeout = config.timeout if config and config.timeout else 600
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=timeout)

        except RuntimeError:
            return asyncio.run(self.arun(messages, tools, mcp_servers, config))

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
