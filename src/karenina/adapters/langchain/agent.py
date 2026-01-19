"""LangChain Agent adapter implementing the AgentPort interface.

This module provides the LangChainAgentAdapter class that wraps existing
LangGraph agent infrastructure behind the unified AgentPort interface.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import TYPE_CHECKING, Any

from karenina.ports import (
    AgentConfig,
    AgentExecutionError,
    AgentPort,
    AgentResult,
    AgentTimeoutError,
    MCPHttpServerConfig,
    MCPServerConfig,
    Message,
    Tool,
    UsageMetadata,
)

from .messages import LangChainMessageConverter

if TYPE_CHECKING:
    from karenina.schemas.workflow.models import ModelConfig

logger = logging.getLogger(__name__)


class LangChainAgentAdapter:
    """Agent adapter using LangGraph with MCP tools.

    This adapter implements the AgentPort Protocol and wraps the existing
    karenina infrastructure for LangGraph-based agentic execution with
    optional MCP server integration.

    The adapter handles:
    - Message conversion between unified Message and LangChain formats
    - MCP server configuration conversion to LangChain format
    - Agent loop execution with recursion limit handling
    - Dual trace output (raw_trace string and trace_messages list)
    - Usage metadata extraction from LangChain responses

    Example:
        >>> from karenina.schemas.workflow.models import ModelConfig
        >>> config = ModelConfig(
        ...     id="claude-sonnet",
        ...     model_name="claude-sonnet-4-20250514",
        ...     model_provider="anthropic"
        ... )
        >>> adapter = LangChainAgentAdapter(config)
        >>> result = await adapter.run(
        ...     messages=[Message.user("What files are in /tmp?")],
        ...     mcp_servers={
        ...         "filesystem": {
        ...             "type": "http",
        ...             "url": "https://mcp.example.com/filesystem",
        ...         }
        ...     }
        ... )
        >>> print(result.final_response)
    """

    def __init__(self, model_config: ModelConfig) -> None:
        """Initialize the LangChain Agent adapter.

        Args:
            model_config: Configuration specifying model, provider, and interface.
        """
        self._config = model_config
        self._converter = LangChainMessageConverter()

    def _convert_mcp_servers_to_urls(
        self,
        mcp_servers: dict[str, MCPServerConfig] | None,
    ) -> dict[str, str] | None:
        """Convert MCPServerConfig dict to URL-based dict for LangChain.

        The existing LangChain infrastructure expects mcp_urls_dict with format:
        {"server_name": "url"} for HTTP/SSE servers.

        Note: Stdio servers are not currently supported by the existing
        init_chat_model_unified infrastructure - only HTTP/SSE.

        Args:
            mcp_servers: Dict mapping server names to MCPServerConfig.

        Returns:
            Dict mapping server names to URLs, or None if no servers.

        Raises:
            AgentExecutionError: If stdio transport is requested (not supported).
        """
        if not mcp_servers:
            return None

        urls: dict[str, str] = {}
        for name, config in mcp_servers.items():
            # Check transport type
            transport_type = config.get("type", "http")

            if transport_type == "stdio":
                # Stdio transport not supported via URL-based infrastructure
                raise AgentExecutionError(
                    f"MCP server '{name}' uses stdio transport which is not "
                    "supported by the LangChain adapter. Use HTTP/SSE transport instead."
                )

            # HTTP or SSE transport - extract URL
            http_config: MCPHttpServerConfig = config  # type: ignore[assignment]
            url = http_config.get("url")
            if not url:
                raise AgentExecutionError(f"MCP server '{name}' has type '{transport_type}' but no URL configured.")
            urls[name] = url

        return urls if urls else None

    def _convert_tools_to_names(self, tools: list[Tool] | None) -> list[str] | None:
        """Convert Tool list to tool name filter list.

        The existing infrastructure supports tool filtering by name.

        Args:
            tools: List of Tool definitions.

        Returns:
            List of tool names for filtering, or None if no filter.
        """
        if not tools:
            return None
        return [tool.name for tool in tools]

    def _extract_usage(self, messages: list[Any]) -> UsageMetadata:
        """Extract cumulative usage metadata from message history.

        LangGraph agents accumulate usage across multiple model calls.
        This extracts usage from AIMessage response_metadata.

        Args:
            messages: List of LangChain messages from agent execution.

        Returns:
            UsageMetadata with cumulative token counts.
        """
        total_input = 0
        total_output = 0
        total_cache_read = 0
        total_cache_creation = 0

        try:
            from langchain_core.messages import AIMessage
        except ImportError:
            return UsageMetadata(
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                model=self._config.model_name,
            )

        for msg in messages:
            if not isinstance(msg, AIMessage):
                continue

            # Try response_metadata first
            usage_data: dict[str, Any] = {}
            if hasattr(msg, "response_metadata") and msg.response_metadata:
                metadata = msg.response_metadata
                usage_data = metadata.get("token_usage") or metadata.get("usage") or {}

            # Fallback to usage_metadata (convert to dict if needed)
            if not usage_data and hasattr(msg, "usage_metadata") and msg.usage_metadata:
                um = msg.usage_metadata
                # Handle both dict and UsageMetadata object
                if isinstance(um, dict):
                    usage_data = um  # type: ignore[assignment]
                else:
                    # Convert UsageMetadata object to dict
                    usage_data = {
                        "input_tokens": getattr(um, "input_tokens", 0),
                        "output_tokens": getattr(um, "output_tokens", 0),
                        "cache_read_input_tokens": getattr(um, "cache_read_input_tokens", None),
                        "cache_creation_input_tokens": getattr(um, "cache_creation_input_tokens", None),
                    }

            if usage_data:
                total_input += int(usage_data.get("input_tokens") or usage_data.get("prompt_tokens") or 0)
                total_output += int(usage_data.get("output_tokens") or usage_data.get("completion_tokens") or 0)
                cache_read = usage_data.get("cache_read_input_tokens") or usage_data.get("cache_read_tokens")
                cache_creation = usage_data.get("cache_creation_input_tokens") or usage_data.get(
                    "cache_creation_tokens"
                )
                if cache_read:
                    total_cache_read += int(cache_read)
                if cache_creation:
                    total_cache_creation += int(cache_creation)

        return UsageMetadata(
            input_tokens=total_input,
            output_tokens=total_output,
            total_tokens=total_input + total_output,
            cache_read_tokens=total_cache_read if total_cache_read else None,
            cache_creation_tokens=total_cache_creation if total_cache_creation else None,
            model=self._config.model_name,
        )

    def _count_turns(self, messages: list[Any]) -> int:
        """Count the number of agent turns from message history.

        A turn is counted for each AIMessage in the trace.

        Args:
            messages: List of LangChain messages.

        Returns:
            Number of turns (AI message count).
        """
        try:
            from langchain_core.messages import AIMessage
        except ImportError:
            # Count by checking type name
            return sum(1 for m in messages if "ai" in type(m).__name__.lower())

        return sum(1 for m in messages if isinstance(m, AIMessage))

    async def run(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        mcp_servers: dict[str, MCPServerConfig] | None = None,
        config: AgentConfig | None = None,
    ) -> AgentResult:
        """Execute an agent loop with optional tools and MCP servers.

        Args:
            messages: Initial conversation messages.
            tools: Optional list of Tool definitions the agent can invoke.
            mcp_servers: Optional dict of MCP server configurations.
            config: Optional AgentConfig for execution parameters.

        Returns:
            AgentResult with final response, traces, usage, and metadata.

        Raises:
            AgentExecutionError: If the agent fails during execution.
            AgentTimeoutError: If execution exceeds the timeout.
        """
        from karenina.infrastructure.llm.interface import init_chat_model_unified
        from karenina.infrastructure.llm.mcp_utils import (
            extract_final_ai_message_from_response,
            harmonize_agent_response,
        )

        config = config or AgentConfig()

        # Convert MCP servers to URL dict
        mcp_urls = self._convert_mcp_servers_to_urls(mcp_servers)
        tool_filter = self._convert_tools_to_names(tools)

        # Build kwargs for model initialization
        kwargs: dict[str, Any] = {}
        if self._config.temperature is not None:
            kwargs["temperature"] = self._config.temperature
        if self._config.extra_kwargs:
            kwargs.update(self._config.extra_kwargs)

        # Initialize agent via existing infrastructure
        try:
            agent = init_chat_model_unified(
                model=self._config.model_name or "",
                provider=self._config.model_provider,
                interface=self._config.interface,
                endpoint_base_url=self._config.endpoint_base_url,
                endpoint_api_key=self._config.endpoint_api_key,
                max_context_tokens=self._config.max_context_tokens,
                mcp_urls_dict=mcp_urls,
                mcp_tool_filter=tool_filter,
                agent_middleware_config=self._config.agent_middleware,
                **kwargs,
            )
        except Exception as e:
            raise AgentExecutionError(f"Failed to initialize agent: {e}") from e

        # Convert messages to LangChain format
        lc_messages = self._converter.to_provider(messages)

        # Extract original question for trace processing
        original_question: str | None = None
        for msg in messages:
            if msg.text:
                original_question = msg.text
                break

        # Prepare agent invocation
        recursion_limit_reached = False
        agent_response: dict[str, Any] = {"messages": []}

        # Use session-based thread_id for checkpointing
        import uuid

        thread_id = str(uuid.uuid4())
        agent_config = {"configurable": {"thread_id": thread_id}}

        async def invoke_agent() -> tuple[dict[str, Any], bool]:
            """Invoke agent and handle recursion limit."""
            local_limit_reached = False

            # Apply timeout if configured
            coro = agent.ainvoke({"messages": lc_messages}, config=agent_config)
            if config.timeout:
                try:
                    response = await asyncio.wait_for(coro, timeout=config.timeout)
                except TimeoutError as e:
                    raise AgentTimeoutError(f"Agent execution timed out after {config.timeout}s") from e
            else:
                response = await coro

            return response, local_limit_reached

        try:
            agent_response, recursion_limit_reached = await invoke_agent()

        except Exception as e:
            # Check if this is a recursion limit error
            error_str = str(e).lower()
            error_type = type(e).__name__

            if "graphrecursionerror" in error_type.lower() or "recursion_limit" in error_str:
                recursion_limit_reached = True
                logger.warning(f"Agent hit recursion limit: {e}")

                # Try to get partial state from checkpointer
                if hasattr(agent, "checkpointer") and agent.checkpointer is not None:
                    try:
                        if hasattr(agent, "get_state"):
                            state = agent.get_state(agent_config)
                            if state and hasattr(state, "values") and "messages" in state.values:
                                agent_response = {"messages": state.values["messages"]}
                    except Exception:
                        pass

                # If we still don't have messages, use input
                if not agent_response.get("messages"):
                    agent_response = {"messages": lc_messages}
            else:
                raise AgentExecutionError(f"Agent execution failed: {e}") from e

        # Extract messages from response
        response_messages: list[Any] = []
        if isinstance(agent_response, dict) and "messages" in agent_response:
            response_messages = agent_response["messages"]
        elif isinstance(agent_response, list):
            response_messages = agent_response

        # Build raw_trace (legacy string format for backward compatibility)
        raw_trace = harmonize_agent_response(agent_response, original_question)
        if recursion_limit_reached:
            raw_trace += "\n\n[Note: Recursion limit reached - partial response shown]"

        # Build trace_messages (new structured format)
        trace_messages = self._converter.from_provider(response_messages)

        # Extract final response
        final_response, error = extract_final_ai_message_from_response(agent_response)
        if error or not final_response:
            # Fall back to last text in trace
            final_response = raw_trace.split("--- AI Message ---")[-1].strip() if raw_trace else ""
            # Further clean up: get just the text before any "Tool Calls:" section
            if "\nTool Calls:" in final_response:
                final_response = final_response.split("\nTool Calls:")[0].strip()
            if not final_response:
                final_response = "[No final response extracted]"

        # Extract usage and count turns
        usage = self._extract_usage(response_messages)
        turns = self._count_turns(response_messages)

        return AgentResult(
            final_response=final_response,
            raw_trace=raw_trace,
            trace_messages=trace_messages,
            usage=usage,
            turns=turns,
            limit_reached=recursion_limit_reached,
            session_id=thread_id,
            actual_model=self._config.model_name,
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

        Raises:
            Same exceptions as run().
        """
        from karenina.benchmark.verification.batch_runner import get_async_portal

        portal = get_async_portal()

        if portal is not None:
            return portal.call(self.run, messages, tools, mcp_servers, config)

        # No portal - check if we're in an async context
        try:
            asyncio.get_running_loop()
            # We're in an async context - use ThreadPoolExecutor

            def run_in_thread() -> AgentResult:
                return asyncio.run(self.run(messages, tools, mcp_servers, config))

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                timeout = config.timeout if config and config.timeout else 600
                return future.result(timeout=timeout)

        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            return asyncio.run(self.run(messages, tools, mcp_servers, config))


# Verify protocol compliance at import time
def _verify_protocol_compliance() -> None:
    """Verify LangChainAgentAdapter implements AgentPort protocol."""
    adapter_instance: AgentPort = None  # type: ignore[assignment]
    _ = adapter_instance
