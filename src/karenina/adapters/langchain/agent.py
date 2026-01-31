"""LangChain Agent adapter implementing the AgentPort interface.

This module provides the LangChainAgentAdapter class that wraps existing
LangGraph agent infrastructure behind the unified AgentPort interface.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import uuid
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
    Role,
    Tool,
)

from .messages import LangChainMessageConverter
from .usage import count_agent_turns, extract_usage_cumulative

if TYPE_CHECKING:
    from karenina.schemas.workflow.models import ModelConfig

logger = logging.getLogger(__name__)


def extract_partial_agent_state(
    agent: Any,
    messages: list[Any],
    exception: Exception,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract partial agent state after a limit is reached.

    Tries multiple methods to recover accumulated messages:
    1. Exception's state attribute
    2. Checkpointer's get_state method
    3. Exception's messages attribute
    4. Fallback to input messages

    Args:
        agent: The LangGraph agent
        messages: Original input messages
        exception: The limit exception
        config: Optional agent config with thread_id for checkpointer

    Returns:
        Dict with "messages" key containing recovered messages
    """
    # Method 1: Check if exception contains state information
    if hasattr(exception, "state") and exception.state is not None:
        logger.info("Extracted partial state from exception.state")
        state = exception.state
        return state if isinstance(state, dict) else {"messages": messages}

    # Method 2: Try to get current graph state if checkpointer exists
    if hasattr(agent, "checkpointer") and agent.checkpointer is not None:
        try:
            if hasattr(agent, "get_state"):
                state_config = config or {"configurable": {"thread_id": "default"}}
                state = agent.get_state(state_config)
                if state and hasattr(state, "values") and "messages" in state.values:
                    logger.info("Extracted partial state from graph checkpointer")
                    return {"messages": state.values["messages"]}
        except Exception as state_error:
            logger.debug(f"Could not extract state from checkpointer: {state_error}")

    # Method 3: Check if exception has accumulated messages attribute
    if hasattr(exception, "messages") and exception.messages is not None:
        logger.info("Extracted messages from exception.messages attribute")
        return {"messages": exception.messages}

    # FALLBACK: Return input messages with warning
    logger.warning(
        "Could not extract partial agent state after limit reached. "
        "Returning input messages only. Accumulated trace may be lost."
    )
    return {"messages": messages}


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
        ...     model_name="claude-sonnet-4-5",
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

    async def _create_agent(
        self,
        mcp_urls: dict[str, str],
        tool_filter: list[str] | None,
        kwargs: dict[str, Any],
    ) -> Any:
        """Create a LangGraph agent with MCP tools.

        This is the canonical agent creation path for the LangChain adapter.
        It handles:
        - Base model initialization via init_chat_model_unified
        - Async MCP tool loading (avoids threading issues)
        - Full middleware stack from AgentMiddlewareConfig

        The AgentPort is for tool-based interactions. For simple LLM calls
        without tools, use LLMPort instead.

        Args:
            mcp_urls: Dict mapping server names to URLs (required).
            tool_filter: List of tool names to filter, or None for all tools.
            kwargs: Additional kwargs for model initialization.

        Returns:
            A LangGraph agent ready for invocation.
        """
        from karenina.adapters.langchain.initialization import init_chat_model_unified
        from karenina.adapters.langchain.middleware import build_agent_middleware

        # Create base chat model using init_chat_model_unified
        # This handles all interface types: langchain, openrouter, openai_endpoint
        base_model = init_chat_model_unified(
            model=self._config.model_name or "",
            provider=self._config.model_provider,
            interface=self._config.interface or "langchain",
            endpoint_base_url=self._config.endpoint_base_url,
            endpoint_api_key=self._config.endpoint_api_key,
            **kwargs,
        )

        # Import agent creation utilities
        try:
            from langchain.agents import create_agent
            from langgraph.checkpoint.memory import InMemorySaver

            from karenina.adapters.langchain.mcp import create_mcp_client_and_tools
        except ImportError as e:
            raise ImportError(
                "langchain>=1.1.0, langgraph, and langchain-mcp-adapters are required. "
                "Install with: uv add 'langchain>=1.1.0' langgraph langchain-mcp-adapters"
            ) from e

        # Get MCP client and tools asynchronously (avoids threading issues)
        _, tools = await create_mcp_client_and_tools(mcp_urls, tool_filter)

        # Auto-detect context size for openai_endpoint interface if not explicitly configured
        max_context_tokens = self._config.max_context_tokens
        if (
            max_context_tokens is None
            and self._config.interface == "openai_endpoint"
            and self._config.endpoint_base_url
        ):
            from karenina.adapters.langchain.middleware import fetch_openai_endpoint_context_size

            max_context_tokens = fetch_openai_endpoint_context_size(
                base_url=self._config.endpoint_base_url,
                api_key=self._config.endpoint_api_key or "",
                model_name=self._config.model_name or "",
            )

        # Build middleware from configuration
        # Pass base_model so summarization uses the same model by default
        # Pass provider to enable provider-specific middleware (e.g., Anthropic prompt caching)
        middleware = build_agent_middleware(
            config=self._config.agent_middleware,
            max_context_tokens=max_context_tokens,
            interface=self._config.interface or "langchain",
            base_model=base_model,
            provider=self._config.model_provider,
        )

        # Create agent with tools, middleware, and checkpointer
        # InMemorySaver enables partial state recovery when limits are hit
        memory = InMemorySaver()
        agent: Any = create_agent(
            model=base_model,
            tools=tools,
            middleware=middleware,
            checkpointer=memory,
        )

        logger.info(f"Created agent with {len(tools)} MCP tools and {len(middleware)} middleware components")

        return agent

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
        from karenina.adapters.langchain.trace import (
            extract_final_ai_message_from_response,
            harmonize_agent_response,
        )

        config = config or AgentConfig()

        # Convert MCP servers to URL dict - required for agent adapter
        mcp_urls = self._convert_mcp_servers_to_urls(mcp_servers)
        if mcp_urls is None:
            raise AgentExecutionError(
                "AgentPort requires MCP servers or tools. For simple LLM calls without tools, use LLMPort instead."
            )

        tool_filter = self._convert_tools_to_names(tools)

        # Build kwargs for model initialization
        kwargs: dict[str, Any] = {}
        if self._config.temperature is not None:
            kwargs["temperature"] = self._config.temperature
        if self._config.extra_kwargs:
            kwargs.update(self._config.extra_kwargs)

        # Create agent with MCP tools
        try:
            agent = await self._create_agent(mcp_urls, tool_filter, kwargs)
        except Exception as e:
            raise AgentExecutionError(f"Failed to initialize agent: {e}") from e

        # Convert messages to LangChain format
        lc_messages = self._converter.to_provider(messages)

        # Prepare agent invocation
        recursion_limit_reached = False
        agent_response: dict[str, Any] = {"messages": []}

        # Use session-based thread_id for checkpointing
        thread_id = str(uuid.uuid4())
        agent_config = {"configurable": {"thread_id": thread_id}}

        # LangGraph agent expects dict with "messages" key
        coro = agent.ainvoke({"messages": lc_messages}, config=agent_config)

        try:
            if config.timeout:
                try:
                    agent_response = await asyncio.wait_for(coro, timeout=config.timeout)
                except TimeoutError as e:
                    raise AgentTimeoutError(f"Agent execution timed out after {config.timeout}s") from e
            else:
                agent_response = await coro

        except Exception as e:
            # Check if this is a recursion limit error
            error_str = str(e).lower()
            error_type = type(e).__name__

            if "graphrecursionerror" in error_type.lower() or "recursion_limit" in error_str:
                recursion_limit_reached = True
                logger.warning(f"Agent hit recursion limit: {e}")

                # Extract partial state using the shared recovery function
                agent_response = extract_partial_agent_state(agent, lc_messages, e, agent_config)
            else:
                raise AgentExecutionError(f"Agent execution failed: {e}") from e

        # Extract messages from response
        response_messages: list[Any] = []
        if isinstance(agent_response, dict) and "messages" in agent_response:
            response_messages = agent_response["messages"]
        elif isinstance(agent_response, list):
            response_messages = agent_response

        # Build raw_trace (legacy string format for backward compatibility)
        raw_trace = harmonize_agent_response(agent_response)
        if recursion_limit_reached:
            raw_trace += "\n\n[Note: Recursion limit reached - partial response shown]"

        # Build trace_messages (new structured format)
        # Exclude user messages — the trace should only contain assistant and
        # tool messages, matching the claude_tool adapter behavior.
        all_trace_messages = self._converter.from_provider(response_messages)
        trace_messages = [m for m in all_trace_messages if m.role != Role.USER]

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
        usage = extract_usage_cumulative(response_messages, model_name=self._config.model_name)
        turns = count_agent_turns(response_messages)

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
        from karenina.benchmark.verification.executor import get_async_portal

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

    async def aclose(self) -> None:
        """Close underlying resources.

        LangChain adapters delegate to LangChain's model management which
        handles its own HTTP client lifecycle. This is a no-op provided
        for interface consistency with other adapters.
        """
        pass


# Verify protocol compliance at import time
def _verify_protocol_compliance() -> None:
    """Verify LangChainAgentAdapter implements AgentPort protocol."""
    adapter_instance: AgentPort = None  # type: ignore[assignment]
    _ = adapter_instance
