"""Claude Agent SDK Agent adapter implementing the AgentPort interface.

This module provides the ClaudeSDKAgentAdapter class that implements the AgentPort
interface using the Claude Agent SDK's ClaudeSDKClient for agent loops with MCP
tool support.

IMPORTANT: Uses ClaudeSDKClient (NOT query()) because:
- ClaudeSDKClient provides MCP server integration
- Supports hooks and interrupts for complex workflows
- Manages multi-turn conversation state

Key differences from LangChain:
- Claude SDK uses string prompts, not message arrays
- System prompts go in ClaudeAgentOptions.system_prompt
- MCP servers are configured in ClaudeAgentOptions.mcp_servers
- Uses ClaudeSDKClient context manager pattern
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
    AgentResponseError,
    AgentResult,
    AgentTimeoutError,
    MCPServerConfig,
    Message,
    Tool,
    UsageMetadata,
)

from .mcp import convert_mcp_config
from .messages import ClaudeSDKMessageConverter
from .trace import sdk_messages_to_raw_trace
from .usage import extract_sdk_usage

if TYPE_CHECKING:
    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage

    from karenina.schemas.workflow.models import ModelConfig

logger = logging.getLogger(__name__)


class ClaudeSDKAgentAdapter:
    """Agent adapter using Claude Agent SDK's ClaudeSDKClient.

    This adapter implements the AgentPort Protocol for agent execution with
    MCP tools and multi-turn conversation support. Uses ClaudeSDKClient for
    full agent capabilities including tool invocation and hooks.

    The adapter handles:
    - Message conversion from unified Message to prompt string
    - MCP server configuration conversion to SDK format
    - Agent loop execution with turn limit handling
    - Dual trace output (raw_trace string and trace_messages list)
    - Usage metadata extraction from SDK responses

    Example:
        >>> from karenina.schemas.workflow.models import ModelConfig
        >>> config = ModelConfig(
        ...     id="claude-sonnet",
        ...     model_name="claude-sonnet-4-20250514",
        ...     model_provider="anthropic",
        ...     interface="claude_agent_sdk"
        ... )
        >>> adapter = ClaudeSDKAgentAdapter(config)
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
        """Initialize the Claude SDK Agent adapter.

        Args:
            model_config: Configuration specifying model, provider, and interface.
        """
        self._config = model_config
        self._converter = ClaudeSDKMessageConverter()

    def _build_options(
        self,
        system_prompt: str | None,
        mcp_servers: dict[str, Any] | None,
        config: AgentConfig,
        tools: list[Tool] | None = None,
    ) -> ClaudeAgentOptions:
        """Build ClaudeAgentOptions for the agent run.

        Args:
            system_prompt: System prompt extracted from messages.
            mcp_servers: SDK-formatted MCP server configuration.
            config: Agent execution configuration.
            tools: Optional list of tool definitions.

        Returns:
            Configured ClaudeAgentOptions.
        """
        from claude_agent_sdk import ClaudeAgentOptions

        options_kwargs: dict[str, Any] = {
            # Use bypassPermissions for batch/automated calls
            "permission_mode": "bypassPermissions",
            # Map AgentConfig.max_turns to SDK
            "max_turns": config.max_turns,
        }

        # Add system prompt if provided
        if system_prompt:
            options_kwargs["system_prompt"] = system_prompt
        elif self._config.system_prompt:
            options_kwargs["system_prompt"] = self._config.system_prompt

        # Add model specification if provided
        if self._config.model_name:
            options_kwargs["model"] = self._config.model_name

        # Add MCP servers if provided
        if mcp_servers:
            options_kwargs["mcp_servers"] = mcp_servers

            # IMPORTANT: When MCP servers are configured, restrict tools to ONLY MCP tools.
            # This prevents the agent from using default Claude Code tools (Read, Grep, Bash, etc.)
            # which can distract from the actual MCP tool usage.
            #
            # Build allowed_tools list from MCP server names and tool filter.
            # MCP tools follow naming convention: mcp__<server_name>__<tool_name>
            mcp_tool_filter = self._config.mcp_tool_filter
            if mcp_tool_filter:
                # User specified specific tools - build full MCP tool names
                allowed_tools = []
                for server_name in mcp_servers:
                    for tool_name in mcp_tool_filter:
                        allowed_tools.append(f"mcp__{server_name}__{tool_name}")
                options_kwargs["allowed_tools"] = allowed_tools
                logger.info(f"Restricting SDK to MCP tools: {allowed_tools}")
            # If no tool filter specified but MCP servers are configured,
            # we cannot pre-filter (don't know tool names yet).
            # The agent will have access to all tools including defaults.
            # TODO: Consider fetching tool list from MCP servers to build filter.

        # Add tool filter if specific tools are requested (non-MCP case)
        if tools and "allowed_tools" not in options_kwargs:
            # SDK supports tool filtering via allowed_tools
            options_kwargs["allowed_tools"] = [t.name for t in tools]

        # Apply any extra options from config
        if config.extra:
            for key, value in config.extra.items():
                # Don't override critical options
                if key not in ("permission_mode", "max_turns", "mcp_servers", "allowed_tools"):
                    options_kwargs[key] = value

        # Build env dict for Anthropic settings (api_key, base_url)
        # The Claude Agent SDK reads ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL from env
        env_vars: dict[str, str] = {}
        if self._config.anthropic_api_key:
            env_vars["ANTHROPIC_API_KEY"] = self._config.anthropic_api_key.get_secret_value()
        if self._config.anthropic_base_url:
            env_vars["ANTHROPIC_BASE_URL"] = self._config.anthropic_base_url

        if env_vars:
            options_kwargs["env"] = env_vars

        return ClaudeAgentOptions(**options_kwargs)

    def _build_raw_trace(self, messages: list[Any]) -> str:
        """Build raw_trace string from SDK messages.

        Delegates to sdk_messages_to_raw_trace() from trace module.
        Matches the format used by LangChain adapter's harmonize_agent_response().

        Args:
            messages: List of SDK message objects.

        Returns:
            Formatted trace string compatible with existing infrastructure.
        """
        return sdk_messages_to_raw_trace(messages, include_user_messages=False)

    def _extract_final_response(self, messages: list[Any], result_msg: Any) -> str:
        """Extract final text response from SDK messages.

        Args:
            messages: List of SDK message objects.
            result_msg: ResultMessage from SDK.

        Returns:
            The final text response.
        """
        # First try ResultMessage.result (SDK's extraction)
        if hasattr(result_msg, "result") and result_msg.result:
            return str(result_msg.result)

        # Fall back to last AssistantMessage text content
        try:
            from claude_agent_sdk import AssistantMessage
            from claude_agent_sdk.types import TextBlock
        except ImportError:
            return "[Unable to extract response - SDK types not available]"

        for msg in reversed(messages):
            if isinstance(msg, AssistantMessage):
                text_parts = []
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        text_parts.append(block.text)
                if text_parts:
                    return "\n".join(text_parts)

        return "[No final response extracted]"

    def _extract_actual_model(self, messages: list[Any]) -> str | None:
        """Extract actual model name from AssistantMessage.model field.

        The SDK includes the actual model that generated each response
        in the AssistantMessage.model field.

        Args:
            messages: List of SDK message objects.

        Returns:
            The model name from the last AssistantMessage, or None.
        """
        try:
            from claude_agent_sdk import AssistantMessage
        except ImportError:
            return None

        for msg in reversed(messages):
            if isinstance(msg, AssistantMessage) and hasattr(msg, "model") and msg.model:
                return msg.model

        return None

    def _convert_mcp_servers(
        self,
        mcp_servers: dict[str, MCPServerConfig] | None,
    ) -> dict[str, Any] | None:
        """Convert MCPServerConfig to SDK format.

        Handles both already-SDK-formatted configs and karenina's simplified format.

        Args:
            mcp_servers: MCP server configuration.

        Returns:
            SDK-formatted MCP configuration, or None.
        """
        if not mcp_servers:
            return None

        # Check if already in SDK format (has 'type' or 'command' keys)
        first_config: Any = next(iter(mcp_servers.values()), {})
        if isinstance(first_config, dict) and ("type" in first_config or "command" in first_config):
            # Already SDK format - return as-is
            return mcp_servers

        # Assume simplified format {"name": "url"} - convert
        # This handles the case where mcp_servers is still in karenina's
        # simplified URL dict format
        url_dict: dict[str, str] = {}
        for name, config in mcp_servers.items():
            if isinstance(config, str):
                url_dict[name] = config
            elif isinstance(config, dict):
                url = config.get("url")
                if url and isinstance(url, str):
                    url_dict[name] = url

        if url_dict:
            return convert_mcp_config(url_dict)

        return mcp_servers

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
            AgentResponseError: If the response is malformed or invalid.
        """
        from claude_agent_sdk import ClaudeSDKClient, ResultMessage

        config = config or AgentConfig()

        # Convert messages to SDK format
        prompt_string = self._converter.to_prompt_string(messages)
        system_prompt = self._converter.extract_system_prompt(messages)

        # Convert MCP servers to SDK format
        sdk_mcp_servers = self._convert_mcp_servers(mcp_servers)

        # Build options
        options = self._build_options(system_prompt, sdk_mcp_servers, config, tools)

        # Execute agent loop using ClaudeSDKClient
        collected_messages: list[Any] = []
        result_message: ResultMessage | None = None
        limit_reached = False

        async def execute_agent() -> None:
            nonlocal collected_messages, result_message, limit_reached

            async with ClaudeSDKClient(options) as client:
                # Send initial query
                await client.query(prompt_string)

                # Collect all messages from response
                # IMPORTANT: Let iteration complete naturally - no break
                async for msg in client.receive_response():
                    collected_messages.append(msg)

                    if isinstance(msg, ResultMessage):
                        result_message = msg
                        # Check if limit was reached
                        if hasattr(msg, "subtype"):
                            subtype = msg.subtype or ""
                            if "limit" in subtype.lower() or "max" in subtype.lower():
                                limit_reached = True

        try:
            if config.timeout:
                await asyncio.wait_for(execute_agent(), timeout=config.timeout)
            else:
                await execute_agent()

        except TimeoutError as e:
            raise AgentTimeoutError(f"Agent execution timed out after {config.timeout}s") from e
        except Exception as e:
            error_str = str(e).lower()

            # Check for limit-related errors
            if "recursion" in error_str or "limit" in error_str or "max_turns" in error_str:
                limit_reached = True
                logger.warning(f"Agent hit turn limit: {e}")
            else:
                raise AgentExecutionError(f"Agent execution failed: {e}") from e

        if result_message is None and not collected_messages:
            raise AgentResponseError("No messages received from SDK agent")

        # Build raw_trace (legacy string format)
        raw_trace = self._build_raw_trace(collected_messages)
        if limit_reached:
            raw_trace += "\n\n[Note: Recursion limit reached - partial response shown]"

        # Build trace_messages (structured format)
        trace_messages = self._converter.from_provider(collected_messages)

        # Extract final response
        final_response = self._extract_final_response(collected_messages, result_message)

        # Extract usage
        usage = (
            extract_sdk_usage(result_message, model=self._config.model_name)
            if result_message
            else UsageMetadata(
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                model=self._config.model_name,
            )
        )

        # Extract turns from ResultMessage
        turns = result_message.num_turns if result_message and hasattr(result_message, "num_turns") else 0

        # Extract actual model
        actual_model = self._extract_actual_model(collected_messages) or self._config.model_name

        # Session ID - SDK may provide one
        session_id = None
        if result_message and hasattr(result_message, "session_id"):
            session_id = result_message.session_id

        return AgentResult(
            final_response=final_response,
            raw_trace=raw_trace,
            trace_messages=trace_messages,
            usage=usage,
            turns=turns or len([m for m in trace_messages if m.role.value == "assistant"]),
            limit_reached=limit_reached,
            session_id=session_id,
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

            timeout = config.timeout if config and config.timeout else 600
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=timeout)

        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            return asyncio.run(self.run(messages, tools, mcp_servers, config))

    async def aclose(self) -> None:
        """Close underlying resources.

        The Claude SDK agent adapter uses ClaudeSDKClient as a context manager
        which handles its own cleanup, so this is a no-op. Provided for
        interface consistency with other adapters that do require cleanup.
        """
        pass


# Verify protocol compliance at import time
def _verify_protocol_compliance() -> None:
    """Verify ClaudeSDKAgentAdapter implements AgentPort protocol."""
    adapter_instance: AgentPort = None  # type: ignore[assignment]
    _ = adapter_instance
