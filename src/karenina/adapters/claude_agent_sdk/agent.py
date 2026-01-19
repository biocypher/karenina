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

        # Add tool filter if specific tools are requested
        if tools:
            # SDK supports tool filtering via allowed_tools
            options_kwargs["allowed_tools"] = [t.name for t in tools]

        # Apply any extra options from config
        if config.extra:
            for key, value in config.extra.items():
                # Don't override critical options
                if key not in ("permission_mode", "max_turns", "mcp_servers"):
                    options_kwargs[key] = value

        return ClaudeAgentOptions(**options_kwargs)

    def _build_raw_trace(self, messages: list[Any]) -> str:
        """Build raw_trace string from SDK messages.

        Matches the format used by LangChain adapter's harmonize_agent_response().
        Uses delimiters like "--- AI Message ---" and "--- Tool Message ---".

        Args:
            messages: List of SDK message objects.

        Returns:
            Formatted trace string compatible with existing infrastructure.
        """
        try:
            from claude_agent_sdk import AssistantMessage, UserMessage
            from claude_agent_sdk.types import (
                TextBlock,
                ThinkingBlock,
                ToolResultBlock,
                ToolUseBlock,
            )
        except ImportError:
            return "[SDK types not available - cannot format trace]"

        trace_parts: list[str] = []

        for msg in messages:
            if isinstance(msg, UserMessage):
                # Skip user messages in trace (matches LangChain behavior)
                continue

            elif isinstance(msg, AssistantMessage):
                parts = self._format_assistant_message_for_trace(
                    msg, TextBlock, ToolUseBlock, ToolResultBlock, ThinkingBlock
                )
                trace_parts.extend(parts)

        return "\n\n".join(trace_parts)

    def _format_assistant_message_for_trace(
        self,
        msg: Any,
        TextBlock: type,
        ToolUseBlock: type,
        ToolResultBlock: type,
        ThinkingBlock: type,
    ) -> list[str]:
        """Format an AssistantMessage for trace output.

        Args:
            msg: AssistantMessage from SDK.
            TextBlock: SDK TextBlock type.
            ToolUseBlock: SDK ToolUseBlock type.
            ToolResultBlock: SDK ToolResultBlock type.
            ThinkingBlock: SDK ThinkingBlock type.

        Returns:
            List of formatted trace sections.
        """
        result: list[str] = []
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[tuple[str, str, bool]] = []  # (call_id, content, is_error)

        for block in msg.content:
            if isinstance(block, TextBlock):
                text_parts.append(block.text)  # type: ignore[attr-defined]

            elif isinstance(block, ThinkingBlock):
                # Include thinking in trace with its own header
                result.append(f"--- Thinking ---\n{block.thinking}")  # type: ignore[attr-defined]

            elif isinstance(block, ToolUseBlock):
                tool_calls.append(
                    {
                        "name": block.name,  # type: ignore[attr-defined]
                        "id": block.id,  # type: ignore[attr-defined]
                        "args": block.input if isinstance(block.input, dict) else {},  # type: ignore[attr-defined]
                    }
                )

            elif isinstance(block, ToolResultBlock):
                content_str = ""
                if block.content:  # type: ignore[attr-defined]
                    if isinstance(block.content, str):  # type: ignore[attr-defined]
                        content_str = block.content  # type: ignore[attr-defined]
                    elif isinstance(block.content, list):  # type: ignore[attr-defined]
                        parts = []
                        for c in block.content:  # type: ignore[attr-defined]
                            if hasattr(c, "text"):
                                parts.append(c.text)
                        content_str = "\n".join(parts)
                    else:
                        content_str = str(block.content)  # type: ignore[attr-defined]

                is_error = getattr(block, "is_error", False)
                if not isinstance(is_error, bool):
                    is_error = False

                tool_results.append((block.tool_use_id, content_str, is_error))  # type: ignore[attr-defined]

        # Build AI Message section (matches LangChain format)
        if text_parts or tool_calls:
            header = "--- AI Message ---"
            content_parts: list[str] = []

            if text_parts:
                content_parts.append("\n".join(text_parts))

            if tool_calls:
                content_parts.append("\nTool Calls:")
                for tc in tool_calls:
                    content_parts.append(f"  {tc['name']} (call_{tc['id']})")
                    content_parts.append(f"   Call ID: {tc['id']}")
                    if tc["args"]:
                        content_parts.append(f"   Args: {tc['args']}")

            result.append(f"{header}\n{''.join(content_parts)}")

        # Add tool result sections
        for call_id, content, is_error in tool_results:
            header = f"--- Tool Message (call_id: {call_id}) ---"
            if is_error:
                content = f"[ERROR] {content}"
            result.append(f"{header}\n{content}")

        return result

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

            timeout = config.timeout if config and config.timeout else 600
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=timeout)

        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            return asyncio.run(self.run(messages, tools, mcp_servers, config))


# Verify protocol compliance at import time
def _verify_protocol_compliance() -> None:
    """Verify ClaudeSDKAgentAdapter implements AgentPort protocol."""
    adapter_instance: AgentPort = None  # type: ignore[assignment]
    _ = adapter_instance
