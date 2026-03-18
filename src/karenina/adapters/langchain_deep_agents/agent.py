"""LangChain Deep Agents adapter implementing the AgentPort interface.

This module provides the DeepAgentsAgentAdapter class that implements AgentPort
using Deep Agents' create_deep_agent() for agent loops with built-in planning,
context management, and subagent orchestration.

Key differences from the Claude Agent SDK adapter:
- Deep Agents uses LangGraph's compiled graph API (.ainvoke/.invoke)
- Messages are LangGraph BaseMessage instances in the result state
- System prompts go to the create_deep_agent() system_prompt parameter
- Recursion limit controls max agent steps via LangGraph config
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import TYPE_CHECKING, Any

from karenina.ports import (
    AgentConfig,
    AgentResponseError,
    AgentResult,
    AgentTimeoutError,
    MCPServerConfig,
    Message,
    Tool,
    UsageMetadata,
)

from .errors import wrap_deep_agents_error
from .initialization import create_chat_model
from .messages import DeepAgentsMessageConverter
from .trace import deep_agents_messages_to_raw_trace
from .usage import extract_actual_model, extract_deep_agents_usage

if TYPE_CHECKING:
    from karenina.schemas.config import ModelConfig

# Lazy import: resolved at first arun() call. Allows module to load
# without deepagents installed (availability check gates actual use).
_create_deep_agent = None

logger = logging.getLogger(__name__)


class DeepAgentsAgentAdapter:
    """Agent adapter using LangChain Deep Agents' create_deep_agent.

    This adapter implements the AgentPort Protocol for agent execution with
    built-in planning tools, filesystem operations, subagent delegation,
    and context management. Uses create_deep_agent() which returns a
    compiled LangGraph graph.

    The adapter handles:
    - Message conversion from unified Message to prompt string
    - Model initialization via init_chat_model
    - Agent creation and invocation via LangGraph
    - Dual trace output (raw_trace string and trace_messages list)
    - Usage metadata extraction from AIMessage response_metadata
    - Recursion limit detection from LangGraph state

    Example:
        >>> config = ModelConfig(
        ...     id="test",
        ...     model_name="claude-sonnet-4-20250514",
        ...     model_provider="anthropic",
        ...     interface="langchain_deep_agents",
        ... )
        >>> adapter = DeepAgentsAgentAdapter(config)
        >>> result = await adapter.arun(
        ...     messages=[Message.user("What files are in /tmp?")],
        ...     config=AgentConfig(max_turns=10),
        ... )
        >>> print(result.final_response)
    """

    def __init__(self, model_config: ModelConfig) -> None:
        """Initialize the Deep Agents adapter.

        Args:
            model_config: Configuration specifying model, provider, and interface.
        """
        self._config = model_config
        self._converter = DeepAgentsMessageConverter()

    def _extract_final_response(self, lc_messages: list[Any]) -> str:
        """Extract final text response from the last AIMessage.

        Args:
            lc_messages: List of LangGraph BaseMessage objects.

        Returns:
            The final text response string.
        """
        from langchain_core.messages import AIMessage

        for msg in reversed(lc_messages):
            if isinstance(msg, AIMessage):
                if isinstance(msg.content, str) and msg.content:
                    return msg.content
                if isinstance(msg.content, list):
                    text_parts = []
                    for block in msg.content:
                        if isinstance(block, str):
                            text_parts.append(block)
                        elif isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block["text"])
                    if text_parts:
                        return "\n".join(text_parts)

        return "[No final response extracted]"

    def _count_turns(self, lc_messages: list[Any]) -> int:
        """Count the number of agent turns (AIMessage instances).

        Args:
            lc_messages: List of LangGraph BaseMessage objects.

        Returns:
            Number of AIMessage instances in the conversation.
        """
        from langchain_core.messages import AIMessage

        return sum(1 for msg in lc_messages if isinstance(msg, AIMessage))

    async def arun(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,  # noqa: ARG002 - required by AgentPort protocol
        mcp_servers: dict[str, MCPServerConfig] | None = None,  # noqa: ARG002 - required by AgentPort protocol
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
        global _create_deep_agent  # noqa: PLW0603
        if _create_deep_agent is None:
            from deepagents import create_deep_agent as _cda

            _create_deep_agent = _cda

        config = config or AgentConfig()

        # Convert messages to prompt string and extract system prompt
        prompt_string = self._converter.to_prompt_string(messages)
        system_prompt = self._converter.extract_system_prompt(messages)

        # Use config system_prompt as fallback
        if not system_prompt and config.system_prompt:
            system_prompt = config.system_prompt
        elif not system_prompt and self._config.system_prompt:
            system_prompt = self._config.system_prompt

        # Initialize model
        chat_model = create_chat_model(self._config)

        # Build agent kwargs
        agent_kwargs: dict[str, Any] = {"model": chat_model}
        if system_prompt:
            agent_kwargs["system_prompt"] = system_prompt

        # Configure backend for real filesystem access when workspace is available
        workspace_path = config.workspace_path
        if workspace_path:
            from deepagents.backends import FilesystemBackend

            agent_kwargs["backend"] = FilesystemBackend(root_dir=str(workspace_path))
            logger.info("Using FilesystemBackend with root_dir=%s", workspace_path)
        else:
            # Default: use FilesystemBackend rooted at cwd for real filesystem access.
            # StateBackend (virtual/in-memory) is NOT suitable for benchmarking because
            # the agent cannot see real files on disk.
            from deepagents.backends import FilesystemBackend

            agent_kwargs["backend"] = FilesystemBackend()
            logger.info("Using FilesystemBackend with default root (cwd)")

        # Pass through extra config to create_deep_agent
        if config.extra:
            for key, value in config.extra.items():
                if key not in ("model", "system_prompt", "backend"):
                    agent_kwargs[key] = value

        # Create the agent
        agent = _create_deep_agent(**agent_kwargs)

        # Build invocation input
        invoke_input: dict[str, Any] = {
            "messages": [{"role": "user", "content": prompt_string}],
        }

        # LangGraph config for recursion limit
        # Each tool call + response = 2 steps, so double max_turns
        langgraph_config: dict[str, Any] = {
            "recursion_limit": config.max_turns * 2,
        }

        # Execute agent
        result: dict[str, Any] = {}
        limit_reached = False

        async def execute_agent() -> None:
            nonlocal result, limit_reached

            result = await agent.ainvoke(invoke_input, config=langgraph_config)

            # Check if limit was reached via state
            if result.get("is_last_step", False):
                limit_reached = True

        try:
            if config.timeout:
                await asyncio.wait_for(execute_agent(), timeout=config.timeout)
            else:
                await execute_agent()

        except TimeoutError as e:
            raise AgentTimeoutError(f"Agent execution timed out after {config.timeout}s") from e
        except Exception as e:
            mapped_error, was_limit = wrap_deep_agents_error(e)
            if was_limit:
                limit_reached = True
                logger.warning("Agent hit turn limit: %s", e)
            else:
                raise mapped_error from e

        # Extract messages from result
        lc_messages: list[Any] = result.get("messages", [])

        if not lc_messages and not limit_reached:
            raise AgentResponseError("No messages received from Deep Agents")

        # If limit was reached but no messages, return a partial result
        if not lc_messages and limit_reached:
            return AgentResult(
                final_response="[Agent hit recursion limit before producing a response]",
                raw_trace="[Note: Recursion limit reached, no response produced]",
                trace_messages=[],
                usage=UsageMetadata(model=self._config.model_name),
                turns=0,
                limit_reached=True,
                session_id=None,
                actual_model=self._config.model_name,
            )

        # Build raw_trace (legacy string format)
        raw_trace = deep_agents_messages_to_raw_trace(lc_messages)
        if limit_reached:
            raw_trace += "\n\n[Note: Recursion limit reached, partial response shown]"

        # Build trace_messages (structured format)
        trace_messages = self._converter.from_provider(lc_messages)

        # Extract final response
        final_response = self._extract_final_response(lc_messages)

        # Extract usage
        usage = extract_deep_agents_usage(lc_messages, model=self._config.model_name)

        # Count turns
        turns = self._count_turns(lc_messages)

        # Extract actual model
        actual_model = extract_actual_model(lc_messages) or self._config.model_name

        return AgentResult(
            final_response=final_response,
            raw_trace=raw_trace,
            trace_messages=trace_messages,
            usage=usage,
            turns=turns,
            limit_reached=limit_reached,
            session_id=None,
            actual_model=actual_model,
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

        Raises:
            AgentExecutionError: If the agent fails during execution.
            AgentTimeoutError: If execution exceeds the timeout.
            AgentResponseError: If the response is malformed or invalid.
        """
        from karenina.benchmark.verification.executor import get_async_portal

        portal = get_async_portal()

        if portal is not None:
            return portal.call(self.arun, messages, tools, mcp_servers, config)

        # No portal: check if we're in an async context
        try:
            asyncio.get_running_loop()
            # We're in an async context: use ThreadPoolExecutor

            def run_in_thread() -> AgentResult:
                return asyncio.run(self.arun(messages, tools, mcp_servers, config))

            timeout = config.timeout if config and config.timeout else 600
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=timeout)

        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            return asyncio.run(self.arun(messages, tools, mcp_servers, config))

    async def aclose(self) -> None:
        """Close underlying resources.

        Deep Agents manages its own cleanup via LangGraph's compiled graph,
        so this is a no-op. Provided for interface consistency with other
        adapters that do require cleanup.
        """
