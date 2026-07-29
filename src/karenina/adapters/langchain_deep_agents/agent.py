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
import threading
from collections.abc import Callable
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any, TypeVar, cast

from karenina.adapters.agent_runtime import (
    CONTAINER_BACKEND,
    get_agent_runtime_access_mode,
    get_agent_runtime_capabilities,
    get_agent_runtime_option,
    get_container_runtime_config,
    get_deepagents_backend,
)
from karenina.ports import (
    AdapterUnavailableError,
    AgentConfig,
    AgentExecutionError,
    AgentResponseError,
    AgentResult,
    MCPServerConfig,
    Message,
    Tool,
    UsageMetadata,
)
from karenina.ports.capabilities import PortCapabilities
from karenina.utils.retry_policy import RetryPolicy

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

T = TypeVar("T")


class _PartialResultStore:
    """Thread-safe handoff for timeout partial results."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._result: AgentResult | None = None

    def set(self, result: AgentResult) -> None:
        with self._lock:
            self._result = result

    def get(self) -> AgentResult | None:
        with self._lock:
            return self._result


def _run_in_fresh_loop(
    coro_func: Any,
    *args: Any,
    timeout: float = 600,
    timeout_grace: float = 30,
    timeout_result: Callable[[], T | None] | None = None,
) -> Any:
    """Run an async callable in a dedicated thread with a fresh event loop.

    DeepAgents runs a LangGraph async task graph backed by LangChain chat
    models. Some OpenAI-compatible async transports are loop-affine; running
    multiple DeepAgents calls through Karenina's shared BlockingPortal can
    surface httpcore errors such as "Event is bound to a different event loop".
    Keeping each synchronous DeepAgents invocation on its own loop avoids
    sharing loop-bound SDK state across answerer/judge calls.
    """

    def _target() -> Any:
        return asyncio.run(coro_func(*args))

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    shutdown_started = False
    future: concurrent.futures.Future[Any] = executor.submit(_target)
    try:
        # The coroutine enforces the user-facing agent timeout with
        # asyncio.wait_for and can return a partial trace. Give that inner
        # timeout a grace period so this wrapper does not race it and replace
        # the partial result with a bare concurrent.futures.TimeoutError.
        return future.result(timeout=timeout + timeout_grace)
    except concurrent.futures.TimeoutError:
        if timeout_result is not None:
            partial_result = timeout_result()
            if partial_result is not None:
                logger.warning(
                    "DeepAgents sync wrapper timed out after %ss but salvaged a partial result",
                    timeout + timeout_grace,
                )
                future.cancel()
                shutdown_started = True
                executor.shutdown(wait=False, cancel_futures=True)
                return partial_result
        raise
    finally:
        if not shutdown_started:
            executor.shutdown(wait=future.done(), cancel_futures=True)


def _runtime_int_option(model_config: ModelConfig, key: str, default: int) -> int:
    """Read an integer runtime option from ModelConfig.extra_kwargs."""

    raw_value = get_agent_runtime_option(model_config, key, default)
    if isinstance(raw_value, int | float | str):
        return int(raw_value)
    raise TypeError(f"agent_runtime option '{key}' must be an integer-compatible value")


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

    Global concurrency cap: ``max_concurrent_requests`` (GlobalLLMLimiter)
    gates single-turn LLM adapter calls and langchain-based agent model
    calls only. This agent's internal model calls run outside that cap.
    They were never semaphore-gated, so this is the honest current state
    rather than a regression.

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

    @property
    def capabilities(self) -> PortCapabilities:
        """Declare adapter capabilities.

        Returns:
            PortCapabilities implied by the configured DeepAgents backend.
        """
        return get_agent_runtime_capabilities(self._config)

    def _build_backend(self, workspace_path: Any) -> Any:
        """Create the DeepAgents backend configured for this model."""

        backend = get_deepagents_backend(self._config)
        access_mode = get_agent_runtime_access_mode(self._config)
        read_max_bytes = _runtime_int_option(self._config, "read_max_bytes", 0)
        if backend == CONTAINER_BACKEND:
            if workspace_path is None:
                raise AdapterUnavailableError(
                    "agent_runtime backend='container' requires an AgentConfig.workspace_path",
                    reason="missing_workspace",
                )
            from .docker_backend import ContainerSandboxBackend

            container_backend = ContainerSandboxBackend(
                root_dir=workspace_path,
                container_config=get_container_runtime_config(self._config),
                timeout=_runtime_int_option(self._config, "execute_timeout", 120),
                max_output_bytes=_runtime_int_option(self._config, "execute_max_output_bytes", 100_000),
            )
            if access_mode == "read_only":
                from .read_only_backend import ReadOnlyBackend

                return ReadOnlyBackend(container_backend, read_max_bytes=read_max_bytes)
            return container_backend

        if backend == "local_shell":
            if workspace_path is None:
                raise AdapterUnavailableError(
                    "agent_runtime backend='local_shell' requires an AgentConfig.workspace_path",
                    reason="missing_workspace",
                )
            from deepagents.backends import LocalShellBackend

            logger.warning(
                "Using unsafe DeepAgents LocalShellBackend with root_dir=%s",
                workspace_path,
            )
            local_backend = LocalShellBackend(
                root_dir=str(workspace_path),
                virtual_mode=True,
                timeout=_runtime_int_option(self._config, "execute_timeout", 120),
                max_output_bytes=_runtime_int_option(self._config, "execute_max_output_bytes", 100_000),
                inherit_env=True,
            )
            if access_mode == "read_only":
                from .read_only_backend import ReadOnlyBackend

                return ReadOnlyBackend(local_backend, read_max_bytes=read_max_bytes)
            return local_backend

        from deepagents.backends import FilesystemBackend

        if workspace_path:
            logger.info("Using FilesystemBackend with root_dir=%s", workspace_path)
            filesystem_backend = FilesystemBackend(root_dir=str(workspace_path), virtual_mode=True)
            if access_mode == "read_only":
                from .read_only_backend import ReadOnlyBackend

                return ReadOnlyBackend(filesystem_backend, read_max_bytes=read_max_bytes)
            return filesystem_backend

        logger.info("Using FilesystemBackend with default root (cwd)")
        filesystem_backend = FilesystemBackend(virtual_mode=True)
        if access_mode == "read_only":
            from .read_only_backend import ReadOnlyBackend

            return ReadOnlyBackend(filesystem_backend, read_max_bytes=read_max_bytes)
        return filesystem_backend

    def _build_raw_trace(
        self,
        lc_messages: list[Any],
        subgraph_results: dict[tuple[Any, ...], dict[str, Any]],
        *,
        limit_reached: bool,
        timeout_reached: bool,
    ) -> str:
        """Build a raw trace, including any streamed subgraph state."""

        raw_trace = deep_agents_messages_to_raw_trace(lc_messages) if lc_messages else ""
        subgraph_trace_parts = []
        for namespace, subgraph_result in subgraph_results.items():
            subgraph_messages = subgraph_result.get("messages", [])
            subgraph_trace = deep_agents_messages_to_raw_trace(subgraph_messages)
            if subgraph_trace:
                namespace_text = " / ".join(str(part) for part in namespace)
                subgraph_trace_parts.append(f"--- Subgraph Trace: {namespace_text} ---\n{subgraph_trace}")
        if subgraph_trace_parts:
            parts = [part for part in [raw_trace, *subgraph_trace_parts] if part]
            raw_trace = "\n\n".join(parts).strip()
        if limit_reached:
            raw_trace += "\n\n[Note: Recursion limit reached, partial response shown]"
        if timeout_reached:
            raw_trace += "\n\n[Note: Wall-clock timeout reached, partial response shown]"
        return raw_trace.strip()

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
        tools: list[Tool] | None = None,
        mcp_servers: dict[str, MCPServerConfig] | None = None,
        config: AgentConfig | None = None,
        _partial_result_store: _PartialResultStore | None = None,
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

        # Initialize model. Agent loops need SDK-level retries at the model
        # call boundary; retrying the whole agent can repeat workspace writes.
        retry_policy = self._config.retry_policy or RetryPolicy()
        chat_model = create_chat_model(
            self._config,
            max_retries=retry_policy.derive_sdk_max_retries(),
        )

        # Build agent kwargs
        agent_kwargs: dict[str, Any] = {"model": chat_model}
        if system_prompt:
            agent_kwargs["system_prompt"] = system_prompt

        agent_kwargs["backend"] = self._build_backend(config.workspace_path)

        # Pass through extra config to create_deep_agent
        if config.extra:
            for key, value in config.extra.items():
                if key not in ("model", "system_prompt", "backend"):
                    agent_kwargs[key] = value

        # Convert MCP servers to LangChain tools and combine with explicit tools
        all_tools: list[Any] = []
        if tools:
            all_tools.extend(tools)

        # Use AsyncExitStack for persistent MCP sessions. Sessions stay alive
        # for all tool calls during agent execution. Exceptions are captured
        # inside the block and re-raised after clean exit, because MCP session
        # cleanup can wrap errors in ExceptionGroup if an exception propagates
        # through the exit stack.
        result: dict[str, Any] = {}
        subgraph_results: dict[tuple[Any, ...], dict[str, Any]] = {}
        limit_reached = False
        timeout_reached = False
        deferred_error: Exception | None = None

        async with AsyncExitStack() as exit_stack:
            # Convert MCP servers to LangChain tools
            if mcp_servers:
                from .mcp import convert_mcp_to_tools

                try:
                    mcp_tools = await convert_mcp_to_tools(mcp_servers, exit_stack)
                    all_tools.extend(mcp_tools)
                    logger.info(
                        "Loaded %d MCP tools from %d servers",
                        len(mcp_tools),
                        len(mcp_servers),
                    )
                except Exception as e:
                    logger.warning("Failed to load MCP tools: %s", e)
                    deferred_error = AgentExecutionError(f"Failed to initialize MCP tools: {e}")
                    deferred_error.__cause__ = e

            if deferred_error is None:
                # Pass tools to agent if any were collected
                if all_tools:
                    agent_kwargs["tools"] = all_tools

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
                async def execute_agent() -> None:
                    nonlocal result, subgraph_results, limit_reached
                    async for chunk in agent.astream(
                        invoke_input,
                        config=langgraph_config,
                        stream_mode="values",
                        subgraphs=True,
                    ):
                        namespace: tuple[Any, ...] = ()
                        state = chunk
                        if isinstance(chunk, tuple) and len(chunk) == 2:
                            namespace_raw, state = chunk
                            namespace = tuple(namespace_raw)
                        if isinstance(state, dict):
                            if namespace:
                                subgraph_results[namespace] = state
                            else:
                                result = state
                    if result.get("is_last_step", False):
                        limit_reached = True

                try:
                    if config.timeout:
                        await asyncio.wait_for(execute_agent(), timeout=config.timeout)
                    else:
                        await execute_agent()

                except TimeoutError:
                    timeout_reached = True
                    logger.warning("Agent timed out after %ss; returning partial trace", config.timeout)
                    if _partial_result_store is not None:
                        partial_lc_messages = result.get("messages", [])
                        raw_trace = self._build_raw_trace(
                            partial_lc_messages,
                            subgraph_results,
                            limit_reached=limit_reached,
                            timeout_reached=True,
                        )
                        if partial_lc_messages:
                            partial_result = AgentResult(
                                final_response=self._extract_final_response(partial_lc_messages),
                                raw_trace=raw_trace,
                                trace_messages=self._converter.from_provider(partial_lc_messages),
                                usage=extract_deep_agents_usage(partial_lc_messages, model=self._config.model_name),
                                turns=self._count_turns(partial_lc_messages),
                                limit_reached=limit_reached,
                                session_id=None,
                                actual_model=extract_actual_model(partial_lc_messages) or self._config.model_name,
                                timeout_reached=True,
                            )
                        else:
                            partial_result = AgentResult(
                                final_response="[Agent stopped before producing a final response]",
                                raw_trace=raw_trace or "[Note: Agent stopped before producing trace messages]",
                                trace_messages=[],
                                usage=UsageMetadata(model=self._config.model_name),
                                turns=0,
                                limit_reached=limit_reached,
                                session_id=None,
                                actual_model=self._config.model_name,
                                timeout_reached=True,
                            )
                        _partial_result_store.set(partial_result)
                except Exception as e:
                    mapped_error, was_limit = wrap_deep_agents_error(e)
                    if was_limit:
                        limit_reached = True
                        logger.warning("Agent hit turn limit: %s", e)
                    else:
                        deferred_error = mapped_error
                        deferred_error.__cause__ = e

        # exit_stack closed: MCP sessions cleaned up

        if deferred_error is not None:
            raise deferred_error

        # Extract messages from result
        lc_messages: list[Any] = result.get("messages", [])

        if not lc_messages and not limit_reached and not timeout_reached:
            raise AgentResponseError("No messages received from Deep Agents")

        # If limit was reached but no messages, return a partial result
        if not lc_messages and (limit_reached or timeout_reached):
            raw_trace = self._build_raw_trace(
                lc_messages,
                subgraph_results,
                limit_reached=limit_reached,
                timeout_reached=timeout_reached,
            )
            return AgentResult(
                final_response="[Agent stopped before producing a final response]",
                raw_trace=raw_trace or "[Note: Agent stopped before producing trace messages]",
                trace_messages=[],
                usage=UsageMetadata(model=self._config.model_name),
                turns=0,
                limit_reached=limit_reached,
                session_id=None,
                actual_model=self._config.model_name,
                timeout_reached=timeout_reached,
            )

        # Build raw_trace (legacy string format)
        raw_trace = self._build_raw_trace(
            lc_messages,
            subgraph_results,
            limit_reached=limit_reached,
            timeout_reached=timeout_reached,
        )

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

        Raises:
            AgentExecutionError: If the agent fails during execution.
            AgentTimeoutError: If execution exceeds the timeout.
            AgentResponseError: If the response is malformed or invalid.
        """
        timeout = config.timeout if config and config.timeout else 600
        partial_result_store = _PartialResultStore()
        return cast(
            AgentResult,
            _run_in_fresh_loop(
                self.arun,
                messages,
                tools,
                mcp_servers,
                config,
                partial_result_store,
                timeout=timeout,
                timeout_result=partial_result_store.get,
            ),
        )

    async def aclose(self) -> None:
        """Close underlying resources.

        Deep Agents manages its own cleanup via LangGraph's compiled graph,
        so this is a no-op. Provided for interface consistency with other
        adapters that do require cleanup.
        """
