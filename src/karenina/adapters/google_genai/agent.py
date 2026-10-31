"""google_genai agent adapter implementing the AgentPort interface.

A karenina-owned tool loop over the google-genai SDK (design D2). Each turn
is one ``generate_content`` call. The adapter runs function and MCP calls
itself, appends the model's ``Content`` to the history unchanged (thought
signatures must round-trip), and lets Gemini use its server-side tools
(Google Search, URL reading) inside a call. Owning the loop gives per-turn
usage and grounding, a Google Search budget enforced between calls, and a
partial trace on timeout.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from karenina.adapters._mcp_http import connect_all_mcp_servers, get_all_mcp_tools
from karenina.adapters._parallel_base import run_coro_in_thread
from karenina.adapters._timeouts import SYNC_WRAPPER_GRACE_SECONDS, compute_sync_wrapper_timeout
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
from karenina.utils.retry_policy import RetryExecutor, RetryPolicy

from .calls import build_generate_config, classify_finish_reason, finish_error, generate
from .client import GenAIClientHolder, running_loop
from .errors import build_error_registry
from .grounding import grounding_to_trace, resolve_source_urls
from .messages import FunctionCallRef, extract_system_instruction, model_parts_to_message, to_contents
from .settings import BUILTIN_GOOGLE_SEARCH, GoogleGenAISettings, RunOptions
from .tools import Toolset, builtin_tools, mcp_tool_bindings, static_tool_bindings, truncate_tool_result
from .trace import build_raw_trace
from .usage import aggregate_usage, extract_usage, per_call_usage

logger = logging.getLogger(__name__)

NO_FINAL_RESPONSE = "[No final response extracted]"
TURN_LIMIT_NOTE = "Turn limit reached - partial response shown"
TIMEOUT_NOTE = "Agent timed out - partial response shown"

_CORRECTIONS = {
    "MALFORMED_FUNCTION_CALL": (
        "Your previous function call could not be parsed (finish reason MALFORMED_FUNCTION_CALL). "
        "Call the tools again with arguments that match their schemas, or answer directly."
    ),
    "UNEXPECTED_TOOL_CALL": (
        "Your previous turn called a tool that is not available (finish reason UNEXPECTED_TOOL_CALL). "
        "Use only the tools provided, or answer directly."
    ),
}


@dataclass
class _RunState:
    """Mutable state of one agent run, kept outside the loop coroutine.

    The loop updates it as it goes, so the trace, usage and turn count
    survive a wall-clock timeout that cancels the loop.
    """

    contents: list[Any]
    trace: list[Message] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    usage: UsageMetadata = field(default_factory=UsageMetadata)
    turns: int = 0
    search_queries: int = 0
    search_disabled: bool = False
    limit_reached: bool = False
    model_version: str | None = None


class GoogleGenAIAgentAdapter:
    """Agent adapter running a karenina-owned tool loop over google-genai.

    Every model call, agent turns included, holds one ``GlobalLLMLimiter``
    permit (design D13) and goes through ``RetryExecutor``, so a transient
    failure repeats only that call.

    Example:
        >>> config = ModelConfig(
        ...     id="argon-judge",
        ...     model_name="gemini-4-argon",
        ...     interface="google_genai",
        ...     mcp_urls_dict={"epmc": "http://localhost:8765/mcp"},
        ...     extra_kwargs={"builtin_tools": ["google_search"], "max_search_queries": 5},
        ... )
        >>> agent = GoogleGenAIAgentAdapter(config)
        >>> result = await agent.arun(
        ...     [Message.user("Check this citation.")],
        ...     mcp_servers={"epmc": {"type": "http", "url": "http://localhost:8765/mcp"}},
        ... )
    """

    def __init__(self, model_config: ModelConfig) -> None:
        """Initialize the adapter.

        Args:
            model_config: Configuration with ``interface="google_genai"``.

        Raises:
            ValueError: If the configuration is invalid (see
                ``GoogleGenAISettings.from_model_config``).
        """
        self._config = model_config
        self._settings = GoogleGenAISettings.from_model_config(model_config)
        self._client = GenAIClientHolder(self._settings)
        self._retry_executor = RetryExecutor(model_config.retry_policy or RetryPolicy(), build_error_registry())

    @property
    def capabilities(self) -> PortCapabilities:
        """Declare adapter capabilities.

        Gemini's code execution sandbox cannot see karenina's workspace, so
        neither file tools nor code execution are offered (design D12).
        """
        return PortCapabilities(
            supports_system_prompt=True,
            supports_structured_output=True,
            supports_streaming=False,
            supports_file_tools=False,
            supports_code_execution=False,
        )

    async def arun(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        mcp_servers: dict[str, MCPServerConfig] | None = None,
        config: AgentConfig | None = None,
    ) -> AgentResult:
        """Run the agent loop with optional tools and MCP servers.

        Args:
            messages: Initial conversation messages.
            tools: Optional static tool definitions.
            mcp_servers: Optional MCP servers (HTTP/SSE).
            config: Optional run configuration. ``extra`` may override
                ``builtin_tools``, ``max_search_queries`` and
                ``max_tool_result_chars`` for this run.

        Returns:
            AgentResult with the final response, both trace formats, the
            usage summed over every call, and the run flags.

        Raises:
            AgentExecutionError: If the run fails.
            AgentTimeoutError: If the timeout fired before any trace was kept.
            AgentResponseError: If the model returned nothing at all.
        """
        config = config or AgentConfig()
        exit_stack = AsyncExitStack()
        error: Exception | None = None
        result: AgentResult | None = None
        try:
            result = await self._arun(exit_stack, messages, tools, mcp_servers, config)
        except Exception as e:
            # Deferred so the error does not propagate through the MCP
            # transports' task groups (which would wrap it in an ExceptionGroup).
            error = e
        finally:
            await _close_quietly(exit_stack)

        if error is not None:
            if isinstance(error, AgentExecutionError | AgentResponseError):
                raise error
            raise AgentExecutionError(f"Agent execution failed: {type(error).__name__}: {error}") from error
        assert result is not None
        return result

    async def _arun(
        self,
        exit_stack: AsyncExitStack,
        messages: list[Message],
        tools: list[Tool] | None,
        mcp_servers: dict[str, MCPServerConfig] | None,
        config: AgentConfig,
    ) -> AgentResult:
        """Set up tools and contents, run the loop, and build the result."""
        options = self._settings.run_options(config.extra)
        system_instruction = (
            extract_system_instruction(messages) or config.system_prompt or self._settings.system_prompt
        )
        contents = to_contents(messages)
        if not contents:
            raise AgentExecutionError("The google_genai agent needs at least one non-system message")

        state = _RunState(contents=contents)
        timeout_reached = False
        scope = asyncio.timeout(config.timeout)
        try:
            async with scope:
                # MCP setup counts against the run's timeout, so a hung
                # initialize or list_tools cannot outlive it.
                toolset = await self._build_toolset(exit_stack, tools, mcp_servers)
                await self._turn_loop(state, toolset, options, system_instruction, config.max_turns)
        except TimeoutError:
            if not scope.expired():
                raise  # a per-call timeout that outlived its retries
            timeout_reached = True
            logger.warning(
                "google_genai agent timed out after %ss (%d turns, %d trace messages kept)",
                config.timeout,
                state.turns,
                len(state.trace),
            )

        return self._build_result(state, timeout_reached=timeout_reached, timeout=config.timeout)

    async def _build_toolset(
        self,
        exit_stack: AsyncExitStack,
        tools: list[Tool] | None,
        mcp_servers: dict[str, MCPServerConfig] | None,
    ) -> Toolset:
        """Collect static and MCP tools. MCP sessions live on ``exit_stack``."""
        bindings = static_tool_bindings(tools or [])
        if mcp_servers:
            sessions = await connect_all_mcp_servers(exit_stack, self._with_mcp_timeouts(mcp_servers))
            mcp_tools = await get_all_mcp_tools(sessions)
            bindings.extend(
                mcp_tool_bindings(
                    mcp_tools,
                    tool_filter=self._settings.mcp_tool_filter,
                    description_overrides=self._settings.mcp_tool_description_overrides,
                )
            )
        return Toolset(bindings)

    def _with_mcp_timeouts(self, mcp_servers: dict[str, MCPServerConfig]) -> dict[str, MCPServerConfig]:
        """Fill MCP transport timeouts from ModelConfig where a server sets none."""
        result: dict[str, MCPServerConfig] = {}
        for name, server in mcp_servers.items():
            server_dict: dict[str, Any] = dict(server)
            if "url" in server_dict:
                if self._settings.mcp_http_timeout is not None:
                    server_dict.setdefault("timeout", self._settings.mcp_http_timeout)
                if self._settings.mcp_sse_read_timeout is not None:
                    server_dict.setdefault("sse_read_timeout", self._settings.mcp_sse_read_timeout)
            result[name] = server_dict  # type: ignore[assignment]
        return result

    def _builtins_for_turn(self, state: _RunState, options: RunOptions) -> tuple[str, ...]:
        """Return the built-in tools of the next call, applying the search budget.

        The budget is soft (design Q1): Gemini searches inside a call, so the
        budget can only remove Google Search for later calls.
        """
        names = options.builtin_tools
        budget = options.max_search_queries
        if budget is None or BUILTIN_GOOGLE_SEARCH not in names or state.search_queries < budget:
            return names
        if not state.search_disabled:
            state.search_disabled = True
            note = (
                f"Google Search budget of {budget} queries reached after {state.search_queries} queries. "
                "Google Search was removed for the remaining turns"
            )
            state.notes.append(note)
            logger.info("google_genai: %s", note)
        return tuple(n for n in names if n != BUILTIN_GOOGLE_SEARCH)

    async def _turn_loop(
        self,
        state: _RunState,
        toolset: Toolset,
        options: RunOptions,
        system_instruction: str | None,
        max_turns: int,
    ) -> None:
        """Run turns until the model answers, a limit is hit, or it stops."""
        function_tool = toolset.function_declaration_tool()
        while True:
            state.turns += 1
            turn = state.turns
            builtins = self._builtins_for_turn(state, options)
            tool_objects = ([function_tool] if function_tool is not None else []) + builtin_tools(builtins)
            generate_config = build_generate_config(
                self._settings, system_instruction=system_instruction, tools=tool_objects
            )
            outcome = await generate(
                self._client,
                model=self._settings.model_name,
                contents=state.contents,
                config=generate_config,
                stream=self._settings.stream,
                request_timeout=self._settings.request_timeout,
                retry_executor=self._retry_executor,
            )
            if self._settings.resolve_grounding_urls:
                await resolve_source_urls(outcome.grounding)

            queries = outcome.search_query_count
            search_available = BUILTIN_GOOGLE_SEARCH in builtins
            call_usage = extract_usage(
                outcome.usage_metadata,
                model=self._settings.model_name,
                web_search_queries=queries if search_available or queries else None,
            )
            state.usage = aggregate_usage(state.usage, call_usage)
            state.search_queries += queries
            if outcome.model_version:
                state.model_version = outcome.model_version

            # Server-side tool activity happened before the model's text.
            state.trace.extend(grounding_to_trace(outcome.grounding, turn))
            message, calls = model_parts_to_message(
                outcome.parts,
                turn=turn,
                name_for=toolset.original_name,
                usage_metadata=per_call_usage(call_usage),
            )
            if message is not None:
                state.trace.append(message)

            kind = classify_finish_reason(outcome.finish_reason)
            if kind == "blocked":
                raise finish_error(outcome)
            if kind == "tool_limit":
                if not _has_assistant_text(state.trace):
                    raise AgentResponseError(
                        "Gemini stopped with TOO_MANY_TOOL_CALLS (its server-side tool loop ran too long) "
                        "before writing any text"
                    )
                state.limit_reached = True
                state.notes.append(
                    "Gemini stopped with TOO_MANY_TOOL_CALLS (its server-side tool loop ran too long), "
                    "partial response shown"
                )
                return
            if kind == "bad_tool_call":
                # The broken output stays in the trace but not in the history.
                self._add_correction(state, outcome.finish_reason or "")
            else:
                if outcome.content is not None:
                    state.contents.append(outcome.content)
                if kind == "max_tokens":
                    state.notes.append(f"Turn {turn} stopped at MAX_TOKENS: its response may be truncated")
                if not calls:
                    if not outcome.text:
                        # An empty STOP is already retried inside calls.generate,
                        # so this is a MAX_TOKENS turn that wrote nothing.
                        raise AgentResponseError(
                            f"Gemini ended the run with an empty final turn (finish reason "
                            f"{outcome.finish_reason}), so it wrote no answer"
                        )
                    return
                await self._run_function_calls(state, toolset, calls, options)

            if state.turns >= max_turns:
                state.limit_reached = True
                logger.warning("google_genai agent hit the turn limit (%d)", max_turns)
                return

    def _add_correction(self, state: _RunState, finish_reason: str) -> None:
        """Answer a malformed or unexpected tool call with a corrective user note."""
        from google.genai import types

        note = _CORRECTIONS.get(finish_reason, f"Your previous turn ended with {finish_reason}. Try again.")
        state.trace.append(Message.user(note))
        last = state.contents[-1] if state.contents else None
        if last is not None and getattr(last, "role", None) == "user":
            # Keep roles alternating: extend the last user turn (built by the
            # adapter, never a model Content) instead of adding another.
            state.contents[-1] = types.Content(role="user", parts=[*(last.parts or []), types.Part(text=note)])
        else:
            state.contents.append(types.Content(role="user", parts=[types.Part(text=note)]))

    async def _run_function_calls(
        self,
        state: _RunState,
        toolset: Toolset,
        calls: list[FunctionCallRef],
        options: RunOptions,
    ) -> None:
        """Run the turn's function calls and send all results back in one user turn."""
        from google.genai import types

        response_parts: list[Any] = []
        for call in calls:
            text, is_error = await toolset.execute(call.name, call.args)
            text = truncate_tool_result(text, options.max_tool_result_chars)
            state.trace.append(Message.tool_result(tool_use_id=call.tool_use_id, content=text, is_error=is_error))
            response_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        id=call.api_id,
                        name=call.name,
                        response={"error" if is_error else "result": text},
                    )
                )
            )
        state.contents.append(types.Content(role="user", parts=response_parts))

    def _build_result(self, state: _RunState, *, timeout_reached: bool, timeout: float | None) -> AgentResult:
        """Build the AgentResult from the run state."""
        if not state.trace:
            if timeout_reached:
                raise AgentTimeoutError(f"Agent execution timed out after {timeout}s with no messages")
            raise AgentResponseError("Gemini returned no content")

        notes = list(state.notes)
        if state.limit_reached:
            notes.append(TURN_LIMIT_NOTE)
        if timeout_reached:
            notes.append(TIMEOUT_NOTE)

        return AgentResult(
            final_response=_final_response(state.trace),
            raw_trace=build_raw_trace(state.trace, notes),
            trace_messages=state.trace,
            usage=state.usage,
            turns=state.turns,
            limit_reached=state.limit_reached,
            session_id=None,
            actual_model=state.model_version or self._settings.model_name,
            timeout_reached=timeout_reached,
        )

    def run(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        mcp_servers: dict[str, MCPServerConfig] | None = None,
        config: AgentConfig | None = None,
    ) -> AgentResult:
        """Synchronous wrapper for ``arun()``.

        Uses the worker's async portal when one is set, otherwise a fresh
        thread (when called from a running loop) or ``asyncio.run``.

        Args:
            messages: Initial conversation messages.
            tools: Optional static tool definitions.
            mcp_servers: Optional MCP server configurations.
            config: Optional run configuration.

        Returns:
            AgentResult from the run.
        """
        from karenina.benchmark.verification.async_lifecycle import get_async_portal

        portal = get_async_portal()
        if portal is not None:
            return portal.call(self.arun, messages, tools, mcp_servers, config)

        if running_loop() is None:
            return asyncio.run(self.arun(messages, tools, mcp_servers, config))

        effective = config or AgentConfig()
        if effective.timeout:
            thread_timeout = effective.timeout + SYNC_WRAPPER_GRACE_SECONDS
        else:
            thread_timeout = compute_sync_wrapper_timeout(
                self._settings.request_timeout,
                retry_policy=self._config.retry_policy,
                internal_call_sequences=max(effective.max_turns, 1),
            )
        return run_coro_in_thread(self.arun, messages, tools, mcp_servers, config, timeout=thread_timeout)

    async def aclose(self) -> None:
        """Close the google-genai clients. MCP sessions close per run. Safe to call twice."""
        await self._client.aclose()


def _has_assistant_text(trace: list[Message]) -> bool:
    """True when some assistant message in the trace carries text (thoughts excluded)."""
    return any(message.role == Role.ASSISTANT and message.text for message in trace)


def _final_response(trace: list[Message]) -> str:
    """The last assistant text in the trace (thoughts excluded)."""
    for message in reversed(trace):
        if message.role == Role.ASSISTANT and message.text:
            return message.text
    return NO_FINAL_RESPONSE


async def _close_quietly(exit_stack: AsyncExitStack) -> None:
    """Close the run's MCP sessions, logging (not raising) teardown errors."""
    try:
        await exit_stack.aclose()
    except Exception:
        logger.warning("Closing the google_genai MCP sessions failed", exc_info=True)


def _verify_protocol_compliance() -> None:
    """Verify GoogleGenAIAgentAdapter implements the AgentPort protocol."""
    adapter_instance: AgentPort = GoogleGenAIAgentAdapter.__new__(GoogleGenAIAgentAdapter)
    _ = adapter_instance
