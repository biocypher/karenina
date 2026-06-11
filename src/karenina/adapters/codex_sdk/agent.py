"""Codex SDK agent adapter implementing the AgentPort interface.

Wraps OpenAI's Codex Python SDK (openai-codex). Each arun() call spawns a
fresh ``codex app-server`` subprocess via ``async with AsyncCodex(...)``,
starts one thread, runs one turn, and tears everything down. This mirrors
the per-run subprocess tradeoff of the Claude Agent SDK adapter: higher
startup cost per call, but no cross-run state leakage.

Custom endpoints: when ``ModelConfig.endpoint_base_url`` is set, the
adapter starts a per-run localhost rewriting shim (see endpoint_shim.py)
and points the codex provider config at it, because strict /v1/responses
implementations such as stock vLLM reject codex's raw request shape.

Timeout salvage: the adapter drains ``TurnHandle.stream()`` itself,
accumulating completed items and usage notifications, racing the drain
against ``config.timeout``. On timeout it stores a provisional partial
result, calls ``handle.interrupt()`` (errors suppressed), and returns a
refined partial with ``timeout_reached=True``.

Known gap: codex has no per-turn iteration cap, so
``AgentConfig.max_turns`` is best effort. ``limit_reached`` is set only
when a turn error message indicates a limit.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import logging
import shutil
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, TypeVar, cast

from karenina.adapters.agent_runtime import get_agent_runtime_capabilities
from karenina.ports import (
    AgentConfig,
    AgentResponseError,
    AgentResult,
    MCPServerConfig,
    Message,
    Role,
    Tool,
)
from karenina.ports.capabilities import PortCapabilities

from .config_builder import (
    build_codex_config_overrides,
    build_codex_env,
    build_thread_start_kwargs,
)
from .endpoint_shim import DEFAULT_UPSTREAM_TIMEOUT_SECONDS, EndpointShim
from .errors import wrap_codex_error
from .mcp import convert_mcp_to_codex_config
from .messages import CodexMessageConverter
from .trace import codex_items_to_raw_trace, extract_final_response
from .usage import extract_codex_usage

if TYPE_CHECKING:
    from karenina.schemas.config import ModelConfig

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Placeholder for a turn that completed normally but produced no final
# assistant text, matching the other agent adapters' phrasing.
_NO_FINAL_RESPONSE_PLACEHOLDER = "[No final response extracted]"
# Placeholder for limit or timeout partials where the agent was stopped.
_STOPPED_PLACEHOLDER = "[Agent stopped before producing a final response]"

# Bounded waits for the timeout-salvage path. The interrupt RPC and the
# subsequent turn/completed normally resolve within a couple of seconds.
_INTERRUPT_RPC_TIMEOUT = 10.0
_INTERRUPT_DRAIN_GRACE = 15.0
_POST_CLOSE_DRAIN_GRACE = 5.0


def _merge_agent_message_deltas(items: list[Any], deltas_by_item_id: dict[str, list[str]]) -> list[Any]:
    """Fold streamed agentMessage delta text into the completed item list.

    The codex app-server does not always emit ``item/completed`` for the
    final agentMessage of a turn: verified live against vLLM, the answer
    text arrives only through ``item/agentMessage/delta`` events and
    ``turn/completed`` carries an empty ``turn.items``. The SDK's own
    collector loses that text the same way (``final_response`` comes back
    empty). This helper repairs the item list:

    - A completed agentMessage with empty text whose id received deltas
      gets the joined delta text.
    - Delta ids with no completed item at all are appended as synthesized
      agentMessage stand-ins (duck-typed, matching the converter's
      ``type``/``id``/``text`` access).

    Args:
        items: Items collected from ``item/completed`` notifications.
        deltas_by_item_id: Accumulated delta strings per agentMessage id.

    Returns:
        New list with delta text folded in. Input list is not mutated.
    """
    from .messages import item_type, unwrap_item

    remaining = {item_id: "".join(parts).strip() for item_id, parts in deltas_by_item_id.items()}
    merged: list[Any] = []
    for item in items:
        inner = unwrap_item(item)
        if item_type(item) == "agentMessage":
            item_id = getattr(inner, "id", None)
            if item_id in remaining:
                delta_text = remaining.pop(item_id)
                if not (getattr(inner, "text", "") or "").strip() and delta_text:
                    inner.text = delta_text
        merged.append(item)

    for item_id, delta_text in remaining.items():
        if delta_text:
            merged.append(SimpleNamespace(type="agentMessage", id=item_id, text=delta_text, phase=None))
    return merged


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

    The codex app-server connection is loop-affine (asyncio subprocess
    transport plus reader tasks bound to the loop that spawned them).
    Running each synchronous call on its own fresh loop avoids sharing
    loop-bound state with Karenina's BlockingPortal, matching the
    langchain_deep_agents sync wrapper.
    """

    def _target() -> Any:
        return asyncio.run(coro_func(*args))

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    shutdown_started = False
    future: concurrent.futures.Future[Any] = executor.submit(_target)
    try:
        # The coroutine enforces the user-facing agent timeout with
        # asyncio.wait_for and can return a partial trace. Give that inner
        # timeout a grace period so this wrapper does not race it and
        # replace the partial result with a bare TimeoutError.
        return future.result(timeout=timeout + timeout_grace)
    except concurrent.futures.TimeoutError:
        if timeout_result is not None:
            partial_result = timeout_result()
            if partial_result is not None:
                logger.warning(
                    "Codex sync wrapper timed out after %ss but salvaged a partial result",
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


class CodexSDKAgentAdapter:
    """Agent adapter using the OpenAI Codex Python SDK.

    Implements the AgentPort Protocol. Codex provides its own deep-agent
    runtime: built-in shell (exec_command), apply_patch, planning, and web
    search tools driven by an OS-level sandbox (read_only or
    workspace_write), so explicit Tool definitions are not forwarded.

    Global concurrency cap: ``max_concurrent_requests`` (GlobalLLMLimiter)
    gates single-turn LLM adapter calls and langchain-based agent model
    calls only. This agent's internal model calls run outside that cap.

    Example:
        >>> from karenina.schemas.config import ModelConfig
        >>> config = ModelConfig(
        ...     id="qwen-codex",
        ...     model_name="qwen3.5-122b-a10b",
        ...     model_provider="openai",
        ...     interface="codex_sdk",
        ...     endpoint_base_url="http://my-vllm-host:8000/v1",
        ... )
        >>> adapter = CodexSDKAgentAdapter(config)
        >>> result = adapter.run([Message.user("Create hello.txt and list the directory.")])
        >>> print(result.final_response)
    """

    def __init__(self, model_config: ModelConfig) -> None:
        """Initialize the Codex SDK agent adapter.

        Args:
            model_config: Configuration specifying model, endpoint, and interface.
        """
        self._config = model_config
        self._converter = CodexMessageConverter()

    @property
    def capabilities(self) -> PortCapabilities:
        """Declare adapter capabilities from the registered runtime profile."""
        return get_agent_runtime_capabilities(self._config)

    def _resolve_system_prompt(self, messages: list[Message], config: AgentConfig) -> str | None:
        """Resolve the system prompt from messages, AgentConfig, or ModelConfig."""
        system_prompt = self._converter.extract_system_prompt(messages)
        if not system_prompt and config.system_prompt:
            system_prompt = config.system_prompt
        if not system_prompt and self._config.system_prompt:
            system_prompt = self._config.system_prompt
        return system_prompt

    def _build_agent_result(
        self,
        items: list[Any],
        *,
        usage: Any,
        session_id: str | None,
        limit_reached: bool,
        timeout_reached: bool,
    ) -> AgentResult:
        """Build an AgentResult from completed or partial codex items."""
        raw_trace = codex_items_to_raw_trace(items)
        if limit_reached:
            raw_trace += "\n\n[Note: Turn limit reached - partial response shown]"
        if timeout_reached:
            raw_trace += "\n\n[Note: Agent timed out - partial trace shown]"

        # Exclude user and system messages: system prompts are captured
        # separately via tagged_messages injection.
        trace_messages = [m for m in self._converter.from_provider(items) if m.role not in (Role.USER, Role.SYSTEM)]

        # Codex's own final_response can be empty even on success, so the
        # response text is always reconstructed from items.
        placeholder = _STOPPED_PLACEHOLDER if (limit_reached or timeout_reached) else _NO_FINAL_RESPONSE_PLACEHOLDER
        response_text = extract_final_response(items) or placeholder

        turns = sum(1 for m in trace_messages if m.role == Role.ASSISTANT and (m.text or m.tool_calls))

        return AgentResult(
            final_response=response_text,
            raw_trace=raw_trace.strip(),
            trace_messages=trace_messages,
            usage=extract_codex_usage(usage, model=self._config.model_name),
            turns=turns,
            limit_reached=limit_reached,
            session_id=session_id,
            actual_model=self._config.model_name,
            timeout_reached=timeout_reached,
        )

    async def arun(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        mcp_servers: dict[str, MCPServerConfig] | None = None,
        config: AgentConfig | None = None,
        _partial_result_store: _PartialResultStore | None = None,
    ) -> AgentResult:
        """Execute one codex turn for the given conversation.

        Args:
            messages: Initial conversation messages.
            tools: Ignored. Codex ships its own built-in tool set and does
                not accept ad hoc tool definitions per turn.
            mcp_servers: Not yet supported. Configured servers are skipped
                with a warning (see mcp.py).
            config: Optional AgentConfig for execution parameters.

        Returns:
            AgentResult with final response, traces, usage, and metadata.

        Raises:
            AdapterUnavailableError: If the codex binary cannot be launched.
            AgentExecutionError: If the agent fails during execution.
            AgentTimeoutError: If execution exceeds the timeout with no
                partial items to salvage.
            AgentResponseError: If the response is malformed or invalid.
        """
        from openai_codex import ApprovalMode, AsyncCodex, CodexConfig, Sandbox

        config = config or AgentConfig()

        if tools:
            logger.warning(
                "codex_sdk adapter ignores %d explicit tool definition(s). Codex uses its built-in tool set",
                len(tools),
            )
        convert_mcp_to_codex_config(mcp_servers)

        prompt_string = self._converter.to_prompt_string(messages)
        system_prompt = self._resolve_system_prompt(messages, config)

        # Workspace: codex workspace_write sandboxing restricts writes to
        # cwd plus its configured writable roots, so always pin cwd. Fall
        # back to a fresh temp dir, never to the host cwd or full access.
        workspace_path = config.workspace_path
        temp_workspace: str | None = None
        if workspace_path is None:
            temp_workspace = await asyncio.to_thread(tempfile.mkdtemp, prefix="karenina-codex-")
            workspace_path = Path(temp_workspace)

        shim: EndpointShim | None = None
        items: list[Any] = []
        agent_message_deltas: dict[str, list[str]] = {}
        usage: Any = None
        completed_turn: Any = None
        thread_id: str | None = None
        limit_reached = False
        timeout_reached = False

        try:
            base_url: str | None = None
            if self._config.endpoint_base_url:
                # Per-arun shim: strict /v1/responses endpoints (stock vLLM)
                # reject codex's developer-role messages and echoed reasoning
                # items, so route requests through the local rewriter. Widen
                # the upstream socket timeout when the agent timeout exceeds
                # the shim default.
                upstream_timeout = max(DEFAULT_UPSTREAM_TIMEOUT_SECONDS, config.timeout or 0.0)
                shim = EndpointShim(self._config.endpoint_base_url, upstream_timeout=upstream_timeout)
                await asyncio.to_thread(shim.start)
                base_url = shim.base_url

            codex_config = CodexConfig(
                config_overrides=build_codex_config_overrides(self._config, base_url),
                env=build_codex_env(self._config),
            )
            thread_kwargs = build_thread_start_kwargs(
                self._config,
                base_instructions=system_prompt,
                cwd=str(workspace_path),
            )
            sandbox_mode = thread_kwargs.pop("sandbox")
            sandbox = Sandbox.read_only if sandbox_mode == "read_only" else Sandbox.workspace_write

            drain_task: asyncio.Task[None] | None = None
            try:
                async with AsyncCodex(codex_config) as codex:
                    thread = await codex.thread_start(
                        sandbox=sandbox,
                        # auto_review keeps escalations non-interactive: the
                        # SDK's default approval handler auto-accepts.
                        approval_mode=ApprovalMode.auto_review,
                        **thread_kwargs,
                    )
                    thread_id = thread.id
                    handle = await thread.turn(prompt_string)

                    async def drain_stream() -> None:
                        nonlocal usage, completed_turn
                        # Drain the single-consume notification stream
                        # ourselves and never call handle.run() afterwards.
                        async for event in handle.stream():
                            method = getattr(event, "method", "")
                            payload = getattr(event, "payload", None)
                            if payload is None:
                                continue
                            if getattr(payload, "turn_id", handle.id) != handle.id:
                                continue
                            if method == "item/completed":
                                items.append(payload.item)
                            elif method == "item/agentMessage/delta":
                                # The final agentMessage of a turn often never
                                # gets an item/completed (verified live: only
                                # deltas, then turn/completed with empty
                                # turn.items), so the answer text must be
                                # accumulated from deltas and synthesized.
                                delta = getattr(payload, "delta", None)
                                delta_item_id = getattr(payload, "item_id", None)
                                if delta and delta_item_id:
                                    agent_message_deltas.setdefault(delta_item_id, []).append(delta)
                            elif method == "thread/tokenUsage/updated":
                                usage = payload.token_usage
                            elif method == "turn/completed":
                                turn = getattr(payload, "turn", None)
                                if turn is not None and getattr(turn, "id", None) == handle.id:
                                    completed_turn = turn

                    # Timeout handling deliberately avoids cancelling the
                    # drain task. The SDK reads notifications via a blocking
                    # queue.get on an executor thread. Cancelling mid-get
                    # unregisters the queue and strands that thread forever,
                    # which then hangs event-loop shutdown. Instead: race the
                    # drain with asyncio.wait, interrupt the turn on timeout,
                    # and let the stream end naturally on the interrupt's
                    # turn/completed. AsyncCodex.close() at context exit is
                    # the fail-safe waker (fail_all on registered queues).
                    drain_task = asyncio.create_task(drain_stream())
                    if config.timeout:
                        done, _ = await asyncio.wait({drain_task}, timeout=config.timeout)
                        if not done:
                            timeout_reached = True
                            logger.warning(
                                "Codex agent timed out after %ss with %d partial item(s). Interrupting turn",
                                config.timeout,
                                len(items),
                            )
                            # Store a provisional partial immediately. The
                            # remaining salvage steps (interrupt RPC, drain
                            # grace, AsyncCodex.close) can outlast the sync
                            # wrapper's deadline against an unresponsive
                            # app-server, and the wrapper falls back to
                            # whatever the store holds at that moment.
                            if _partial_result_store is not None:
                                _partial_result_store.set(
                                    self._build_agent_result(
                                        _merge_agent_message_deltas(list(items), agent_message_deltas),
                                        usage=usage,
                                        session_id=thread_id,
                                        limit_reached=limit_reached,
                                        timeout_reached=True,
                                    )
                                )
                            with contextlib.suppress(Exception):
                                await asyncio.wait_for(handle.interrupt(), timeout=_INTERRUPT_RPC_TIMEOUT)
                            # Wait for the interrupted turn to emit its final
                            # turn/completed so the stream finishes cleanly.
                            await asyncio.wait({drain_task}, timeout=_INTERRUPT_DRAIN_GRACE)
                        else:
                            await drain_task
                    else:
                        await drain_task
            except Exception as e:
                mapped_error, was_limit = wrap_codex_error(e)
                if was_limit:
                    limit_reached = True
                    logger.warning("Codex agent hit a limit: %s", e)
                else:
                    raise mapped_error from e
            finally:
                # The context exit above closed the app-server, which wakes
                # any still-blocked notification read via fail_all. Give the
                # drain task a short window to observe that, then detach.
                if drain_task is not None and not drain_task.done():
                    await asyncio.wait({drain_task}, timeout=_POST_CLOSE_DRAIN_GRACE)
                    if not drain_task.done():
                        drain_task.cancel()
                if drain_task is not None and drain_task.done() and not drain_task.cancelled():
                    # Retrieve to silence "exception was never retrieved" on
                    # the expected TransportClosedError after a forced close.
                    with contextlib.suppress(Exception):
                        drain_task.exception()

            items = _merge_agent_message_deltas(items, agent_message_deltas)

            if timeout_reached:
                partial_result = self._build_agent_result(
                    items,
                    usage=usage,
                    session_id=thread_id,
                    limit_reached=limit_reached,
                    timeout_reached=True,
                )
                if _partial_result_store is not None:
                    _partial_result_store.set(partial_result)
                return partial_result

            if completed_turn is not None:
                status = getattr(getattr(completed_turn, "status", None), "value", None)
                turn_error = getattr(completed_turn, "error", None)
                if status == "failed":
                    error_message = getattr(turn_error, "message", None) or "turn failed"
                    mapped_error, was_limit = wrap_codex_error(RuntimeError(error_message))
                    if was_limit:
                        limit_reached = True
                        logger.warning("Codex turn failed at a limit: %s", error_message)
                    else:
                        raise mapped_error

            # A limit partial with zero items still returns a placeholder
            # result (mirrors the deep_agents adapter) rather than raising.
            if completed_turn is None and not items and not limit_reached:
                raise AgentResponseError("No turn completion or items received from Codex SDK agent")

            return self._build_agent_result(
                items,
                usage=usage,
                session_id=thread_id,
                limit_reached=limit_reached,
                timeout_reached=False,
            )
        finally:
            if shim is not None:
                await asyncio.to_thread(shim.stop)
            if temp_workspace is not None:
                await asyncio.to_thread(shutil.rmtree, temp_workspace, ignore_errors=True)

    def run(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        mcp_servers: dict[str, MCPServerConfig] | None = None,
        config: AgentConfig | None = None,
    ) -> AgentResult:
        """Synchronous wrapper for arun().

        Runs the coroutine on a dedicated thread with a fresh event loop
        and salvages timeout partial results via a thread-safe store, the
        same pattern as the langchain_deep_agents adapter.

        Args:
            messages: Initial conversation messages.
            tools: Ignored (codex uses its built-in tool set).
            mcp_servers: Not yet supported, skipped with a warning.
            config: Optional AgentConfig for execution parameters.

        Returns:
            AgentResult from the agent execution.

        Raises:
            AdapterUnavailableError: If the codex binary cannot be launched.
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

        Each arun() owns its AsyncCodex context and endpoint shim and tears
        them down before returning, so this is a no-op. Provided for
        interface consistency with other adapters.
        """
