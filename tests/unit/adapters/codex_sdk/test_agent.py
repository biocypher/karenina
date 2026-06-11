"""Tests for CodexSDKAgentAdapter using a fake openai_codex module.

The fake module is installed in sys.modules per test, so no codex
app-server subprocess is spawned and no network is touched. The adapter
imports the SDK lazily inside arun(), which makes per-test substitution
via monkeypatch.setitem reliable.
"""

from __future__ import annotations

import asyncio
import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest

from karenina.ports import (
    AgentConfig,
    AgentExecutionError,
    AgentResponseError,
    Message,
)
from karenina.schemas.config import ModelConfig

from .conftest import make_agent_message, make_command_execution, make_reasoning, make_usage

TURN_ID = "turn_1"
THREAD_ID = "thread_1"


def _notification(method: str, payload: Any) -> SimpleNamespace:
    return SimpleNamespace(method=method, payload=payload)


def _item_event(item: Any) -> SimpleNamespace:
    return _notification("item/completed", SimpleNamespace(turn_id=TURN_ID, item=item))


def _usage_event(usage: Any) -> SimpleNamespace:
    return _notification("thread/tokenUsage/updated", SimpleNamespace(turn_id=TURN_ID, token_usage=usage))


def _agent_delta_event(item_id: str, delta: str) -> SimpleNamespace:
    return _notification(
        "item/agentMessage/delta",
        SimpleNamespace(turn_id=TURN_ID, item_id=item_id, delta=delta),
    )


def _completed_event(status: str = "completed", error_message: str | None = None) -> SimpleNamespace:
    error = SimpleNamespace(message=error_message) if error_message else None
    turn = SimpleNamespace(id=TURN_ID, status=SimpleNamespace(value=status), error=error)
    return _notification("turn/completed", SimpleNamespace(turn=turn))


def _successful_events() -> list[SimpleNamespace]:
    return [
        _item_event(make_reasoning(content=["plan the command"])),
        _item_event(make_command_execution(item_id="cmd_1", command="echo hi > hello.txt", output="", exit_code=0)),
        _item_event(make_agent_message("Created hello.txt.", phase="final_answer")),
        _usage_event(make_usage(input_tokens=200, output_tokens=40)),
        _completed_event("completed"),
    ]


class _FakeTurnHandle:
    def __init__(
        self,
        events: list[SimpleNamespace],
        hang: bool,
        stream_error: Exception | None,
        ignore_interrupt: bool = False,
    ) -> None:
        self.id = TURN_ID
        self._events = events
        self._hang = hang
        self._stream_error = stream_error
        self._ignore_interrupt = ignore_interrupt
        self._interrupted = asyncio.Event()
        self.interrupt_called = False

    async def interrupt(self) -> None:
        self.interrupt_called = True
        if not self._ignore_interrupt:
            self._interrupted.set()

    async def stream(self) -> Any:
        for event in self._events:
            yield event
        if self._stream_error is not None:
            raise self._stream_error
        if self._hang:
            # Like the real app-server: an in-flight turn blocks the stream
            # until interrupt, which then emits turn/completed (interrupted).
            await self._interrupted.wait()
            yield _completed_event("interrupted")


class _FakeThread:
    def __init__(self, handle: _FakeTurnHandle, record: dict[str, Any]) -> None:
        self.id = THREAD_ID
        self._handle = handle
        self._record = record

    async def turn(self, prompt: str) -> _FakeTurnHandle:
        self._record["prompt"] = prompt
        return self._handle


def build_fake_codex_module(
    events: list[SimpleNamespace],
    *,
    hang: bool = False,
    stream_error: Exception | None = None,
    ignore_interrupt: bool = False,
    close_hang_s: float = 0.0,
) -> tuple[types.ModuleType, dict[str, Any]]:
    """Build a fake openai_codex module plus a record of observed calls."""
    record: dict[str, Any] = {}
    handle = _FakeTurnHandle(events, hang, stream_error, ignore_interrupt)
    record["handle"] = handle

    class FakeCodexConfig:
        def __init__(self, config_overrides: tuple[str, ...] = (), env: dict[str, str] | None = None) -> None:
            record["config_overrides"] = config_overrides
            record["env"] = env

    class FakeAsyncCodex:
        def __init__(self, config: Any) -> None:
            record["codex_config"] = config

        async def __aenter__(self) -> FakeAsyncCodex:
            return self

        async def __aexit__(self, *_exc: Any) -> None:
            if close_hang_s:
                # Bounded hang so abandoned worker threads still finish
                # before interpreter exit joins them.
                await asyncio.sleep(close_hang_s)
            record["closed"] = True

        async def thread_start(self, **kwargs: Any) -> _FakeThread:
            record["thread_start_kwargs"] = kwargs
            return _FakeThread(handle, record)

    module = types.ModuleType("openai_codex")
    module.AsyncCodex = FakeAsyncCodex
    module.CodexConfig = FakeCodexConfig
    module.ApprovalMode = SimpleNamespace(auto_review="auto_review", deny_all="deny_all")
    module.Sandbox = SimpleNamespace(
        read_only="read_only", workspace_write="workspace_write", full_access="full_access"
    )
    return module, record


@pytest.fixture
def codex_model_config() -> ModelConfig:
    return ModelConfig(
        id="qwen-codex",
        model_name="qwen3.5-122b-a10b",
        interface="codex_sdk",
        endpoint_base_url="http://example-endpoint:8000/v1",
    )


def _make_adapter(model_config: ModelConfig) -> Any:
    from karenina.adapters.codex_sdk.agent import CodexSDKAgentAdapter

    return CodexSDKAgentAdapter(model_config)


class TestSuccessfulRun:
    def test_full_turn_builds_agent_result(
        self, monkeypatch: pytest.MonkeyPatch, codex_model_config: ModelConfig
    ) -> None:
        module, record = build_fake_codex_module(_successful_events())
        monkeypatch.setitem(sys.modules, "openai_codex", module)
        adapter = _make_adapter(codex_model_config)

        result = asyncio.run(adapter.arun([Message.system("Be helpful"), Message.user("Create hello.txt")]))

        assert result.final_response == "Created hello.txt."
        assert "--- Thinking ---" in result.raw_trace
        assert "--- AI Message ---" in result.raw_trace
        assert "--- Tool Message (call_id: cmd_1) ---" in result.raw_trace
        assert result.usage.input_tokens == 200
        assert result.usage.output_tokens == 40
        assert result.session_id == THREAD_ID
        assert result.timeout_reached is False
        assert result.limit_reached is False
        assert result.turns == 2  # shell tool-use turn plus final text turn
        roles = [m.role.value for m in result.trace_messages]
        assert "user" not in roles
        assert record["prompt"] == "Create hello.txt"

    def test_thread_start_receives_expected_kwargs(
        self, monkeypatch: pytest.MonkeyPatch, codex_model_config: ModelConfig
    ) -> None:
        module, record = build_fake_codex_module(_successful_events())
        monkeypatch.setitem(sys.modules, "openai_codex", module)
        adapter = _make_adapter(codex_model_config)

        asyncio.run(adapter.arun([Message.system("Be helpful"), Message.user("hi")]))

        kwargs = record["thread_start_kwargs"]
        assert kwargs["model"] == "qwen3.5-122b-a10b"
        assert kwargs["model_provider"] == "karenina"
        assert kwargs["base_instructions"] == "Be helpful"
        assert kwargs["sandbox"] == "workspace_write"
        assert kwargs["approval_mode"] == "auto_review"
        assert kwargs["cwd"]  # temp workspace fallback was created

    def test_endpoint_overrides_point_at_local_shim(
        self, monkeypatch: pytest.MonkeyPatch, codex_model_config: ModelConfig
    ) -> None:
        module, record = build_fake_codex_module(_successful_events())
        monkeypatch.setitem(sys.modules, "openai_codex", module)
        adapter = _make_adapter(codex_model_config)

        asyncio.run(adapter.arun([Message.user("hi")]))

        overrides = "\n".join(record["config_overrides"])
        assert 'wire_api="responses"' in overrides
        assert "http://127.0.0.1:" in overrides
        # The raw endpoint URL must not be handed to codex directly.
        assert "example-endpoint" not in overrides

    def test_final_message_synthesized_from_deltas(
        self, monkeypatch: pytest.MonkeyPatch, codex_model_config: ModelConfig
    ) -> None:
        # Live-observed behavior: the final agentMessage of a turn never gets
        # item/completed. Its text arrives only via item/agentMessage/delta
        # events, so the adapter must synthesize the assistant message.
        events = [
            _item_event(make_reasoning(content=["simple factual question"])),
            _agent_delta_event("msg_final", "\n\nPar"),
            _agent_delta_event("msg_final", "is"),
            _usage_event(make_usage(input_tokens=50, output_tokens=5)),
            _completed_event("completed"),
        ]
        module, _ = build_fake_codex_module(events)
        monkeypatch.setitem(sys.modules, "openai_codex", module)
        adapter = _make_adapter(codex_model_config)

        result = asyncio.run(adapter.arun([Message.user("Capital of France?")]))

        assert result.final_response == "Paris"
        assert "--- AI Message ---\nParis" in result.raw_trace
        # The synthesized message is the last assistant entry.
        assert result.trace_messages[-1].role.value == "assistant"
        assert result.trace_messages[-1].text == "Paris"

    def test_empty_completed_message_filled_from_deltas(
        self, monkeypatch: pytest.MonkeyPatch, codex_model_config: ModelConfig
    ) -> None:
        # Variant: the agentMessage item DOES complete but with empty text
        # (observed in the step-0 smoke). Delta text fills it in place.
        events = [
            _agent_delta_event("msg_1", "Paris"),
            _item_event(make_agent_message("", item_id="msg_1", phase="final_answer")),
            _completed_event("completed"),
        ]
        module, _ = build_fake_codex_module(events)
        monkeypatch.setitem(sys.modules, "openai_codex", module)
        adapter = _make_adapter(codex_model_config)

        result = asyncio.run(adapter.arun([Message.user("Capital of France?")]))

        assert result.final_response == "Paris"
        # No duplicate synthesized message.
        assistant_texts = [m.text for m in result.trace_messages if m.role.value == "assistant" and m.text]
        assert assistant_texts == ["Paris"]

    def test_sync_run_wrapper(self, monkeypatch: pytest.MonkeyPatch, codex_model_config: ModelConfig) -> None:
        module, _ = build_fake_codex_module(_successful_events())
        monkeypatch.setitem(sys.modules, "openai_codex", module)
        adapter = _make_adapter(codex_model_config)

        result = adapter.run([Message.user("hi")])
        assert result.final_response == "Created hello.txt."


class TestTimeoutSalvage:
    def test_timeout_sets_flag_and_interrupts(
        self, monkeypatch: pytest.MonkeyPatch, codex_model_config: ModelConfig
    ) -> None:
        events = [
            _item_event(make_command_execution(item_id="cmd_1", command="sleep 100", output="", exit_code=None)),
        ]
        module, record = build_fake_codex_module(events, hang=True)
        monkeypatch.setitem(sys.modules, "openai_codex", module)
        adapter = _make_adapter(codex_model_config)

        result = asyncio.run(adapter.arun([Message.user("hi")], config=AgentConfig(timeout=0.3)))

        assert result.timeout_reached is True
        assert record["handle"].interrupt_called is True
        assert "[Note: Agent timed out - partial trace shown]" in result.raw_trace
        assert "--- Tool Message (call_id: cmd_1) ---" in result.raw_trace
        assert result.final_response == "[Agent stopped before producing a final response]"

    def test_timeout_with_no_items_still_returns_partial(
        self, monkeypatch: pytest.MonkeyPatch, codex_model_config: ModelConfig
    ) -> None:
        module, record = build_fake_codex_module([], hang=True)
        monkeypatch.setitem(sys.modules, "openai_codex", module)
        adapter = _make_adapter(codex_model_config)

        result = asyncio.run(adapter.arun([Message.user("hi")], config=AgentConfig(timeout=0.2)))

        assert result.timeout_reached is True
        assert result.trace_messages == []
        assert record["handle"].interrupt_called is True

    def test_unresponsive_interrupt_still_returns_within_grace(
        self, monkeypatch: pytest.MonkeyPatch, codex_model_config: ModelConfig
    ) -> None:
        # Pathological case: the app-server never reacts to the interrupt,
        # so the stream stays open. The adapter must still return a partial
        # result after its bounded grace waits instead of hanging.
        import karenina.adapters.codex_sdk.agent as agent_module

        monkeypatch.setattr(agent_module, "_INTERRUPT_DRAIN_GRACE", 0.2)
        monkeypatch.setattr(agent_module, "_POST_CLOSE_DRAIN_GRACE", 0.2)
        events = [_item_event(make_agent_message("partial"))]
        module, record = build_fake_codex_module(events, hang=True, ignore_interrupt=True)
        monkeypatch.setitem(sys.modules, "openai_codex", module)
        adapter = _make_adapter(codex_model_config)

        result = asyncio.run(adapter.arun([Message.user("hi")], config=AgentConfig(timeout=0.2)))

        assert result.timeout_reached is True
        assert record["handle"].interrupt_called is True
        assert result.final_response == "partial"

    def test_hanging_close_still_returns_partial_from_run(
        self, monkeypatch: pytest.MonkeyPatch, codex_model_config: ModelConfig
    ) -> None:
        # AsyncCodex.close() hangs after the timeout was detected. The
        # provisional partial stored at timeout detection must reach the
        # sync wrapper before its deadline, so run() returns a salvaged
        # result instead of leaking concurrent.futures.TimeoutError.
        import karenina.adapters.codex_sdk.agent as agent_module

        original_run_in_fresh_loop = agent_module._run_in_fresh_loop

        def fast_wrapper(coro_func: Any, *args: Any, **kwargs: Any) -> Any:
            kwargs["timeout_grace"] = 1.0
            return original_run_in_fresh_loop(coro_func, *args, **kwargs)

        monkeypatch.setattr(agent_module, "_run_in_fresh_loop", fast_wrapper)

        events = [_item_event(make_agent_message("salvaged partial"))]
        module, record = build_fake_codex_module(events, hang=True, close_hang_s=5.0)
        monkeypatch.setitem(sys.modules, "openai_codex", module)
        adapter = _make_adapter(codex_model_config)

        result = adapter.run([Message.user("hi")], config=AgentConfig(timeout=0.3))

        assert result.timeout_reached is True
        assert result.final_response == "salvaged partial"
        assert record["handle"].interrupt_called is True


class TestFailures:
    def test_failed_turn_raises_mapped_error(
        self, monkeypatch: pytest.MonkeyPatch, codex_model_config: ModelConfig
    ) -> None:
        events = [_completed_event("failed", error_message="upstream exploded")]
        module, _ = build_fake_codex_module(events)
        monkeypatch.setitem(sys.modules, "openai_codex", module)
        adapter = _make_adapter(codex_model_config)

        with pytest.raises(AgentExecutionError, match="upstream exploded"):
            asyncio.run(adapter.arun([Message.user("hi")]))

    def test_failed_turn_with_limit_message_sets_limit_reached(
        self, monkeypatch: pytest.MonkeyPatch, codex_model_config: ModelConfig
    ) -> None:
        events = [
            _item_event(make_agent_message("partial answer")),
            _completed_event("failed", error_message="model context window exceeded"),
        ]
        module, _ = build_fake_codex_module(events)
        monkeypatch.setitem(sys.modules, "openai_codex", module)
        adapter = _make_adapter(codex_model_config)

        result = asyncio.run(adapter.arun([Message.user("hi")]))
        assert result.limit_reached is True
        assert result.final_response == "partial answer"

    def test_limit_error_with_zero_items_returns_placeholder_partial(
        self, monkeypatch: pytest.MonkeyPatch, codex_model_config: ModelConfig
    ) -> None:
        # A stream error that classifies as a limit before anything was
        # drained returns a placeholder partial (mirrors deep_agents)
        # instead of raising AgentResponseError.
        limit_error = RuntimeError("recursion limit reached for this turn")
        module, _ = build_fake_codex_module([], stream_error=limit_error)
        monkeypatch.setitem(sys.modules, "openai_codex", module)
        adapter = _make_adapter(codex_model_config)

        result = asyncio.run(adapter.arun([Message.user("hi")]))

        assert result.limit_reached is True
        assert result.trace_messages == []
        assert result.final_response == "[Agent stopped before producing a final response]"
        assert "[Note: Turn limit reached - partial response shown]" in result.raw_trace

    def test_stream_error_is_wrapped_and_chained(
        self, monkeypatch: pytest.MonkeyPatch, codex_model_config: ModelConfig
    ) -> None:
        sdk_error = type("ServerBusyError", (Exception,), {})("server overloaded")
        module, _ = build_fake_codex_module([], stream_error=sdk_error)
        monkeypatch.setitem(sys.modules, "openai_codex", module)
        adapter = _make_adapter(codex_model_config)

        with pytest.raises(AgentExecutionError, match="overloaded") as exc_info:
            asyncio.run(adapter.arun([Message.user("hi")]))
        assert exc_info.value.__cause__ is sdk_error

    def test_empty_stream_raises_response_error(
        self, monkeypatch: pytest.MonkeyPatch, codex_model_config: ModelConfig
    ) -> None:
        module, _ = build_fake_codex_module([])
        monkeypatch.setitem(sys.modules, "openai_codex", module)
        adapter = _make_adapter(codex_model_config)

        with pytest.raises(AgentResponseError, match="No turn completion"):
            asyncio.run(adapter.arun([Message.user("hi")]))
