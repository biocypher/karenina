from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("acp", reason="OMP optional dependency not installed")

import acp
from acp.schema import AgentMessageChunk, PromptResponse, TextContentBlock, Usage

from karenina.adapters.omp.agent import OmpAgentAdapter
from karenina.ports import AgentConfig, AgentExecutionError, AgentResponseError, AgentTimeoutError, Message, Tool
from karenina.schemas.config import ModelConfig


class _FakeStream:
    async def read(self, _size: int = -1) -> bytes:
        return b""


class _FakeProcess:
    def __init__(self) -> None:
        self.stdin = object()
        self.stdout = object()
        self.stderr = _FakeStream()
        self.returncode: int | None = None
        self.terminated = False

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.returncode = -9

    async def wait(self) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


class _FakeConnection:
    def __init__(
        self,
        client: Any,
        *,
        hang_until_cancel: bool = False,
        protocol_version: int = acp.PROTOCOL_VERSION,
    ) -> None:
        self.client = client
        self.hang_until_cancel = hang_until_cancel
        self.protocol_version = protocol_version
        self.cancelled = asyncio.Event()
        self.options: list[tuple[str, str]] = []
        self.closed_session: str | None = None

    async def initialize(self, **_kwargs: Any) -> Any:
        return SimpleNamespace(protocol_version=self.protocol_version)

    async def new_session(self, **_kwargs: Any) -> Any:
        return SimpleNamespace(session_id="omp-session")

    async def set_config_option(self, *, config_id: str, value: str, **_kwargs: Any) -> None:
        self.options.append((config_id, value))

    async def prompt(self, **_kwargs: Any) -> PromptResponse:
        await self.client.session_update(
            "omp-session",
            AgentMessageChunk(
                session_update="agent_message_chunk",
                message_id="final",
                content=TextContentBlock(type="text", text="OMP_ADAPTER_OK"),
            ),
        )
        if self.hang_until_cancel:
            await self.cancelled.wait()
            return PromptResponse(stop_reason="cancelled")
        return PromptResponse(
            stop_reason="end_turn",
            usage=Usage(input_tokens=20, output_tokens=4, total_tokens=24),
        )

    async def cancel(self, **_kwargs: Any) -> None:
        self.cancelled.set()

    async def close_session(self, *, session_id: str) -> None:
        self.closed_session = session_id

    async def close(self) -> None:
        return None


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        id="glm",
        model_provider="zai",
        model_name="glm-5.3",
        interface="omp",
        extra_kwargs={"omp_cli_path": "/fake/omp", "thinking": "high"},
    )


def _install_fake_transport(
    monkeypatch: pytest.MonkeyPatch,
    *,
    hang_until_cancel: bool = False,
    protocol_version: int = acp.PROTOCOL_VERSION,
) -> tuple[_FakeProcess, dict[str, Any]]:
    process = _FakeProcess()
    record: dict[str, Any] = {}

    async def create_process(*args: str, **_kwargs: Any) -> _FakeProcess:
        record["command"] = args
        return process

    def connect(client: Any, *_args: Any, **_kwargs: Any) -> _FakeConnection:
        connection = _FakeConnection(
            client,
            hang_until_cancel=hang_until_cancel,
            protocol_version=protocol_version,
        )
        record["connection"] = connection
        return connection

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(acp, "connect_to_agent", connect)
    return process, record


def test_successful_acp_run_maps_model_usage_and_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path, model_config: ModelConfig
) -> None:
    process, record = _install_fake_transport(monkeypatch)
    adapter = OmpAgentAdapter(model_config)

    result = asyncio.run(
        adapter.arun(
            [Message.system("Be concise"), Message.user("Return the marker")],
            config=AgentConfig(workspace_path=tmp_path),
        )
    )

    assert result.final_response == "OMP_ADAPTER_OK"
    assert result.actual_model == "zai/glm-5.3"
    assert result.usage.total_tokens == 24
    assert result.session_id == "omp-session"
    assert result.timeout_reached is False
    assert ("model", "zai/glm-5.3") in record["connection"].options
    assert ("thinking", "high") in record["connection"].options
    assert "--no-skills" in record["command"]
    assert process.terminated is True


def test_timeout_sends_acp_cancel_and_returns_partial(
    monkeypatch: pytest.MonkeyPatch, tmp_path, model_config: ModelConfig
) -> None:
    _, record = _install_fake_transport(monkeypatch, hang_until_cancel=True)
    adapter = OmpAgentAdapter(model_config)

    result = asyncio.run(
        adapter.arun(
            [Message.user("Return the marker")],
            config=AgentConfig(timeout=0.001, workspace_path=tmp_path),
        )
    )

    assert result.timeout_reached is True
    assert result.final_response == "OMP_ADAPTER_OK"
    assert record["connection"].cancelled.is_set()
    assert "timed out" in result.raw_trace


def test_static_tools_are_rejected(model_config: ModelConfig) -> None:
    adapter = OmpAgentAdapter(model_config)
    tool = Tool(name="custom", description="custom", input_schema={"type": "object"})

    with pytest.raises(AgentExecutionError, match="cannot accept Karenina Tool"):
        asyncio.run(adapter.arun([Message.user("hi")], tools=[tool]))


def test_protocol_version_mismatch_fails_before_session_creation(
    monkeypatch: pytest.MonkeyPatch, tmp_path, model_config: ModelConfig
) -> None:
    process, _ = _install_fake_transport(monkeypatch, protocol_version=acp.PROTOCOL_VERSION + 1)
    adapter = OmpAgentAdapter(model_config)

    with pytest.raises(AgentResponseError, match="ACP protocol mismatch"):
        asyncio.run(
            adapter.arun(
                [Message.user("Return the marker")],
                config=AgentConfig(workspace_path=tmp_path),
            )
        )

    assert process.terminated is True


@pytest.mark.unit
class TestOmpLifecycle:
    def test_spawn_failure_is_a_port_error(self, monkeypatch, model_config):
        async def fail(*_args, **_kwargs):
            raise FileNotFoundError("configured OMP executable disappeared")

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fail)
        with pytest.raises(AgentExecutionError, match="subprocess could not start") as error:
            asyncio.run(OmpAgentAdapter(model_config).arun([Message.user("hello")]))
        assert isinstance(error.value.__cause__, FileNotFoundError)

    def test_process_ignoring_terminate_is_killed(self, monkeypatch, model_config):
        monkeypatch.setattr("karenina.adapters.omp.agent._PROCESS_EXIT_GRACE_SECONDS", 0.001)

        async def run():
            killed = asyncio.Event()

            class StubbornProcess(_FakeProcess):
                def terminate(self):
                    self.terminated = True

                def kill(self):
                    super().kill()
                    killed.set()

                async def wait(self):
                    await killed.wait()
                    return self.returncode

            process = StubbornProcess()
            await OmpAgentAdapter(model_config)._stop_process(process)
            assert process.terminated
            assert killed.is_set()
            assert process.returncode == -9

        asyncio.run(run())

    @pytest.mark.parametrize("reason", ["end_turn", "refusal"])
    def test_failed_turn_is_not_a_successful_answer(self, monkeypatch, tmp_path, model_config, reason):
        process, _ = _install_fake_transport(monkeypatch)

        async def failed_prompt(self, **_kwargs):
            await self.client.session_update(
                "omp-session",
                AgentMessageChunk(
                    session_update="agent_message_chunk",
                    content=TextContentBlock(type="text", text="401 token expired or incorrect"),
                ),
            )
            return PromptResponse(stop_reason=reason, usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0))

        monkeypatch.setattr(_FakeConnection, "prompt", failed_prompt)
        with pytest.raises(AgentResponseError, match="401 token expired"):
            asyncio.run(
                OmpAgentAdapter(model_config).arun([Message.user("hello")], config=AgentConfig(workspace_path=tmp_path))
            )
        assert process.terminated

    def test_missing_usage_cannot_mask_provider_failure(self, monkeypatch, tmp_path, model_config):
        _install_fake_transport(monkeypatch)
        original = _FakeConnection.prompt

        async def without_usage(self, **kwargs):
            response = await original(self, **kwargs)
            response.usage = None
            return response

        monkeypatch.setattr(_FakeConnection, "prompt", without_usage)
        with pytest.raises(AgentResponseError, match="without any recorded model usage"):
            asyncio.run(
                OmpAgentAdapter(model_config).arun([Message.user("hello")], config=AgentConfig(workspace_path=tmp_path))
            )

    def test_unadvertised_mcp_transport_fails_before_session_creation(self, monkeypatch, tmp_path, model_config):
        process, record = _install_fake_transport(monkeypatch)
        with pytest.raises(AgentExecutionError, match="did not advertise MCP http"):
            asyncio.run(
                OmpAgentAdapter(model_config).arun(
                    [Message.user("hello")],
                    mcp_servers={"remote": {"type": "http", "url": "https://example.test/mcp"}},
                    config=AgentConfig(workspace_path=tmp_path),
                )
            )
        assert record["connection"].options == []
        assert record["connection"].closed_session is None
        assert process.terminated

    @pytest.mark.parametrize("phase", ["initialize", "new_session", "set_config_option"])
    def test_timeout_includes_session_setup(self, monkeypatch, tmp_path, model_config, phase):
        process, _ = _install_fake_transport(monkeypatch)
        model_config.extra_kwargs["timeout_cancel_grace"] = 0.001

        async def stalled(self, **_kwargs):
            await asyncio.Event().wait()

        monkeypatch.setattr(_FakeConnection, phase, stalled)

        async def run():
            with pytest.raises(AgentTimeoutError, match="before producing a trace"):
                await asyncio.wait_for(
                    OmpAgentAdapter(model_config).arun(
                        [Message.user("hello")], config=AgentConfig(timeout=0.01, workspace_path=tmp_path)
                    ),
                    timeout=1,
                )

        asyncio.run(run())
        assert process.terminated

    def test_stalled_cleanup_does_not_hang_a_successful_run(self, monkeypatch, tmp_path, model_config):
        process, _ = _install_fake_transport(monkeypatch)
        monkeypatch.setattr("karenina.adapters.omp.agent._CLEANUP_GRACE_SECONDS", 0.01)

        async def stalled(self, **_kwargs):
            await asyncio.Event().wait()

        monkeypatch.setattr(_FakeConnection, "close_session", stalled)
        monkeypatch.setattr(_FakeConnection, "close", stalled)

        async def run():
            return await asyncio.wait_for(
                OmpAgentAdapter(model_config).arun(
                    [Message.user("hello")], config=AgentConfig(workspace_path=tmp_path)
                ),
                timeout=1,
            )

        result = asyncio.run(run())
        assert result.final_response == "OMP_ADAPTER_OK"
        assert process.terminated

    def test_stalled_cancel_keeps_partial_result(self, monkeypatch, tmp_path, model_config):
        process, _ = _install_fake_transport(monkeypatch, hang_until_cancel=True)
        monkeypatch.setattr("karenina.adapters.omp.agent._CLEANUP_GRACE_SECONDS", 0.01)
        model_config.extra_kwargs["timeout_cancel_grace"] = 0.001

        async def stalled(self, **_kwargs):
            await asyncio.Event().wait()

        monkeypatch.setattr(_FakeConnection, "cancel", stalled)

        async def run():
            return await asyncio.wait_for(
                OmpAgentAdapter(model_config).arun(
                    [Message.user("hello")], config=AgentConfig(timeout=0.01, workspace_path=tmp_path)
                ),
                timeout=1,
            )

        result = asyncio.run(run())
        assert result.timeout_reached
        assert result.final_response == "OMP_ADAPTER_OK"
        assert process.terminated

    def test_caller_cancellation_cancels_session_and_reaps_process(self, monkeypatch, tmp_path, model_config):
        process, record = _install_fake_transport(monkeypatch, hang_until_cancel=True)

        async def run():
            started = asyncio.Event()

            async def stalled(self, **_kwargs):
                started.set()
                await asyncio.Event().wait()

            monkeypatch.setattr(_FakeConnection, "prompt", stalled)
            task = asyncio.create_task(
                OmpAgentAdapter(model_config).arun([Message.user("hello")], config=AgentConfig(workspace_path=tmp_path))
            )
            await asyncio.wait_for(started.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert record["connection"].cancelled.is_set()
            assert process.terminated
            assert not (asyncio.all_tasks() - {asyncio.current_task()})

        asyncio.run(run())

    def test_stderr_is_drained_with_bounded_memory(self, model_config):
        async def run():
            stream = asyncio.StreamReader()
            stream.feed_data(b"x" * 50_000 + b"diagnostic tail")
            stream.feed_eof()
            tail = bytearray()
            await OmpAgentAdapter(model_config)._collect_stderr(stream, tail)
            return tail

        tail = asyncio.run(run())
        assert len(tail) == 4_000
        assert tail.endswith(b"diagnostic tail")

    @pytest.mark.parametrize("value", [float("inf"), float("nan"), -1])
    def test_invalid_cancellation_grace_fails_before_spawning(self, monkeypatch, model_config, value):
        _, record = _install_fake_transport(monkeypatch)
        model_config.extra_kwargs["timeout_cancel_grace"] = value
        with pytest.raises(AgentExecutionError, match="finite and non-negative"):
            asyncio.run(OmpAgentAdapter(model_config).arun([Message.user("hello")]))
        assert "command" not in record
