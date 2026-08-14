from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("acp", reason="OMP optional dependency not installed")

import acp
from acp.schema import AgentMessageChunk, PromptResponse, TextContentBlock, Usage

from karenina.adapters.omp.agent import OmpAgentAdapter
from karenina.ports import AgentConfig, AgentExecutionError, AgentResponseError, Message, Tool
from karenina.schemas.config import ModelConfig


class _FakeStream:
    async def read(self) -> bytes:
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
        model_provider="zhipu-coding-plan",
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
    assert result.actual_model == "zhipu-coding-plan/glm-5.3"
    assert result.usage.total_tokens == 24
    assert result.session_id == "omp-session"
    assert result.timeout_reached is False
    assert ("model", "zhipu-coding-plan/glm-5.3") in record["connection"].options
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
