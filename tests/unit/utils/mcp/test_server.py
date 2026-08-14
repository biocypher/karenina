"""Unit tests for managed local MCP server processes."""

from __future__ import annotations

import signal
import subprocess
from pathlib import Path

import pytest

from karenina.exceptions import McpServerError
from karenina.utils.mcp import McpServerSpec, managed_mcp_server, managed_mcp_servers


class _Connection:
    def __enter__(self) -> _Connection:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class _Process:
    def __init__(self, *, running: bool = True, pid: int = 1234) -> None:
        self.pid = pid
        self.running = running
        self.wait_calls: list[float | None] = []

    def poll(self) -> int | None:
        return None if self.running else 7

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls.append(timeout)
        self.running = False
        return 0


@pytest.mark.unit
class TestMcpServerSpec:
    def test_builds_url_and_validates_port(self) -> None:
        spec = McpServerSpec(name="otp", command=("otp-mcp",), port=8765, path="mcp")

        assert spec.url == "http://127.0.0.1:8765/mcp"
        with pytest.raises(ValueError, match="port"):
            McpServerSpec(name="otp", command=("otp-mcp",), port=0)


@pytest.mark.unit
class TestManagedMcpServer:
    def test_waits_for_readiness_and_stops_process_group(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        process = _Process()
        popen_kwargs: dict[str, object] = {}
        signals: list[tuple[int, signal.Signals]] = []

        def fake_popen(_command: list[str], **kwargs: object) -> _Process:
            popen_kwargs.update(kwargs)
            return process

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        monkeypatch.setattr("socket.create_connection", lambda *_args, **_kwargs: _Connection())
        monkeypatch.setattr("os.killpg", lambda pid, sig: signals.append((pid, sig)))
        monkeypatch.setattr("karenina.utils.mcp.server._wait_for_process_group_exit", lambda *_args: True)

        spec = McpServerSpec(name="otp", command=("otp-mcp",), port=8765)
        with managed_mcp_server(spec) as server:
            assert server.url == "http://127.0.0.1:8765/mcp"
            assert server.process is process

        assert popen_kwargs["start_new_session"] is True
        assert signals == [(1234, signal.SIGTERM)]
        assert process.wait_calls == [10.0]

    def test_stops_server_when_context_body_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        process = _Process()
        signals: list[signal.Signals] = []
        monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)
        monkeypatch.setattr("socket.create_connection", lambda *_args, **_kwargs: _Connection())
        monkeypatch.setattr("os.killpg", lambda _pid, sig: signals.append(sig))
        monkeypatch.setattr("karenina.utils.mcp.server._wait_for_process_group_exit", lambda *_args: True)

        with (
            pytest.raises(RuntimeError, match="analysis failed"),
            managed_mcp_server(McpServerSpec(name="otp", command=("otp-mcp",))),
        ):
            raise RuntimeError("analysis failed")

        assert signals == [signal.SIGTERM]

    def test_reports_early_exit_with_log_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        process = _Process(running=False)
        monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)
        log_path = tmp_path / "otp.log"
        monkeypatch.setattr("os.killpg", lambda *_args: None)
        monkeypatch.setattr("karenina.utils.mcp.server._wait_for_process_group_exit", lambda *_args: True)

        with (
            pytest.raises(McpServerError, match=str(log_path)),
            managed_mcp_server(McpServerSpec(name="otp", command=("otp-mcp",), log_path=log_path)),
        ):
            pass

    def test_forces_kill_after_termination_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        process = _Process()
        signals: list[signal.Signals] = []
        wait_count = 0

        def fake_wait(timeout: float | None = None) -> int:
            nonlocal wait_count
            wait_count += 1
            if wait_count == 1:
                raise subprocess.TimeoutExpired("otp-mcp", timeout)
            process.running = False
            return 0

        monkeypatch.setattr(process, "wait", fake_wait)
        monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)
        monkeypatch.setattr("socket.create_connection", lambda *_args, **_kwargs: _Connection())
        monkeypatch.setattr("os.killpg", lambda _pid, sig: signals.append(sig))
        group_waits = iter((False, True))
        monkeypatch.setattr(
            "karenina.utils.mcp.server._wait_for_process_group_exit",
            lambda *_args: next(group_waits),
        )

        with managed_mcp_server(McpServerSpec(name="otp", command=("otp-mcp",))):
            pass

        assert signals == [signal.SIGTERM, signal.SIGKILL]

    def test_manages_multiple_servers_in_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        processes = [_Process(pid=1), _Process(pid=2)]
        signals: list[int] = []
        monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: processes.pop(0))
        monkeypatch.setattr("socket.create_connection", lambda *_args, **_kwargs: _Connection())
        monkeypatch.setattr("os.killpg", lambda pid, _sig: signals.append(pid))
        monkeypatch.setattr("karenina.utils.mcp.server._wait_for_process_group_exit", lambda *_args: True)
        specs = [
            McpServerSpec(name="one", command=("otp-mcp",), port=8001),
            McpServerSpec(name="two", command=("otp-mcp",), port=8002),
        ]

        with managed_mcp_servers(specs) as servers:
            assert [server.url for server in servers] == [
                "http://127.0.0.1:8001/mcp",
                "http://127.0.0.1:8002/mcp",
            ]

        assert signals == [2, 1]
