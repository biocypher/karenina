from __future__ import annotations

import pytest

pytest.importorskip("acp", reason="OMP optional dependency not installed")

from karenina.adapters.omp.mcp import convert_mcp_servers
from karenina.ports import AgentExecutionError


def test_convert_http_and_sse_servers() -> None:
    servers = convert_mcp_servers(
        {
            "remote": {"type": "http", "url": "https://example.test/mcp", "headers": {"X-Test": "yes"}},
            "legacy": {"type": "sse", "url": "https://example.test/sse"},
        }
    )

    assert [server.name for server in servers] == ["remote", "legacy"]
    assert servers[0].type == "http"
    assert servers[0].headers[0].name == "X-Test"
    assert servers[1].type == "sse"


def test_convert_stdio_resolves_executable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda command: "/usr/bin/tool" if command == "tool" else None)

    [server] = convert_mcp_servers(
        {"local": {"type": "stdio", "command": "tool", "args": ["--serve"], "env": {"A": "1"}}}
    )

    assert server.command == "/usr/bin/tool"
    assert server.args == ["--serve"]
    assert server.env[0].value == "1"


def test_convert_stdio_rejects_missing_executable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _command: None)

    with pytest.raises(AgentExecutionError, match="executable not found"):
        convert_mcp_servers({"local": {"type": "stdio", "command": "missing"}})
