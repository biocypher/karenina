from __future__ import annotations

import pytest

pytest.importorskip("acp", reason="OMP optional dependency not installed")

from acp.schema import AgentCapabilities, McpCapabilities

from karenina.adapters.omp.mcp import convert_mcp_servers, validate_mcp_capabilities
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


def test_convert_stdio_preserves_virtualenv_executable(tmp_path) -> None:
    interpreter = tmp_path / "venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(tmp_path / "base-python")

    [server] = convert_mcp_servers({"local": {"command": str(interpreter)}})

    assert server.command == str(interpreter)


@pytest.mark.unit
class TestNegotiatedMcpCapabilities:
    @pytest.mark.parametrize("transport", ["http", "sse"])
    @pytest.mark.parametrize("capabilities", [None, AgentCapabilities()])
    def test_missing_capability_rejects_optional_transport(self, transport, capabilities):
        with pytest.raises(AgentExecutionError, match=f"did not advertise MCP {transport}"):
            validate_mcp_capabilities({"remote": {"type": transport, "url": "https://example.test"}}, capabilities)

    def test_advertised_transports_are_accepted(self):
        validate_mcp_capabilities(
            {
                "http": {"type": "http", "url": "https://example.test/mcp"},
                "sse": {"type": "sse", "url": "https://example.test/sse"},
            },
            AgentCapabilities(mcp_capabilities=McpCapabilities(http=True, sse=True)),
        )

    def test_stdio_needs_no_optional_capability(self):
        validate_mcp_capabilities({"local": {"command": "python"}}, None)
