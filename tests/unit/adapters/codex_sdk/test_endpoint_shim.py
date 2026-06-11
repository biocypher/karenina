"""Tests for the request-rewriting endpoint shim.

The rewrite cases mirror the recorded request shapes from the step-0
validation (specs/codex_sdk_findings.md): codex sends its system text as
``instructions`` plus developer-role input messages, and echoes prior
``reasoning`` items back into ``input`` on follow-up turns. Stock vLLM
rejects both.
"""

from __future__ import annotations

import http.server
import json
import threading
import time
import urllib.request
from typing import Any

import pytest

from karenina.adapters.codex_sdk.endpoint_shim import EndpointShim, rewrite_responses_payload


def _recorded_codex_request() -> dict[str, Any]:
    """First-turn request shape as recorded through the reference proxy."""
    return {
        "model": "qwen3.5-122b-a10b",
        "instructions": "You are a helpful coding agent.",
        "input": [
            {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": "Permissions: workspace-write."}],
            },
            {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": "Apps and skills catalog."}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Create hello.txt"}],
            },
        ],
        "tools": [{"type": "function", "name": "exec_command"}],
        "stream": True,
    }


def _recorded_followup_request() -> dict[str, Any]:
    """Follow-up turn: echoed reasoning plus function call round trip."""
    return {
        "model": "qwen3.5-122b-a10b",
        "instructions": "You are a helpful coding agent.",
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Create hello.txt"}]},
            {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "I should run a command."}]},
            {"type": "function_call", "name": "exec_command", "call_id": "call_1", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "exit 0"},
            {"type": "message", "role": "system", "content": [{"type": "input_text", "text": "Late system note."}]},
        ],
    }


class TestRewriteResponsesPayload:
    def test_developer_messages_folded_into_instructions(self) -> None:
        rewritten = rewrite_responses_payload(_recorded_codex_request())
        assert rewritten["instructions"] == (
            "You are a helpful coding agent.\n\nPermissions: workspace-write.\n\nApps and skills catalog."
        )
        roles = [item.get("role") for item in rewritten["input"] if item.get("type") == "message"]
        assert roles == ["user"]

    def test_reasoning_items_stripped(self) -> None:
        rewritten = rewrite_responses_payload(_recorded_followup_request())
        types = [item["type"] for item in rewritten["input"]]
        assert "reasoning" not in types
        # Function call round trips survive untouched.
        assert "function_call" in types
        assert "function_call_output" in types

    def test_system_messages_folded(self) -> None:
        rewritten = rewrite_responses_payload(_recorded_followup_request())
        assert rewritten["instructions"].endswith("Late system note.")
        assert all(item.get("role") != "system" for item in rewritten["input"])

    def test_original_payload_not_mutated(self) -> None:
        payload = _recorded_codex_request()
        rewrite_responses_payload(payload)
        assert len(payload["input"]) == 3
        assert payload["instructions"] == "You are a helpful coding agent."

    def test_payload_without_system_text_passes_through(self) -> None:
        payload = {"model": "m", "input": [{"type": "message", "role": "user", "content": "hi"}]}
        rewritten = rewrite_responses_payload(payload)
        assert rewritten["input"] == payload["input"]
        assert "instructions" not in rewritten


class _RecordingUpstreamHandler(http.server.BaseHTTPRequestHandler):
    """Loopback upstream that records the request it receives."""

    recorded: dict[str, Any] = {}

    def log_message(self, *args: Any) -> None:
        pass

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        type(self).recorded = {"path": self.path, "body": json.loads(body)}
        response = json.dumps({"ok": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)


@pytest.fixture
def upstream_server() -> Any:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RecordingUpstreamHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


class TestEndpointShimRoundTrip:
    def test_shim_rewrites_and_forwards_to_upstream(self, upstream_server: Any) -> None:
        port = upstream_server.server_address[1]
        shim = EndpointShim(f"http://127.0.0.1:{port}/v1")
        shim.start()
        try:
            request = urllib.request.Request(
                shim.base_url + "/responses",
                data=json.dumps(_recorded_codex_request()).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                assert response.status == 200
        finally:
            shim.stop()

        recorded = _RecordingUpstreamHandler.recorded
        assert recorded["path"] == "/v1/responses"
        body = recorded["body"]
        assert "Permissions: workspace-write." in body["instructions"]
        assert all(item.get("role") not in ("developer", "system") for item in body["input"])

    def test_base_url_requires_start(self) -> None:
        shim = EndpointShim("http://127.0.0.1:1/v1")
        with pytest.raises(RuntimeError, match="not running"):
            _ = shim.base_url

    def test_stop_is_idempotent(self) -> None:
        shim = EndpointShim("http://127.0.0.1:1/v1")
        shim.start()
        shim.stop()
        shim.stop()

    def test_upstream_timeout_configurable(self) -> None:
        shim = EndpointShim("http://127.0.0.1:1/v1", upstream_timeout=1234.5)
        assert shim._upstream_timeout == 1234.5


class _StreamingUpstreamHandler(http.server.BaseHTTPRequestHandler):
    """Loopback upstream that streams two SSE chunks with a delay between."""

    protocol_version = "HTTP/1.1"
    second_chunk_written = threading.Event()
    first_chunk = b"data: first\n\n"
    second_chunk = b"data: second\n\n"

    def log_message(self, *args: Any) -> None:
        pass

    def _write_chunk(self, data: bytes) -> None:
        self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        self._write_chunk(self.first_chunk)
        time.sleep(0.5)
        self._write_chunk(self.second_chunk)
        type(self).second_chunk_written.set()
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


class TestEndpointShimStreaming:
    def test_sse_body_is_relayed_incrementally(self) -> None:
        """The first SSE chunk must reach the client before upstream finishes.

        Codex aborts streams that stay idle past its stream idle timeout,
        so the shim cannot buffer the upstream body to EOF.
        """
        _StreamingUpstreamHandler.second_chunk_written.clear()
        upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _StreamingUpstreamHandler)
        upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        upstream_thread.start()
        shim = EndpointShim(f"http://127.0.0.1:{upstream.server_address[1]}/v1")
        shim.start()
        try:
            request = urllib.request.Request(
                shim.base_url + "/stream-test",
                data=b"{}",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                first = response.read(len(_StreamingUpstreamHandler.first_chunk))
                # The first chunk arrived while upstream was still sleeping
                # before its second chunk, proving incremental relay.
                assert first == _StreamingUpstreamHandler.first_chunk
                assert not _StreamingUpstreamHandler.second_chunk_written.is_set()
                rest = response.read()
            assert _StreamingUpstreamHandler.second_chunk in rest
        finally:
            shim.stop()
            upstream.shutdown()
            upstream.server_close()
