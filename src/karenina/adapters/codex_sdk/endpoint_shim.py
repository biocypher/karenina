"""Local request-rewriting shim for custom OpenAI-compatible endpoints.

The codex CLI (0.132.0) only speaks ``wire_api="responses"`` to custom
providers, and stock vLLM's /v1/responses implementation rejects codex's
request shape in three ways (verified live, see
specs/codex_sdk_findings.md):

1. ``role="developer"`` input messages return 400 "Unexpected message role".
2. Mapping developer to system trips 400 "System message must be at the
   beginning" because codex emits several late system blocks.
3. ``type="reasoning"`` input items, which codex echoes back on follow-up
   turns, fail vLLM validation with 400.

This module runs a localhost HTTP server on an ephemeral port that forwards
requests to the real endpoint after rewriting /responses POST bodies:
developer and system input messages are folded into the top-level
``instructions`` string and reasoning items are stripped from ``input``.
All other requests pass through untouched.

Streaming: codex requests SSE (``stream: true``) and aborts streams that
stay idle past its ``stream_idle_timeout_ms``. The shim therefore relays
upstream response bodies incrementally as chunked transfer encoding,
flushing each piece as it arrives, instead of buffering to EOF.

Lifecycle: the agent adapter starts one shim per arun() call and stops it
in a finally block. A shim is a daemon thread plus a loopback socket, so
per-run startup cost is negligible and there is no shared state to leak
across concurrent runs.
"""

from __future__ import annotations

import http.server
import json
import logging
import threading
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

_HOP_BY_HOP_REQUEST_HEADERS = frozenset({"host", "content-length", "connection"})
_HOP_BY_HOP_RESPONSE_HEADERS = frozenset({"transfer-encoding", "connection", "content-length"})
DEFAULT_UPSTREAM_TIMEOUT_SECONDS = 600.0
_RELAY_CHUNK_SIZE = 8192


def _text_of_message_item(item: dict[str, Any]) -> str:
    """Extract plain text from a Responses API input message item."""
    content = item.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(part.get("text", "") for part in content if isinstance(part, dict) and "text" in part)
    return ""


def rewrite_responses_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a codex Responses API request body for strict endpoints.

    Folds the top-level ``instructions`` plus every developer-role and
    system-role input message into a single ``instructions`` string (in
    original order) and drops those messages from ``input``. Also strips
    ``type="reasoning"`` items, which codex echoes back on follow-up turns
    and strict endpoints cannot validate.

    Args:
        payload: Parsed JSON request body.

    Returns:
        The rewritten payload. The input dict is not mutated.
    """
    result = dict(payload)
    system_parts: list[str] = []
    instructions = result.get("instructions")
    if instructions:
        system_parts.append(str(instructions))

    kept: list[Any] = []
    for item in result.get("input") or []:
        if isinstance(item, dict) and item.get("type") == "message" and item.get("role") in ("developer", "system"):
            system_parts.append(_text_of_message_item(item))
            continue
        if isinstance(item, dict) and item.get("type") == "reasoning":
            continue
        kept.append(item)

    if system_parts:
        result["instructions"] = "\n\n".join(part for part in system_parts if part)
    result["input"] = kept
    return result


class _ShimRequestHandler(http.server.BaseHTTPRequestHandler):
    """Forward one request to the upstream endpoint, rewriting if needed."""

    # HTTP/1.1 is required for the chunked transfer encoding used to relay
    # streaming (SSE) upstream bodies incrementally.
    protocol_version = "HTTP/1.1"

    # Set by EndpointShim when building the server.
    upstream_base_url: str = ""
    upstream_timeout: float = DEFAULT_UPSTREAM_TIMEOUT_SECONDS

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Route handler logging through the module logger at debug level."""
        logger.debug("codex endpoint shim: " + format, *args)

    def _maybe_rewrite(self, body: bytes) -> bytes:
        if "/responses" not in self.path or not body:
            return body
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            return body
        if not isinstance(payload, dict):
            return body
        return json.dumps(rewrite_responses_payload(payload)).encode()

    def _upstream_url(self) -> str:
        # The shim serves under /v1/... and maps /v1/<rest> onto the
        # configured base URL (which itself usually ends in /v1).
        path = self.path
        if path.startswith("/v1/"):
            path = path[len("/v1") :]
        elif path == "/v1":
            path = ""
        return self.upstream_base_url.rstrip("/") + path

    def _send_buffered_response(self, status: int, headers: Any, data: bytes) -> None:
        """Send a fully buffered response with an explicit Content-Length."""
        self.send_response(status)
        for key, value in headers.items():
            if key.lower() not in _HOP_BY_HOP_RESPONSE_HEADERS:
                self.send_header(key, value)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _relay_streaming_response(self, response: Any) -> None:
        """Relay the upstream body incrementally as chunked transfer encoding.

        Codex streams SSE and enforces a stream idle timeout, so the body
        must reach it as upstream produces it. ``read1`` returns whatever
        the current upstream chunk holds without blocking for a full
        buffer, and each piece is flushed immediately in chunked framing.
        """
        self.send_response(response.status)
        for key, value in response.headers.items():
            if key.lower() not in _HOP_BY_HOP_RESPONSE_HEADERS:
                self.send_header(key, value)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        read_available = getattr(response, "read1", response.read)
        while True:
            chunk = read_available(_RELAY_CHUNK_SIZE)
            if not chunk:
                break
            self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
            self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _proxy(self, method: str) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        body = self._maybe_rewrite(body)

        request = urllib.request.Request(  # noqa: S310
            self._upstream_url(),
            data=body if body else None,
            method=method,
        )
        for key, value in self.headers.items():
            if key.lower() not in _HOP_BY_HOP_REQUEST_HEADERS:
                request.add_header(key, value)

        try:
            with urllib.request.urlopen(request, timeout=self.upstream_timeout) as response:  # noqa: S310
                self._relay_streaming_response(response)
        except urllib.error.HTTPError as e:
            data = e.read()
            logger.debug(
                "codex endpoint shim: upstream returned HTTP %s for %s: %s",
                e.code,
                self.path,
                data[:500].decode(errors="replace"),
            )
            self._send_buffered_response(e.code, e.headers, data)
        except OSError as e:
            logger.warning("codex endpoint shim: upstream request failed for %s: %s", self.path, e)
            message = json.dumps({"error": {"message": f"karenina codex shim upstream failure: {e}"}}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(message)))
            self.end_headers()
            self.wfile.write(message)

    def do_GET(self) -> None:  # noqa: N802
        """Proxy a GET request upstream."""
        self._proxy("GET")

    def do_POST(self) -> None:  # noqa: N802
        """Proxy a POST request upstream."""
        self._proxy("POST")


class EndpointShim:
    """Localhost rewriting proxy in front of a custom OpenAI-compatible endpoint.

    Example:
        >>> shim = EndpointShim("http://my-vllm-host:8000/v1")
        >>> shim.start()
        >>> shim.base_url
        'http://127.0.0.1:54321/v1'
        >>> shim.stop()
    """

    def __init__(
        self,
        upstream_base_url: str,
        upstream_timeout: float = DEFAULT_UPSTREAM_TIMEOUT_SECONDS,
    ) -> None:
        """Create a shim for the given upstream base URL.

        Args:
            upstream_base_url: The real endpoint base URL, typically ending
                in /v1 (e.g. ``http://host:8000/v1``).
            upstream_timeout: Per-request socket timeout (seconds) for the
                upstream connection. Applies per socket operation, so a
                streaming response only times out when it stays idle this
                long. The adapter widens this when AgentConfig.timeout is
                larger than the default.
        """
        self._upstream_base_url = upstream_base_url
        self._upstream_timeout = upstream_timeout
        self._server: http.server.ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        """The localhost base URL codex should be pointed at."""
        if self._server is None:
            raise RuntimeError("EndpointShim is not running. Call start() first.")
        port = self._server.server_address[1]
        return f"http://127.0.0.1:{port}/v1"

    def start(self) -> None:
        """Bind an ephemeral loopback port and start serving in a daemon thread."""
        if self._server is not None:
            return
        handler = type(
            "BoundShimRequestHandler",
            (_ShimRequestHandler,),
            {
                "upstream_base_url": self._upstream_base_url,
                "upstream_timeout": self._upstream_timeout,
            },
        )
        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="karenina-codex-endpoint-shim",
            daemon=True,
        )
        self._thread.start()
        logger.debug("codex endpoint shim listening on %s for upstream %s", self.base_url, self._upstream_base_url)

    def stop(self) -> None:
        """Shut down the server and join the serving thread. Safe to call twice."""
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None

    def __enter__(self) -> EndpointShim:
        """Start the shim on context entry."""
        self.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        """Stop the shim on context exit."""
        self.stop()
