"""Managed lifecycle for local MCP server processes."""

from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from karenina.exceptions import McpServerError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class McpServerSpec:
    """Configuration for one managed local MCP server.

    Args:
        name: Human-readable server name used in logs and errors.
        command: Executable and arguments used to start the server.
        host: Host used for the readiness check and returned URL.
        port: TCP port used for the readiness check and returned URL.
        path: HTTP path appended to the returned URL.
        env: Environment additions passed to the child process.
        cwd: Optional child working directory.
        log_path: Optional file receiving combined standard output and error.
        startup_timeout: Seconds to wait for the TCP port to accept connections.
        shutdown_timeout: Seconds to wait after termination before forcing kill.
        poll_interval: Seconds between readiness checks.
    """

    name: str
    command: tuple[str, ...]
    host: str = "127.0.0.1"
    port: int = 8000
    path: str = "/mcp"
    env: Mapping[str, str] = field(default_factory=dict)
    cwd: Path | None = None
    log_path: Path | None = None
    startup_timeout: float = 60.0
    shutdown_timeout: float = 10.0
    poll_interval: float = 0.1

    def __post_init__(self) -> None:
        """Validate values needed for safe process management."""
        if not self.name.strip():
            raise ValueError("MCP server name must not be empty")
        if not self.command or not self.command[0].strip():
            raise ValueError("MCP server command must not be empty")
        if not 1 <= self.port <= 65_535:
            raise ValueError("MCP server port must be between 1 and 65535")
        if self.startup_timeout <= 0 or self.shutdown_timeout <= 0:
            raise ValueError("MCP server timeouts must be positive")
        if self.poll_interval <= 0:
            raise ValueError("MCP server poll interval must be positive")

    @property
    def url(self) -> str:
        """Return the server's streamable HTTP URL."""
        path = self.path if self.path.startswith("/") else f"/{self.path}"
        return f"http://{self.host}:{self.port}{path}"


@dataclass(frozen=True)
class ManagedMcpServer:
    """A running managed MCP server."""

    spec: McpServerSpec
    process: subprocess.Popen[str]

    @property
    def url(self) -> str:
        """Return the server's streamable HTTP URL."""
        return self.spec.url


def _diagnostic_suffix(spec: McpServerSpec) -> str:
    if spec.log_path is None:
        return ""
    return f" See server log: {spec.log_path}"


def _open_log(spec: McpServerSpec) -> TextIO | int:
    if spec.log_path is None:
        return subprocess.DEVNULL
    spec.log_path.parent.mkdir(parents=True, exist_ok=True)
    return spec.log_path.open("a", encoding="utf-8")


def _wait_until_ready(spec: McpServerSpec, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + spec.startup_timeout
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise McpServerError(
                f"MCP server {spec.name!r} exited during startup with code {return_code}.{_diagnostic_suffix(spec)}",
                server_name=spec.name,
                log_path=str(spec.log_path) if spec.log_path else None,
            )
        try:
            with socket.create_connection((spec.host, spec.port), timeout=min(1.0, spec.poll_interval)):
                return
        except OSError:
            time.sleep(spec.poll_interval)
    raise McpServerError(
        f"MCP server {spec.name!r} did not accept connections on "
        f"{spec.host}:{spec.port} within {spec.startup_timeout:g} seconds."
        f"{_diagnostic_suffix(spec)}",
        server_name=spec.name,
        log_path=str(spec.log_path) if spec.log_path else None,
    )


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_group_exit(process_group_id: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _process_group_exists(process_group_id):
            return True
        time.sleep(0.1)
    return not _process_group_exists(process_group_id)


def _stop_process(spec: McpServerSpec, process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError as exc:
        raise McpServerError(
            f"Could not stop MCP server {spec.name!r}: {exc}.{_diagnostic_suffix(spec)}",
            server_name=spec.name,
            log_path=str(spec.log_path) if spec.log_path else None,
        ) from exc

    if process.poll() is None:
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=spec.shutdown_timeout)
    if _wait_for_process_group_exit(process.pid, spec.shutdown_timeout):
        return

    logger.warning("MCP server %s did not stop after termination, forcing kill", spec.name)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError as exc:
        raise McpServerError(
            f"Could not stop MCP server {spec.name!r}: {exc}.{_diagnostic_suffix(spec)}",
            server_name=spec.name,
            log_path=str(spec.log_path) if spec.log_path else None,
        ) from exc
    if process.poll() is None:
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=spec.shutdown_timeout)
    if not _wait_for_process_group_exit(process.pid, spec.shutdown_timeout):
        raise McpServerError(
            f"MCP server {spec.name!r} did not stop after a forced kill.{_diagnostic_suffix(spec)}",
            server_name=spec.name,
            log_path=str(spec.log_path) if spec.log_path else None,
        )


@contextmanager
def managed_mcp_server(spec: McpServerSpec) -> Iterator[ManagedMcpServer]:
    """Start one local MCP server and tear it down on context exit.

    Args:
        spec: Server command, address, timing, and logging configuration.

    Yields:
        The running server and its URL.

    Raises:
        McpServerError: If the process cannot be started, become ready, or stop.
    """
    log_handle = _open_log(spec)
    process: subprocess.Popen[str] | None = None
    try:
        child_env = os.environ.copy()
        child_env.update(spec.env)
        try:
            process = subprocess.Popen(
                list(spec.command),
                cwd=spec.cwd,
                env=child_env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        except OSError as exc:
            raise McpServerError(
                f"Could not start MCP server {spec.name!r}: {exc}.{_diagnostic_suffix(spec)}",
                server_name=spec.name,
                log_path=str(spec.log_path) if spec.log_path else None,
            ) from exc
        try:
            _wait_until_ready(spec, process)
        except Exception:
            _stop_process(spec, process)
            raise
        logger.info("Managed MCP server %s is ready at %s", spec.name, spec.url)
        yield ManagedMcpServer(spec=spec, process=process)
    finally:
        if process is not None:
            _stop_process(spec, process)
        if not isinstance(log_handle, int):
            log_handle.close()


@contextmanager
def managed_mcp_servers(specs: Sequence[McpServerSpec]) -> Iterator[list[ManagedMcpServer]]:
    """Start multiple local MCP servers and tear them down in reverse order.

    Args:
        specs: Server specifications in startup order.

    Yields:
        Running servers in the same order as ``specs``.
    """
    with ExitStack() as stack:
        servers = [stack.enter_context(managed_mcp_server(spec)) for spec in specs]
        yield servers
