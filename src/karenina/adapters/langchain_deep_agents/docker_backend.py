"""Docker-backed DeepAgents sandbox backend."""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import cast

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol

from karenina.ports import AdapterUnavailableError

SANDBOX_WORKSPACE_PATH = "/workspace"


class DockerSandboxBackend(FilesystemBackend, SandboxBackendProtocol):  # type: ignore[misc]
    """DeepAgents backend that maps a host workspace to `/workspace` in Docker."""

    def __init__(
        self,
        *,
        root_dir: str | Path,
        image: str,
        network: str = "bridge",
        timeout: int = 120,
        max_output_bytes: int = 100_000,
    ) -> None:
        if not image:
            raise AdapterUnavailableError(
                "deepagents_docker_image is required when deepagents_backend='docker'",
                reason="missing_deepagents_docker_image",
            )
        if network not in {"bridge", "none"}:
            raise AdapterUnavailableError(
                "deepagents_docker_network must be 'bridge' or 'none'",
                reason="invalid_deepagents_docker_network",
            )
        if shutil.which("docker") is None:
            raise AdapterUnavailableError(
                "Docker is required for deepagents_backend='docker', but the docker command was not found",
                reason="docker_unavailable",
            )

        self._host_workspace = Path(root_dir).resolve()
        if not self._host_workspace.is_dir():
            raise AdapterUnavailableError(
                f"Docker sandbox workspace does not exist: {self._host_workspace}",
                reason="missing_workspace",
            )

        super().__init__(
            root_dir=self._host_workspace,
            virtual_mode=True,
            max_file_size_mb=10,
        )
        self._image = image
        self._network = network
        self._default_timeout = timeout
        self._max_output_bytes = max_output_bytes
        self._sandbox_id = f"docker-{uuid.uuid4().hex[:8]}"

    @property
    def id(self) -> str:
        """Unique sandbox id."""

        return self._sandbox_id

    def _strip_workspace_prefix(self, path: str) -> str:
        """Normalize `/workspace` paths to FilesystemBackend virtual paths."""

        if path == SANDBOX_WORKSPACE_PATH:
            return "/"
        prefix = f"{SANDBOX_WORKSPACE_PATH}/"
        if path.startswith(prefix):
            return "/" + path.removeprefix(prefix)
        return path

    def _resolve_path(self, key: str) -> Path:
        return cast(Path, super()._resolve_path(self._strip_workspace_prefix(key)))

    def _to_virtual_path(self, path: Path) -> str:
        virtual_path = super()._to_virtual_path(path)
        if virtual_path == "/":
            return SANDBOX_WORKSPACE_PATH
        return f"{SANDBOX_WORKSPACE_PATH}{virtual_path}"

    def _command_for_container(self, command: str) -> str:
        """Map any leaked host workspace path to `/workspace`."""

        return command.replace(str(self._host_workspace), SANDBOX_WORKSPACE_PATH)

    def _docker_command(self, command: str) -> list[str]:
        docker_cmd = [
            "docker",
            "run",
            "--rm",
            "--network",
            self._network,
            "--workdir",
            SANDBOX_WORKSPACE_PATH,
            "--volume",
            f"{self._host_workspace}:{SANDBOX_WORKSPACE_PATH}:rw",
            "--env",
            "UV_LINK_MODE=copy",
            "--env",
            "UV_CACHE_DIR=/tmp/uv-cache",
            "--env",
            "PATH=/workspace/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "--pids-limit",
            "256",
        ]

        if hasattr(os, "getuid") and hasattr(os, "getgid"):
            docker_cmd.extend(["--user", f"{os.getuid()}:{os.getgid()}"])

        docker_cmd.extend(
            [
                self._image,
                "/bin/sh",
                "-lc",
                self._command_for_container(command),
            ]
        )
        return docker_cmd

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Execute a shell command inside a one-shot Docker container."""

        if not command or not isinstance(command, str):
            return ExecuteResponse(
                output="Error: Command must be a non-empty string.",
                exit_code=1,
                truncated=False,
            )

        effective_timeout = timeout if timeout is not None else self._default_timeout
        if effective_timeout <= 0:
            return ExecuteResponse(
                output=f"Error: timeout must be positive, got {effective_timeout}",
                exit_code=1,
                truncated=False,
            )

        try:
            result = subprocess.run(  # noqa: S603
                self._docker_command(command),
                check=False,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
            )
        except subprocess.TimeoutExpired:
            return ExecuteResponse(
                output=f"Error: Command timed out after {effective_timeout} seconds.",
                exit_code=124,
                truncated=False,
            )
        except Exception as e:  # noqa: BLE001
            return ExecuteResponse(
                output=f"Error executing Docker command ({type(e).__name__}): {e}",
                exit_code=1,
                truncated=False,
            )

        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            stderr_lines = result.stderr.strip().split("\n")
            output_parts.extend(f"[stderr] {line}" for line in stderr_lines)
        output = "\n".join(output_parts) if output_parts else "<no output>"

        truncated = False
        if len(output) > self._max_output_bytes:
            output = output[: self._max_output_bytes]
            output += f"\n\n... Output truncated at {self._max_output_bytes} bytes."
            truncated = True

        if result.returncode != 0:
            output = f"{output.rstrip()}\n\nExit code: {result.returncode}"

        return ExecuteResponse(
            output=output,
            exit_code=result.returncode,
            truncated=truncated,
        )
