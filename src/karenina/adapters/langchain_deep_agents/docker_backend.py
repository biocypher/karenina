"""Container-backed DeepAgents sandbox backend."""

from __future__ import annotations

import subprocess
import uuid
from pathlib import Path
from typing import cast

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol

from karenina.adapters.agent_runtime import (
    SANDBOX_WORKSPACE_PATH,
    ContainerRuntimeConfig,
    build_container_command,
    preflight_container_runtime,
)
from karenina.ports import AdapterUnavailableError


class ContainerSandboxBackend(FilesystemBackend, SandboxBackendProtocol):  # type: ignore[misc]
    """DeepAgents backend that maps a host workspace to `/workspace` in a container."""

    def __init__(
        self,
        *,
        root_dir: str | Path,
        container_config: ContainerRuntimeConfig,
        timeout: int = 120,
        max_output_bytes: int = 100_000,
    ) -> None:
        if not container_config.image:
            raise AdapterUnavailableError(
                "agent_runtime container_image is required when backend='container'",
                reason="missing_deepagents_container_image",
            )
        self._host_workspace = Path(root_dir).resolve()
        if not self._host_workspace.is_dir():
            raise AdapterUnavailableError(
                f"Container sandbox workspace does not exist: {self._host_workspace}",
                reason="missing_workspace",
            )
        preflight_container_runtime(container_config)

        super().__init__(
            root_dir=self._host_workspace,
            virtual_mode=True,
            max_file_size_mb=10,
        )
        self._container_config = container_config
        self._default_timeout = timeout
        self._max_output_bytes = max_output_bytes
        self._sandbox_id = f"{container_config.runtime}-{uuid.uuid4().hex[:8]}"

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

    def _container_command(self, command: str) -> list[str]:
        return build_container_command(
            config=self._container_config,
            host_workspace=self._host_workspace,
            argv=["/bin/sh", "-lc", self._command_for_container(command)],
            env={
                "UV_LINK_MODE": "copy",
                "UV_CACHE_DIR": "/tmp/uv-cache",
                # /opt/renv/bin is appended last so a baked R env (when present in
                # the image) exposes R/Rscript without shadowing the baked system
                # python3 or the agent's own /workspace/.venv. R_LIBS_USER gives a
                # writable target for BiocManager::install at run time. Both are
                # inert for images that lack the baked profile.
                "PATH": "/workspace/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/renv/bin",
                "R_LIBS_USER": "/tmp/rlibs",
            },
        )

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
                self._container_command(command),
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


class DockerSandboxBackend(ContainerSandboxBackend):
    """Compatibility wrapper for callers importing the old Docker backend."""

    def __init__(
        self,
        *,
        root_dir: str | Path,
        image: str,
        network: str = "bridge",
        timeout: int = 120,
        max_output_bytes: int = 100_000,
    ) -> None:
        super().__init__(
            root_dir=root_dir,
            container_config=ContainerRuntimeConfig(
                runtime="docker",
                image=image,
                network=network,
            ),
            timeout=timeout,
            max_output_bytes=max_output_bytes,
        )
