#!/usr/bin/env python3
"""Container wrapper used as Claude Agent SDK ``cli_path``.

The Claude Agent SDK expects ``cli_path`` to be an executable compatible with
the Claude Code CLI. This wrapper preserves that contract while launching
the real ``claude`` executable inside a configured container image.
"""

from __future__ import annotations

import os
import socket
import sys
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlparse

from karenina.adapters.agent_runtime import (
    SANDBOX_WORKSPACE_PATH,
    ContainerRuntimeConfig,
)
from karenina.adapters.agent_runtime import (
    build_container_command as build_runtime_container_command,
)


def _env_with_legacy(preferred: str, legacy: str | None = None, default: str | None = None) -> str | None:
    value = os.environ.get(preferred)
    if value:
        return value
    if legacy:
        value = os.environ.get(legacy)
        if value:
            return value
    return default


def _forward_env_dict() -> dict[str, str | None]:
    passthrough = [
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "API_TIMEOUT_MS",
        "CLAUDE_AGENT_SDK_VERSION",
        "CLAUDE_CODE_ENTRYPOINT",
        "CLAUDE_CODE_ENABLE_SDK_FILE_CHECKPOINTING",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_CODE_STREAM_CLOSE_TIMEOUT",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    ]
    env: dict[str, str | None] = {name: None for name in passthrough if name in os.environ}
    env.update(
        {
            "HOME": "/tmp",
            "CLAUDE_CONFIG_DIR": os.environ.get("CLAUDE_CONFIG_DIR", "/tmp/claude-config"),
            "UV_LINK_MODE": "copy",
            "UV_CACHE_DIR": "/tmp/uv-cache",
            "PATH": "/workspace/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        }
    )
    return env


def _rewrite_workspace_args(args: list[str], host_workspace: str) -> list[str]:
    return [arg.replace(host_workspace, SANDBOX_WORKSPACE_PATH) for arg in args]


def _add_host_flags() -> list[str]:
    flags: list[str] = []
    configured = _env_with_legacy("KARENINA_CLAUDE_CONTAINER_ADD_HOSTS", "KARENINA_CLAUDE_DOCKER_ADD_HOSTS")
    if configured:
        for host_mapping in configured.split(","):
            host_mapping = host_mapping.strip()
            if host_mapping:
                flags.extend(["--add-host", host_mapping])

    endpoint = os.environ.get("ANTHROPIC_BASE_URL")
    if not endpoint:
        return flags
    hostname = urlparse(endpoint).hostname
    if not hostname or hostname in {"localhost", "127.0.0.1", "::1"}:
        return flags
    try:
        socket.inet_aton(hostname)
        return flags
    except OSError:
        pass
    with suppress(OSError):
        flags.extend(["--add-host", f"{hostname}:{socket.gethostbyname(hostname)}"])
    return flags


def _container_config() -> ContainerRuntimeConfig:
    runtime = _env_with_legacy("KARENINA_CLAUDE_CONTAINER_RUNTIME", default="docker") or "docker"
    image = _env_with_legacy("KARENINA_CLAUDE_CONTAINER_IMAGE", "KARENINA_CLAUDE_DOCKER_IMAGE")
    network = _env_with_legacy("KARENINA_CLAUDE_CONTAINER_NETWORK", "KARENINA_CLAUDE_DOCKER_NETWORK", "bridge")
    add_hosts = _env_with_legacy("KARENINA_CLAUDE_CONTAINER_ADD_HOSTS", "KARENINA_CLAUDE_DOCKER_ADD_HOSTS")
    return ContainerRuntimeConfig(
        runtime=runtime,
        image=image,
        network=network or "bridge",
        add_hosts=tuple(host.strip() for host in (add_hosts or "").split(",") if host.strip()),
    )


def build_container_command(argv: list[str] | None = None) -> list[str]:
    """Build the configured container command for tests and for ``main``."""

    argv = list(sys.argv[1:] if argv is None else argv)
    host_workspace = _env_with_legacy("KARENINA_CLAUDE_CONTAINER_WORKSPACE", "KARENINA_CLAUDE_DOCKER_WORKSPACE")
    if not host_workspace:
        print("KARENINA_CLAUDE_CONTAINER_WORKSPACE is required for Claude SDK container runtime", file=sys.stderr)
        raise SystemExit(2)

    config = _container_config()
    if not config.image:
        print("KARENINA_CLAUDE_CONTAINER_IMAGE is required for Claude SDK container runtime", file=sys.stderr)
        raise SystemExit(2)
    if config.runtime == "docker" and config.network not in {"bridge", "none"}:
        print("KARENINA_CLAUDE_CONTAINER_NETWORK must be 'bridge' or 'none' for Docker", file=sys.stderr)
        raise SystemExit(2)
    if config.runtime != "docker" and config.add_hosts:
        print("KARENINA_CLAUDE_CONTAINER_ADD_HOSTS is Docker-only", file=sys.stderr)
        raise SystemExit(2)
    if config.runtime != "docker" and Path(config.image).suffix != ".sif":
        print("KARENINA_CLAUDE_CONTAINER_IMAGE must be a local .sif file for Singularity/Apptainer", file=sys.stderr)
        raise SystemExit(2)

    add_hosts = list(config.add_hosts)
    if config.runtime == "docker":
        auto_hosts = _add_host_flags()
        for index, token in enumerate(auto_hosts):
            if token == "--add-host" and index + 1 < len(auto_hosts):
                add_hosts.append(auto_hosts[index + 1])
        config = ContainerRuntimeConfig(
            runtime=config.runtime,
            image=config.image,
            network=config.network,
            add_hosts=tuple(add_hosts),
        )

    return build_runtime_container_command(
        config=config,
        host_workspace=host_workspace,
        argv=["claude", *_rewrite_workspace_args(argv, host_workspace)],
        env=_forward_env_dict(),
        interactive=True,
    )


def build_docker_command(argv: list[str] | None = None) -> list[str]:
    """Compatibility alias for existing tests and imports."""

    return build_container_command(argv)


def main() -> None:
    command = build_container_command()
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
