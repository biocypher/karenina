#!/usr/bin/env python3
"""Docker wrapper used as Claude Agent SDK ``cli_path``.

The Claude Agent SDK expects ``cli_path`` to be an executable compatible with
the Claude Code CLI. This wrapper preserves that contract while launching the
real ``claude`` executable inside a configured Docker image.
"""

from __future__ import annotations

import os
import socket
import sys
from contextlib import suppress
from urllib.parse import urlparse

SANDBOX_WORKSPACE_PATH = "/workspace"


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        print(f"{name} is required for Claude SDK Docker runtime", file=sys.stderr)
        raise SystemExit(2)
    return value


def _env_flag(name: str, value: str | None = None) -> list[str]:
    if value is None:
        return ["--env", name]
    return ["--env", f"{name}={value}"]


def _forward_env_flags() -> list[str]:
    flags: list[str] = []
    passthrough = [
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_API_KEY",
        "CLAUDE_AGENT_SDK_VERSION",
        "CLAUDE_CODE_ENTRYPOINT",
        "CLAUDE_CODE_ENABLE_SDK_FILE_CHECKPOINTING",
        "CLAUDE_CODE_STREAM_CLOSE_TIMEOUT",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    ]
    for name in passthrough:
        if name in os.environ:
            flags.extend(_env_flag(name))

    flags.extend(
        [
            *_env_flag("HOME", "/tmp"),
            *_env_flag("CLAUDE_CONFIG_DIR", os.environ.get("CLAUDE_CONFIG_DIR", "/tmp/claude-config")),
            *_env_flag("UV_LINK_MODE", "copy"),
            *_env_flag("UV_CACHE_DIR", "/tmp/uv-cache"),
            *_env_flag(
                "PATH",
                "/workspace/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            ),
        ]
    )
    return flags


def _rewrite_workspace_args(args: list[str], host_workspace: str) -> list[str]:
    return [arg.replace(host_workspace, SANDBOX_WORKSPACE_PATH) for arg in args]


def _add_host_flags() -> list[str]:
    flags: list[str] = []
    configured = os.environ.get("KARENINA_CLAUDE_DOCKER_ADD_HOSTS")
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


def build_docker_command(argv: list[str] | None = None) -> list[str]:
    """Build the Docker command for tests and for ``main``."""

    argv = list(sys.argv[1:] if argv is None else argv)
    host_workspace = _required_env("KARENINA_CLAUDE_DOCKER_WORKSPACE")
    image = _required_env("KARENINA_CLAUDE_DOCKER_IMAGE")
    network = os.environ.get("KARENINA_CLAUDE_DOCKER_NETWORK", "bridge")
    if network not in {"bridge", "none"}:
        print("KARENINA_CLAUDE_DOCKER_NETWORK must be 'bridge' or 'none'", file=sys.stderr)
        raise SystemExit(2)

    command = [
        "docker",
        "run",
        "--rm",
        "-i",
        "--network",
        network,
        "--workdir",
        SANDBOX_WORKSPACE_PATH,
        "--volume",
        f"{host_workspace}:{SANDBOX_WORKSPACE_PATH}:rw",
        *_add_host_flags(),
        *_forward_env_flags(),
        "--pids-limit",
        "256",
    ]

    if hasattr(os, "getuid") and hasattr(os, "getgid"):
        command.extend(["--user", f"{os.getuid()}:{os.getgid()}"])

    command.extend([image, "claude", *_rewrite_workspace_args(argv, host_workspace)])
    return command


def main() -> None:
    command = build_docker_command()
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
