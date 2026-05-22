"""Shared runtime helpers for agent adapters."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from karenina.ports.capabilities import PortCapabilities

if TYPE_CHECKING:
    from karenina.schemas.config import ModelConfig


SANDBOX_WORKSPACE_PATH = "/workspace"
AGENT_RUNTIME_EXTRA_KEY = "agent_runtime"
AGENT_RUNTIME_ACCESS_MODES = {"read_write", "read_only"}
CONTAINER_BACKEND = "container"
LEGACY_DOCKER_BACKEND = "docker"
CONTAINER_BACKEND_ALIASES = {CONTAINER_BACKEND, LEGACY_DOCKER_BACKEND}
CONTAINER_RUNTIMES = {"docker", "singularity", "apptainer"}
CLAUDE_SDK_BACKENDS = {"native", CONTAINER_BACKEND, LEGACY_DOCKER_BACKEND}
DEEPAGENTS_BACKENDS = {"filesystem", CONTAINER_BACKEND, LEGACY_DOCKER_BACKEND, "local_shell"}
DOCKER_PREFLIGHT_TIMEOUT_SECONDS = 10


@dataclass(frozen=True)
class AgentRuntimeProfile:
    """Adapter-specific runtime behavior used by shared pipeline prompts."""

    capabilities: Callable[[ModelConfig], PortCapabilities]
    workspace_path_for_prompt: Callable[[ModelConfig, Path | None], str | None] | None = None
    map_path_for_prompt: Callable[[ModelConfig, Path | None, Path | None], str | None] | None = None


@dataclass(frozen=True)
class ContainerRuntimeConfig:
    """Resolved configuration for one-shot container command execution."""

    runtime: str
    image: str | None
    network: str = "bridge"
    add_hosts: tuple[str, ...] = ()


_runtime_profiles: dict[str, AgentRuntimeProfile] = {}


def register_agent_runtime_profile(
    interface: str,
    profile: AgentRuntimeProfile,
    *,
    force: bool = False,
) -> AgentRuntimeProfile:
    """Register runtime behavior for an adapter interface.

    External adapters can call this during their own registration to customize
    prompt-visible paths and capability reporting without editing built-in code.
    """

    if interface in _runtime_profiles and not force:
        raise ValueError(
            f"Runtime profile for interface '{interface}' is already registered. "
            "Use force=True to intentionally overwrite it."
        )
    _runtime_profiles[interface] = profile
    return profile


def get_deepagents_backend(model_config: ModelConfig) -> str:
    """Return the configured DeepAgents backend name."""

    backend = str(
        get_agent_runtime_option(
            model_config,
            "backend",
            "filesystem",
            legacy_attr="deepagents_backend",
        )
    )
    if backend not in DEEPAGENTS_BACKENDS:
        allowed = ", ".join(sorted(DEEPAGENTS_BACKENDS))
        raise ValueError(f"agent_runtime backend for langchain_deep_agents must be one of: {allowed}")
    if backend == LEGACY_DOCKER_BACKEND:
        return CONTAINER_BACKEND
    return backend


def get_claude_sdk_backend(model_config: ModelConfig) -> str:
    """Return the configured Claude SDK runtime backend."""

    backend = str(get_agent_runtime_option(model_config, "backend", "native"))
    if backend not in CLAUDE_SDK_BACKENDS:
        allowed = ", ".join(sorted(CLAUDE_SDK_BACKENDS))
        raise ValueError(f"agent_runtime backend for claude_agent_sdk must be one of: {allowed}")
    if backend == LEGACY_DOCKER_BACKEND:
        return CONTAINER_BACKEND
    return backend


def is_container_backend(model_config: ModelConfig) -> bool:
    """Return whether this model is configured for containerized execution."""

    backend = str(get_agent_runtime_option(model_config, "backend", "native"))
    return backend in CONTAINER_BACKEND_ALIASES


def get_agent_runtime_access_mode(model_config: ModelConfig) -> str:
    """Return the configured runtime access mode."""

    access_mode = str(get_agent_runtime_option(model_config, "access_mode", "read_write"))
    if access_mode not in AGENT_RUNTIME_ACCESS_MODES:
        allowed = ", ".join(sorted(AGENT_RUNTIME_ACCESS_MODES))
        raise ValueError(f"agent_runtime access_mode must be one of: {allowed}")
    return access_mode


def claude_sdk_sandbox_enabled(model_config: ModelConfig) -> bool:
    """Return whether Claude Agent SDK native sandboxing is enabled."""

    if get_claude_sdk_backend(model_config) == CONTAINER_BACKEND:
        return False
    return bool(
        get_agent_runtime_option(
            model_config,
            "sandbox_enabled",
            True,
            legacy_attr="claude_sdk_sandbox_enabled",
        )
    )


def get_agent_runtime_options(model_config: ModelConfig) -> dict[str, object]:
    """Return adapter runtime options from ModelConfig.extra_kwargs.

    Adapter-specific runtime settings live under ``extra_kwargs["agent_runtime"]``
    so the shared ModelConfig schema does not grow a field for every adapter.
    """

    extra_kwargs = getattr(model_config, "extra_kwargs", None) or {}
    raw_options = extra_kwargs.get(AGENT_RUNTIME_EXTRA_KEY)
    if isinstance(raw_options, dict):
        return raw_options
    return {}


def get_agent_runtime_option(
    model_config: ModelConfig,
    key: str,
    default: object = None,
    *,
    legacy_attr: str | None = None,
) -> object:
    """Read one adapter runtime option with a temporary legacy fallback."""

    options = get_agent_runtime_options(model_config)
    if key in options:
        return options[key]
    if legacy_attr and hasattr(model_config, legacy_attr):
        return getattr(model_config, legacy_attr)
    return default


def _runtime_option_alias(
    model_config: ModelConfig,
    preferred_key: str,
    legacy_key: str,
    default: object = None,
) -> object:
    options = get_agent_runtime_options(model_config)
    if preferred_key in options:
        return options[preferred_key]
    if legacy_key in options:
        return options[legacy_key]
    return default


def _string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list | tuple):
        return tuple(str(item) for item in value)
    return (str(value),)


def get_container_runtime_config(model_config: ModelConfig) -> ContainerRuntimeConfig:
    """Resolve neutral container runtime options with Docker compatibility aliases."""

    runtime = str(get_agent_runtime_option(model_config, "container_runtime", "docker"))
    if runtime not in CONTAINER_RUNTIMES:
        allowed = ", ".join(sorted(CONTAINER_RUNTIMES))
        raise ValueError(f"agent_runtime container_runtime must be one of: {allowed}")

    image = _runtime_option_alias(model_config, "container_image", "docker_image")
    network = str(_runtime_option_alias(model_config, "container_network", "docker_network", "bridge"))
    add_hosts = _string_tuple(_runtime_option_alias(model_config, "container_add_hosts", "docker_add_hosts"))
    if runtime == "docker":
        if network not in {"bridge", "none"}:
            from karenina.ports import AdapterUnavailableError

            raise AdapterUnavailableError(
                "agent_runtime container_network must be 'bridge' or 'none' for Docker",
                reason="invalid_container_network",
            )
    else:
        if network not in {"bridge", ""}:
            from karenina.ports import AdapterUnavailableError

            raise AdapterUnavailableError(
                f"agent_runtime container_network={network!r} is Docker-only and is not supported for {runtime}",
                reason="unsupported_container_network",
            )
        if add_hosts:
            from karenina.ports import AdapterUnavailableError

            raise AdapterUnavailableError(
                f"agent_runtime container_add_hosts is Docker-only and is not supported for {runtime}",
                reason="unsupported_container_add_hosts",
            )

    return ContainerRuntimeConfig(
        runtime=runtime,
        image=str(image) if image else None,
        network=network or "bridge",
        add_hosts=add_hosts,
    )


def _docker_command_output(result: subprocess.CompletedProcess[str]) -> str:
    """Return compact Docker command output for preflight errors."""

    output = "\n".join(part.strip() for part in (result.stderr, result.stdout) if part and part.strip())
    if not output:
        output = f"Docker command exited with status {result.returncode}"
    max_chars = 1_000
    if len(output) > max_chars:
        return f"{output[:max_chars]}..."
    return output


def preflight_docker_runtime(
    *,
    image: str | None,
    timeout: int = DOCKER_PREFLIGHT_TIMEOUT_SECONDS,
    check_image: bool = True,
) -> None:
    """Validate that the host Docker runtime is ready before starting an agent.

    The agent-facing Docker backends execute shell commands by wrapping them in
    ``docker run``. If the daemon or configured image is unavailable, failing
    during backend setup prevents wasted model turns where the agent repeatedly
    retries normal shell commands.
    """

    if shutil.which("docker") is None:
        from karenina.ports import AdapterUnavailableError

        raise AdapterUnavailableError(
            "Docker is required for agent_runtime backend='docker', but the docker command was not found",
            reason="docker_unavailable",
        )

    try:
        info = subprocess.run(  # noqa: S603
            ["docker", "info"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        from karenina.ports import AdapterUnavailableError

        raise AdapterUnavailableError(
            f"Docker daemon preflight timed out after {timeout} seconds while running 'docker info'",
            reason="docker_daemon_unavailable",
        ) from e

    if info.returncode != 0:
        from karenina.ports import AdapterUnavailableError

        raise AdapterUnavailableError(
            "Docker daemon is not reachable for agent_runtime backend='docker'. "
            f"'docker info' failed: {_docker_command_output(info)}",
            reason="docker_daemon_unavailable",
        )

    if not check_image or not image:
        return

    try:
        inspect = subprocess.run(  # noqa: S603
            ["docker", "image", "inspect", image],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        from karenina.ports import AdapterUnavailableError

        raise AdapterUnavailableError(
            f"Docker image preflight timed out after {timeout} seconds while checking image '{image}'",
            reason="docker_image_unavailable",
        ) from e

    if inspect.returncode != 0:
        from karenina.ports import AdapterUnavailableError

        raise AdapterUnavailableError(
            "Docker image required by agent_runtime backend='docker' is not available locally: "
            f"{image}. 'docker image inspect {image}' failed: {_docker_command_output(inspect)}",
            reason="docker_image_unavailable",
        )


def preflight_container_runtime(
    config: ContainerRuntimeConfig,
    *,
    timeout: int = DOCKER_PREFLIGHT_TIMEOUT_SECONDS,
    check_image: bool = True,
) -> None:
    """Validate a configured container runtime before an agent starts."""

    if config.runtime == "docker":
        preflight_docker_runtime(image=config.image, timeout=timeout, check_image=check_image)
        return

    executable = config.runtime
    if shutil.which(executable) is None:
        from karenina.ports import AdapterUnavailableError

        raise AdapterUnavailableError(
            f"{executable} is required for agent_runtime container_runtime='{config.runtime}', "
            f"but the {executable} command was not found",
            reason=f"{config.runtime}_unavailable",
        )

    if not check_image:
        return
    if not config.image:
        from karenina.ports import AdapterUnavailableError

        raise AdapterUnavailableError(
            f"agent_runtime container_image is required when container_runtime='{config.runtime}'",
            reason="missing_container_image",
        )

    image_path = Path(config.image).expanduser()
    if image_path.suffix != ".sif" or not image_path.is_file():
        from karenina.ports import AdapterUnavailableError

        raise AdapterUnavailableError(
            f"agent_runtime container_image for {config.runtime} must be an existing local .sif file: {config.image}",
            reason="container_image_unavailable",
        )


def _env_flags_for_docker(env: dict[str, str | None]) -> list[str]:
    flags: list[str] = []
    for name, value in env.items():
        if value is None:
            flags.extend(["--env", name])
        else:
            flags.extend(["--env", f"{name}={value}"])
    return flags


def _env_flags_for_singularity(env: dict[str, str | None]) -> list[str]:
    flags: list[str] = []
    for name, value in env.items():
        env_value = os.environ.get(name, "") if value is None else value
        flags.extend(["--env", f"{name}={env_value}"])
    return flags


def build_container_command(
    *,
    config: ContainerRuntimeConfig,
    host_workspace: str | Path,
    argv: list[str],
    env: dict[str, str | None] | None = None,
    interactive: bool = False,
) -> list[str]:
    """Build a one-shot command for Docker, Singularity, or Apptainer."""

    if not config.image:
        from karenina.ports import AdapterUnavailableError

        raise AdapterUnavailableError(
            "agent_runtime container_image is required for container execution",
            reason="missing_container_image",
        )
    host_workspace = str(Path(host_workspace).resolve())
    env = env or {}

    if config.runtime == "docker":
        command = ["docker", "run", "--rm"]
        if interactive:
            command.append("-i")
        command.extend(
            [
                "--network",
                config.network,
                "--workdir",
                SANDBOX_WORKSPACE_PATH,
                "--volume",
                f"{host_workspace}:{SANDBOX_WORKSPACE_PATH}:rw",
            ]
        )
        for host_mapping in config.add_hosts:
            command.extend(["--add-host", host_mapping])
        command.extend(_env_flags_for_docker(env))
        command.extend(["--pids-limit", "256"])
        if hasattr(os, "getuid") and hasattr(os, "getgid"):
            command.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
        command.extend([config.image, *argv])
        return command

    # Sandbox isolation: prevent the host filesystem from leaking host-installed
    # Python packages into the model's sandbox.
    #   --no-home              skips Singularity's default $HOME auto-mount
    #   --no-mount bind-paths  drops system-default binds from singularity.conf
    #                          (e.g. /homes, /hps/software, /nfs/*); only the
    #                          explicit --bind below remains, so the model
    #                          cannot reach $HOME/.local/.../site-packages
    #                          either implicitly or via sys.path.insert.
    #   PYTHONNOUSERSITE=1     belt-and-braces: ignore user-site even if a
    #                          future caller adds a bind that exposes one.
    command = [
        config.runtime,
        "exec",
        "--no-home",
        "--no-mount",
        "bind-paths",
        "--bind",
        f"{host_workspace}:{SANDBOX_WORKSPACE_PATH}:rw",
        "--pwd",
        SANDBOX_WORKSPACE_PATH,
        "--env",
        "PYTHONNOUSERSITE=1",
        *_env_flags_for_singularity(env),
        config.image,
        *argv,
    ]
    return command


def _deepagents_capabilities(model_config: ModelConfig) -> PortCapabilities:
    backend = get_deepagents_backend(model_config)
    access_mode = get_agent_runtime_access_mode(model_config)
    return PortCapabilities(
        supports_system_prompt=True,
        supports_file_tools=True,
        supports_code_execution=access_mode != "read_only" and backend in {CONTAINER_BACKEND, "local_shell"},
        uses_sandboxed_execution=backend == CONTAINER_BACKEND,
    )


def _claude_sdk_capabilities(model_config: ModelConfig) -> PortCapabilities:
    access_mode = get_agent_runtime_access_mode(model_config)
    backend = get_claude_sdk_backend(model_config)
    return PortCapabilities(
        supports_system_prompt=True,
        supports_file_tools=True,
        supports_code_execution=access_mode != "read_only",
        uses_sandboxed_execution=backend == CONTAINER_BACKEND or claude_sdk_sandbox_enabled(model_config),
    )


def _default_workspace_path_for_prompt(_model_config: ModelConfig, workspace_path: Path | None) -> str | None:
    if workspace_path is None:
        return None
    return str(workspace_path)


def _default_map_path_for_prompt(
    _model_config: ModelConfig,
    path: Path | None,
    _workspace_path: Path | None,
) -> str | None:
    if path is None:
        return None
    return str(path)


def _deepagents_workspace_path_for_prompt(model_config: ModelConfig, workspace_path: Path | None) -> str | None:
    if workspace_path is None:
        return None
    if get_deepagents_backend(model_config) == CONTAINER_BACKEND:
        return SANDBOX_WORKSPACE_PATH
    return str(workspace_path)


def _deepagents_map_path_for_prompt(
    model_config: ModelConfig,
    path: Path | None,
    workspace_path: Path | None,
) -> str | None:
    if path is None:
        return None
    if get_deepagents_backend(model_config) != CONTAINER_BACKEND or workspace_path is None:
        return str(path)

    try:
        rel = path.resolve().relative_to(workspace_path.resolve())
    except ValueError:
        return str(path)

    if rel.as_posix() == ".":
        return SANDBOX_WORKSPACE_PATH
    return f"{SANDBOX_WORKSPACE_PATH}/{rel.as_posix()}"


def _claude_sdk_workspace_path_for_prompt(model_config: ModelConfig, workspace_path: Path | None) -> str | None:
    if workspace_path is None:
        return None
    if get_claude_sdk_backend(model_config) == CONTAINER_BACKEND:
        return SANDBOX_WORKSPACE_PATH
    return str(workspace_path)


def _claude_sdk_map_path_for_prompt(
    model_config: ModelConfig,
    path: Path | None,
    workspace_path: Path | None,
) -> str | None:
    if path is None:
        return None
    if get_claude_sdk_backend(model_config) != CONTAINER_BACKEND or workspace_path is None:
        return str(path)

    try:
        rel = path.resolve().relative_to(workspace_path.resolve())
    except ValueError:
        return str(path)

    if rel.as_posix() == ".":
        return SANDBOX_WORKSPACE_PATH
    return f"{SANDBOX_WORKSPACE_PATH}/{rel.as_posix()}"


def get_agent_runtime_profile(interface: str) -> AgentRuntimeProfile | None:
    """Return the registered runtime profile for an interface, if any."""

    return _runtime_profiles.get(interface)


def get_agent_runtime_capabilities(model_config: ModelConfig) -> PortCapabilities:
    """Return agent capabilities implied by the registered runtime profile."""

    profile = get_agent_runtime_profile(model_config.interface)
    if profile is None:
        return PortCapabilities(supports_system_prompt=True)
    return profile.capabilities(model_config)


def workspace_path_for_prompt(model_config: ModelConfig, workspace_path: Path | None) -> str | None:
    """Return the workspace path that should be shown to the model."""

    profile = get_agent_runtime_profile(model_config.interface)
    if profile and profile.workspace_path_for_prompt:
        return profile.workspace_path_for_prompt(model_config, workspace_path)
    return _default_workspace_path_for_prompt(model_config, workspace_path)


def map_path_for_prompt(
    model_config: ModelConfig,
    path: Path | None,
    workspace_path: Path | None,
) -> str | None:
    """Map a host path to the sandbox-visible path when needed."""

    profile = get_agent_runtime_profile(model_config.interface)
    if profile and profile.map_path_for_prompt:
        return profile.map_path_for_prompt(model_config, path, workspace_path)
    return _default_map_path_for_prompt(model_config, path, workspace_path)


register_agent_runtime_profile(
    "langchain_deep_agents",
    AgentRuntimeProfile(
        capabilities=_deepagents_capabilities,
        workspace_path_for_prompt=_deepagents_workspace_path_for_prompt,
        map_path_for_prompt=_deepagents_map_path_for_prompt,
    ),
)
register_agent_runtime_profile(
    "claude_agent_sdk",
    AgentRuntimeProfile(
        capabilities=_claude_sdk_capabilities,
        workspace_path_for_prompt=_claude_sdk_workspace_path_for_prompt,
        map_path_for_prompt=_claude_sdk_map_path_for_prompt,
    ),
)
