"""Shared runtime helpers for agent adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from karenina.ports.capabilities import PortCapabilities

if TYPE_CHECKING:
    from karenina.schemas.config import ModelConfig


SANDBOX_WORKSPACE_PATH = "/workspace"
AGENT_RUNTIME_EXTRA_KEY = "agent_runtime"


@dataclass(frozen=True)
class AgentRuntimeProfile:
    """Adapter-specific runtime behavior used by shared pipeline prompts."""

    capabilities: Callable[[ModelConfig], PortCapabilities]
    workspace_path_for_prompt: Callable[[ModelConfig, Path | None], str | None] | None = None
    map_path_for_prompt: Callable[[ModelConfig, Path | None, Path | None], str | None] | None = None


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

    return str(
        get_agent_runtime_option(
            model_config,
            "backend",
            "filesystem",
            legacy_attr="deepagents_backend",
        )
    )


def claude_sdk_sandbox_enabled(model_config: ModelConfig) -> bool:
    """Return whether Claude Agent SDK native sandboxing is enabled."""

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


def _deepagents_capabilities(model_config: ModelConfig) -> PortCapabilities:
    backend = get_deepagents_backend(model_config)
    return PortCapabilities(
        supports_system_prompt=True,
        supports_file_tools=True,
        supports_code_execution=backend in {"docker", "local_shell"},
        uses_sandboxed_execution=backend == "docker",
    )


def _claude_sdk_capabilities(model_config: ModelConfig) -> PortCapabilities:
    return PortCapabilities(
        supports_system_prompt=True,
        supports_file_tools=True,
        supports_code_execution=True,
        uses_sandboxed_execution=claude_sdk_sandbox_enabled(model_config),
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
    if get_deepagents_backend(model_config) == "docker":
        return SANDBOX_WORKSPACE_PATH
    return str(workspace_path)


def _deepagents_map_path_for_prompt(
    model_config: ModelConfig,
    path: Path | None,
    workspace_path: Path | None,
) -> str | None:
    if path is None:
        return None
    if get_deepagents_backend(model_config) != "docker" or workspace_path is None:
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
        workspace_path_for_prompt=_default_workspace_path_for_prompt,
        map_path_for_prompt=_default_map_path_for_prompt,
    ),
)
