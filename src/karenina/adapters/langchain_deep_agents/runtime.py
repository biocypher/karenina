"""Runtime helpers for the LangChain DeepAgents adapter."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from karenina.ports.capabilities import PortCapabilities

if TYPE_CHECKING:
    from karenina.schemas.config import ModelConfig


SANDBOX_WORKSPACE_PATH = "/workspace"


def get_deepagents_backend(model_config: ModelConfig) -> str:
    """Return the configured DeepAgents backend name."""

    return getattr(model_config, "deepagents_backend", "filesystem")


def get_deepagents_capabilities(model_config: ModelConfig) -> PortCapabilities:
    """Return agent capabilities implied by the DeepAgents backend."""

    backend = get_deepagents_backend(model_config)
    return PortCapabilities(
        supports_system_prompt=True,
        supports_file_tools=True,
        supports_code_execution=backend in {"docker", "local_shell"},
        uses_sandboxed_execution=backend == "docker",
    )


def workspace_path_for_prompt(model_config: ModelConfig, workspace_path: Path | None) -> str | None:
    """Return the workspace path that should be shown to the model."""

    if workspace_path is None:
        return None
    if get_deepagents_backend(model_config) == "docker":
        return SANDBOX_WORKSPACE_PATH
    return str(workspace_path)


def map_path_for_prompt(
    model_config: ModelConfig,
    path: Path | None,
    workspace_path: Path | None,
) -> str | None:
    """Map a host workspace path to the sandbox-visible path when needed."""

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
