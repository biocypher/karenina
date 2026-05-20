"""Compatibility wrapper for DeepAgents runtime helpers."""

from __future__ import annotations

from karenina.adapters.agent_runtime import (
    SANDBOX_WORKSPACE_PATH,
    get_agent_runtime_capabilities,
    get_deepagents_backend,
    map_path_for_prompt,
    workspace_path_for_prompt,
)

get_deepagents_capabilities = get_agent_runtime_capabilities

__all__ = [
    "SANDBOX_WORKSPACE_PATH",
    "get_deepagents_backend",
    "get_deepagents_capabilities",
    "map_path_for_prompt",
    "workspace_path_for_prompt",
]
