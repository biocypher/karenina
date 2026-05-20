"""Tests for shared agent runtime helpers."""

from __future__ import annotations

from karenina.adapters.agent_runtime import (
    AgentRuntimeProfile,
    get_agent_runtime_capabilities,
    map_path_for_prompt,
    register_agent_runtime_profile,
    workspace_path_for_prompt,
)
from karenina.ports.capabilities import PortCapabilities
from karenina.schemas.config import ModelConfig


def test_deepagents_docker_workspace_maps_to_container_path(tmp_path):
    config = ModelConfig(
        id="deep",
        model_name="claude-sonnet-4-20250514",
        interface="langchain_deep_agents",
        extra_kwargs={"agent_runtime": {"backend": "docker"}},
    )

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trace_path = workspace / "traces" / "trace.md"

    assert workspace_path_for_prompt(config, workspace) == "/workspace"
    assert map_path_for_prompt(config, trace_path, workspace) == "/workspace/traces/trace.md"


def test_claude_sdk_workspace_uses_host_path(tmp_path):
    config = ModelConfig(
        id="claude",
        model_name="claude-sonnet-4-20250514",
        interface="claude_agent_sdk",
    )

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trace_path = workspace / "traces" / "trace.md"

    assert workspace_path_for_prompt(config, workspace) == str(workspace)
    assert map_path_for_prompt(config, trace_path, workspace) == str(trace_path)


def test_claude_sdk_docker_workspace_maps_to_container_path(tmp_path):
    config = ModelConfig(
        id="claude",
        model_name="claude-sonnet-4-20250514",
        interface="claude_agent_sdk",
        extra_kwargs={"agent_runtime": {"backend": "docker", "docker_image": "karenina-bixbench-claude:latest"}},
    )

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trace_path = workspace / "traces" / "trace.md"

    assert workspace_path_for_prompt(config, workspace) == "/workspace"
    assert map_path_for_prompt(config, trace_path, workspace) == "/workspace/traces/trace.md"


def test_deepagents_read_only_access_mode_disables_code_execution_capability():
    config = ModelConfig(
        id="deep",
        model_name="claude-sonnet-4-20250514",
        interface="langchain_deep_agents",
        extra_kwargs={"agent_runtime": {"backend": "docker", "access_mode": "read_only"}},
    )

    capabilities = get_agent_runtime_capabilities(config)

    assert capabilities.supports_file_tools is True
    assert capabilities.supports_code_execution is False
    assert capabilities.uses_sandboxed_execution is True


def test_claude_sdk_read_only_access_mode_disables_code_execution_capability():
    config = ModelConfig(
        id="claude",
        model_name="claude-sonnet-4-20250514",
        interface="claude_agent_sdk",
        extra_kwargs={"agent_runtime": {"access_mode": "read_only"}},
    )

    capabilities = get_agent_runtime_capabilities(config)

    assert capabilities.supports_file_tools is True
    assert capabilities.supports_code_execution is False
    assert capabilities.uses_sandboxed_execution is True


def test_claude_sdk_docker_capabilities_report_sandboxed_execution():
    config = ModelConfig(
        id="claude",
        model_name="claude-sonnet-4-20250514",
        interface="claude_agent_sdk",
        extra_kwargs={"agent_runtime": {"backend": "docker", "docker_image": "karenina-bixbench-claude:latest"}},
    )

    capabilities = get_agent_runtime_capabilities(config)

    assert capabilities.supports_file_tools is True
    assert capabilities.supports_code_execution is True
    assert capabilities.uses_sandboxed_execution is True


def test_runtime_profile_registry_allows_extension_adapters():
    profile = AgentRuntimeProfile(
        capabilities=lambda _config: PortCapabilities(
            supports_system_prompt=True,
            supports_file_tools=True,
            supports_code_execution=False,
        ),
        workspace_path_for_prompt=lambda _config, _workspace: "custom://workspace",
        map_path_for_prompt=lambda _config, _path, _workspace: "custom://trace",
    )
    register_agent_runtime_profile("test_runtime_profile_adapter", profile, force=True)
    config = ModelConfig.model_construct(
        id="custom",
        model_name="custom-model",
        interface="test_runtime_profile_adapter",
    )

    assert workspace_path_for_prompt(config, None) == "custom://workspace"
    assert map_path_for_prompt(config, None, None) == "custom://trace"
