"""Tests for shared agent runtime helpers."""

from __future__ import annotations

import subprocess

import pytest

from karenina.adapters.agent_runtime import (
    AgentRuntimeProfile,
    ContainerRuntimeConfig,
    build_container_command,
    get_agent_runtime_capabilities,
    get_claude_sdk_backend,
    get_container_runtime_config,
    map_path_for_prompt,
    preflight_container_runtime,
    preflight_docker_runtime,
    register_agent_runtime_profile,
    workspace_path_for_prompt,
)
from karenina.ports import AdapterUnavailableError
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


def test_legacy_docker_backend_normalizes_to_container_runtime():
    config = ModelConfig(
        id="claude",
        model_name="claude-sonnet-4-20250514",
        interface="claude_agent_sdk",
        extra_kwargs={
            "agent_runtime": {
                "backend": "docker",
                "docker_image": "karenina-bixbench-claude:latest",
            }
        },
    )

    assert get_claude_sdk_backend(config) == "container"
    container_config = get_container_runtime_config(config)
    assert container_config.runtime == "docker"
    assert container_config.image == "karenina-bixbench-claude:latest"


def test_singularity_container_workspace_maps_to_container_path(tmp_path):
    config = ModelConfig(
        id="claude",
        model_name="claude-sonnet-4-20250514",
        interface="claude_agent_sdk",
        extra_kwargs={
            "agent_runtime": {
                "backend": "container",
                "container_runtime": "singularity",
                "container_image": str(tmp_path / "image.sif"),
            }
        },
    )

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trace_path = workspace / "traces" / "trace.md"

    assert workspace_path_for_prompt(config, workspace) == "/workspace"
    assert map_path_for_prompt(config, trace_path, workspace) == "/workspace/traces/trace.md"


def test_singularity_preflight_requires_existing_sif(monkeypatch, tmp_path):
    monkeypatch.setattr("karenina.adapters.agent_runtime.shutil.which", lambda _name: "/usr/bin/singularity")

    with pytest.raises(AdapterUnavailableError) as exc_info:
        preflight_container_runtime(
            ContainerRuntimeConfig(
                runtime="singularity",
                image=str(tmp_path / "missing.sif"),
            )
        )

    assert exc_info.value.reason == "container_image_unavailable"


def test_singularity_command_binds_workspace_and_sets_env(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    image = tmp_path / "image.sif"
    image.write_text("")

    command = build_container_command(
        config=ContainerRuntimeConfig(runtime="singularity", image=str(image)),
        host_workspace=workspace,
        argv=["/bin/sh", "-lc", "pwd"],
        env={"UV_LINK_MODE": "copy"},
    )

    assert command[:2] == ["singularity", "exec"]
    assert f"{workspace.resolve()}:/workspace:rw" in command
    assert "--pwd" in command
    assert command[command.index("--pwd") + 1] == "/workspace"
    assert "--env" in command
    assert "UV_LINK_MODE=copy" in command
    assert str(image) in command


def test_singularity_rejects_docker_only_add_hosts():
    config = ModelConfig(
        id="claude",
        model_name="claude-sonnet-4-20250514",
        interface="claude_agent_sdk",
        extra_kwargs={
            "agent_runtime": {
                "backend": "container",
                "container_runtime": "singularity",
                "container_image": "/tmp/image.sif",
                "container_add_hosts": ["internal:10.0.0.1"],
            }
        },
    )

    with pytest.raises(AdapterUnavailableError) as exc_info:
        get_container_runtime_config(config)

    assert exc_info.value.reason == "unsupported_container_add_hosts"


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


def test_docker_preflight_requires_docker_cli(monkeypatch):
    monkeypatch.setattr("karenina.adapters.agent_runtime.shutil.which", lambda _name: None)

    with pytest.raises(AdapterUnavailableError) as exc_info:
        preflight_docker_runtime(image="karenina-bixbench:latest")

    assert exc_info.value.reason == "docker_unavailable"


def test_docker_preflight_reports_daemon_unavailable(monkeypatch):
    monkeypatch.setattr("karenina.adapters.agent_runtime.shutil.which", lambda _name: "/usr/bin/docker")

    def fake_run(command, **_kwargs):
        assert command == ["docker", "info"]
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="Cannot connect to Docker daemon")

    monkeypatch.setattr("karenina.adapters.agent_runtime.subprocess.run", fake_run)

    with pytest.raises(AdapterUnavailableError) as exc_info:
        preflight_docker_runtime(image="karenina-bixbench:latest")

    assert exc_info.value.reason == "docker_daemon_unavailable"
    assert "Cannot connect to Docker daemon" in str(exc_info.value)


def test_docker_preflight_reports_missing_image(monkeypatch):
    monkeypatch.setattr("karenina.adapters.agent_runtime.shutil.which", lambda _name: "/usr/bin/docker")
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        if command == ["docker", "info"]:
            return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="No such image")

    monkeypatch.setattr("karenina.adapters.agent_runtime.subprocess.run", fake_run)

    with pytest.raises(AdapterUnavailableError) as exc_info:
        preflight_docker_runtime(image="karenina-bixbench:latest")

    assert calls == [
        ["docker", "info"],
        ["docker", "image", "inspect", "karenina-bixbench:latest"],
    ]
    assert exc_info.value.reason == "docker_image_unavailable"
    assert "No such image" in str(exc_info.value)


def test_docker_preflight_accepts_ready_runtime(monkeypatch):
    monkeypatch.setattr("karenina.adapters.agent_runtime.shutil.which", lambda _name: "/usr/bin/docker")
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr("karenina.adapters.agent_runtime.subprocess.run", fake_run)

    preflight_docker_runtime(image="karenina-bixbench:latest")

    assert calls == [
        ["docker", "info"],
        ["docker", "image", "inspect", "karenina-bixbench:latest"],
    ]
