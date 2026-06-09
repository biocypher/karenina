"""Tests for workspace_path to cwd wiring in ClaudeSDKAgentAdapter."""

import asyncio
import sys
import types
from pathlib import Path

import pytest

from karenina.adapters.claude_agent_sdk.agent import CLAUDE_SDK_DOCKER_WRAPPER, ClaudeSDKAgentAdapter
from karenina.adapters.claude_agent_sdk.docker_cli_wrapper import build_docker_command
from karenina.ports import AdapterUnavailableError, AgentConfig, Message
from karenina.schemas.config.models import ModelConfig


def _make_fake_claude_agent_sdk():
    """Create a minimal stub for claude_agent_sdk with a real ClaudeAgentOptions."""

    class ClaudeAgentOptions:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
            # Set defaults for fields not provided
            if not hasattr(self, "cwd"):
                self.cwd = None

    mod = types.ModuleType("claude_agent_sdk")
    mod.ClaudeAgentOptions = ClaudeAgentOptions
    return mod


@pytest.fixture(autouse=True)
def patch_claude_agent_sdk(monkeypatch):
    """Patch claude_agent_sdk into sys.modules so tests run without the real package."""
    if "claude_agent_sdk" not in sys.modules:
        fake_mod = _make_fake_claude_agent_sdk()
        monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake_mod)
    yield


@pytest.fixture
def no_docker_preflight(monkeypatch):
    monkeypatch.setattr(
        "karenina.adapters.claude_agent_sdk.agent.preflight_container_runtime",
        lambda *_args, **_kwargs: None,
    )


@pytest.mark.unit
class TestWorkspacePath:
    def _make_adapter(self):
        config = ModelConfig(
            id="test",
            model_name="claude-sonnet-4-20250514",
            interface="claude_agent_sdk",
        )
        return ClaudeSDKAgentAdapter(config)

    def test_build_options_sets_cwd_when_workspace_path_provided(self):
        adapter = self._make_adapter()
        config = AgentConfig(workspace_path=Path("/tmp/test_workspace"))
        options = adapter._build_options(
            system_prompt="test",
            mcp_servers=None,
            config=config,
        )
        assert str(options.cwd) == "/tmp/test_workspace"

    def test_build_options_uses_process_cwd_when_workspace_path_is_none(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        adapter = self._make_adapter()
        config = AgentConfig()
        options = adapter._build_options(
            system_prompt="test",
            mcp_servers=None,
            config=config,
        )
        assert str(options.cwd) == str(tmp_path)

    def test_build_options_enables_native_sandbox_by_default(self):
        adapter = self._make_adapter()
        options = adapter._build_options(
            system_prompt="test",
            mcp_servers=None,
            config=AgentConfig(),
        )

        assert options.permission_mode == "acceptEdits"
        assert options.sandbox == {
            "enabled": True,
            "failIfUnavailable": True,
            "autoAllowBashIfSandboxed": True,
            "allowUnsandboxedCommands": False,
        }

    def test_capabilities_report_sandboxed_execution_by_default(self):
        adapter = self._make_adapter()

        assert adapter.capabilities.supports_file_tools is True
        assert adapter.capabilities.supports_code_execution is True
        assert adapter.capabilities.uses_sandboxed_execution is True

    def test_build_options_can_disable_sandbox(self):
        config = ModelConfig(
            id="test",
            model_name="claude-sonnet-4-20250514",
            interface="claude_agent_sdk",
            extra_kwargs={"agent_runtime": {"sandbox_enabled": False}},
        )
        adapter = ClaudeSDKAgentAdapter(config)

        options = adapter._build_options(
            system_prompt="test",
            mcp_servers=None,
            config=AgentConfig(),
        )

        assert getattr(options, "sandbox", None) is None
        assert adapter.capabilities.uses_sandboxed_execution is False

    def test_native_mode_sets_workspace_local_uv_env(self, tmp_path):
        adapter = self._make_adapter()
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        options = adapter._build_options(
            system_prompt="test",
            mcp_servers=None,
            config=AgentConfig(workspace_path=workspace),
        )

        assert options.env["UV_CACHE_DIR"] == str(workspace / ".uv-cache")
        assert options.env["XDG_CACHE_HOME"] == str(workspace / ".cache")
        assert options.env["UV_PROJECT_ENVIRONMENT"] == str(workspace / ".venv")

    @pytest.mark.usefixtures("no_docker_preflight")
    def test_docker_backend_uses_wrapper_and_disables_native_sandbox(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        config = ModelConfig(
            id="test",
            model_name="claude-sonnet-4-20250514",
            interface="claude_agent_sdk",
            extra_kwargs={
                "agent_runtime": {
                    "backend": "docker",
                    "docker_image": "karenina-bixbench-claude:latest",
                    "docker_network": "none",
                    "docker_add_hosts": ["hl-codon-gpu-020:10.0.0.20"],
                }
            },
        )
        adapter = ClaudeSDKAgentAdapter(config)

        options = adapter._build_options(
            system_prompt="test",
            mcp_servers=None,
            config=AgentConfig(workspace_path=workspace),
        )

        assert options.cli_path == str(CLAUDE_SDK_DOCKER_WRAPPER)
        assert getattr(options, "sandbox", None) is None
        assert options.cwd == str(workspace)
        assert options.env["KARENINA_CLAUDE_DOCKER_WORKSPACE"] == str(workspace.resolve())
        assert options.env["KARENINA_CLAUDE_DOCKER_IMAGE"] == "karenina-bixbench-claude:latest"
        assert options.env["KARENINA_CLAUDE_DOCKER_NETWORK"] == "none"
        assert options.env["KARENINA_CLAUDE_DOCKER_ADD_HOSTS"] == "hl-codon-gpu-020:10.0.0.20"
        assert options.env["CLAUDE_CONFIG_DIR"] == "/tmp/claude-config"
        assert options.permission_mode == "bypassPermissions"
        assert adapter.capabilities.uses_sandboxed_execution is True

    def test_docker_backend_runs_preflight(self, tmp_path, monkeypatch):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        captured: dict[str, str | None] = {}
        monkeypatch.setattr(
            "karenina.adapters.claude_agent_sdk.agent.preflight_container_runtime",
            lambda config, **_kwargs: captured.update({"image": config.image, "runtime": config.runtime}),
        )
        config = ModelConfig(
            id="test",
            model_name="claude-sonnet-4-20250514",
            interface="claude_agent_sdk",
            extra_kwargs={
                "agent_runtime": {
                    "backend": "docker",
                    "docker_image": "karenina-bixbench-claude:latest",
                }
            },
        )
        adapter = ClaudeSDKAgentAdapter(config)

        adapter._build_options(
            system_prompt="test",
            mcp_servers=None,
            config=AgentConfig(workspace_path=workspace),
        )

        assert captured == {"image": "karenina-bixbench-claude:latest", "runtime": "docker"}

    @pytest.mark.usefixtures("no_docker_preflight")
    def test_zai_anthropic_endpoint_sets_auth_token_and_glm_mapping(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        config = ModelConfig(
            id="test",
            model_name="glm-5.1",
            interface="claude_agent_sdk",
            anthropic_base_url="https://api.z.ai/api/anthropic",
            anthropic_api_key="zai-key",
            extra_kwargs={
                "agent_runtime": {
                    "backend": "docker",
                    "docker_image": "karenina-bixbench-claude:latest",
                }
            },
        )
        adapter = ClaudeSDKAgentAdapter(config)

        options = adapter._build_options(
            system_prompt="test",
            mcp_servers=None,
            config=AgentConfig(workspace_path=workspace),
        )

        assert options.model == "claude-sonnet-4-5"
        assert options.env["ANTHROPIC_AUTH_TOKEN"] == "zai-key"
        assert options.env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "glm-5.1"
        assert options.env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "glm-5.1"
        assert options.env["API_TIMEOUT_MS"] == "3000000"

    @pytest.mark.usefixtures("no_docker_preflight")
    def test_docker_read_only_keeps_safe_permission_mode(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        config = ModelConfig(
            id="test",
            model_name="claude-sonnet-4-20250514",
            interface="claude_agent_sdk",
            extra_kwargs={
                "agent_runtime": {
                    "backend": "docker",
                    "access_mode": "read_only",
                    "docker_image": "karenina-bixbench-claude:latest",
                }
            },
        )
        adapter = ClaudeSDKAgentAdapter(config)

        options = adapter._build_options(
            system_prompt="test",
            mcp_servers=None,
            config=AgentConfig(workspace_path=workspace),
        )

        assert options.permission_mode == "acceptEdits"
        assert options.allowed_tools == ["Read", "Grep", "Glob", "LS"]

    def test_docker_backend_requires_workspace(self):
        config = ModelConfig(
            id="test",
            model_name="claude-sonnet-4-20250514",
            interface="claude_agent_sdk",
            extra_kwargs={
                "agent_runtime": {
                    "backend": "docker",
                    "docker_image": "karenina-bixbench-claude:latest",
                }
            },
        )
        adapter = ClaudeSDKAgentAdapter(config)

        with pytest.raises(AdapterUnavailableError, match="requires an AgentConfig.workspace_path"):
            adapter._build_options(
                system_prompt="test",
                mcp_servers=None,
                config=AgentConfig(),
            )

    def test_read_only_access_mode_allows_only_read_tools(self):
        config = ModelConfig(
            id="test",
            model_name="claude-sonnet-4-20250514",
            interface="claude_agent_sdk",
            extra_kwargs={"agent_runtime": {"access_mode": "read_only"}},
        )
        adapter = ClaudeSDKAgentAdapter(config)

        options = adapter._build_options(
            system_prompt="test",
            mcp_servers=None,
            config=AgentConfig(),
        )

        assert options.tools == ["Read", "Grep", "Glob", "LS"]
        assert options.allowed_tools == ["Read", "Grep", "Glob", "LS"]
        assert adapter.capabilities.supports_code_execution is False

    def test_build_options_ignores_plain_permission_mode_extra(self):
        adapter = self._make_adapter()

        options = adapter._build_options(
            system_prompt="test",
            mcp_servers=None,
            config=AgentConfig(extra={"permission_mode": "bypassPermissions"}),
        )

        assert options.permission_mode == "acceptEdits"

    def test_build_options_allows_named_unsafe_permission_override(self):
        adapter = self._make_adapter()

        options = adapter._build_options(
            system_prompt="test",
            mcp_servers=None,
            config=AgentConfig(
                extra={
                    "permission_mode": "bypassPermissions",
                    "allow_unsafe_permission_mode_override": True,
                }
            ),
        )

        assert options.permission_mode == "bypassPermissions"


@pytest.mark.unit
class TestDockerCliWrapper:
    def test_build_docker_command_maps_workspace_and_forwards_endpoint(self, monkeypatch, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        monkeypatch.setenv("KARENINA_CLAUDE_DOCKER_WORKSPACE", str(workspace))
        monkeypatch.setenv("KARENINA_CLAUDE_DOCKER_IMAGE", "karenina-bixbench-claude:latest")
        monkeypatch.setenv("KARENINA_CLAUDE_DOCKER_NETWORK", "bridge")
        monkeypatch.setenv("KARENINA_CLAUDE_DOCKER_ADD_HOSTS", "hl-codon-gpu-020:10.0.0.20")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://hl-codon-gpu-020:8000")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "EMPTY")
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "zai-key")
        monkeypatch.setenv("ANTHROPIC_DEFAULT_SONNET_MODEL", "glm-5.1")
        monkeypatch.setenv("ANTHROPIC_DEFAULT_OPUS_MODEL", "glm-5.1")

        command = build_docker_command(["--version", "--settings", f'{{"cwd":"{workspace}"}}'])

        assert command[:4] == ["docker", "run", "--rm", "-i"]
        assert "--network" in command
        assert command[command.index("--network") + 1] == "bridge"
        assert "--add-host" in command
        assert command[command.index("--add-host") + 1] == "hl-codon-gpu-020:10.0.0.20"
        assert f"{workspace}:/workspace:rw" in command
        assert "ANTHROPIC_BASE_URL" in command
        assert "ANTHROPIC_API_KEY" in command
        assert "ANTHROPIC_AUTH_TOKEN" in command
        assert "ANTHROPIC_DEFAULT_SONNET_MODEL" in command
        assert "ANTHROPIC_DEFAULT_OPUS_MODEL" in command
        assert "karenina-bixbench-claude:latest" in command
        image_index = command.index("karenina-bixbench-claude:latest")
        # The CLI is launched through a login shell so /etc/profile provisioning
        # (per-task scratch root in the BixBench images) applies to claude and
        # every Bash tool subshell it spawns.
        assert command[image_index + 1 : image_index + 5] == ["/bin/sh", "-lc", 'exec claude "$@"', "claude"]
        assert str(workspace) not in command[-1]
        assert "/workspace" in command[-1]

    def test_build_singularity_command_maps_workspace(self, monkeypatch, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        image = tmp_path / "karenina.sif"
        image.write_text("")
        monkeypatch.setenv("KARENINA_CLAUDE_CONTAINER_WORKSPACE", str(workspace))
        monkeypatch.setenv("KARENINA_CLAUDE_CONTAINER_RUNTIME", "singularity")
        monkeypatch.setenv("KARENINA_CLAUDE_CONTAINER_IMAGE", str(image))
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://hl-codon-gpu-020:8000")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "EMPTY")

        command = build_docker_command(["--settings", f'{{"cwd":"{workspace}"}}'])

        assert command[:2] == ["singularity", "exec"]
        assert f"{workspace.resolve()}:/workspace:rw" in command
        assert "--pwd" in command
        assert command[command.index("--pwd") + 1] == "/workspace"
        assert str(image) in command
        image_index = command.index(str(image))
        assert command[image_index + 1 : image_index + 5] == ["/bin/sh", "-lc", 'exec claude "$@"', "claude"]
        assert str(workspace) not in command[-1]
        assert "/workspace" in command[-1]


@pytest.mark.unit
class TestBuildOptionsIsolation:
    """The adapter must build ClaudeAgentOptions that do not load personal MCP servers.

    Regression guard for issue 089 (claude-agent-sdk subprocess loading personal
    ~/.claude/ MCP servers). The adapter MUST unconditionally set setting_sources=[]
    and forward CLAUDE_CONFIG_DIR from the parent process environment.
    """

    def _adapter(self, **overrides) -> ClaudeSDKAgentAdapter:
        defaults = {
            "id": "test",
            "model_name": "qwen3.5-122b-a10b",
            "interface": "claude_agent_sdk",
            "anthropic_base_url": "http://codon-gpu-001:8000",
            "anthropic_api_key": "EMPTY",
        }
        defaults.update(overrides)
        return ClaudeSDKAgentAdapter(ModelConfig(**defaults))

    def test_build_options_sets_empty_setting_sources(self):
        adapter = self._adapter()
        options = adapter._build_options(
            system_prompt=None,
            mcp_servers=None,
            config=AgentConfig(),
            tools=None,
        )
        assert options.setting_sources == [], (
            "setting_sources must be [] to suppress loading of personal MCP servers "
            "(see issues/089-claude-agent-sdk-subprocess-incompatible-with-sglang-vllm-"
            "anthropic-endpoint.md)"
        )

    def test_build_options_propagates_claude_config_dir(self, monkeypatch, tmp_path):
        fake_cfg = tmp_path / "fake-config"
        fake_cfg.mkdir()
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(fake_cfg))

        adapter = self._adapter()
        options = adapter._build_options(
            system_prompt=None,
            mcp_servers=None,
            config=AgentConfig(),
            tools=None,
        )
        assert options.env is not None
        assert options.env.get("CLAUDE_CONFIG_DIR") == str(fake_cfg)
        assert options.setting_sources == [], (
            "setting_sources must remain [] when CLAUDE_CONFIG_DIR is set; issue 089 requires both simultaneously"
        )

    def test_build_options_no_claude_config_dir_when_parent_unset(self, monkeypatch):
        monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)

        adapter = self._adapter()
        options = adapter._build_options(
            system_prompt=None,
            mcp_servers=None,
            config=AgentConfig(),
            tools=None,
        )
        # env may still exist for ANTHROPIC_* keys but should not fabricate CLAUDE_CONFIG_DIR
        env = getattr(options, "env", None) or {}
        assert env.get("CLAUDE_CONFIG_DIR") is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_arun_returns_partial_trace_when_timeout_has_messages(monkeypatch):
    class TextBlock:
        def __init__(self, text):
            self.text = text

    class ThinkingBlock:
        pass

    class ToolUseBlock:
        pass

    class ToolResultBlock:
        pass

    class UserMessage:
        def __init__(self, content):
            self.content = content

    class AssistantMessage:
        def __init__(self, content, model="qwen3.5-122b-a10b"):
            self.content = content
            self.model = model

    class ResultMessage:
        pass

    class ClaudeAgentOptions:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class ClaudeSDKClient:
        def __init__(self, _options):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def query(self, _prompt):
            return None

        async def receive_response(self):
            yield AssistantMessage([TextBlock("partial answer")])
            await asyncio.sleep(10)

    fake_sdk = types.ModuleType("claude_agent_sdk")
    fake_sdk.ClaudeAgentOptions = ClaudeAgentOptions
    fake_sdk.ClaudeSDKClient = ClaudeSDKClient
    fake_sdk.AssistantMessage = AssistantMessage
    fake_sdk.UserMessage = UserMessage
    fake_sdk.ResultMessage = ResultMessage

    fake_types = types.ModuleType("claude_agent_sdk.types")
    fake_types.TextBlock = TextBlock
    fake_types.ThinkingBlock = ThinkingBlock
    fake_types.ToolUseBlock = ToolUseBlock
    fake_types.ToolResultBlock = ToolResultBlock

    monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake_sdk)
    monkeypatch.setitem(sys.modules, "claude_agent_sdk.types", fake_types)

    config = ModelConfig(
        id="test",
        model_name="qwen3.5-122b-a10b",
        interface="claude_agent_sdk",
    )
    adapter = ClaudeSDKAgentAdapter(config)

    result = await adapter.arun(
        [Message.user("run a long task")],
        config=AgentConfig(timeout=0.01),
    )

    assert result.timeout_reached is True
    assert result.limit_reached is False
    assert "partial answer" in result.raw_trace
    assert "[Note: Agent timed out - partial trace shown]" in result.raw_trace
    assert result.final_response == "partial answer"
    assert result.trace_messages
