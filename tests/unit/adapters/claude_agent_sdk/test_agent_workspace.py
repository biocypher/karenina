"""Tests for workspace_path to cwd wiring in ClaudeSDKAgentAdapter."""

import sys
import types
from pathlib import Path

import pytest

from karenina.adapters.claude_agent_sdk.agent import ClaudeSDKAgentAdapter
from karenina.ports import AgentConfig
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
