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

    def test_build_options_no_cwd_when_workspace_path_is_none(self):
        adapter = self._make_adapter()
        config = AgentConfig()
        options = adapter._build_options(
            system_prompt="test",
            mcp_servers=None,
            config=config,
        )
        assert options.cwd is None
