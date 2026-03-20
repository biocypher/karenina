"""Conformance tests for workspace filesystem access.

Validates that adapters marked as natively_agentic correctly configure
real filesystem access when workspace_path is provided. This catches
the critical bug where a virtual/in-memory backend makes the agent
unable to see real files on disk.

This test exists because:
- Natively agentic adapters have built-in filesystem tools (ls, glob, read_file)
- If these tools use a virtual backend, they return empty results for real paths
- Simple tests ("What is the capital of France?") pass fine without filesystem
- The bug only surfaces when the agent needs to read workspace files
- This is a silent failure: the agent generates synthetic data instead of erroring
"""

from __future__ import annotations

import pytest

from karenina.adapters.registry import AdapterRegistry


@pytest.mark.unit
class TestWorkspaceAccessConformance:
    """Verify that natively agentic adapters can access real workspace files."""

    def test_natively_agentic_adapters_must_not_use_virtual_backend(
        self, all_registered_interfaces, mock_model_config_for_interface
    ):
        """Natively agentic adapters must configure real filesystem access.

        This test verifies that the adapter does not use a virtual/in-memory
        backend that would make the agent blind to real files on disk.

        The test is necessarily adapter-specific because different SDKs use
        different backend abstractions. We check the known problematic ones.
        """
        for interface in all_registered_interfaces:
            spec = AdapterRegistry.get_spec(interface)
            if not spec.natively_agentic:
                continue

            # For langchain_deep_agents: verify FilesystemBackend is used
            if interface == "langchain_deep_agents":
                self._verify_deep_agents_backend(mock_model_config_for_interface(interface))

    def _verify_deep_agents_backend(self, model_config):
        """Verify Deep Agents adapter uses FilesystemBackend, not StateBackend.

        Captures the kwargs passed to create_deep_agent to inspect the backend.
        """
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        try:
            from deepagents.backends import FilesystemBackend, StateBackend
        except ImportError:
            pytest.skip("deepagents not installed")

        from langchain_core.messages import AIMessage

        from karenina.adapters.langchain_deep_agents import agent as agent_module
        from karenina.adapters.langchain_deep_agents.agent import DeepAgentsAgentAdapter
        from karenina.ports import AgentConfig, Message

        captured_kwargs: dict = {}
        original_create = agent_module._create_deep_agent

        def capture_create(**kwargs):
            captured_kwargs.update(kwargs)
            mock_agent = MagicMock()
            mock_agent.ainvoke = AsyncMock(
                return_value={
                    "messages": [AIMessage(content="ok")],
                    "is_last_step": False,
                }
            )
            return mock_agent

        agent_module._create_deep_agent = capture_create
        try:
            adapter = DeepAgentsAgentAdapter(model_config)
            asyncio.run(
                adapter.arun(
                    messages=[Message.user("test")],
                    config=AgentConfig(max_turns=2),
                )
            )

            backend = captured_kwargs.get("backend")
            assert backend is not None, "create_deep_agent must receive a 'backend' kwarg"
            assert isinstance(backend, FilesystemBackend), (
                f"Expected FilesystemBackend, got {type(backend).__name__}. "
                "StateBackend would make the agent unable to see real files."
            )
            assert not isinstance(backend, StateBackend), "StateBackend (virtual/in-memory) must never be used."
        finally:
            agent_module._create_deep_agent = original_create

    def test_workspace_path_is_accepted_by_agent_config(self):
        """AgentConfig must accept workspace_path parameter."""
        from pathlib import Path

        from karenina.ports import AgentConfig

        config = AgentConfig(workspace_path=Path("/tmp/test_workspace"))
        assert config.workspace_path == Path("/tmp/test_workspace")

    def test_workspace_path_none_is_valid(self):
        """AgentConfig with no workspace_path must also work."""
        from karenina.ports import AgentConfig

        config = AgentConfig()
        assert config.workspace_path is None
