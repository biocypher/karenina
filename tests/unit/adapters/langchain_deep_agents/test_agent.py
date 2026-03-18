"""Tests for DeepAgentsAgentAdapter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from karenina.adapters.langchain_deep_agents.agent import DeepAgentsAgentAdapter
from karenina.ports import AgentConfig, AgentResult, Message


@pytest.mark.unit
class TestDeepAgentsBackendConfiguration:
    """Verify the adapter configures a real filesystem backend, not virtual state.

    This test class exists because a virtual/in-memory backend (StateBackend)
    causes the agent to see an empty filesystem: all ls/glob/read_file calls
    return empty results even when real files exist on disk. This is a critical
    correctness issue for benchmarking with workspace files.
    """

    @pytest.mark.asyncio
    async def test_default_uses_filesystem_backend(self, deep_agents_model_config, monkeypatch):
        """Without workspace_path, adapter must use FilesystemBackend (not StateBackend)."""
        from deepagents.backends import FilesystemBackend
        from langchain_core.messages import AIMessage

        captured_kwargs: dict = {}

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

        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.agent._create_deep_agent",
            capture_create,
        )
        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.agent.create_chat_model",
            lambda _config, **_kw: MagicMock(),
        )

        adapter = DeepAgentsAgentAdapter(deep_agents_model_config)
        await adapter.arun(messages=[Message.user("test")], config=AgentConfig(max_turns=2))

        assert "backend" in captured_kwargs, "backend kwarg must be passed to create_deep_agent"
        backend = captured_kwargs["backend"]
        assert isinstance(backend, FilesystemBackend), (
            f"Expected FilesystemBackend, got {type(backend).__name__}. "
            "StateBackend would make the agent unable to see real files on disk."
        )

    @pytest.mark.asyncio
    async def test_workspace_path_configures_rooted_backend(self, deep_agents_model_config, tmp_path, monkeypatch):
        """When workspace_path is set, FilesystemBackend must be rooted at that path."""
        from deepagents.backends import FilesystemBackend
        from langchain_core.messages import AIMessage

        captured_kwargs: dict = {}

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

        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.agent._create_deep_agent",
            capture_create,
        )
        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.agent.create_chat_model",
            lambda _config, **_kw: MagicMock(),
        )

        workspace = tmp_path / "my_workspace"
        workspace.mkdir()
        (workspace / "data.xlsx").write_text("fake data")

        adapter = DeepAgentsAgentAdapter(deep_agents_model_config)
        await adapter.arun(
            messages=[Message.user("analyze data.xlsx")],
            config=AgentConfig(max_turns=2, workspace_path=workspace),
        )

        backend = captured_kwargs["backend"]
        assert isinstance(backend, FilesystemBackend)
        # FilesystemBackend stores root_dir as 'cwd' attribute
        assert backend.cwd == workspace

    @pytest.mark.asyncio
    async def test_backend_is_never_state_backend(self, deep_agents_model_config, monkeypatch):
        """StateBackend must never be used; it makes the agent blind to real files."""
        from deepagents.backends import StateBackend
        from langchain_core.messages import AIMessage

        captured_kwargs: dict = {}

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

        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.agent._create_deep_agent",
            capture_create,
        )
        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.agent.create_chat_model",
            lambda _config, **_kw: MagicMock(),
        )

        adapter = DeepAgentsAgentAdapter(deep_agents_model_config)
        await adapter.arun(messages=[Message.user("test")], config=AgentConfig(max_turns=2))

        backend = captured_kwargs.get("backend")
        assert not isinstance(backend, StateBackend), (
            "StateBackend (virtual/in-memory) must never be used. "
            "The agent would see empty results for all filesystem operations."
        )


@pytest.mark.unit
class TestDeepAgentsAgentAdapter:
    def test_init_stores_config(self, deep_agents_model_config):
        adapter = DeepAgentsAgentAdapter(deep_agents_model_config)
        assert adapter._config == deep_agents_model_config

    @pytest.mark.asyncio
    async def test_arun_returns_agent_result(self, deep_agents_model_config, monkeypatch):
        """Test that arun produces a valid AgentResult with mocked Deep Agents."""
        from langchain_core.messages import AIMessage

        mock_result = {
            "messages": [AIMessage(content="The answer is 42.")],
            "is_last_step": False,
        }

        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(return_value=mock_result)

        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.agent._create_deep_agent",
            lambda **_kwargs: mock_agent,
        )
        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.agent.create_chat_model",
            lambda _config, **_kw: MagicMock(),
        )

        adapter = DeepAgentsAgentAdapter(deep_agents_model_config)
        result = await adapter.arun(
            messages=[Message.user("What is the meaning of life?")],
            config=AgentConfig(max_turns=5),
        )

        assert isinstance(result, AgentResult)
        assert "42" in result.final_response
        assert "--- AI Message ---" in result.raw_trace
        assert len(result.trace_messages) >= 1
        assert result.usage.input_tokens >= 0
        assert result.limit_reached is False
        assert result.turns == 1

    @pytest.mark.asyncio
    async def test_arun_detects_limit_reached(self, deep_agents_model_config, monkeypatch):
        """Test that is_last_step=True in result sets limit_reached."""
        from langchain_core.messages import AIMessage

        mock_result = {
            "messages": [AIMessage(content="Partial answer.")],
            "is_last_step": True,
        }

        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(return_value=mock_result)

        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.agent._create_deep_agent",
            lambda **_kwargs: mock_agent,
        )
        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.agent.create_chat_model",
            lambda _config, **_kw: MagicMock(),
        )

        adapter = DeepAgentsAgentAdapter(deep_agents_model_config)
        result = await adapter.arun(
            messages=[Message.user("Complex question")],
            config=AgentConfig(max_turns=2),
        )

        assert result.limit_reached is True

    @pytest.mark.asyncio
    async def test_arun_with_tool_calls(self, deep_agents_model_config, monkeypatch):
        """Test trace extraction with tool calls in the conversation."""
        from langchain_core.messages import AIMessage, ToolMessage

        mock_result = {
            "messages": [
                AIMessage(
                    content="Let me search.",
                    tool_calls=[{"name": "search", "args": {"q": "test"}, "id": "call_1"}],
                ),
                ToolMessage(content="search result", tool_call_id="call_1"),
                AIMessage(content="Based on the search, the answer is X."),
            ],
            "is_last_step": False,
        }

        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(return_value=mock_result)

        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.agent._create_deep_agent",
            lambda **_kwargs: mock_agent,
        )
        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.agent.create_chat_model",
            lambda _config, **_kw: MagicMock(),
        )

        adapter = DeepAgentsAgentAdapter(deep_agents_model_config)
        result = await adapter.arun(
            messages=[Message.user("Search for something")],
            config=AgentConfig(max_turns=10),
        )

        assert "answer is X" in result.final_response
        assert "--- Tool Call ---" in result.raw_trace
        assert "--- Tool Result ---" in result.raw_trace
        assert result.turns == 2  # Two AIMessages

    @pytest.mark.asyncio
    async def test_aclose_is_noop(self, deep_agents_model_config):
        """aclose() should not raise."""
        adapter = DeepAgentsAgentAdapter(deep_agents_model_config)
        await adapter.aclose()
