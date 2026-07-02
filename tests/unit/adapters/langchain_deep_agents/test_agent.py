"""Tests for DeepAgentsAgentAdapter."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from karenina.adapters.langchain_deep_agents.agent import DeepAgentsAgentAdapter
from karenina.benchmark.verification.executor import set_async_portal
from karenina.ports import AgentConfig, AgentResult, Message


def _mock_deep_agent(result):
    """Return a minimal DeepAgents-like object that streams one state."""

    class MockDeepAgent:
        async def astream(self, *_args, **_kwargs):
            yield result

    return MockDeepAgent()


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
            return _mock_deep_agent(
                {
                    "messages": [AIMessage(content="ok")],
                    "is_last_step": False,
                }
            )

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
            return _mock_deep_agent(
                {
                    "messages": [AIMessage(content="ok")],
                    "is_last_step": False,
                }
            )

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
    async def test_read_only_access_mode_wraps_backend(self, deep_agents_model_config, tmp_path, monkeypatch):
        """Read-only access mode should keep reads but block writes and execution."""
        from deepagents.backends import FilesystemBackend
        from langchain_core.messages import AIMessage

        from karenina.adapters.langchain_deep_agents.read_only_backend import ReadOnlyBackend

        captured_kwargs: dict = {}

        def capture_create(**kwargs):
            captured_kwargs.update(kwargs)
            return _mock_deep_agent(
                {
                    "messages": [AIMessage(content="ok")],
                    "is_last_step": False,
                }
            )

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
        (workspace / "results.txt").write_text("answer: 42")
        read_only_config = deep_agents_model_config.model_copy(
            update={"extra_kwargs": {"agent_runtime": {"access_mode": "read_only"}}}
        )

        adapter = DeepAgentsAgentAdapter(read_only_config)
        await adapter.arun(
            messages=[Message.user("inspect results")],
            config=AgentConfig(max_turns=2, workspace_path=workspace),
        )

        backend = captured_kwargs["backend"]
        assert isinstance(backend, ReadOnlyBackend)
        assert isinstance(backend.delegate, FilesystemBackend)
        assert backend.read("/results.txt") == "     1\tanswer: 42"
        assert backend.write("/new.txt", "content").error is not None
        assert backend.edit("/results.txt", "42", "43").error is not None
        assert (await backend.awrite("/async-new.txt", "content")).error is not None
        assert (await backend.aedit("/results.txt", "42", "43")).error is not None
        assert not hasattr(backend, "execute")

    @pytest.mark.asyncio
    async def test_docker_backend_runs_preflight_before_agent(self, deep_agents_model_config, tmp_path, monkeypatch):
        """Docker-backed command execution should fail fast when host Docker is unavailable."""
        from langchain_core.messages import AIMessage

        captured_preflight: dict[str, str | None] = {}
        captured_kwargs: dict = {}

        def capture_preflight(config, **_kwargs):
            captured_preflight["image"] = config.image
            captured_preflight["runtime"] = config.runtime

        def capture_create(**kwargs):
            captured_kwargs.update(kwargs)
            return _mock_deep_agent(
                {
                    "messages": [AIMessage(content="ok")],
                    "is_last_step": False,
                }
            )

        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.docker_backend.preflight_container_runtime",
            capture_preflight,
        )
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
        docker_config = deep_agents_model_config.model_copy(
            update={
                "extra_kwargs": {
                    "agent_runtime": {
                        "backend": "docker",
                        "docker_image": "karenina-bixbench:latest",
                    }
                }
            }
        )

        adapter = DeepAgentsAgentAdapter(docker_config)
        await adapter.arun(
            messages=[Message.user("analyze data")],
            config=AgentConfig(max_turns=2, workspace_path=workspace),
        )

        assert captured_preflight == {"image": "karenina-bixbench:latest", "runtime": "docker"}
        assert captured_kwargs["backend"].id.startswith("docker-")

    @pytest.mark.asyncio
    async def test_backend_is_never_state_backend(self, deep_agents_model_config, monkeypatch):
        """StateBackend must never be used; it makes the agent blind to real files."""
        from deepagents.backends import StateBackend
        from langchain_core.messages import AIMessage

        captured_kwargs: dict = {}

        def capture_create(**kwargs):
            captured_kwargs.update(kwargs)
            return _mock_deep_agent(
                {
                    "messages": [AIMessage(content="ok")],
                    "is_last_step": False,
                }
            )

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

        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.agent._create_deep_agent",
            lambda **_kwargs: _mock_deep_agent(mock_result),
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

        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.agent._create_deep_agent",
            lambda **_kwargs: _mock_deep_agent(mock_result),
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

        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.agent._create_deep_agent",
            lambda **_kwargs: _mock_deep_agent(mock_result),
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
        # DA now emits the canonical inline-tool-calls + "Tool Message" format
        # shared with langchain / csdk / manual.
        assert "--- AI Message ---" in result.raw_trace
        assert "Tool Calls:" in result.raw_trace
        assert "--- Tool Message (call_id:" in result.raw_trace
        assert result.turns == 2  # Two AIMessages

    @pytest.mark.asyncio
    async def test_aclose_is_noop(self, deep_agents_model_config):
        """aclose() should not raise."""
        adapter = DeepAgentsAgentAdapter(deep_agents_model_config)
        await adapter.aclose()

    def test_run_uses_fresh_loop_even_when_portal_is_active(self, deep_agents_model_config, monkeypatch):
        """The sync agent-loop path must not run DeepAgents on the shared batch portal."""
        from langchain_core.messages import AIMessage

        class FailingPortal:
            def call(self, *_args, **_kwargs):
                raise AssertionError("DeepAgentsAgentAdapter.run() should not use the shared portal")

        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.agent._create_deep_agent",
            lambda **_kwargs: _mock_deep_agent(
                {
                    "messages": [AIMessage(content="ok")],
                    "is_last_step": False,
                }
            ),
        )
        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.agent.create_chat_model",
            lambda _config, **_kw: MagicMock(),
        )

        adapter = DeepAgentsAgentAdapter(deep_agents_model_config)
        set_async_portal(FailingPortal())
        try:
            result = adapter.run(
                messages=[Message.user("test")],
                config=AgentConfig(max_turns=2),
            )
        finally:
            set_async_portal(None)

        assert result.final_response == "ok"

    @pytest.mark.asyncio
    async def test_arun_enables_model_call_retries(self, deep_agents_model_config, monkeypatch):
        """Agent loops should pass retry budget to LangChain model calls."""
        from langchain_core.messages import AIMessage

        captured_model_kwargs: dict = {}

        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.agent._create_deep_agent",
            lambda **_kwargs: _mock_deep_agent(
                {
                    "messages": [AIMessage(content="ok")],
                    "is_last_step": False,
                }
            ),
        )

        def capture_chat_model(_config, **kwargs):
            captured_model_kwargs.update(kwargs)
            return MagicMock()

        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.agent.create_chat_model",
            capture_chat_model,
        )

        adapter = DeepAgentsAgentAdapter(deep_agents_model_config)
        await adapter.arun(
            messages=[Message.user("test")],
            config=AgentConfig(max_turns=2),
        )

        assert captured_model_kwargs["max_retries"] >= 1


@pytest.mark.unit
class TestDeepAgentsAgentCapabilities:
    def test_capabilities_returns_port_capabilities(self, deep_agents_model_config):
        """DeepAgentsAgentAdapter should expose a capabilities property."""
        from karenina.ports.capabilities import PortCapabilities

        adapter = DeepAgentsAgentAdapter(deep_agents_model_config)
        caps = adapter.capabilities
        assert isinstance(caps, PortCapabilities)
        assert caps.supports_system_prompt is True


@pytest.mark.unit
class TestDeepAgentsMCPToolsWiring:
    """Test that arun() passes tools and MCP-derived tools to the agent."""

    @pytest.mark.asyncio
    async def test_explicit_tools_passed_to_agent(self, deep_agents_model_config, monkeypatch):
        """Explicit tools should be forwarded to create_deep_agent."""
        from langchain_core.messages import AIMessage

        from karenina.ports import AgentConfig, Message, Tool
        from karenina.ports.usage import UsageMetadata

        captured_kwargs = {}

        def mock_create_deep_agent(**kwargs):
            captured_kwargs.update(kwargs)
            return _mock_deep_agent(
                {
                    "messages": [AIMessage(content="Done")],
                    "is_last_step": False,
                }
            )

        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.agent._create_deep_agent",
            mock_create_deep_agent,
        )
        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.agent.deep_agents_messages_to_raw_trace",
            lambda _msgs: "trace",
        )
        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.agent.extract_deep_agents_usage",
            lambda _msgs, model: UsageMetadata(model=model),
        )
        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.agent.extract_actual_model",
            lambda _msgs: None,
        )

        adapter = DeepAgentsAgentAdapter(deep_agents_model_config)
        test_tool = Tool(name="test_tool", description="A test tool", input_schema={})

        await adapter.arun(
            messages=[Message.user("Hello")],
            tools=[test_tool],
            config=AgentConfig(max_turns=5),
        )

        assert "tools" in captured_kwargs
        assert len(captured_kwargs["tools"]) >= 1

    @pytest.mark.asyncio
    async def test_mcp_tools_loaded_and_combined(self, deep_agents_model_config, monkeypatch):
        """MCP servers should be converted to tools and combined with explicit tools."""
        from langchain_core.messages import AIMessage

        from karenina.ports import AgentConfig, Message, Tool
        from karenina.ports.usage import UsageMetadata

        captured_kwargs = {}
        fake_mcp_tool = MagicMock()
        fake_mcp_tool.name = "mcp_tool"

        def mock_create_deep_agent(**kwargs):
            captured_kwargs.update(kwargs)
            return _mock_deep_agent(
                {
                    "messages": [AIMessage(content="Done")],
                    "is_last_step": False,
                }
            )

        async def mock_convert_mcp(servers, exit_stack):
            return [fake_mcp_tool]

        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.agent._create_deep_agent",
            mock_create_deep_agent,
        )
        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.mcp.convert_mcp_to_tools",
            mock_convert_mcp,
        )
        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.agent.deep_agents_messages_to_raw_trace",
            lambda _msgs: "trace",
        )
        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.agent.extract_deep_agents_usage",
            lambda _msgs, model: UsageMetadata(model=model),
        )
        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.agent.extract_actual_model",
            lambda _msgs: None,
        )

        adapter = DeepAgentsAgentAdapter(deep_agents_model_config)
        test_tool = Tool(name="explicit_tool", description="Explicit", input_schema={})

        await adapter.arun(
            messages=[Message.user("Hello")],
            tools=[test_tool],
            mcp_servers={"server1": {"type": "stdio", "command": "echo"}},
            config=AgentConfig(max_turns=5),
        )

        assert "tools" in captured_kwargs
        tool_names = [t.name if hasattr(t, "name") else str(t) for t in captured_kwargs["tools"]]
        assert "mcp_tool" in tool_names

    @pytest.mark.asyncio
    async def test_no_tools_kwarg_when_none_provided(self, deep_agents_model_config, monkeypatch):
        """When neither tools nor mcp_servers are given, tools kwarg should be absent."""
        from langchain_core.messages import AIMessage

        from karenina.ports import AgentConfig, Message
        from karenina.ports.usage import UsageMetadata

        captured_kwargs = {}

        def mock_create_deep_agent(**kwargs):
            captured_kwargs.update(kwargs)
            return _mock_deep_agent(
                {
                    "messages": [AIMessage(content="Done")],
                    "is_last_step": False,
                }
            )

        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.agent._create_deep_agent",
            mock_create_deep_agent,
        )
        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.agent.deep_agents_messages_to_raw_trace",
            lambda _msgs: "trace",
        )
        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.agent.extract_deep_agents_usage",
            lambda _msgs, model: UsageMetadata(model=model),
        )
        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.agent.extract_actual_model",
            lambda _msgs: None,
        )

        adapter = DeepAgentsAgentAdapter(deep_agents_model_config)
        await adapter.arun(
            messages=[Message.user("Hello")],
            config=AgentConfig(max_turns=5),
        )

        assert "tools" not in captured_kwargs
