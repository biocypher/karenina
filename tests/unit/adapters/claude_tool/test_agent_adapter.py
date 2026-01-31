"""Tests for ClaudeToolAgentAdapter.

Tests agent adapter functionality for multi-turn tool execution.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from karenina.adapters.claude_tool import ClaudeToolAgentAdapter
from karenina.ports import (
    AgentConfig,
    AgentExecutionError,
    AgentResponseError,
    AgentResult,
    AgentTimeoutError,
    Message,
    Role,
    Tool,
)
from karenina.ports.messages import TextContent


class MockTextBlock:
    """Mock text content block."""

    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class MockToolUseBlock:
    """Mock tool use content block."""

    def __init__(self, id: str, name: str, input: dict[str, Any]) -> None:
        self.type = "tool_use"
        self.id = id
        self.name = name
        self.input = input


class MockUsage:
    """Mock usage object."""

    def __init__(self, input_tokens: int = 100, output_tokens: int = 50) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class MockResponse:
    """Mock Anthropic API response."""

    def __init__(
        self,
        content: list[Any] | None = None,
        usage: MockUsage | None = None,
        role: str = "assistant",
        model: str = "claude-haiku-4-5",
    ) -> None:
        self.content = content or [MockTextBlock("Hello!")]
        self.usage = usage or MockUsage()
        self.role = role
        self.model = model


@pytest.fixture
def model_config() -> Any:
    """Create a mock ModelConfig for claude_tool interface."""
    from karenina.schemas.workflow.models import ModelConfig

    return ModelConfig(
        id="test-claude-tool",
        model_name="claude-haiku-4-5",
        model_provider="anthropic",
        interface="claude_tool",
        max_tokens=1024,
    )


class TestAgentAdapterInitialization:
    """Tests for agent adapter initialization."""

    def test_initialization_stores_config(self, model_config: Any) -> None:
        """Test initialization stores config."""
        adapter = ClaudeToolAgentAdapter(model_config)

        assert adapter._config == model_config

    def test_initialization_lazy_client(self, model_config: Any) -> None:
        """Test async client is lazily initialized."""
        adapter = ClaudeToolAgentAdapter(model_config)

        assert adapter._async_client is None


class TestAgentAdapterClientManagement:
    """Tests for agent adapter client management."""

    def test_get_async_client_creates_client(self, model_config: Any) -> None:
        """Test _get_async_client creates async Anthropic client."""
        adapter = ClaudeToolAgentAdapter(model_config)

        with patch("anthropic.AsyncAnthropic") as mock_async:
            mock_instance = MagicMock()
            mock_async.return_value = mock_instance

            client = adapter._get_async_client()

            mock_async.assert_called_once()
            assert client == mock_instance

    def test_get_async_client_caches_client(self, model_config: Any) -> None:
        """Test _get_async_client caches client instance."""
        adapter = ClaudeToolAgentAdapter(model_config)

        with patch("anthropic.AsyncAnthropic") as mock_async:
            mock_instance = MagicMock()
            mock_async.return_value = mock_instance

            client1 = adapter._get_async_client()
            client2 = adapter._get_async_client()

            assert client1 is client2
            mock_async.assert_called_once()


class TestAgentAdapterTraceBuild:
    """Tests for trace building functionality."""

    def test_build_raw_trace_from_assistant_messages(self) -> None:
        """Test claude_tool_messages_to_raw_trace from assistant messages."""
        from karenina.adapters.claude_tool.trace import claude_tool_messages_to_raw_trace

        messages = [
            Message(role=Role.ASSISTANT, content=[TextContent(text="Hello there!")]),
        ]

        trace = claude_tool_messages_to_raw_trace(messages)

        assert "--- AI Message ---" in trace
        assert "Hello there!" in trace

    def test_build_raw_trace_skips_user_messages(self) -> None:
        """Test claude_tool_messages_to_raw_trace skips user messages."""
        from karenina.adapters.claude_tool.trace import claude_tool_messages_to_raw_trace

        messages = [
            Message(role=Role.USER, content=[TextContent(text="User question")]),
            Message(role=Role.ASSISTANT, content=[TextContent(text="Response")]),
        ]

        trace = claude_tool_messages_to_raw_trace(messages)

        assert "User question" not in trace
        assert "Response" in trace

    def test_extract_final_response_from_last_assistant(self, model_config: Any) -> None:
        """Test _extract_final_response gets last assistant text."""
        adapter = ClaudeToolAgentAdapter(model_config)

        messages = [
            Message(role=Role.ASSISTANT, content=[TextContent(text="First response")]),
            Message(role=Role.ASSISTANT, content=[TextContent(text="Final response")]),
        ]

        response = adapter._extract_final_response(messages)

        assert response == "Final response"

    def test_extract_final_response_no_assistant(self, model_config: Any) -> None:
        """Test _extract_final_response with no assistant messages."""
        adapter = ClaudeToolAgentAdapter(model_config)

        messages = [
            Message(role=Role.USER, content=[TextContent(text="Just user")]),
        ]

        response = adapter._extract_final_response(messages)

        assert response == "[No final response extracted]"


class MockAsyncIterator:
    """Helper class to create a proper async iterator for mocking tool_runner."""

    def __init__(self, items: list[Any]) -> None:
        self.items = items
        self.index = 0

    def __aiter__(self) -> MockAsyncIterator:
        return self

    async def __anext__(self) -> Any:
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item


class TestAgentAdapterRun:
    """Tests for agent run functionality."""

    @pytest.mark.asyncio
    async def test_run_returns_agent_result(self, model_config: Any) -> None:
        """Test run returns AgentResult."""
        adapter = ClaudeToolAgentAdapter(model_config)

        mock_response = MockResponse(content=[MockTextBlock("Hello!")])
        mock_iterator = MockAsyncIterator([mock_response])

        mock_client = MagicMock()
        mock_client.beta.messages.tool_runner = MagicMock(return_value=mock_iterator)

        with patch.object(adapter, "_get_async_client", return_value=mock_client):
            result = await adapter.run(
                messages=[Message.user("Hello")],
            )

            assert isinstance(result, AgentResult)
            assert result.final_response is not None
            assert result.raw_trace is not None
            assert result.trace_messages is not None

    @pytest.mark.asyncio
    async def test_run_uses_tool_runner(self, model_config: Any) -> None:
        """Test run uses tool_runner for execution."""
        adapter = ClaudeToolAgentAdapter(model_config)

        mock_response = MockResponse()
        mock_iterator = MockAsyncIterator([mock_response])

        mock_client = MagicMock()
        mock_client.beta.messages.tool_runner = MagicMock(return_value=mock_iterator)

        with patch.object(adapter, "_get_async_client", return_value=mock_client):
            await adapter.run(messages=[Message.user("Hello")])

            mock_client.beta.messages.tool_runner.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_passes_model_config(self, model_config: Any) -> None:
        """Test run passes model configuration."""
        adapter = ClaudeToolAgentAdapter(model_config)

        mock_response = MockResponse()
        mock_iterator = MockAsyncIterator([mock_response])

        mock_client = MagicMock()
        mock_client.beta.messages.tool_runner = MagicMock(return_value=mock_iterator)

        with patch.object(adapter, "_get_async_client", return_value=mock_client):
            await adapter.run(messages=[Message.user("Hello")])

            call_kwargs = mock_client.beta.messages.tool_runner.call_args.kwargs
            assert call_kwargs["model"] == "claude-haiku-4-5"
            assert call_kwargs["max_tokens"] == 1024

    @pytest.mark.asyncio
    async def test_run_wraps_timeout_as_agent_timeout_error(self, model_config: Any) -> None:
        """Test run wraps TimeoutError as AgentTimeoutError."""
        adapter = ClaudeToolAgentAdapter(model_config)

        with patch.object(adapter, "_execute_agent_loop") as mock_execute:
            mock_execute.side_effect = TimeoutError("Timed out")

            with pytest.raises(AgentTimeoutError):
                await adapter.run(
                    messages=[Message.user("Hello")],
                    config=AgentConfig(timeout=30),
                )

    @pytest.mark.asyncio
    async def test_run_wraps_exception_as_agent_execution_error(self, model_config: Any) -> None:
        """Test run wraps general exceptions as AgentExecutionError."""
        adapter = ClaudeToolAgentAdapter(model_config)

        with patch.object(adapter, "_execute_agent_loop") as mock_execute:
            mock_execute.side_effect = RuntimeError("Something broke")

            with pytest.raises(AgentExecutionError, match="Something broke"):
                await adapter.run(messages=[Message.user("Hello")])

    def test_model_config_requires_model_name(self) -> None:
        """Test ModelConfig validates model_name is required for claude_tool.

        Note: This validation happens at the Pydantic level in ModelConfig,
        not in the adapter itself.
        """
        from pydantic import ValidationError

        from karenina.schemas.workflow.models import ModelConfig

        with pytest.raises(ValidationError, match="model_name is required"):
            ModelConfig(
                id="test",
                model_name=None,
                model_provider="anthropic",
                interface="claude_tool",
            )


class TestAgentAdapterWithTools:
    """Tests for agent adapter with tools."""

    @pytest.mark.asyncio
    async def test_run_wraps_static_tools(self, model_config: Any) -> None:
        """Test run wraps static tools for tool_runner."""
        adapter = ClaudeToolAgentAdapter(model_config)

        tool = Tool(
            name="my_tool",
            description="A tool",
            input_schema={"type": "object", "properties": {}},
        )

        mock_response = MockResponse()
        mock_iterator = MockAsyncIterator([mock_response])

        mock_client = MagicMock()
        mock_client.beta.messages.tool_runner = MagicMock(return_value=mock_iterator)

        with (
            patch.object(adapter, "_get_async_client", return_value=mock_client),
            patch("karenina.adapters.claude_tool.agent.wrap_static_tool") as mock_wrap,
        ):
            mock_wrapped = MagicMock()
            mock_wrap.return_value = mock_wrapped

            await adapter.run(
                messages=[Message.user("Hello")],
                tools=[tool],
            )

            mock_wrap.assert_called_once_with(tool)


class TestAgentAdapterWithMCPServers:
    """Tests for agent adapter with MCP servers."""

    @pytest.mark.asyncio
    async def test_run_connects_to_mcp_servers(self, model_config: Any) -> None:
        """Test run connects to MCP servers."""
        adapter = ClaudeToolAgentAdapter(model_config)

        mcp_servers = {
            "test_server": {
                "type": "http",
                "url": "https://mcp.example.com/mcp",
            }
        }

        mock_response = MockResponse()
        mock_iterator = MockAsyncIterator([mock_response])

        mock_client = MagicMock()
        mock_client.beta.messages.tool_runner = MagicMock(return_value=mock_iterator)

        with (
            patch.object(adapter, "_get_async_client", return_value=mock_client),
            patch("karenina.adapters.claude_tool.agent.connect_all_mcp_servers") as mock_connect,
            patch("karenina.adapters.claude_tool.agent.get_all_mcp_tools") as mock_get_tools,
        ):
            mock_connect.return_value = {"test_server": MagicMock()}
            mock_get_tools.return_value = []

            await adapter.run(
                messages=[Message.user("Hello")],
                mcp_servers=mcp_servers,
            )

            mock_connect.assert_called_once()


class TestAgentAdapterExecuteLoop:
    """Tests for agent loop execution."""

    @pytest.mark.asyncio
    async def test_execute_loop_collects_messages(self, model_config: Any) -> None:
        """Test _execute_agent_loop collects messages from responses."""
        adapter = ClaudeToolAgentAdapter(model_config)

        mock_response1 = MockResponse(content=[MockTextBlock("First")])
        mock_response2 = MockResponse(content=[MockTextBlock("Second")])
        mock_iterator = MockAsyncIterator([mock_response1, mock_response2])

        mock_client = MagicMock()
        mock_client.beta.messages.tool_runner = MagicMock(return_value=mock_iterator)

        with patch.object(adapter, "_get_async_client", return_value=mock_client):
            result = await adapter._execute_agent_loop(
                messages=[Message.user("Hello")],
                tools=[],
                config=AgentConfig(),
            )

            assert len(result.trace_messages) == 2
            assert result.turns == 2

    @pytest.mark.asyncio
    async def test_execute_loop_raises_on_empty_response(self, model_config: Any) -> None:
        """Test _execute_agent_loop raises when no messages received."""
        adapter = ClaudeToolAgentAdapter(model_config)

        mock_iterator = MockAsyncIterator([])  # Empty iterator

        mock_client = MagicMock()
        mock_client.beta.messages.tool_runner = MagicMock(return_value=mock_iterator)

        with (
            patch.object(adapter, "_get_async_client", return_value=mock_client),
            pytest.raises(AgentResponseError, match="No messages received"),
        ):
            await adapter._execute_agent_loop(
                messages=[Message.user("Hello")],
                tools=[],
                config=AgentConfig(),
            )

    @pytest.mark.asyncio
    async def test_execute_loop_respects_max_turns(self, model_config: Any) -> None:
        """Test _execute_agent_loop respects max_turns limit."""
        adapter = ClaudeToolAgentAdapter(model_config)

        responses = [MockResponse() for _ in range(10)]
        mock_iterator = MockAsyncIterator(responses)

        mock_client = MagicMock()
        mock_client.beta.messages.tool_runner = MagicMock(return_value=mock_iterator)

        with patch.object(adapter, "_get_async_client", return_value=mock_client):
            result = await adapter._execute_agent_loop(
                messages=[Message.user("Hello")],
                tools=[],
                config=AgentConfig(max_turns=3),
            )

            assert result.turns == 3
            assert result.limit_reached is True


class TestAgentAdapterSync:
    """Tests for synchronous agent execution."""

    def test_run_sync_exists(self, model_config: Any) -> None:
        """Test run_sync method exists."""
        adapter = ClaudeToolAgentAdapter(model_config)

        assert hasattr(adapter, "run_sync")
        assert callable(adapter.run_sync)
