"""Tests for Claude Tool agent timeout recovery.

Verifies that the ClaudeToolAgentAdapter handles timeout scenarios correctly:
partial AgentResult on timeout with accumulated messages, normal completion
without timeout, and AgentTimeoutError when no messages were collected.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from karenina.adapters.claude_tool import ClaudeToolAgentAdapter
from karenina.ports import (
    AgentConfig,
    AgentResult,
    AgentTimeoutError,
    Message,
)


class MockTextBlock:
    """Mock Anthropic text content block."""

    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class MockUsage:
    """Mock Anthropic usage object."""

    def __init__(self, input_tokens: int = 100, output_tokens: int = 50) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class MockResponse:
    """Mock Anthropic API response yielded by tool_runner."""

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


class SlowAsyncIterator:
    """Async iterator that yields some items then blocks, simulating a timeout.

    Yields ``fast_items`` immediately, then sleeps for ``slow_delay`` seconds
    before yielding ``slow_items``. When wrapped in ``asyncio.wait_for``
    with a short timeout, the fast items are collected and the slow ones
    trigger ``TimeoutError``.
    """

    def __init__(
        self,
        fast_items: list[Any],
        slow_items: list[Any] | None = None,
        slow_delay: float = 10.0,
    ) -> None:
        self.fast_items = fast_items
        self.slow_items = slow_items or []
        self.slow_delay = slow_delay
        self._items = list(fast_items)
        self._slow_items = list(slow_items or [])
        self._index = 0
        self._in_slow = False

    def __aiter__(self) -> SlowAsyncIterator:
        return self

    async def __anext__(self) -> Any:
        if self._index < len(self._items):
            item = self._items[self._index]
            self._index += 1
            return item

        if not self._in_slow:
            self._in_slow = True
            self._slow_index = 0

        if self._slow_index < len(self._slow_items):
            await asyncio.sleep(self.slow_delay)
            item = self._slow_items[self._slow_index]
            self._slow_index += 1
            return item

        # No slow items: just block
        await asyncio.sleep(self.slow_delay)
        raise StopAsyncIteration


class FastAsyncIterator:
    """Async iterator that yields all items immediately."""

    def __init__(self, items: list[Any]) -> None:
        self.items = items
        self.index = 0

    def __aiter__(self) -> FastAsyncIterator:
        return self

    async def __anext__(self) -> Any:
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item


@pytest.fixture
def model_config() -> Any:
    """Create a ModelConfig for claude_tool interface."""
    from karenina.schemas.config import ModelConfig

    return ModelConfig(
        id="test-claude-tool-timeout",
        model_name="claude-haiku-4-5",
        model_provider="anthropic",
        interface="claude_tool",
        max_tokens=1024,
    )


@pytest.mark.unit
class TestClaudeToolAgentTimeoutWithMessages:
    """Test timeout recovery when the agent has accumulated messages."""

    @pytest.mark.asyncio
    async def test_timeout_returns_partial_result(self, model_config: Any) -> None:
        """Verify that a timeout with accumulated messages returns a partial AgentResult."""
        adapter = ClaudeToolAgentAdapter(model_config)

        fast_responses = [
            MockResponse(content=[MockTextBlock("Turn 1 response")]),
            MockResponse(content=[MockTextBlock("Turn 2 response")]),
        ]
        # After yielding 2 fast responses, the iterator blocks until timeout
        mock_iterator = SlowAsyncIterator(fast_items=fast_responses)

        mock_client = MagicMock()
        mock_client.beta.messages.tool_runner = MagicMock(return_value=mock_iterator)

        sentinel_tool = MagicMock()
        with patch.object(adapter, "_get_async_client", return_value=mock_client):
            result = await adapter._execute_agent_loop(
                messages=[Message.user("Hello")],
                tools=[sentinel_tool],
                config=AgentConfig(timeout=0.05),
            )

        assert isinstance(result, AgentResult)
        assert result.timeout_reached is True
        assert result.turns == 2
        assert len(result.trace_messages) == 2
        assert "Turn 1 response" in result.raw_trace
        assert "Turn 2 response" in result.raw_trace
        assert "[Note: Agent timed out" in result.raw_trace

    @pytest.mark.asyncio
    async def test_timeout_preserves_usage(self, model_config: Any) -> None:
        """Verify that usage metadata is accumulated even on timeout."""
        adapter = ClaudeToolAgentAdapter(model_config)

        fast_responses = [
            MockResponse(
                content=[MockTextBlock("First")],
                usage=MockUsage(input_tokens=100, output_tokens=50),
            ),
            MockResponse(
                content=[MockTextBlock("Second")],
                usage=MockUsage(input_tokens=80, output_tokens=40),
            ),
        ]
        mock_iterator = SlowAsyncIterator(fast_items=fast_responses)

        mock_client = MagicMock()
        mock_client.beta.messages.tool_runner = MagicMock(return_value=mock_iterator)

        sentinel_tool = MagicMock()
        with patch.object(adapter, "_get_async_client", return_value=mock_client):
            result = await adapter._execute_agent_loop(
                messages=[Message.user("Hello")],
                tools=[sentinel_tool],
                config=AgentConfig(timeout=0.05),
            )

        assert result.timeout_reached is True
        assert result.usage.input_tokens > 0
        assert result.usage.output_tokens > 0

    @pytest.mark.asyncio
    async def test_timeout_extracts_final_response(self, model_config: Any) -> None:
        """Verify that the final response is extracted from accumulated messages on timeout."""
        adapter = ClaudeToolAgentAdapter(model_config)

        fast_responses = [
            MockResponse(content=[MockTextBlock("Partial answer so far")]),
        ]
        mock_iterator = SlowAsyncIterator(fast_items=fast_responses)

        mock_client = MagicMock()
        mock_client.beta.messages.tool_runner = MagicMock(return_value=mock_iterator)

        sentinel_tool = MagicMock()
        with patch.object(adapter, "_get_async_client", return_value=mock_client):
            result = await adapter._execute_agent_loop(
                messages=[Message.user("Hello")],
                tools=[sentinel_tool],
                config=AgentConfig(timeout=0.05),
            )

        assert result.timeout_reached is True
        assert result.final_response == "Partial answer so far"


@pytest.mark.unit
class TestClaudeToolAgentNormalCompletion:
    """Test normal completion (no timeout)."""

    @pytest.mark.asyncio
    async def test_normal_completion_not_timed_out(self, model_config: Any) -> None:
        """Verify that normal completion sets timeout_reached=False."""
        adapter = ClaudeToolAgentAdapter(model_config)

        responses = [
            MockResponse(content=[MockTextBlock("Step 1")]),
            MockResponse(content=[MockTextBlock("Final answer")]),
        ]
        mock_iterator = FastAsyncIterator(responses)

        mock_client = MagicMock()
        mock_client.beta.messages.tool_runner = MagicMock(return_value=mock_iterator)

        sentinel_tool = MagicMock()
        with patch.object(adapter, "_get_async_client", return_value=mock_client):
            result = await adapter._execute_agent_loop(
                messages=[Message.user("Hello")],
                tools=[sentinel_tool],
                config=AgentConfig(),
            )

        assert result.timeout_reached is False
        assert result.turns == 2
        assert result.limit_reached is False
        assert result.final_response == "Final answer"

    @pytest.mark.asyncio
    async def test_normal_completion_with_timeout_config(self, model_config: Any) -> None:
        """Verify normal completion with a timeout configured but not hit."""
        adapter = ClaudeToolAgentAdapter(model_config)

        responses = [MockResponse(content=[MockTextBlock("Done")])]
        mock_iterator = FastAsyncIterator(responses)

        mock_client = MagicMock()
        mock_client.beta.messages.tool_runner = MagicMock(return_value=mock_iterator)

        sentinel_tool = MagicMock()
        with patch.object(adapter, "_get_async_client", return_value=mock_client):
            result = await adapter._execute_agent_loop(
                messages=[Message.user("Hello")],
                tools=[sentinel_tool],
                config=AgentConfig(timeout=60.0),
            )

        assert result.timeout_reached is False
        assert result.turns == 1
        assert "[Note: Agent timed out" not in result.raw_trace


@pytest.mark.unit
class TestClaudeToolAgentTimeoutNoMessages:
    """Test timeout when no messages have been accumulated."""

    @pytest.mark.asyncio
    async def test_timeout_with_no_messages_raises(self, model_config: Any) -> None:
        """Verify that timeout with zero accumulated messages raises AgentTimeoutError."""
        adapter = ClaudeToolAgentAdapter(model_config)

        # Iterator that immediately blocks (no fast items)
        mock_iterator = SlowAsyncIterator(fast_items=[])

        mock_client = MagicMock()
        mock_client.beta.messages.tool_runner = MagicMock(return_value=mock_iterator)

        sentinel_tool = MagicMock()
        with (
            patch.object(adapter, "_get_async_client", return_value=mock_client),
            pytest.raises(AgentTimeoutError, match="timed out.*with no messages"),
        ):
            await adapter._execute_agent_loop(
                messages=[Message.user("Hello")],
                tools=[sentinel_tool],
                config=AgentConfig(timeout=0.05),
            )
