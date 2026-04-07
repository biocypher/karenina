"""Tests for ClaudeToolLLMAdapter streaming methods.

Tests astream() and _astream_with_timeout() using mocked Anthropic client.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from karenina.adapters.claude_tool import ClaudeToolLLMAdapter
from karenina.ports import Message


class _MockUsage:
    """Mock Anthropic usage object."""

    def __init__(self, input_tokens: int = 100, output_tokens: int = 50) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = None
        self.cache_creation_input_tokens = None


class _MockFinalMessage:
    """Mock final message returned by stream.get_final_message()."""

    def __init__(self, usage: _MockUsage | None = None) -> None:
        self.usage = usage or _MockUsage()


@pytest.fixture
def model_config() -> Any:
    """Create a ModelConfig for claude_tool interface."""
    from karenina.schemas.config import ModelConfig

    return ModelConfig(
        id="test-stream",
        model_name="claude-haiku-4-5",
        model_provider="anthropic",
        interface="claude_tool",
        max_tokens=1024,
    )


def _make_fast_stream(chunks: list[str], final_message: _MockFinalMessage | None = None) -> MagicMock:
    """Build a mock Anthropic stream context manager that yields chunks quickly.

    Args:
        chunks: Text chunks to yield from text_stream.
        final_message: Final message for usage extraction.
    """
    if final_message is None:
        final_message = _MockFinalMessage()

    async def _text_stream() -> Any:
        for chunk in chunks:
            yield chunk

    stream = MagicMock()
    stream.text_stream = _text_stream()
    stream.get_final_message = AsyncMock(return_value=final_message)

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=stream)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


def _make_slow_stream(chunks: list[str], delay: float = 5.0) -> MagicMock:
    """Build a mock Anthropic stream that sleeps between chunks.

    Args:
        chunks: Text chunks to yield.
        delay: Seconds to sleep between each chunk.
    """

    async def _text_stream() -> Any:
        for chunk in chunks:
            yield chunk
            await asyncio.sleep(delay)

    stream = MagicMock()
    stream.text_stream = _text_stream()
    stream.get_final_message = AsyncMock(side_effect=Exception("stream interrupted"))

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=stream)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


@pytest.mark.unit
class TestClaudeToolStreaming:
    """Tests for ClaudeToolLLMAdapter streaming."""

    @pytest.mark.asyncio
    async def test_astream_yields_text_chunks(self, model_config: Any) -> None:
        """astream() yields text chunks and accumulates content."""
        adapter = ClaudeToolLLMAdapter(model_config)

        mock_client = AsyncMock()
        mock_client.messages.stream = MagicMock(return_value=_make_fast_stream(["Hello", ", ", "world!"]))

        with patch.object(adapter, "_get_async_client", return_value=mock_client):
            async with adapter.astream([Message.user("Hi")]) as sr:
                collected: list[str] = []
                async for chunk in sr:
                    collected.append(chunk)

        assert collected == ["Hello", ", ", "world!"]
        assert sr.accumulated_content == "Hello, world!"

    @pytest.mark.asyncio
    async def test_astream_extracts_usage(self, model_config: Any) -> None:
        """astream() extracts usage from the final message."""
        adapter = ClaudeToolLLMAdapter(model_config)

        final_msg = _MockFinalMessage(_MockUsage(input_tokens=200, output_tokens=80))
        mock_client = AsyncMock()
        mock_client.messages.stream = MagicMock(return_value=_make_fast_stream(["OK"], final_message=final_msg))

        with patch.object(adapter, "_get_async_client", return_value=mock_client):
            async with adapter.astream([Message.user("Hi")]) as sr:
                async for _ in sr:
                    pass

        assert sr.usage.input_tokens == 200
        assert sr.usage.output_tokens == 80

    @pytest.mark.asyncio
    async def test_astream_with_timeout_completes(self, model_config: Any) -> None:
        """_astream_with_timeout() completes without partial flag on fast responses."""
        adapter = ClaudeToolLLMAdapter(model_config)

        mock_client = AsyncMock()
        mock_client.messages.stream = MagicMock(return_value=_make_fast_stream(["All ", "done."]))

        with patch.object(adapter, "_get_async_client", return_value=mock_client):
            result = await adapter._astream_with_timeout([Message.user("Hi")], timeout=10.0)

        assert result.content == "All done."
        assert result.is_partial is False
        assert result.usage_unavailable is False

    @pytest.mark.asyncio
    async def test_astream_with_timeout_captures_partial(self, model_config: Any) -> None:
        """_astream_with_timeout() sets is_partial and usage_unavailable on timeout."""
        adapter = ClaudeToolLLMAdapter(model_config)

        mock_client = AsyncMock()
        mock_client.messages.stream = MagicMock(return_value=_make_slow_stream(["First", "Second", "Third"], delay=5.0))

        with patch.object(adapter, "_get_async_client", return_value=mock_client):
            result = await adapter._astream_with_timeout([Message.user("Hi")], timeout=0.05)

        assert result.is_partial is True
        assert result.usage_unavailable is True
        # Should have captured at least the first chunk
        assert "First" in result.content
