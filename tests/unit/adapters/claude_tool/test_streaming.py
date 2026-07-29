"""Tests for ClaudeToolLLMAdapter streaming methods.

Tests astream() and _astream_with_timeout() using mocked Anthropic client.
The mock stream yields Anthropic streaming events (message_start, text,
message_delta) so the inline usage capture path is exercised, and the
mock stream manager is entered via __aenter__ so the establishment-retry
path is exercised.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from karenina.adapters.claude_tool import ClaudeToolLLMAdapter
from karenina.ports import Message


class _MockUsage:
    """Mock Anthropic usage object."""

    def __init__(self, input_tokens: int | None = 100, output_tokens: int | None = 50) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = None
        self.cache_creation_input_tokens = None


class _MockFinalMessage:
    """Mock final message returned by stream.get_final_message()."""

    def __init__(self, usage: _MockUsage | None = None) -> None:
        self.usage = usage or _MockUsage()


def _text_event(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _message_start_event(usage: _MockUsage) -> SimpleNamespace:
    return SimpleNamespace(type="message_start", message=SimpleNamespace(usage=usage))


def _message_delta_event(usage: _MockUsage) -> SimpleNamespace:
    return SimpleNamespace(type="message_delta", usage=usage)


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
    """Build a mock Anthropic stream context manager that yields events quickly.

    Args:
        chunks: Text chunks to yield as text events.
        final_message: Final message for usage extraction.
    """
    if final_message is None:
        final_message = _MockFinalMessage()

    async def _events() -> Any:
        yield _message_start_event(_MockUsage(input_tokens=100, output_tokens=1))
        for chunk in chunks:
            yield _text_event(chunk)
        yield _message_delta_event(_MockUsage(input_tokens=None, output_tokens=50))

    stream = MagicMock()
    stream.__aiter__ = lambda _self: _events()
    stream.get_final_message = AsyncMock(return_value=final_message)

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=stream)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


def _make_slow_stream(chunks: list[str], delay: float = 5.0) -> MagicMock:
    """Build a mock Anthropic stream that sleeps between text events.

    Usage arrives in message_start before the first text chunk, so a
    mid-stream timeout still leaves a partial usage snapshot.

    Args:
        chunks: Text chunks to yield.
        delay: Seconds to sleep between each chunk.
    """

    async def _events() -> Any:
        yield _message_start_event(_MockUsage(input_tokens=200, output_tokens=1))
        for chunk in chunks:
            yield _text_event(chunk)
            await asyncio.sleep(delay)

    stream = MagicMock()
    stream.__aiter__ = lambda _self: _events()
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
        """astream() extracts usage from the final message on success."""
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
    async def test_astream_captures_usage_inline_from_events(self, model_config: Any) -> None:
        """Usage is captured from message_start and message_delta as events arrive."""
        adapter = ClaudeToolLLMAdapter(model_config)

        mock_client = AsyncMock()
        ctx = _make_fast_stream(["chunk"])
        # Make the final-message enrichment fail so only inline capture remains.
        ctx.__aenter__.return_value.get_final_message = AsyncMock(side_effect=Exception("no final message"))
        mock_client.messages.stream = MagicMock(return_value=ctx)

        with patch.object(adapter, "_get_async_client", return_value=mock_client):
            async with adapter.astream([Message.user("Hi")]) as sr:
                async for _ in sr:
                    pass

        # message_start carried input_tokens=100, message_delta carried output_tokens=50.
        assert sr.usage.input_tokens == 100
        assert sr.usage.output_tokens == 50

    @pytest.mark.asyncio
    async def test_mid_stream_timeout_leaves_partial_usage(self, model_config: Any) -> None:
        """A mid-stream timeout leaves whatever usage arrived before the cut."""
        adapter = ClaudeToolLLMAdapter(model_config)

        mock_client = AsyncMock()
        mock_client.messages.stream = MagicMock(return_value=_make_slow_stream(["First", "Second"], delay=5.0))

        with patch.object(adapter, "_get_async_client", return_value=mock_client):
            async with adapter.astream([Message.user("Hi")]) as sr:
                with pytest.raises(TimeoutError):
                    async with asyncio.timeout(0.05):
                        async for _ in sr:
                            pass

        # message_start arrived before the stall, so input tokens are present.
        assert sr.usage.input_tokens == 200
        assert "First" in sr.accumulated_content

    @pytest.mark.asyncio
    async def test_astream_establishment_retries_transient_failure(self, model_config: Any) -> None:
        """A transient connection error during stream establishment is retried."""
        adapter = ClaudeToolLLMAdapter(model_config)

        failing_ctx = AsyncMock()
        failing_ctx.__aenter__ = AsyncMock(side_effect=ConnectionError("setup failed"))
        failing_ctx.__aexit__ = AsyncMock(return_value=False)

        good_ctx = _make_fast_stream(["recovered"])

        mock_client = AsyncMock()
        mock_client.messages.stream = MagicMock(side_effect=[failing_ctx, good_ctx])

        with patch.object(adapter, "_get_async_client", return_value=mock_client):
            async with adapter.astream([Message.user("Hi")]) as sr:
                async for _ in sr:
                    pass

        assert sr.accumulated_content == "recovered"
        assert mock_client.messages.stream.call_count == 2

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
    async def test_astream_with_timeout_raises_streaming_timeout_error(self, model_config: Any) -> None:
        """_astream_with_timeout() raises StreamingTimeoutError on timeout."""
        from karenina.exceptions import StreamingTimeoutError

        adapter = ClaudeToolLLMAdapter(model_config)

        mock_client = AsyncMock()
        mock_client.messages.stream = MagicMock(return_value=_make_slow_stream(["First", "Second", "Third"], delay=5.0))

        with (
            patch.object(adapter, "_get_async_client", return_value=mock_client),
            pytest.raises(StreamingTimeoutError) as exc_info,
        ):
            await adapter._astream_with_timeout([Message.user("Hi")], timeout=0.05)

        # Should have captured at least the first chunk in partial_content
        assert "First" in exc_info.value.partial_content
