"""Tests for LangChainLLMAdapter streaming methods.

Tests astream() and _astream_with_timeout() using mocked LangChain model.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from karenina.adapters.langchain import LangChainLLMAdapter
from karenina.ports import Message


class _FakeChunk:
    """Minimal mock of a LangChain AIMessageChunk."""

    def __init__(self, content: str = "", usage_metadata: dict[str, Any] | None = None) -> None:
        self.content = content
        self.usage_metadata = usage_metadata


@pytest.fixture
def model_config() -> Any:
    """Create a ModelConfig for the langchain interface."""
    from karenina.schemas.config import ModelConfig

    return ModelConfig(
        id="test-lc-stream",
        model_name="claude-sonnet-4-20250514",
        model_provider="anthropic",
        interface="langchain",
    )


def _build_adapter(model_config: Any, chunks: list[_FakeChunk]) -> LangChainLLMAdapter:
    """Build a LangChainLLMAdapter with a mocked model that yields chunks.

    Args:
        model_config: ModelConfig fixture.
        chunks: Fake AIMessageChunk objects to yield from astream.
    """

    async def _fake_astream(messages: Any) -> Any:
        for chunk in chunks:
            yield chunk

    mock_model = MagicMock()
    mock_model.astream = _fake_astream

    with patch("karenina.adapters.langchain.initialization.init_chat_model_unified") as mock_init:
        mock_init.return_value = mock_model
        adapter = LangChainLLMAdapter(model_config)

    return adapter


def _build_slow_adapter(model_config: Any, chunks: list[_FakeChunk], delay: float = 5.0) -> LangChainLLMAdapter:
    """Build a LangChainLLMAdapter whose astream sleeps between chunks.

    Args:
        model_config: ModelConfig fixture.
        chunks: Fake AIMessageChunk objects to yield.
        delay: Seconds to sleep between chunks.
    """

    async def _fake_astream(messages: Any) -> Any:
        for chunk in chunks:
            yield chunk
            await asyncio.sleep(delay)

    mock_model = MagicMock()
    mock_model.astream = _fake_astream

    with patch("karenina.adapters.langchain.initialization.init_chat_model_unified") as mock_init:
        mock_init.return_value = mock_model
        adapter = LangChainLLMAdapter(model_config)

    return adapter


@pytest.mark.unit
class TestLangChainStreaming:
    """Tests for LangChainLLMAdapter streaming."""

    @pytest.mark.asyncio
    async def test_astream_yields_text_chunks(self, model_config: Any) -> None:
        """astream() yields text chunks and accumulates content."""
        chunks = [
            _FakeChunk(content="Hello"),
            _FakeChunk(content=", "),
            _FakeChunk(content="world!"),
        ]
        adapter = _build_adapter(model_config, chunks)

        async with adapter.astream([Message.user("Hi")]) as sr:
            collected: list[str] = []
            async for chunk in sr:
                collected.append(chunk)

        assert collected == ["Hello", ", ", "world!"]
        assert sr.accumulated_content == "Hello, world!"

    @pytest.mark.asyncio
    async def test_astream_extracts_usage_from_final_chunk(self, model_config: Any) -> None:
        """astream() extracts usage from chunk with usage_metadata."""
        chunks = [
            _FakeChunk(content="Response"),
            _FakeChunk(
                content="",
                usage_metadata={"input_tokens": 120, "output_tokens": 60},
            ),
        ]
        adapter = _build_adapter(model_config, chunks)

        async with adapter.astream([Message.user("Hi")]) as sr:
            async for _ in sr:
                pass

        assert sr.usage.input_tokens == 120
        assert sr.usage.output_tokens == 60

    @pytest.mark.asyncio
    async def test_astream_with_timeout_completes(self, model_config: Any) -> None:
        """_astream_with_timeout() completes without partial flag on fast responses."""
        chunks = [
            _FakeChunk(content="All "),
            _FakeChunk(content="done."),
        ]
        adapter = _build_adapter(model_config, chunks)

        result = await adapter._astream_with_timeout([Message.user("Hi")], timeout=10.0)

        assert result.content == "All done."
        assert result.is_partial is False
        assert result.usage_unavailable is False

    @pytest.mark.asyncio
    async def test_astream_with_timeout_raises_streaming_timeout_error(self, model_config: Any) -> None:
        """_astream_with_timeout() raises StreamingTimeoutError on timeout."""
        from karenina.exceptions import StreamingTimeoutError

        chunks = [
            _FakeChunk(content="First"),
            _FakeChunk(content="Second"),
            _FakeChunk(content="Third"),
        ]
        adapter = _build_slow_adapter(model_config, chunks, delay=5.0)

        with pytest.raises(StreamingTimeoutError) as exc_info:
            await adapter._astream_with_timeout([Message.user("Hi")], timeout=0.05)

        assert "First" in exc_info.value.partial_content

    @pytest.mark.asyncio
    async def test_astream_with_timeout_no_partial_flag(self, model_config: Any) -> None:
        """_astream_with_timeout() does not set is_partial on success."""
        chunks = [
            _FakeChunk(content="All "),
            _FakeChunk(content="done."),
        ]
        adapter = _build_adapter(model_config, chunks)

        result = await adapter._astream_with_timeout([Message.user("Hi")], timeout=10.0)

        assert result.is_partial is False

    @pytest.mark.asyncio
    async def test_astream_accumulates_text_from_thinking_blocks(self, model_config: Any) -> None:
        """Anthropic extended thinking chunks arrive as list[dict] with thinking + text blocks.

        Regression: the old ``isinstance(chunk.content, str)`` gate dropped
        every block-list chunk, producing empty ``accumulated_content``.
        The fix must extract only ``type == "text"`` blocks while dropping
        ``thinking`` blocks.
        """
        chunks = [
            _FakeChunk(content=[{"type": "thinking", "thinking": "Reasoning...", "index": 0}]),
            _FakeChunk(content=[{"type": "thinking", "thinking": " more reasoning", "index": 0}]),
            _FakeChunk(content=[{"type": "text", "text": "Hello, "}]),
            _FakeChunk(content=[{"type": "text", "text": "world!"}]),
        ]
        adapter = _build_adapter(model_config, chunks)

        async with adapter.astream([Message.user("Hi")]) as sr:
            async for _ in sr:
                pass

        assert sr.accumulated_content == "Hello, world!"

    @pytest.mark.asyncio
    async def test_astream_with_timeout_extracts_text_from_list_chunks(self, model_config: Any) -> None:
        """_astream_with_timeout returns text-only content for list-block streams."""
        chunks = [
            _FakeChunk(content=[{"type": "thinking", "thinking": "secret", "index": 0}]),
            _FakeChunk(content=[{"type": "text", "text": "Final answer."}]),
        ]
        adapter = _build_adapter(model_config, chunks)

        result = await adapter._astream_with_timeout([Message.user("Hi")], timeout=10.0)

        assert result.content == "Final answer."
        assert "secret" not in result.content
