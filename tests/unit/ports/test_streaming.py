"""Tests for StreamingLLMResponse accumulation behavior."""

import asyncio

import pytest

from karenina.ports.llm import StreamingLLMResponse


async def _make_chunks(*texts: str):
    """Async generator yielding text chunks."""
    for t in texts:
        yield t


async def _make_slow_chunks(*texts: str, delay: float = 0.5):
    """Async generator yielding text chunks with delay."""
    for t in texts:
        await asyncio.sleep(delay)
        yield t


async def _make_error_chunks(*texts: str):
    """Async generator that yields some chunks then raises."""
    for t in texts:
        yield t
    raise RuntimeError("stream interrupted")


@pytest.mark.unit
class TestStreamingLLMResponse:
    """Tests for StreamingLLMResponse chunk accumulation."""

    @pytest.mark.asyncio
    async def test_accumulates_chunks(self):
        sr = StreamingLLMResponse()
        sr._set_chunk_source(_make_chunks("Hello", " ", "world"))
        chunks = []
        async for chunk in sr:
            chunks.append(chunk)
        assert chunks == ["Hello", " ", "world"]
        assert sr.accumulated_content == "Hello world"

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        sr = StreamingLLMResponse()
        sr._set_chunk_source(_make_chunks())
        async for _chunk in sr:
            pass
        assert sr.accumulated_content == ""

    @pytest.mark.asyncio
    async def test_no_chunk_source_stops_immediately(self):
        sr = StreamingLLMResponse()
        chunks = [c async for c in sr]
        assert chunks == []
        assert sr.accumulated_content == ""

    @pytest.mark.asyncio
    async def test_error_midstream_preserves_partial(self):
        sr = StreamingLLMResponse()
        sr._set_chunk_source(_make_error_chunks("partial", " content"))
        with pytest.raises(RuntimeError, match="stream interrupted"):
            async for _chunk in sr:
                pass
        assert sr.accumulated_content == "partial content"

    @pytest.mark.asyncio
    async def test_is_complete_default_false(self):
        sr = StreamingLLMResponse()
        assert sr.is_complete is False

    @pytest.mark.asyncio
    async def test_usage_default_empty(self):
        sr = StreamingLLMResponse()
        assert sr.usage.input_tokens == 0
        assert sr.usage.output_tokens == 0
