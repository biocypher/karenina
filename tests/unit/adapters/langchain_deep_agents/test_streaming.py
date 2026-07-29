"""Tests for DeepAgentsLLMAdapter streaming methods.

Tests astream() and _astream_with_timeout() using mocked LangChain chat model.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from karenina.adapters.langchain_deep_agents.llm import DeepAgentsLLMAdapter
from karenina.ports import Message


class _FakeChunk:
    """Minimal mock of a LangChain AIMessageChunk."""

    def __init__(self, content: str = "", usage_metadata: dict[str, Any] | None = None) -> None:
        self.content = content
        self.usage_metadata = usage_metadata


@pytest.mark.unit
class TestDeepAgentsStreaming:
    """Tests for DeepAgentsLLMAdapter streaming."""

    @pytest.mark.asyncio
    async def test_astream_yields_text_chunks(self, deep_agents_model_config: Any, monkeypatch: Any) -> None:
        """astream() yields text chunks and accumulates content."""
        chunks = [
            _FakeChunk(content="Hello"),
            _FakeChunk(content=", "),
            _FakeChunk(content="world!"),
        ]

        async def _fake_astream(messages: Any) -> Any:
            for chunk in chunks:
                yield chunk

        mock_model = MagicMock()
        mock_model.astream = _fake_astream

        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.llm.create_chat_model",
            lambda _config, **_kw: mock_model,
        )

        adapter = DeepAgentsLLMAdapter(deep_agents_model_config)

        async with adapter.astream([Message.user("Hi")]) as sr:
            collected: list[str] = []
            async for chunk in sr:
                collected.append(chunk)

        assert collected == ["Hello", ", ", "world!"]
        assert sr.accumulated_content == "Hello, world!"

    @pytest.mark.asyncio
    async def test_astream_extracts_usage_from_final_chunk(
        self, deep_agents_model_config: Any, monkeypatch: Any
    ) -> None:
        """astream() extracts usage from chunk with usage_metadata."""
        chunks = [
            _FakeChunk(content="Response"),
            _FakeChunk(
                content="",
                usage_metadata={"input_tokens": 120, "output_tokens": 60},
            ),
        ]

        async def _fake_astream(messages: Any) -> Any:
            for chunk in chunks:
                yield chunk

        mock_model = MagicMock()
        mock_model.astream = _fake_astream

        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.llm.create_chat_model",
            lambda _config, **_kw: mock_model,
        )

        adapter = DeepAgentsLLMAdapter(deep_agents_model_config)

        async with adapter.astream([Message.user("Hi")]) as sr:
            async for _ in sr:
                pass

        assert sr.usage.input_tokens == 120
        assert sr.usage.output_tokens == 60

    @pytest.mark.asyncio
    async def test_astream_establishment_retries_transient_failure(
        self, deep_agents_model_config: Any, monkeypatch: Any
    ) -> None:
        """A transient error while opening the stream is retried from scratch."""
        attempts = {"count": 0}

        async def _flaky_astream(messages: Any) -> Any:
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise ConnectionError("cross-loop pooled connection error")
            yield _FakeChunk(content="recovered")

        mock_model = MagicMock()
        mock_model.astream = _flaky_astream

        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.llm.create_chat_model",
            lambda _config, **_kw: mock_model,
        )

        adapter = DeepAgentsLLMAdapter(deep_agents_model_config)

        async with adapter.astream([Message.user("Hi")]) as sr:
            async for _ in sr:
                pass

        assert sr.accumulated_content == "recovered"
        assert attempts["count"] == 2

    @pytest.mark.asyncio
    async def test_establishment_hang_raises_streaming_timeout_promptly(
        self, deep_agents_model_config: Any, monkeypatch: Any
    ) -> None:
        """An establishment stall is bounded by the streaming wall-clock timeout.

        request_timeout is unset (the default), so only the stream-level
        timeout window can bound establishment. The half-opened stream
        iterator must also be closed, leaving no never-closed debris.
        """
        from karenina.exceptions import StreamingTimeoutError

        closed = {"value": False}

        async def _hanging_astream(messages: Any) -> Any:
            try:
                await asyncio.sleep(30)
                yield _FakeChunk(content="too late")
            finally:
                closed["value"] = True

        mock_model = MagicMock()
        mock_model.astream = _hanging_astream

        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.llm.create_chat_model",
            lambda _config, **_kw: mock_model,
        )

        adapter = DeepAgentsLLMAdapter(deep_agents_model_config)

        start = time.monotonic()
        with pytest.raises(StreamingTimeoutError) as exc_info:
            await adapter._astream_with_timeout([Message.user("Hi")], timeout=0.05)
        elapsed = time.monotonic() - start

        assert elapsed < 2.0, f"timeout fired after {elapsed:.2f}s, expected near 0.05s"
        assert exc_info.value.partial_content == ""
        assert closed["value"], "half-opened stream iterator must be closed on establishment timeout"

    @pytest.mark.asyncio
    async def test_astream_with_timeout_completes(self, deep_agents_model_config: Any, monkeypatch: Any) -> None:
        """_astream_with_timeout() completes without partial flag on fast responses."""
        chunks = [
            _FakeChunk(content="All "),
            _FakeChunk(content="done."),
        ]

        async def _fake_astream(messages: Any) -> Any:
            for chunk in chunks:
                yield chunk

        mock_model = MagicMock()
        mock_model.astream = _fake_astream

        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.llm.create_chat_model",
            lambda _config, **_kw: mock_model,
        )

        adapter = DeepAgentsLLMAdapter(deep_agents_model_config)
        result = await adapter._astream_with_timeout([Message.user("Hi")], timeout=10.0)

        assert result.content == "All done."
        assert result.is_partial is False
        assert result.usage_unavailable is False

    @pytest.mark.asyncio
    async def test_astream_with_timeout_raises_streaming_timeout_error(
        self, deep_agents_model_config: Any, monkeypatch: Any
    ) -> None:
        """_astream_with_timeout() raises StreamingTimeoutError on timeout."""
        from karenina.exceptions import StreamingTimeoutError

        chunks = [
            _FakeChunk(content="First"),
            _FakeChunk(content="Second"),
            _FakeChunk(content="Third"),
        ]

        async def _slow_astream(messages: Any) -> Any:
            for chunk in chunks:
                yield chunk
                await asyncio.sleep(5.0)

        mock_model = MagicMock()
        mock_model.astream = _slow_astream

        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.llm.create_chat_model",
            lambda _config, **_kw: mock_model,
        )

        adapter = DeepAgentsLLMAdapter(deep_agents_model_config)

        with pytest.raises(StreamingTimeoutError) as exc_info:
            await adapter._astream_with_timeout([Message.user("Hi")], timeout=0.05)

        assert "First" in exc_info.value.partial_content
