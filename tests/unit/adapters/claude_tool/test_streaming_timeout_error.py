"""Tests for Claude Tool adapter StreamingTimeoutError on timeout.

Verifies that _astream_with_timeout raises StreamingTimeoutError
instead of returning a partial LLMResponse with is_partial=True.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest

from karenina.adapters.claude_tool import ClaudeToolLLMAdapter
from karenina.exceptions import StreamingTimeoutError
from karenina.ports import Message
from karenina.ports.llm import StreamingLLMResponse
from karenina.schemas.config import ModelConfig


def _make_config() -> ModelConfig:
    return ModelConfig(
        id="test-claude-tool",
        model_name="claude-haiku-4-5",
        model_provider="anthropic",
        interface="claude_tool",
        max_tokens=1024,
    )


@pytest.mark.unit
class TestClaudeToolStreamingTimeoutError:
    """Verify _astream_with_timeout raises StreamingTimeoutError."""

    @pytest.mark.asyncio
    async def test_streaming_timeout_raises_error(self) -> None:
        """Timeout during streaming raises StreamingTimeoutError."""
        adapter = ClaudeToolLLMAdapter(_make_config())

        @asynccontextmanager
        async def _mock_astream(messages: list[Message]) -> AsyncIterator[StreamingLLMResponse]:
            sr = StreamingLLMResponse()

            async def _slow_chunks() -> AsyncIterator[str]:
                import asyncio

                sr._accumulated = "partial content"
                await asyncio.sleep(10)  # Will be cancelled by timeout
                yield "never reached"

            sr._set_chunk_source(_slow_chunks())
            yield sr

        with (
            patch.object(adapter, "astream", _mock_astream),
            pytest.raises(StreamingTimeoutError, match="timed out after 0.01s"),
        ):
            await adapter._astream_with_timeout(
                [Message.user("Hello")],
                timeout=0.01,
            )

    @pytest.mark.asyncio
    async def test_streaming_timeout_includes_partial_content(self) -> None:
        """StreamingTimeoutError carries partial content."""
        adapter = ClaudeToolLLMAdapter(_make_config())

        @asynccontextmanager
        async def _mock_astream(messages: list[Message]) -> AsyncIterator[StreamingLLMResponse]:
            sr = StreamingLLMResponse()

            async def _slow_chunks() -> AsyncIterator[str]:
                import asyncio

                yield "partial"
                await asyncio.sleep(10)
                yield "never"

            sr._set_chunk_source(_slow_chunks())
            yield sr

        with (
            patch.object(adapter, "astream", _mock_astream),
            pytest.raises(StreamingTimeoutError) as exc_info,
        ):
            await adapter._astream_with_timeout(
                [Message.user("Hello")],
                timeout=0.01,
            )
        assert exc_info.value.partial_content is not None
