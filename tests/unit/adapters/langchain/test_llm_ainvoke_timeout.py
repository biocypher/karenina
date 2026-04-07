"""Tests for issue 192: per-attempt ``ainvoke`` timeout in LangChainLLMAdapter.

The LangChain ``ainvoke`` paths used by the parser previously had no
karenina-layer wall-clock guard. A stalled coroutine inside LangChain's
structured-output / usage-metadata-callback machinery could hang a worker
thread forever, since httpx's ``request_timeout`` does not catch every
such case.

These tests verify that:
1. ``LangChainLLMAdapter._ainvoke_text`` enforces ``request_timeout``.
2. ``LangChainLLMAdapter._ainvoke_structured`` enforces ``request_timeout``
   on the structured invocation path.
3. ``LangChainLLMAdapter._ainvoke_with_fallback_parsing`` enforces
   ``request_timeout`` on the fallback parsing path.
4. When ``request_timeout`` is None, no timeout is imposed.
5. The fired ``asyncio.TimeoutError`` flows through ``RetryExecutor`` and
   is classified as ``TIMEOUT``, so the configured timeout retry budget
   applies and a subsequent successful attempt completes the call.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from karenina.adapters.langchain import LangChainLLMAdapter
from karenina.ports import Message
from karenina.schemas.config import ModelConfig
from karenina.utils.retry_policy import (
    CategoryRetryConfig,
    RetryPolicy,
    track_retries,
)


class _Schema(BaseModel):
    """Tiny schema for structured-output tests."""

    value: str


def _make_config(*, request_timeout: float | None, retry_policy: RetryPolicy | None = None) -> ModelConfig:
    """Build a minimal ModelConfig for the langchain interface."""
    return ModelConfig(
        id="test-timeout",
        model_name="test-model",
        model_provider="openai",
        interface="langchain",
        request_timeout=request_timeout,
        retry_policy=retry_policy,
    )


def _build_adapter_with_mock_model(model_config: ModelConfig, mock_model: Any) -> LangChainLLMAdapter:
    """Build an adapter whose underlying model is replaced with ``mock_model``."""
    with patch("karenina.adapters.langchain.initialization.init_chat_model_unified") as mock_init:
        mock_init.return_value = mock_model
        return LangChainLLMAdapter(model_config)


@pytest.mark.unit
class TestAinvokeTextTimeout:
    """Tests for ``_ainvoke_text`` per-attempt timeout enforcement."""

    @pytest.mark.asyncio
    async def test_text_invoke_times_out_when_model_hangs(self) -> None:
        """A hanging ``ainvoke`` raises ``TimeoutError`` within the configured budget."""
        # No retries; we want the timeout to surface immediately on the first attempt.
        policy = RetryPolicy(timeout=CategoryRetryConfig(max_attempts=0))
        config = _make_config(request_timeout=0.2, retry_policy=policy)

        async def _hang(*args: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(60)
            return MagicMock()

        mock_model = MagicMock()
        mock_model.ainvoke = _hang
        adapter = _build_adapter_with_mock_model(config, mock_model)

        start = time.monotonic()
        with pytest.raises(TimeoutError):
            await adapter._ainvoke_text([Message.user("Hello")])
        elapsed = time.monotonic() - start
        assert elapsed < 1.5, f"Timeout fired but took {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_text_invoke_no_timeout_when_request_timeout_none(self) -> None:
        """With ``request_timeout=None``, a fast model returns normally."""
        config = _make_config(request_timeout=None)

        mock_response = MagicMock()
        mock_response.content = "ok"
        mock_response.response_metadata = {"usage": {"input_tokens": 1, "output_tokens": 1}}

        async def _fast(*args: Any, **kwargs: Any) -> Any:
            return mock_response

        mock_model = MagicMock()
        mock_model.ainvoke = _fast
        adapter = _build_adapter_with_mock_model(config, mock_model)

        result = await adapter._ainvoke_text([Message.user("Hello")])
        assert result.content == "ok"


@pytest.mark.unit
class TestAinvokeStructuredTimeout:
    """Tests for ``_ainvoke_structured`` per-attempt timeout enforcement."""

    @pytest.mark.asyncio
    async def test_structured_invoke_times_out_when_model_hangs(self) -> None:
        """A hanging structured ``ainvoke`` raises ``TimeoutError`` quickly."""
        policy = RetryPolicy(timeout=CategoryRetryConfig(max_attempts=0))
        config = _make_config(request_timeout=0.2, retry_policy=policy)

        async def _hang(*args: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(60)
            return MagicMock()

        # Base model with a structured wrapper that also hangs.
        structured_model = MagicMock()
        structured_model.ainvoke = _hang

        base_model = MagicMock()
        base_model.with_structured_output = MagicMock(return_value=structured_model)
        base_model.ainvoke = _hang

        adapter = _build_adapter_with_mock_model(config, base_model)
        structured_adapter = adapter.with_structured_output(_Schema, max_retries=0)

        start = time.monotonic()
        with pytest.raises(TimeoutError):
            await structured_adapter._ainvoke_structured([Message.user("Hello")])
        elapsed = time.monotonic() - start
        assert elapsed < 2.0, f"Timeout fired but took {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_structured_invoke_no_timeout_when_request_timeout_none(self) -> None:
        """With ``request_timeout=None``, the structured path returns normally."""
        config = _make_config(request_timeout=None)

        async def _fast(*args: Any, **kwargs: Any) -> Any:
            return _Schema(value="ok")

        structured_model = MagicMock()
        structured_model.ainvoke = _fast

        base_model = MagicMock()
        base_model.with_structured_output = MagicMock(return_value=structured_model)
        base_model.ainvoke = _fast

        adapter = _build_adapter_with_mock_model(config, base_model)
        structured_adapter = adapter.with_structured_output(_Schema, max_retries=0)

        result = await structured_adapter._ainvoke_structured([Message.user("Hello")])
        assert isinstance(result.raw, _Schema)
        assert result.raw.value == "ok"


@pytest.mark.unit
class TestAinvokeFallbackParsingTimeout:
    """Tests for ``_ainvoke_with_fallback_parsing`` per-attempt timeout enforcement."""

    @pytest.mark.asyncio
    async def test_fallback_invoke_times_out_when_model_hangs(self) -> None:
        """A hanging fallback ``ainvoke`` raises ``TimeoutError`` quickly."""
        policy = RetryPolicy(timeout=CategoryRetryConfig(max_attempts=0))
        config = _make_config(request_timeout=0.2, retry_policy=policy)

        async def _hang(*args: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(60)
            return MagicMock()

        # Base model that has no working structured output: force the
        # fallback path by making with_structured_output raise.
        base_model = MagicMock()
        base_model.with_structured_output = MagicMock(side_effect=RuntimeError("not supported"))
        base_model.ainvoke = _hang

        adapter = _build_adapter_with_mock_model(config, base_model)
        structured_adapter = adapter.with_structured_output(_Schema, max_retries=0)

        start = time.monotonic()
        with pytest.raises(TimeoutError):
            await structured_adapter._ainvoke_with_fallback_parsing([Message.user("Hello")])
        elapsed = time.monotonic() - start
        assert elapsed < 2.0, f"Timeout fired but took {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_fallback_invoke_no_timeout_when_request_timeout_none(self) -> None:
        """With ``request_timeout=None``, the fallback path returns normally."""
        config = _make_config(request_timeout=None)

        mock_response = MagicMock()
        mock_response.content = '{"value": "ok"}'
        mock_response.response_metadata = {"usage": {"input_tokens": 1, "output_tokens": 1}}

        async def _fast(*args: Any, **kwargs: Any) -> Any:
            return mock_response

        base_model = MagicMock()
        base_model.with_structured_output = MagicMock(side_effect=RuntimeError("not supported"))
        base_model.ainvoke = _fast

        adapter = _build_adapter_with_mock_model(config, base_model)
        structured_adapter = adapter.with_structured_output(_Schema, max_retries=0)

        result = await structured_adapter._ainvoke_with_fallback_parsing([Message.user("Hello")])
        assert isinstance(result.raw, _Schema)
        assert result.raw.value == "ok"


@pytest.mark.unit
class TestRetryExecutorTimeoutIntegration:
    """Integration test: timeouts flow through RetryExecutor as TIMEOUT category."""

    @pytest.mark.asyncio
    async def test_timeout_then_success_consumes_retry_budget(self) -> None:
        """Two timeouts followed by success: third attempt wins, tracker shows used=2."""
        policy = RetryPolicy(
            timeout=CategoryRetryConfig(
                max_attempts=2,
                backoff_min=0.0,
                backoff_max=0.0,
            ),
        )
        config = _make_config(request_timeout=0.1, retry_policy=policy)

        mock_response = MagicMock()
        mock_response.content = "ok"
        mock_response.response_metadata = {"usage": {"input_tokens": 1, "output_tokens": 1}}

        call_count = 0

        async def _fail_twice_then_succeed(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                # Sleep longer than request_timeout so the wait_for fires.
                await asyncio.sleep(5)
                return mock_response
            return mock_response

        mock_model = MagicMock()
        mock_model.ainvoke = _fail_twice_then_succeed
        adapter = _build_adapter_with_mock_model(config, mock_model)

        with track_retries(policy) as tracker:
            result = await adapter.ainvoke([Message.user("Hello")])

        assert result.content == "ok"
        assert call_count == 3
        assert tracker["timeout"]["used"] == 2
        assert tracker["timeout"]["budget"] == 2
