"""Tests for the request_timeout wall-clock guard on ClaudeSDKLLMAdapter.ainvoke.

When ModelConfig.request_timeout is set, the query() drain loop must be
bounded by asyncio.wait_for and surface a stock TimeoutError (classified
as TIMEOUT by the ErrorRegistry). The SDK module is replaced per test via
monkeypatch.setitem on sys.modules, so no module-level stub leaks into
other test files. The adapter resolves claude_agent_sdk at call time
inside ainvoke, which makes the per-test swap sufficient.
"""

from __future__ import annotations

import asyncio
import sys
import time
import types
from typing import Any

import pytest

from karenina.adapters.claude_agent_sdk.llm import ClaudeSDKLLMAdapter
from karenina.ports import Message


def _build_fake_sdk() -> types.ModuleType:
    """Build a minimal claude_agent_sdk stub with all needed symbols."""

    class ClaudeAgentOptions:
        def __init__(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

        def __getattr__(self, name: str) -> None:
            return None

    class TextBlock:
        def __init__(self, text: str = "") -> None:
            self.text = text

    class AssistantMessage:
        def __init__(self, content: list[Any] | None = None) -> None:
            self.content = content or []

    class ResultMessage:
        def __init__(
            self,
            result: str = "",
            usage: dict[str, int] | None = None,
            total_cost_usd: float | None = None,
            structured_output: dict[str, Any] | None = None,
        ) -> None:
            self.result = result
            self.usage = usage or {}
            self.total_cost_usd = total_cost_usd
            self.structured_output = structured_output

    mod = types.ModuleType("claude_agent_sdk")
    for _name, _obj in [
        ("ClaudeAgentOptions", ClaudeAgentOptions),
        ("TextBlock", TextBlock),
        ("AssistantMessage", AssistantMessage),
        ("ResultMessage", ResultMessage),
        # query is a sentinel. Every test sets it before invoking the adapter.
        ("query", None),
    ]:
        setattr(mod, _name, _obj)
    return mod


@pytest.fixture
def fake_sdk(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install a fresh SDK stub for this test only (monkeypatch restores it)."""
    sdk = _build_fake_sdk()
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", sdk)
    return sdk


def _model_config(request_timeout: float | None) -> Any:
    from karenina.schemas.config import ModelConfig

    return ModelConfig(
        id="test-sdk-timeout",
        model_name="claude-sonnet-4-20250514",
        model_provider="anthropic",
        interface="claude_agent_sdk",
        request_timeout=request_timeout,
    )


@pytest.mark.unit
class TestAinvokeRequestTimeout:
    """request_timeout bounds the wall clock of the query() drain."""

    @pytest.mark.asyncio
    async def test_slow_query_times_out_near_configured_value(self, fake_sdk: types.ModuleType) -> None:
        async def slow_query(prompt: str, options: Any) -> Any:
            await asyncio.sleep(30)
            yield fake_sdk.ResultMessage(result="too late")

        fake_sdk.query = slow_query

        adapter = ClaudeSDKLLMAdapter(_model_config(request_timeout=0.2))

        start = time.monotonic()
        with pytest.raises(TimeoutError):
            await adapter.ainvoke([Message.user("Hi")])
        elapsed = time.monotonic() - start
        assert elapsed < 2.0, f"timeout fired after {elapsed:.2f}s, expected near 0.2s"

    @pytest.mark.asyncio
    async def test_timeout_error_classified_as_timeout(self, fake_sdk: types.ModuleType) -> None:
        from karenina.utils.errors import ErrorCategory, ErrorRegistry

        async def slow_query(prompt: str, options: Any) -> Any:
            await asyncio.sleep(30)
            yield fake_sdk.ResultMessage(result="too late")

        fake_sdk.query = slow_query

        adapter = ClaudeSDKLLMAdapter(_model_config(request_timeout=0.1))

        with pytest.raises(TimeoutError) as exc_info:
            await adapter.ainvoke([Message.user("Hi")])
        assert ErrorRegistry().classify(exc_info.value) is ErrorCategory.TIMEOUT

    @pytest.mark.asyncio
    async def test_fast_query_succeeds_under_timeout(self, fake_sdk: types.ModuleType) -> None:
        async def fast_query(prompt: str, options: Any) -> Any:
            yield fake_sdk.ResultMessage(
                result="quick answer",
                usage={"input_tokens": 5, "output_tokens": 3},
            )

        fake_sdk.query = fast_query

        adapter = ClaudeSDKLLMAdapter(_model_config(request_timeout=5.0))

        response = await adapter.ainvoke([Message.user("Hi")])
        assert response.content == "quick answer"

    @pytest.mark.asyncio
    async def test_no_timeout_when_request_timeout_none(self, fake_sdk: types.ModuleType) -> None:
        async def fast_query(prompt: str, options: Any) -> Any:
            yield fake_sdk.ResultMessage(result="unbounded answer")

        fake_sdk.query = fast_query

        adapter = ClaudeSDKLLMAdapter(_model_config(request_timeout=None))

        response = await adapter.ainvoke([Message.user("Hi")])
        assert response.content == "unbounded answer"
