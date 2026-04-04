"""Tests for ClaudeSDKLLMAdapter streaming methods.

Tests astream() and _astream_with_timeout() using mocked claude_agent_sdk.query().
"""

from __future__ import annotations

import asyncio
import sys
import types
from typing import Any

import pytest

from karenina.ports import Message

# ---------------------------------------------------------------------------
# Stub out claude_agent_sdk before importing the adapter
# ---------------------------------------------------------------------------


def _build_fake_sdk() -> types.ModuleType:
    """Build a minimal claude_agent_sdk stub with all needed symbols."""

    class ClaudeAgentOptions:
        def __init__(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

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
        # query is a sentinel; every test replaces it via monkeypatch before use.
        ("query", None),
    ]:
        setattr(mod, _name, _obj)
    return mod


# Install the stub once at module level so that `from claude_agent_sdk import ...`
# inside the adapter resolves all names. Individual tests replace mod.query.
_SDK = _build_fake_sdk()
if "claude_agent_sdk" not in sys.modules:
    sys.modules["claude_agent_sdk"] = _SDK


from karenina.adapters.claude_agent_sdk.llm import ClaudeSDKLLMAdapter  # noqa: E402


@pytest.fixture
def model_config() -> Any:
    """Create a ModelConfig for the claude_agent_sdk interface."""
    from karenina.schemas.config import ModelConfig

    return ModelConfig(
        id="test-sdk-stream",
        model_name="claude-sonnet-4-20250514",
        model_provider="anthropic",
        interface="claude_agent_sdk",
    )


@pytest.mark.unit
class TestClaudeSDKStreaming:
    """Tests for ClaudeSDKLLMAdapter streaming."""

    @pytest.mark.asyncio
    async def test_astream_yields_text_chunks(self, model_config: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """astream() yields text deltas and accumulates content."""

        async def fake_query(prompt: str, options: Any) -> Any:
            yield _SDK.AssistantMessage(content=[_SDK.TextBlock("Hello")])
            yield _SDK.AssistantMessage(content=[_SDK.TextBlock("Hello, world!")])
            yield _SDK.ResultMessage(
                result="Hello, world!",
                usage={"input_tokens": 50, "output_tokens": 30},
            )

        monkeypatch.setattr(_SDK, "query", fake_query)

        adapter = ClaudeSDKLLMAdapter(model_config)

        async with adapter.astream([Message.user("Hi")]) as sr:
            collected: list[str] = []
            async for chunk in sr:
                collected.append(chunk)

        # First message yields "Hello", second yields delta ", world!"
        assert sr.accumulated_content == "Hello, world!"
        assert len(collected) == 2

    @pytest.mark.asyncio
    async def test_astream_extracts_usage_from_result(self, model_config: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """astream() extracts usage from the ResultMessage."""

        async def fake_query(prompt: str, options: Any) -> Any:
            yield _SDK.AssistantMessage(content=[_SDK.TextBlock("OK")])
            yield _SDK.ResultMessage(
                result="OK",
                usage={"input_tokens": 200, "output_tokens": 80},
                total_cost_usd=0.005,
            )

        monkeypatch.setattr(_SDK, "query", fake_query)

        adapter = ClaudeSDKLLMAdapter(model_config)

        async with adapter.astream([Message.user("Hi")]) as sr:
            async for _ in sr:
                pass

        assert sr.usage.input_tokens == 200
        assert sr.usage.output_tokens == 80

    @pytest.mark.asyncio
    async def test_astream_with_timeout_completes(self, model_config: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """_astream_with_timeout() completes without partial flag on fast responses."""

        async def fake_query(prompt: str, options: Any) -> Any:
            yield _SDK.AssistantMessage(content=[_SDK.TextBlock("All done.")])
            yield _SDK.ResultMessage(
                result="All done.",
                usage={"input_tokens": 10, "output_tokens": 5},
            )

        monkeypatch.setattr(_SDK, "query", fake_query)

        adapter = ClaudeSDKLLMAdapter(model_config)
        result = await adapter._astream_with_timeout([Message.user("Hi")], timeout=10.0)

        assert result.content == "All done."
        assert result.is_partial is False
        assert result.usage_unavailable is False

    @pytest.mark.asyncio
    async def test_astream_with_timeout_captures_partial(
        self, model_config: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_astream_with_timeout() sets is_partial and usage_unavailable on timeout."""

        async def slow_query(prompt: str, options: Any) -> Any:
            yield _SDK.AssistantMessage(content=[_SDK.TextBlock("First")])
            await asyncio.sleep(5.0)
            yield _SDK.AssistantMessage(content=[_SDK.TextBlock("First Second")])
            await asyncio.sleep(5.0)
            yield _SDK.ResultMessage(result="First Second Third")

        monkeypatch.setattr(_SDK, "query", slow_query)

        adapter = ClaudeSDKLLMAdapter(model_config)
        result = await adapter._astream_with_timeout([Message.user("Hi")], timeout=0.05)

        assert result.is_partial is True
        assert result.usage_unavailable is True
        assert "First" in result.content
