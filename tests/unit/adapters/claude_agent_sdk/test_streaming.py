"""Tests for ClaudeSDKLLMAdapter streaming methods.

Tests astream() and _astream_with_timeout() using mocked claude_agent_sdk.query().
"""

from __future__ import annotations

import asyncio
import contextlib
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

        def __getattr__(self, name: str) -> None:
            # Return None for any attribute not explicitly set (matches real SDK defaults)
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

    class StreamEvent:
        def __init__(self, event: dict[str, Any] | None = None) -> None:
            self.event = event or {}

    mod = types.ModuleType("claude_agent_sdk")
    for _name, _obj in [
        ("ClaudeAgentOptions", ClaudeAgentOptions),
        ("TextBlock", TextBlock),
        ("AssistantMessage", AssistantMessage),
        ("ResultMessage", ResultMessage),
        ("StreamEvent", StreamEvent),
        # query is a sentinel; every test replaces it via monkeypatch before use.
        ("query", None),
    ]:
        setattr(mod, _name, _obj)
    return mod


# Install the stub so that `from claude_agent_sdk import ...` inside the adapter
# resolves all names. We eagerly try to import the real SDK first so the
# original module (if installed) is preserved and restored after these tests
# finish. Without this preflight, the stub would shadow the real package even
# in environments where it is installed, causing other test files that depend
# on the real SDK to fail or be incorrectly skipped.
_SDK = _build_fake_sdk()

with contextlib.suppress(ImportError):
    import claude_agent_sdk as _real_sdk  # noqa: F401

_ORIGINAL_SDK = sys.modules.get("claude_agent_sdk")
sys.modules["claude_agent_sdk"] = _SDK


from karenina.adapters.claude_agent_sdk.llm import ClaudeSDKLLMAdapter  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_sdk_module():
    """Ensure the real SDK module is restored after each test."""
    sys.modules["claude_agent_sdk"] = _SDK
    yield
    if _ORIGINAL_SDK is not None:
        sys.modules["claude_agent_sdk"] = _ORIGINAL_SDK
    else:
        sys.modules.pop("claude_agent_sdk", None)


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
    async def test_astream_with_timeout_raises_streaming_timeout_error(
        self, model_config: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_astream_with_timeout() raises StreamingTimeoutError on timeout."""
        from karenina.exceptions import StreamingTimeoutError

        async def slow_query(prompt: str, options: Any) -> Any:
            yield _SDK.AssistantMessage(content=[_SDK.TextBlock("First")])
            await asyncio.sleep(5.0)
            yield _SDK.AssistantMessage(content=[_SDK.TextBlock("First Second")])
            await asyncio.sleep(5.0)
            yield _SDK.ResultMessage(result="First Second Third")

        monkeypatch.setattr(_SDK, "query", slow_query)

        adapter = ClaudeSDKLLMAdapter(model_config)
        with pytest.raises(StreamingTimeoutError) as exc_info:
            await adapter._astream_with_timeout([Message.user("Hi")], timeout=0.05)

        assert "First" in exc_info.value.partial_content

    @pytest.mark.asyncio
    async def test_astream_captures_usage_from_stream_events(
        self, model_config: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Raw StreamEvent message_start and message_delta usage is captured inline.

        No ResultMessage is emitted (as on a mid-stream cut), so the
        inline StreamEvent snapshot is the only usage signal.
        """

        async def fake_query(prompt: str, options: Any) -> Any:
            yield _SDK.StreamEvent(
                event={
                    "type": "message_start",
                    "message": {"id": "msg_1", "usage": {"input_tokens": 75, "output_tokens": 1}},
                }
            )
            yield _SDK.AssistantMessage(content=[_SDK.TextBlock("partial text")])
            yield _SDK.StreamEvent(event={"type": "message_delta", "usage": {"output_tokens": 42}})

        monkeypatch.setattr(_SDK, "query", fake_query)

        adapter = ClaudeSDKLLMAdapter(model_config)

        async with adapter.astream([Message.user("Hi")]) as sr:
            async for _ in sr:
                pass

        assert sr.accumulated_content == "partial text"
        assert sr.usage.input_tokens == 75
        assert sr.usage.output_tokens == 42

    @pytest.mark.asyncio
    async def test_astream_result_message_overrides_stream_event_usage(
        self, model_config: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On the success path the ResultMessage usage wins over inline events."""

        async def fake_query(prompt: str, options: Any) -> Any:
            yield _SDK.StreamEvent(
                event={
                    "type": "message_start",
                    "message": {"id": "msg_1", "usage": {"input_tokens": 75, "output_tokens": 1}},
                }
            )
            yield _SDK.AssistantMessage(content=[_SDK.TextBlock("done")])
            yield _SDK.ResultMessage(
                result="done",
                usage={"input_tokens": 80, "output_tokens": 20},
            )

        monkeypatch.setattr(_SDK, "query", fake_query)

        adapter = ClaudeSDKLLMAdapter(model_config)

        async with adapter.astream([Message.user("Hi")]) as sr:
            async for _ in sr:
                pass

        assert sr.usage.input_tokens == 80
        assert sr.usage.output_tokens == 20
