"""Retry routing tests for the claude_agent_sdk parser (T2, design decision D1).

Verifies that _parse_via_anthropic and _parse_via_openai route their API
calls through RetryExecutor: classified-retryable errors are retried up to
the per-category budget and a bound track_retries tracker records each
retry. Exhausted transient errors still surface as ParseError (the public
parser contract).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, Field

from karenina.adapters.claude_agent_sdk.parser import ClaudeSDKParserAdapter
from karenina.ports import Message, ParseError
from karenina.schemas.config import ModelConfig
from karenina.utils.retry_policy import CategoryRetryConfig, RetryPolicy, track_retries

anthropic = pytest.importorskip("anthropic")
openai = pytest.importorskip("openai")


class CityAnswer(BaseModel):
    """Tiny schema for parser retry tests."""

    city: str = Field(description="A city name")


def _tight_policy() -> RetryPolicy:
    """Zero-backoff policy with a 3-attempt connection budget."""
    return RetryPolicy(
        connection=CategoryRetryConfig(max_attempts=3, backoff_min=0.0, backoff_max=0.0),
        timeout=CategoryRetryConfig(max_attempts=1, backoff_min=0.0, backoff_max=0.0),
        rate_limit=CategoryRetryConfig(max_attempts=0),
        server_error=CategoryRetryConfig(max_attempts=0),
    )


def _model_config(*, anthropic_base_url: str | None = None) -> ModelConfig:
    """claude_agent_sdk ModelConfig with the tight retry policy bound."""
    return ModelConfig(
        id="claude-sdk-parser-retry-test",
        model_name="claude-haiku-4-5",
        model_provider="anthropic",
        interface="claude_agent_sdk",
        max_tokens=256,
        retry_policy=_tight_policy(),
        anthropic_base_url=anthropic_base_url,
    )


def _messages() -> list[Message]:
    return [
        Message.system("Extract the city."),
        Message.user("The capital of France is Paris."),
    ]


def _anthropic_connection_error() -> Exception:
    request = httpx.Request("POST", "http://localhost:8000/v1/messages")
    return anthropic.APIConnectionError(request=request)


def _openai_connection_error() -> Exception:
    request = httpx.Request("POST", "http://localhost:8000/v1/chat/completions")
    return openai.APIConnectionError(request=request)


def _anthropic_response(text: str = '{"city": "Paris"}') -> SimpleNamespace:
    """Minimal anthropic messages.create response shape."""
    return SimpleNamespace(
        content=[SimpleNamespace(text=text)],
        usage=SimpleNamespace(input_tokens=3, output_tokens=5),
        model="claude-haiku-4-5",
    )


def _openai_response(text: str = '{"city": "Paris"}') -> SimpleNamespace:
    """Minimal openai chat.completions.create response shape."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=5, total_tokens=8),
        model="claude-haiku-4-5",
    )


class FlakyCall:
    """Async callable that fails N times with a given error, then succeeds."""

    def __init__(self, failures: int, response: Any, exc_factory: Any) -> None:
        self.failures = failures
        self.response = response
        self.exc_factory = exc_factory
        self.calls = 0

    async def __call__(self, **_kwargs: Any) -> Any:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc_factory()
        return self.response


@pytest.mark.unit
class TestClaudeSDKParserAnthropicRetryRouting:
    def _install(self, monkeypatch: pytest.MonkeyPatch, parser: ClaudeSDKParserAdapter, flaky: FlakyCall) -> None:
        client = SimpleNamespace(messages=SimpleNamespace(create=flaky))
        monkeypatch.setattr(parser, "_get_anthropic_client", lambda: client)

    @pytest.mark.asyncio
    async def test_parse_via_anthropic_retries_and_records_telemetry(self, monkeypatch):
        """Two connection failures are retried and recorded, third call succeeds."""
        flaky = FlakyCall(failures=2, response=_anthropic_response(), exc_factory=_anthropic_connection_error)
        parser = ClaudeSDKParserAdapter(_model_config())
        self._install(monkeypatch, parser, flaky)

        with track_retries(_tight_policy()) as tracker:
            result = await parser.aparse_to_pydantic(_messages(), CityAnswer)

        assert result.parsed.city == "Paris"
        assert flaky.calls == 3
        assert tracker["connection"]["used"] == 2
        assert tracker["connection"]["budget"] == 3

    @pytest.mark.asyncio
    async def test_parse_via_anthropic_exhaustion_raises_parse_error(self, monkeypatch):
        """Exhausted transient retries surface as ParseError, telemetry intact."""
        flaky = FlakyCall(failures=10, response=_anthropic_response(), exc_factory=_anthropic_connection_error)
        parser = ClaudeSDKParserAdapter(_model_config())
        self._install(monkeypatch, parser, flaky)

        with track_retries(_tight_policy()) as tracker, pytest.raises(ParseError):
            await parser.aparse_to_pydantic(_messages(), CityAnswer)

        assert flaky.calls == 4
        assert tracker["connection"]["used"] == 3


@pytest.mark.unit
class TestClaudeSDKParserOpenAIRetryRouting:
    def _install(self, monkeypatch: pytest.MonkeyPatch, parser: ClaudeSDKParserAdapter, flaky: FlakyCall) -> None:
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=flaky)))
        monkeypatch.setattr(parser, "_get_openai_client", lambda: client)

    @pytest.mark.asyncio
    async def test_parse_via_openai_retries_and_records_telemetry(self, monkeypatch):
        """The custom-endpoint OpenAI path routes through RetryExecutor too."""
        flaky = FlakyCall(failures=2, response=_openai_response(), exc_factory=_openai_connection_error)
        parser = ClaudeSDKParserAdapter(_model_config(anthropic_base_url="http://localhost:8000"))
        self._install(monkeypatch, parser, flaky)

        with track_retries(_tight_policy()) as tracker:
            result = await parser.aparse_to_pydantic(_messages(), CityAnswer)

        assert result.parsed.city == "Paris"
        assert flaky.calls == 3
        assert tracker["connection"]["used"] == 2

    @pytest.mark.asyncio
    async def test_parse_via_openai_does_not_retry_permanent_errors(self, monkeypatch):
        """Permanent errors surface as ParseError immediately without retries."""
        flaky = FlakyCall(
            failures=10,
            response=_openai_response(),
            exc_factory=lambda: ValueError("bad request payload"),
        )
        parser = ClaudeSDKParserAdapter(_model_config(anthropic_base_url="http://localhost:8000"))
        self._install(monkeypatch, parser, flaky)

        with track_retries(_tight_policy()) as tracker, pytest.raises(ParseError):
            await parser.aparse_to_pydantic(_messages(), CityAnswer)

        assert flaky.calls == 1
        assert sum(entry["used"] for entry in tracker.values()) == 0
