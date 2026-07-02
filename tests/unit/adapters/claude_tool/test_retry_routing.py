"""Retry routing tests for the claude_tool LLM adapter (T2, design decision D1).

Verifies that the text and structured invocation paths route the Anthropic
SDK calls through RetryExecutor: classified-retryable errors are retried up
to the per-category budget, a bound track_retries tracker records each
retry, and the structured-output validation loop stays outside the
transient executor (validation failures do not consume transient budgets).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, Field

from karenina.adapters.claude_tool.llm import ClaudeToolLLMAdapter
from karenina.ports import Message, ParseError
from karenina.schemas.config import ModelConfig
from karenina.utils.retry_policy import CategoryRetryConfig, RetryPolicy, track_retries

anthropic = pytest.importorskip("anthropic")


class CityAnswer(BaseModel):
    """Tiny schema for structured-output retry tests."""

    city: str = Field(description="A city name")


def _tight_policy() -> RetryPolicy:
    """Zero-backoff policy with a 3-attempt connection budget."""
    return RetryPolicy(
        connection=CategoryRetryConfig(max_attempts=3, backoff_min=0.0, backoff_max=0.0),
        timeout=CategoryRetryConfig(max_attempts=1, backoff_min=0.0, backoff_max=0.0),
        rate_limit=CategoryRetryConfig(max_attempts=0),
        server_error=CategoryRetryConfig(max_attempts=0),
    )


def _model_config() -> ModelConfig:
    """claude_tool ModelConfig with the tight retry policy bound."""
    return ModelConfig(
        id="claude-tool-retry-test",
        model_name="claude-haiku-4-5",
        model_provider="anthropic",
        interface="claude_tool",
        max_tokens=256,
        retry_policy=_tight_policy(),
    )


def _connection_error() -> Exception:
    """A real anthropic.APIConnectionError (classifies as CONNECTION)."""
    request = httpx.Request("POST", "http://localhost:8000/v1/messages")
    return anthropic.APIConnectionError(request=request)


def _text_response(text: str = "Paris") -> SimpleNamespace:
    """Minimal messages.create response shape."""
    return SimpleNamespace(
        content=[SimpleNamespace(text=text)],
        usage=SimpleNamespace(input_tokens=3, output_tokens=5),
    )


def _parsed_response(parsed: BaseModel | None) -> SimpleNamespace:
    """Minimal beta.messages.parse response shape."""
    return SimpleNamespace(
        parsed_output=parsed,
        content=[],
        usage=SimpleNamespace(input_tokens=3, output_tokens=5),
    )


class FlakyCall:
    """Async callable that fails N times with a given error, then succeeds."""

    def __init__(self, failures: int, response: Any, exc_factory: Any = _connection_error) -> None:
        self.failures = failures
        self.response = response
        self.exc_factory = exc_factory
        self.calls = 0

    async def __call__(self, **_kwargs: Any) -> Any:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc_factory()
        return self.response


def _install_client(monkeypatch: pytest.MonkeyPatch, adapter: ClaudeToolLLMAdapter, flaky: FlakyCall) -> None:
    """Wire a fake async client exposing the flaky call on both API paths."""
    client = SimpleNamespace(
        messages=SimpleNamespace(create=flaky),
        beta=SimpleNamespace(messages=SimpleNamespace(parse=flaky)),
    )
    monkeypatch.setattr(adapter, "_get_async_client", lambda: client)


@pytest.mark.unit
class TestClaudeToolTextRetryRouting:
    @pytest.mark.asyncio
    async def test_ainvoke_text_retries_transient_errors_and_records_telemetry(self, monkeypatch):
        """Two connection failures are retried and recorded, third call succeeds."""
        flaky = FlakyCall(failures=2, response=_text_response())
        adapter = ClaudeToolLLMAdapter(_model_config())
        _install_client(monkeypatch, adapter, flaky)

        with track_retries(_tight_policy()) as tracker:
            result = await adapter.ainvoke([Message.user("Capital of France?")])

        assert result.content == "Paris"
        assert result.usage.total_tokens == 8
        assert flaky.calls == 3
        assert tracker["connection"]["used"] == 2
        assert tracker["connection"]["budget"] == 3

    @pytest.mark.asyncio
    async def test_ainvoke_text_raises_after_budget_exhaustion(self, monkeypatch):
        """The budget bounds the attempts: 1 original call + 3 retries."""
        flaky = FlakyCall(failures=10, response=_text_response())
        adapter = ClaudeToolLLMAdapter(_model_config())
        _install_client(monkeypatch, adapter, flaky)

        with track_retries(_tight_policy()) as tracker, pytest.raises(anthropic.APIConnectionError):
            await adapter.ainvoke([Message.user("Capital of France?")])

        assert flaky.calls == 4
        assert tracker["connection"]["used"] == 3

    @pytest.mark.asyncio
    async def test_ainvoke_text_does_not_retry_permanent_errors(self, monkeypatch):
        """Permanent errors propagate immediately without retries."""
        flaky = FlakyCall(
            failures=10,
            response=_text_response(),
            exc_factory=lambda: ValueError("bad request payload"),
        )
        adapter = ClaudeToolLLMAdapter(_model_config())
        _install_client(monkeypatch, adapter, flaky)

        with track_retries(_tight_policy()) as tracker, pytest.raises(ValueError):
            await adapter.ainvoke([Message.user("Capital of France?")])

        assert flaky.calls == 1
        assert sum(entry["used"] for entry in tracker.values()) == 0


@pytest.mark.unit
class TestClaudeToolStructuredRetryRouting:
    @pytest.mark.asyncio
    async def test_structured_retries_transient_errors_and_records_telemetry(self, monkeypatch):
        """The beta.messages.parse path routes through RetryExecutor."""
        flaky = FlakyCall(failures=1, response=_parsed_response(CityAnswer(city="Paris")))
        adapter = ClaudeToolLLMAdapter(_model_config()).with_structured_output(CityAnswer, max_retries=0)
        _install_client(monkeypatch, adapter, flaky)

        with track_retries(_tight_policy()) as tracker:
            result = await adapter.ainvoke([Message.user("Capital of France?")])

        assert json.loads(result.content) == {"city": "Paris"}
        assert flaky.calls == 2
        assert tracker["connection"]["used"] == 1

    @pytest.mark.asyncio
    async def test_validation_failure_does_not_consume_transient_budget(self, monkeypatch):
        """Missing parsed_output is a validation failure, not a transient retry."""
        flaky = FlakyCall(failures=0, response=_parsed_response(None))
        adapter = ClaudeToolLLMAdapter(_model_config()).with_structured_output(CityAnswer, max_retries=0)
        _install_client(monkeypatch, adapter, flaky)

        with track_retries(_tight_policy()) as tracker, pytest.raises(ParseError):
            await adapter.ainvoke([Message.user("Capital of France?")])

        assert flaky.calls == 1
        assert sum(entry["used"] for entry in tracker.values()) == 0

    @pytest.mark.asyncio
    async def test_validation_retry_loop_reinvokes_without_transient_budget(self, monkeypatch):
        """Validation retries (error feedback) call again with fresh transient budgets."""
        flaky = FlakyCall(failures=0, response=_parsed_response(None))
        adapter = ClaudeToolLLMAdapter(_model_config()).with_structured_output(CityAnswer, max_retries=2)
        _install_client(monkeypatch, adapter, flaky)

        with track_retries(_tight_policy()) as tracker, pytest.raises(ParseError):
            await adapter.ainvoke([Message.user("Capital of France?")])

        # 1 original + 2 validation retries, none counted as transient
        assert flaky.calls == 3
        assert sum(entry["used"] for entry in tracker.values()) == 0
