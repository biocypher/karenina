"""Retry routing tests for the deep_agents LLM and parser adapters (T1).

Verifies that ainvoke() and aparse_to_pydantic() route their LLM calls
through RetryExecutor: classified-retryable errors are retried up to the
per-category budget, a bound track_retries tracker records each retry, and
the chat model is constructed once per call (retries repeat the API call,
not model construction).
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from karenina.adapters.langchain_deep_agents.llm import DeepAgentsLLMAdapter
from karenina.adapters.langchain_deep_agents.parser import DeepAgentsParserAdapter
from karenina.ports import Message, ParsePortResult
from karenina.schemas.config import ModelConfig
from karenina.utils.retry_policy import CategoryRetryConfig, RetryPolicy, track_retries


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
    """Deep agents ModelConfig with the tight retry policy bound."""
    return ModelConfig(
        id="deep-agents-retry-test",
        model_name="test-model",
        model_provider="openai",
        interface="langchain_deep_agents",
        temperature=0.0,
        retry_policy=_tight_policy(),
    )


class FlakyAsyncModel:
    """Stand-in chat model whose ainvoke fails N times, then succeeds."""

    def __init__(self, failures: int, response: Any, exc_factory: Any = None) -> None:
        self.failures = failures
        self.response = response
        self.exc_factory = exc_factory or (lambda: ConnectionError("connection refused"))
        self.calls = 0
        self.structured_kwargs: dict[str, Any] | None = None

    async def ainvoke(self, _lc_messages: list[Any]) -> Any:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc_factory()
        return self.response

    def with_structured_output(self, _schema: type[BaseModel], **kwargs: Any) -> FlakyAsyncModel:
        self.structured_kwargs = kwargs
        return self


@pytest.mark.unit
class TestDeepAgentsLLMRetryRouting:
    def _install_factory(self, monkeypatch: pytest.MonkeyPatch, model: FlakyAsyncModel) -> dict[str, int]:
        """Patch create_chat_model to return ``model`` and count constructions."""
        constructions = {"count": 0}

        def _factory(_config: ModelConfig, **_kw: Any) -> FlakyAsyncModel:
            constructions["count"] += 1
            return model

        monkeypatch.setattr("karenina.adapters.langchain_deep_agents.llm.create_chat_model", _factory)
        return constructions

    @pytest.mark.asyncio
    async def test_ainvoke_retries_transient_errors_and_records_telemetry(self, monkeypatch):
        """Two connection failures are retried and recorded, third call succeeds."""
        model = FlakyAsyncModel(failures=2, response=AIMessage(content="Paris"))
        self._install_factory(monkeypatch, model)
        adapter = DeepAgentsLLMAdapter(_model_config())

        with track_retries(_tight_policy()) as tracker:
            result = await adapter.ainvoke([Message.user("Capital of France?")])

        assert result.content == "Paris"
        assert model.calls == 3
        assert tracker["connection"]["used"] == 2
        assert tracker["connection"]["budget"] == 3

    @pytest.mark.asyncio
    async def test_ainvoke_constructs_model_once_across_retries(self, monkeypatch):
        """Retries repeat the API call, not model construction."""
        model = FlakyAsyncModel(failures=2, response=AIMessage(content="Paris"))
        constructions = self._install_factory(monkeypatch, model)
        adapter = DeepAgentsLLMAdapter(_model_config())

        await adapter.ainvoke([Message.user("Capital of France?")])

        assert constructions["count"] == 1
        assert model.calls == 3

    @pytest.mark.asyncio
    async def test_ainvoke_raises_after_budget_exhaustion(self, monkeypatch):
        """The budget bounds the attempts: 1 original call + 3 retries."""
        model = FlakyAsyncModel(failures=10, response=AIMessage(content="never"))
        self._install_factory(monkeypatch, model)
        adapter = DeepAgentsLLMAdapter(_model_config())

        with track_retries(_tight_policy()) as tracker, pytest.raises(ConnectionError):
            await adapter.ainvoke([Message.user("Capital of France?")])

        assert model.calls == 4
        assert tracker["connection"]["used"] == 3

    @pytest.mark.asyncio
    async def test_ainvoke_does_not_retry_permanent_errors(self, monkeypatch):
        """Permanent errors propagate immediately without retries."""
        model = FlakyAsyncModel(
            failures=10,
            response=AIMessage(content="never"),
            exc_factory=lambda: ValueError("bad request payload"),
        )
        self._install_factory(monkeypatch, model)
        adapter = DeepAgentsLLMAdapter(_model_config())

        with track_retries(_tight_policy()) as tracker, pytest.raises(ValueError):
            await adapter.ainvoke([Message.user("Capital of France?")])

        assert model.calls == 1
        assert sum(entry["used"] for entry in tracker.values()) == 0

    @pytest.mark.asyncio
    async def test_structured_ainvoke_retries_transient_errors(self, monkeypatch):
        """The structured-output path routes through RetryExecutor too."""
        parsed = CityAnswer(city="Paris")
        model = FlakyAsyncModel(failures=1, response=parsed)
        self._install_factory(monkeypatch, model)
        adapter = DeepAgentsLLMAdapter(_model_config()).with_structured_output(CityAnswer)

        with track_retries(_tight_policy()) as tracker:
            result = await adapter.ainvoke([Message.user("Capital of France?")])

        assert json.loads(result.content) == {"city": "Paris"}
        assert model.calls == 2
        assert tracker["connection"]["used"] == 1


@pytest.mark.unit
class TestDeepAgentsParserRetryRouting:
    def _install_factory(self, monkeypatch: pytest.MonkeyPatch, model: FlakyAsyncModel) -> dict[str, int]:
        """Patch the parser module's create_chat_model and count constructions."""
        constructions = {"count": 0}

        def _factory(_config: ModelConfig, **_kw: Any) -> FlakyAsyncModel:
            constructions["count"] += 1
            return model

        monkeypatch.setattr("karenina.adapters.langchain_deep_agents.parser.create_chat_model", _factory)
        return constructions

    @staticmethod
    def _messages() -> list[Message]:
        return [
            Message.system("Extract the city."),
            Message.user("The capital of France is Paris."),
        ]

    @pytest.mark.asyncio
    async def test_aparse_retries_transient_errors_and_records_telemetry(self, monkeypatch):
        """Structured parse retries connection errors and records them."""
        raw_response = {
            "raw": AIMessage(content='{"city": "Paris"}'),
            "parsed": CityAnswer(city="Paris"),
            "parsing_error": None,
        }
        model = FlakyAsyncModel(failures=2, response=raw_response)
        constructions = self._install_factory(monkeypatch, model)
        parser = DeepAgentsParserAdapter(_model_config())

        with track_retries(_tight_policy()) as tracker:
            result = await parser.aparse_to_pydantic(self._messages(), CityAnswer)

        assert isinstance(result, ParsePortResult)
        assert result.parsed.city == "Paris"
        assert model.calls == 3
        assert constructions["count"] == 1
        assert tracker["connection"]["used"] == 2

    @pytest.mark.asyncio
    async def test_text_fallback_path_retries_transient_errors(self, monkeypatch):
        """When structured output is unavailable, the text fallback retries too."""

        class FallbackModel(FlakyAsyncModel):
            def with_structured_output(self, _schema: type[BaseModel], **_kwargs: Any) -> FlakyAsyncModel:
                raise TypeError("structured output not supported by this model")

        model = FallbackModel(failures=1, response=AIMessage(content='{"city": "Paris"}'))
        self._install_factory(monkeypatch, model)
        parser = DeepAgentsParserAdapter(_model_config())

        with track_retries(_tight_policy()) as tracker:
            result = await parser.aparse_to_pydantic(self._messages(), CityAnswer)

        assert result.parsed.city == "Paris"
        assert model.calls == 2
        assert tracker["connection"]["used"] == 1

    @pytest.mark.asyncio
    async def test_exhausted_transient_error_skips_text_fallback(self, monkeypatch):
        """A dead-endpoint style error re-raises instead of paying a second
        retry budget on the text fallback."""
        model = FlakyAsyncModel(failures=10, response=None)
        self._install_factory(monkeypatch, model)
        parser = DeepAgentsParserAdapter(_model_config())

        with track_retries(_tight_policy()) as tracker, pytest.raises(ConnectionError):
            await parser.aparse_to_pydantic(self._messages(), CityAnswer)

        # Only the structured attempt's budget is spent: 1 original call
        # plus 3 retries. The text fallback never runs.
        assert model.calls == 4
        assert tracker["connection"]["used"] == 3
