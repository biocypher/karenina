"""Conformance: LLM adapter retries route through RetryExecutor.

Asserts a shared contract across the single-turn LLM adapters that own
their transient retries (langchain, claude_tool, langchain_deep_agents):
a classified-retryable error raised by the underlying SDK call is retried
by RetryExecutor (not an SDK-internal layer), so a bound track_retries
tracker observes every retry. This is the telemetry guarantee that the
live B9 suite checks against a dead port.

The claude_agent_sdk LLM adapter is excluded by design: its subprocess
query() transport is out of scope for retry routing (spec decision D1).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from karenina.ports import Message
from karenina.schemas.config import ModelConfig
from karenina.utils.retry_policy import CategoryRetryConfig, RetryPolicy, track_retries

FAILURES = 2


def _tight_policy() -> RetryPolicy:
    """Zero-backoff policy with a 3-attempt connection budget."""
    return RetryPolicy(
        connection=CategoryRetryConfig(max_attempts=3, backoff_min=0.0, backoff_max=0.0),
        timeout=CategoryRetryConfig(max_attempts=1, backoff_min=0.0, backoff_max=0.0),
        rate_limit=CategoryRetryConfig(max_attempts=0),
        server_error=CategoryRetryConfig(max_attempts=0),
    )


class _Counter:
    """Shared mutable call counter for the flaky fakes."""

    def __init__(self) -> None:
        self.calls = 0


def _make_langchain(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, _Counter]:
    """LangChainLLMAdapter over a flaky injected base model."""
    pytest.importorskip("langchain_core", reason="langchain-core not installed")
    from langchain_core.messages import AIMessage

    from karenina.adapters.langchain.llm import LangChainLLMAdapter

    counter = _Counter()

    class FlakyModel:
        async def ainvoke(self, _lc_messages: list[Any]) -> Any:
            counter.calls += 1
            if counter.calls <= FAILURES:
                raise ConnectionError("connection refused")
            return AIMessage(content="Paris")

    config = ModelConfig(
        id="conformance-langchain",
        model_name="test-model",
        model_provider="openai",
        interface="langchain",
        retry_policy=_tight_policy(),
    )
    return LangChainLLMAdapter(config, _base_model=FlakyModel()), counter


def _make_claude_tool(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, _Counter]:
    """ClaudeToolLLMAdapter over a flaky fake async client."""
    pytest.importorskip("anthropic", reason="anthropic not installed")
    from karenina.adapters.claude_tool.llm import ClaudeToolLLMAdapter

    counter = _Counter()

    async def flaky_create(**_kwargs: Any) -> Any:
        counter.calls += 1
        if counter.calls <= FAILURES:
            raise ConnectionError("connection refused")
        return SimpleNamespace(
            content=[SimpleNamespace(text="Paris")],
            usage=SimpleNamespace(input_tokens=3, output_tokens=5),
        )

    config = ModelConfig(
        id="conformance-claude-tool",
        model_name="claude-haiku-4-5",
        model_provider="anthropic",
        interface="claude_tool",
        max_tokens=256,
        retry_policy=_tight_policy(),
    )
    adapter = ClaudeToolLLMAdapter(config)
    client = SimpleNamespace(messages=SimpleNamespace(create=flaky_create))
    monkeypatch.setattr(adapter, "_get_async_client", lambda: client)
    return adapter, counter


def _make_deep_agents(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, _Counter]:
    """DeepAgentsLLMAdapter over a flaky model from a patched factory."""
    pytest.importorskip("deepagents", reason="deepagents not installed")
    pytest.importorskip("langchain_core", reason="langchain-core not installed")
    from langchain_core.messages import AIMessage

    from karenina.adapters.langchain_deep_agents.llm import DeepAgentsLLMAdapter

    counter = _Counter()

    class FlakyModel:
        async def ainvoke(self, _lc_messages: list[Any]) -> Any:
            counter.calls += 1
            if counter.calls <= FAILURES:
                raise ConnectionError("connection refused")
            return AIMessage(content="Paris")

    monkeypatch.setattr(
        "karenina.adapters.langchain_deep_agents.llm.create_chat_model",
        lambda _config, **_kw: FlakyModel(),
    )
    config = ModelConfig(
        id="conformance-deep-agents",
        model_name="test-model",
        model_provider="openai",
        interface="langchain_deep_agents",
        retry_policy=_tight_policy(),
    )
    return DeepAgentsLLMAdapter(config), counter


ADAPTER_FACTORIES = [
    pytest.param(_make_langchain, id="langchain"),
    pytest.param(_make_claude_tool, id="claude_tool"),
    pytest.param(_make_deep_agents, id="langchain_deep_agents"),
]


@pytest.mark.unit
class TestRetryRoutingConformance:
    """All retry-owning LLM adapters surface retries to track_retries."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("factory", ADAPTER_FACTORIES)
    async def test_ainvoke_retries_route_through_retry_executor(self, factory, monkeypatch):
        """Classified-retryable errors increment the bound tracker on every adapter."""
        adapter, counter = factory(monkeypatch)

        with track_retries(_tight_policy()) as tracker:
            result = await adapter.ainvoke([Message.user("Capital of France?")])

        assert "Paris" in result.content
        assert counter.calls == FAILURES + 1
        assert tracker["connection"]["used"] == FAILURES
