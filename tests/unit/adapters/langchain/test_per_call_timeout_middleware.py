"""Tests for PerCallTimeoutMiddleware (issue 195).

The LangChain agent loop runs many ``model.ainvoke()`` calls under a single
outer ``asyncio.wait_for`` that budgets the full ``agent_timeout``. If a
single call wedges silently, the only safety net is that outer budget,
which pays a full timeout per rescue. Issue 195 documented 2-in-576 such
wedges on vLLM streaming responses where httpx's read timeout never fired.

PerCallTimeoutMiddleware wraps every ``handler(request)`` invocation in
an ``asyncio.wait_for`` so each individual model call is bounded. When
the budget expires the stock ``asyncio.TimeoutError`` propagates to the
outer ``ModelRetryMiddleware`` which retries with backoff, mirroring the
LangChainLLMAdapter._ainvoke_with_timeout pattern at the agent layer.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from karenina.adapters.langchain.middleware import (
    PerCallTimeoutMiddleware,
    build_agent_middleware,
)


class _FakeRequest:
    """Minimal stand-in for langchain.agents.middleware.types.ModelRequest."""


class _FakeResponse:
    """Minimal stand-in for ModelResponse."""

    def __init__(self, value: str = "ok") -> None:
        self.value = value


@pytest.mark.unit
class TestPerCallTimeoutMiddleware:
    """Direct tests on PerCallTimeoutMiddleware.awrap_model_call."""

    def test_rejects_non_positive_timeout(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            PerCallTimeoutMiddleware(timeout=0)
        with pytest.raises(ValueError, match="must be positive"):
            PerCallTimeoutMiddleware(timeout=-1.5)

    @pytest.mark.asyncio
    async def test_passes_through_fast_call(self) -> None:
        """A handler that completes quickly should return unchanged."""
        mw = PerCallTimeoutMiddleware(timeout=1.0)
        expected = _FakeResponse("fast")

        async def handler(_req: Any) -> _FakeResponse:
            return expected

        result = await mw.awrap_model_call(_FakeRequest(), handler)
        assert result is expected

    @pytest.mark.asyncio
    async def test_raises_timeout_on_slow_call(self) -> None:
        """A handler that blocks longer than timeout raises asyncio.TimeoutError."""
        mw = PerCallTimeoutMiddleware(timeout=0.05)

        async def handler(_req: Any) -> _FakeResponse:
            await asyncio.sleep(5.0)  # deliberately longer than timeout
            return _FakeResponse("should-not-reach")

        with pytest.raises(asyncio.TimeoutError):
            await mw.awrap_model_call(_FakeRequest(), handler)

    @pytest.mark.asyncio
    async def test_cancels_underlying_handler_on_timeout(self) -> None:
        """When the budget fires, the underlying handler coroutine is cancelled."""
        mw = PerCallTimeoutMiddleware(timeout=0.05)
        cancelled = asyncio.Event()

        async def handler(_req: Any) -> _FakeResponse:
            try:
                await asyncio.sleep(5.0)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return _FakeResponse("should-not-reach")

        with pytest.raises(asyncio.TimeoutError):
            await mw.awrap_model_call(_FakeRequest(), handler)
        # Give the event loop a tick to observe the cancellation
        await asyncio.sleep(0)
        assert cancelled.is_set(), "handler was not cancelled on timeout"

    @pytest.mark.asyncio
    async def test_propagates_handler_exceptions_unchanged(self) -> None:
        """A non-timeout exception from the handler should propagate untouched."""
        mw = PerCallTimeoutMiddleware(timeout=1.0)

        class _BoomError(RuntimeError):
            pass

        async def handler(_req: Any) -> _FakeResponse:
            raise _BoomError("synthetic")

        with pytest.raises(_BoomError, match="synthetic"):
            await mw.awrap_model_call(_FakeRequest(), handler)

    def test_sync_wrap_is_passthrough(self) -> None:
        """The sync path is intentionally a passthrough (see class docstring)."""
        mw = PerCallTimeoutMiddleware(timeout=0.001)
        sentinel = object()
        called = {"n": 0}

        def handler(_req: Any) -> object:
            called["n"] += 1
            return sentinel

        result = mw.wrap_model_call(_FakeRequest(), handler)
        assert result is sentinel
        assert called["n"] == 1


def _build_config_without_summarization() -> Any:
    """Return an AgentMiddlewareConfig with summarization disabled.

    ``build_agent_middleware`` otherwise raises when no base_model is
    provided, because the default summarization middleware needs a model
    instance. These tests only care about timeout wiring.
    """
    from karenina.schemas.config import AgentMiddlewareConfig

    cfg = AgentMiddlewareConfig()
    cfg.summarization.enabled = False
    return cfg


@pytest.mark.unit
class TestBuildAgentMiddlewareTimeoutWiring:
    """Tests that build_agent_middleware installs the timeout MW correctly."""

    def test_not_added_when_request_timeout_is_none(self) -> None:
        middleware = build_agent_middleware(
            config=_build_config_without_summarization(),
            request_timeout=None,
        )
        assert not any(isinstance(m, PerCallTimeoutMiddleware) for m in middleware)

    def test_not_added_when_request_timeout_is_zero(self) -> None:
        middleware = build_agent_middleware(
            config=_build_config_without_summarization(),
            request_timeout=0.0,
        )
        assert not any(isinstance(m, PerCallTimeoutMiddleware) for m in middleware)

    def test_not_added_when_request_timeout_is_negative(self) -> None:
        middleware = build_agent_middleware(
            config=_build_config_without_summarization(),
            request_timeout=-10.0,
        )
        assert not any(isinstance(m, PerCallTimeoutMiddleware) for m in middleware)

    def test_added_with_configured_timeout_when_request_timeout_is_positive(self) -> None:
        middleware = build_agent_middleware(
            config=_build_config_without_summarization(),
            request_timeout=45.5,
        )
        timeout_mws = [m for m in middleware if isinstance(m, PerCallTimeoutMiddleware)]
        assert len(timeout_mws) == 1
        assert timeout_mws[0].timeout == 45.5

    def test_timeout_middleware_is_inner_to_model_retry_middleware(self) -> None:
        """PerCallTimeoutMiddleware must sit INSIDE ModelRetryMiddleware.

        The outer ModelRetryMiddleware catches asyncio.TimeoutError and retries
        the handler. If the order were reversed, the retry layer could not
        observe timeouts and the guardrail would not compose with backoff.
        """
        from langchain.agents.middleware import ModelRetryMiddleware

        middleware = build_agent_middleware(
            config=_build_config_without_summarization(),
            request_timeout=30.0,
        )
        retry_idx = next(i for i, m in enumerate(middleware) if isinstance(m, ModelRetryMiddleware))
        timeout_idx = next(i for i, m in enumerate(middleware) if isinstance(m, PerCallTimeoutMiddleware))
        assert retry_idx < timeout_idx, (
            f"ModelRetryMiddleware (idx {retry_idx}) must come before "
            f"PerCallTimeoutMiddleware (idx {timeout_idx}) so retry is the outer layer"
        )


@pytest.mark.unit
class TestAgentCreateAgentPassesRequestTimeout:
    """Integration: LangChainAgentAdapter._create_agent wires request_timeout through."""

    def test_create_agent_passes_request_timeout_to_build_agent_middleware(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from karenina.adapters.langchain import agent as agent_module
        from karenina.adapters.langchain.agent import LangChainAgentAdapter
        from karenina.schemas.config import ModelConfig

        config = ModelConfig(
            id="test",
            model_name="test-model",
            model_provider="openai",
            interface="langchain",
            request_timeout=42.0,
        )

        adapter = LangChainAgentAdapter(config)

        captured_kwargs: dict[str, Any] = {}

        def fake_init_chat_model_unified(**kwargs: Any) -> Any:
            return MagicMock(name="base_model")

        def fake_build_agent_middleware(**kwargs: Any) -> list[Any]:
            captured_kwargs.update(kwargs)
            return []

        def fake_create_agent(**_kwargs: Any) -> Any:
            return MagicMock(name="agent")

        # Patch the three things _create_agent imports via lazy imports
        monkeypatch.setattr(
            "karenina.adapters.langchain.initialization.init_chat_model_unified",
            fake_init_chat_model_unified,
        )
        monkeypatch.setattr(
            "karenina.adapters.langchain.middleware.build_agent_middleware",
            fake_build_agent_middleware,
        )
        monkeypatch.setattr(
            "langchain.agents.create_agent",
            fake_create_agent,
        )
        # InMemorySaver is also imported inside _create_agent
        fake_memory = MagicMock(name="InMemorySaver")
        monkeypatch.setattr(
            "langgraph.checkpoint.memory.InMemorySaver",
            lambda: fake_memory,
        )
        # Prevent the openai_endpoint branch from trying to call the network
        monkeypatch.setattr(
            "karenina.adapters.langchain.middleware.fetch_openai_endpoint_context_size",
            lambda **_kwargs: None,
        )
        # Silence module-level references just in case
        _ = agent_module

        adapter._create_agent(tools=[], kwargs={})

        assert captured_kwargs.get("request_timeout") == 42.0
