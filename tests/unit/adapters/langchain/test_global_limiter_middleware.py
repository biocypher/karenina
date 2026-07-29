"""Tests for GlobalLimiterMiddleware (T13, design decision D2).

LangGraph agent loops call the raw LangChain model directly, bypassing the
adapter ainvoke leaves where the GlobalLLMLimiter permit is normally
borrowed. GlobalLimiterMiddleware puts each model call inside the agent
loop under ``async with limiter.borrow()`` so agent model calls count
against ``max_concurrent_requests``. The borrow is a no-op while no batch
has configured the limiter, so the middleware is attached unconditionally.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from karenina.adapters.langchain.middleware import (
    GlobalLimiterMiddleware,
    build_agent_middleware,
)
from karenina.benchmark.verification.async_lifecycle import get_global_llm_limiter


class _FakeRequest:
    """Minimal stand-in for langchain.agents.middleware.types.ModelRequest."""


class _InFlight:
    """Thread-safe in-flight counter."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current = 0
        self.max_observed = 0
        self.total = 0

    def enter(self) -> None:
        with self._lock:
            self._current += 1
            self.total += 1
            self.max_observed = max(self.max_observed, self._current)

    def exit(self) -> None:
        with self._lock:
            self._current -= 1


@pytest.mark.unit
class TestGlobalLimiterMiddleware:
    """Direct tests on GlobalLimiterMiddleware.awrap_model_call."""

    @pytest.mark.asyncio
    async def test_passthrough_when_unconfigured(self) -> None:
        """Without an active configure, the handler runs unimpeded."""
        mw = GlobalLimiterMiddleware()
        calls: list[Any] = []

        async def handler(req: Any) -> str:
            calls.append(req)
            return "ok"

        request = _FakeRequest()
        result = await mw.awrap_model_call(request, handler)
        assert result == "ok"
        assert calls == [request]
        # The singleton limiter was never activated.
        assert get_global_llm_limiter().capacity is None

    @pytest.mark.asyncio
    async def test_borrows_per_call_and_serializes_at_cap_one(self) -> None:
        """Each model call borrows one permit: cap=1 serializes handlers."""
        mw = GlobalLimiterMiddleware()
        counter = _InFlight()

        async def handler(_req: Any) -> str:
            counter.enter()
            try:
                await asyncio.sleep(0.05)
            finally:
                counter.exit()
            return "ok"

        with get_global_llm_limiter().configure(1):
            results = await asyncio.gather(*(mw.awrap_model_call(_FakeRequest(), handler) for _ in range(3)))

        assert results == ["ok", "ok", "ok"]
        assert counter.total == 3
        assert counter.max_observed == 1

    @pytest.mark.asyncio
    async def test_handler_exception_releases_permit(self) -> None:
        """A failing model call must not leak its permit."""
        mw = GlobalLimiterMiddleware()

        async def failing_handler(_req: Any) -> str:
            raise RuntimeError("model exploded")

        limiter = get_global_llm_limiter()
        with limiter.configure(1):
            with pytest.raises(RuntimeError, match="model exploded"):
                await mw.awrap_model_call(_FakeRequest(), failing_handler)
            # Permit recovered: a fresh borrow succeeds promptly.
            async with asyncio.timeout(5):
                async with limiter.borrow():
                    pass

    def test_sync_wrap_is_passthrough(self) -> None:
        """The sync path delegates without touching the limiter."""
        mw = GlobalLimiterMiddleware()
        calls: list[Any] = []

        def handler(req: Any) -> str:
            calls.append(req)
            return "sync-ok"

        request = _FakeRequest()
        assert mw.wrap_model_call(request, handler) == "sync-ok"
        assert calls == [request]


def _build_config_without_summarization() -> Any:
    """Return an AgentMiddlewareConfig with summarization disabled.

    ``build_agent_middleware`` otherwise raises when no base_model is
    provided, because the default summarization middleware needs a model
    instance. These tests only care about limiter wiring.
    """
    from karenina.schemas.config import AgentMiddlewareConfig

    cfg = AgentMiddlewareConfig()
    cfg.summarization.enabled = False
    return cfg


@pytest.mark.unit
class TestBuildAgentMiddlewareRegistration:
    """build_agent_middleware attaches the limiter middleware unconditionally."""

    def test_added_unconditionally(self) -> None:
        middleware = build_agent_middleware(config=_build_config_without_summarization())
        limiter_mws = [m for m in middleware if isinstance(m, GlobalLimiterMiddleware)]
        assert len(limiter_mws) == 1

    def test_added_last_so_it_is_innermost(self) -> None:
        """The limiter must wrap only the innermost wire call.

        LangGraph composes wrap_model_call with earlier list entries as
        outer layers, so being last keeps retry backoff and per-call
        timeout bookkeeping outside the permit hold.
        """
        middleware = build_agent_middleware(
            config=_build_config_without_summarization(),
            request_timeout=30.0,
        )
        assert isinstance(middleware[-1], GlobalLimiterMiddleware)

    def test_added_even_with_request_timeout_disabled(self) -> None:
        middleware = build_agent_middleware(
            config=_build_config_without_summarization(),
            request_timeout=None,
        )
        assert isinstance(middleware[-1], GlobalLimiterMiddleware)
