"""Retry telemetry propagation across dispatch boundaries (T3).

Covers the track_retries guarantee documented in retry_policy.py:
(a) the sync wrappers' ThreadPoolExecutor fallback (sync invoke called
    while a loop runs in the calling thread) records retries, because the
    fallback re-enters a copy of the caller's context in the fresh thread,
(b) the BlockingPortal path still records (regression guard: anyio copies
    the caller's contextvars into the portal task),
(c) concurrent trackers stay isolated between contexts, and shared-tracker
    increments are thread-safe under the _RECORD_RETRY_LOCK.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
from typing import Any

import pytest

from karenina.ports import Message
from karenina.schemas.config import ModelConfig
from karenina.utils.errors import ErrorCategory
from karenina.utils.retry_policy import (
    CategoryRetryConfig,
    RetryPolicy,
    _record_retry,
    track_retries,
)

pytest.importorskip("langchain_core", reason="langchain-core not installed")

FAILURES = 2


def _tight_policy() -> RetryPolicy:
    """Zero-backoff policy with a 3-attempt connection budget."""
    return RetryPolicy(
        connection=CategoryRetryConfig(max_attempts=3, backoff_min=0.0, backoff_max=0.0),
        timeout=CategoryRetryConfig(max_attempts=1, backoff_min=0.0, backoff_max=0.0),
        rate_limit=CategoryRetryConfig(max_attempts=0),
        server_error=CategoryRetryConfig(max_attempts=0),
    )


def _make_flaky_adapter() -> Any:
    """LangChainLLMAdapter over an injected base model that fails twice."""
    from langchain_core.messages import AIMessage

    from karenina.adapters.langchain.llm import LangChainLLMAdapter

    state = {"calls": 0}

    class FlakyModel:
        async def ainvoke(self, _lc_messages: list[Any]) -> Any:
            state["calls"] += 1
            if state["calls"] <= FAILURES:
                raise ConnectionError("connection refused")
            return AIMessage(content="Paris")

    config = ModelConfig(
        id="telemetry-propagation-test",
        model_name="test-model",
        model_provider="openai",
        interface="langchain",
        retry_policy=_tight_policy(),
    )
    return LangChainLLMAdapter(config, _base_model=FlakyModel())


@pytest.mark.unit
class TestRetryTelemetryThreadFallback:
    """The ThreadPoolExecutor fallback carries the caller's context."""

    def test_sync_invoke_from_running_loop_records_retries(self):
        """Calling sync invoke inside a running loop uses the thread fallback
        and still increments the caller's tracker."""
        adapter = _make_flaky_adapter()
        messages = [Message.user("Capital of France?")]

        with track_retries(_tight_policy()) as tracker:

            async def call_sync_from_coroutine() -> Any:
                # A loop is running in this thread, so invoke() dispatches
                # to a fresh ThreadPoolExecutor thread.
                return adapter.invoke(messages)

            result = asyncio.run(call_sync_from_coroutine())

        assert result.content == "Paris"
        assert tracker["connection"]["used"] == FAILURES

    def test_sync_parser_thread_fallback_records_retries(self, monkeypatch):
        """The parser sync wrapper's thread fallback also carries context."""
        pytest.importorskip("deepagents", reason="deepagents not installed")
        from pydantic import BaseModel, Field

        from karenina.adapters.langchain_deep_agents.parser import DeepAgentsParserAdapter

        class CityAnswer(BaseModel):
            city: str = Field(description="A city name")

        state = {"calls": 0}

        class FlakyModel:
            def with_structured_output(self, _schema: type[BaseModel], **_kw: Any) -> FlakyModel:
                return self

            async def ainvoke(self, _lc_messages: list[Any]) -> Any:
                state["calls"] += 1
                if state["calls"] <= FAILURES:
                    raise ConnectionError("connection refused")
                return {"raw": None, "parsed": CityAnswer(city="Paris"), "parsing_error": None}

        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.parser.create_chat_model",
            lambda _config, **_kw: FlakyModel(),
        )

        config = ModelConfig(
            id="telemetry-parser-test",
            model_name="test-model",
            model_provider="openai",
            interface="langchain_deep_agents",
            retry_policy=_tight_policy(),
        )
        parser = DeepAgentsParserAdapter(config)
        parser_messages = [Message.system("Extract the city."), Message.user("Paris is the capital.")]

        with track_retries(_tight_policy()) as tracker:

            async def call_sync_from_coroutine() -> Any:
                return parser.parse_to_pydantic(parser_messages, CityAnswer)

            result = asyncio.run(call_sync_from_coroutine())

        assert result.parsed.city == "Paris"
        assert tracker["connection"]["used"] == FAILURES


@pytest.mark.unit
class TestRetryTelemetryPortalPath:
    """Regression guard: the BlockingPortal path keeps recording."""

    def test_sync_invoke_via_portal_records_retries(self):
        """anyio's portal.call copies the caller's contextvars (measured on
        the baseline), so telemetry must keep flowing through it."""
        from anyio.from_thread import start_blocking_portal

        from karenina.benchmark.verification.executor import set_async_portal

        adapter = _make_flaky_adapter()
        messages = [Message.user("Capital of France?")]

        with track_retries(_tight_policy()) as tracker, start_blocking_portal() as portal:
            set_async_portal(portal)
            try:
                result = adapter.invoke(messages)
            finally:
                set_async_portal(None)

        assert result.content == "Paris"
        assert tracker["connection"]["used"] == FAILURES


@pytest.mark.unit
class TestRetryTelemetryIsolationAndThreadSafety:
    """Concurrent trackers stay isolated, shared increments stay exact."""

    def test_concurrent_trackers_are_isolated(self):
        """Two threads with their own track_retries bindings never bleed
        counts into each other."""

        results: dict[str, int] = {}
        errors: list[Exception] = []

        def worker(name: str, failures: int) -> None:
            try:
                from langchain_core.messages import AIMessage

                from karenina.adapters.langchain.llm import LangChainLLMAdapter

                state = {"calls": 0}

                class FlakyModel:
                    async def ainvoke(self, _lc_messages: list[Any]) -> Any:
                        state["calls"] += 1
                        if state["calls"] <= failures:
                            raise ConnectionError("connection refused")
                        return AIMessage(content="ok")

                config = ModelConfig(
                    id=f"isolation-{name}",
                    model_name="test-model",
                    model_provider="openai",
                    interface="langchain",
                    retry_policy=_tight_policy(),
                )
                adapter = LangChainLLMAdapter(config, _base_model=FlakyModel())
                with track_retries(_tight_policy()) as tracker:
                    asyncio.run(adapter.ainvoke([Message.user("ping")]))
                results[name] = tracker["connection"]["used"]
            except Exception as exc:  # surfaced via the errors list
                errors.append(exc)

        t1 = threading.Thread(target=worker, args=("one", 1))
        t2 = threading.Thread(target=worker, args=("two", 3))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f"worker errors: {errors}"
        assert results["one"] == 1
        assert results["two"] == 3

    def test_shared_tracker_increments_are_thread_safe(self):
        """Many threads incrementing one shared tracker via copied contexts
        lose no increments (guarded by _RECORD_RETRY_LOCK)."""
        increments_per_thread = 200
        thread_count = 8

        with track_retries(_tight_policy()) as tracker:
            ctx = contextvars.copy_context()

            def hammer() -> None:
                for _ in range(increments_per_thread):
                    ctx.run(_record_retry, ErrorCategory.CONNECTION)

            threads = [threading.Thread(target=hammer) for _ in range(thread_count)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert tracker["connection"]["used"] == increments_per_thread * thread_count
