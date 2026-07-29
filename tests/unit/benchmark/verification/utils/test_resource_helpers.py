"""Tests for resource cleanup helper functions."""

import asyncio
import concurrent.futures
from unittest.mock import AsyncMock, MagicMock

import pytest
from anyio.from_thread import start_blocking_portal

from karenina.adapters.registry import (
    _active_adapters,
    _adapters_lock,
    register_adapter,
)
from karenina.benchmark.verification.async_lifecycle import set_async_portal
from karenina.benchmark.verification.utils.resource_helpers import cleanup_resources


class FakeAdapter:
    """Adapter stub whose aclose() is an AsyncMock we can assert on."""

    def __init__(self) -> None:
        self.aclose = AsyncMock()


@pytest.fixture(autouse=True)
def _clear_adapters():
    """Ensure adapter list is empty before and after each test."""
    with _adapters_lock:
        _active_adapters.clear()
    yield
    with _adapters_lock:
        _active_adapters.clear()


@pytest.fixture(autouse=True)
def _clean_portal():
    """Keep thread-local portal state clean around each test."""
    set_async_portal(None)
    yield
    set_async_portal(None)


def test_cleanup_awaits_adapter_close_no_loop():
    """cleanup_resources() awaits aclose() on every registered adapter
    when no event loop is running (asyncio.run path)."""
    adapters = [FakeAdapter(), FakeAdapter(), FakeAdapter()]
    for a in adapters:
        register_adapter(a)

    cleanup_resources()

    for a in adapters:
        a.aclose.assert_awaited_once()


def test_cleanup_awaits_adapter_close_with_running_loop():
    """cleanup_resources() awaits aclose() even when called from inside
    an async context (run_coroutine_threadsafe path)."""
    adapters = [FakeAdapter(), FakeAdapter()]
    for a in adapters:
        register_adapter(a)

    async def _run_in_loop():
        # Run cleanup from a different thread so the current loop stays alive
        # while run_coroutine_threadsafe schedules the coroutine onto it.
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, cleanup_resources)

    asyncio.run(_run_in_loop())

    for a in adapters:
        a.aclose.assert_awaited_once()


def test_cleanup_handles_adapter_close_error():
    """cleanup_resources() does not raise even if an adapter's aclose() fails."""
    bad = FakeAdapter()
    bad.aclose = AsyncMock(side_effect=RuntimeError("boom"))
    register_adapter(bad)

    # Should not raise
    cleanup_resources()


def test_cleanup_clears_adapter_list():
    """After cleanup, the adapter list is empty."""
    adapters = [FakeAdapter(), FakeAdapter()]
    for a in adapters:
        register_adapter(a)

    cleanup_resources()

    with _adapters_lock:
        assert len(_active_adapters) == 0


def test_cleanup_uses_portal_loop_when_active(monkeypatch):
    """With an active portal and no running loop in the calling thread,
    cleanup runs on the portal's own event loop (loop affinity)."""
    ran_on: list[asyncio.AbstractEventLoop] = []

    async def fake_cleanup():
        ran_on.append(asyncio.get_running_loop())

    monkeypatch.setattr("karenina.adapters.registry.cleanup_all_adapters", fake_cleanup)

    with start_blocking_portal(backend="asyncio") as portal:
        set_async_portal(portal)
        portal_loop = portal.call(asyncio.get_running_loop)
        cleanup_resources()

    assert ran_on == [portal_loop], "Adapter cleanup must run on the shared portal's loop"


def test_cleanup_running_loop_guard_skips_portal(monkeypatch):
    """With a loop running in the calling thread, the portal is NOT used:
    the existing run_coroutine_threadsafe path is kept to avoid nesting."""
    calls: dict = {}

    def fake_run_coroutine_threadsafe(coro, loop):
        calls["loop"] = loop
        coro.close()
        future: concurrent.futures.Future = concurrent.futures.Future()
        future.set_result(None)
        return future

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", fake_run_coroutine_threadsafe)

    stub_portal = MagicMock()
    stub_portal._event_loop_thread_id = 12345

    async def main():
        set_async_portal(stub_portal)
        cleanup_resources()

    asyncio.run(main())

    assert "loop" in calls, "Expected the run_coroutine_threadsafe path with a running loop"
    stub_portal.start_task_soon.assert_not_called()
    stub_portal.call.assert_not_called()


def test_cleanup_resources_noop_after_pre_teardown_aclose():
    """cleanup_resources() is safe to call after an adapter has already been
    aclose()'d by the executor's pre-teardown path.

    The parallel executor now closes httpx-owning adapters on the worker
    portal's loop BEFORE tearing down the portal. The downstream
    cleanup_resources() still iterates registered adapters and awaits their
    aclose() again. This second call must be safe (httpx documents aclose
    as idempotent; adapters that wrap it must not raise).
    """
    already_closed = FakeAdapter()
    register_adapter(already_closed)

    # Simulate the pre-teardown aclose having already run.
    asyncio.run(already_closed.aclose())
    already_closed.aclose.reset_mock()

    # cleanup_resources() must not raise, and must still call aclose()
    # exactly once. The adapter's aclose is expected to be idempotent.
    cleanup_resources()

    already_closed.aclose.assert_awaited_once()
