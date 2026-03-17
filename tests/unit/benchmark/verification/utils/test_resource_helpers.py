"""Tests for resource cleanup helper functions."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from karenina.adapters.registry import (
    _active_adapters,
    _adapters_lock,
    register_adapter,
)
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
