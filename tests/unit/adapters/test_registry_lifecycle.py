"""Tests for immediate adapter lifecycle cleanup helpers."""

from __future__ import annotations

import asyncio
import logging

import pytest

from karenina.adapters.registry import (
    _active_adapters,
    _adapters_lock,
    cleanup_all_adapters,
    close_adapter,
    register_adapter,
)


@pytest.fixture(autouse=True)
def _clear_adapters():
    """Isolate the module-global adapter list (other tests may leak entries)."""
    with _adapters_lock:
        _active_adapters.clear()
    yield
    with _adapters_lock:
        _active_adapters.clear()


class CountingAdapter:
    def __init__(self) -> None:
        self.close_count = 0

    async def aclose(self) -> None:
        self.close_count += 1


def test_close_adapter_closes_and_unregisters_transient_adapter() -> None:
    adapter = CountingAdapter()
    register_adapter(adapter)

    close_adapter(adapter)

    assert adapter.close_count == 1

    # If close_adapter did not unregister, global cleanup would close it again.
    asyncio.run(cleanup_all_adapters())
    assert adapter.close_count == 1


@pytest.mark.asyncio
async def test_cleanup_all_adapters_still_closes_registered_adapter() -> None:
    adapter = CountingAdapter()
    register_adapter(adapter)

    await cleanup_all_adapters()

    assert adapter.close_count == 1


class FailingAdapter:
    async def aclose(self) -> None:
        raise RuntimeError("close boom")


@pytest.mark.asyncio
async def test_cleanup_all_adapters_logs_failures_at_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Per-adapter close failures are logged at warning with traceback,
    plus one aggregate line with the failure count."""
    register_adapter(FailingAdapter())
    register_adapter(CountingAdapter())

    caplog.set_level(logging.DEBUG, logger="karenina.adapters.registry")
    await cleanup_all_adapters()

    per_adapter = [
        rec
        for rec in caplog.records
        if rec.levelno == logging.WARNING and rec.getMessage() == "Error closing adapter FailingAdapter"
    ]
    assert len(per_adapter) == 1, "Expected one warning for the failing adapter"
    assert per_adapter[0].exc_info is not None, "Failure warning must carry exc_info"

    aggregate = [
        rec
        for rec in caplog.records
        if rec.levelno == logging.WARNING and "1 failure(s) out of 2 adapter(s)" in rec.getMessage()
    ]
    assert len(aggregate) == 1, "Expected one aggregate failure-count warning"

    success = [
        rec for rec in caplog.records if rec.levelno == logging.DEBUG and rec.getMessage() == "Closed 1 adapter(s)"
    ]
    assert len(success) == 1, "Expected the success debug line with the closed count"


@pytest.mark.asyncio
async def test_cleanup_all_adapters_no_failure_logs_when_all_close(caplog: pytest.LogCaptureFixture) -> None:
    """With no failures there is no aggregate warning, only the debug count."""
    register_adapter(CountingAdapter())
    register_adapter(CountingAdapter())

    caplog.set_level(logging.DEBUG, logger="karenina.adapters.registry")
    await cleanup_all_adapters()

    warnings = [rec for rec in caplog.records if rec.levelno == logging.WARNING]
    assert warnings == []
    success = [
        rec for rec in caplog.records if rec.levelno == logging.DEBUG and rec.getMessage() == "Closed 2 adapter(s)"
    ]
    assert len(success) == 1
