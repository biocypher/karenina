"""Tests for immediate adapter lifecycle cleanup helpers."""

from __future__ import annotations

import asyncio

import pytest

from karenina.adapters.registry import cleanup_all_adapters, close_adapter, register_adapter


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
