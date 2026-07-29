"""Tests for loop affinity in search_provider async tool dispatch.

The async branch of _invoke_tool must prefer the shared BlockingPortal when
one is active for the calling thread (so async tools run on the same loop as
the adapters), guard against nesting when a loop is already running in the
calling thread, and keep the new-loop thread fallback intact when no portal
is active.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from anyio.from_thread import start_blocking_portal

from karenina.benchmark.verification.async_lifecycle import set_async_portal
from karenina.benchmark.verification.utils.search_provider import _wrap_langchain_tool


class AsyncRecordingTool:
    """Async search tool that records the event loop it ran on."""

    def __init__(self) -> None:
        self.loops: list[asyncio.AbstractEventLoop] = []

    async def ainvoke(self, query: str) -> list[dict]:
        self.loops.append(asyncio.get_running_loop())
        return [{"title": "t", "content": f"result for {query}", "url": "http://example.com"}]


@pytest.fixture(autouse=True)
def _clean_portal():
    """Keep thread-local portal state clean around each test."""
    set_async_portal(None)
    yield
    set_async_portal(None)


@pytest.mark.unit
class TestSearchProviderPortalAffinity:
    def test_portal_used_when_active(self) -> None:
        """With an active portal and no running loop, the tool runs on the portal loop."""
        tool = AsyncRecordingTool()
        search = _wrap_langchain_tool(tool)

        with start_blocking_portal(backend="asyncio") as portal:
            set_async_portal(portal)
            portal_loop = portal.call(asyncio.get_running_loop)
            results = search("q1")

        assert tool.loops == [portal_loop], "Async tool must run on the shared portal's event loop"
        assert len(results) == 1
        assert results[0].content == "result for q1"

    def test_on_loop_guard_falls_back_to_thread(self) -> None:
        """With a running loop in the calling thread, the portal is NOT used."""
        tool = AsyncRecordingTool()
        search = _wrap_langchain_tool(tool)

        stub_portal = MagicMock()
        stub_portal._event_loop_thread_id = 12345

        results_holder: list = []

        async def main() -> None:
            set_async_portal(stub_portal)
            # Direct sync call while this thread runs a loop: the guard must
            # route to the thread fallback, never to portal dispatch.
            results_holder.append(search("q2"))

        asyncio.run(main())

        stub_portal.start_task_soon.assert_not_called()
        stub_portal.call.assert_not_called()
        results = results_holder[0]
        assert len(results) == 1
        assert results[0].content == "result for q2"
        # The fallback ran the tool on a fresh loop in a worker thread.
        assert len(tool.loops) == 1

    def test_no_portal_fallback_unchanged(self) -> None:
        """Without a portal, the historical thread + new loop fallback is used."""
        tool = AsyncRecordingTool()
        search = _wrap_langchain_tool(tool)

        results = search("q3")

        assert len(results) == 1
        assert results[0].content == "result for q3"
        assert len(tool.loops) == 1

    def test_batch_queries_through_portal(self) -> None:
        """Batch dispatch reuses the portal loop for every query."""
        tool = AsyncRecordingTool()
        search = _wrap_langchain_tool(tool)

        with start_blocking_portal(backend="asyncio") as portal:
            set_async_portal(portal)
            portal_loop = portal.call(asyncio.get_running_loop)
            results = search(["a", "b"])

        assert tool.loops == [portal_loop, portal_loop]
        assert len(results) == 2
        assert results[0][0].content == "result for a"
        assert results[1][0].content == "result for b"
