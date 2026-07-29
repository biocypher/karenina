"""Tests for deterministic MCP client teardown in the langchain adapter.

cleanup_mcp_client must always await async teardown to completion (portal,
run_coroutine_threadsafe, or asyncio.run), never schedule-and-drop. When
called on a running loop's own thread it must warn and direct callers to
acleanup_mcp_client instead of deadlocking or dropping a coroutine.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from typing import Any

import pytest

from karenina.adapters.langchain.mcp import acleanup_mcp_client, cleanup_mcp_client

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")


class _FakePortal:
    """Minimal portal exposing start_task_soon, runs the close to completion."""

    def __init__(self) -> None:
        self.started_with: Any = None

    def start_task_soon(self, func: Any) -> concurrent.futures.Future:
        self.started_with = func
        future: concurrent.futures.Future = concurrent.futures.Future()
        try:
            future.set_result(asyncio.run(func()))
        except Exception as e:  # noqa: BLE001 - test stub mirrors Future semantics
            future.set_exception(e)
        return future


class _AsyncCloseClient:
    """Client exposing only aclose, records whether the close was awaited."""

    def __init__(self, fail: bool = False) -> None:
        self.aclose_awaited = False
        self.fail = fail

    async def aclose(self) -> None:
        await asyncio.sleep(0)
        self.aclose_awaited = True
        if self.fail:
            raise RuntimeError("boom during aclose")


class _SyncCloseClient:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _ExitOnlyClient:
    def __init__(self) -> None:
        self.exited = False

    def __exit__(self, *args: Any) -> None:
        self.exited = True


@pytest.mark.unit
class TestAcleanupMcpClient:
    async def test_aclose_is_awaited(self) -> None:
        client = _AsyncCloseClient()
        await acleanup_mcp_client(client)
        assert client.aclose_awaited

    async def test_none_client_is_noop(self) -> None:
        await acleanup_mcp_client(None)

    async def test_aclose_failure_is_logged_not_raised(self, caplog: pytest.LogCaptureFixture) -> None:
        client = _AsyncCloseClient(fail=True)
        with caplog.at_level(logging.WARNING):
            await acleanup_mcp_client(client)
        assert client.aclose_awaited
        assert any("aclose() failed" in record.message for record in caplog.records)

    async def test_sync_close_fallback(self) -> None:
        client = _SyncCloseClient()
        await acleanup_mcp_client(client)
        assert client.closed

    async def test_context_manager_fallback(self) -> None:
        client = _ExitOnlyClient()
        await acleanup_mcp_client(client)
        assert client.exited


@pytest.mark.unit
class TestCleanupMcpClientSync:
    def test_sync_close_preferred(self) -> None:
        client = _SyncCloseClient()
        cleanup_mcp_client(client)
        assert client.closed

    def test_none_client_is_noop(self) -> None:
        cleanup_mcp_client(None)

    def test_aclose_awaited_via_asyncio_run(self) -> None:
        """No portal and no running loop: aclose runs to completion in a fresh loop."""
        client = _AsyncCloseClient()
        cleanup_mcp_client(client)
        assert client.aclose_awaited

    def test_aclose_awaited_via_portal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A shared portal awaits the close (bounded) before cleanup_mcp_client returns."""
        from karenina.benchmark.verification import executor

        portal = _FakePortal()
        monkeypatch.setattr(executor, "get_async_portal", lambda: portal)

        client = _AsyncCloseClient()
        cleanup_mcp_client(client)
        assert portal.started_with is not None
        assert client.aclose_awaited

    def test_portal_dispatch_is_timeout_bounded(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A stalled close on the portal cannot wedge the calling thread."""
        import karenina.adapters.langchain.mcp as mcp_module
        from karenina.benchmark.verification import executor

        stuck_future: concurrent.futures.Future = concurrent.futures.Future()

        class _StuckPortal:
            def start_task_soon(self, _func: Any) -> concurrent.futures.Future:
                return stuck_future

        monkeypatch.setattr(executor, "get_async_portal", lambda: _StuckPortal())
        monkeypatch.setattr(mcp_module, "MCP_CLEANUP_TIMEOUT", 0.05)

        client = _AsyncCloseClient()
        with caplog.at_level(logging.WARNING):
            cleanup_mcp_client(client)

        assert stuck_future.cancelled()
        assert any("Failed to cleanup MCP client" in record.message for record in caplog.records)

    def test_aclose_awaited_via_run_coroutine_threadsafe(self) -> None:
        """An explicit loop running in another thread receives the close."""
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        try:
            client = _AsyncCloseClient()
            cleanup_mcp_client(client, loop=loop)
            assert client.aclose_awaited
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=5)
            loop.close()

    def test_explicit_loop_preferred_over_portal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The client's home loop wins over the shared portal when both exist."""
        from karenina.benchmark.verification import executor

        portal = _FakePortal()
        monkeypatch.setattr(executor, "get_async_portal", lambda: portal)

        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        try:
            client = _AsyncCloseClient()
            cleanup_mcp_client(client, loop=loop)
            assert client.aclose_awaited
            assert portal.started_with is None
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=5)
            loop.close()

    async def test_on_running_loop_warns_even_with_portal(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The own-running-loop guard fires before any portal dispatch."""
        from karenina.benchmark.verification import executor

        portal = _FakePortal()
        monkeypatch.setattr(executor, "get_async_portal", lambda: portal)

        client = _AsyncCloseClient()
        with caplog.at_level(logging.WARNING):
            cleanup_mcp_client(client)

        assert not client.aclose_awaited
        assert portal.started_with is None
        assert any("acleanup_mcp_client" in record.message for record in caplog.records)

    async def test_on_running_loop_warns_and_does_not_block(self, caplog: pytest.LogCaptureFixture) -> None:
        """Calling the sync wrapper on a running loop warns instead of deadlocking."""
        client = _AsyncCloseClient()
        with caplog.at_level(logging.WARNING):
            cleanup_mcp_client(client)
        assert not client.aclose_awaited
        assert any("acleanup_mcp_client" in record.message for record in caplog.records)

    def test_aclose_failure_is_logged_not_raised(self, caplog: pytest.LogCaptureFixture) -> None:
        client = _AsyncCloseClient(fail=True)
        with caplog.at_level(logging.WARNING):
            cleanup_mcp_client(client)
        assert any("Failed to cleanup MCP client" in record.message for record in caplog.records)

    def test_context_manager_fallback(self) -> None:
        client = _ExitOnlyClient()
        cleanup_mcp_client(client)
        assert client.exited
