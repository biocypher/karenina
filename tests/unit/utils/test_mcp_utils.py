"""Tests for karenina.utils.mcp after the dead-duplicate removal (T8).

connect_mcp_session survives (it is live via the langchain adapter's
persistent MCP tools). The unused multi-server duplicates
connect_all_mcp_servers and get_all_mcp_tools were deleted, the live
copies in karenina.adapters.claude_tool.mcp keep their own tests.
fetch_tool_descriptions now bounds its portal dispatch.
"""

from __future__ import annotations

import concurrent.futures
from typing import Any

import pytest

from karenina.exceptions import McpTimeoutError


@pytest.mark.unit
class TestMcpUtilsSurface:
    """The public surface of karenina.utils.mcp after T8."""

    def test_surviving_symbols_import(self) -> None:
        from karenina.utils.mcp import (
            afetch_tool_descriptions,
            apply_tool_description_overrides,
            connect_mcp_session,
            fetch_tool_descriptions,
        )

        assert callable(connect_mcp_session)
        assert callable(afetch_tool_descriptions)
        assert callable(fetch_tool_descriptions)
        assert callable(apply_tool_description_overrides)

    def test_deleted_duplicates_are_gone(self) -> None:
        import karenina.utils.mcp as mcp_pkg
        import karenina.utils.mcp.client as mcp_client

        assert not hasattr(mcp_pkg, "connect_all_mcp_servers")
        assert not hasattr(mcp_pkg, "get_all_mcp_tools")
        assert not hasattr(mcp_client, "connect_all_mcp_servers")
        assert not hasattr(mcp_client, "get_all_mcp_tools")
        assert "connect_all_mcp_servers" not in mcp_pkg.__all__
        assert "get_all_mcp_tools" not in mcp_pkg.__all__

    def test_live_copies_still_exist_in_claude_tool(self) -> None:
        from karenina.adapters.claude_tool.mcp import connect_all_mcp_servers, get_all_mcp_tools

        assert callable(connect_all_mcp_servers)
        assert callable(get_all_mcp_tools)


@pytest.mark.unit
class TestFetchToolDescriptionsPortalBound:
    """The portal dispatch in fetch_tool_descriptions is timeout bounded."""

    def test_portal_path_returns_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from karenina.benchmark.verification import executor
        from karenina.utils.mcp.tools import fetch_tool_descriptions

        expected = {"tool_a": "does a thing"}

        class _FakePortal:
            def start_task_soon(self, _fn: Any, *_args: Any) -> concurrent.futures.Future:
                future: concurrent.futures.Future = concurrent.futures.Future()
                future.set_result(expected)
                return future

        monkeypatch.setattr(executor, "get_async_portal", lambda: _FakePortal())

        result = fetch_tool_descriptions({"server": "http://example.invalid/mcp"})
        assert result == expected

    def test_portal_path_times_out_and_cancels(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import karenina.adapters._timeouts as timeouts_mod
        from karenina.benchmark.verification import executor
        from karenina.utils.mcp.tools import fetch_tool_descriptions

        stuck_future: concurrent.futures.Future = concurrent.futures.Future()

        class _FakePortal:
            def start_task_soon(self, _fn: Any, *_args: Any) -> concurrent.futures.Future:
                return stuck_future

        monkeypatch.setattr(executor, "get_async_portal", lambda: _FakePortal())
        monkeypatch.setattr(timeouts_mod, "compute_sync_wrapper_timeout", lambda *_a, **_kw: 0.05)

        with pytest.raises(McpTimeoutError):
            fetch_tool_descriptions({"server": "http://example.invalid/mcp"})

        assert stuck_future.cancelled()
