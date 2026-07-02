"""Tests for dead portal detection in get_async_portal()."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from karenina.benchmark.verification.executor import get_async_portal, set_async_portal


@pytest.fixture(autouse=True)
def _clean_portal():
    """Ensure portal is cleared before and after each test."""
    set_async_portal(None)
    yield
    set_async_portal(None)


def _make_live_portal() -> MagicMock:
    """Create a mock portal that appears alive."""
    portal = MagicMock()
    portal._event_loop_thread_id = 12345  # Non-None = alive
    return portal


def _make_dead_portal() -> MagicMock:
    """Create a mock portal that appears dead (event loop thread ended)."""
    portal = MagicMock()
    portal._event_loop_thread_id = None  # None = dead
    return portal


@pytest.mark.unit
class TestGetAsyncPortalHealthCheck:
    def test_returns_live_portal(self) -> None:
        portal = _make_live_portal()
        set_async_portal(portal)
        assert get_async_portal() is portal

    def test_returns_none_for_dead_portal(self) -> None:
        portal = _make_dead_portal()
        set_async_portal(portal)
        assert get_async_portal() is None

    def test_clears_stale_reference_on_dead_portal(self) -> None:
        """After detecting a dead portal, the stale reference is cleared."""
        portal = _make_dead_portal()
        set_async_portal(portal)

        # First call detects and clears
        assert get_async_portal() is None
        # Second call returns None without needing to detect again
        assert get_async_portal() is None

    def test_returns_none_when_no_portal_set(self) -> None:
        assert get_async_portal() is None

    def test_portal_without_thread_id_attr_treated_as_stale(self) -> None:
        """Portal missing _event_loop_thread_id is treated as stale.

        If anyio renames the private attribute, the health check can no
        longer prove the loop is alive, so the portal is cleared instead of
        being trusted blindly (T10 sentinel fix).
        """
        portal = MagicMock(spec=[])  # No attributes at all
        set_async_portal(portal)
        assert get_async_portal() is None
        # The stale reference is cleared, subsequent calls stay None.
        assert get_async_portal() is None
