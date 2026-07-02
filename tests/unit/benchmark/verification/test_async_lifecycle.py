"""Tests for the async_lifecycle leaf module and its executor re-exports."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from karenina.benchmark.verification import async_lifecycle, executor
from karenina.benchmark.verification.async_lifecycle import (
    get_async_portal,
    set_async_portal,
)


@pytest.fixture(autouse=True)
def _clean_portal():
    """Ensure portal state is cleared before and after each test."""
    set_async_portal(None)
    yield
    set_async_portal(None)


@pytest.mark.unit
class TestExecutorReExports:
    """The executor module re-exports the leaf module's objects by identity.

    This is the leaf-module invariant: async_lifecycle owns the canonical
    state, executor merely re-exports. A single parametrized identity check
    documents the surface; the behavioral test_shared_semaphore_state below
    then proves the re-exports actually share underlying state.
    """

    @pytest.mark.parametrize(
        ("executor_attr", "leaf_attr"),
        [
            ("get_global_llm_semaphore", "get_global_llm_semaphore"),
            ("set_global_llm_semaphore", "set_global_llm_semaphore"),
            ("get_async_portal", "get_async_portal"),
            ("set_async_portal", "set_async_portal"),
            ("_SENTINEL", "_SENTINEL"),
            ("PRE_TEARDOWN_ACLOSE_TIMEOUT", "PRE_TEARDOWN_ACLOSE_TIMEOUT"),
        ],
    )
    def test_reexport_identity(self, executor_attr: str, leaf_attr: str) -> None:
        assert getattr(executor, executor_attr) is getattr(async_lifecycle, leaf_attr)

    def test_shared_semaphore_state(self) -> None:
        """Setting via the executor path is visible via the leaf path."""
        import threading

        sem = threading.Semaphore(2)
        executor.set_global_llm_semaphore(sem)
        try:
            assert async_lifecycle.get_global_llm_semaphore() is sem
        finally:
            async_lifecycle.set_global_llm_semaphore(None)
        assert executor.get_global_llm_semaphore() is None

    def test_leaf_has_no_karenina_module_level_imports(self) -> None:
        """The leaf must stay import-cycle safe (stdlib/anyio only)."""
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(async_lifecycle))
        offenders: list[str] = []
        # Only direct module-level statements: imports inside TYPE_CHECKING
        # blocks or function bodies do not execute at import time.
        for node in tree.body:
            if isinstance(node, ast.Import):
                offenders.extend(alias.name for alias in node.names if alias.name.startswith("karenina"))
            elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("karenina"):
                offenders.append(node.module or "")
        assert not offenders, f"Leaf module imports karenina modules at top level: {offenders}"


@pytest.mark.unit
class TestSentinelStaleness:
    """A portal lacking _event_loop_thread_id is treated as stale."""

    def test_missing_attribute_treated_as_stale(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(async_lifecycle, "_sentinel_stale_warned", False)
        portal = MagicMock(spec=[])  # no attributes at all
        set_async_portal(portal)
        assert get_async_portal() is None
        # Reference was cleared, not just hidden.
        assert get_async_portal() is None

    def test_missing_attribute_warns_once(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(async_lifecycle, "_sentinel_stale_warned", False)
        caplog.set_level(logging.WARNING, logger="karenina.benchmark.verification.async_lifecycle")

        for _ in range(3):
            set_async_portal(MagicMock(spec=[]))
            assert get_async_portal() is None

        warnings = [rec for rec in caplog.records if "without _event_loop_thread_id" in rec.getMessage()]
        assert len(warnings) == 1, f"Expected exactly one staleness warning, got {len(warnings)}"

    def test_none_thread_id_still_treated_as_stale(self) -> None:
        portal = MagicMock()
        portal._event_loop_thread_id = None
        set_async_portal(portal)
        assert get_async_portal() is None

    def test_live_portal_returned(self) -> None:
        portal = MagicMock()
        portal._event_loop_thread_id = 12345
        set_async_portal(portal)
        assert get_async_portal() is portal
