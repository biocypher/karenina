"""Async lifecycle primitives shared by the verification executors.

This leaf module owns the cross-cutting async execution state used by both
:class:`~karenina.benchmark.verification.executor.VerificationExecutor` and
:class:`~karenina.benchmark.verification.scenario_executor.ScenarioExecutor`:

- Thread-local BlockingPortal storage (``get_async_portal`` /
  ``set_async_portal``) so worker threads can run async code on a shared
  event loop without passing the portal through every signature.
- The module-global LLM request semaphore (``get_global_llm_semaphore`` /
  ``set_global_llm_semaphore``) used to cap concurrent LLM requests.
- The pre-teardown adapter aclose bound (``PRE_TEARDOWN_ACLOSE_TIMEOUT``).

Import discipline: this module imports only the standard library (anyio is
referenced for type checking only). It must never import from
``karenina.adapters`` or any other ``karenina.benchmark`` module at the top
level, so adapters can import it at module import time without cycles.

Back-compat: ``karenina.benchmark.verification.executor`` re-exports every
public name here, so existing imports and monkeypatch targets through the
executor module path remain valid.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anyio.from_thread import BlockingPortal

logger = logging.getLogger(__name__)

_SENTINEL = object()  # Distinguishes "attribute missing" from "attribute is None"

# Bound (seconds) on each adapter.aclose() call issued via the worker portal
# before the portal is torn down. A stuck aclose must not wedge the finally
# block. Mirrors the pattern in langchain/parser.py:297-312. To exercise the
# timeout branch in tests, monkey-patch executor.PRE_TEARDOWN_ACLOSE_TIMEOUT
# or scenario_executor.PRE_TEARDOWN_ACLOSE_TIMEOUT (the executors pass their
# own module bindings as timeout=, so patching this leaf constant is a no-op).
PRE_TEARDOWN_ACLOSE_TIMEOUT = 5.0

# ============================================================================
# Thread-Local Portal Storage
# ============================================================================

_portal_storage = threading.local()

# One-time flag for the missing-attribute staleness warning below. If anyio
# ever renames ``_event_loop_thread_id``, every portal would look attribute-
# less, so warn loudly once instead of spamming every call.
_sentinel_stale_warned = False


def get_async_portal() -> BlockingPortal | None:
    """Get the current async portal for running async code from threads.

    Each worker thread has its own thread-local portal reference.
    Returns None (and clears the stale reference) if the portal's event loop
    has ended, allowing callers to fall back to asyncio.run().

    The health check reads anyio's private ``_event_loop_thread_id``
    attribute: a live portal carries the loop thread id, a stopped portal
    carries None. A portal that lacks the attribute entirely (anyio renamed
    it, or a non-anyio object was stored) is also treated as stale, because
    the health check can no longer prove the loop is alive. That case emits
    a one-time warning since it indicates an anyio API drift.

    Returns:
        The BlockingPortal if one is active for this thread, None otherwise
    """
    global _sentinel_stale_warned  # noqa: PLW0603

    portal = getattr(_portal_storage, "portal", None)
    if portal is not None:
        thread_id = getattr(portal, "_event_loop_thread_id", _SENTINEL)
        if thread_id is None:
            logger.warning("Clearing stale portal reference (event loop thread ended)")
            _portal_storage.portal = None
            return None
        if thread_id is _SENTINEL:
            if not _sentinel_stale_warned:
                _sentinel_stale_warned = True
                logger.warning(
                    "Clearing portal reference without _event_loop_thread_id attribute "
                    "(anyio may have renamed it), treating the portal as stale. "
                    "This warning is emitted once per process."
                )
            _portal_storage.portal = None
            return None
    return portal


def set_async_portal(portal: BlockingPortal | None) -> None:
    """Set the async portal for the current thread.

    Args:
        portal: The BlockingPortal to use, or None to clear
    """
    _portal_storage.portal = portal


# ============================================================================
# Global LLM Semaphore
# ============================================================================

_global_llm_semaphore: threading.Semaphore | None = None


def get_global_llm_semaphore() -> threading.Semaphore | None:
    """Get the global LLM request semaphore.

    Module-level (not thread-local) because the semaphore must be visible
    from any thread, including the BlockingPortal event loop thread.
    The semaphore itself is thread-safe.

    Returns:
        The active Semaphore if set, None otherwise.
    """
    return _global_llm_semaphore


def set_global_llm_semaphore(sem: threading.Semaphore | None) -> None:
    """Set the global LLM request semaphore.

    Called by ScenarioExecutor before spawning workers and cleared after
    all workers finish.

    Args:
        sem: The Semaphore to use, or None to clear.
    """
    global _global_llm_semaphore  # noqa: PLW0603
    _global_llm_semaphore = sem


# ============================================================================
# Pre-Teardown Adapter Close
# ============================================================================


def aclose_portal_adapters(portal: BlockingPortal, timeout: float = PRE_TEARDOWN_ACLOSE_TIMEOUT) -> None:
    """Close portal-pinned adapter clients on the portal's own event loop.

    Adapters created inside a worker thread with an active BlockingPortal
    open httpx transports bound to that portal's event loop. They must be
    closed on the same loop BEFORE the portal is torn down: the downstream
    cleanup_resources() call runs on a fresh loop and httpx raises
    "Event loop is closed" when its transports are pinned to a dead loop.

    Each aclose() is dispatched with the bounded ``portal.start_task_soon``
    + ``future.result(timeout)`` + cancel pattern (mirrors
    langchain/parser.py) so a stuck aclose cannot wedge the caller's
    finally block. Per-adapter failures are logged rather than raised and
    the per-portal adapter tracking is always cleared.

    Args:
        portal: The BlockingPortal whose tracked adapters should be closed.
        timeout: Per-adapter wall-clock bound in seconds for each aclose()
            dispatch. Defaults to ``PRE_TEARDOWN_ACLOSE_TIMEOUT``.
    """
    # Lazy import: the registry lives in karenina.adapters, which this leaf
    # must not import at module level (import-cycle discipline).
    from karenina.adapters.registry import (
        clear_portal_adapter_refs,
        snapshot_adapters_for_portal,
    )

    try:
        for adapter in snapshot_adapters_for_portal(portal):
            if not hasattr(adapter, "aclose"):
                continue
            try:
                future = portal.start_task_soon(adapter.aclose)
                future.result(timeout=timeout)
            except TimeoutError:
                # Cancel the abandoned coroutine so the portal's loop does
                # not block the context manager's __exit__ waiting for it
                # to finish. TimeoutError only comes from future.result, so
                # the future is always bound here.
                future.cancel()
                logger.warning(
                    "Pre-teardown aclose timed out on %s (>%ss), proceeding with portal teardown",
                    type(adapter).__name__,
                    timeout,
                )
            except Exception as exc:
                logger.warning(
                    "Pre-teardown aclose failed on %s: %s",
                    type(adapter).__name__,
                    exc,
                )
    finally:
        clear_portal_adapter_refs(portal)
