"""Async lifecycle primitives shared by the verification executors.

This leaf module owns the cross-cutting async execution state used by both
:class:`~karenina.benchmark.verification.executor.VerificationExecutor` and
:class:`~karenina.benchmark.verification.scenario_executor.ScenarioExecutor`:

- Thread-local BlockingPortal storage (``get_async_portal`` /
  ``set_async_portal``) so worker threads can run async code on a shared
  event loop without passing the portal through every signature.
- The process-wide LLM concurrency cap (``GlobalLLMLimiter`` /
  ``get_global_llm_limiter``): a ref-counted, cross-event-loop limiter on
  concurrent LLM request setups, configured by the executors from
  ``max_concurrent_requests`` and borrowed at the adapter async leaves.
- The legacy module-global LLM request semaphore (``get_global_llm_semaphore``
  / ``set_global_llm_semaphore``), deprecated: production no longer sets it,
  the GlobalLLMLimiter supersedes it.
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

import asyncio
import contextlib
import logging
import threading
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import AsyncIterable, AsyncIterator, Iterator

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
# Global LLM Limiter
# ============================================================================

# Type variable for gate_stream_establishment item pass-through.
T = TypeVar("T")


class GlobalLLMLimiter:
    """Process-wide cap on concurrent LLM request setups.

    Configured by the verification executors from
    ``VerificationConfig.max_concurrent_requests`` and borrowed at the
    adapter async leaves: each single-turn request attempt, each stream
    establishment attempt, and each model call inside langchain-based
    agent loops holds one permit. The cap therefore bounds concurrent
    request SETUPS: an established stream releases its permit before its
    chunks are consumed, so open streams may outnumber the cap.

    Uniform per-attempt policy: every borrow wraps exactly one wire
    attempt (the per-attempt timeout helpers in the adapters, one stream
    establishment attempt, or one agent-loop model call). Retry backoff
    sleeps therefore never hold a permit, and a retrying request
    re-borrows for each attempt.

    Cross-event-loop safety: karenina runs one BlockingPortal event loop
    per worker thread, so the shared core must not be loop-affine
    (``anyio.CapacityLimiter`` and ``asyncio.Semaphore`` bind to a single
    loop). The core is a ``threading.BoundedSemaphore``, and the async
    ``borrow()`` acquires it without blocking the event loop by running
    the blocking acquire in the loop's default thread pool.

    The default acquire behavior is UNBOUNDED (block until a permit
    frees), matching the legacy global-semaphore semantics. This is safe
    in production because every permit is released when its attempt
    completes or hits its own request_timeout, so waiters always make
    progress as long as request paths are bounded. A bounded acquire
    would instead misfire on perfectly healthy backlogged batches (a
    small cap with many workers legitimately waits a long time, and
    threading.Semaphore wakeups are not FIFO). An acquire timeout is
    available as an opt-in via the constructor for callers that prefer a
    ``GlobalLimiterTimeoutError`` over waiting.

    Configuration is ref-counted: the first ``configure(n)`` with a
    non-None ``n`` sets the capacity, nested ``configure`` calls share it
    (a differing ``n`` logs a warning and keeps the first, the capacity is
    never resized mid-flight), and ``configure(None)`` participates in the
    ref-count without setting a capacity. The capacity is dropped only
    when the last active ``configure`` exits. While no ``configure`` is
    active, ``borrow()`` is a near-zero-cost no-op (it takes only the
    uncontended instance lock).
    """

    def __init__(self, acquire_timeout: float | None = None) -> None:
        """Initialize an inactive limiter.

        Args:
            acquire_timeout: Optional wall-clock bound in seconds for a
                single borrow acquisition while the limiter is
                configured. None (the default) blocks until a permit
                frees, matching the legacy global-semaphore semantics.
                When set, an expired acquire raises
                GlobalLimiterTimeoutError.
        """
        self._lock = threading.Lock()
        self._semaphore: threading.BoundedSemaphore | None = None
        self._capacity: int | None = None
        self._refcount = 0
        self._acquire_timeout = acquire_timeout

    @property
    def capacity(self) -> int | None:
        """Return the active capacity, or None while unconfigured."""
        with self._lock:
            return self._capacity

    @contextlib.contextmanager
    def configure(self, capacity: int | None) -> Iterator[GlobalLLMLimiter]:
        """Enable the limiter for the duration of a batch (ref-counted).

        The first active ``configure`` with a non-None capacity creates
        the shared semaphore. Nested ``configure`` calls with the same
        capacity share it; a different capacity logs a warning and keeps
        the first (never resized mid-flight). ``configure(None)`` is a
        no-op enable: it only participates in the ref-count. On exit the
        ref-count is decremented and the capacity is dropped when it
        reaches zero, so an outer batch keeps its cap when a nested batch
        exits.

        Args:
            capacity: Maximum concurrent borrows, or None for no cap.

        Yields:
            This limiter instance.
        """
        with self._lock:
            self._refcount += 1
            if capacity is not None:
                if self._capacity is None:
                    self._capacity = capacity
                    self._semaphore = threading.BoundedSemaphore(capacity)
                elif self._capacity != capacity:
                    logger.warning(
                        "GlobalLLMLimiter already configured with capacity %d. "
                        "Ignoring nested configure(%d) and keeping the first "
                        "(the cap is never resized mid-flight)",
                        self._capacity,
                        capacity,
                    )
        try:
            yield self
        finally:
            with self._lock:
                self._refcount -= 1
                if self._refcount <= 0:
                    self._refcount = 0
                    self._semaphore = None
                    self._capacity = None

    @contextlib.asynccontextmanager
    async def borrow(self) -> AsyncIterator[None]:
        """Hold one permit for the enclosed block.

        Near-zero-cost no-op while the limiter is unconfigured (only the
        uncontended instance lock is taken). The permit is acquired
        without blocking the calling event loop and is released on exit,
        including on exception or cancellation.

        Snapshot semantics: the semaphore reference is read once on
        entry. A configuration teardown racing with an in-flight borrow
        releases into the snapshotted semaphore (harmless, the dropped
        semaphore is unreferenced), and a borrow already past its
        unconfigured snapshot when a ``configure`` activates stays
        ungated for that one call.

        Thread-pool note: a contended acquire occupies one thread of the
        loop's default executor for the duration of the wait, and a
        CANCELLED contended borrow can leave that thread blocked until a
        permit frees (the thread then returns the permit via the
        cancellation handshake). Waiters beyond the pool size simply
        queue for an executor slot. This cannot deadlock the limiter:
        permit holders never need that pool to finish their work.

        Raises:
            GlobalLimiterTimeoutError: Only when an opt-in acquire
                timeout was set on the constructor and no permit became
                available within it.
        """
        with self._lock:
            semaphore = self._semaphore
            acquire_timeout = self._acquire_timeout
        if semaphore is None:
            yield
            return
        await self._acquire_permit(semaphore, acquire_timeout)
        try:
            yield
        finally:
            semaphore.release()

    @staticmethod
    async def _acquire_permit(semaphore: threading.BoundedSemaphore, timeout: float | None) -> None:
        """Acquire one permit without blocking the running event loop.

        Fast path: an uncontended permit is taken with a non-blocking
        acquire directly on the loop thread (no executor round-trip and
        no await, so there is no cancellation window between acquisition
        and the caller's try/finally).

        Contended path: the blocking acquire runs in the loop's default
        thread pool. If the awaiting task is cancelled after the
        underlying acquire already succeeded, the permit is released by
        whichever side observes the completed handshake second (the
        worker thread and the cancellation handler serialize on a state
        lock), so cancellation never leaks permits.

        Args:
            semaphore: The shared semaphore to acquire.
            timeout: Optional wall-clock bound for the blocking acquire
                in seconds. None blocks until a permit frees.

        Raises:
            GlobalLimiterTimeoutError: If an opt-in timeout was set and
                the acquire timed out.
        """
        if semaphore.acquire(blocking=False):
            return

        loop = asyncio.get_running_loop()
        state_lock = threading.Lock()
        state = {"cancelled": False, "held": False}

        def _blocking_acquire() -> bool:
            ok = semaphore.acquire(timeout=timeout)
            with state_lock:
                if ok and state["cancelled"]:
                    semaphore.release()
                    return False
                state["held"] = ok
            return ok

        try:
            acquired = await loop.run_in_executor(None, _blocking_acquire)
        except asyncio.CancelledError:
            with state_lock:
                state["cancelled"] = True
                if state["held"]:
                    semaphore.release()
                    state["held"] = False
            raise
        if not acquired:
            # Lazy import: karenina.exceptions pulls in non-leaf modules at
            # import time, which would break this module's import-cycle
            # discipline (stdlib-only at module level).
            from karenina.exceptions import GlobalLimiterTimeoutError

            raise GlobalLimiterTimeoutError(
                f"GlobalLLMLimiter could not acquire a permit within {timeout}s. "
                "The global LLM concurrency cap stayed saturated, so a wedged "
                "endpoint may be holding permits."
            )


_GLOBAL_LLM_LIMITER = GlobalLLMLimiter()


def get_global_llm_limiter() -> GlobalLLMLimiter:
    """Return the process-wide GlobalLLMLimiter singleton.

    Module-level (not thread-local) because the cap must be shared across
    every worker thread and every per-thread portal event loop.

    Returns:
        The singleton GlobalLLMLimiter.
    """
    return _GLOBAL_LLM_LIMITER


async def gate_stream_establishment(source: AsyncIterable[T]) -> AsyncIterator[T]:
    """Yield from ``source``, holding a limiter permit for the first item only.

    Streaming adapters whose underlying stream starts lazily on first
    iteration (the wire request fires inside the first ``__anext__``)
    wrap their raw stream with this helper so establishment counts
    against the global cap. The permit is released before the first item
    is yielded: the cap bounds concurrent stream SETUPS, not concurrent
    open streams.

    Args:
        source: The raw async iterable whose first item fetch performs
            the request establishment.

    Yields:
        The items of ``source``, unchanged.
    """
    iterator = source.__aiter__()
    async with get_global_llm_limiter().borrow():
        try:
            first = await iterator.__anext__()
        except StopAsyncIteration:
            return
    yield first
    async for item in iterator:
        yield item


# ============================================================================
# Global LLM Semaphore (legacy)
# ============================================================================

_global_llm_semaphore: threading.Semaphore | None = None


def get_global_llm_semaphore() -> threading.Semaphore | None:
    """Get the global LLM request semaphore.

    Deprecated: production code no longer sets this semaphore. The
    GlobalLLMLimiter (``get_global_llm_limiter``) supersedes it as the
    enforcement mechanism for ``max_concurrent_requests``. The accessor
    pair and the ``with_llm_semaphore`` decorator remain functional for
    direct callers and existing tests.

    Module-level (not thread-local) because the semaphore must be visible
    from any thread, including the BlockingPortal event loop thread.
    The semaphore itself is thread-safe.

    Returns:
        The active Semaphore if set, None otherwise.
    """
    return _global_llm_semaphore


def set_global_llm_semaphore(sem: threading.Semaphore | None) -> None:
    """Set the global LLM request semaphore.

    Deprecated: production code no longer calls this (the executors enter
    ``GlobalLLMLimiter.configure`` instead). Kept functional for direct
    callers and existing tests.

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
