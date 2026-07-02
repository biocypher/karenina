"""Shared portal-backed thread pool for the verification executors (T14).

This module owns the parallel execution skeleton that was previously
duplicated between :class:`~karenina.benchmark.verification.executor.VerificationExecutor`
and :class:`~karenina.benchmark.verification.scenario_executor.ScenarioExecutor`:

- Lazy per-worker BlockingPortal lifecycle (one portal per worker thread,
  reused across items, tracked for teardown).
- ThreadPoolExecutor submit + ``as_completed`` drain with the optional
  wall-clock batch timeout.
- ``pool.shutdown(wait=True, cancel_futures=timed_out)`` followed by the
  post-shutdown recovered-from-drain sweep, so in-flight items are never
  silently dropped when the batch times out (issue 189).
- The drop-detection invariant: every submitted index must end in
  ``results_by_index`` or ``failed_items``. Futures that escape both sweeps
  get a synthetic TimeoutError entry.
- The bounded pre-teardown adapter aclose on each worker portal BEFORE the
  portal context manager exits.
- Progress dispatch (item-start previews and completion callbacks) under
  ONE lock. Sinks rely on this serialization (until T17 gives sinks their
  own locking).
- A ``gate`` hook entered around the pooled run, intended for
  ``GlobalLLMLimiter.configure`` (ref-counted save-and-restore, so nesting
  cannot lift an outer cap).

The executors stay thin translators: the QA worker keeps its cache requeue
loop, per-answerer caps, hydration, and ``VerificationBatchError`` contract.
The scenario worker keeps ``ScenarioManager.run`` dispatch, per-turn progress
wiring, and its ``(results, errors)`` return contract.

Monkeypatch compatibility: callers inject ``portal_factory`` and
``pre_teardown`` as closures defined in their own modules, so existing test
patch targets (``executor.start_blocking_portal``,
``executor.aclose_portal_adapters``, ``executor.PRE_TEARDOWN_ACLOSE_TIMEOUT``
and the scenario_executor equivalents) keep resolving at call time. The
``get_async_portal`` / ``set_async_portal`` accessors are bound from
``async_lifecycle`` at import time and are intentionally not injectable: the
injectable seams are ``portal_factory``, ``pre_teardown``, and the timeout.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

from anyio.from_thread import start_blocking_portal

from karenina.schemas.verification.config import DEFAULT_ASYNC_MAX_WORKERS

from .async_lifecycle import (
    PRE_TEARDOWN_ACLOSE_TIMEOUT,
    aclose_portal_adapters,
    get_async_portal,
    set_async_portal,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from contextlib import AbstractContextManager

    from anyio.from_thread import BlockingPortal

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


# ============================================================================
# Shared tuning
# ============================================================================


@dataclass
class ExecutionTuning:
    """Execution knobs shared by both verification executors.

    ``ExecutorConfig`` and ``ScenarioExecutorConfig`` remain the public
    configuration surfaces. Each derives an ``ExecutionTuning`` via its
    ``to_tuning()`` method. Fields that one path does not use are simply
    left at their defaults by that path.

    The pool itself consumes only ``max_workers`` and ``timeout_seconds``.
    The other four fields are carried for future consolidation but are NOT
    read by ``run_in_portal_pool``: the executors read them from their own
    configs (the QA worker closure reads ``retry_wait_seconds``,
    ``max_requeue_count``, and ``answerer_concurrency_limits``, and the
    callers enter the limiter with ``max_concurrent_requests``). Setting
    them on a directly constructed ``ExecutionTuning`` has no effect on the
    pool.

    Attributes:
        max_workers: Maximum number of parallel worker threads.
        timeout_seconds: Maximum wall-clock seconds for a parallel batch.
            None (the default) disables the batch-level timeout.
        retry_wait_seconds: Seconds the QA cache requeue loop waits for
            IN_PROGRESS entries (QA path only, unread by the pool).
        max_requeue_count: Maximum QA cache requeues before forcing a fresh
            generation (QA path only, unread by the pool).
        answerer_concurrency_limits: Optional per-answerer concurrency caps
            (QA path only, unread by the pool).
        max_concurrent_requests: GlobalLLMLimiter capacity entered for the
            duration of a batch (unread by the pool: the scenario path
            passes it through its gate, and the QA path is configured by
            batch_runner).
    """

    max_workers: int = DEFAULT_ASYNC_MAX_WORKERS
    timeout_seconds: float | None = None
    retry_wait_seconds: float = 5.0
    max_requeue_count: int = 5
    answerer_concurrency_limits: dict[str, int] | None = None
    max_concurrent_requests: int | None = None


# ============================================================================
# Outcome
# ============================================================================


@dataclass
class PortalPoolOutcome:
    """Neutral outcome of a pooled run.

    Attributes:
        results_by_index: Successful worker return values keyed by the item's
            submission index (original ordering is the caller's to restore).
        failed_items: ``(description, exception)`` pairs for items that
            raised, were cancelled, or were synthetically failed by the
            drop-detection invariant.
        timed_out: True when the batch-level wall-clock timeout fired.
        diagnostics: Timeout snapshot for operator-facing messages:
            ``completed_at_timeout`` (results harvested when the timeout
            fired, after the immediate done-future sweep),
            ``in_flight_at_timeout`` (indices not yet in results at that
            moment, including not-started ones, matching the scenario twin's
            convention), and ``recovered_from_drain`` (how many of those
            indices the post-shutdown drain recovered into results).
    """

    results_by_index: dict[int, Any] = field(default_factory=dict)
    failed_items: list[tuple[str, BaseException]] = field(default_factory=list)
    timed_out: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Lifecycle defaults
# ============================================================================


def _default_portal_factory() -> AbstractContextManager[BlockingPortal]:
    """Open a fresh asyncio BlockingPortal context manager."""
    return start_blocking_portal(backend="asyncio")


def _default_pre_teardown(portal: BlockingPortal) -> None:
    """Run the bounded pre-teardown adapter aclose sweep on ``portal``."""
    aclose_portal_adapters(portal, timeout=PRE_TEARDOWN_ACLOSE_TIMEOUT)


@contextlib.contextmanager
def sequential_portal(
    *,
    portal_factory: Callable[[], AbstractContextManager[BlockingPortal]] | None = None,
    pre_teardown: Callable[[BlockingPortal], None] | None = None,
) -> Iterator[BlockingPortal]:
    """Shared sequential-mode portal lifecycle.

    Opens one BlockingPortal, publishes it via the thread-local portal
    storage so sync ``invoke()`` calls reuse one event loop (instead of
    creating/destroying one per call, which causes httpx connection pool
    errors), and on exit runs the pre-teardown aclose sweep on the still-live
    portal loop before clearing the thread-local and tearing the portal down.

    Args:
        portal_factory: Optional factory returning the portal context
            manager. Callers inject a closure over their own module's
            ``start_blocking_portal`` binding so monkeypatch targets keep
            working. Defaults to a fresh asyncio portal.
        pre_teardown: Optional hook invoked with the live portal before
            teardown. Callers inject a closure over their own module's
            ``aclose_portal_adapters`` / ``PRE_TEARDOWN_ACLOSE_TIMEOUT``
            bindings. Defaults to the bounded adapter aclose sweep.

    Yields:
        The live BlockingPortal.
    """
    factory = portal_factory or _default_portal_factory
    sweep = pre_teardown or _default_pre_teardown
    with factory() as portal:
        set_async_portal(portal)
        try:
            yield portal
        finally:
            # Pre-teardown aclose: close portal-pinned adapter clients on the
            # shared portal's live loop and drop the per-portal tracking
            # BEFORE the enclosing `with` tears the portal down. Downstream
            # cleanup_resources() runs on a different loop and cannot safely
            # close httpx transports pinned to this one.
            sweep(portal)
            set_async_portal(None)


# ============================================================================
# Parallel pool
# ============================================================================


def run_in_portal_pool(
    items: Sequence[T],
    worker_fn: Callable[[int, T], R],
    *,
    tuning: ExecutionTuning,
    on_progress: Callable[[int, int, R], None] | None = None,
    on_item_start: Callable[[int, int, T], None] | None = None,
    describe: Callable[[int, T], str] | None = None,
    gate: AbstractContextManager[Any] | None = None,
    portal_factory: Callable[[], AbstractContextManager[BlockingPortal]] | None = None,
    pre_teardown: Callable[[BlockingPortal], None] | None = None,
    progress_lock: threading.Lock | None = None,
    item_noun: str = "Task",
) -> PortalPoolOutcome:
    """Run ``worker_fn`` over ``items`` on a portal-backed thread pool.

    Faithful extraction of the parallel skeleton shared by the two
    executor twins. Every submitted index ends in
    ``outcome.results_by_index`` or ``outcome.failed_items`` (drop-detection
    invariant). The helper never raises for per-item failures or batch
    timeout. Translating the outcome into each executor's public contract
    (``VerificationBatchError`` vs ``(results, errors)``) is the caller's
    job.

    Args:
        items: Work items, submitted in order with their index.
        worker_fn: ``(index, item) -> result``, executed on a worker thread
            after the worker's portal is ensured. Exceptions (including
            BaseException) become per-item failure entries.
        tuning: Shared execution knobs (worker count, batch timeout).
        on_progress: Optional completion callback ``(completed, total,
            result)``, dispatched under the progress lock for results
            harvested by the live ``as_completed`` loop. Results recovered
            by the timeout sweeps do NOT dispatch it (twin behavior).
        on_item_start: Optional preview callback ``(completed + 1, total,
            item)``, dispatched under the progress lock from the worker
            thread before ``worker_fn`` runs.
        describe: ``(index, item) -> label`` used in failure entries and
            logs. Defaults to ``repr(item)``.
        gate: Optional context manager entered around the pooled run,
            intended for ``GlobalLLMLimiter.configure`` (ref-counted, so a
            nested gate cannot lift an outer cap).
        portal_factory: Optional per-worker portal factory (see
            :func:`sequential_portal` for the monkeypatch rationale).
        pre_teardown: Optional per-portal pre-teardown hook (defaults to the
            bounded adapter aclose sweep).
        progress_lock: Optional caller-supplied lock for progress dispatch.
            Callers that update their own progress state from worker threads
            (the scenario per-turn callback) pass their lock here so ALL
            progress mutation is serialized under one lock.
        item_noun: Capitalized noun used in synthetic failure messages
            ("Task" for QA, "Combo" for scenarios).

    Returns:
        A :class:`PortalPoolOutcome` with results, failures, the timeout
        flag, and timeout diagnostics.
    """
    factory = portal_factory or _default_portal_factory
    sweep = pre_teardown or _default_pre_teardown
    label = describe or (lambda _idx, item: repr(item))
    lock = progress_lock or threading.Lock()
    total = len(items)
    completed_count = [0]

    # Per-worker portal management: each worker thread creates one portal
    # that is reused across all items on that thread. This preserves httpx
    # connection pools and avoids "Event loop is closed" errors from rapid
    # portal churn.
    #
    # Each entry is (context_manager, portal). The portal reference is kept
    # separately so the finally block can look up adapters tracked against
    # that portal in the registry and call their aclose() on the portal's
    # own event loop BEFORE the portal is torn down.
    _portal_resources: list[tuple[Any, BlockingPortal]] = []
    _portal_init_lock = threading.Lock()

    def _ensure_worker_portal() -> None:
        """Lazily create a BlockingPortal for this worker thread."""
        if get_async_portal() is not None:
            return
        cm = factory()
        portal = cm.__enter__()
        set_async_portal(portal)
        with _portal_init_lock:
            _portal_resources.append((cm, portal))

    def _run_item(idx: int, item: T) -> R:
        """Worker wrapper: ensure the portal, fire the preview, run the item."""
        _ensure_worker_portal()
        if on_item_start is not None:
            with lock:
                on_item_start(completed_count[0] + 1, total, item)
        return worker_fn(idx, item)

    results_by_index: dict[int, Any] = {}
    failed_items: list[tuple[str, BaseException]] = []
    timed_out = False
    in_flight_at_timeout: set[int] = set()
    completed_at_timeout = 0

    with gate if gate is not None else contextlib.nullcontext():
        pool = ThreadPoolExecutor(max_workers=tuning.max_workers)
        try:
            # Submit all items.
            future_to_meta: dict[Future[R], tuple[int, str]] = {}
            for idx, item in enumerate(items):
                future = pool.submit(_run_item, idx, item)
                future_to_meta[future] = (idx, label(idx, item))

            # Collect results as they complete.
            collected: set[Future[R]] = set()
            try:
                for future in as_completed(future_to_meta, timeout=tuning.timeout_seconds):
                    collected.add(future)
                    idx, desc = future_to_meta[future]
                    try:
                        results_by_index[idx] = future.result()
                        if on_progress is not None:
                            with lock:
                                completed_count[0] += 1
                                on_progress(completed_count[0], total, results_by_index[idx])
                    except BaseException as e:
                        logger.error("Parallel %s failed: %s: %s", item_noun.lower(), desc, e)
                        failed_items.append((desc, e))
            except TimeoutError:
                timed_out = True
                # Sweep futures that completed between last yield and timeout.
                for future, (idx, desc) in future_to_meta.items():
                    if future.done() and future not in collected:
                        collected.add(future)
                        try:
                            results_by_index[idx] = future.result()
                        except BaseException as e:
                            failed_items.append((desc, e))
                # Snapshot indices that were in-flight (not in
                # results_by_index) at this moment, before the post-shutdown
                # drain runs. The drain may recover some of these, but the
                # diagnostics still report them so operators can see what was
                # running at the critical moment.
                completed_at_timeout = len(results_by_index)
                in_flight_at_timeout = {idx for idx in range(total) if idx not in results_by_index}
        finally:
            # Always wait for pool workers to finish before tearing down
            # worker portals. With wait=False, in-flight workers could
            # outlive their portals and crash on callback dispatch with
            # "cannot schedule new futures after shutdown", or set Future
            # results that are never harvested (silent drop).
            pool.shutdown(wait=True, cancel_futures=timed_out)

            # Post-shutdown sweep: after wait=True returns, every future is
            # in a terminal state. Harvest any futures that finished while
            # the pool was draining, so in-flight items are reflected in the
            # outcome instead of being silently dropped.
            if timed_out:
                for future, (idx, desc) in future_to_meta.items():
                    if future in collected:
                        continue
                    if future.cancelled():
                        failed_items.append(
                            (
                                desc,
                                TimeoutError(f"{item_noun} cancelled before start: {desc}"),
                            )
                        )
                        collected.add(future)
                        continue
                    try:
                        results_by_index[idx] = future.result()
                    except BaseException as e:
                        failed_items.append((desc, e))
                    collected.add(future)

            # Drop-detection invariant. After pool.shutdown(wait=True) every
            # future must be in a terminal state, so the sweeps above should
            # have covered everything. If this fires, the shutdown race has
            # reopened and items were being lost.
            uncollected = [(future, meta) for future, meta in future_to_meta.items() if future not in collected]
            if uncollected:
                logger.error(
                    "Portal pool dropped %d %ss after pool.shutdown(wait=True). Emitting synthetic failure entries.",
                    len(uncollected),
                    item_noun.lower(),
                )
                for future, (_idx, desc) in uncollected:
                    failed_items.append(
                        (
                            desc,
                            TimeoutError(f"{item_noun} left uncollected by parallel executor: {desc}"),
                        )
                    )
                    collected.add(future)

            # Pre-teardown aclose: close adapter-owned httpx clients on the
            # portal loop that opened them, BEFORE the portal is torn down.
            # Without this, the downstream cleanup_resources() call runs on a
            # fresh loop and httpx raises "Event loop is closed" because its
            # transports are pinned to the dead portal loop.
            for _cm, portal in _portal_resources:
                sweep(portal)

            # Clean up worker portals (event loops) only after workers have
            # fully exited, so no worker can still be using a portal when its
            # context manager exits.
            for cm, _portal in _portal_resources:
                try:
                    cm.__exit__(None, None, None)
                except Exception:
                    logger.debug("Portal cleanup error", exc_info=True)

    recovered_from_drain = sum(1 for idx in in_flight_at_timeout if idx in results_by_index)
    return PortalPoolOutcome(
        results_by_index=results_by_index,
        failed_items=failed_items,
        timed_out=timed_out,
        diagnostics={
            "completed_at_timeout": completed_at_timeout,
            "in_flight_at_timeout": in_flight_at_timeout,
            "recovered_from_drain": recovered_from_drain,
        },
    )
