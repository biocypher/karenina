"""Verification task execution with parallel/sequential support.

This module provides the VerificationExecutor class for running verification
tasks either sequentially or in parallel using ThreadPoolExecutor with asyncio
BlockingPortal for proper async event loop management.

The thread-local portal storage allows worker threads to share async execution
context without passing the portal through all function signatures.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from anyio.from_thread import start_blocking_portal

if TYPE_CHECKING:
    from anyio.from_thread import BlockingPortal

from karenina.schemas.verification import VerificationResult
from karenina.schemas.verification.config import DEFAULT_ASYNC_MAX_WORKERS
from karenina.utils.answer_cache import AnswerTraceCache

logger = logging.getLogger(__name__)

_SENTINEL = object()  # Distinguishes "attribute missing" from "attribute is None"

# Bound (seconds) on each adapter.aclose() call issued via the worker portal
# before the portal is torn down. A stuck aclose must not wedge the finally
# block. Mirrors the pattern in langchain/parser.py:297-312. Tests can
# monkey-patch this to a shorter value to exercise the timeout branch.
PRE_TEARDOWN_ACLOSE_TIMEOUT = 5.0

# ============================================================================
# Thread-Local Portal Storage
# ============================================================================

_portal_storage = threading.local()


def get_async_portal() -> BlockingPortal | None:
    """Get the current async portal for running async code from threads.

    Each worker thread has its own thread-local portal reference.
    Returns None (and clears the stale reference) if the portal's event loop
    has ended, allowing callers to fall back to asyncio.run().

    Returns:
        The BlockingPortal if one is active for this thread, None otherwise
    """
    portal = getattr(_portal_storage, "portal", None)
    if portal is not None:
        thread_id = getattr(portal, "_event_loop_thread_id", _SENTINEL)
        if thread_id is None:
            logger.warning("Clearing stale portal reference (event loop thread ended)")
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
# Configuration
# ============================================================================


@dataclass
class ExecutorConfig:
    """Configuration for verification execution.

    Attributes:
        max_workers: Maximum number of parallel workers (default: 2)
        enable_cache: Whether to enable answer caching (default: True)
        retry_wait_seconds: Seconds to wait for IN_PROGRESS cache entries (default: 5.0)
        timeout_seconds: Maximum wall-clock seconds for a parallel batch.
            Set to None (the default) to disable the batch-level timeout; the
            executor then runs until all tasks finish. Set to a positive float
            to enforce a ceiling.
        max_requeue_count: Maximum times a task can be requeued before forcing a
            fresh generation (default: 5). When exceeded, the cache entry is reset
            and the task restarts with retry_count=0.
    """

    max_workers: int = DEFAULT_ASYNC_MAX_WORKERS
    enable_cache: bool = True
    retry_wait_seconds: float = 5.0
    timeout_seconds: float | None = None
    max_requeue_count: int = 5


# ============================================================================
# Executor
# ============================================================================


class VerificationExecutor:
    """Executes verification tasks with parallel/sequential support.

    This class encapsulates the execution logic for running verification tasks,
    supporting both sequential and parallel execution modes with proper async
    event loop management via AnyIO BlockingPortal.

    Example:
        >>> executor = VerificationExecutor(parallel=True, config=ExecutorConfig(max_workers=4))
        >>> results = executor.run_batch(tasks, progress_callback=my_callback)
    """

    def __init__(self, parallel: bool = True, config: ExecutorConfig | None = None):
        """Initialize the verification executor.

        Args:
            parallel: Whether to run tasks in parallel (default: True)
            config: Optional execution configuration
        """
        self.parallel = parallel
        self.config = config or ExecutorConfig()

    def run_batch(
        self,
        tasks: list[dict[str, Any]],
        progress_callback: Callable[[int, int, VerificationResult | None], None] | None = None,
    ) -> dict[str, VerificationResult]:
        """Execute verification tasks.

        Args:
            tasks: List of task dictionaries with verification parameters
            progress_callback: Optional callback(current, total, result | None) for progress updates.
                             Called before starting each task (with preview result) and
                             after completion (with actual result).

        Returns:
            Dictionary mapping result keys to verification results
        """
        if self.parallel:
            return self._run_parallel(tasks, progress_callback)
        else:
            return self._run_sequential(tasks, progress_callback)

    def _run_sequential(
        self,
        tasks: list[dict[str, Any]],
        progress_callback: Callable[[int, int, VerificationResult | None], None] | None,
    ) -> dict[str, VerificationResult]:
        """Execute tasks one at a time.

        On failure, logs the error, records it, and continues processing the
        remaining tasks. If any tasks failed, raises VerificationBatchError
        with partial results after the loop completes.

        Args:
            tasks: List of task dictionaries
            progress_callback: Optional progress callback

        Returns:
            Dictionary mapping result keys to verification results

        Raises:
            VerificationBatchError: If one or more tasks failed during execution.
        """
        from karenina.exceptions import VerificationBatchError

        from .batch_runner import execute_task
        from .utils.cache_helpers import log_cache_stats
        from .utils.task_helpers import create_preview_result

        answer_cache = AnswerTraceCache() if self.config.enable_cache else None
        results: dict[str, VerificationResult] = {}
        errors: list[tuple[str, BaseException]] = []
        total = len(tasks)

        # Use a shared BlockingPortal so that sync invoke() calls reuse the
        # same event loop instead of creating/destroying one per call via
        # asyncio.run(). This prevents httpx connection pool errors caused
        # by closed event loops.
        with start_blocking_portal(backend="asyncio") as portal:
            set_async_portal(portal)
            try:
                for idx, task in enumerate(tasks, 1):
                    # Call progress callback BEFORE starting task (with preview result)
                    if progress_callback:
                        preview_result = create_preview_result(task)
                        progress_callback(idx, total, preview_result)

                    try:
                        # Execute the task with answer cache
                        result_key, result = execute_task(task, answer_cache)
                        results[result_key] = result
                    except Exception as e:
                        logger.error("Sequential task %s failed: %s", task["question_id"], e)
                        errors.append((task["question_id"], e))
                        continue

                    # Call progress callback AFTER completion (with actual result)
                    if progress_callback:
                        progress_callback(idx, total, result)
            finally:
                # Drop per-portal adapter tracking so a sequential run does
                # not leak entries into the module-global map across batches.
                # The shared portal is about to be torn down by the enclosing
                # `with` statement, and downstream cleanup_resources() will
                # still close the adapters (on its own fresh loop). Sequential
                # mode was never affected by the parallel teardown ordering
                # bug, so there is no need to pre-close here.
                from karenina.adapters.registry import clear_portal_adapter_refs

                clear_portal_adapter_refs(portal)
                set_async_portal(None)

        # Log cache statistics
        if answer_cache:
            log_cache_stats(answer_cache, mode="sequential")

        if errors:
            raise VerificationBatchError(
                f"{len(errors)} of {total} verification tasks failed",
                partial_results=results,
                errors=errors,
            )

        return results

    def _run_parallel(
        self,
        tasks: list[dict[str, Any]],
        progress_callback: Callable[[int, int, VerificationResult | None], None] | None,
    ) -> dict[str, VerificationResult]:
        """Execute tasks in parallel using ThreadPoolExecutor.

        Each task is submitted as a separate Future with its own BlockingPortal.
        Cache retry logic runs internally within each callable. Every submission
        produces a Future, so no task can be silently lost.

        Args:
            tasks: List of task dictionaries
            progress_callback: Optional progress callback

        Returns:
            Dictionary mapping result keys to verification results (in original task order)

        Raises:
            VerificationBatchError: If the batch times out or any tasks fail.
        """
        from karenina.exceptions import VerificationBatchError

        from .batch_runner import execute_task
        from .utils.cache_helpers import generate_answer_cache_key, log_cache_stats
        from .utils.task_helpers import create_preview_result

        max_workers = self.config.max_workers
        logger.info("Parallel execution: %d tasks with %d workers", len(tasks), max_workers)

        answer_cache = AnswerTraceCache() if self.config.enable_cache else None
        total = len(tasks)
        max_requeue = self.config.max_requeue_count
        retry_wait = self.config.retry_wait_seconds

        # Thread-safe progress tracking
        progress_lock = threading.Lock()
        completed_count = [0]

        # Per-worker portal management: each worker thread creates one portal
        # that is reused across all tasks on that thread. This preserves httpx
        # connection pools and avoids "Event loop is closed" errors from rapid
        # portal churn.
        #
        # Each entry is (context_manager, portal). The portal reference is
        # kept separately so the finally block can look up adapters tracked
        # against that portal in the registry and call their aclose() on the
        # portal's own event loop BEFORE the portal is torn down.
        _portal_resources: list[tuple[Any, BlockingPortal]] = []
        _portal_init_lock = threading.Lock()

        def _ensure_worker_portal() -> None:
            """Lazily create a BlockingPortal for this worker thread."""
            if get_async_portal() is not None:
                return
            cm = start_blocking_portal(backend="asyncio")
            portal = cm.__enter__()
            set_async_portal(portal)
            with _portal_init_lock:
                _portal_resources.append((cm, portal))

        def execute_task_with_retry(idx: int, task: dict[str, Any]) -> tuple[int, tuple[str, VerificationResult]]:
            """Execute a single verification task with cache retry."""
            _ensure_worker_portal()

            # Call preview progress callback
            if progress_callback:
                preview_result = create_preview_result(task)
                with progress_lock:
                    progress_callback(completed_count[0] + 1, total, preview_result)

            if not answer_cache:
                return idx, execute_task(task, answer_cache)

            cache_key = generate_answer_cache_key(task)
            for attempt in range(max_requeue + 1):
                status, cached_answer_data = answer_cache.get_or_reserve(cache_key)

                if status == "IN_PROGRESS":
                    if attempt == 0:
                        logger.debug(
                            "Task %s in progress (retry=%d), retrying immediately",
                            cache_key,
                            attempt,
                        )
                        continue

                    completed = answer_cache.wait_for_completion(cache_key, timeout=retry_wait)
                    if not completed:
                        logger.debug(
                            "Task %s still in progress after %ss, retrying",
                            cache_key,
                            retry_wait,
                        )
                    continue

                # MISS or HIT: execute
                return idx, execute_task(
                    task,
                    answer_cache,
                    cache_status=status,
                    cached_answer_data=cached_answer_data,
                )

            # Exceeded max requeue: force reset, generate fresh
            logger.warning(
                "Task %s exceeded max requeue count (%d), generating fresh",
                cache_key,
                max_requeue,
            )
            answer_cache.force_reset(cache_key)
            return idx, execute_task(task, answer_cache)

        # Preserve incoming order (ordering is controlled by task_ordering config)
        indexed_tasks = list(enumerate(tasks))

        results_by_index: dict[int, tuple[str, VerificationResult]] = {}
        failed_tasks: list[tuple[str, BaseException]] = []

        pool = ThreadPoolExecutor(max_workers=max_workers)
        try:
            # Submit all tasks
            future_to_meta: dict[Future[tuple[int, tuple[str, VerificationResult]]], tuple[int, str]] = {}
            for idx, task in indexed_tasks:
                future = pool.submit(execute_task_with_retry, idx, task)
                future_to_meta[future] = (idx, task["question_id"])

            # Collect results as they complete
            collected: set[Future[tuple[int, tuple[str, VerificationResult]]]] = set()
            timed_out = False
            try:
                for future in as_completed(future_to_meta, timeout=self.config.timeout_seconds):
                    collected.add(future)
                    idx, question_id = future_to_meta[future]
                    try:
                        result_idx, (result_key, verification_result) = future.result()
                        results_by_index[result_idx] = (result_key, verification_result)

                        if progress_callback:
                            with progress_lock:
                                completed_count[0] += 1
                                progress_callback(completed_count[0], total, verification_result)
                    except BaseException as e:
                        logger.error("Parallel task %s failed: %s", question_id, e)
                        failed_tasks.append((question_id, e))
            except TimeoutError:
                timed_out = True
                # Sweep futures that completed between last yield and timeout
                for future, (_idx, question_id) in future_to_meta.items():
                    if future.done() and future not in collected:
                        collected.add(future)
                        try:
                            result_idx, (result_key, verification_result) = future.result()
                            results_by_index[result_idx] = (result_key, verification_result)
                        except BaseException as e:
                            failed_tasks.append((question_id, e))
        finally:
            # Always wait for pool workers to finish before tearing down
            # worker portals. With wait=False, in-flight workers could
            # outlive their portals and crash on callback dispatch with
            # "cannot schedule new futures after shutdown", or set Future
            # results that are never harvested (silent drop).
            pool.shutdown(wait=True, cancel_futures=timed_out)

            # Post-shutdown sweep: after wait=True returns, every future is
            # in a terminal state. Harvest any futures that finished while
            # the pool was draining, so in-flight tasks are reflected in
            # partial_results or errors instead of being silently dropped.
            if timed_out:
                for future, (_idx, question_id) in future_to_meta.items():
                    if future in collected:
                        continue
                    if future.cancelled():
                        failed_tasks.append(
                            (
                                question_id,
                                TimeoutError(f"Task cancelled before start: {question_id}"),
                            )
                        )
                        collected.add(future)
                        continue
                    try:
                        result_idx, (result_key, verification_result) = future.result()
                        results_by_index[result_idx] = (result_key, verification_result)
                    except BaseException as e:
                        failed_tasks.append((question_id, e))
                    collected.add(future)

            # Drop-detection invariant. After pool.shutdown(wait=True) every
            # future must be in a terminal state, so the sweeps above should
            # have covered everything. If this fires, the shutdown race has
            # reopened and tasks were being lost.
            uncollected = [(future, meta) for future, meta in future_to_meta.items() if future not in collected]
            if uncollected:
                logger.error(
                    "Parallel executor dropped %d tasks after pool.shutdown(wait=True). "
                    "Emitting synthetic failure entries.",
                    len(uncollected),
                )
                for future, (_idx, question_id) in uncollected:
                    failed_tasks.append(
                        (
                            question_id,
                            TimeoutError(f"Task left uncollected by parallel executor: {question_id}"),
                        )
                    )
                    collected.add(future)

            # Pre-teardown aclose: close adapter-owned httpx clients on the
            # portal loop that opened them, BEFORE the portal is torn down.
            # Without this, the downstream cleanup_resources() call in
            # batch_runner runs on a fresh loop and httpx raises
            # "Event loop is closed" because its transports are pinned to
            # the dead portal loop. Bounded timeout mirrors the
            # start_task_soon + future.result pattern in
            # langchain/parser.py:297-312 to avoid wedging the finally
            # block on a stuck aclose.
            from karenina.adapters.registry import (
                clear_portal_adapter_refs,
                snapshot_adapters_for_portal,
            )

            for _cm, portal in _portal_resources:
                for adapter in snapshot_adapters_for_portal(portal):
                    if not hasattr(adapter, "aclose"):
                        continue
                    future = portal.start_task_soon(adapter.aclose)
                    try:
                        future.result(timeout=PRE_TEARDOWN_ACLOSE_TIMEOUT)
                    except TimeoutError:
                        # Cancel the abandoned coroutine so the portal's loop
                        # does not block the context manager's __exit__
                        # waiting for it to finish.
                        future.cancel()
                        logger.warning(
                            "Pre-teardown aclose timed out on %s (>%ss); proceeding with portal teardown",
                            type(adapter).__name__,
                            PRE_TEARDOWN_ACLOSE_TIMEOUT,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Pre-teardown aclose failed on %s: %s",
                            type(adapter).__name__,
                            exc,
                        )
                clear_portal_adapter_refs(portal)

            # Clean up worker portals (event loops) only after workers have
            # fully exited, so no worker can still be using a portal when
            # its context manager exits.
            for cm, _portal in _portal_resources:
                try:
                    cm.__exit__(None, None, None)
                except Exception:
                    logger.debug("Portal cleanup error", exc_info=True)

        # Restore original order and convert to dictionary
        results: dict[str, VerificationResult] = {}
        for idx in sorted(results_by_index.keys()):
            result_key, verification_result = results_by_index[idx]
            results[result_key] = verification_result

        # Log cache statistics
        if answer_cache:
            log_cache_stats(answer_cache, mode="parallel mode")

        # Handle timeout. as_completed only raises TimeoutError when a finite
        # timeout was passed, so timeout_seconds is guaranteed non-None here.
        if timed_out:
            timeout_label = (
                f"{self.config.timeout_seconds:.0f} seconds"
                if self.config.timeout_seconds is not None
                else "unknown timeout"
            )
            raise VerificationBatchError(
                f"Parallel batch timed out after {timeout_label} ({len(results)} of {total} tasks completed)",
                partial_results=results,
                errors=failed_tasks,
            )

        # Handle partial failures
        if failed_tasks:
            raise VerificationBatchError(
                f"{len(failed_tasks)} of {total} verification tasks failed",
                partial_results=results,
                errors=failed_tasks,
            )

        return results


def get_default_max_workers() -> int:
    """Get the default max workers from environment variable or default.

    Returns:
        Max workers value from KARENINA_ASYNC_MAX_WORKERS env var or 2
    """
    return int(os.getenv("KARENINA_ASYNC_MAX_WORKERS", str(DEFAULT_ASYNC_MAX_WORKERS)))
