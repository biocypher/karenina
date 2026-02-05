"""Verification task execution with parallel/sequential support.

This module provides the VerificationExecutor class for running verification
tasks either sequentially or in parallel using a thread pool with asyncio
BlockingPortal for proper async event loop management.

The thread-local portal storage allows worker threads to share async execution
context without passing the portal through all function signatures.
"""

from __future__ import annotations

import contextlib
import logging
import os
import queue
import random
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from anyio.from_thread import start_blocking_portal

if TYPE_CHECKING:
    from anyio.from_thread import BlockingPortal

from ...schemas.verification.config import DEFAULT_ASYNC_MAX_WORKERS
from ...schemas.workflow import VerificationResult
from ...utils.answer_cache import AnswerTraceCache

logger = logging.getLogger(__name__)

# ============================================================================
# Thread-Local Portal Storage
# ============================================================================

_portal_storage = threading.local()


def get_async_portal() -> BlockingPortal | None:
    """Get the current async portal for running async code from threads.

    Each worker thread has its own thread-local portal reference.

    Returns:
        The BlockingPortal if one is active for this thread, None otherwise
    """
    return getattr(_portal_storage, "portal", None)


def set_async_portal(portal: BlockingPortal | None) -> None:
    """Set the async portal for the current thread.

    Args:
        portal: The BlockingPortal to use, or None to clear
    """
    _portal_storage.portal = portal


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
    """

    max_workers: int = DEFAULT_ASYNC_MAX_WORKERS
    enable_cache: bool = True
    retry_wait_seconds: float = 5.0


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

        Args:
            tasks: List of task dictionaries
            progress_callback: Optional progress callback

        Returns:
            Dictionary mapping result keys to verification results
        """
        from .batch_runner import execute_task
        from .utils.cache_helpers import log_cache_stats
        from .utils.task_helpers import create_preview_result

        answer_cache = AnswerTraceCache() if self.config.enable_cache else None
        results: dict[str, VerificationResult] = {}
        total = len(tasks)

        for idx, task in enumerate(tasks, 1):
            # Call progress callback BEFORE starting task (with preview result)
            if progress_callback:
                preview_result = create_preview_result(task)
                progress_callback(idx, total, preview_result)

            # Execute the task with answer cache
            result_key, result = execute_task(task, answer_cache)
            results[result_key] = result

            # Call progress callback AFTER completion (with actual result)
            if progress_callback:
                progress_callback(idx, total, result)

        # Log cache statistics
        if answer_cache:
            log_cache_stats(answer_cache, mode="sequential")

        return results

    def _run_parallel(
        self,
        tasks: list[dict[str, Any]],
        progress_callback: Callable[[int, int, VerificationResult | None], None] | None,
    ) -> dict[str, VerificationResult]:
        """Execute tasks in parallel with thread pool.

        Uses intelligent retry and answer cache optimization with task shuffling
        and progressive retry to maximize cache hits while avoiding blocking.

        Uses AnyIO BlockingPortal to properly manage async event loops across
        worker threads, preventing connection pool degradation.

        Args:
            tasks: List of task dictionaries
            progress_callback: Optional progress callback

        Returns:
            Dictionary mapping result keys to verification results (in original task order)
        """
        from .batch_runner import execute_task
        from .utils.cache_helpers import generate_answer_cache_key, log_cache_stats
        from .utils.task_helpers import create_preview_result

        max_workers = self.config.max_workers
        logger.info(f"Parallel execution: {len(tasks)} tasks with {max_workers} workers")

        # Create thread-safe answer cache for sharing traces across judges
        answer_cache = AnswerTraceCache() if self.config.enable_cache else None

        # Preserve original order and shuffle for better cache distribution
        indexed_tasks = [(idx, task) for idx, task in enumerate(tasks)]
        random.shuffle(indexed_tasks)

        # Create work queue with (original_index, task, retry_count)
        work_queue: queue.Queue[tuple[int, dict[str, Any], int] | None] = queue.Queue()
        for idx, task in indexed_tasks:
            work_queue.put((idx, task, 0))

        # Thread-safe storage for results (indexed by original position)
        results_lock = threading.Lock()
        results_by_index: dict[int, tuple[str, VerificationResult]] = {}

        # Thread-safe progress tracking
        progress_lock = threading.Lock()
        completed_count = [0]
        total = len(tasks)

        # Completion tracking
        tasks_completed_event = threading.Event()

        retry_wait_seconds = self.config.retry_wait_seconds

        def worker(portal: BlockingPortal) -> None:
            """Worker function that processes tasks from queue with retry logic."""
            # Each worker sets its own thread-local portal
            set_async_portal(portal)
            try:
                while True:
                    try:
                        # Get next task (non-blocking to allow checking for shutdown)
                        try:
                            item = work_queue.get(timeout=1.0)
                        except queue.Empty:
                            # Check if all tasks are completed
                            with results_lock:
                                if len(results_by_index) == total:
                                    tasks_completed_event.set()
                            continue

                        # Check for shutdown signal
                        if item is None:
                            work_queue.task_done()
                            break

                        original_index, task, retry_count = item

                        # Check answer cache first
                        if answer_cache:
                            cache_key = generate_answer_cache_key(task)
                            status, cached_answer_data = answer_cache.get_or_reserve(cache_key)

                            if status == "IN_PROGRESS":
                                # Another worker is generating this answer
                                # Always call task_done() immediately after get()
                                work_queue.task_done()

                                if retry_count == 0:
                                    # First encounter: immediately requeue and move to next task
                                    logger.debug(
                                        f"Task {cache_key} in progress (retry={retry_count}), requeueing immediately"
                                    )
                                    work_queue.put((original_index, task, retry_count + 1))
                                    continue
                                else:
                                    # Subsequent encounter: use cache event-based waiting
                                    completed = answer_cache.wait_for_completion(cache_key, timeout=retry_wait_seconds)
                                    if not completed:
                                        logger.debug(
                                            f"Task {cache_key} still in progress after {retry_wait_seconds}s, requeueing"
                                        )
                                    work_queue.put((original_index, task, retry_count + 1))
                                    continue
                        else:
                            status, cached_answer_data = "MISS", None

                        # Status is MISS or HIT - ready to execute
                        # Call preview progress callback
                        if progress_callback:
                            preview_result = create_preview_result(task)
                            with progress_lock:
                                progress_callback(completed_count[0] + 1, total, preview_result)

                        # Execute the task with pre-checked cache status
                        try:
                            result_key, verification_result = execute_task(
                                task, answer_cache, cache_status=status, cached_answer_data=cached_answer_data
                            )

                            # Store result at original index
                            with results_lock:
                                results_by_index[original_index] = (result_key, verification_result)
                                # Check if all tasks are now complete
                                if len(results_by_index) == total:
                                    tasks_completed_event.set()

                            # Call completion progress callback
                            if progress_callback:
                                with progress_lock:
                                    completed_count[0] += 1
                                    progress_callback(completed_count[0], total, verification_result)

                        except Exception as e:
                            logger.error(f"Task execution failed: {e}")
                            # Don't raise - log and continue

                        finally:
                            work_queue.task_done()

                    except Exception as e:
                        logger.error(f"Worker error: {e}")
                        with contextlib.suppress(ValueError):
                            # task_done() called more times than get()
                            work_queue.task_done()

            finally:
                # Clear the portal when worker exits
                set_async_portal(None)

        # Use BlockingPortal to properly manage async event loop for all worker threads
        with start_blocking_portal(backend="asyncio") as portal:
            # Start worker threads, each with a reference to the shared portal
            workers = []
            for _ in range(max_workers):
                t = threading.Thread(target=worker, args=(portal,), daemon=True)
                t.start()
                workers.append(t)

            # Wait for all tasks to actually complete (not just queue empty)
            tasks_completed_event.wait()

            # Send shutdown signal to workers
            for _ in range(max_workers):
                work_queue.put(None)

            # Wait for workers to finish
            for t in workers:
                t.join(timeout=5.0)

        # Restore original order and convert to dictionary
        results: dict[str, VerificationResult] = {}
        for idx in sorted(results_by_index.keys()):
            result_key, verification_result = results_by_index[idx]
            results[result_key] = verification_result

        # Log cache statistics
        if answer_cache:
            log_cache_stats(answer_cache, mode="parallel mode")

        return results


def get_default_max_workers() -> int:
    """Get the default max workers from environment variable or default.

    Returns:
        Max workers value from KARENINA_ASYNC_MAX_WORKERS env var or 2
    """
    return int(os.getenv("KARENINA_ASYNC_MAX_WORKERS", str(DEFAULT_ASYNC_MAX_WORKERS)))
