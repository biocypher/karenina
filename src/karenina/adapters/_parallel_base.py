"""Shared utilities for parallel invokers.

This module provides common utility functions used by parallel invokers
(LLMParallelInvoker).

Uses composition over inheritance to avoid complex generic typing issues
while sharing async execution patterns.

Environment Variables:
- KARENINA_ASYNC_ENABLED: Enable/disable parallel execution (default: true)
- KARENINA_ASYNC_MAX_WORKERS: Max concurrent workers (default: 2)
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import logging
import os
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from anyio.from_thread import BlockingPortal

logger = logging.getLogger(__name__)

T = TypeVar("T")


def read_async_config() -> tuple[bool, int]:
    """Read async configuration from environment variables.

    This function implements the standard pattern for reading KARENINA_ASYNC_*
    environment variables, used by all parallel invokers.

    Returns:
        Tuple of (async_enabled, max_workers) with defaults applied.

    Example:
        >>> async_enabled, max_workers = read_async_config()
        >>> if async_enabled:
        ...     invoker.max_workers = max_workers
    """
    # Read async_enabled
    async_enabled = True  # Default
    env_val = os.getenv("KARENINA_ASYNC_ENABLED")
    if env_val is not None:
        async_enabled = env_val.lower() in ("true", "1", "yes")

    # Read max_workers
    max_workers = 2  # Default
    env_val = os.getenv("KARENINA_ASYNC_MAX_WORKERS")
    if env_val is not None:
        with contextlib.suppress(ValueError):
            max_workers = int(env_val)

    return async_enabled, max_workers


def get_max_workers(override: int | None = None) -> int:
    """Get max_workers with env var fallback.

    Args:
        override: Explicit max_workers value. If provided, takes precedence.

    Returns:
        The effective max_workers value.

    Example:
        >>> workers = get_max_workers(override=None)  # Uses env var or default
        >>> workers = get_max_workers(override=8)     # Uses explicit value
    """
    if override is not None:
        return override

    env_val = os.getenv("KARENINA_ASYNC_MAX_WORKERS")
    if env_val is not None:
        with contextlib.suppress(ValueError):
            return int(env_val)

    return 2  # Default


async def run_with_semaphore(
    tasks: list[Coroutine[Any, Any, T]],
    max_workers: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[tuple[T | None, Exception | None]]:
    """Run coroutines with semaphore-based concurrency limiting.

    This utility executes multiple coroutines concurrently while limiting
    the number of simultaneous executions via a semaphore.

    Args:
        tasks: List of coroutines to execute.
        max_workers: Maximum number of concurrent tasks.
        progress_callback: Optional callback(completed, total) for progress.

    Returns:
        List of (result, error) tuples in the same order as input tasks.
        - result: The coroutine's return value, or None if error
        - error: Exception if the task failed, or None if success

    Example:
        >>> async def my_task(i: int) -> str:
        ...     await asyncio.sleep(0.1)
        ...     return f"result_{i}"
        >>> tasks = [my_task(i) for i in range(10)]
        >>> results = await run_with_semaphore(tasks, max_workers=3)
    """
    if not tasks:
        return []

    total = len(tasks)
    semaphore = asyncio.Semaphore(max_workers)
    progress_lock = asyncio.Lock()
    completed_count = 0

    async def execute_task(index: int, coro: Coroutine[Any, Any, T]) -> tuple[int, T | None, Exception | None]:
        """Execute a single task and return (index, result, error)."""
        nonlocal completed_count

        async with semaphore:
            try:
                result = await coro
                return index, result, None
            except Exception as e:
                logger.debug(f"Task {index} failed: {e}")
                return index, None, e
            finally:
                if progress_callback:
                    async with progress_lock:
                        completed_count += 1
                        progress_callback(completed_count, total)

    # Execute all tasks concurrently with gather
    task_coroutines = [execute_task(i, coro) for i, coro in enumerate(tasks)]
    raw_results = await asyncio.gather(*task_coroutines, return_exceptions=False)

    # Build ordered results list
    results: list[tuple[T | None, Exception | None]] = [(None, None)] * total
    for index, result, error in raw_results:
        results[index] = (result, error)

    return results


def sync_invoke_via_portal(
    async_fn: Callable[..., Coroutine[Any, Any, T]],
    *args: Any,
    **kwargs: Any,
) -> T:
    """Invoke an async function synchronously using BlockingPortal if available.

    This utility provides a standard pattern for calling async functions
    from sync code, using the shared BlockingPortal from batch_runner.py
    when available, or falling back to appropriate alternatives.

    Args:
        async_fn: The async function to call.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        The result of the async function.

    Raises:
        Any exception raised by the async function.

    Example:
        >>> async def fetch_data(url: str) -> dict:
        ...     ...
        >>> data = sync_invoke_via_portal(fetch_data, "https://example.com")
    """
    from ..benchmark.verification.executor import get_async_portal

    portal = get_async_portal()

    if portal is not None:
        # Use the shared BlockingPortal for proper event loop management
        result: T = portal.call(async_fn, *args, **kwargs)
        return result

    # No portal available - check if we're already in an async context
    try:
        asyncio.get_running_loop()
        # We're in an async context - use ThreadPoolExecutor to avoid
        # nested event loop issues
        logger.debug("sync_invoke_via_portal: Running in async context, using ThreadPoolExecutor")

        def run_in_thread() -> T:
            return asyncio.run(async_fn(*args, **kwargs))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            return future.result(timeout=600)  # 10 minute timeout

    except RuntimeError:
        # No event loop running, safe to use asyncio.run
        logger.debug("sync_invoke_via_portal: No event loop, using asyncio.run()")
        return asyncio.run(async_fn(*args, **kwargs))


def get_portal() -> BlockingPortal | None:
    """Get the current async portal for running async code from threads.

    This is a convenience wrapper around batch_runner.get_async_portal()
    that avoids import issues in type hints.

    Returns:
        The BlockingPortal if one is active, None otherwise
    """
    from ..benchmark.verification.executor import get_async_portal

    return get_async_portal()
