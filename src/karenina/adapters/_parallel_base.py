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
import contextvars
import functools
import logging
import os
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any, TypeVar

# Leaf module import (stdlib-only at module level), hoisted out of the
# per-invoke wrapper body. The semaphore lookup point for monkeypatching is
# now this module's global name: karenina.adapters._parallel_base.
# get_global_llm_semaphore.
from karenina.benchmark.verification.async_lifecycle import get_global_llm_semaphore

from ..schemas.verification.config import DEFAULT_ASYNC_ENABLED, DEFAULT_ASYNC_MAX_WORKERS

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
    async_enabled = DEFAULT_ASYNC_ENABLED
    env_val = os.getenv("KARENINA_ASYNC_ENABLED")
    if env_val is not None:
        async_enabled = env_val.lower() in ("true", "1", "yes")

    # Read max_workers
    max_workers = DEFAULT_ASYNC_MAX_WORKERS
    env_val = os.getenv("KARENINA_ASYNC_MAX_WORKERS")
    if env_val is not None:
        with contextlib.suppress(ValueError):
            max_workers = int(env_val)

    return async_enabled, max_workers


def with_llm_semaphore(fn: Callable[..., T]) -> Callable[..., T]:
    """Wrap a sync LLM invoke() call with global semaphore acquisition.

    When a global LLM semaphore is active (set by ScenarioExecutor or
    VerificationExecutor), this decorator acquires one permit before calling
    the wrapped function and releases it after (including on exception).

    When no global semaphore is set, the function is called directly with
    no overhead.

    Args:
        fn: The sync invoke() method to wrap.

    Returns:
        Wrapped function with semaphore acquisition.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        sem = get_global_llm_semaphore()
        if sem is not None:
            sem.acquire()
        try:
            return fn(*args, **kwargs)
        finally:
            if sem is not None:
                sem.release()

    return wrapper


def run_coro_in_thread(
    coro_func: Callable[..., Coroutine[Any, Any, T]],
    *args: Any,
    timeout: float | None,
) -> T:
    """Run an async function in a fresh thread, propagating contextvars.

    Shared helper for the adapters' sync-wrapper ThreadPoolExecutor
    fallbacks (sync invoke called while an event loop is already running
    in the calling thread). A plain ``executor.submit(asyncio.run, ...)``
    would run the coroutine without the caller's context, silently losing
    context-bound state such as the ``track_retries`` telemetry tracker.
    This helper captures the caller's context with
    ``contextvars.copy_context()`` and re-enters it in the worker thread,
    so the coroutine (and every retry decision inside it) sees the same
    contextvars as the caller.

    Args:
        coro_func: Async function to call.
        *args: Positional arguments forwarded to coro_func.
        timeout: Wall-clock bound for the thread result in seconds.
            None waits indefinitely.

    Returns:
        The return value of the coroutine.

    Raises:
        concurrent.futures.TimeoutError: If the thread does not finish
            within the timeout.
        Exception: Any exception raised by the coroutine is re-raised.
    """
    ctx = contextvars.copy_context()

    def _create_and_run() -> T:
        # Both coroutine creation and asyncio.run happen inside ctx, so a
        # sync prelude in coro_func sees the caller's contextvars too, and
        # the task created by asyncio.run copies ctx for the async body.
        return asyncio.run(coro_func(*args))

    def _target() -> T:
        return ctx.run(_create_and_run)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_target)
        return future.result(timeout=timeout)


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

    return DEFAULT_ASYNC_MAX_WORKERS


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
        # We're in an async context - use a fresh thread (with the caller's
        # context propagated) to avoid nested event loop issues
        logger.debug("sync_invoke_via_portal: Running in async context, using ThreadPoolExecutor")

        def _call() -> Coroutine[Any, Any, T]:
            return async_fn(*args, **kwargs)

        from karenina.adapters._timeouts import PORTAL_DISPATCH_FLOOR, compute_sync_wrapper_timeout

        # No model config is available here, so the bound is the historical
        # floor for the generic portal dispatch path (10 minutes).
        return run_coro_in_thread(_call, timeout=compute_sync_wrapper_timeout(None, floor=PORTAL_DISPATCH_FLOOR))

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
