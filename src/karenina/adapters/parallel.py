"""Parallel invocation utility for ParserPort adapters.

This module provides a reusable utility for executing multiple parser invocations
in parallel using asyncio.gather() with semaphore-based concurrency limiting.

Key characteristics:
- Uses asyncio.gather() for true async parallelism
- Preserves result ordering (critical requirement)
- Per-task error isolation (failed tasks don't block others)
- Integrates with existing BlockingPortal from batch_runner.py
- Same max_workers configuration pattern as ParallelLLMInvoker

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
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from ..ports import ParserPort

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class AdapterParallelInvoker:
    """Execute multiple ParserPort invocations in parallel using asyncio.gather().

    This utility is designed for scenarios where multiple independent parser calls
    need to be made (e.g., evaluating traits one-by-one in "sequential" mode when
    using a ParserPort adapter). By running these calls in parallel, we achieve
    significant speedup.

    Similar to ParallelLLMInvoker but works with ParserPort instead of LangChain:
    - asyncio.gather() with semaphore for concurrency control
    - Thread-safe result collection
    - Per-task error handling (failed tasks don't block others)

    Example usage:
        invoker = AdapterParallelInvoker(parser_adapter, max_workers=4)
        tasks = [
            ("Parse this response: ...", ResponseModel),
            ("Parse this response: ...", ResponseModel),
        ]
        results = invoker.invoke_batch(tasks)
        for result, usage, error in results:
            if error:
                print(f"Task failed: {error}")
            else:
                print(f"Result: {result}")
    """

    def __init__(
        self,
        parser: ParserPort,
        max_workers: int | None = None,
    ):
        """Initialize the adapter parallel invoker.

        Args:
            parser: ParserPort implementation for parsing operations.
            max_workers: Maximum number of concurrent parser calls. If None,
                        defaults to KARENINA_ASYNC_MAX_WORKERS env var or 2.
        """
        self.parser = parser
        self._max_workers = max_workers

    @property
    def max_workers(self) -> int:
        """Get the effective max_workers value."""
        if self._max_workers is not None:
            return self._max_workers

        env_val = os.getenv("KARENINA_ASYNC_MAX_WORKERS")
        if env_val is not None:
            with contextlib.suppress(ValueError):
                return int(env_val)

        return 2  # Default

    async def ainvoke_batch(
        self,
        tasks: Sequence[tuple[str, type[T]]],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[tuple[T | None, dict[str, Any] | None, Exception | None]]:
        """Invoke parser for multiple prompt/schema pairs in parallel (async).

        Each task is a tuple of (prompt_text, response_model_class). The parser is
        invoked with structured output for each task, and results are collected.

        Args:
            tasks: Sequence of (prompt_text, response_model_class) tuples.
                   prompt_text: The text to parse (typically LLM response text).
                   response_model_class: Pydantic model for structured output.
            progress_callback: Optional callback(completed, total) for progress.

        Returns:
            List of (result, usage_metadata, error) tuples in same order as input.
            - result: Parsed Pydantic model instance, or None if error
            - usage_metadata: Dict of usage info (currently empty for adapters), or None
            - error: Exception if the task failed, or None if success
        """
        if not tasks:
            return []

        total = len(tasks)
        # Use asyncio.Lock for thread-safe progress tracking
        progress_lock = asyncio.Lock()
        completed_count = 0

        # Semaphore for concurrency limiting
        semaphore = asyncio.Semaphore(self.max_workers)

        async def execute_task(
            index: int,
        ) -> tuple[int, T | None, dict[str, Any] | None, Exception | None]:
            """Execute a single task and return (index, result, usage, error)."""
            nonlocal completed_count

            prompt_text, model_class = tasks[index]

            async with semaphore:
                try:
                    result = await self.parser.aparse_to_pydantic(prompt_text, model_class)
                    # Note: ParserPort currently doesn't return usage metadata
                    # This could be extended in the future
                    usage_metadata: dict[str, Any] = {}
                    return index, result, usage_metadata, None
                except Exception as e:
                    logger.debug(f"AdapterParallelInvoker: Task {index} failed: {e}")
                    return index, None, None, e
                finally:
                    # Update progress
                    if progress_callback:
                        async with progress_lock:
                            completed_count += 1
                            progress_callback(completed_count, total)

        # Execute all tasks concurrently with gather
        task_coroutines = [execute_task(i) for i in range(total)]
        raw_results = await asyncio.gather(*task_coroutines, return_exceptions=False)

        # Build ordered results list
        results: list[tuple[T | None, dict[str, Any] | None, Exception | None]] = [(None, None, None)] * total
        for index, result, usage, error in raw_results:
            results[index] = (result, usage, error)

        return results

    def invoke_batch(
        self,
        tasks: Sequence[tuple[str, type[T]]],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[tuple[T | None, dict[str, Any] | None, Exception | None]]:
        """Invoke parser for multiple prompt/schema pairs in parallel (sync).

        This is a sync wrapper around ainvoke_batch(). Uses the shared BlockingPortal
        if available, otherwise falls back to asyncio.run() with proper event loop
        handling.

        Args:
            tasks: Sequence of (prompt_text, response_model_class) tuples.
                   prompt_text: The text to parse (typically LLM response text).
                   response_model_class: Pydantic model for structured output.
            progress_callback: Optional callback(completed, total) for progress.

        Returns:
            List of (result, usage_metadata, error) tuples in same order as input.
            - result: Parsed Pydantic model instance, or None if error
            - usage_metadata: Dict of usage info (currently empty for adapters), or None
            - error: Exception if the task failed, or None if success
        """
        from ..benchmark.verification.batch_runner import get_async_portal

        portal = get_async_portal()

        if portal is not None:
            # Use the shared BlockingPortal for proper event loop management
            return portal.call(self.ainvoke_batch, tasks, progress_callback)

        # No portal available - check if we're already in an async context
        try:
            asyncio.get_running_loop()
            # We're in an async context - use ThreadPoolExecutor to avoid
            # nested event loop issues
            logger.debug("AdapterParallelInvoker: Running in async context, using ThreadPoolExecutor")

            def run_in_thread() -> list[tuple[T | None, dict[str, Any] | None, Exception | None]]:
                return asyncio.run(self.ainvoke_batch(tasks, progress_callback))

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=600)  # 10 minute timeout

        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            logger.debug("AdapterParallelInvoker: No event loop, using asyncio.run()")
            return asyncio.run(self.ainvoke_batch(tasks, progress_callback))

    def invoke_batch_with_aggregated_usage(
        self,
        tasks: Sequence[tuple[str, type[T]]],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[list[tuple[T | None, Exception | None]], dict[str, Any]]:
        """Invoke parser for multiple tasks and aggregate usage metadata.

        Similar to invoke_batch but aggregates usage metadata across all calls
        into a single dictionary with totals.

        Args:
            tasks: Sequence of (prompt_text, response_model_class) tuples.
            progress_callback: Optional callback(completed, total) for progress.

        Returns:
            Tuple of (results_list, aggregated_usage) where:
            - results_list: List of (result, error) tuples in input order
            - aggregated_usage: Dict with aggregated token counts and call count
        """
        raw_results = self.invoke_batch(tasks, progress_callback)

        # Separate results and aggregate usage
        results: list[tuple[T | None, Exception | None]] = []
        aggregated_usage: dict[str, Any] = {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

        for result, usage, error in raw_results:
            results.append((result, error))

            if usage:
                aggregated_usage["calls"] += 1
                aggregated_usage["input_tokens"] += usage.get("input_tokens", 0)
                aggregated_usage["output_tokens"] += usage.get("output_tokens", 0)
                aggregated_usage["total_tokens"] += usage.get("total_tokens", 0)

        return results, aggregated_usage
