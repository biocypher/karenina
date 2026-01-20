"""Parallel invocation utility for LLMPort adapters with structured output.

This module provides a reusable utility for executing multiple LLM invocations
in parallel using asyncio.gather() with semaphore-based concurrency limiting.

Key characteristics:
- Uses asyncio.gather() for true async parallelism
- Preserves result ordering (critical requirement)
- Per-task error isolation (failed tasks don't block others)
- Integrates with existing BlockingPortal from batch_runner.py
- Same max_workers configuration pattern as AdapterParallelInvoker

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
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from ..ports import LLMPort, Message

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def read_async_config() -> tuple[bool, int]:
    """Read async configuration from environment variables.

    This function implements the standard pattern for reading KARENINA_ASYNC_*
    environment variables, used by QuestionClassifier and LLMTraitEvaluator.

    Returns:
        Tuple of (async_enabled, max_workers) with defaults applied.
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


class LLMParallelInvoker:
    """Execute multiple LLMPort invocations with structured output in parallel.

    This utility is designed for scenarios where multiple independent LLM calls
    with structured output need to be made (e.g., evaluating traits one-by-one
    in "sequential" mode). By running these calls in parallel, we achieve
    significant speedup.

    Similar to AdapterParallelInvoker but works with LLMPort.with_structured_output():
    - asyncio.gather() with semaphore for concurrency control
    - Thread-safe result collection
    - Per-task error handling (failed tasks don't block others)

    Example usage:
        invoker = LLMParallelInvoker(llm_adapter, max_workers=4)
        tasks = [
            ([Message.system("..."), Message.user("...")], ResponseModel),
            ([Message.system("..."), Message.user("...")], ResponseModel),
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
        llm: LLMPort,
        max_workers: int | None = None,
    ):
        """Initialize the LLM parallel invoker.

        Args:
            llm: LLMPort implementation for LLM operations.
            max_workers: Maximum number of concurrent LLM calls. If None,
                        defaults to KARENINA_ASYNC_MAX_WORKERS env var or 2.
        """
        self.llm = llm
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
        tasks: Sequence[tuple[list[Message], type[T]]],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[tuple[T | None, dict[str, Any] | None, Exception | None]]:
        """Invoke LLM for multiple message/schema pairs in parallel (async).

        Each task is a tuple of (messages, response_model_class). The LLM is
        invoked with structured output for each task, and results are collected.

        Args:
            tasks: Sequence of (messages, response_model_class) tuples.
                   messages: List of Message objects to send.
                   response_model_class: Pydantic model for structured output.
            progress_callback: Optional callback(completed, total) for progress.

        Returns:
            List of (result, usage_metadata, error) tuples in same order as input.
            - result: Parsed Pydantic model instance, or None if error
            - usage_metadata: Dict of usage info from LLMResponse.usage, or None
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

            messages, model_class = tasks[index]

            async with semaphore:
                try:
                    # Create structured output adapter and invoke
                    structured_llm = self.llm.with_structured_output(model_class)
                    response = await structured_llm.ainvoke(messages)

                    # Extract the parsed model from response.raw
                    result = response.raw
                    if not isinstance(result, model_class):
                        # Fallback: try to validate from response content
                        result = model_class.model_validate_json(response.content)

                    # Convert UsageMetadata to dict
                    usage_metadata = asdict(response.usage) if response.usage else {}

                    return index, result, usage_metadata, None
                except Exception as e:
                    logger.debug(f"LLMParallelInvoker: Task {index} failed: {e}")
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
        tasks: Sequence[tuple[list[Message], type[T]]],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[tuple[T | None, dict[str, Any] | None, Exception | None]]:
        """Invoke LLM for multiple message/schema pairs in parallel (sync).

        This is a sync wrapper around ainvoke_batch(). Uses the shared BlockingPortal
        if available, otherwise falls back to asyncio.run() with proper event loop
        handling.

        Args:
            tasks: Sequence of (messages, response_model_class) tuples.
                   messages: List of Message objects to send.
                   response_model_class: Pydantic model for structured output.
            progress_callback: Optional callback(completed, total) for progress.

        Returns:
            List of (result, usage_metadata, error) tuples in same order as input.
            - result: Parsed Pydantic model instance, or None if error
            - usage_metadata: Dict of usage info from LLMResponse.usage, or None
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
            logger.debug("LLMParallelInvoker: Running in async context, using ThreadPoolExecutor")

            def run_in_thread() -> list[tuple[T | None, dict[str, Any] | None, Exception | None]]:
                return asyncio.run(self.ainvoke_batch(tasks, progress_callback))

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=600)  # 10 minute timeout

        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            logger.debug("LLMParallelInvoker: No event loop, using asyncio.run()")
            return asyncio.run(self.ainvoke_batch(tasks, progress_callback))
