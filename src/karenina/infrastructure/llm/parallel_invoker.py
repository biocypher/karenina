"""Parallel LLM invocation utility for concurrent trait evaluation.

This module provides a reusable utility for executing multiple LLM invocations
in parallel using ThreadPoolExecutor. It reuses patterns from batch_runner.py:
- ThreadPoolExecutor for parallel execution
- Thread-safe result collection
- Per-task error handling (failed tasks don't block others)
- Optional progress callback

Environment Variables:
- KARENINA_ASYNC_ENABLED: Enable/disable parallel execution (default: true)
- KARENINA_ASYNC_MAX_WORKERS: Max concurrent workers (default: 2)
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class ParallelLLMInvoker:
    """
    Execute multiple LLM invocations in parallel using ThreadPoolExecutor.

    This utility is designed for scenarios where multiple independent LLM calls
    need to be made (e.g., evaluating traits one-by-one in "sequential" mode).
    By running these calls in parallel, we achieve significant speedup.

    Reuses patterns from batch_runner.py:
    - ThreadPoolExecutor for parallel execution
    - Thread-safe result collection
    - Per-task error handling (failed tasks don't block others)

    Example usage:
        invoker = ParallelLLMInvoker(llm, max_workers=4)
        tasks = [
            (messages1, ResponseModel),
            (messages2, ResponseModel),
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
        llm: BaseChatModel,
        max_workers: int | None = None,
    ):
        """
        Initialize the parallel LLM invoker.

        Args:
            llm: LangChain chat model instance for LLM calls.
            max_workers: Maximum number of parallel workers. If None, defaults to
                        KARENINA_ASYNC_MAX_WORKERS env var or 2.
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

    def invoke_batch(
        self,
        tasks: Sequence[tuple[list[BaseMessage], type[T]]],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[tuple[T | None, dict[str, Any] | None, Exception | None]]:
        """
        Invoke LLM for multiple message/schema pairs in parallel.

        Each task is a tuple of (messages, response_model_class). The LLM is
        invoked with structured output for each task, and results are collected.

        Args:
            tasks: Sequence of (messages, response_model_class) tuples.
                   messages: List of BaseMessage for the LLM call.
                   response_model_class: Pydantic model for structured output.
            progress_callback: Optional callback(completed, total) for progress.

        Returns:
            List of (result, usage_metadata, error) tuples in same order as input.
            - result: Parsed Pydantic model instance, or None if error
            - usage_metadata: Dict of usage info, or None if error
            - error: Exception if the task failed, or None if success
        """
        from ...benchmark.verification.evaluators.rubric_parsing import (
            invoke_with_structured_output,
        )

        if not tasks:
            return []

        total = len(tasks)
        results: list[tuple[T | None, dict[str, Any] | None, Exception | None]] = [(None, None, None)] * total
        completed_count = 0

        def execute_task(index: int) -> tuple[int, T | None, dict[str, Any] | None, Exception | None]:
            """Execute a single task and return (index, result, usage, error)."""
            nonlocal completed_count

            messages, model_class = tasks[index]
            try:
                result, usage_metadata = invoke_with_structured_output(self.llm, messages, model_class)
                return index, result, usage_metadata, None
            except Exception as e:
                logger.debug(f"Task {index} failed: {e}")
                return index, None, None, e

        # Execute tasks in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(execute_task, i): i for i in range(total)}

            for future in as_completed(futures):
                index, result, usage, error = future.result()
                results[index] = (result, usage, error)
                completed_count += 1

                if progress_callback:
                    progress_callback(completed_count, total)

        return results

    def invoke_batch_with_aggregated_usage(
        self,
        tasks: Sequence[tuple[list[BaseMessage], type[T]]],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[list[tuple[T | None, Exception | None]], dict[str, Any]]:
        """
        Invoke LLM for multiple tasks and aggregate usage metadata.

        Similar to invoke_batch but aggregates usage metadata across all calls
        into a single dictionary with totals.

        Args:
            tasks: Sequence of (messages, response_model_class) tuples.
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


def read_async_config() -> tuple[bool, int]:
    """
    Read async configuration from environment variables.

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
