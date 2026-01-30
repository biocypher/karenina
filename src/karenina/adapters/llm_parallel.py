"""Parallel invocation utility for LLMPort adapters.

This module provides a reusable utility for executing multiple LLM invocations
in parallel using asyncio.gather() with semaphore-based concurrency limiting.

Two modes are supported:
- Plain text mode: `invoke_batch()` for answer generation using LLMPort.ainvoke()
- Structured output mode: `invoke_batch_structured()` for rubric evaluation

Key characteristics:
- Uses asyncio.gather() for true async parallelism
- Preserves result ordering (critical requirement)
- Per-task error isolation (failed tasks don't block others)
- Integrates with existing BlockingPortal from batch_runner.py

Environment Variables:
- KARENINA_ASYNC_ENABLED: Enable/disable parallel execution (default: true)
- KARENINA_ASYNC_MAX_WORKERS: Max concurrent workers (default: 2)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

from ._parallel_base import get_max_workers, read_async_config, sync_invoke_via_portal

if TYPE_CHECKING:
    from ..ports import LLMPort, LLMResponse, Message

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Re-export read_async_config for backward compatibility
__all__ = ["LLMParallelInvoker", "read_async_config"]


class LLMParallelInvoker:
    """Execute multiple LLMPort invocations in parallel.

    This utility is designed for scenarios where multiple independent LLM calls
    need to be made. By running these calls in parallel, we achieve significant
    speedup.

    Provides two separate methods for clean type separation:

    1. `invoke_batch()` - Plain text mode (answer generation):
       - Task: list[Message]
       - Uses LLMPort.ainvoke() directly
       - Returns: list[tuple[LLMResponse | None, Exception | None]]

    2. `invoke_batch_structured()` - Structured output mode (rubric evaluation):
       - Task: tuple[list[Message], type[T]]
       - Uses LLMPort.with_structured_output().ainvoke()
       - Returns: list[tuple[T | None, dict | None, Exception | None]]

    Example usage (plain text):
        invoker = LLMParallelInvoker(llm_adapter, max_workers=4)
        tasks = [
            [Message.system("..."), Message.user("Question 1")],
            [Message.system("..."), Message.user("Question 2")],
        ]
        results = invoker.invoke_batch(tasks)
        for response, error in results:
            if error:
                print(f"Task failed: {error}")
            else:
                print(f"Response: {response.content}")

    Example usage (structured output):
        invoker = LLMParallelInvoker(llm_adapter, max_workers=4)
        tasks = [
            ([Message.system("..."), Message.user("...")], ResponseModel),
            ([Message.system("..."), Message.user("...")], ResponseModel),
        ]
        results = invoker.invoke_batch_structured(tasks)
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
        return get_max_workers(self._max_workers)

    # =========================================================================
    # Plain Text Mode (answer generation)
    # =========================================================================

    async def ainvoke_batch(
        self,
        tasks: Sequence[list[Message]],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[tuple[LLMResponse | None, Exception | None]]:
        """Invoke LLM for multiple message lists in parallel (async, plain text).

        Each task is a list of messages. The LLM is invoked for each task,
        and results are collected in order.

        Args:
            tasks: Sequence of message lists to send to the LLM.
            progress_callback: Optional callback(completed, total) for progress.

        Returns:
            List of (response, error) tuples in same order as input.
            - response: LLMResponse with content and usage metadata, or None if error
            - error: Exception if the task failed, or None if success
        """
        if not tasks:
            return []

        total = len(tasks)
        progress_lock = asyncio.Lock()
        completed_count = 0
        semaphore = asyncio.Semaphore(self.max_workers)

        async def execute_task(
            index: int,
        ) -> tuple[int, LLMResponse | None, Exception | None]:
            """Execute a single task and return (index, response, error)."""
            nonlocal completed_count

            messages = tasks[index]

            async with semaphore:
                try:
                    response = await self.llm.ainvoke(messages)
                    return index, response, None
                except Exception as e:
                    logger.debug(f"LLMParallelInvoker: Task {index} failed: {e}")
                    return index, None, e
                finally:
                    if progress_callback:
                        async with progress_lock:
                            completed_count += 1
                            progress_callback(completed_count, total)

        task_coroutines = [execute_task(i) for i in range(total)]
        raw_results = await asyncio.gather(*task_coroutines, return_exceptions=False)

        # Build ordered results list
        results: list[tuple[LLMResponse | None, Exception | None]] = [(None, None)] * total
        for index, response, error in raw_results:
            results[index] = (response, error)

        return results

    def invoke_batch(
        self,
        tasks: Sequence[list[Message]],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[tuple[LLMResponse | None, Exception | None]]:
        """Invoke LLM for multiple message lists in parallel (sync, plain text).

        This is a sync wrapper around ainvoke_batch(). Uses the shared BlockingPortal
        if available, otherwise falls back to asyncio.run().

        Args:
            tasks: Sequence of message lists to send to the LLM.
            progress_callback: Optional callback(completed, total) for progress.

        Returns:
            List of (response, error) tuples in same order as input.
            - response: LLMResponse with content and usage metadata, or None if error
            - error: Exception if the task failed, or None if success
        """
        return sync_invoke_via_portal(self.ainvoke_batch, tasks, progress_callback)

    # =========================================================================
    # Structured Output Mode (rubric evaluation)
    # =========================================================================

    async def ainvoke_batch_structured(
        self,
        tasks: Sequence[tuple[list[Message], type[T]]],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[tuple[T | None, dict[str, Any] | None, Exception | None]]:
        """Invoke LLM for multiple message/schema pairs in parallel (async, structured).

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

    def invoke_batch_structured(
        self,
        tasks: Sequence[tuple[list[Message], type[T]]],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[tuple[T | None, dict[str, Any] | None, Exception | None]]:
        """Invoke LLM for multiple message/schema pairs in parallel (sync, structured).

        This is a sync wrapper around ainvoke_batch_structured(). Uses the shared
        BlockingPortal if available, otherwise falls back to asyncio.run().

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
        return sync_invoke_via_portal(self.ainvoke_batch_structured, tasks, progress_callback)
