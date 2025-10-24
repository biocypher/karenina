"""Async utilities for parallelizing synchronous operations.

This module provides thin wrappers around synchronous functions to enable
parallel execution while maintaining a single source of truth for business logic.
"""

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T")
R = TypeVar("R")


async def run_async_chunked(
    items: list[T],
    sync_function: Callable[[T], R],
    chunk_size: int = 5,
    progress_callback: Callable[[float, str], None] | None = None,
    max_workers: int | None = None,
    on_task_start: Callable[[T], None] | None = None,
    on_task_done: Callable[[T, R | Exception], None] | None = None,
) -> list[R]:
    """
    Run a synchronous function on a list of items in parallel chunks.

    This is a thin wrapper that enables async execution of sync functions
    without modifying the core business logic. Items are processed in chunks
    to avoid overwhelming the system or hitting rate limits.

    Args:
        items: List of items to process
        sync_function: Synchronous function to apply to each item
        chunk_size: Number of items to process in parallel (default: 5)
        progress_callback: Optional callback for progress updates (percentage, message)
        max_workers: Maximum number of worker threads (default: min(32, len(items)))
        on_task_start: Optional callback when a task starts processing
        on_task_done: Optional callback when a task completes (with result or exception)

    Returns:
        List of results maintaining the same order as input items

    Example:
        ```python
        def process_item(item: str) -> str:
            return item.upper()

        items = ["hello", "world", "async"]
        results = await run_async_chunked(items, process_item, chunk_size=2)
        # Results: ["HELLO", "WORLD", "ASYNC"]
        ```
    """
    if not items:
        return []

    if max_workers is None:
        max_workers = min(32, len(items))

    results: list[R | Exception] = []
    for _ in range(len(items)):
        results.append(None)  # type: ignore[arg-type]
    total_items = len(items)
    processed_count = 0

    # Process items in chunks
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit items in chunks to avoid overwhelming the executor
        for chunk_start in range(0, total_items, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_items)
            chunk_items = items[chunk_start:chunk_end]

            # Submit chunk to executor
            future_to_index = {}
            future_to_item = {}
            for i, item in enumerate(chunk_items):
                original_index = chunk_start + i

                # Call on_task_start callback
                if on_task_start:
                    on_task_start(item)

                future = executor.submit(sync_function, item)
                future_to_index[future] = original_index
                future_to_item[future] = item

            # Collect results as they complete
            for future in as_completed(future_to_index):
                original_index = future_to_index[future]
                item = future_to_item[future]
                try:
                    result = future.result()
                    results[original_index] = result
                    processed_count += 1

                    # Call on_task_done callback
                    if on_task_done:
                        on_task_done(item, result)

                    # Update progress
                    if progress_callback:
                        percentage = (processed_count / total_items) * 100
                        message = f"Processed {processed_count}/{total_items} items"
                        progress_callback(percentage, message)

                except Exception as e:
                    # Store exception as result to be handled by caller
                    results[original_index] = e
                    processed_count += 1

                    # Call on_task_done callback with exception
                    if on_task_done:
                        on_task_done(item, e)

            # Small delay between chunks to be respectful to APIs
            if chunk_end < total_items:
                await asyncio.sleep(0.1)

    return results  # type: ignore[return-value]


async def run_async_batched(
    items: list[T],
    sync_function: Callable[[T], R],
    batch_size: int = 10,
    concurrent_batches: int = 2,
    progress_callback: Callable[[float, str], None] | None = None,
    delay_between_batches: float = 0.5,
    on_task_start: Callable[[T], None] | None = None,
    on_task_done: Callable[[T, R | Exception], None] | None = None,
) -> list[R]:
    """
    Run a synchronous function on batches of items with controlled concurrency.

    This is useful when you need finer control over the execution pattern,
    such as processing items in sequential batches with limited parallelism
    within each batch.

    Args:
        items: List of items to process
        sync_function: Synchronous function to apply to each item
        batch_size: Number of items per batch
        concurrent_batches: Number of batches to process concurrently
        progress_callback: Optional callback for progress updates
        delay_between_batches: Delay in seconds between batch submissions

    Returns:
        List of results maintaining the same order as input items
    """
    if not items:
        return []

    results: list[R | Exception] = []
    for _ in range(len(items)):
        results.append(None)  # type: ignore[arg-type]
    total_items = len(items)
    processed_count = 0

    # Create batches
    batches = []
    for i in range(0, total_items, batch_size):
        batch_items = [(idx, items[idx]) for idx in range(i, min(i + batch_size, total_items))]
        batches.append(batch_items)

    async def process_batch(batch: list[tuple[int, T]]) -> list[tuple[int, R]]:
        """Process a single batch using the chunked approach."""
        batch_items = [item for _, item in batch]
        batch_results = await run_async_chunked(
            items=batch_items,
            sync_function=sync_function,
            chunk_size=len(batch_items),  # Process entire batch concurrently
            on_task_start=on_task_start,
            on_task_done=on_task_done,
        )
        return [(idx, result) for (idx, _), result in zip(batch, batch_results, strict=True)]

    # Process batches with controlled concurrency
    batch_semaphore = asyncio.Semaphore(concurrent_batches)

    async def process_batch_with_semaphore(batch: list[tuple[int, T]]) -> list[tuple[int, R]]:
        async with batch_semaphore:
            return await process_batch(batch)

    # Submit all batches
    batch_tasks = []
    for i, batch in enumerate(batches):
        # Add delay between batch submissions
        if i > 0:
            await asyncio.sleep(delay_between_batches)

        batch_tasks.append(process_batch_with_semaphore(batch))

    # Collect results
    for task in asyncio.as_completed(batch_tasks):
        batch_results = await task
        for idx, result in batch_results:
            results[idx] = result
            processed_count += 1

            if progress_callback:
                percentage = (processed_count / total_items) * 100
                message = f"Processed {processed_count}/{total_items} items (batched)"
                progress_callback(percentage, message)

    return results  # type: ignore[return-value]


def run_sync_with_progress(
    items: list[T],
    sync_function: Callable[[T], R],
    progress_callback: Callable[[float, str], None] | None = None,
) -> list[R]:
    """
    Run a synchronous function sequentially with progress tracking.

    This provides the synchronous equivalent of the async functions above,
    maintaining the same interface for easy switching between sync and async modes.

    Args:
        items: List of items to process
        sync_function: Synchronous function to apply to each item
        progress_callback: Optional callback for progress updates

    Returns:
        List of results maintaining the same order as input items
    """
    if not items:
        return []

    results = []
    total_items = len(items)

    for i, item in enumerate(items):
        try:
            result = sync_function(item)
            results.append(result)

            # Update progress
            if progress_callback:
                percentage = ((i + 1) / total_items) * 100
                message = f"Processed {i + 1}/{total_items} items (sync)"
                progress_callback(percentage, message)

        except Exception as e:
            results.append(e)  # type: ignore[arg-type]

    return results


class AsyncConfig(BaseModel):
    """Configuration for async processing behavior."""

    enabled: bool = True
    chunk_size: int = 5
    max_workers: int | None = None
    batch_size: int | None = None
    concurrent_batches: int | None = None
    delay_between_batches: float = 0.5

    @classmethod
    def from_env(cls) -> "AsyncConfig":
        """Create configuration from environment variables."""
        import os

        enabled = os.getenv("KARENINA_ASYNC_ENABLED", "true").lower() == "true"
        chunk_size = int(os.getenv("KARENINA_ASYNC_CHUNK_SIZE", "5"))
        max_workers = None
        if workers_env := os.getenv("KARENINA_ASYNC_MAX_WORKERS"):
            max_workers = int(workers_env)

        batch_size = None
        if batch_env := os.getenv("KARENINA_ASYNC_BATCH_SIZE"):
            batch_size = int(batch_env)

        concurrent_batches = None
        if concurrent_env := os.getenv("KARENINA_ASYNC_CONCURRENT_BATCHES"):
            concurrent_batches = int(concurrent_env)

        delay = float(os.getenv("KARENINA_ASYNC_DELAY_BETWEEN_BATCHES", "0.5"))

        return cls(
            enabled=enabled,
            chunk_size=chunk_size,
            max_workers=max_workers,
            batch_size=batch_size,
            concurrent_batches=concurrent_batches,
            delay_between_batches=delay,
        )


async def execute_with_config(
    items: list[T],
    sync_function: Callable[[T], R],
    config: AsyncConfig,
    progress_callback: Callable[[float, str], None] | None = None,
    on_task_start: Callable[[T], None] | None = None,
    on_task_done: Callable[[T, R | Exception], None] | None = None,
) -> list[R]:
    """
    Execute function with the given configuration (sync or async).

    This is the main entry point that switches between sync and async
    execution based on configuration.

    Args:
        items: List of items to process
        sync_function: Synchronous function to apply to each item
        config: Async configuration
        progress_callback: Optional callback for progress updates
        on_task_start: Optional callback when a task starts processing
        on_task_done: Optional callback when a task completes

    Returns:
        List of results maintaining the same order as input items
    """
    if not config.enabled:
        # Synchronous execution
        return run_sync_with_progress(items, sync_function, progress_callback)

    # Choose async execution strategy
    if config.batch_size and config.concurrent_batches:
        # Batched async execution
        return await run_async_batched(
            items=items,
            sync_function=sync_function,
            batch_size=config.batch_size,
            concurrent_batches=config.concurrent_batches,
            progress_callback=progress_callback,
            delay_between_batches=config.delay_between_batches,
            on_task_start=on_task_start,
            on_task_done=on_task_done,
        )
    else:
        # Chunked async execution (default)
        return await run_async_chunked(
            items=items,
            sync_function=sync_function,
            chunk_size=config.chunk_size,
            progress_callback=progress_callback,
            max_workers=config.max_workers,
            on_task_start=on_task_start,
            on_task_done=on_task_done,
        )


# Convenience functions for common patterns
def create_task_from_params(**kwargs: Any) -> dict[str, Any]:
    """Create a task dictionary from keyword arguments."""
    return kwargs


def unpack_task_and_call(sync_function: Callable[..., Any], task: dict[str, Any]) -> Any:
    """Unpack a task dictionary and call the sync function with its parameters."""
    return sync_function(**task)
