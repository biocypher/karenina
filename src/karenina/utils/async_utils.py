"""Async utilities for parallelizing synchronous operations.

This module provides utilities for sequential execution with progress tracking.
For parallel execution, use ThreadPoolExecutor directly (see generation_service.py or batch_runner.py for examples).
"""

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def run_sync_with_progress(
    items: list[T],
    sync_function: Callable[[T], R],
    progress_callback: Callable[[float, str], None] | None = None,
) -> list[R]:
    """
    Run a synchronous function sequentially with progress tracking.

    This provides sequential execution with progress callbacks.
    For parallel execution, use ThreadPoolExecutor directly (see generation_service.py or batch_runner.py).

    Args:
        items: List of items to process
        sync_function: Synchronous function to apply to each item
        progress_callback: Optional callback for progress updates (percentage, message)

    Returns:
        List of results maintaining the same order as input items.
        Exceptions are returned as Exception objects in the results list.

    Example:
        ```python
        def process_item(item: str) -> str:
            return item.upper()

        items = ["hello", "world"]
        results = run_sync_with_progress(items, process_item)
        # Results: ["HELLO", "WORLD"]
        ```
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
