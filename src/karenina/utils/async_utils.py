"""Async utilities for parallelizing synchronous operations.

This module provides utilities for parallel execution.
For new code, use ThreadPoolExecutor directly (see generation_service.py or batch_runner.py for examples).
"""

import os
from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T")
R = TypeVar("R")


class AsyncConfig(BaseModel):
    """Simple async configuration for backward compatibility.

    DEPRECATED: Use direct boolean flags and max_workers instead.
    This class is maintained only for schema compatibility.
    """

    enabled: bool = True
    max_workers: int | None = None

    @classmethod
    def from_env(cls) -> "AsyncConfig":
        """Create configuration from environment variables."""
        enabled = os.getenv("KARENINA_ASYNC_ENABLED", "true").lower() == "true"
        max_workers = None
        if workers_env := os.getenv("KARENINA_ASYNC_MAX_WORKERS"):
            max_workers = int(workers_env)

        return cls(enabled=enabled, max_workers=max_workers)


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
