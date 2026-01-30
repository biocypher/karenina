"""Module-level helper functions for manual trace operations.

These functions provide convenient access to the global ManualTraceManager singleton.
"""

from typing import Any

from .manager import ManualTraceManager, _trace_manager, get_trace_manager


def load_manual_traces(json_data: dict[str, Any]) -> None:
    """
    Load manual traces from JSON data into the global manager.

    Args:
        json_data: Dictionary with question hashes as keys and traces as values

    Raises:
        ManualTraceError: If validation fails
    """
    _trace_manager.load_traces_from_json(json_data)


def get_manual_trace(question_hash: str) -> str | None:
    """
    Get a manual trace for a specific question hash.

    Args:
        question_hash: MD5 hash of the question

    Returns:
        The precomputed trace or None if not found
    """
    return _trace_manager.get_trace(question_hash)


def has_manual_trace(question_hash: str) -> bool:
    """
    Check if a manual trace exists for a question hash.

    Args:
        question_hash: MD5 hash of the question

    Returns:
        True if trace exists, False otherwise
    """
    return _trace_manager.has_trace(question_hash)


def clear_manual_traces() -> None:
    """Clear all loaded manual traces."""
    _trace_manager.clear_traces()


def get_manual_trace_count() -> int:
    """Get the number of loaded manual traces."""
    return _trace_manager.get_trace_count()


def get_memory_usage_info() -> dict[str, Any]:
    """Get memory usage information for manual traces."""
    return _trace_manager.get_memory_usage_info()


def set_manual_trace(question_hash: str, trace: str, agent_metrics: dict[str, Any] | None = None) -> None:
    """
    Set a manual trace programmatically with optional agent metrics.

    Args:
        question_hash: MD5 hash of the question
        trace: The precomputed trace string
        agent_metrics: Optional agent metrics dictionary (tool calls, failures, etc.)

    Raises:
        ManualTraceError: If question_hash format is invalid
    """
    _trace_manager.set_trace(question_hash, trace, agent_metrics)


def get_manual_trace_with_metrics(question_hash: str) -> tuple[str | None, dict[str, Any] | None]:
    """
    Get a manual trace and its agent metrics for a specific question hash.

    Args:
        question_hash: MD5 hash of the question

    Returns:
        Tuple of (trace, agent_metrics) where either can be None
    """
    return _trace_manager.get_trace_with_metrics(question_hash)


def get_manual_trace_manager() -> ManualTraceManager:
    """
    Get the global trace manager instance.

    This is useful for programmatic access to the trace manager,
    particularly for the ManualTraces class.
    """
    return _trace_manager


__all__ = [
    "load_manual_traces",
    "get_manual_trace",
    "has_manual_trace",
    "clear_manual_traces",
    "get_manual_trace_count",
    "get_memory_usage_info",
    "set_manual_trace",
    "get_manual_trace_with_metrics",
    "get_manual_trace_manager",
    "get_trace_manager",
]
