"""Module-level helper functions for manual trace operations.

These functions provide convenient access to the global ManualTraceManager singleton.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .manager import ManualTraceManager, _trace_manager, get_trace_manager

if TYPE_CHECKING:
    from pathlib import Path

    from .traces import ManualTraces


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


def load_manual_traces_from_file(trace_file: Path, benchmark: Any) -> ManualTraces:
    """
    Load manual traces from JSON file and create ManualTraces object.

    Args:
        trace_file: Path to JSON file with traces (question_hash -> trace_string mapping)
        benchmark: Benchmark for question mapping

    Returns:
        ManualTraces object populated with traces

    Raises:
        FileNotFoundError: If trace file doesn't exist
        json.JSONDecodeError: If JSON is invalid
        ValueError: If trace data is not a dict
        ManualTraceError: If trace validation fails

    Example JSON format:
        {
            "936dbc8755f623c951d96ea2b03e13bc": "Answer for question 1",
            "8f2e2b1e4d5c6a7b8c9d0e1f2a3b4c5d": "Answer for question 2"
        }
    """
    import json
    from pathlib import Path

    from .traces import ManualTraces

    trace_file = Path(trace_file)

    if not trace_file.exists():
        raise FileNotFoundError(f"Manual traces file not found: {trace_file}")

    # Load JSON
    with open(trace_file) as f:
        traces_data = json.load(f)

    # Validate it's a dictionary
    if not isinstance(traces_data, dict):
        raise ValueError(f"Invalid trace file format: expected JSON object (dict), got {type(traces_data).__name__}")

    # Load into global manager (validation happens here)
    load_manual_traces(traces_data)

    # Create ManualTraces object linked to benchmark
    manual_traces = ManualTraces(benchmark)

    return manual_traces


__all__ = [
    "load_manual_traces",
    "load_manual_traces_from_file",
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
