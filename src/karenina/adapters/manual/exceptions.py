"""Exceptions for manual trace operations.

This module defines exceptions used by the manual adapter module.
Kept separate to avoid circular imports between __init__.py and manager.py.
"""

from karenina.ports import PortError


class ManualTraceError(PortError):
    """Raised when there's an error with manual trace operations.

    This includes:
    - Invalid question hash format
    - Validation errors when loading traces
    - Preprocessing errors for message lists

    Attributes:
        message: Description of what went wrong.

    Example:
        >>> try:
        ...     set_manual_trace("invalid", "trace")
        ... except ManualTraceError as e:
        ...     print(f"Trace error: {e}")
    """

    pass


class ManualInterfaceError(PortError):
    """Raised when attempting LLM operations with manual interface.

    Manual interface is for pre-recorded traces only. If this error is raised,
    it means code is incorrectly trying to invoke an LLM when it should be
    using pre-recorded trace data instead.

    This is a safety net - properly written call sites should check
    `interface != "manual"` before attempting adapter operations.

    Attributes:
        operation: The operation that was attempted (e.g., "agent.run_sync()").

    Example:
        >>> try:
        ...     agent.run_sync(messages)  # Manual interface
        ... except ManualInterfaceError as e:
        ...     print(f"Cannot {e.operation}: manual interface uses pre-recorded traces")
    """

    def __init__(self, operation: str) -> None:
        self.operation = operation
        super().__init__(
            f"Cannot {operation} with manual interface. "
            f"Manual interface uses pre-recorded traces, not live LLM calls. "
            f"Check that interface != 'manual' before calling adapter methods, "
            f"or use model_config.manual_traces for pre-recorded data."
        )


class ManualTraceNotFoundError(PortError):
    """Raised when a manual trace is not found for a question hash.

    This error indicates that the ManualTraceManager does not contain a
    pre-recorded trace for the requested question. This typically happens when:
    - Traces were not loaded before running verification
    - The question hash doesn't match any loaded traces
    - The trace file has different question hashes than expected

    Attributes:
        question_hash: The hash that was not found.
        loaded_count: Number of traces currently loaded.

    Example:
        >>> try:
        ...     result = agent.run_sync(
        ...         messages=[...],
        ...         config=AgentConfig(question_hash="abc123...")
        ...     )
        ... except ManualTraceNotFoundError as e:
        ...     print(f"Missing trace for hash: {e.question_hash}")
    """

    def __init__(self, question_hash: str, loaded_count: int) -> None:
        self.question_hash = question_hash
        self.loaded_count = loaded_count
        super().__init__(
            f"No manual trace found for hash: '{question_hash}'. "
            f"Loaded {loaded_count} trace(s). "
            "Ensure traces are loaded via ManualTraceManager before verification."
        )
