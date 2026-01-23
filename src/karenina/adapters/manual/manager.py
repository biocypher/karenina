"""Manual trace manager for precomputed answer traces.

This module provides the ManualTraceManager class for session-based, thread-safe
storage of precomputed answer traces. It also exports the global singleton instance
used throughout the application.
"""

import re
import threading
import time
from typing import Any

from . import ManualTraceError


class ManualTraceManager:
    """Manages precomputed answer traces for manual verification with memory management."""

    def __init__(self, session_timeout_seconds: int = 3600):
        """
        Initialize the trace manager.

        Args:
            session_timeout_seconds: Timeout for trace session in seconds (default: 1 hour)
        """
        # Session-based storage for manual traces
        self._traces: dict[str, str] = {}
        self._trace_timestamps: dict[str, float] = {}
        self._trace_metrics: dict[str, dict[str, Any] | None] = {}  # Agent metrics for each trace
        self._session_timeout = session_timeout_seconds
        self._cleanup_timer: threading.Timer | None = None
        self._last_access = time.time()

        # Thread safety
        self._lock = threading.RLock()

        # Start automatic cleanup
        self._start_cleanup_timer()

    def load_traces_from_json(self, json_data: dict[str, Any]) -> None:
        """
        Load manual traces from JSON data.

        Args:
            json_data: Dictionary with question hashes as keys and traces as values

        Raises:
            ManualTraceError: If validation fails
        """
        self._validate_trace_data(json_data)

        with self._lock:
            # Store traces after validation with timestamps
            current_time = time.time()
            for question_hash, trace in json_data.items():
                self._traces[question_hash] = str(trace)
                self._trace_timestamps[question_hash] = current_time
                self._trace_metrics[question_hash] = None  # No metrics for JSON-loaded traces

            self._last_access = current_time
            self._restart_cleanup_timer()

    def set_trace(self, question_hash: str, trace: str, agent_metrics: dict[str, Any] | None = None) -> None:
        """
        Set a manual trace programmatically with optional agent metrics.

        Args:
            question_hash: MD5 hash of the question
            trace: The precomputed trace string
            agent_metrics: Optional agent metrics dictionary (tool calls, failures, etc.)

        Raises:
            ManualTraceError: If question_hash format is invalid
        """
        if not self._is_valid_md5_hash(question_hash):
            raise ManualTraceError(
                f"Invalid question hash format: '{question_hash}'. "
                "Question hashes must be 32-character hexadecimal MD5 hashes."
            )

        with self._lock:
            current_time = time.time()
            self._traces[question_hash] = str(trace)
            self._trace_timestamps[question_hash] = current_time
            self._trace_metrics[question_hash] = agent_metrics
            self._last_access = current_time
            self._restart_cleanup_timer()

    def get_trace(self, question_hash: str) -> str | None:
        """
        Get a manual trace for a specific question hash.

        Args:
            question_hash: MD5 hash of the question

        Returns:
            The precomputed trace or None if not found
        """
        with self._lock:
            self._last_access = time.time()
            return self._traces.get(question_hash)

    def get_trace_with_metrics(self, question_hash: str) -> tuple[str | None, dict[str, Any] | None]:
        """
        Get a manual trace and its agent metrics for a specific question hash.

        Args:
            question_hash: MD5 hash of the question

        Returns:
            Tuple of (trace, agent_metrics) where either can be None
        """
        with self._lock:
            self._last_access = time.time()
            trace = self._traces.get(question_hash)
            metrics = self._trace_metrics.get(question_hash)
            return trace, metrics

    def has_trace(self, question_hash: str) -> bool:
        """
        Check if a trace exists for a question hash.

        Args:
            question_hash: MD5 hash of the question

        Returns:
            True if trace exists, False otherwise
        """
        with self._lock:
            self._last_access = time.time()
            return question_hash in self._traces

    def get_all_traces(self) -> dict[str, str]:
        """
        Get all loaded traces.

        Returns:
            Dictionary of all traces keyed by question hash
        """
        with self._lock:
            return self._traces.copy()

    def clear_traces(self) -> None:
        """Clear all loaded traces and stop cleanup timer."""
        with self._lock:
            self._traces.clear()
            self._trace_timestamps.clear()
            self._trace_metrics.clear()
            if self._cleanup_timer:
                self._cleanup_timer.cancel()
                self._cleanup_timer = None

    def get_trace_count(self) -> int:
        """Get the number of loaded traces."""
        with self._lock:
            return len(self._traces)

    def get_memory_usage_info(self) -> dict[str, Any]:
        """
        Get information about current memory usage.

        Returns:
            Dictionary with memory usage statistics
        """
        with self._lock:
            total_traces = len(self._traces)
            total_chars = sum(len(trace) for trace in self._traces.values())
            estimated_bytes = total_chars * 4  # Rough estimate for UTF-8

            return {
                "trace_count": total_traces,
                "total_characters": total_chars,
                "estimated_memory_bytes": estimated_bytes,
                "session_timeout_seconds": self._session_timeout,
                "last_access_timestamp": self._last_access,
                "seconds_since_last_access": time.time() - self._last_access,
            }

    def _start_cleanup_timer(self) -> None:
        """Start the automatic cleanup timer."""
        with self._lock:
            if self._cleanup_timer:
                self._cleanup_timer.cancel()

            self._cleanup_timer = threading.Timer(self._session_timeout, self._cleanup_expired_traces)
            self._cleanup_timer.daemon = True
            self._cleanup_timer.start()

    def _restart_cleanup_timer(self) -> None:
        """Restart the cleanup timer after activity."""
        self._start_cleanup_timer()

    def _cleanup_expired_traces(self) -> None:
        """Clean up expired traces based on session timeout."""
        with self._lock:
            current_time = time.time()

            # Check if the entire session has expired
            if current_time - self._last_access >= self._session_timeout:
                self._traces.clear()
                self._trace_timestamps.clear()
                self._trace_metrics.clear()
                if self._cleanup_timer:
                    self._cleanup_timer.cancel()
                    self._cleanup_timer = None
                return

            # Clean up individual expired traces
            expired_hashes = [
                hash_key
                for hash_key, timestamp in self._trace_timestamps.items()
                if current_time - timestamp >= self._session_timeout
            ]

            for hash_key in expired_hashes:
                self._traces.pop(hash_key, None)
                self._trace_timestamps.pop(hash_key, None)
                self._trace_metrics.pop(hash_key, None)

            # Restart timer if we still have traces
            if self._traces:
                self._start_cleanup_timer()

    def _validate_trace_data(self, json_data: dict[str, Any]) -> None:
        """
        Validate the structure and content of trace data.

        Args:
            json_data: The trace data to validate

        Raises:
            ManualTraceError: If validation fails
        """
        if not isinstance(json_data, dict):
            raise ManualTraceError(
                "Invalid trace data format: Expected a JSON object with question hash keys. "
                f"Received {type(json_data).__name__}. "
                "Please ensure your file contains a JSON object like: "
                '{"hash1": "trace1", "hash2": "trace2"}'
            )

        if not json_data:
            raise ManualTraceError(
                "Empty trace data: No traces found in the uploaded file. "
                "Please ensure your JSON file contains question hash to trace mappings like: "
                '{"d41d8cd98f00b204e9800998ecf8427e": "Your answer trace here"}'
            )

        # Validate each entry
        for key, value in json_data.items():
            # Validate question hash format (32 character hexadecimal)
            if not self._is_valid_md5_hash(key):
                raise ManualTraceError(
                    f"Invalid question hash format: '{key}'. "
                    "Question hashes must be 32-character hexadecimal MD5 hashes. "
                    "These are typically generated during question extraction. "
                    "Example of valid hash: 'd41d8cd98f00b204e9800998ecf8427e'. "
                    "Use the 'Download CSV mapper' feature to see valid hashes for your questions."
                )

            # Validate trace content
            if not isinstance(value, str) or not value.strip():
                raise ManualTraceError(
                    f"Invalid trace content for question hash '{key}'. "
                    f"Expected a non-empty string, but got {type(value).__name__}. "
                    "Trace content should be the precomputed answer text from your LLM. "
                    "Example: 'This is the answer to the question.'"
                )

    def _is_valid_md5_hash(self, hash_string: str) -> bool:
        """
        Check if a string is a valid MD5 hash.

        Args:
            hash_string: String to validate

        Returns:
            True if valid MD5 hash, False otherwise
        """
        if not isinstance(hash_string, str):
            return False

        # MD5 hash is exactly 32 hexadecimal characters
        md5_pattern = re.compile(r"^[a-fA-F0-9]{32}$")
        return bool(md5_pattern.match(hash_string))


# Global instance for session-based trace storage
_trace_manager = ManualTraceManager()


def get_trace_manager() -> ManualTraceManager:
    """Get the global trace manager instance."""
    return _trace_manager
