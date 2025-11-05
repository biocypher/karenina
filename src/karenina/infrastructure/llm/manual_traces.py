"""Manual trace management for precomputed answer traces."""

import re
import threading
import time
from typing import Any


class LLMError(Exception):
    """Base exception for LLM-related errors."""

    pass


class ManualTraceError(LLMError):
    """Raised when there's an error with manual trace operations."""

    pass


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


class ManualTraces:
    """
    Manages manual traces for a specific benchmark.

    This class provides a high-level API for registering traces programmatically,
    with support for both string traces and LangChain message lists (with tool call metrics).
    """

    def __init__(self, benchmark: "Any") -> None:
        """
        Initialize ManualTraces with a benchmark.

        Args:
            benchmark: The Benchmark object containing questions

        Note:
            This enables question text to hash mapping via the benchmark's question cache.
        """
        self._benchmark = benchmark
        self._trace_manager = get_manual_trace_manager()

    def register_trace(self, question_identifier: str, trace: str | list[Any], map_to_id: bool = False) -> None:
        """
        Register a single trace by question ID or text.

        Args:
            question_identifier: Either a question hash (32-char MD5) or question text
            trace: Either a string trace or list of LangChain messages
            map_to_id: If True, treat question_identifier as text and convert to hash

        Raises:
            ValueError: If question not found in benchmark (when map_to_id=True)
            ManualTraceError: If question_hash format is invalid
            TypeError: If trace format is invalid
        """
        # Convert text to hash if needed
        question_hash = self._question_text_to_hash(question_identifier) if map_to_id else question_identifier

        # Validate hash format
        if not self._trace_manager._is_valid_md5_hash(question_hash):
            raise ManualTraceError(
                f"Invalid question hash: '{question_hash}'. "
                "Question hashes must be 32-character hexadecimal MD5 hashes."
            )

        # Preprocess trace (handle both string and message list formats)
        harmonized_trace, agent_metrics = self._preprocess_trace(trace)

        # Store in trace manager
        self._trace_manager.set_trace(question_hash=question_hash, trace=harmonized_trace, agent_metrics=agent_metrics)

    def register_traces(self, traces_dict: dict[str, str | list[Any]], map_to_id: bool = False) -> None:
        """
        Batch register traces.

        Args:
            traces_dict: Dictionary with question identifiers as keys and traces as values
            map_to_id: If True, treat keys as question text and convert to hashes

        Raises:
            ValueError: If any question not found in benchmark (when map_to_id=True)
            ManualTraceError: If any question_hash format is invalid
            TypeError: If any trace format is invalid
        """
        for identifier, trace in traces_dict.items():
            self.register_trace(identifier, trace, map_to_id=map_to_id)

    def _question_text_to_hash(self, question_text: str) -> str:
        """
        Convert question text to MD5 hash using the same algorithm as Question.id.

        Args:
            question_text: The question text

        Returns:
            MD5 hash of the question text

        Raises:
            ValueError: If question not found in benchmark
        """
        import hashlib

        computed_hash = hashlib.md5(question_text.encode("utf-8")).hexdigest()

        # Search through benchmark's questions to find matching question
        # Note: _questions_cache uses URN format IDs as keys, but we need the MD5 hash
        for question_urn_id in self._benchmark._questions_cache:
            question_data = self._benchmark.get_question(question_urn_id)
            if question_data["question"] == question_text:
                # Found the question, return the MD5 hash (not the URN ID)
                return computed_hash

        # Question not found
        available_count = len(self._benchmark._questions_cache)
        raise ValueError(
            f"Question not found in benchmark: '{question_text[:50]}...'\n"
            f"Computed hash: {computed_hash}\n"
            f"Available questions in benchmark: {available_count}\n"
            "Note: Question text must match EXACTLY (case-sensitive, including whitespace)."
        )

    def _preprocess_trace(self, trace: str | list[Any]) -> tuple[str, dict[str, Any] | None]:
        """
        Process trace and extract agent metrics if applicable.

        For string traces: Returns as-is with no metrics.
        For LangChain message lists: Extracts tool calls, failures, and harmonizes to string.

        Args:
            trace: Either a string trace or list of LangChain messages

        Returns:
            Tuple of (harmonized_trace_string, agent_metrics_dict_or_None)

        Raises:
            TypeError: If trace format is invalid
        """
        if isinstance(trace, str):
            # Plain string trace - no metrics
            return trace, None

        elif isinstance(trace, list):
            # LangChain message format - extract metrics and harmonize
            try:
                from karenina.benchmark.verification.verification_utils import _extract_agent_metrics
                from karenina.infrastructure.llm.mcp_utils import harmonize_agent_response

                # Build response object expected by extraction function
                response = {"messages": trace}

                # Extract metrics (tool calls, failures, etc.)
                agent_metrics = _extract_agent_metrics(response)

                # Convert to string trace
                harmonized_trace = harmonize_agent_response(response)

                return harmonized_trace, agent_metrics

            except ImportError as e:
                raise ManualTraceError(
                    f"Failed to import required preprocessing functions: {e}\n"
                    "This may indicate a dependency issue in the verification module."
                ) from e

            except Exception as e:
                raise ManualTraceError(
                    f"Failed to preprocess LangChain message list: {e}\n"
                    "Ensure the message list contains valid LangChain message objects."
                ) from e

        else:
            raise TypeError(f"Invalid trace format: expected str or list[BaseMessage], got {type(trace).__name__}")
