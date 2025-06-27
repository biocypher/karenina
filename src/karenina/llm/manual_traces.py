"""Manual trace management for precomputed answer traces."""

import re
from typing import Any


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class ManualTraceError(LLMError):
    """Raised when there's an error with manual trace operations."""
    pass


class ManualTraceManager:
    """Manages precomputed answer traces for manual verification."""

    def __init__(self):
        """Initialize the trace manager."""
        # Session-based storage for manual traces
        self._traces: dict[str, str] = {}

    def load_traces_from_json(self, json_data: dict[str, Any]) -> None:
        """
        Load manual traces from JSON data.
        
        Args:
            json_data: Dictionary with question hashes as keys and traces as values
            
        Raises:
            ManualTraceError: If validation fails
        """
        self._validate_trace_data(json_data)

        # Store traces after validation
        for question_hash, trace in json_data.items():
            self._traces[question_hash] = str(trace)

    def get_trace(self, question_hash: str) -> str | None:
        """
        Get a manual trace for a specific question hash.
        
        Args:
            question_hash: MD5 hash of the question
            
        Returns:
            The precomputed trace or None if not found
        """
        return self._traces.get(question_hash)

    def has_trace(self, question_hash: str) -> bool:
        """
        Check if a trace exists for a question hash.
        
        Args:
            question_hash: MD5 hash of the question
            
        Returns:
            True if trace exists, False otherwise
        """
        return question_hash in self._traces

    def get_all_traces(self) -> dict[str, str]:
        """
        Get all loaded traces.
        
        Returns:
            Dictionary of all traces keyed by question hash
        """
        return self._traces.copy()

    def clear_traces(self) -> None:
        """Clear all loaded traces."""
        self._traces.clear()

    def get_trace_count(self) -> int:
        """Get the number of loaded traces."""
        return len(self._traces)

    def _validate_trace_data(self, json_data: dict[str, Any]) -> None:
        """
        Validate the structure and content of trace data.
        
        Args:
            json_data: The trace data to validate
            
        Raises:
            ManualTraceError: If validation fails
        """
        if not isinstance(json_data, dict):
            raise ManualTraceError("Trace data must be a JSON object")

        if not json_data:
            raise ManualTraceError("Trace data cannot be empty")

        # Validate each entry
        for key, value in json_data.items():
            # Validate question hash format (32 character hexadecimal)
            if not self._is_valid_md5_hash(key):
                raise ManualTraceError(
                    f"Invalid question hash format: '{key}'. "
                    "Must be a 32-character hexadecimal MD5 hash."
                )

            # Validate trace content
            if not isinstance(value, str) or not value.strip():
                raise ManualTraceError(
                    f"Invalid trace content for question hash '{key}'. "
                    "Trace must be a non-empty string."
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
        md5_pattern = re.compile(r'^[a-fA-F0-9]{32}$')
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
