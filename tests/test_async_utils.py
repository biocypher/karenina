"""Tests for async utilities module."""

import time

import pytest

from karenina.utils.async_utils import run_sync_with_progress


def slow_function(item: str) -> str:
    """A slow synchronous function for testing."""
    time.sleep(0.01)  # Small delay to simulate work
    return item.upper()


def error_function(item: str) -> str:
    """A function that raises an error for testing error handling."""
    if item == "error":
        raise ValueError("Test error")
    return item.upper()


class TestRunSyncWithProgress:
    """Test the synchronous function with progress tracking."""

    def test_sync_with_progress(self) -> None:
        """Test synchronous execution with progress callback."""
        items = ["hello", "world", "test"]
        progress_calls = []

        def progress_callback(percentage: float, message: str) -> None:
            progress_calls.append((percentage, message))

        results = run_sync_with_progress(items, slow_function, progress_callback)

        # Check results
        assert results == ["HELLO", "WORLD", "TEST"]

        # Check progress callbacks
        assert len(progress_calls) == 3
        assert progress_calls[0][0] == pytest.approx(33.33, abs=0.01)
        assert progress_calls[1][0] == pytest.approx(66.67, abs=0.01)
        assert progress_calls[2][0] == 100.0
        assert "1/3" in progress_calls[0][1]
        assert "2/3" in progress_calls[1][1]
        assert "3/3" in progress_calls[2][1]

    def test_sync_empty_list(self) -> None:
        """Test synchronous execution with empty list."""
        results = run_sync_with_progress([], slow_function)
        assert results == []

    def test_sync_with_errors(self) -> None:
        """Test synchronous execution with errors."""
        items = ["hello", "error", "world"]
        results = run_sync_with_progress(items, error_function)

        assert results[0] == "HELLO"
        assert isinstance(results[1], ValueError)
        assert results[2] == "WORLD"

    def test_progress_callback_none(self) -> None:
        """Test that None progress callback doesn't cause errors."""
        items = ["hello", "world"]
        results = run_sync_with_progress(items, slow_function, None)
        assert results == ["HELLO", "WORLD"]


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_all_errors(self) -> None:
        """Test handling when all items result in errors."""
        items = ["error", "error", "error"]
        results = run_sync_with_progress(items, error_function)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, ValueError)


if __name__ == "__main__":
    pytest.main([__file__])
