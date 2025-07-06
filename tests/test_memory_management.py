"""Test memory management features of manual trace manager."""

import time

import pytest

from karenina.llm.manual_traces import ManualTraceManager, get_memory_usage_info


def test_session_timeout_initialization():
    """Test that ManualTraceManager initializes with correct timeout."""
    # Test default timeout
    manager = ManualTraceManager()
    assert manager._session_timeout == 3600  # 1 hour default

    # Test custom timeout
    manager = ManualTraceManager(session_timeout_seconds=1800)
    assert manager._session_timeout == 1800  # 30 minutes


def test_memory_usage_info():
    """Test memory usage information collection."""
    manager = ManualTraceManager()

    # Initially empty
    info = manager.get_memory_usage_info()
    assert info["trace_count"] == 0
    assert info["total_characters"] == 0
    assert info["estimated_memory_bytes"] == 0
    assert info["session_timeout_seconds"] == 3600

    # Load some traces
    test_traces = {
        "d41d8cd98f00b204e9800998ecf8427e": "Short trace",
        "c4ca4238a0b923820dcc509a6f75849b": "A longer trace with more content",
    }

    manager.load_traces_from_json(test_traces)

    info = manager.get_memory_usage_info()
    assert info["trace_count"] == 2
    assert info["total_characters"] == len("Short trace") + len("A longer trace with more content")
    assert info["estimated_memory_bytes"] > 0
    assert info["seconds_since_last_access"] < 1  # Just accessed


def test_automatic_cleanup_timer():
    """Test that cleanup timer is properly managed."""
    manager = ManualTraceManager(session_timeout_seconds=1)  # 1 second timeout

    # Load traces
    test_traces = {"d41d8cd98f00b204e9800998ecf8427e": "Test trace"}
    manager.load_traces_from_json(test_traces)

    # Verify traces are loaded
    assert manager.get_trace_count() == 1
    assert manager._cleanup_timer is not None

    # Wait for cleanup (longer than timeout)
    time.sleep(1.5)

    # Traces should be cleaned up
    assert manager.get_trace_count() == 0


def test_manual_cleanup():
    """Test manual cleanup functionality."""
    manager = ManualTraceManager()

    # Load traces
    test_traces = {
        "d41d8cd98f00b204e9800998ecf8427e": "Test trace 1",
        "c4ca4238a0b923820dcc509a6f75849b": "Test trace 2",
    }
    manager.load_traces_from_json(test_traces)

    # Verify traces are loaded
    assert manager.get_trace_count() == 2
    assert manager._cleanup_timer is not None

    # Manual cleanup
    manager.clear_traces()

    # Verify cleanup
    assert manager.get_trace_count() == 0
    assert len(manager._trace_timestamps) == 0
    assert manager._cleanup_timer is None


def test_access_updates_last_access():
    """Test that accessing traces updates the last access timestamp."""
    manager = ManualTraceManager()

    test_traces = {"d41d8cd98f00b204e9800998ecf8427e": "Test trace"}
    manager.load_traces_from_json(test_traces)

    initial_access = manager._last_access
    time.sleep(0.1)  # Small delay

    # Access trace
    manager.get_trace("d41d8cd98f00b204e9800998ecf8427e")
    assert manager._last_access > initial_access

    # Check trace existence
    manager.has_trace("d41d8cd98f00b204e9800998ecf8427e")
    assert manager._last_access > initial_access


def test_global_memory_usage_function():
    """Test the global memory usage function."""
    from karenina.llm.manual_traces import clear_manual_traces, load_manual_traces

    # Clear any existing traces
    clear_manual_traces()

    # Load traces
    test_traces = {"d41d8cd98f00b204e9800998ecf8427e": "Global test trace"}
    load_manual_traces(test_traces)

    # Check memory usage
    info = get_memory_usage_info()
    assert info["trace_count"] == 1
    assert info["total_characters"] == len("Global test trace")

    # Cleanup
    clear_manual_traces()


def test_timer_restart_on_activity():
    """Test that timer restarts when new traces are loaded."""
    manager = ManualTraceManager(session_timeout_seconds=10)

    # Load initial traces
    test_traces = {"d41d8cd98f00b204e9800998ecf8427e": "Test trace 1"}
    manager.load_traces_from_json(test_traces)

    initial_timer = manager._cleanup_timer
    time.sleep(0.1)

    # Load more traces - should restart timer
    more_traces = {"c4ca4238a0b923820dcc509a6f75849b": "Test trace 2"}
    manager.load_traces_from_json(more_traces)

    # Timer should be different (restarted)
    assert manager._cleanup_timer is not initial_timer

    # Cleanup
    manager.clear_traces()


if __name__ == "__main__":
    pytest.main([__file__])
