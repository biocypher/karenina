"""Test thread safety features of manual trace manager."""

import threading
import time

import pytest

from karenina.llm.manual_traces import ManualTraceManager


def test_concurrent_access():
    """Test concurrent access to trace manager from multiple threads."""
    manager = ManualTraceManager(session_timeout_seconds=10)
    results = []
    errors = []

    def load_traces(thread_id: int):
        """Load traces from a thread."""
        try:
            test_traces = {f"d41d8cd98f00b204e9800998ecf8427{thread_id % 10}": f"Trace from thread {thread_id}"}
            manager.load_traces_from_json(test_traces)
            results.append(f"Thread {thread_id} loaded traces")
        except Exception as e:
            errors.append(f"Thread {thread_id} error: {e}")

    def access_traces(thread_id: int):
        """Access traces from a thread."""
        try:
            # Try to access various traces
            for i in range(5):
                hash_key = f"d41d8cd98f00b204e9800998ecf8427{i}"
                trace = manager.get_trace(hash_key)
                has_trace = manager.has_trace(hash_key)
                results.append(f"Thread {thread_id} accessed trace {i}: {trace is not None}, {has_trace}")
        except Exception as e:
            errors.append(f"Thread {thread_id} access error: {e}")

    # Create and start multiple threads
    threads = []
    for i in range(5):
        # Mix of loaders and accessors
        if i % 2 == 0:
            thread = threading.Thread(target=load_traces, args=(i,))
        else:
            thread = threading.Thread(target=access_traces, args=(i,))
        threads.append(thread)

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Check results
    assert not errors, f"Thread errors occurred: {errors}"
    assert len(results) > 0, "No results from any thread"

    # Clean up
    manager.clear_traces()


def test_concurrent_read_write():
    """Test concurrent read and write operations."""
    manager = ManualTraceManager(session_timeout_seconds=10)

    # Load initial traces
    initial_traces = {"d41d8cd98f00b204e9800998ecf8427e": "Initial trace"}
    manager.load_traces_from_json(initial_traces)

    read_results = []
    write_results = []
    errors = []

    def reader_thread():
        """Continuously read traces."""
        try:
            for _i in range(10):
                trace = manager.get_trace("d41d8cd98f00b204e9800998ecf8427e")
                count = manager.get_trace_count()
                read_results.append((trace is not None, count))
                time.sleep(0.01)  # Small delay
        except Exception as e:
            errors.append(f"Reader error: {e}")

    def writer_thread(thread_id: int):
        """Write new traces."""
        try:
            # Use valid 32-character MD5 hashes
            valid_hashes = [
                [
                    "c4ca4238a0b923820dcc509a6f75849b",
                    "c81e728d9d4c2f636f067f89cc14862c",
                    "eccbc87e4b5ce2fe28308fd9f2a7baf3",
                ],
                [
                    "a87ff679a2f3e71d9181a67b7542122c",
                    "e4da3b7fbbce2345d7772b0674a318d5",
                    "1679091c5a880faf6fb5e6087eb1b2dc",
                ],
                [
                    "8f14e45fceea167a5a36dedd4bea2543",
                    "c9f0f895fb98ab9159f51fd0297e236d",
                    "45c48cce2e2d7fbdea1afc51c7c6ad26",
                ],
            ][thread_id % 3]

            for i in range(3):
                hash_key = valid_hashes[i]
                new_traces = {hash_key: f"New trace {thread_id}-{i}"}
                manager.load_traces_from_json(new_traces)
                write_results.append(f"Writer {thread_id} iteration {i}")
                time.sleep(0.02)  # Small delay
        except Exception as e:
            errors.append(f"Writer {thread_id} error: {e}")

    # Start reader and writer threads
    threads = []

    # Start reader
    reader = threading.Thread(target=reader_thread)
    threads.append(reader)

    # Start multiple writers
    for i in range(3):
        writer = threading.Thread(target=writer_thread, args=(i,))
        threads.append(writer)

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for all threads
    for thread in threads:
        thread.join()

    # Check results
    assert not errors, f"Thread errors occurred: {errors}"
    assert len(read_results) > 0, "No read results"
    assert len(write_results) > 0, "No write results"

    # All reads should have succeeded
    for _trace_found, count in read_results:
        assert isinstance(count, int) and count >= 0

    # Clean up
    manager.clear_traces()


def test_concurrent_memory_usage():
    """Test concurrent access to memory usage information."""
    manager = ManualTraceManager(session_timeout_seconds=10)

    results = []
    errors = []

    def check_memory_usage(thread_id: int):
        """Check memory usage from a thread."""
        try:
            for _i in range(5):
                info = manager.get_memory_usage_info()
                assert isinstance(info["trace_count"], int)
                assert isinstance(info["total_characters"], int)
                assert isinstance(info["estimated_memory_bytes"], int)
                results.append(f"Thread {thread_id} checked memory usage")
                time.sleep(0.01)
        except Exception as e:
            errors.append(f"Thread {thread_id} error: {e}")

    def load_traces_concurrent(thread_id: int):
        """Load traces concurrently."""
        try:
            # Use valid MD5 hashes
            valid_hashes = [
                "d41d8cd98f00b204e9800998ecf8427e",
                "c4ca4238a0b923820dcc509a6f75849b",
                "c81e728d9d4c2f636f067f89cc14862c",
                "eccbc87e4b5ce2fe28308fd9f2a7baf3",
                "a87ff679a2f3e71d9181a67b7542122c",
                "e4da3b7fbbce2345d7772b0674a318d5",
            ]
            test_traces = {
                valid_hashes[thread_id % len(valid_hashes)]: f"Long trace content from thread {thread_id} " * 10
            }
            manager.load_traces_from_json(test_traces)
            results.append(f"Thread {thread_id} loaded traces")
        except Exception as e:
            errors.append(f"Thread {thread_id} error: {e}")

    # Create mixed threads
    threads = []
    for i in range(6):
        if i % 2 == 0:
            thread = threading.Thread(target=check_memory_usage, args=(i,))
        else:
            thread = threading.Thread(target=load_traces_concurrent, args=(i,))
        threads.append(thread)

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for completion
    for thread in threads:
        thread.join()

    # Check results
    assert not errors, f"Thread errors occurred: {errors}"
    assert len(results) > 0, "No results from threads"

    # Clean up
    manager.clear_traces()


def test_concurrent_cleanup():
    """Test that cleanup doesn't interfere with concurrent operations."""
    manager = ManualTraceManager(session_timeout_seconds=1)  # Very short timeout

    results = []
    errors = []

    def continuous_operations():
        """Perform continuous operations while cleanup might happen."""
        try:
            # Use valid MD5 hashes
            valid_hashes = [
                "d41d8cd98f00b204e9800998ecf8427e",
                "c4ca4238a0b923820dcc509a6f75849b",
                "c81e728d9d4c2f636f067f89cc14862c",
            ]
            for i in range(20):
                # Load traces
                test_traces = {valid_hashes[i % 3]: f"Trace {i}"}
                manager.load_traces_from_json(test_traces)

                # Access traces
                manager.get_trace_count()
                manager.has_trace("d41d8cd98f00b204e9800998ecf8427e")

                results.append(f"Operation {i} completed")
                time.sleep(0.05)  # Small delay to allow cleanup
        except Exception as e:
            errors.append(f"Operation error: {e}")

    # Start operation thread
    op_thread = threading.Thread(target=continuous_operations)
    op_thread.start()

    # Let cleanup happen naturally
    time.sleep(1.5)  # Wait longer than timeout

    # Wait for operations to complete
    op_thread.join()

    # Check results - some operations should have succeeded
    # Cleanup might clear traces, but operations should still work
    assert not errors, f"Operation errors occurred: {errors}"
    assert len(results) > 0, "No operations completed"

    # Clean up
    manager.clear_traces()


def test_lock_type():
    """Test that we're using the correct lock type (RLock for reentrant access)."""
    manager = ManualTraceManager()

    # Verify lock type - check the class name since RLock type checking is complex
    assert manager._lock.__class__.__name__ == "RLock", "Should use RLock for reentrant access"

    # Test reentrant access (calling method that calls another method with lock)
    test_traces = {"d41d8cd98f00b204e9800998ecf8427e": "Test trace"}

    # This should work without deadlock
    manager.load_traces_from_json(test_traces)
    assert manager.get_trace_count() == 1

    # Clean up
    manager.clear_traces()


if __name__ == "__main__":
    pytest.main([__file__])
