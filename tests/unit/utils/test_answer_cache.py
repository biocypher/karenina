"""Unit tests for answer cache module.

Tests for thread-safe answer caching used in verification pipeline.
"""

import threading
import time

from karenina.utils.answer_cache import AnswerTraceCache, CacheEntry


class TestCacheEntry:
    """Tests for CacheEntry class."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        entry = CacheEntry()
        assert entry.is_complete is False
        assert entry.answer_data is None
        assert entry.error is None
        assert entry.timestamp is not None
        assert isinstance(entry.event, threading.Event)

    def test_init_with_complete(self) -> None:
        """Test initialization with is_complete=True."""
        entry = CacheEntry(is_complete=True)
        assert entry.is_complete is True


class TestAnswerTraceCacheInit:
    """Tests for AnswerTraceCache initialization."""

    def test_init_creates_empty_cache(self) -> None:
        """Test that new cache has empty storage and zero stats."""
        cache = AnswerTraceCache()
        assert cache._cache == {}
        assert cache._stats_hits == 0
        assert cache._stats_misses == 0
        assert cache._stats_waits == 0
        assert cache._stats_timeouts == 0

    def test_two_caches_are_independent(self) -> None:
        """Test that separate cache instances are independent."""
        cache1 = AnswerTraceCache()
        cache2 = AnswerTraceCache()
        cache1._cache["key1"] = CacheEntry(is_complete=True)
        assert "key1" not in cache2._cache


class TestGetOrReserve:
    """Tests for get_or_reserve method."""

    def test_get_or_reserve_miss_on_empty_cache(self) -> None:
        """Test MISS status when cache is empty."""
        cache = AnswerTraceCache()
        status, data = cache.get_or_reserve("key1")
        assert status == "MISS"
        assert data is None
        assert cache._stats_misses == 1

    def test_get_or_reserve_miss_on_new_key(self) -> None:
        """Test MISS status for a new key."""
        cache = AnswerTraceCache()
        cache._cache["existing"] = CacheEntry(is_complete=True)
        status, data = cache.get_or_reserve("new_key")
        assert status == "MISS"
        assert data is None
        assert cache._stats_misses == 1

    def test_get_or_reserve_creates_entry_on_miss(self) -> None:
        """Test that MISS creates a cache entry."""
        cache = AnswerTraceCache()
        cache.get_or_reserve("key1")
        assert "key1" in cache._cache
        assert cache._cache["key1"].is_complete is False

    def test_get_or_reserve_hit_on_completed(self) -> None:
        """Test HIT status when entry is complete."""
        cache = AnswerTraceCache()
        cache._cache["key1"] = CacheEntry(is_complete=True)
        cache._cache["key1"].answer_data = {"result": "success"}
        status, data = cache.get_or_reserve("key1")
        assert status == "HIT"
        assert data == {"result": "success"}
        assert cache._stats_hits == 1

    def test_get_or_reserve_in_progress(self) -> None:
        """Test IN_PROGRESS status when entry is not complete."""
        cache = AnswerTraceCache()
        cache._cache["key1"] = CacheEntry(is_complete=False)
        status, data = cache.get_or_reserve("key1")
        assert status == "IN_PROGRESS"
        assert data is None
        assert cache._stats_waits == 1

    def test_get_or_reserve_failed_entry_allows_retry(self) -> None:
        """Test that failed entries are treated as MISS to allow retry."""
        cache = AnswerTraceCache()
        entry = CacheEntry(is_complete=True)
        entry.error = Exception("Generation failed")
        cache._cache["key1"] = entry

        status, data = cache.get_or_reserve("key1")
        assert status == "MISS"  # Should allow retry
        assert data is None
        # Entry should have been replaced with a new incomplete entry
        assert cache._cache["key1"].error is None

    def test_get_or_reserve_multiple_calls_increment_stats(self) -> None:
        """Test that multiple calls increment statistics correctly."""
        cache = AnswerTraceCache()

        # First call - MISS
        cache.get_or_reserve("key1")
        assert cache._stats_misses == 1

        # Second call while in progress - IN_PROGRESS
        cache.get_or_reserve("key1")
        assert cache._stats_waits == 1

        # Complete and try again - HIT
        cache._cache["key1"].is_complete = True
        cache._cache["key1"].answer_data = {"value": 1}
        cache.get_or_reserve("key1")
        assert cache._stats_hits == 1


class TestComplete:
    """Tests for complete method."""

    def test_complete_sets_answer_data(self) -> None:
        """Test that complete stores answer data."""
        cache = AnswerTraceCache()
        cache.get_or_reserve("key1")
        answer_data = {"trace": "answer text", "parsed": {"value": 42}}
        cache.complete("key1", answer_data)

        entry = cache._cache["key1"]
        assert entry.is_complete is True
        assert entry.answer_data == answer_data
        assert entry.error is None

    def test_complete_with_error(self) -> None:
        """Test that complete can store error state."""
        cache = AnswerTraceCache()
        cache.get_or_reserve("key1")
        error = ValueError("Generation failed")
        cache.complete("key1", None, error=error)

        entry = cache._cache["key1"]
        assert entry.is_complete is True
        assert entry.answer_data is None
        assert entry.error is error

    def test_complete_nonexistent_key_logs_warning(self, caplog) -> None:
        """Test that completing non-existent key logs warning."""
        import logging

        cache = AnswerTraceCache()
        with caplog.at_level(logging.WARNING):
            cache.complete("nonexistent", {"data": "value"})
        assert "non-existent cache entry" in caplog.text

    def test_complete_signals_event(self) -> None:
        """Test that complete signals the event."""
        cache = AnswerTraceCache()
        cache.get_or_reserve("key1")
        event = cache._cache["key1"].event
        assert not event.is_set()
        cache.complete("key1", {"data": "value"})
        assert event.is_set()


class TestWaitForCompletion:
    """Tests for wait_for_completion method."""

    def test_wait_for_completed_entry_returns_immediately(self) -> None:
        """Test that waiting on completed entry returns immediately."""
        cache = AnswerTraceCache()
        cache._cache["key1"] = CacheEntry(is_complete=True)
        assert cache.wait_for_completion("key1") is True

    def test_wait_for_nonexistent_key_returns_true(self) -> None:
        """Test that waiting on non-existent key returns True."""
        cache = AnswerTraceCache()
        assert cache.wait_for_completion("nonexistent") is True

    def test_wait_for_in_progress_waits_for_completion(self) -> None:
        """Test that waiting on in-progress entry waits for completion."""
        cache = AnswerTraceCache()
        cache.get_or_reserve("key1")

        # Start a thread that waits and completes after a delay
        result = {"waited": False}

        def wait_then_complete():
            result["waited"] = cache.wait_for_completion("key1", timeout=1.0)
            assert result["waited"] is True

        thread = threading.Thread(target=wait_then_complete)
        thread.start()

        # Complete after a short delay
        time.sleep(0.1)
        cache.complete("key1", {"data": "value"})

        thread.join(timeout=2.0)
        assert result["waited"] is True

    def test_wait_for_in_progress_times_out(self) -> None:
        """Test that waiting on in-progress entry times out."""
        cache = AnswerTraceCache()
        cache.get_or_reserve("key1")
        # Don't complete it
        result = cache.wait_for_completion("key1", timeout=0.1)
        assert result is False
        assert cache._stats_timeouts == 1

    def test_wait_timeout_increments_stats(self) -> None:
        """Test that timeout increments stats counter."""
        cache = AnswerTraceCache()
        cache.get_or_reserve("key1")
        cache.wait_for_completion("key1", timeout=0.01)
        assert cache._stats_timeouts == 1


class TestGetStats:
    """Tests for get_stats method."""

    def test_get_stats_returns_all_counters(self) -> None:
        """Test that get_stats returns all statistics."""
        cache = AnswerTraceCache()
        stats = cache.get_stats()
        assert stats == {
            "hits": 0,
            "misses": 0,
            "waits": 0,
            "timeouts": 0,
        }

    def test_get_stats_reflects_operations(self) -> None:
        """Test that stats reflect cache operations."""
        cache = AnswerTraceCache()

        # Generate some activity
        cache.get_or_reserve("key1")  # MISS
        cache.get_or_reserve("key1")  # IN_PROGRESS
        cache._cache["key1"].is_complete = True
        cache._cache["key1"].answer_data = {"data": "value"}
        cache.get_or_reserve("key1")  # HIT

        cache.wait_for_completion("key2", timeout=0)  # Nonexistent, no stats change

        stats = cache.get_stats()
        assert stats["misses"] == 1
        assert stats["waits"] == 1
        assert stats["hits"] == 1


class TestConcurrency:
    """Tests for concurrent access."""

    def test_concurrent_get_or_reserve(self) -> None:
        """Test that multiple threads can call get_or_reserve safely."""
        cache = AnswerTraceCache()
        results = []

        def worker(thread_id: int):
            for _ in range(10):
                status, _ = cache.get_or_reserve(f"key{thread_id}")
                results.append((thread_id, status))

        threads = [
            threading.Thread(target=worker, args=(i,))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Each thread should have 1 MISS and 9 IN_PROGRESS results
        thread_results = [r for r in results if r[0] == 0]
        miss_count = sum(1 for _, s in thread_results if s == "MISS")
        progress_count = sum(1 for _, s in thread_results if s == "IN_PROGRESS")
        assert miss_count == 1
        assert progress_count == 9

    def test_concurrent_complete_and_wait(self) -> None:
        """Test that complete and wait work correctly with threads."""
        cache = AnswerTraceCache()
        cache.get_or_reserve("shared_key")
        results = {"waiter1": False, "waiter2": False}

        def completer():
            time.sleep(0.1)
            cache.complete("shared_key", {"result": "success"})

        def waiter(name: str):
            cache.wait_for_completion("shared_key", timeout=1.0)
            results[name] = True

        threads = [
            threading.Thread(target=completer),
            threading.Thread(target=waiter, args=("waiter1",)),
            threading.Thread(target=waiter, args=("waiter2",)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results["waiter1"] is True
        assert results["waiter2"] is True


class TestCacheEntryTimestamp:
    """Tests for CacheEntry timestamp."""

    def test_timestamp_is_set_on_creation(self) -> None:
        """Test that timestamp is set when entry is created."""
        before = time.time()
        entry = CacheEntry()
        after = time.time()
        assert before <= entry.timestamp <= after

    def test_timestamp_different_for_each_entry(self) -> None:
        """Test that each entry gets a unique timestamp."""
        entry1 = CacheEntry()
        time.sleep(0.01)  # Small delay
        entry2 = CacheEntry()
        assert entry1.timestamp < entry2.timestamp


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_complete_with_none_data_and_no_error(self) -> None:
        """Test completing with None data but no error (empty success)."""
        cache = AnswerTraceCache()
        cache.get_or_reserve("key1")
        cache.complete("key1", None)
        entry = cache._cache["key1"]
        assert entry.is_complete is True
        assert entry.answer_data is None
        assert entry.error is None

    def test_multiple_keys_independent(self) -> None:
        """Test that multiple keys are handled independently."""
        cache = AnswerTraceCache()
        cache.get_or_reserve("key1")
        cache.get_or_reserve("key2")
        cache.get_or_reserve("key3")

        # Complete each with different data
        cache.complete("key1", {"id": 1})
        cache.complete("key2", {"id": 2})
        cache.complete("key3", {"id": 3})

        assert cache._cache["key1"].answer_data == {"id": 1}
        assert cache._cache["key2"].answer_data == {"id": 2}
        assert cache._cache["key3"].answer_data == {"id": 3}

    def test_same_key_overwrites_previous_entry(self) -> None:
        """Test that failed entry is replaced on retry."""
        cache = AnswerTraceCache()
        cache._cache["key1"] = CacheEntry(is_complete=True)
        cache._cache["key1"].error = Exception("Old error")
        cache._cache["key1"].answer_data = {"old": "data"}
        cache._cache["key1"].timestamp = 123.0

        # Getting a failed entry should create a new one
        status, _ = cache.get_or_reserve("key1")
        assert status == "MISS"
        assert cache._cache["key1"].error is None
        assert cache._cache["key1"].answer_data is None
        assert cache._cache["key1"].timestamp != 123.0
