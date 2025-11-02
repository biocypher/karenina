"""Tests for answer trace caching optimization.

This module tests the thread-safe answer caching that prevents duplicate
answer generation when multiple judges evaluate the same answering model output.
"""

import threading
import time
from unittest.mock import patch

from karenina.benchmark.verification.batch_runner import (
    _extract_answer_data_from_result,
    _generate_answer_cache_key,
    execute_task,
)
from karenina.schemas.workflow import ModelConfig, VerificationResult
from karenina.utils.answer_cache import AnswerTraceCache, CacheEntry


class TestCacheEntry:
    """Test CacheEntry class."""

    def test_initial_state(self):
        """Test cache entry initial state."""
        entry = CacheEntry(is_complete=False)
        assert entry.is_complete is False
        assert entry.answer_data is None
        assert entry.error is None
        assert isinstance(entry.timestamp, float)

    def test_completed_state(self):
        """Test cache entry completed state."""
        entry = CacheEntry(is_complete=True)
        assert entry.is_complete is True


class TestAnswerTraceCache:
    """Test AnswerTraceCache functionality."""

    def test_cache_miss_reserves_slot(self):
        """Test that cache miss reserves slot for generation."""
        cache = AnswerTraceCache()
        key = "test_question_model1"

        status, answer_data = cache.get_or_reserve(key)

        assert status == "MISS"
        assert answer_data is None
        assert key in cache._cache
        assert cache._cache[key].is_complete is False

    def test_cache_hit_returns_data(self):
        """Test that cache hit returns cached data."""
        cache = AnswerTraceCache()
        key = "test_question_model1"

        # First call reserves slot
        cache.get_or_reserve(key)

        # Complete with data
        test_data = {"raw_llm_response": "Test answer"}
        cache.complete(key, test_data, error=None)

        # Second call should hit cache
        status, answer_data = cache.get_or_reserve(key)

        assert status == "HIT"
        assert answer_data == test_data

    def test_cache_handles_failure(self):
        """Test that cache handles failed answer generation."""
        cache = AnswerTraceCache()
        key = "test_question_model1"

        # First call reserves slot
        cache.get_or_reserve(key)

        # Complete with error
        error = Exception("LLM failed")
        cache.complete(key, None, error=error)

        # Second call should allow retry (treated as MISS)
        status, answer_data = cache.get_or_reserve(key)

        assert status == "MISS"
        assert answer_data is None

    def test_in_progress_returns_immediately(self):
        """Test that IN_PROGRESS status is returned immediately without blocking."""
        cache = AnswerTraceCache()
        key = "test_question_model1"

        # First task reserves slot
        status1, _ = cache.get_or_reserve(key)
        assert status1 == "MISS"

        # Second task should get IN_PROGRESS immediately (no blocking)
        start_time = time.time()
        status2, answer_data = cache.get_or_reserve(key)
        elapsed = time.time() - start_time

        assert status2 == "IN_PROGRESS"
        assert answer_data is None
        assert elapsed < 0.1  # Should return immediately

        # After completion, third task should hit cache
        test_data = {"raw_llm_response": "Test answer"}
        cache.complete(key, test_data, error=None)

        status3, answer_data = cache.get_or_reserve(key)
        assert status3 == "HIT"
        assert answer_data == test_data

    def test_in_progress_statistics(self):
        """Test that IN_PROGRESS encounters are tracked in statistics."""
        cache = AnswerTraceCache()
        key = "test_question_model1"

        # First call - MISS
        cache.get_or_reserve(key)
        stats = cache.get_stats()
        assert stats["misses"] == 1
        assert stats["waits"] == 0

        # Second call while in progress - IN_PROGRESS
        cache.get_or_reserve(key)
        stats = cache.get_stats()
        assert stats["waits"] == 1  # Tracks IN_PROGRESS returns

        # Complete the entry
        cache.complete(key, {"data": "test"}, error=None)

        # Third call - HIT
        cache.get_or_reserve(key)
        stats = cache.get_stats()
        assert stats["hits"] == 1

    def test_cache_statistics(self):
        """Test cache statistics tracking."""
        cache = AnswerTraceCache()

        # Cache miss
        cache.get_or_reserve("key1")
        stats = cache.get_stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 0

        # Complete and hit
        cache.complete("key1", {"data": "test"}, error=None)
        cache.get_or_reserve("key1")
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_thread_safety_multiple_callers(self):
        """Test that multiple tasks can safely check the cache concurrently."""
        cache = AnswerTraceCache()
        key = "test_question_model1"
        test_data = {"raw_llm_response": "Test answer"}

        # First task reserves slot
        status, _ = cache.get_or_reserve(key)
        assert status == "MISS"

        results = []

        def concurrent_check():
            status, answer_data = cache.get_or_reserve(key)
            results.append((status, answer_data))

        # Start multiple concurrent checks
        threads = [threading.Thread(target=concurrent_check) for _ in range(5)]
        for t in threads:
            t.start()

        # All should get IN_PROGRESS immediately
        for t in threads:
            t.join()

        assert len(results) == 5
        for status, answer_data in results:
            assert status == "IN_PROGRESS"
            assert answer_data is None

        # After completion, all new checks should HIT
        cache.complete(key, test_data, error=None)

        results_after = []

        def concurrent_check_after():
            status, answer_data = cache.get_or_reserve(key)
            results_after.append((status, answer_data))

        threads_after = [threading.Thread(target=concurrent_check_after) for _ in range(5)]
        for t in threads_after:
            t.start()
        for t in threads_after:
            t.join()

        assert len(results_after) == 5
        for status, answer_data in results_after:
            assert status == "HIT"
            assert answer_data == test_data


class TestCacheKeyGeneration:
    """Test cache key generation."""

    def test_cache_key_without_replicate(self):
        """Test cache key generation without replicate."""
        model = ModelConfig(
            id="model1",
            model_provider="openai",
            model_name="gpt-4.1-mini",
            temperature=0.0,
            system_prompt="Test",
        )

        task = {
            "question_id": "q1",
            "answering_model": model,
            "replicate": None,
        }

        key = _generate_answer_cache_key(task)
        assert key == "q1_model1"

    def test_cache_key_with_replicate(self):
        """Test cache key generation with replicate."""
        model = ModelConfig(
            id="model1",
            model_provider="openai",
            model_name="gpt-4.1-mini",
            temperature=0.0,
            system_prompt="Test",
        )

        task = {
            "question_id": "q1",
            "answering_model": model,
            "replicate": 2,
        }

        key = _generate_answer_cache_key(task)
        assert key == "q1_model1_rep2"


class TestAnswerDataExtraction:
    """Test answer data extraction from results."""

    def test_extract_answer_data(self):
        """Test extracting answer data from verification result."""
        result = VerificationResult(
            question_id="q1",
            template_id="t1",
            completed_without_errors=True,
            question_text="Test question",
            raw_llm_response="Test answer",
            answering_model="openai/gpt-4.1-mini",
            parsing_model="openai/gpt-4.1-mini",
            execution_time=1.0,
            timestamp="2025-01-01T00:00:00",
            recursion_limit_reached=False,
            answering_mcp_servers=["server1"],
            usage_metadata={"answer_generation": {"input_tokens": 100}},
            agent_metrics={"iterations": 3},
        )

        data = _extract_answer_data_from_result(result)

        assert data["raw_llm_response"] == "Test answer"
        assert data["recursion_limit_reached"] is False
        assert data["answering_mcp_servers"] == ["server1"]
        assert data["usage_metadata"] == {"input_tokens": 100}
        assert data["agent_metrics"] == {"iterations": 3}


class TestCacheIntegration:
    """Integration tests for answer caching in execute_task."""

    @patch("karenina.benchmark.verification.runner.run_single_model_verification")
    def test_execute_task_caches_answer(self, mock_verify):
        """Test that execute_task caches generated answers."""
        # Setup mock
        mock_result = VerificationResult(
            question_id="q1",
            template_id="t1",
            completed_without_errors=True,
            question_text="Test",
            raw_llm_response="Answer",
            answering_model="model1",
            parsing_model="model2",
            execution_time=1.0,
            timestamp="2025-01-01",
        )
        mock_verify.return_value = mock_result

        # Create task
        answering_model = ModelConfig(
            id="model1",
            model_provider="openai",
            model_name="gpt-4.1-mini",
            temperature=0.0,
            system_prompt="Test",
        )
        parsing_model = ModelConfig(
            id="model2",
            model_provider="openai",
            model_name="gpt-4.1-mini",
            temperature=0.0,
            system_prompt="Test",
        )

        task = {
            "question_id": "q1",
            "question_text": "Test question",
            "template_code": "class Answer: pass",
            "answering_model": answering_model,
            "parsing_model": parsing_model,
            "replicate": None,
            "rubric": None,
        }

        cache = AnswerTraceCache()

        # First execution - should generate and cache
        result_key1, result1 = execute_task(task, cache)
        assert mock_verify.call_count == 1
        assert mock_verify.call_args[1]["cached_answer_data"] is None

        # Second execution - should use cache
        result_key2, result2 = execute_task(task, cache)
        assert mock_verify.call_count == 2
        assert mock_verify.call_args[1]["cached_answer_data"] is not None
        assert mock_verify.call_args[1]["cached_answer_data"]["raw_llm_response"] == "Answer"

        # Cache stats should show one hit
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    @patch("karenina.benchmark.verification.runner.run_single_model_verification")
    def test_different_replicates_not_cached(self, mock_verify):
        """Test that different replicates generate separate answers."""
        mock_result = VerificationResult(
            question_id="q1",
            template_id="t1",
            completed_without_errors=True,
            question_text="Test",
            raw_llm_response="Answer",
            answering_model="model1",
            parsing_model="model2",
            execution_time=1.0,
            timestamp="2025-01-01",
        )
        mock_verify.return_value = mock_result

        answering_model = ModelConfig(
            id="model1",
            model_provider="openai",
            model_name="gpt-4.1-mini",
            temperature=0.0,
            system_prompt="Test",
        )
        parsing_model = ModelConfig(
            id="model2",
            model_provider="openai",
            model_name="gpt-4.1-mini",
            temperature=0.0,
            system_prompt="Test",
        )

        task_rep1 = {
            "question_id": "q1",
            "question_text": "Test",
            "template_code": "class Answer: pass",
            "answering_model": answering_model,
            "parsing_model": parsing_model,
            "replicate": 1,
            "rubric": None,
        }

        task_rep2 = {**task_rep1, "replicate": 2}

        cache = AnswerTraceCache()

        # Execute both replicates
        execute_task(task_rep1, cache)
        execute_task(task_rep2, cache)

        # Both should generate (no cache hit)
        assert mock_verify.call_count == 2
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 2
