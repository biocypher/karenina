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

        answer_data, should_generate = cache.get_or_reserve(key)

        assert answer_data is None
        assert should_generate is True
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
        answer_data, should_generate = cache.get_or_reserve(key)

        assert answer_data == test_data
        assert should_generate is False

    def test_cache_handles_failure(self):
        """Test that cache handles failed answer generation."""
        cache = AnswerTraceCache()
        key = "test_question_model1"

        # First call reserves slot
        cache.get_or_reserve(key)

        # Complete with error
        error = Exception("LLM failed")
        cache.complete(key, None, error=error)

        # Second call should allow retry
        answer_data, should_generate = cache.get_or_reserve(key)

        assert answer_data is None
        assert should_generate is True

    def test_waiting_for_in_progress_answer(self):
        """Test that tasks wait for in-progress answers."""
        cache = AnswerTraceCache(wait_timeout=2.0)
        key = "test_question_model1"

        # First task reserves slot
        cache.get_or_reserve(key)

        # Simulate second task waiting
        test_data = {"raw_llm_response": "Test answer"}

        def complete_after_delay():
            time.sleep(0.5)
            cache.complete(key, test_data, error=None)

        # Start background thread to complete the answer
        thread = threading.Thread(target=complete_after_delay)
        thread.start()

        # Second task should wait and receive cached answer
        start_time = time.time()
        answer_data, should_generate = cache.get_or_reserve(key)
        elapsed = time.time() - start_time

        thread.join()

        assert answer_data == test_data
        assert should_generate is False
        assert 0.4 < elapsed < 1.0  # Should have waited ~0.5s

    def test_timeout_prevents_deadlock(self):
        """Test that timeout prevents deadlock."""
        cache = AnswerTraceCache(wait_timeout=0.5)
        key = "test_question_model1"

        # First task reserves slot but never completes
        cache.get_or_reserve(key)

        # Second task should timeout and proceed to generate
        start_time = time.time()
        answer_data, should_generate = cache.get_or_reserve(key)
        elapsed = time.time() - start_time

        assert answer_data is None
        assert should_generate is True
        assert 0.4 < elapsed < 1.0  # Should have timed out after ~0.5s

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

    def test_thread_safety_multiple_waiters(self):
        """Test that multiple tasks can wait for the same answer."""
        cache = AnswerTraceCache(wait_timeout=2.0)
        key = "test_question_model1"
        test_data = {"raw_llm_response": "Test answer"}

        # First task reserves slot
        cache.get_or_reserve(key)

        results = []

        def waiter_task():
            answer_data, should_generate = cache.get_or_reserve(key)
            results.append((answer_data, should_generate))

        # Start multiple waiting tasks
        threads = [threading.Thread(target=waiter_task) for _ in range(5)]
        for t in threads:
            t.start()

        # Complete after brief delay
        time.sleep(0.2)
        cache.complete(key, test_data, error=None)

        # Wait for all threads
        for t in threads:
            t.join()

        # All waiters should receive the cached answer
        assert len(results) == 5
        for answer_data, should_generate in results:
            assert answer_data == test_data
            assert should_generate is False


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
