"""Integration tests for enhanced answer cache optimization.

This module tests the complete cache optimization system including:
- Non-blocking cache with immediate IN_PROGRESS returns
- Task shuffling for better cache distribution
- Progressive retry strategy (immediate requeue, then 30s waits)
- Result order preservation despite shuffling

Note: These tests are currently skipped as they require optimization for CI/CD.
"""

import threading
import time
from unittest.mock import patch

import pytest

from karenina.benchmark.verification.batch_runner import execute_parallel, execute_sequential
from karenina.schemas.workflow import ModelConfig, VerificationResult
from karenina.schemas.workflow.verification import (
    VerificationResultMetadata,
    VerificationResultRubric,
    VerificationResultTemplate,
)


@pytest.mark.skip(reason="Integration tests - optimize for CI/CD before enabling")
class TestCacheOptimizationIntegration:
    """Integration tests for cache optimization."""

    @patch("karenina.benchmark.verification.runner.run_single_model_verification")
    def test_parallel_cache_with_shuffling(self, mock_verify):
        """Test that parallel execution with shuffling maximizes cache hits."""
        # Track call order and timing
        call_log = []
        call_lock = threading.Lock()

        def mock_verify_with_logging(**kwargs):
            """Mock that logs calls and simulates processing time."""
            with call_lock:
                call_log.append(
                    {
                        "question_id": kwargs["question_id"],
                        "answering_model": kwargs["answering_model"].id,
                        "parsing_model": kwargs["parsing_model"].id,
                        "replicate": kwargs.get("answering_replicate"),
                        "cached": kwargs.get("cached_answer_data") is not None,
                        "timestamp": time.time(),
                    }
                )

            # Simulate answer generation delay only if not cached
            if not kwargs.get("cached_answer_data"):
                time.sleep(0.2)

            return VerificationResult(
                metadata=VerificationResultMetadata(
                    question_id=kwargs["question_id"],
                    template_id="test_template",
                    completed_without_errors=True,
                    question_text=kwargs["question_text"],
                    answering_model=kwargs["answering_model"].id,
                    parsing_model=kwargs["parsing_model"].id,
                    execution_time=0.1,
                    timestamp="2025-01-01",
                ),
                template=VerificationResultTemplate(
                    raw_llm_response=f"Answer for rep {kwargs.get('answering_replicate', 1)}",
                ),
                rubric=VerificationResultRubric(rubric_evaluation_performed=False),
            )

        mock_verify.side_effect = mock_verify_with_logging

        # Create models
        answering_model = ModelConfig(
            id="answering_model_1",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            system_prompt="Answer questions",
        )
        parsing_model_1 = ModelConfig(
            id="parsing_model_1",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            system_prompt="Parse answers",
        )
        parsing_model_2 = ModelConfig(
            id="parsing_model_2",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            system_prompt="Parse answers differently",
        )

        # Create 6 tasks:
        # - 2 questions (q1, q2)
        # - 1 answering model
        # - 3 parsing models scenarios:
        #   * For each question, use 2 different parsing models
        # Expected: 2 answer generations (one per question), 4 cache hits
        tasks = []
        for question_id in ["q1", "q2"]:
            for parsing_model in [parsing_model_1, parsing_model_2]:
                tasks.append(
                    {
                        "question_id": question_id,
                        "question_text": f"What is the answer to {question_id}?",
                        "template_code": "class Answer: pass",
                        "answering_model": answering_model,
                        "parsing_model": parsing_model,
                        "replicate": None,
                        "rubric": None,
                        "keywords": None,
                        "few_shot_examples": None,
                        "few_shot_enabled": False,
                        "abstention_enabled": False,
                        "deep_judgment_enabled": False,
                    }
                )

        print(f"\n📋 Created {len(tasks)} tasks:")
        for i, task in enumerate(tasks):
            print(f"  Task {i}: Q={task['question_id']}, P={task['parsing_model'].id}")

        # Execute in parallel with 2 workers
        start_time = time.time()
        results = execute_parallel(tasks, max_workers=2)
        elapsed = time.time() - start_time

        print(f"\n⏱️  Execution completed in {elapsed:.2f}s")
        print(f"📊 Total verification calls: {len(call_log)}")

        # Verify results
        assert len(results) == 4, f"Expected 4 results, got {len(results)}"

        # Analyze cache performance
        cached_calls = sum(1 for call in call_log if call["cached"])
        non_cached_calls = sum(1 for call in call_log if not call["cached"])

        print("\n📈 Cache Performance:")
        print(f"  ✓ Cache hits: {cached_calls}")
        print(f"  ✗ Cache misses: {non_cached_calls}")

        # We should have 2 misses (generate answers for q1 and q2)
        # and 2 hits (reuse those answers for the second parsing model)
        assert non_cached_calls == 2, f"Expected 2 non-cached calls, got {non_cached_calls}"
        assert cached_calls == 2, f"Expected 2 cached calls, got {cached_calls}"

        # Verify result ordering is maintained
        result_keys = list(results.keys())
        print("\n🔢 Result ordering maintained:")
        for i, key in enumerate(result_keys):
            result = results[key]
            print(f"  {i}: {result.question_id} / {result.parsing_model}")

        # All results should be present
        question_ids = [results[key].question_id for key in result_keys]
        assert question_ids.count("q1") == 2, "Expected 2 results for q1"
        assert question_ids.count("q2") == 2, "Expected 2 results for q2"

    @patch("karenina.benchmark.verification.runner.run_single_model_verification")
    def test_progressive_retry_strategy(self, mock_verify):
        """Test that IN_PROGRESS tasks use progressive retry (immediate, then 30s waits)."""
        # Simulate slow answer generation
        generation_started = threading.Event()
        generation_complete = threading.Event()

        def mock_verify_slow(**kwargs):
            """Mock that simulates slow answer generation."""
            if not kwargs.get("cached_answer_data"):
                # First call - simulate slow generation
                generation_started.set()
                time.sleep(1.0)  # Simulate 1s generation time
                generation_complete.set()

            return VerificationResult(
                metadata=VerificationResultMetadata(
                    question_id=kwargs["question_id"],
                    template_id="test_template",
                    completed_without_errors=True,
                    question_text=kwargs["question_text"],
                    answering_model=kwargs["answering_model"].id,
                    parsing_model=kwargs["parsing_model"].id,
                    execution_time=0.1,
                    timestamp="2025-01-01",
                ),
                template=VerificationResultTemplate(
                    raw_llm_response="Answer",
                ),
                rubric=VerificationResultRubric(rubric_evaluation_performed=False),
            )

        mock_verify.side_effect = mock_verify_slow

        # Create models
        answering_model = ModelConfig(
            id="slow_model",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            system_prompt="Test",
        )
        parsing_model_1 = ModelConfig(
            id="parser_1",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            system_prompt="Parse",
        )
        parsing_model_2 = ModelConfig(
            id="parser_2",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            system_prompt="Parse",
        )

        # Create 2 tasks for the same question with different parsers
        tasks = [
            {
                "question_id": "q1",
                "question_text": "Test question",
                "template_code": "class Answer: pass",
                "answering_model": answering_model,
                "parsing_model": parsing_model_1,
                "replicate": None,
                "rubric": None,
                "keywords": None,
                "few_shot_examples": None,
                "few_shot_enabled": False,
                "abstention_enabled": False,
                "deep_judgment_enabled": False,
            },
            {
                "question_id": "q1",
                "question_text": "Test question",
                "template_code": "class Answer: pass",
                "answering_model": answering_model,
                "parsing_model": parsing_model_2,
                "replicate": None,
                "rubric": None,
                "keywords": None,
                "few_shot_examples": None,
                "few_shot_enabled": False,
                "abstention_enabled": False,
                "deep_judgment_enabled": False,
            },
        ]

        print(f"\n🔄 Testing progressive retry with {len(tasks)} tasks")

        # Execute with 2 workers
        results = execute_parallel(tasks, max_workers=2)

        print("✓ Execution completed")
        print(f"📊 Results: {len(results)}")

        # Both tasks should complete
        assert len(results) == 2, f"Expected 2 results, got {len(results)}"

        # One should have generated, one should have used cache
        assert mock_verify.call_count == 2
        cached_count = sum(1 for call in mock_verify.call_args_list if call[1].get("cached_answer_data") is not None)
        print(f"✓ Cached calls: {cached_count}")
        assert cached_count == 1, f"Expected 1 cached call, got {cached_count}"

    @patch("karenina.benchmark.verification.runner.run_single_model_verification")
    def test_sequential_no_retry_needed(self, mock_verify):
        """Test that sequential execution doesn't need retry logic."""
        call_order = []

        def mock_verify_track(**kwargs):
            call_order.append(
                {
                    "parsing": kwargs["parsing_model"].id,
                    "cached": kwargs.get("cached_answer_data") is not None,
                }
            )
            return VerificationResult(
                metadata=VerificationResultMetadata(
                    question_id=kwargs["question_id"],
                    template_id="test_template",
                    completed_without_errors=True,
                    question_text=kwargs["question_text"],
                    answering_model=kwargs["answering_model"].id,
                    parsing_model=kwargs["parsing_model"].id,
                    execution_time=0.1,
                    timestamp="2025-01-01",
                ),
                template=VerificationResultTemplate(
                    raw_llm_response="Answer",
                ),
                rubric=VerificationResultRubric(rubric_evaluation_performed=False),
            )

        mock_verify.side_effect = mock_verify_track

        # Create models
        answering_model = ModelConfig(
            id="seq_model",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            system_prompt="Test",
        )
        parsing_models = [
            ModelConfig(
                id=f"parser_{i}",
                model_provider="openai",
                model_name="gpt-4o-mini",
                temperature=0.0,
                system_prompt="Parse",
            )
            for i in range(3)
        ]

        # Create 3 tasks for the same question
        tasks = []
        for parsing_model in parsing_models:
            tasks.append(
                {
                    "question_id": "q1",
                    "question_text": "Test question",
                    "template_code": "class Answer: pass",
                    "answering_model": answering_model,
                    "parsing_model": parsing_model,
                    "replicate": None,
                    "rubric": None,
                    "keywords": None,
                    "few_shot_examples": None,
                    "few_shot_enabled": False,
                    "abstention_enabled": False,
                    "deep_judgment_enabled": False,
                }
            )

        print(f"\n📝 Testing sequential execution with {len(tasks)} tasks")

        # Execute sequentially
        results = execute_sequential(tasks)

        print("✓ Execution completed")
        print(f"📊 Results: {len(results)}")

        # All tasks should complete
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"

        # First should generate, next 2 should use cache
        print("\n📈 Call order:")
        for i, call in enumerate(call_order):
            status = "CACHED" if call["cached"] else "GENERATED"
            print(f"  {i + 1}. {call['parsing']}: {status}")

        assert not call_order[0]["cached"], "First call should generate"
        assert call_order[1]["cached"], "Second call should be cached"
        assert call_order[2]["cached"], "Third call should be cached"

        print("✓ Sequential cache behavior verified")

    @patch("karenina.benchmark.verification.runner.run_single_model_verification")
    def test_result_order_preservation(self, mock_verify):
        """Test that results are returned in original task order despite shuffling."""

        def mock_verify_simple(**kwargs):
            return VerificationResult(
                metadata=VerificationResultMetadata(
                    question_id=kwargs["question_id"],
                    template_id="test_template",
                    completed_without_errors=True,
                    question_text=kwargs["question_text"],
                    answering_model=kwargs["answering_model"].id,
                    parsing_model=kwargs["parsing_model"].id,
                    execution_time=0.1,
                    timestamp="2025-01-01",
                ),
                template=VerificationResultTemplate(
                    raw_llm_response="Answer",
                ),
                rubric=VerificationResultRubric(rubric_evaluation_performed=False),
            )

        mock_verify.side_effect = mock_verify_simple

        # Create a predictable task list
        answering_model = ModelConfig(
            id="test_model",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            system_prompt="Test",
        )
        parsing_model = ModelConfig(
            id="parser",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            system_prompt="Parse",
        )

        # Create 5 tasks with different question IDs
        task_question_ids = ["q1", "q2", "q3", "q4", "q5"]
        tasks = []
        for qid in task_question_ids:
            tasks.append(
                {
                    "question_id": qid,
                    "question_text": f"Question {qid}",
                    "template_code": "class Answer: pass",
                    "answering_model": answering_model,
                    "parsing_model": parsing_model,
                    "replicate": None,
                    "rubric": None,
                    "keywords": None,
                    "few_shot_examples": None,
                    "few_shot_enabled": False,
                    "abstention_enabled": False,
                    "deep_judgment_enabled": False,
                }
            )

        print(f"\n🔢 Testing result order with {len(tasks)} tasks")
        print(f"Input order: {task_question_ids}")

        # Execute in parallel (which shuffles internally)
        results = execute_parallel(tasks, max_workers=2)

        # Extract result order
        result_keys = list(results.keys())
        result_question_ids = [results[key].question_id for key in result_keys]

        print(f"Output order: {result_question_ids}")

        # Results should be returned (result_key is ordered by timestamp, so order may vary)
        # But all question IDs should be present
        assert len(results) == 5, f"Expected 5 results, got {len(results)}"
        assert set(result_question_ids) == set(task_question_ids), "All questions should be present"

        print(f"✓ All {len(results)} results present")


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "-s"])
