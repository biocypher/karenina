"""Integration tests for enhanced answer cache optimization.

This module tests the cache optimization system including:
- Answer sharing across multiple parsing models
- Cache key correctness (question_id + answering_model + replicate)
- Sequential and parallel execution modes

Note: Most tests use execute_sequential to avoid anyio portal overhead.
One test verifies parallel execution works correctly.
"""

from unittest.mock import patch

import pytest

from karenina.benchmark.verification.batch_runner import execute_sequential
from karenina.schemas.workflow import ModelConfig, VerificationResult
from karenina.schemas.workflow.verification import (
    VerificationResultMetadata,
    VerificationResultRubric,
    VerificationResultTemplate,
)


def create_mock_result(kwargs: dict) -> VerificationResult:
    """Create a standard mock VerificationResult from kwargs."""
    timestamp = "2025-01-01"
    result_id = VerificationResultMetadata.compute_result_id(
        question_id=kwargs["question_id"],
        answering_model=kwargs["answering_model"].id,
        parsing_model=kwargs["parsing_model"].id,
        timestamp=timestamp,
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
            timestamp=timestamp,
            result_id=result_id,
        ),
        template=VerificationResultTemplate(
            raw_llm_response=f"Answer for {kwargs['question_id']}",
        ),
        rubric=VerificationResultRubric(rubric_evaluation_performed=False),
    )


def create_model(model_id: str, system_prompt: str = "Test") -> ModelConfig:
    """Create a test ModelConfig."""
    return ModelConfig(
        id=model_id,
        model_provider="anthropic",
        model_name="claude-haiku-4-5",
        temperature=0.0,
        system_prompt=system_prompt,
    )


def create_task(
    question_id: str,
    answering_model: ModelConfig,
    parsing_model: ModelConfig,
    replicate: int | None = None,
) -> dict:
    """Create a standard verification task dict."""
    return {
        "question_id": question_id,
        "question_text": f"What is the answer to {question_id}?",
        "template_code": "class Answer: pass",
        "answering_model": answering_model,
        "parsing_model": parsing_model,
        "replicate": replicate,
        "rubric": None,
        "keywords": None,
        "few_shot_examples": None,
        "few_shot_enabled": False,
        "abstention_enabled": False,
        "deep_judgment_enabled": False,
    }


@pytest.mark.integration
class TestCacheOptimizationIntegration:
    """Integration tests for cache optimization.

    These tests verify the answer cache behavior by tracking which calls
    receive cached_answer_data vs which generate new answers.
    """

    @patch("karenina.benchmark.verification.runner.run_single_model_verification")
    def test_sequential_cache_shares_answers_across_parsers(self, mock_verify):
        """Test that sequential execution shares answers across different parsing models."""
        call_log = []

        def mock_verify_track(**kwargs):
            call_log.append(
                {
                    "question_id": kwargs["question_id"],
                    "parsing_model": kwargs["parsing_model"].id,
                    "cached": kwargs.get("cached_answer_data") is not None,
                }
            )
            return create_mock_result(kwargs)

        mock_verify.side_effect = mock_verify_track

        answering_model = create_model("answering_1")
        parsing_model_1 = create_model("parser_1")
        parsing_model_2 = create_model("parser_2")
        parsing_model_3 = create_model("parser_3")

        # Create 3 tasks for the same question with different parsers
        tasks = [
            create_task("q1", answering_model, parsing_model_1),
            create_task("q1", answering_model, parsing_model_2),
            create_task("q1", answering_model, parsing_model_3),
        ]

        results = execute_sequential(tasks)

        assert len(results) == 3, f"Expected 3 results, got {len(results)}"

        # First call should generate (not cached)
        assert not call_log[0]["cached"], "First call should generate answer"

        # Subsequent calls should use cache
        assert call_log[1]["cached"], "Second call should use cached answer"
        assert call_log[2]["cached"], "Third call should use cached answer"

    @patch("karenina.benchmark.verification.runner.run_single_model_verification")
    def test_cache_key_includes_question_id(self, mock_verify):
        """Test that different questions don't share cache entries."""
        call_log = []

        def mock_verify_track(**kwargs):
            call_log.append(
                {
                    "question_id": kwargs["question_id"],
                    "cached": kwargs.get("cached_answer_data") is not None,
                }
            )
            return create_mock_result(kwargs)

        mock_verify.side_effect = mock_verify_track

        answering_model = create_model("answering_1")
        parsing_model = create_model("parser_1")

        # Create tasks for different questions
        tasks = [
            create_task("q1", answering_model, parsing_model),
            create_task("q2", answering_model, parsing_model),
            create_task("q3", answering_model, parsing_model),
        ]

        results = execute_sequential(tasks)

        assert len(results) == 3
        # All should be non-cached since they have different questions
        cached_calls = sum(1 for call in call_log if call["cached"])
        assert cached_calls == 0, f"Expected 0 cached calls (different questions), got {cached_calls}"

    @patch("karenina.benchmark.verification.runner.run_single_model_verification")
    def test_cache_key_includes_answering_model(self, mock_verify):
        """Test that different answering models don't share cache entries."""
        call_log = []

        def mock_verify_track(**kwargs):
            call_log.append(
                {
                    "answering_model": kwargs["answering_model"].id,
                    "cached": kwargs.get("cached_answer_data") is not None,
                }
            )
            return create_mock_result(kwargs)

        mock_verify.side_effect = mock_verify_track

        answering_model_1 = create_model("answering_1")
        answering_model_2 = create_model("answering_2")
        parsing_model = create_model("parser_1")

        # Same question, different answering models
        tasks = [
            create_task("q1", answering_model_1, parsing_model),
            create_task("q1", answering_model_2, parsing_model),
        ]

        results = execute_sequential(tasks)

        assert len(results) == 2
        # Both should be non-cached since they use different answering models
        cached_calls = sum(1 for call in call_log if call["cached"])
        assert cached_calls == 0, f"Expected 0 cached calls (different answering models), got {cached_calls}"

    @patch("karenina.benchmark.verification.runner.run_single_model_verification")
    def test_cache_key_includes_replicate(self, mock_verify):
        """Test that different replicates don't share cache entries."""
        call_log = []

        def mock_verify_track(**kwargs):
            call_log.append(
                {
                    "replicate": kwargs.get("replicate"),
                    "cached": kwargs.get("cached_answer_data") is not None,
                }
            )
            return create_mock_result(kwargs)

        mock_verify.side_effect = mock_verify_track

        answering_model = create_model("answering_1")
        parsing_model = create_model("parser_1")

        # Same question and models, but different replicates
        tasks = [
            create_task("q1", answering_model, parsing_model, replicate=1),
            create_task("q1", answering_model, parsing_model, replicate=2),
            create_task("q1", answering_model, parsing_model, replicate=3),
        ]

        results = execute_sequential(tasks)

        assert len(results) == 3
        # All should be non-cached since they have different replicates
        cached_calls = sum(1 for call in call_log if call["cached"])
        assert cached_calls == 0, f"Expected 0 cached calls (different replicates), got {cached_calls}"

    @patch("karenina.benchmark.verification.runner.run_single_model_verification")
    def test_multiple_questions_multiple_parsers(self, mock_verify):
        """Test cache behavior with multiple questions and multiple parsing models."""
        call_log = []

        def mock_verify_track(**kwargs):
            call_log.append(
                {
                    "question_id": kwargs["question_id"],
                    "parsing_model": kwargs["parsing_model"].id,
                    "cached": kwargs.get("cached_answer_data") is not None,
                }
            )
            return create_mock_result(kwargs)

        mock_verify.side_effect = mock_verify_track

        answering_model = create_model("answering_1")
        parser_1 = create_model("parser_1")
        parser_2 = create_model("parser_2")

        # 2 questions × 2 parsers = 4 tasks
        # Expected: 2 generations (one per question), 2 cache hits
        tasks = [
            create_task("q1", answering_model, parser_1),
            create_task("q1", answering_model, parser_2),
            create_task("q2", answering_model, parser_1),
            create_task("q2", answering_model, parser_2),
        ]

        results = execute_sequential(tasks)

        assert len(results) == 4

        # Count by question
        q1_calls = [c for c in call_log if c["question_id"] == "q1"]
        q2_calls = [c for c in call_log if c["question_id"] == "q2"]

        # First call for each question should generate
        assert not q1_calls[0]["cached"], "First q1 call should generate"
        assert not q2_calls[0]["cached"], "First q2 call should generate"

        # Second call for each question should use cache
        assert q1_calls[1]["cached"], "Second q1 call should use cache"
        assert q2_calls[1]["cached"], "Second q2 call should use cache"

    @patch("karenina.benchmark.verification.runner.run_single_model_verification")
    def test_cache_statistics_tracked(self, mock_verify):
        """Test that cache statistics are properly tracked."""
        mock_verify.side_effect = lambda **kwargs: create_mock_result(kwargs)

        answering_model = create_model("answering_1")
        parser_1 = create_model("parser_1")
        parser_2 = create_model("parser_2")

        # 2 tasks for same question: 1 miss, 1 hit
        tasks = [
            create_task("q1", answering_model, parser_1),
            create_task("q1", answering_model, parser_2),
        ]

        # Note: We can't easily inspect cache stats without accessing internals,
        # but the logs will show "Answer cache statistics: X hits, Y misses"
        results = execute_sequential(tasks)

        assert len(results) == 2
        assert mock_verify.call_count == 2

    @patch("karenina.benchmark.verification.runner.run_single_model_verification")
    def test_all_results_have_correct_metadata(self, mock_verify):
        """Test that all results have correct question and model metadata."""
        mock_verify.side_effect = lambda **kwargs: create_mock_result(kwargs)

        answering_model = create_model("answering_1")
        parser_1 = create_model("parser_1")
        parser_2 = create_model("parser_2")

        tasks = [
            create_task("q1", answering_model, parser_1),
            create_task("q1", answering_model, parser_2),
            create_task("q2", answering_model, parser_1),
        ]

        results = execute_sequential(tasks)

        # Verify all results are present with correct metadata
        result_list = list(results.values())
        assert len(result_list) == 3

        question_ids = {r.question_id for r in result_list}
        assert question_ids == {"q1", "q2"}

        parsing_models = {r.parsing_model for r in result_list}
        assert parsing_models == {"parser_1", "parser_2"}


@pytest.mark.integration
class TestParallelExecution:
    """Test parallel execution specifically.

    Note: These tests are slower due to anyio portal overhead.
    They verify that parallel execution maintains correctness.
    """

    @patch("karenina.benchmark.verification.runner.run_single_model_verification")
    def test_parallel_execution_completes(self, mock_verify):
        """Test that parallel execution completes and returns all results."""
        from karenina.benchmark.verification.batch_runner import execute_parallel

        mock_verify.side_effect = lambda **kwargs: create_mock_result(kwargs)

        answering_model = create_model("answering_1")
        parsing_model = create_model("parser_1")

        # Simple case: 3 different questions
        tasks = [
            create_task("q1", answering_model, parsing_model),
            create_task("q2", answering_model, parsing_model),
            create_task("q3", answering_model, parsing_model),
        ]

        results = execute_parallel(tasks, max_workers=2)

        assert len(results) == 3
        question_ids = {results[key].question_id for key in results}
        assert question_ids == {"q1", "q2", "q3"}


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
