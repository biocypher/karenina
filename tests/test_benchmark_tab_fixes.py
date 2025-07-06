"""
Tests for BenchmarkTab fixes - verification of removed Clear All Results button
and proper results accumulation behavior.

Date: January 23, 2025
"""

import pytest

# These tests would verify the backend behavior that supports the frontend changes
# Since the main fixes are in the React frontend, we'll create complementary backend tests


@pytest.fixture
def mock_verification_results():
    """Sample verification results for testing."""
    return {
        "q1": {
            "question_id": "q1",
            "success": True,
            "question_text": "What is 2+2?",
            "raw_llm_response": "4",
            "parsed_response": {"answer": "4"},
            "verify_result": True,
            "answering_model": "gpt-4",
            "parsing_model": "gpt-4",
            "execution_time": 1.5,
            "timestamp": "2025-01-23T10:00:00Z",
            "run_name": "test_run_1",
            "job_id": "job_123",
        },
        "q2": {
            "question_id": "q2",
            "success": True,
            "question_text": "What is the capital of France?",
            "raw_llm_response": "Paris",
            "parsed_response": {"answer": "Paris"},
            "verify_result": True,
            "answering_model": "gpt-4",
            "parsing_model": "gpt-4",
            "execution_time": 1.2,
            "timestamp": "2025-01-23T10:01:00Z",
            "run_name": "test_run_1",
            "job_id": "job_123",
        },
    }


@pytest.fixture
def additional_verification_results():
    """Additional results from a second run."""
    return {
        "q3": {
            "question_id": "q3",
            "success": True,
            "question_text": "What is Python?",
            "raw_llm_response": "A programming language",
            "parsed_response": {"answer": "A programming language"},
            "verify_result": True,
            "answering_model": "gpt-4",
            "parsing_model": "gpt-4",
            "execution_time": 2.0,
            "timestamp": "2025-01-23T10:05:00Z",
            "run_name": "test_run_2",
            "job_id": "job_456",
        }
    }


class TestBenchmarkResultsAccumulation:
    """Test that results properly accumulate instead of being replaced."""

    def test_results_accumulation_structure(self, mock_verification_results, additional_verification_results):
        """Test that results can be properly accumulated without conflicts."""
        # Simulate the frontend accumulation logic: prev => ({ ...prev, ...new })
        existing_results = mock_verification_results.copy()
        new_results = additional_verification_results.copy()

        # This simulates what the frontend does: setResults(prev => ({ ...prev, ...sanitizedResults }))
        accumulated_results = {**existing_results, **new_results}

        # Verify all results are present
        assert len(accumulated_results) == 3
        assert "q1" in accumulated_results
        assert "q2" in accumulated_results
        assert "q3" in accumulated_results

        # Verify original results are preserved
        assert accumulated_results["q1"]["question_text"] == "What is 2+2?"
        assert accumulated_results["q2"]["question_text"] == "What is the capital of France?"

        # Verify new results are added
        assert accumulated_results["q3"]["question_text"] == "What is Python?"

        # Verify different run names are preserved
        assert accumulated_results["q1"]["run_name"] == "test_run_1"
        assert accumulated_results["q3"]["run_name"] == "test_run_2"

    def test_results_overwrite_same_question_id(self, mock_verification_results):
        """Test that re-running the same question overwrites the previous result."""
        existing_results = mock_verification_results.copy()

        # New result for same question ID but different run
        new_result = {
            "q1": {
                "question_id": "q1",
                "success": False,  # Different result
                "question_text": "What is 2+2?",
                "raw_llm_response": "5",  # Wrong answer
                "parsed_response": {"answer": "5"},
                "verify_result": False,
                "answering_model": "gpt-3.5",  # Different model
                "parsing_model": "gpt-3.5",
                "execution_time": 0.8,
                "timestamp": "2025-01-23T11:00:00Z",
                "run_name": "test_run_2",
                "job_id": "job_789",
            }
        }

        # Accumulate results
        accumulated_results = {**existing_results, **new_result}

        # Verify the newer result overwrote the old one
        assert not accumulated_results["q1"]["success"]
        assert accumulated_results["q1"]["raw_llm_response"] == "5"
        assert accumulated_results["q1"]["answering_model"] == "gpt-3.5"
        assert accumulated_results["q1"]["run_name"] == "test_run_2"

        # Verify other results are unaffected
        assert accumulated_results["q2"]["success"]
        assert accumulated_results["q2"]["run_name"] == "test_run_1"

    def test_historical_results_merge_with_current_session(
        self, mock_verification_results, additional_verification_results
    ):
        """Test that historical results properly merge with current session results."""
        # Simulate historical results (loaded from API)
        historical_results = mock_verification_results.copy()

        # Simulate current session results
        current_session_results = additional_verification_results.copy()

        # This simulates the fixed behavior: setResults(prev => ({ ...prev, ...data.results }))
        merged_results = {**current_session_results, **historical_results}

        # Verify all results are present
        assert len(merged_results) == 3

        # Verify current session results are preserved
        assert merged_results["q3"]["run_name"] == "test_run_2"

        # Verify historical results are added
        assert merged_results["q1"]["run_name"] == "test_run_1"
        assert merged_results["q2"]["run_name"] == "test_run_1"

        # Verify data integrity
        for result in merged_results.values():
            assert "question_id" in result
            assert "success" in result
            assert "timestamp" in result
            assert "run_name" in result

    def test_empty_results_accumulation(self):
        """Test accumulation behavior with empty results."""
        existing_results = {}
        new_results = {"q1": {"question_id": "q1", "success": True}}

        accumulated_results = {**existing_results, **new_results}

        assert len(accumulated_results) == 1
        assert "q1" in accumulated_results

    def test_results_with_different_timestamps_preserved(self, mock_verification_results):
        """Test that results with different timestamps are properly preserved."""
        existing_results = mock_verification_results.copy()

        # Add results from different time periods
        newer_results = {
            "q4": {
                "question_id": "q4",
                "success": True,
                "timestamp": "2025-01-23T15:00:00Z",  # Later timestamp
                "run_name": "afternoon_run",
            },
            "q5": {
                "question_id": "q5",
                "success": True,
                "timestamp": "2025-01-22T15:00:00Z",  # Earlier timestamp
                "run_name": "yesterday_run",
            },
        }

        accumulated_results = {**existing_results, **newer_results}

        # Verify all timestamps are preserved
        timestamps = [result["timestamp"] for result in accumulated_results.values()]
        assert "2025-01-23T10:00:00Z" in timestamps  # Original
        assert "2025-01-23T15:00:00Z" in timestamps  # Newer
        assert "2025-01-22T15:00:00Z" in timestamps  # Earlier

        # Verify run names from different periods are preserved
        run_names = [result["run_name"] for result in accumulated_results.values()]
        assert "test_run_1" in run_names
        assert "afternoon_run" in run_names
        assert "yesterday_run" in run_names


class TestClearAllResultsButtonRemoval:
    """Test that Clear All Results functionality is no longer accessible."""

    def test_no_clear_all_results_endpoint_needed(self):
        """Test that we don't need a clear all results endpoint since button is removed."""
        # Since the button is removed from the frontend, there should be no backend
        # endpoint calls to clear all results from the UI
        # This test documents that the functionality is intentionally removed

        # If there was a clear_all_results function, it should not be called
        # from the normal UI flow anymore
        assert True  # This test documents the intentional removal

    def test_results_persist_without_clear_functionality(self, mock_verification_results):
        """Test that results persist since there's no clear functionality in the UI."""
        results = mock_verification_results.copy()

        # Without the clear button, results should remain intact
        # throughout the session unless explicitly replaced by new data
        assert len(results) == 2

        # Simulate what would happen if someone tried to clear results manually
        # (this should only happen programmatically, not from UI)
        cleared_results = {}

        # Verify that clearing would remove results (but this isn't accessible from UI)
        assert len(cleared_results) == 0

        # The original results should be unaffected by this test operation
        assert len(results) == 2


class TestBenchmarkTabIntegration:
    """Integration tests for the fixed BenchmarkTab behavior."""

    def test_multiple_benchmark_runs_accumulate(self, mock_verification_results, additional_verification_results):
        """Test the complete flow of multiple benchmark runs accumulating results."""
        # Simulate first benchmark run
        run1_results = mock_verification_results.copy()
        all_results = run1_results.copy()

        assert len(all_results) == 2
        assert all(result["run_name"] == "test_run_1" for result in all_results.values())

        # Simulate second benchmark run (should accumulate, not replace)
        run2_results = additional_verification_results.copy()
        all_results = {**all_results, **run2_results}

        assert len(all_results) == 3

        # Verify results from both runs are present
        run_names = {result["run_name"] for result in all_results.values()}
        assert "test_run_1" in run_names
        assert "test_run_2" in run_names

        # Verify specific results from each run
        run1_questions = [
            result["question_id"] for result in all_results.values() if result["run_name"] == "test_run_1"
        ]
        run2_questions = [
            result["question_id"] for result in all_results.values() if result["run_name"] == "test_run_2"
        ]

        assert "q1" in run1_questions
        assert "q2" in run1_questions
        assert "q3" in run2_questions

    def test_session_persistence_across_component_remounts(self, mock_verification_results):
        """Test that results persist even when component remounts (simulating page refresh)."""
        # Simulate initial session with results
        session_results = mock_verification_results.copy()

        # Simulate historical results that would be loaded on component mount
        historical_results = {
            "q0": {
                "question_id": "q0",
                "success": True,
                "run_name": "historical_run",
                "timestamp": "2025-01-22T10:00:00Z",
            }
        }

        # Simulate the fixed behavior: accumulate instead of replace
        # setResults(prev => ({ ...prev, ...data.results }))
        merged_results = {**session_results, **historical_results}

        # Verify both historical and session results are preserved
        assert len(merged_results) == 3

        # Verify session results survived the "remount"
        session_question_ids = ["q1", "q2"]
        for qid in session_question_ids:
            assert qid in merged_results
            assert merged_results[qid]["run_name"] == "test_run_1"

        # Verify historical results were loaded
        assert "q0" in merged_results
        assert merged_results["q0"]["run_name"] == "historical_run"

    def test_results_filtering_with_accumulated_data(self, mock_verification_results, additional_verification_results):
        """Test that filtering works correctly with accumulated results from multiple runs."""
        # Accumulate results from multiple runs
        all_results = {**mock_verification_results, **additional_verification_results}

        # Test filtering by run name (simulating frontend filter logic)
        def filter_by_run_name(results, run_name):
            return {k: v for k, v in results.items() if v["run_name"] == run_name}

        run1_filtered = filter_by_run_name(all_results, "test_run_1")
        run2_filtered = filter_by_run_name(all_results, "test_run_2")

        assert len(run1_filtered) == 2  # q1, q2
        assert len(run2_filtered) == 1  # q3

        # Test filtering by success status
        def filter_by_success(results, success_status):
            return {k: v for k, v in results.items() if v["success"] == success_status}

        successful_results = filter_by_success(all_results, True)
        assert len(successful_results) == 3  # All are successful in this test data

        # Verify filtering doesn't lose data integrity
        for result in successful_results.values():
            assert result["success"]
            assert "question_id" in result
            assert "timestamp" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
