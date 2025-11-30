"""Test TaskEval formatting and export functions."""

import json
from datetime import datetime

import pytest

from karenina.benchmark.task_eval.models import LogEvent, StepEval, TaskEvalResult
from karenina.schemas.workflow.verification import (
    VerificationResult,
    VerificationResultMetadata,
    VerificationResultRubric,
    VerificationResultTemplate,
)


class TestTaskEvalFormatting:
    """Test formatting methods for TaskEval results."""

    @pytest.fixture
    def sample_step_eval(self) -> None:
        """Create a sample StepEval with various data."""
        return StepEval(
            verification_results={
                "q1": [
                    VerificationResult(
                        metadata=VerificationResultMetadata(
                            question_id="q1",
                            template_id="test_template",
                            completed_without_errors=True,
                            question_text="What is 15 + 27?",
                            answering_model="gpt-4.1-mini",
                            parsing_model="gpt-4.1-mini",
                            execution_time=0.5,
                            timestamp="2025-11-11T00:00:00",
                        ),
                        template=VerificationResultTemplate(
                            raw_llm_response="The answer is 42 because 15 + 27 equals 42",
                            parsed_gt_response={"result": "42"},
                            parsed_llm_response={"result": "42"},
                            template_verification_performed=True,
                            verify_result=True,
                        ),
                        rubric=VerificationResultRubric(
                            rubric_evaluation_performed=True,
                            llm_trait_scores={"accuracy": True, "clarity": 3},
                        ),
                    )
                ],
                "q2": [
                    VerificationResult(
                        metadata=VerificationResultMetadata(
                            question_id="q2",
                            template_id="test_template",
                            completed_without_errors=False,
                            error="Calculation failed",
                            question_text="What is the error case?",
                            answering_model="gpt-4.1-mini",
                            parsing_model="gpt-4.1-mini",
                            execution_time=0.2,
                            timestamp="2025-11-11T00:00:00",
                        ),
                        template=VerificationResultTemplate(
                            raw_llm_response="Error in calculation",
                            parsed_gt_response={"result": "42"},
                            parsed_llm_response={"result": "43"},
                            template_verification_performed=True,
                            verify_result=False,
                        ),
                        rubric=VerificationResultRubric(
                            rubric_evaluation_performed=True,
                            llm_trait_scores={"accuracy": False},
                        ),
                    )
                ],
            }
        )

    @pytest.fixture
    def sample_task_result(self, sample_step_eval) -> None:
        """Create a sample TaskEvalResult."""
        return TaskEvalResult(
            task_id="test_task_123",
            metadata={"purpose": "Test formatting", "version": "1.0"},
            global_eval=sample_step_eval,
            per_step={"step1": sample_step_eval},
            timestamp=datetime.now().isoformat(),
        )

    def test_step_eval_format_rubric_scores(self, sample_step_eval) -> None:
        """Test rubric scores formatting."""
        formatted = sample_step_eval.format_rubric_scores()

        # Check that questions are formatted
        assert "Question: q1" in formatted
        assert "Question: q2" in formatted

        # Check that verification status is shown
        assert "✓ PASSED" in formatted
        assert "✗ FAILED" in formatted

        # Check that rubric scores are shown
        assert "accuracy=✓" in formatted  # q1 has True
        assert "accuracy=✗" in formatted  # q2 has False
        assert "clarity=3" in formatted  # q1 has score 3

        # Check that output text is shown
        assert "The answer is 42" in formatted
        assert "Error in calculation" in formatted

    def test_step_eval_format_question_results(self, sample_step_eval) -> None:
        """Test question verification results formatting."""
        formatted = sample_step_eval.format_verification_results()

        assert "Question: q1" in formatted
        assert "Question: q2" in formatted
        assert "✓ PASSED" in formatted
        assert "✗ FAILED" in formatted
        assert "The answer is 42" in formatted
        assert "Error in calculation" in formatted
        # Rubric scores are shown inline
        assert "accuracy=✓" in formatted  # True shows as checkmark
        assert "clarity=3" in formatted  # Integer scores shown as-is
        assert "accuracy=✗" in formatted  # False shows as X

    def test_step_eval_get_summary_stats(self, sample_step_eval) -> None:
        """Test summary statistics calculation."""
        stats = sample_step_eval.get_summary_stats()

        assert stats["traces_total"] == 2
        assert stats["traces_passed"] == 1  # Only q1 passed
        assert stats["results_total"] == 2
        assert stats["rubric_traits_passed"] == 2  # q1: accuracy=True, clarity=3>0; q2: accuracy=False
        assert stats["rubric_traits_total"] == 3  # q1: 2 traits, q2: 1 trait
        assert stats["success_rate"] == 50.0  # 1/2 traces passed

    def test_task_result_display(self, sample_task_result) -> None:
        """Test display method formatting."""
        display_output = sample_task_result.display()

        assert "TASK EVALUATION RESULTS" in display_output
        assert "Task ID: test_task_123" in display_output
        assert "Purpose: Test formatting" in display_output
        assert "GLOBAL EVALUATION" in display_output
        assert "STEP EVALUATION: step1" in display_output
        assert "Verification Results:" in display_output
        assert "SUMMARY:" in display_output

    def test_task_result_display_minimal(self, sample_task_result) -> None:
        """Test display method includes all sections."""
        display_output = sample_task_result.display()

        assert "TASK EVALUATION RESULTS" in display_output
        assert "GLOBAL EVALUATION" in display_output
        assert "STEP EVALUATION:" in display_output
        assert "SUMMARY:" in display_output

    def test_task_result_summary(self, sample_task_result) -> None:
        """Test summary method."""
        summary = sample_task_result.summary()

        assert "rubric traits passed" in summary
        # Should include both global and step stats (3 traits each: q1 has 2, q2 has 1)
        # Both global and step have same data, so: 3 + 3 = 6 total, 2 + 2 = 4 passed
        assert "4/6 rubric traits passed" in summary

    def test_task_result_summary_compact(self, sample_task_result) -> None:
        """Test compact summary method."""
        compact = sample_task_result.summary_compact()

        assert "TaskEval [test_task_123]:" in compact
        assert "rubric traits" in compact
        assert "success" in compact

    def test_task_result_to_dict_clean(self, sample_task_result) -> None:
        """Test clean dictionary export."""
        clean_dict = sample_task_result.to_dict_clean()

        assert clean_dict["task_id"] == "test_task_123"
        assert clean_dict["metadata"]["purpose"] == "Test formatting"
        assert "global" in clean_dict
        assert "steps" in clean_dict
        assert "summary" in clean_dict
        assert clean_dict["global"]["traces_total"] == 2
        assert clean_dict["steps"]["step1"]["traces_total"] == 2

    def test_task_result_export_json(self, sample_task_result) -> None:
        """Test JSON export functionality."""
        json_str = sample_task_result.export_json()

        # Verify it's valid JSON
        parsed = json.loads(json_str)

        assert parsed["task_id"] == "test_task_123"
        assert parsed["metadata"]["purpose"] == "Test formatting"
        assert "global_evaluation" in parsed
        assert "step_evaluations" in parsed
        assert "summary" in parsed

        # Check structure - contains verification results and summary stats
        assert "verification_results" in parsed["global_evaluation"]
        assert "summary_stats" in parsed["global_evaluation"]
        assert "step1" in parsed["step_evaluations"]

    def test_task_result_export_json_with_logs(self, sample_task_result) -> None:
        """Test JSON export with logs included."""
        # Add some logs
        sample_task_result.logs = {
            "global": [
                LogEvent(level="info", text="Starting evaluation"),
                LogEvent(level="debug", text="Processing question q1"),
            ]
        }

        json_str = sample_task_result.export_json(include_logs=True)
        parsed = json.loads(json_str)

        assert "logs" in parsed
        assert "global" in parsed["logs"]
        assert len(parsed["logs"]["global"]) == 2
        assert parsed["logs"]["global"][0]["text"] == "Starting evaluation"

    def test_task_result_export_json_compact(self, sample_task_result) -> None:
        """Test JSON export with no indentation."""
        json_str = sample_task_result.export_json(indent=None)

        # Should be compact (no newlines/indentation)
        assert "\n" not in json_str

        # But still valid JSON
        parsed = json.loads(json_str)
        assert parsed["task_id"] == "test_task_123"

    def test_task_result_export_markdown(self, sample_task_result) -> None:
        """Test Markdown export functionality."""
        md_str = sample_task_result.export_markdown()

        assert "# Task Evaluation Results" in md_str
        assert "**Task ID:** test_task_123" in md_str
        assert "**Purpose:** Test formatting" in md_str
        assert "## Verification Results" in md_str
        assert "**Traces Passed**:" in md_str
        assert "**Rubric Traits Passed**:" in md_str
        assert "## Summary" in md_str

    def test_task_result_status_emoji(self, sample_task_result) -> None:
        """Test status emoji generation."""
        # Default case (50% success rate)
        emoji = sample_task_result._get_status_emoji()
        assert "❌ Some Failed" in emoji

        # Test with 100% success rate - replace results with all passing
        sample_task_result.global_eval.verification_results = {
            "q1": [
                VerificationResult(
                    metadata=VerificationResultMetadata(
                        question_id="q1",
                        template_id="test_template",
                        completed_without_errors=True,
                        question_text="Test",
                        answering_model="gpt-4.1-mini",
                        parsing_model="gpt-4.1-mini",
                        execution_time=1.0,
                        timestamp="2025-11-11T00:00:00",
                    ),
                    template=VerificationResultTemplate(
                        raw_llm_response="Success",
                        verify_result=True,
                    ),
                )
            ]
        }
        emoji = sample_task_result._get_status_emoji()
        assert "✅ All Passed" in emoji

        # Test with no data
        sample_task_result.global_eval = None
        emoji = sample_task_result._get_status_emoji()
        assert "❓ No Data" in emoji

    def test_empty_task_result_formatting(self) -> None:
        """Test formatting methods with empty/minimal data."""
        empty_result = TaskEvalResult(task_id="empty_test")

        # Should not crash
        display_output = empty_result.display()
        assert "TASK EVALUATION RESULTS" in display_output
        assert "No evaluations performed" in empty_result.summary()

        # JSON export should work
        json_str = empty_result.export_json()
        parsed = json.loads(json_str)
        assert parsed["task_id"] == "empty_test"
        assert parsed["global_evaluation"] is None

    def test_step_eval_empty_formatting(self) -> None:
        """Test StepEval formatting with empty data."""
        empty_step = StepEval()

        rubric_formatted = empty_step.format_rubric_scores()
        assert "No verification results" in rubric_formatted

        question_formatted = empty_step.format_verification_results()
        assert "No verification results" in question_formatted

        stats = empty_step.get_summary_stats()
        assert stats["traces_total"] == 0
        assert stats["success_rate"] == 0

    def test_formatting_with_special_characters(self) -> None:
        """Test formatting handles special characters properly."""
        step_eval = StepEval(
            verification_results={
                "q_special": [
                    VerificationResult(
                        metadata=VerificationResultMetadata(
                            question_id="q_special",
                            template_id="test_template",
                            completed_without_errors=True,
                            question_text="Test with special chars",
                            answering_model="gpt-4.1-mini",
                            parsing_model="gpt-4.1-mini",
                            execution_time=1.0,
                            timestamp="2025-11-11T00:00:00",
                        ),
                        template=VerificationResultTemplate(
                            raw_llm_response="Answer with émojis 🎉 and unicode ñ",
                            verify_result=True,
                        ),
                        rubric=VerificationResultRubric(
                            rubric_evaluation_performed=True,
                            llm_trait_scores={"trait_with_special_chars": True},
                        ),
                    )
                ]
            }
        )

        result = TaskEvalResult(task_id="special_chars_test", global_eval=step_eval)

        # Should handle special characters in all formats
        display_output = result.display()
        json_str = result.export_json()
        md_str = result.export_markdown()

        # Check special characters are preserved
        assert "émojis 🎉" in display_output
        parsed = json.loads(json_str)
        assert "émojis 🎉" in str(parsed)
        assert "special_chars_test" in md_str
