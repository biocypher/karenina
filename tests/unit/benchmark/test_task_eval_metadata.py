"""Tests for TaskEval metadata coherence fixes (issues 024, 114, 115, 117, 160, 165, 166, 168, 179)."""

import pytest

from karenina.benchmark.task_eval import StepEval
from karenina.schemas.config import ModelConfig


@pytest.mark.unit
class TestTaskEvalInterface:
    """Issue 166: taskeval interface registration."""

    def test_taskeval_interface_registered(self):
        """ModelConfig with interface='taskeval' should not raise."""
        config = ModelConfig(
            id="taskeval_user_provided",
            model_name="user-provided",
            model_provider="user-provided",
            interface="taskeval",
        )
        assert config.interface == "taskeval"

    def test_taskeval_sentinel_fields(self):
        """Sentinel model has expected field values."""
        config = ModelConfig(
            id="taskeval_user_provided",
            model_name="user-provided",
            model_provider="user-provided",
            interface="taskeval",
        )
        assert config.model_name == "user-provided"
        assert config.model_provider == "user-provided"


@pytest.mark.unit
class TestStepEvalFailedQuestions:
    """Issue 117: StepEval tracks failed questions."""

    def test_failed_questions_defaults_empty(self):
        """failed_questions defaults to empty dict."""
        step = StepEval()
        assert step.failed_questions == {}

    def test_failed_questions_records_errors(self):
        """failed_questions stores error messages by question_id."""
        step = StepEval()
        step.failed_questions["q1"] = ["Some error"]
        step.failed_questions["q1"].append("Another error")
        assert len(step.failed_questions["q1"]) == 2
        assert step.failed_questions["q1"][0] == "Some error"
