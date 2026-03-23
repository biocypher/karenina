"""Tests for TaskEval metadata coherence fixes (issues 024, 114, 115, 117, 160, 165, 166, 168, 179)."""

import pytest

from karenina.benchmark.task_eval import StepEval
from karenina.schemas.config import ModelConfig
from karenina.schemas.verification import VerificationResult
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.schemas.verification.result_components import (
    VerificationResultMetadata,
    VerificationResultRubric,
    VerificationResultTemplate,
)


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


def _make_metadata(question_id: str = "test_q") -> VerificationResultMetadata:
    """Helper to create test metadata."""
    return VerificationResultMetadata(
        question_id=question_id,
        template_id="test_t",
        completed_without_errors=True,
        question_text="test question",
        answering=ModelIdentity(interface="mock", model_name="mock"),
        parsing=ModelIdentity(interface="mock", model_name="mock"),
        execution_time=0.1,
        timestamp="2026-01-01T00:00:00",
        result_id="abcd1234abcd1234",
    )


@pytest.mark.unit
class TestSuccessRateRubricOnly:
    """Issue 024: success_rate should not be zero in rubric_only mode when traits pass."""

    def test_rubric_only_all_pass(self):
        """success_rate > 0 when all rubric traits pass and no template verification."""
        rubric = VerificationResultRubric(
            rubric_evaluation_performed=True,
            llm_trait_scores={"trait_a": True, "trait_b": True},
        )
        template = VerificationResultTemplate(
            template_verification_performed=False,
            verify_result=None,
        )
        result = VerificationResult(
            metadata=_make_metadata(),
            template=template,
            rubric=rubric,
        )
        step = StepEval(verification_results={"q1": [result]})
        stats = step.get_summary_stats()
        assert stats["success_rate"] == 100.0
        assert stats["traces_passed"] == 1

    def test_rubric_only_some_fail(self):
        """success_rate == 0 when rubric traits fail."""
        rubric = VerificationResultRubric(
            rubric_evaluation_performed=True,
            llm_trait_scores={"trait_a": True, "trait_b": False},
        )
        template = VerificationResultTemplate(
            template_verification_performed=False,
            verify_result=None,
        )
        result = VerificationResult(
            metadata=_make_metadata(),
            template=template,
            rubric=rubric,
        )
        step = StepEval(verification_results={"q1": [result]})
        stats = step.get_summary_stats()
        assert stats["success_rate"] == 0.0
        assert stats["traces_passed"] == 0

    def test_template_mode_unchanged(self):
        """Template-verified traces still use verify_result for pass counting."""
        template = VerificationResultTemplate(
            template_verification_performed=True,
            verify_result=True,
        )
        result = VerificationResult(
            metadata=_make_metadata(),
            template=template,
        )
        step = StepEval(verification_results={"q1": [result]})
        stats = step.get_summary_stats()
        assert stats["success_rate"] == 100.0
