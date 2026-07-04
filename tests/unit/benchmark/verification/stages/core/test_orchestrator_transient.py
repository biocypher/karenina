"""Tests for StageOrchestrator error category classification."""

from __future__ import annotations

import pytest

from karenina.benchmark.verification.stages.core.base import (
    BaseVerificationStage,
    VerificationContext,
)
from karenina.benchmark.verification.stages.core.orchestrator import StageOrchestrator
from karenina.schemas.config import ModelConfig
from karenina.utils.errors import ErrorCategory


def _make_context() -> VerificationContext:
    model = ModelConfig(id="test", model_name="test-model")
    return VerificationContext(
        question_id="q1",
        question_text="What?",
        raw_answer="Y",
        template_code="class Answer: pass",
        answering_model=model,
        parsing_model=model,
        template_id="tpl1",
    )


class _RaisingStage(BaseVerificationStage):
    """Test stage that raises a given exception."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    @property
    def name(self) -> str:
        return "raising_stage"

    def execute(self, context: VerificationContext) -> None:  # noqa: ARG002
        raise self._exc


class _FinalizeStub(BaseVerificationStage):
    """Stub finalize stage that always runs and stores a minimal result."""

    @property
    def name(self) -> str:
        return "FinalizeResultStage"

    def should_run(self, context: VerificationContext) -> bool:  # noqa: ARG002
        return True  # Always runs

    def execute(self, context: VerificationContext) -> None:
        from karenina.benchmark.verification.stages.core.base import ArtifactKeys
        from karenina.schemas.verification import VerificationResult
        from karenina.schemas.verification.model_identity import ModelIdentity
        from karenina.schemas.verification.result_components import (
            VerificationResultDeepJudgment,
            VerificationResultDeepJudgmentRubric,
            VerificationResultMetadata,
            VerificationResultRubric,
            VerificationResultTemplate,
        )

        identity = ModelIdentity(model_name="test", interface="openai")
        metadata = VerificationResultMetadata(
            question_id=context.question_id,
            template_id=context.template_id,
            completed_without_errors=context.completed_without_errors,
            error=context.error,
            error_category=context.error_category.value if context.error_category else None,
            warnings=list(context.warnings),
            question_text=context.question_text,
            answering=identity,
            parsing=identity,
            execution_time=0.0,
            timestamp="2026-01-01T00:00:00Z",
            result_id="abcdef1234567890",
        )
        vr = VerificationResult(
            metadata=metadata,
            template=VerificationResultTemplate(),
            rubric=VerificationResultRubric(),
            deep_judgment=VerificationResultDeepJudgment(),
            deep_judgment_rubric=VerificationResultDeepJudgmentRubric(),
        )
        context.set_artifact(ArtifactKeys.FINAL_RESULT, vr)


@pytest.mark.unit
class TestOrchestratorErrorCategoryClassification:
    def test_connection_error_marks_connection(self) -> None:
        ctx = _make_context()
        orchestrator = StageOrchestrator(stages=[_RaisingStage(ConnectionError("connection refused")), _FinalizeStub()])
        orchestrator.execute(ctx)
        assert ctx.error_category == ErrorCategory.CONNECTION

    def test_value_error_marks_permanent(self) -> None:
        ctx = _make_context()
        orchestrator = StageOrchestrator(stages=[_RaisingStage(ValueError("bad value")), _FinalizeStub()])
        orchestrator.execute(ctx)
        assert ctx.error_category == ErrorCategory.PERMANENT

    def test_timeout_error_marks_timeout(self) -> None:
        ctx = _make_context()
        orchestrator = StageOrchestrator(stages=[_RaisingStage(TimeoutError("request timed out")), _FinalizeStub()])
        orchestrator.execute(ctx)
        assert ctx.error_category == ErrorCategory.TIMEOUT
