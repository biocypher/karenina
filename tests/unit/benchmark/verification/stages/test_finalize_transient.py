"""Tests for FinalizeResultStage transient flag propagation."""

from __future__ import annotations

import pytest

from karenina.benchmark.verification.stages.core.base import VerificationContext
from karenina.benchmark.verification.stages.pipeline.finalize_result import FinalizeResultStage
from karenina.schemas.config import ModelConfig


def _make_context(*, is_transient: bool = False) -> VerificationContext:
    """Create a minimal VerificationContext with error and transient state."""
    model = ModelConfig(id="test", model_name="test-model")
    ctx = VerificationContext(
        question_id="q1",
        question_text="What?",
        raw_answer="Y",
        template_code="class Answer: pass",
        answering_model=model,
        parsing_model=model,
        template_id="tpl1",
    )
    if is_transient:
        ctx.mark_error("connection timeout", transient=True)
    else:
        ctx.mark_error("permanent failure")
    return ctx


@pytest.mark.unit
class TestFinalizeResultTransientFlag:
    def test_transient_flag_copied_to_metadata(self) -> None:
        ctx = _make_context(is_transient=True)
        stage = FinalizeResultStage()
        stage.execute(ctx)

        from karenina.benchmark.verification.stages.core.base import ArtifactKeys

        vr = ctx.get_artifact(ArtifactKeys.FINAL_RESULT)
        assert vr is not None
        assert vr.metadata.is_transient_error is True

    def test_non_transient_flag_copied_to_metadata(self) -> None:
        ctx = _make_context(is_transient=False)
        stage = FinalizeResultStage()
        stage.execute(ctx)

        from karenina.benchmark.verification.stages.core.base import ArtifactKeys

        vr = ctx.get_artifact(ArtifactKeys.FINAL_RESULT)
        assert vr is not None
        assert vr.metadata.is_transient_error is False
