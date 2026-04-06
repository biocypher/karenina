"""Tests for FinalizeResultStage error_category and warnings propagation."""

from __future__ import annotations

import pytest

from karenina.benchmark.verification.stages.core.base import VerificationContext
from karenina.benchmark.verification.stages.pipeline.finalize_result import FinalizeResultStage
from karenina.schemas.config import ModelConfig
from karenina.utils.errors import ErrorCategory


def _make_context(*, category: ErrorCategory | None = None) -> VerificationContext:
    """Create a minimal VerificationContext with error and category state."""
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
    if category is not None:
        ctx.mark_error("test failure", category=category)
    else:
        ctx.mark_error("permanent failure")
    return ctx


@pytest.mark.unit
class TestFinalizeResultErrorCategory:
    def test_connection_category_copied_to_metadata(self) -> None:
        ctx = _make_context(category=ErrorCategory.CONNECTION)
        stage = FinalizeResultStage()
        stage.execute(ctx)

        from karenina.benchmark.verification.stages.core.base import ArtifactKeys

        vr = ctx.get_artifact(ArtifactKeys.FINAL_RESULT)
        assert vr is not None
        assert vr.metadata.error_category == "connection"

    def test_permanent_category_copied_to_metadata(self) -> None:
        ctx = _make_context(category=ErrorCategory.PERMANENT)
        stage = FinalizeResultStage()
        stage.execute(ctx)

        from karenina.benchmark.verification.stages.core.base import ArtifactKeys

        vr = ctx.get_artifact(ArtifactKeys.FINAL_RESULT)
        assert vr is not None
        assert vr.metadata.error_category == "permanent"

    def test_default_mark_error_copies_permanent_to_metadata(self) -> None:
        ctx = _make_context()  # defaults to PERMANENT
        stage = FinalizeResultStage()
        stage.execute(ctx)

        from karenina.benchmark.verification.stages.core.base import ArtifactKeys

        vr = ctx.get_artifact(ArtifactKeys.FINAL_RESULT)
        assert vr is not None
        assert vr.metadata.error_category == "permanent"

    def test_warnings_copied_to_metadata(self) -> None:
        ctx = _make_context(category=ErrorCategory.CONNECTION)
        ctx.add_warning("something happened")
        ctx.add_warning("another thing")
        stage = FinalizeResultStage()
        stage.execute(ctx)

        from karenina.benchmark.verification.stages.core.base import ArtifactKeys

        vr = ctx.get_artifact(ArtifactKeys.FINAL_RESULT)
        assert vr is not None
        assert vr.metadata.warnings == ["something happened", "another thing"]

    def test_no_error_category_none_in_metadata(self) -> None:
        """When no error occurred, error_category should be None."""
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
        stage = FinalizeResultStage()
        stage.execute(ctx)

        from karenina.benchmark.verification.stages.core.base import ArtifactKeys

        vr = ctx.get_artifact(ArtifactKeys.FINAL_RESULT)
        assert vr is not None
        assert vr.metadata.error_category is None
