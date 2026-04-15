"""Tests for FinalizeResultStage failure classification and warnings propagation.

The finalizer delegates Failure attribution to :func:`classify_failure` in
:mod:`karenina.benchmark.verification.failure_classifier`. When a context is
marked with ``mark_error`` but no retry counts are populated (i.e. retry
tracking was not active), the classifier falls through to the unexpected-error
catchall regardless of the ErrorCategory used to call ``mark_error``.
"""

from __future__ import annotations

import pytest

from karenina.benchmark.verification.stages.core.base import ArtifactKeys, VerificationContext
from karenina.benchmark.verification.stages.pipeline.finalize_result import FinalizeResultStage
from karenina.schemas.config import ModelConfig
from karenina.schemas.results.failure import FailureCategory, FailureGroup
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
class TestFinalizeResultFailureClassification:
    def test_connection_without_retry_counts_classifies_as_unexpected(self) -> None:
        """Without retry counts, connection errors hit the unexpected-error catchall.

        The classifier only maps ``ErrorCategory.CONNECTION`` to
        ``FailureCategory.CONNECTION`` when retries were attempted and the
        budget was exhausted (``RETRY_COUNTS`` artifact populated).
        """
        ctx = _make_context(category=ErrorCategory.CONNECTION)
        stage = FinalizeResultStage()
        stage.execute(ctx)

        vr = ctx.get_artifact(ArtifactKeys.FINAL_RESULT)
        assert vr is not None
        assert vr.metadata.failure is not None
        assert vr.metadata.failure.category is FailureCategory.UNEXPECTED_ERROR
        assert vr.metadata.failure.group is FailureGroup.SYSTEM

    def test_connection_with_retry_exhausted_classifies_as_connection(self) -> None:
        """With RETRY_COUNTS showing exhausted budget, connection is preserved."""
        ctx = _make_context(category=ErrorCategory.CONNECTION)
        ctx.set_artifact(
            ArtifactKeys.RETRY_COUNTS,
            {"connection": {"used": 3, "budget": 3}},
        )
        stage = FinalizeResultStage()
        stage.execute(ctx)

        vr = ctx.get_artifact(ArtifactKeys.FINAL_RESULT)
        assert vr is not None
        assert vr.metadata.failure is not None
        assert vr.metadata.failure.category is FailureCategory.CONNECTION
        assert vr.metadata.failure.group is FailureGroup.RETRY_EXHAUSTED
        assert vr.metadata.retry_counts == {"connection": {"used": 3, "budget": 3}}

    def test_permanent_category_classifies_as_unexpected(self) -> None:
        """ErrorCategory.PERMANENT has no matching leaf, falls to catchall."""
        ctx = _make_context(category=ErrorCategory.PERMANENT)
        stage = FinalizeResultStage()
        stage.execute(ctx)

        vr = ctx.get_artifact(ArtifactKeys.FINAL_RESULT)
        assert vr is not None
        assert vr.metadata.failure is not None
        assert vr.metadata.failure.category is FailureCategory.UNEXPECTED_ERROR

    def test_default_mark_error_classifies_as_unexpected(self) -> None:
        """Default mark_error (PERMANENT) falls to catchall."""
        ctx = _make_context()  # defaults to PERMANENT
        stage = FinalizeResultStage()
        stage.execute(ctx)

        vr = ctx.get_artifact(ArtifactKeys.FINAL_RESULT)
        assert vr is not None
        assert vr.metadata.failure is not None
        assert vr.metadata.failure.category is FailureCategory.UNEXPECTED_ERROR

    def test_warnings_copied_to_metadata(self) -> None:
        ctx = _make_context(category=ErrorCategory.CONNECTION)
        ctx.add_warning("something happened")
        ctx.add_warning("another thing")
        stage = FinalizeResultStage()
        stage.execute(ctx)

        vr = ctx.get_artifact(ArtifactKeys.FINAL_RESULT)
        assert vr is not None
        assert vr.metadata.warnings == ["something happened", "another thing"]

    def test_no_error_failure_is_none(self) -> None:
        """When no error occurred, failure should be None (pipeline pass)."""
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

        vr = ctx.get_artifact(ArtifactKeys.FINAL_RESULT)
        assert vr is not None
        assert vr.metadata.failure is None
