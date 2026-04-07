"""Tests for VerificationContext transient error flag."""

from __future__ import annotations

import pytest

from karenina.benchmark.verification.stages.core.base import VerificationContext
from karenina.schemas.config import ModelConfig


def _make_context() -> VerificationContext:
    """Create a minimal VerificationContext for testing."""
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


@pytest.mark.unit
class TestVerificationContextTransientFlag:
    def test_default_is_transient_false(self) -> None:
        ctx = _make_context()
        assert ctx.is_transient_error is False

    def test_mark_error_default_not_transient(self) -> None:
        ctx = _make_context()
        ctx.mark_error("something broke")
        assert ctx.is_transient_error is False
        assert ctx.completed_without_errors is False

    def test_mark_error_transient_true(self) -> None:
        ctx = _make_context()
        ctx.mark_error("timeout", transient=True)
        assert ctx.is_transient_error is True
        assert ctx.completed_without_errors is False

    def test_mark_error_transient_sticky(self) -> None:
        """Once is_transient_error is True, a subsequent non-transient mark_error keeps it True."""
        ctx = _make_context()
        ctx.mark_error("connection error", transient=True)
        assert ctx.is_transient_error is True

        ctx.mark_error("permanent error", transient=False)
        assert ctx.is_transient_error is True

    def test_mark_error_sets_error_and_completed(self) -> None:
        """Existing behavior: mark_error sets error string and completed_without_errors=False."""
        ctx = _make_context()
        ctx.mark_error("test error")
        assert ctx.error == "test error"
        assert ctx.completed_without_errors is False
