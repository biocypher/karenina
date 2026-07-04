"""Tests for VerificationContext error category and warnings."""

from __future__ import annotations

import pytest

from karenina.benchmark.verification.stages.core.base import VerificationContext
from karenina.schemas.config import ModelConfig
from karenina.utils.errors import ErrorCategory


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
class TestVerificationContextErrorCategory:
    def test_default_error_category_none(self) -> None:
        ctx = _make_context()
        assert ctx.error_category is None

    def test_mark_error_default_is_permanent(self) -> None:
        ctx = _make_context()
        ctx.mark_error("something broke")
        assert ctx.error_category == ErrorCategory.PERMANENT
        assert ctx.completed_without_errors is False

    def test_mark_error_with_connection_category(self) -> None:
        ctx = _make_context()
        ctx.mark_error("timeout", category=ErrorCategory.CONNECTION)
        assert ctx.error_category == ErrorCategory.CONNECTION
        assert ctx.completed_without_errors is False

    def test_mark_error_with_timeout_category(self) -> None:
        ctx = _make_context()
        ctx.mark_error("request timed out", category=ErrorCategory.TIMEOUT)
        assert ctx.error_category == ErrorCategory.TIMEOUT

    def test_mark_error_with_rate_limit_category(self) -> None:
        ctx = _make_context()
        ctx.mark_error("rate limited", category=ErrorCategory.RATE_LIMIT)
        assert ctx.error_category == ErrorCategory.RATE_LIMIT

    def test_mark_error_overwrites_previous_and_warns(self) -> None:
        """When mark_error is called twice, the second overwrites the first and a warning is added."""
        ctx = _make_context()
        ctx.mark_error("first error", category=ErrorCategory.CONNECTION)
        ctx.mark_error("second error", category=ErrorCategory.PERMANENT)
        assert ctx.error == "second error"
        assert ctx.error_category == ErrorCategory.PERMANENT
        assert len(ctx.warnings) == 1
        assert "Previous error overwritten" in ctx.warnings[0]
        assert "first error" in ctx.warnings[0]

    def test_mark_error_sets_error_and_completed(self) -> None:
        """mark_error sets error string and completed_without_errors=False."""
        ctx = _make_context()
        ctx.mark_error("test error")
        assert ctx.error == "test error"
        assert ctx.completed_without_errors is False


@pytest.mark.unit
class TestVerificationContextWarnings:
    def test_default_warnings_empty(self) -> None:
        ctx = _make_context()
        assert ctx.warnings == []

    def test_add_warning_appends(self) -> None:
        ctx = _make_context()
        ctx.add_warning("something happened")
        assert ctx.warnings == ["something happened"]

    def test_add_multiple_warnings(self) -> None:
        ctx = _make_context()
        ctx.add_warning("warn 1")
        ctx.add_warning("warn 2")
        assert ctx.warnings == ["warn 1", "warn 2"]

    def test_warnings_capped_at_50(self) -> None:
        """After 50 warnings, a truncation message is added and no more are accepted."""
        ctx = _make_context()
        for i in range(55):
            ctx.add_warning(f"warning {i}")
        # 50 real warnings + 1 truncation message = 51
        assert len(ctx.warnings) == 51
        assert ctx.warnings[50] == "(additional warnings truncated)"
        # Warnings 50-54 are not present (only the truncation message at index 50)
        assert "warning 50" not in ctx.warnings
