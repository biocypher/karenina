"""Tests for ExportManager issues.

Covers:
- Issue 110: get_summary() returns question_count as float instead of int
"""

from unittest.mock import MagicMock

import pytest

from karenina.benchmark.core.exports import ExportManager


def _make_export_manager(question_count: int = 5) -> ExportManager:
    """Create an ExportManager with mocked dependencies."""
    base = MagicMock()
    base.name = "test-benchmark"
    base.version = "1.0"
    base.creator = "tester"
    base.created_at = "2026-01-01"
    base.modified_at = "2026-01-02"
    base.question_count = question_count
    base.finished_count = 3
    base.get_progress.return_value = 60.0
    base.is_complete = False
    base._questions_cache = {f"q{i}": {"question": f"Q{i}"} for i in range(question_count)}

    templates_manager = MagicMock()
    templates_manager.has_template.return_value = True

    rubrics_manager = MagicMock()
    rubrics_manager.get_global_rubric.return_value = None

    return ExportManager(base=base, templates_manager=templates_manager, rubrics_manager=rubrics_manager)


@pytest.mark.unit
class TestIssue110QuestionCountAsFloat:
    """Issue 110: get_summary() wraps question_count in float(), but it should be int."""

    def test_question_count_is_int(self) -> None:
        """question_count in get_summary() should be an int, not a float."""
        manager = _make_export_manager(question_count=5)
        summary = manager.get_summary()

        assert summary["question_count"] == 5
        assert isinstance(summary["question_count"], int), (
            f"question_count should be int, got {type(summary['question_count']).__name__}"
        )

    def test_question_count_zero_is_int(self) -> None:
        """Even zero question_count should be int, not float."""
        manager = _make_export_manager(question_count=0)
        summary = manager.get_summary()

        assert summary["question_count"] == 0
        assert isinstance(summary["question_count"], int)
