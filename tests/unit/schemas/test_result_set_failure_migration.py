"""Tests covering VerificationResultSet reads of the unified failure field.

These tests exercise the read-sites migrated off the legacy
``completed_without_errors`` attribute onto ``metadata.failure is None``:
- ``VerificationResultSet.filter(completed_only=True)``
- ``_calculate_basic_counts`` (exposed through ``get_summary()``)
- ``_calculate_completion_by_combo`` (exposed through ``get_summary()``)
"""

from __future__ import annotations

import pytest

from karenina.schemas.results.failure import Failure, FailureCategory
from karenina.schemas.results.verification_result_set import VerificationResultSet
from tests.schemas._metadata_factory import make_metadata
from tests.unit.benchmark.core._result_factory import make_result


@pytest.mark.unit
class TestResultSetFailureMigration:
    """VerificationResultSet interprets metadata.failure for pass/fail counts."""

    def test_filter_completed_only_drops_results_with_failure(self) -> None:
        passed = make_result(metadata=make_metadata(question_id="q1", result_id="r1", failure=None))
        failed = make_result(
            metadata=make_metadata(
                question_id="q2",
                result_id="r2",
                failure=Failure(
                    category=FailureCategory.TIMEOUT,
                    stage="generate_answer",
                    reason="timed out",
                ),
            )
        )

        result_set = VerificationResultSet(results=[passed, failed])
        filtered = result_set.filter(completed_only=True)

        assert len(filtered) == 1
        assert filtered.results[0].metadata.result_id == "r1"

    def test_basic_counts_treat_failure_none_as_completed(self) -> None:
        passed = make_result(metadata=make_metadata(question_id="q1", result_id="r1", failure=None))
        failed = make_result(
            metadata=make_metadata(
                question_id="q2",
                result_id="r2",
                failure=Failure(
                    category=FailureCategory.CONTENT,
                    stage="verify_template",
                    reason="wrong answer",
                ),
            )
        )

        result_set = VerificationResultSet(results=[passed, failed])
        summary = result_set.get_summary()

        assert summary["num_results"] == 2
        assert summary["num_completed"] == 1

    def test_completion_by_combo_counts_failure_none_per_combo(self) -> None:
        ok = make_result(metadata=make_metadata(question_id="q1", result_id="r1", failure=None))
        bad = make_result(
            metadata=make_metadata(
                question_id="q2",
                result_id="r2",
                failure=Failure(
                    category=FailureCategory.UNEXPECTED_ERROR,
                    stage="generate_answer",
                    reason="boom",
                ),
            )
        )

        result_set = VerificationResultSet(results=[ok, bad])
        summary = result_set.get_summary()

        assert summary["completion_by_combo"]
        combo = next(iter(summary["completion_by_combo"].values()))
        assert combo["total"] == 2
        assert combo["completed"] == 1
