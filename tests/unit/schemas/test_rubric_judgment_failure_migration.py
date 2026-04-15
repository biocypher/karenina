"""Tests covering RubricJudgmentResults DataFrame failure columns.

These tests exercise the status columns written by ``to_dataframe`` that were
migrated off the legacy ``metadata.completed_without_errors`` / ``metadata.error``
onto the unified ``metadata.failure`` schema.
"""

from __future__ import annotations

import pytest

from karenina.schemas.results.failure import Failure, FailureCategory
from karenina.schemas.results.rubric_judgment import RubricJudgmentResults
from karenina.schemas.verification.result_components import (
    VerificationResultDeepJudgmentRubric,
)
from tests.schemas._metadata_factory import make_metadata
from tests.unit.benchmark.core._result_factory import make_result


def _result_with_deep_judgment_rubric(**metadata_overrides):
    """Build a VerificationResult that has a deep_judgment_rubric trait row."""
    deep_judgment_rubric = VerificationResultDeepJudgmentRubric(
        deep_judgment_rubric_performed=True,
        deep_judgment_rubric_scores={"clarity": 4},
        rubric_trait_reasoning={"clarity": "Response was clear."},
        extracted_rubric_excerpts={"clarity": []},
        trait_metadata={
            "clarity": {
                "model_calls": 2,
                "excerpt_retry_count": 0,
                "stages_completed": ["reasoning"],
                "excerpt_validation_failed": False,
            }
        },
    )
    return make_result(
        metadata=make_metadata(**metadata_overrides),
        deep_judgment_rubric=deep_judgment_rubric,
    )


@pytest.mark.unit
class TestRubricJudgmentFailureMigration:
    """to_dataframe surfaces success/failure_* columns derived from Failure."""

    def test_dataframe_pass_row_has_success_true_and_no_failure(self) -> None:
        result = _result_with_deep_judgment_rubric(failure=None, caveats=[])
        judgments = RubricJudgmentResults(results=[result])

        df = judgments.to_dataframe()

        assert len(df) == 1
        row = df.iloc[0]
        assert bool(row["success"]) is True
        assert row["failure_category"] is None
        assert row["failure_group"] is None
        assert row["failure_stage"] is None
        assert row["failure_reason"] is None
        assert row["caveats"] == ""

    def test_dataframe_fail_row_surfaces_failure_fields(self) -> None:
        failure = Failure(
            category=FailureCategory.TIMEOUT,
            stage="generate_answer",
            reason="gone",
            details={"error_message": "timed out after 120s"},
        )
        result = _result_with_deep_judgment_rubric(failure=failure, caveats=[])
        judgments = RubricJudgmentResults(results=[result])

        df = judgments.to_dataframe()

        assert len(df) == 1
        row = df.iloc[0]
        assert bool(row["success"]) is False
        assert row["failure_category"] == "timeout"
        assert row["failure_group"] == "retry"
        assert row["failure_stage"] == "generate_answer"
        assert row["failure_reason"] == "gone"
