"""Failure/caveats column tests for ``RubricDataFrameBuilder``.

These tests lock in the new ``success`` + ``failure_*`` + ``caveats`` column
set that replaces the legacy ``completed_without_errors`` / ``error`` /
``failed_stage`` columns in all trait-row creators. See the 2026-04-15
failure-state-harmonization plan.
"""

from __future__ import annotations

import pytest

from karenina.schemas.dataframes.rubric import RubricDataFrameBuilder
from karenina.schemas.results.caveat import Caveat
from karenina.schemas.results.failure import Failure, FailureCategory, FailureGroup
from karenina.schemas.verification.result_components import (
    VerificationResultRubric,
    VerificationResultTemplate,
)
from tests.schemas._metadata_factory import make_metadata
from tests.unit.benchmark.core._result_factory import make_result

LEGACY_COLUMNS = ("completed_without_errors", "error", "error_category", "failed_stage")
NEW_COLUMNS = (
    "success",
    "failure_category",
    "failure_group",
    "failure_stage",
    "failure_reason",
    "caveats",
)


def _make_template() -> VerificationResultTemplate:
    return VerificationResultTemplate(
        raw_llm_response="resp",
        parsed_gt_response={"answer": "Paris"},
        parsed_llm_response={"answer": "Paris"},
        verify_result=True,
    )


def _make_rubric(**overrides) -> VerificationResultRubric:
    defaults = {
        "rubric_evaluation_performed": True,
        "rubric_evaluation_strategy": "batch",
        "llm_trait_scores": {"Clarity": 4},
    }
    defaults.update(overrides)
    return VerificationResultRubric(**defaults)


@pytest.mark.unit
class TestRubricDataFrameFailureColumns:
    """New failure/caveats columns must appear on all five trait-row creators."""

    def _failure(self) -> Failure:
        return Failure(
            category=FailureCategory.TIMEOUT,
            stage="rubric_evaluation",
            reason="rubric timeout",
        )

    def test_llm_trait_row_has_failure_columns(self):
        md = make_metadata(failure=self._failure(), caveats=[Caveat.RETRIES_USED])
        result = make_result(metadata=md, template=_make_template(), rubric=_make_rubric())
        df = RubricDataFrameBuilder([result]).build_dataframe(trait_type="llm")

        for col in NEW_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"
        for legacy in LEGACY_COLUMNS:
            assert legacy not in df.columns, f"Legacy column still present: {legacy}"

        row = df.iloc[0]
        assert bool(row["success"]) is False
        assert row["failure_category"] == FailureCategory.TIMEOUT.value
        assert row["failure_group"] == FailureGroup.RETRY_EXHAUSTED.value
        assert row["failure_stage"] == "rubric_evaluation"
        assert row["failure_reason"] == "rubric timeout"
        assert row["caveats"] == Caveat.RETRIES_USED.value

    def test_regex_trait_row_has_failure_columns(self):
        md = make_metadata(failure=None, caveats=[])
        rubric = _make_rubric(llm_trait_scores=None, regex_trait_scores={"HasCite": True})
        result = make_result(metadata=md, template=_make_template(), rubric=rubric)
        df = RubricDataFrameBuilder([result]).build_dataframe(trait_type="regex")

        for col in NEW_COLUMNS:
            assert col in df.columns
        for legacy in LEGACY_COLUMNS:
            assert legacy not in df.columns

        row = df.iloc[0]
        assert bool(row["success"]) is True
        assert row["failure_category"] in (None, "")
        assert row["caveats"] == ""

    def test_callable_trait_row_has_failure_columns(self):
        md = make_metadata(failure=None, caveats=[])
        rubric = _make_rubric(llm_trait_scores=None, callable_trait_scores={"LenCheck": True})
        result = make_result(metadata=md, template=_make_template(), rubric=rubric)
        df = RubricDataFrameBuilder([result]).build_dataframe(trait_type="callable")

        for col in NEW_COLUMNS:
            assert col in df.columns
        for legacy in LEGACY_COLUMNS:
            assert legacy not in df.columns

        row = df.iloc[0]
        assert bool(row["success"]) is True

    def test_metric_trait_row_has_failure_columns(self):
        md = make_metadata(
            failure=Failure(
                category=FailureCategory.CONTENT,
                stage="verify_template",
                reason="fail",
            ),
            caveats=[],
        )
        rubric = _make_rubric(
            llm_trait_scores=None,
            metric_trait_scores={"Entity": {"precision": 0.9, "recall": 0.8, "f1": 0.85}},
        )
        result = make_result(metadata=md, template=_make_template(), rubric=rubric)
        df = RubricDataFrameBuilder([result]).build_dataframe(trait_type="metric")

        for col in NEW_COLUMNS:
            assert col in df.columns
        for legacy in LEGACY_COLUMNS:
            assert legacy not in df.columns

        # All three metric rows should carry the same failure info
        assert (df["failure_category"] == FailureCategory.CONTENT.value).all()
        assert (df["failure_group"] == FailureGroup.CONTENT.value).all()
        assert (~df["success"].astype(bool)).all()

    def test_agentic_trait_row_has_failure_columns(self):
        md = make_metadata(failure=None, caveats=[Caveat.PARTIAL_CONTENT])
        rubric = _make_rubric(llm_trait_scores=None, agentic_trait_scores={"Quality": 4})
        result = make_result(metadata=md, template=_make_template(), rubric=rubric)
        df = RubricDataFrameBuilder([result]).build_dataframe(trait_type="agentic")

        for col in NEW_COLUMNS:
            assert col in df.columns
        for legacy in LEGACY_COLUMNS:
            assert legacy not in df.columns

        row = df.iloc[0]
        assert bool(row["success"]) is True
        assert row["caveats"] == Caveat.PARTIAL_CONTENT.value


@pytest.mark.unit
class TestRubricDataFrameColumnOrder:
    """The column-order helper must emit the new status columns instead of legacy."""

    def test_status_columns_order(self):
        md = make_metadata(failure=None, caveats=[])
        result = make_result(metadata=md, template=_make_template(), rubric=_make_rubric())
        df = RubricDataFrameBuilder([result]).build_dataframe()
        cols = list(df.columns)

        assert "success" in cols
        assert "failure_category" in cols
        assert "caveats" in cols
        # success should come before question identification
        assert cols.index("success") < cols.index("question_id")
        # Legacy entries removed
        for legacy in LEGACY_COLUMNS:
            assert legacy not in cols
