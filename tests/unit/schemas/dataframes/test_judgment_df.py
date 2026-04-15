"""Failure/caveats column tests for ``JudgmentDataFrameBuilder``.

These tests lock in the new ``success`` + ``failure_*`` + ``caveats`` column
set that replaces the legacy ``completed_without_errors`` / ``error`` /
``failed_stage`` columns. See the 2026-04-15 failure-state-harmonization plan.
"""

from __future__ import annotations

import pytest

from karenina.schemas.dataframes.judgment import JudgmentDataFrameBuilder
from karenina.schemas.results.caveat import Caveat
from karenina.schemas.results.failure import Failure, FailureCategory, FailureGroup
from karenina.schemas.verification.result_components import (
    VerificationResultDeepJudgment,
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
        raw_llm_response="BCL2 is on chromosome 18.",
        parsed_gt_response={"gene_name": "BCL2"},
        parsed_llm_response={"gene_name": "BCL2"},
        verify_result=True,
    )


def _make_deep_judgment() -> VerificationResultDeepJudgment:
    return VerificationResultDeepJudgment(
        deep_judgment_mode="full",
        deep_judgment_performed=True,
        extracted_excerpts={
            "gene_name": [{"text": "BCL2 is anti-apoptotic", "confidence": "high"}],
        },
        attribute_reasoning={"gene_name": "The response identifies BCL2."},
        deep_judgment_stages_completed=["excerpts", "reasoning"],
        deep_judgment_model_calls=2,
    )


@pytest.mark.unit
class TestJudgmentDataFrameFailureColumns:
    """Covers ``_create_judgment_row`` (happy path with deep judgment data)."""

    def test_failure_columns_present(self):
        md = make_metadata(
            failure=Failure(
                category=FailureCategory.TIMEOUT,
                stage="generate_answer",
                reason="boom",
            ),
            caveats=[Caveat.RETRIES_USED],
        )
        result = make_result(metadata=md, template=_make_template(), deep_judgment=_make_deep_judgment())
        df = JudgmentDataFrameBuilder([result]).build_dataframe()

        for col in NEW_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"
        for legacy in LEGACY_COLUMNS:
            assert legacy not in df.columns, f"Legacy column still present: {legacy}"

    def test_failure_values_populated(self):
        md = make_metadata(
            failure=Failure(
                category=FailureCategory.DEEP_JUDGMENT,
                stage="deep_judgment_autofail",
                reason="dj failed",
            ),
            caveats=[Caveat.EMBEDDING_OVERRIDE],
        )
        result = make_result(metadata=md, template=_make_template(), deep_judgment=_make_deep_judgment())
        df = JudgmentDataFrameBuilder([result]).build_dataframe()
        row = df.iloc[0]

        assert bool(row["success"]) is False
        assert row["failure_category"] == FailureCategory.DEEP_JUDGMENT.value
        assert row["failure_group"] == FailureGroup.AUTOFAIL.value
        assert row["failure_stage"] == "deep_judgment_autofail"
        assert row["failure_reason"] == "dj failed"
        assert row["caveats"] == Caveat.EMBEDDING_OVERRIDE.value

    def test_pass_has_none_failure_fields(self):
        md = make_metadata(failure=None, caveats=[])
        result = make_result(metadata=md, template=_make_template(), deep_judgment=_make_deep_judgment())
        df = JudgmentDataFrameBuilder([result]).build_dataframe()
        row = df.iloc[0]

        assert bool(row["success"]) is True
        assert row["failure_category"] in (None, "")
        assert row["failure_group"] in (None, "")
        assert row["failure_stage"] in (None, "")
        assert row["failure_reason"] in (None, "")
        assert row["caveats"] == ""


@pytest.mark.unit
class TestJudgmentEmptyDataFrameFailureColumns:
    """Covers ``_create_empty_judgment_row`` (no deep judgment data path)."""

    def test_failure_columns_present_empty_row(self):
        md = make_metadata(
            failure=Failure(
                category=FailureCategory.UNEXPECTED_ERROR,
                stage="generate_answer",
                reason="crashed",
            ),
            caveats=[],
        )
        result = make_result(metadata=md, template=None, deep_judgment=None)
        df = JudgmentDataFrameBuilder([result]).build_dataframe()
        row = df.iloc[0]

        for col in NEW_COLUMNS:
            assert col in df.columns
        for legacy in LEGACY_COLUMNS:
            assert legacy not in df.columns

        assert bool(row["success"]) is False
        assert row["failure_category"] == FailureCategory.UNEXPECTED_ERROR.value
        assert row["failure_group"] == FailureGroup.SYSTEM.value
        assert row["failure_stage"] == "generate_answer"
        assert row["failure_reason"] == "crashed"
        assert row["caveats"] == ""
