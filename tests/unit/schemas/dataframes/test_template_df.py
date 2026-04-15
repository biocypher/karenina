"""Failure/caveats column tests for ``TemplateDataFrameBuilder``.

These tests lock in the new ``success`` + ``failure_*`` + ``caveats`` column
set that replaces the legacy ``completed_without_errors`` / ``error`` /
``failed_stage`` columns. See the 2026-04-15 failure-state-harmonization plan.
"""

from __future__ import annotations

import pytest

from karenina.schemas.dataframes.template import TemplateDataFrameBuilder
from karenina.schemas.results.caveat import Caveat
from karenina.schemas.results.failure import Failure, FailureCategory, FailureGroup
from karenina.schemas.verification.result_components import (
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
    """Build a minimal populated template."""
    return VerificationResultTemplate(
        raw_llm_response="resp",
        parsed_gt_response={"answer": "Paris"},
        parsed_llm_response={"answer": "Paris"},
        verify_result=True,
    )


@pytest.mark.unit
class TestTemplateFieldDataFrameFailureColumns:
    """Covers ``build_field_dataframe`` row creators (field + empty rows)."""

    def test_failure_columns_present_with_template(self):
        md = make_metadata(
            failure=Failure(
                category=FailureCategory.TIMEOUT,
                stage="generate_answer",
                reason="boom",
            ),
            caveats=[Caveat.RETRIES_USED],
        )
        result = make_result(metadata=md, template=_make_template())
        df = TemplateDataFrameBuilder([result]).build_field_dataframe()

        for col in NEW_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"
        for legacy in LEGACY_COLUMNS:
            assert legacy not in df.columns, f"Legacy column still present: {legacy}"

    def test_failure_values_populated_with_template(self):
        md = make_metadata(
            failure=Failure(
                category=FailureCategory.TIMEOUT,
                stage="generate_answer",
                reason="boom",
            ),
            caveats=[Caveat.RETRIES_USED],
        )
        result = make_result(metadata=md, template=_make_template())
        df = TemplateDataFrameBuilder([result]).build_field_dataframe()
        row = df.iloc[0]

        assert bool(row["success"]) is False
        assert row["failure_category"] == FailureCategory.TIMEOUT.value
        assert row["failure_group"] == FailureGroup.RETRY_EXHAUSTED.value
        assert row["failure_stage"] == "generate_answer"
        assert row["failure_reason"] == "boom"
        assert row["caveats"] == "retries_used"

    def test_pass_has_none_failure_fields_with_template(self):
        md = make_metadata(failure=None, caveats=[])
        result = make_result(metadata=md, template=_make_template())
        df = TemplateDataFrameBuilder([result]).build_field_dataframe()
        row = df.iloc[0]

        assert bool(row["success"]) is True
        assert row["failure_category"] in (None, "")
        assert row["failure_group"] in (None, "")
        assert row["failure_stage"] in (None, "")
        assert row["failure_reason"] in (None, "")
        assert row["caveats"] == ""

    def test_caveats_are_comma_joined(self):
        md = make_metadata(
            failure=None,
            caveats=[Caveat.PARTIAL_CONTENT, Caveat.EMBEDDING_OVERRIDE],
        )
        result = make_result(metadata=md, template=_make_template())
        df = TemplateDataFrameBuilder([result]).build_field_dataframe()
        row = df.iloc[0]

        parts = set(row["caveats"].split(","))
        assert parts == {Caveat.PARTIAL_CONTENT.value, Caveat.EMBEDDING_OVERRIDE.value}

    def test_failure_columns_present_empty_row(self):
        """Empty-row path (no template) must also carry the new columns."""
        md = make_metadata(
            failure=Failure(
                category=FailureCategory.UNEXPECTED_ERROR,
                stage="generate_answer",
                reason="crashed",
            ),
            caveats=[],
        )
        result = make_result(metadata=md, template=None)
        df = TemplateDataFrameBuilder([result]).build_field_dataframe()
        row = df.iloc[0]

        for col in NEW_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"
        for legacy in LEGACY_COLUMNS:
            assert legacy not in df.columns
        assert bool(row["success"]) is False
        assert row["failure_category"] == FailureCategory.UNEXPECTED_ERROR.value
        assert row["failure_group"] == FailureGroup.SYSTEM.value
        assert row["failure_stage"] == "generate_answer"


@pytest.mark.unit
class TestTemplateRegexDataFrameFailureColumns:
    """Covers ``build_regex_dataframe`` row construction."""

    def _make_regex_template(self) -> VerificationResultTemplate:
        return VerificationResultTemplate(
            raw_llm_response="matched",
            regex_validations_performed=True,
            regex_validation_results={"pattern_1": True},
            regex_validation_details={"pattern_1": {"pattern": "m.+", "match_start": 0}},
            regex_extraction_results={"pattern_1": "matched"},
        )

    def test_failure_columns_present(self):
        md = make_metadata(failure=None, caveats=[])
        result = make_result(metadata=md, template=self._make_regex_template())
        df = TemplateDataFrameBuilder([result]).build_regex_dataframe()

        for col in ("success", "failure_category", "failure_group", "failure_stage", "failure_reason", "caveats"):
            assert col in df.columns
        for legacy in LEGACY_COLUMNS:
            assert legacy not in df.columns

    def test_failure_values_populated(self):
        md = make_metadata(
            failure=Failure(
                category=FailureCategory.CONTENT,
                stage="verify_template",
                reason="mismatch",
            ),
            caveats=[Caveat.EMBEDDING_OVERRIDE],
        )
        result = make_result(metadata=md, template=self._make_regex_template())
        df = TemplateDataFrameBuilder([result]).build_regex_dataframe()
        row = df.iloc[0]

        assert bool(row["success"]) is False
        assert row["failure_category"] == FailureCategory.CONTENT.value
        assert row["failure_group"] == FailureGroup.CONTENT.value
        assert row["failure_stage"] == "verify_template"
        assert row["failure_reason"] == "mismatch"
        assert row["caveats"] == Caveat.EMBEDDING_OVERRIDE.value


@pytest.mark.unit
class TestTemplateUsageDataFrameFailureColumns:
    """Covers ``build_usage_dataframe`` row construction."""

    def _make_usage_template(self) -> VerificationResultTemplate:
        return VerificationResultTemplate(
            raw_llm_response="resp",
            usage_metadata={
                "answer_generation": {
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "total_tokens": 30,
                    "model": "gpt-4o",
                }
            },
        )

    def test_failure_columns_present(self):
        md = make_metadata(failure=None, caveats=[])
        result = make_result(metadata=md, template=self._make_usage_template())
        df = TemplateDataFrameBuilder([result]).build_usage_dataframe()

        for col in ("success", "failure_category", "failure_group", "failure_stage", "failure_reason", "caveats"):
            assert col in df.columns
        for legacy in LEGACY_COLUMNS:
            assert legacy not in df.columns

    def test_failure_values_populated(self):
        md = make_metadata(
            failure=Failure(
                category=FailureCategory.RATE_LIMIT,
                stage="generate_answer",
                reason="429",
            ),
            caveats=[],
        )
        result = make_result(metadata=md, template=self._make_usage_template())
        df = TemplateDataFrameBuilder([result]).build_usage_dataframe()
        row = df.iloc[0]

        assert bool(row["success"]) is False
        assert row["failure_category"] == FailureCategory.RATE_LIMIT.value
        assert row["failure_group"] == FailureGroup.RETRY_EXHAUSTED.value
        assert row["failure_stage"] == "generate_answer"
        assert row["failure_reason"] == "429"
        assert row["caveats"] == ""
