"""Unit tests for TemplateDataFrameBuilder column completeness and compare semantics.

Covers issues 125, 147, 149, 161, and 164: missing columns in _create_field_row
and _create_empty_row, plus _compare_values both-None semantics.
"""

import pytest

from karenina.schemas.dataframes.template import TemplateDataFrameBuilder
from karenina.schemas.results.failure import Failure, FailureCategory
from karenina.schemas.verification import (
    VerificationResult,
    VerificationResultTemplate,
)
from tests.integration.dataframe_helpers import create_metadata

# =============================================================================
# Helpers
# =============================================================================


def _make_result(
    question_id: str = "q001",
    template: VerificationResultTemplate | None = None,
    failed_stage: str | None = None,
    scenario_id: str | None = None,
    scenario_node: str | None = None,
    scenario_turn: int | None = None,
    scenario_path: list[str] | None = None,
) -> VerificationResult:
    """Build a VerificationResult with optional metadata overrides."""
    metadata = create_metadata(question_id)
    if failed_stage is not None:
        metadata.failure = Failure(
            category=FailureCategory.UNEXPECTED_ERROR,
            stage=failed_stage,
            reason="test failure",
        )
    metadata.scenario_id = scenario_id
    metadata.scenario_node = scenario_node
    metadata.scenario_turn = scenario_turn
    metadata.scenario_path = scenario_path
    return VerificationResult(
        metadata=metadata,
        template=template,
        rubric=None,
        deep_judgment=None,
    )


def _build_field_df(result: VerificationResult):
    """Build the field DataFrame from a single result."""
    builder = TemplateDataFrameBuilder([result])
    return builder.build_field_dataframe()


# =============================================================================
# Issue 147: failure_stage column (post failure-state-harmonization)
# =============================================================================


@pytest.mark.unit
class TestFailureStageColumn:
    """Issue 147 + failure-state-harmonization: row creators must expose failure_stage."""

    def test_field_row_contains_failure_stage(self):
        """The field DataFrame must contain a failure_stage column when template data exists."""
        template = VerificationResultTemplate(
            raw_llm_response="test",
            parsed_gt_response={"answer": "Paris"},
            parsed_llm_response={"answer": "Paris"},
            verify_result=True,
        )
        result = _make_result(template=template, failed_stage="abstention_check")
        df = _build_field_df(result)

        assert "failure_stage" in df.columns
        assert df["failure_stage"].iloc[0] == "abstention_check"

    def test_field_row_failure_stage_none_when_not_set(self):
        """failure_stage is None when no stage failed."""
        template = VerificationResultTemplate(
            raw_llm_response="test",
            parsed_gt_response={"answer": "Paris"},
            parsed_llm_response={"answer": "Paris"},
            verify_result=True,
        )
        result = _make_result(template=template, failed_stage=None)
        df = _build_field_df(result)

        assert "failure_stage" in df.columns
        assert df["failure_stage"].iloc[0] is None

    def test_empty_row_contains_failure_stage(self):
        """The empty row (no template) must also include failure_stage."""
        result = _make_result(template=None, failed_stage="generate_answer")
        df = _build_field_df(result)

        assert "failure_stage" in df.columns
        assert df["failure_stage"].iloc[0] == "generate_answer"


# =============================================================================
# Issue 149: verify_granular_result column
# =============================================================================


@pytest.mark.unit
class TestVerifyGranularResultColumn:
    """Issue 149: _create_field_row and _create_empty_row must include verify_granular_result."""

    def test_field_row_contains_verify_granular_result(self):
        """verify_granular_result should appear in the field DataFrame."""
        template = VerificationResultTemplate(
            raw_llm_response="test",
            parsed_gt_response={"answer": "Paris"},
            parsed_llm_response={"answer": "Paris"},
            verify_result=True,
            verify_granular_result={"answer": True},
        )
        result = _make_result(template=template)
        df = _build_field_df(result)

        assert "verify_granular_result" in df.columns
        assert df["verify_granular_result"].iloc[0] == {"answer": True}

    def test_field_row_verify_granular_result_none_when_absent(self):
        """verify_granular_result is None when not set on the template."""
        template = VerificationResultTemplate(
            raw_llm_response="test",
            parsed_gt_response={"answer": "Paris"},
            parsed_llm_response={"answer": "Paris"},
            verify_result=True,
            verify_granular_result=None,
        )
        result = _make_result(template=template)
        df = _build_field_df(result)

        assert "verify_granular_result" in df.columns
        assert df["verify_granular_result"].iloc[0] is None

    def test_empty_row_verify_granular_result_is_none(self):
        """The empty row (no template) must have verify_granular_result as None."""
        result = _make_result(template=None)
        df = _build_field_df(result)

        assert "verify_granular_result" in df.columns
        assert df["verify_granular_result"].iloc[0] is None


# =============================================================================
# Issue 161: sufficiency columns
# =============================================================================


@pytest.mark.unit
class TestSufficiencyColumns:
    """Issue 161: _create_field_row and _create_empty_row must include 4 sufficiency columns."""

    SUFFICIENCY_COLUMNS = [
        "sufficiency_check_performed",
        "sufficiency_detected",
        "sufficiency_override_applied",
        "sufficiency_reasoning",
    ]

    def test_field_row_contains_all_sufficiency_columns(self):
        """All 4 sufficiency columns must appear in the field DataFrame."""
        template = VerificationResultTemplate(
            raw_llm_response="test",
            parsed_gt_response={"answer": "Paris"},
            parsed_llm_response={"answer": "Paris"},
            verify_result=True,
            sufficiency_check_performed=True,
            sufficiency_detected=True,
            sufficiency_override_applied=False,
            sufficiency_reasoning="Response is sufficient.",
        )
        result = _make_result(template=template)
        df = _build_field_df(result)

        for col in self.SUFFICIENCY_COLUMNS:
            assert col in df.columns, f"Missing sufficiency column: {col}"

    def test_field_row_sufficiency_values_populated(self):
        """Sufficiency columns reflect the template values."""
        template = VerificationResultTemplate(
            raw_llm_response="test",
            parsed_gt_response={"answer": "Paris"},
            parsed_llm_response={"answer": "Paris"},
            verify_result=True,
            sufficiency_check_performed=True,
            sufficiency_detected=False,
            sufficiency_override_applied=True,
            sufficiency_reasoning="Insufficient detail provided.",
        )
        result = _make_result(template=template)
        df = _build_field_df(result)

        assert bool(df["sufficiency_check_performed"].iloc[0]) is True
        assert bool(df["sufficiency_detected"].iloc[0]) is False
        assert bool(df["sufficiency_override_applied"].iloc[0]) is True
        assert df["sufficiency_reasoning"].iloc[0] == "Insufficient detail provided."

    def test_field_row_sufficiency_defaults_when_not_performed(self):
        """When sufficiency check is not performed, defaults match template defaults."""
        template = VerificationResultTemplate(
            raw_llm_response="test",
            parsed_gt_response={"answer": "Paris"},
            parsed_llm_response={"answer": "Paris"},
            verify_result=True,
        )
        result = _make_result(template=template)
        df = _build_field_df(result)

        assert bool(df["sufficiency_check_performed"].iloc[0]) is False
        assert df["sufficiency_detected"].iloc[0] is None
        assert bool(df["sufficiency_override_applied"].iloc[0]) is False
        assert df["sufficiency_reasoning"].iloc[0] is None

    def test_empty_row_sufficiency_columns(self):
        """Empty row (no template) must have sufficiency columns with correct defaults."""
        result = _make_result(template=None)
        df = _build_field_df(result)

        for col in self.SUFFICIENCY_COLUMNS:
            assert col in df.columns, f"Missing sufficiency column in empty row: {col}"

        assert bool(df["sufficiency_check_performed"].iloc[0]) is False
        assert df["sufficiency_detected"].iloc[0] is None
        assert bool(df["sufficiency_override_applied"].iloc[0]) is False
        assert df["sufficiency_reasoning"].iloc[0] is None


# =============================================================================
# Issue 164: scenario columns
# =============================================================================


@pytest.mark.unit
class TestScenarioColumns:
    """Issue 164: _create_field_row and _create_empty_row must include 4 scenario columns."""

    SCENARIO_COLUMNS = [
        "scenario_id",
        "scenario_node",
        "scenario_turn",
        "scenario_path",
    ]

    def test_field_row_contains_scenario_columns(self):
        """Scenario columns must appear in the field DataFrame."""
        template = VerificationResultTemplate(
            raw_llm_response="test",
            parsed_gt_response={"answer": "Paris"},
            parsed_llm_response={"answer": "Paris"},
            verify_result=True,
        )
        result = _make_result(
            template=template,
            scenario_id="sc-001",
            scenario_node="greeting",
            scenario_turn=2,
            scenario_path=["greeting", "followup"],
        )
        df = _build_field_df(result)

        for col in self.SCENARIO_COLUMNS:
            assert col in df.columns, f"Missing scenario column: {col}"

    def test_field_row_scenario_values_populated(self):
        """Scenario columns must carry the correct values from metadata."""
        template = VerificationResultTemplate(
            raw_llm_response="test",
            parsed_gt_response={"answer": "Paris"},
            parsed_llm_response={"answer": "Paris"},
            verify_result=True,
        )
        result = _make_result(
            template=template,
            scenario_id="sc-001",
            scenario_node="greeting",
            scenario_turn=2,
            scenario_path=["greeting", "followup"],
        )
        df = _build_field_df(result)

        assert df["scenario_id"].iloc[0] == "sc-001"
        assert df["scenario_node"].iloc[0] == "greeting"
        assert df["scenario_turn"].iloc[0] == 2
        assert df["scenario_path"].iloc[0] == ["greeting", "followup"]

    def test_field_row_scenario_none_for_standalone(self):
        """Scenario columns are None for standalone (non-scenario) questions."""
        template = VerificationResultTemplate(
            raw_llm_response="test",
            parsed_gt_response={"answer": "Paris"},
            parsed_llm_response={"answer": "Paris"},
            verify_result=True,
        )
        result = _make_result(template=template)
        df = _build_field_df(result)

        for col in self.SCENARIO_COLUMNS:
            assert df[col].iloc[0] is None, f"Expected None for standalone {col}"

    def test_empty_row_scenario_columns(self):
        """Empty row (no template) must have scenario columns from metadata."""
        result = _make_result(
            template=None,
            scenario_id="sc-002",
            scenario_node="farewell",
            scenario_turn=5,
            scenario_path=["greeting", "main", "farewell"],
        )
        df = _build_field_df(result)

        for col in self.SCENARIO_COLUMNS:
            assert col in df.columns, f"Missing scenario column in empty row: {col}"

        assert df["scenario_id"].iloc[0] == "sc-002"
        assert df["scenario_node"].iloc[0] == "farewell"
        assert df["scenario_turn"].iloc[0] == 5
        assert df["scenario_path"].iloc[0] == ["greeting", "main", "farewell"]


# =============================================================================
# Issue 125: _compare_values both-None semantics
# =============================================================================


@pytest.mark.unit
class TestCompareValuesBothNone:
    """Issue 125: _compare_values(None, None) must return None, not True."""

    def test_both_none_returns_none(self):
        """When both values are None, _compare_values should return None (not comparable)."""
        builder = TemplateDataFrameBuilder([])
        result = builder._compare_values(None, None)
        assert result is None

    def test_one_none_returns_false(self):
        """When only one value is None, _compare_values should return False."""
        builder = TemplateDataFrameBuilder([])
        assert builder._compare_values(None, "x") is False
        assert builder._compare_values("x", None) is False

    def test_equal_values_returns_true(self):
        """When both values are equal non-None, _compare_values returns True."""
        builder = TemplateDataFrameBuilder([])
        assert builder._compare_values("Paris", "Paris") is True
        assert builder._compare_values(42, 42) is True
        assert builder._compare_values([1, 2], [1, 2]) is True

    def test_unequal_values_returns_false(self):
        """When both values are non-None but different, _compare_values returns False."""
        builder = TemplateDataFrameBuilder([])
        assert builder._compare_values("Paris", "Lyon") is False
        assert builder._compare_values(42, 99) is False

    def test_field_match_none_when_both_values_none(self):
        """In the DataFrame, field_match should be None when both gt and llm are None."""
        template = VerificationResultTemplate(
            raw_llm_response="test",
            parsed_gt_response={"answer": None},
            parsed_llm_response={"answer": None},
        )
        result = _make_result(template=template)
        df = _build_field_df(result)

        row = df[df["field_name"] == "answer"].iloc[0]
        assert row["field_match"] is None
