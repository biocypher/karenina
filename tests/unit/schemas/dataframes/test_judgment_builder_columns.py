"""Unit tests for JudgmentDataFrameBuilder column completeness and private attribute.

Covers issues 124, 147, and 164: missing columns in _create_judgment_row
and _create_empty_judgment_row, plus renaming self.results to self._results.
"""

import pytest

from karenina.schemas.dataframes.judgment import JudgmentDataFrameBuilder
from karenina.schemas.results.failure import Failure, FailureCategory
from karenina.schemas.verification import (
    VerificationResult,
    VerificationResultDeepJudgment,
    VerificationResultTemplate,
)
from tests.integration.dataframe_helpers import create_metadata

# =============================================================================
# Helpers
# =============================================================================


def _make_deep_judgment(**overrides) -> VerificationResultDeepJudgment:
    """Build a deep judgment result with excerpts (and optional overrides)."""
    defaults = {
        "deep_judgment_enabled": True,
        "deep_judgment_performed": True,
        "extracted_excerpts": {
            "gene_name": [
                {"text": "BCL2 is an anti-apoptotic gene", "confidence": "high"},
            ],
        },
        "attribute_reasoning": {
            "gene_name": "The response identifies BCL2.",
        },
        "deep_judgment_stages_completed": ["excerpts", "reasoning"],
        "deep_judgment_model_calls": 2,
    }
    defaults.update(overrides)
    return VerificationResultDeepJudgment(**defaults)


def _make_template() -> VerificationResultTemplate:
    """Build a minimal successful template result."""
    return VerificationResultTemplate(
        raw_llm_response="BCL2 is on chromosome 18.",
        parsed_gt_response={"gene_name": "BCL2"},
        parsed_llm_response={"gene_name": "BCL2"},
        template_verification_performed=True,
        verify_result=True,
    )


def _make_result(
    question_id: str = "q001",
    deep_judgment: VerificationResultDeepJudgment | None = None,
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
        template=template if template is not None else _make_template(),
        rubric=None,
        deep_judgment=deep_judgment,
    )


def _build_df(results: list[VerificationResult]):
    """Build the judgment DataFrame from a list of results."""
    builder = JudgmentDataFrameBuilder(results)
    return builder.build_dataframe()


# =============================================================================
# Issue 147: failed_stage column
# =============================================================================


@pytest.mark.unit
class TestFailedStageColumn:
    """Issue 147: _create_judgment_row and _create_empty_judgment_row must include failed_stage."""

    def test_judgment_row_contains_failed_stage(self):
        """The judgment DataFrame must contain a failed_stage column with deep judgment data."""
        result = _make_result(
            deep_judgment=_make_deep_judgment(),
            failed_stage="abstention_check",
        )
        df = _build_df([result])

        assert "failure_stage" in df.columns
        assert df["failure_stage"].iloc[0] == "abstention_check"

    def test_judgment_row_failed_stage_none_when_not_set(self):
        """failed_stage is None when no stage failed."""
        result = _make_result(
            deep_judgment=_make_deep_judgment(),
            failed_stage=None,
        )
        df = _build_df([result])

        assert "failure_stage" in df.columns
        assert df["failure_stage"].iloc[0] is None

    def test_empty_row_contains_failed_stage(self):
        """The empty row (no deep judgment) must also include failed_stage."""
        result = _make_result(
            deep_judgment=None,
            failed_stage="generate_answer",
        )
        df = _build_df([result])

        assert "failure_stage" in df.columns
        assert df["failure_stage"].iloc[0] == "generate_answer"


# =============================================================================
# Issue 164: scenario columns
# =============================================================================


@pytest.mark.unit
class TestScenarioColumns:
    """Issue 164: _create_judgment_row and _create_empty_judgment_row must include 4 scenario columns."""

    SCENARIO_COLUMNS = [
        "scenario_id",
        "scenario_node",
        "scenario_turn",
        "scenario_path",
    ]

    def test_judgment_row_contains_scenario_columns(self):
        """Scenario columns must appear in the judgment DataFrame with deep judgment data."""
        result = _make_result(
            deep_judgment=_make_deep_judgment(),
            scenario_id="sc-001",
            scenario_node="greeting",
            scenario_turn=2,
            scenario_path=["greeting", "followup"],
        )
        df = _build_df([result])

        for col in self.SCENARIO_COLUMNS:
            assert col in df.columns, f"Missing scenario column: {col}"

    def test_judgment_row_scenario_values_populated(self):
        """Scenario columns must carry the correct values from metadata."""
        result = _make_result(
            deep_judgment=_make_deep_judgment(),
            scenario_id="sc-001",
            scenario_node="greeting",
            scenario_turn=2,
            scenario_path=["greeting", "followup"],
        )
        df = _build_df([result])

        assert df["scenario_id"].iloc[0] == "sc-001"
        assert df["scenario_node"].iloc[0] == "greeting"
        assert df["scenario_turn"].iloc[0] == 2
        assert df["scenario_path"].iloc[0] == ["greeting", "followup"]

    def test_judgment_row_scenario_none_for_standalone(self):
        """Scenario columns are None for standalone (non-scenario) questions."""
        result = _make_result(deep_judgment=_make_deep_judgment())
        df = _build_df([result])

        for col in self.SCENARIO_COLUMNS:
            assert df[col].iloc[0] is None, f"Expected None for standalone {col}"

    def test_empty_row_scenario_columns(self):
        """Empty row (no deep judgment) must have scenario columns from metadata."""
        result = _make_result(
            deep_judgment=None,
            scenario_id="sc-002",
            scenario_node="farewell",
            scenario_turn=5,
            scenario_path=["greeting", "main", "farewell"],
        )
        df = _build_df([result])

        for col in self.SCENARIO_COLUMNS:
            assert col in df.columns, f"Missing scenario column in empty row: {col}"

        assert df["scenario_id"].iloc[0] == "sc-002"
        assert df["scenario_node"].iloc[0] == "farewell"
        assert df["scenario_turn"].iloc[0] == 5
        assert df["scenario_path"].iloc[0] == ["greeting", "main", "farewell"]


# =============================================================================
# Issue 124: private _results attribute
# =============================================================================


@pytest.mark.unit
class TestPrivateResultsAttribute:
    """Issue 124: JudgmentDataFrameBuilder must use self._results, not self.results."""

    def test_no_public_results_attribute(self):
        """JudgmentDataFrameBuilder should NOT have a public .results attribute."""
        builder = JudgmentDataFrameBuilder([])
        assert not hasattr(builder, "results"), (
            "JudgmentDataFrameBuilder should not expose a public 'results' attribute"
        )

    def test_private_results_holds_data(self):
        """JudgmentDataFrameBuilder._results must hold the data passed to __init__."""
        result = _make_result(deep_judgment=_make_deep_judgment())
        builder = JudgmentDataFrameBuilder([result])
        assert hasattr(builder, "_results")
        assert len(builder._results) == 1
        assert builder._results[0] is result

    def test_build_dataframe_uses_private_results(self):
        """build_dataframe() must work correctly using the private _results attribute."""
        result = _make_result(deep_judgment=_make_deep_judgment())
        builder = JudgmentDataFrameBuilder([result])
        df = builder.build_dataframe()

        assert len(df) > 0
        assert df["question_id"].iloc[0] == "q001"
