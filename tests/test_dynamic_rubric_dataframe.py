"""Tests for dynamic rubric _skipped columns in the rubric DataFrame builder.

Validates that {trait_name}_skipped companion columns are correctly added
when VerificationResults contain dynamic rubric metadata (promoted and
skipped traits).

Run with: pytest tests/test_dynamic_rubric_dataframe.py -v
"""

import math

import pytest

from karenina.schemas.dataframes.rubric import RubricDataFrameBuilder
from karenina.schemas.results import RubricResults
from karenina.schemas.verification import (
    VerificationResult,
    VerificationResultRubric,
    VerificationResultTemplate,
)
from tests.integration.dataframe_helpers import create_metadata

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def template_result() -> VerificationResultTemplate:
    """Create a basic successful template result."""
    return VerificationResultTemplate(
        raw_llm_response="BCL2 is an anti-apoptotic protein.",
        parsed_gt_response={"target": "BCL2"},
        parsed_llm_response={"target": "BCL2"},
        template_verification_performed=True,
        verify_result=True,
    )


@pytest.fixture
def rubric_with_dynamic_promoted(template_result: VerificationResultTemplate) -> VerificationResult:
    """Result where the dynamic rubric promoted all traits (concept present)."""
    return VerificationResult(
        metadata=create_metadata("q_promoted"),
        template=template_result,
        rubric=VerificationResultRubric(
            rubric_evaluation_performed=True,
            rubric_evaluation_strategy="batch",
            llm_trait_scores={"safety": True, "conciseness": 4},
            dynamic_rubric_promoted_traits=["safety", "conciseness"],
        ),
    )


@pytest.fixture
def rubric_with_dynamic_skipped(template_result: VerificationResultTemplate) -> VerificationResult:
    """Result where the dynamic rubric skipped some traits (concept absent)."""
    return VerificationResult(
        metadata=create_metadata("q_skipped"),
        template=template_result,
        rubric=VerificationResultRubric(
            rubric_evaluation_performed=True,
            rubric_evaluation_strategy="batch",
            llm_trait_scores={"safety": True},
            dynamic_rubric_skipped_traits={"conciseness": "concept not present in response"},
            dynamic_rubric_promoted_traits=["safety"],
        ),
    )


@pytest.fixture
def rubric_without_dynamic(template_result: VerificationResultTemplate) -> VerificationResult:
    """Result with no dynamic rubric data (standard rubric evaluation)."""
    return VerificationResult(
        metadata=create_metadata("q_static"),
        template=template_result,
        rubric=VerificationResultRubric(
            rubric_evaluation_performed=True,
            rubric_evaluation_strategy="batch",
            llm_trait_scores={"safety": True, "conciseness": 3},
        ),
    )


@pytest.fixture
def result_no_rubric(template_result: VerificationResultTemplate) -> VerificationResult:
    """Result with no rubric at all."""
    return VerificationResult(
        metadata=create_metadata("q_no_rubric"),
        template=template_result,
    )


# =============================================================================
# Tests: promoted and skipped traits produce _skipped columns
# =============================================================================


@pytest.mark.unit
class TestDynamicRubricSkippedColumns:
    """Verify _skipped columns appear when dynamic rubric data is present."""

    def test_promoted_traits_get_false(
        self,
        rubric_with_dynamic_promoted: VerificationResult,
    ):
        """Promoted traits should have _skipped = False on every row."""
        results = RubricResults(results=[rubric_with_dynamic_promoted])
        df = results.to_dataframe(trait_type="all")

        assert "safety_skipped" in df.columns
        assert "conciseness_skipped" in df.columns

        # All rows belong to the promoted result, so all should be False
        assert (df["safety_skipped"] == False).all()  # noqa: E712
        assert (df["conciseness_skipped"] == False).all()  # noqa: E712

    def test_skipped_traits_get_true(
        self,
        rubric_with_dynamic_skipped: VerificationResult,
    ):
        """Skipped traits should have _skipped = True on every row."""
        results = RubricResults(results=[rubric_with_dynamic_skipped])
        df = results.to_dataframe(trait_type="all")

        assert "conciseness_skipped" in df.columns
        assert "safety_skipped" in df.columns

        # conciseness was skipped
        assert (df["conciseness_skipped"] == True).all()  # noqa: E712
        # safety was promoted
        assert (df["safety_skipped"] == False).all()  # noqa: E712

    def test_mixed_promoted_and_skipped(
        self,
        rubric_with_dynamic_promoted: VerificationResult,
        rubric_with_dynamic_skipped: VerificationResult,
    ):
        """Dataset with both promoted and skipped results across questions."""
        results = RubricResults(results=[rubric_with_dynamic_promoted, rubric_with_dynamic_skipped])
        df = results.to_dataframe(trait_type="all")

        assert "safety_skipped" in df.columns
        assert "conciseness_skipped" in df.columns

        # Rows from q_promoted: both traits promoted (False)
        promoted_rows = df[df["question_id"] == "q_promoted"]
        assert (promoted_rows["safety_skipped"] == False).all()  # noqa: E712
        assert (promoted_rows["conciseness_skipped"] == False).all()  # noqa: E712

        # Rows from q_skipped: safety promoted (False), conciseness skipped (True)
        skipped_rows = df[df["question_id"] == "q_skipped"]
        assert (skipped_rows["safety_skipped"] == False).all()  # noqa: E712
        assert (skipped_rows["conciseness_skipped"] == True).all()  # noqa: E712


# =============================================================================
# Tests: no dynamic rubric data produces no _skipped columns
# =============================================================================


@pytest.mark.unit
class TestNoDynamicRubricNoSkippedColumns:
    """When no result has dynamic rubric data, _skipped columns must not appear."""

    def test_standard_rubric_no_skipped_columns(
        self,
        rubric_without_dynamic: VerificationResult,
    ):
        """Standard rubric results should produce no _skipped columns."""
        results = RubricResults(results=[rubric_without_dynamic])
        df = results.to_dataframe(trait_type="all")

        skipped_cols = [c for c in df.columns if c.endswith("_skipped")]
        assert skipped_cols == [], f"Unexpected _skipped columns: {skipped_cols}"

    def test_no_rubric_no_skipped_columns(
        self,
        result_no_rubric: VerificationResult,
    ):
        """Results without any rubric should produce no _skipped columns."""
        results = RubricResults(results=[result_no_rubric])
        df = results.to_dataframe(trait_type="all")

        skipped_cols = [c for c in df.columns if c.endswith("_skipped")]
        assert skipped_cols == [], f"Unexpected _skipped columns: {skipped_cols}"

    def test_empty_results_no_skipped_columns(self):
        """Empty result list should produce no _skipped columns."""
        builder = RubricDataFrameBuilder(results=[], include_deep_judgment=False)
        df = builder.build_dataframe(trait_type="all")

        assert df.empty
        skipped_cols = [c for c in df.columns if c.endswith("_skipped")]
        assert skipped_cols == []


# =============================================================================
# Tests: mixed dynamic and non-dynamic results
# =============================================================================


@pytest.mark.unit
class TestMixedDynamicAndStaticResults:
    """When some results have dynamic rubric data and some do not,
    the _skipped columns should use NaN for results without dynamic data."""

    def test_nan_for_non_dynamic_results(
        self,
        rubric_with_dynamic_promoted: VerificationResult,
        rubric_without_dynamic: VerificationResult,
    ):
        """Results without dynamic rubric get NaN in _skipped columns."""
        results = RubricResults(results=[rubric_with_dynamic_promoted, rubric_without_dynamic])
        df = results.to_dataframe(trait_type="all")

        assert "safety_skipped" in df.columns
        assert "conciseness_skipped" in df.columns

        # Rows from q_promoted: both traits are False (promoted)
        promoted_rows = df[df["question_id"] == "q_promoted"]
        assert (promoted_rows["safety_skipped"] == False).all()  # noqa: E712
        assert (promoted_rows["conciseness_skipped"] == False).all()  # noqa: E712

        # Rows from q_static: should be NaN (no dynamic rubric on this result)
        static_rows = df[df["question_id"] == "q_static"]
        for _, row in static_rows.iterrows():
            assert _is_nan(row["safety_skipped"]), f"Expected NaN for safety_skipped, got {row['safety_skipped']}"
            assert _is_nan(row["conciseness_skipped"]), (
                f"Expected NaN for conciseness_skipped, got {row['conciseness_skipped']}"
            )

    def test_nan_for_no_rubric_results(
        self,
        rubric_with_dynamic_skipped: VerificationResult,
        result_no_rubric: VerificationResult,
    ):
        """Results without any rubric get NaN in _skipped columns."""
        results = RubricResults(results=[rubric_with_dynamic_skipped, result_no_rubric])
        df = results.to_dataframe(trait_type="all")

        # The no-rubric result should produce a single empty row with NaN _skipped
        no_rubric_rows = df[df["question_id"] == "q_no_rubric"]
        assert len(no_rubric_rows) == 1

        for _, row in no_rubric_rows.iterrows():
            assert _is_nan(row["safety_skipped"])
            assert _is_nan(row["conciseness_skipped"])

    def test_trait_type_filter_preserves_skipped_columns(
        self,
        rubric_with_dynamic_promoted: VerificationResult,
    ):
        """Filtering by trait_type still includes _skipped columns."""
        results = RubricResults(results=[rubric_with_dynamic_promoted])

        # Filter to llm_binary only (safety is boolean)
        df = results.to_dataframe(trait_type="llm_binary")
        assert len(df) == 1  # Only the boolean "safety" trait
        assert "safety_skipped" in df.columns
        assert "conciseness_skipped" in df.columns

    def test_skipped_columns_are_sorted(
        self,
        template_result: VerificationResultTemplate,
    ):
        """The _skipped columns should appear in sorted order by trait name."""
        result = VerificationResult(
            metadata=create_metadata("q_sorted"),
            template=template_result,
            rubric=VerificationResultRubric(
                rubric_evaluation_performed=True,
                llm_trait_scores={"z_trait": 3},
                dynamic_rubric_promoted_traits=["z_trait", "a_trait", "m_trait"],
                dynamic_rubric_skipped_traits={"b_trait": "absent"},
            ),
        )
        results = RubricResults(results=[result])
        df = results.to_dataframe(trait_type="all")

        skipped_cols = [c for c in df.columns if c.endswith("_skipped")]
        assert skipped_cols == ["a_trait_skipped", "b_trait_skipped", "m_trait_skipped", "z_trait_skipped"]


# =============================================================================
# Helpers
# =============================================================================


def _is_nan(value: object) -> bool:
    """Check if a value is NaN, handling non-float types safely."""
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
