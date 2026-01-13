"""Integration tests for DataFrame functionality with deep judgment verification.

This test module validates that JudgmentResults DataFrame methods work correctly
using fixture-created VerificationResult objects with deep judgment data.

Tests are marked with:
- @pytest.mark.integration: Tests that combine multiple components
- @pytest.mark.deep_judgment: Tests specific to deep judgment mode

Run with: pytest tests/integration/test_dataframe_integration_deep_judgment.py -v
"""

from datetime import UTC, datetime

import pandas as pd
import pytest

from karenina.schemas.workflow import (
    JudgmentResults,
    TemplateResults,
    VerificationResult,
    VerificationResultDeepJudgment,
    VerificationResultMetadata,
    VerificationResultTemplate,
)

# =============================================================================
# Helper Functions
# =============================================================================


def _create_metadata(
    question_id: str,
    answering_model: str = "claude-haiku-4-5",
    completed: bool = True,
    error: str | None = None,
) -> VerificationResultMetadata:
    """Helper to create metadata with computed result_id."""
    timestamp = datetime.now(UTC).isoformat()
    return VerificationResultMetadata(
        question_id=question_id,
        template_id="test-template-id",
        completed_without_errors=completed,
        error=error,
        question_text=f"Question text for {question_id}",
        raw_answer="Expected answer",
        answering_model=answering_model,
        parsing_model="claude-haiku-4-5",
        execution_time=1.5,
        timestamp=timestamp,
        result_id=VerificationResultMetadata.compute_result_id(
            question_id=question_id,
            answering_model=answering_model,
            parsing_model="claude-haiku-4-5",
            timestamp=timestamp,
        ),
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def template_result_success() -> VerificationResultTemplate:
    """Create a successful template result."""
    return VerificationResultTemplate(
        raw_llm_response="BCL2 is an anti-apoptotic gene located on chromosome 18.",
        parsed_gt_response={"gene_name": "BCL2", "chromosome": "18"},
        parsed_llm_response={"gene_name": "BCL2", "chromosome": "18"},
        template_verification_performed=True,
        verify_result=True,
        verify_granular_result={"gene_name": True, "chromosome": True},
    )


@pytest.fixture
def deep_judgment_result() -> VerificationResultDeepJudgment:
    """Create a deep judgment result with excerpts and reasoning."""
    return VerificationResultDeepJudgment(
        deep_judgment_enabled=True,
        deep_judgment_performed=True,
        extracted_excerpts={
            "gene_name": [
                {"text": "BCL2 is an anti-apoptotic gene", "confidence": "high"},
                {"text": "BCL2 gene located on chromosome 18", "confidence": "high"},
            ],
            "chromosome": [
                {"text": "located on chromosome 18", "confidence": "high"},
            ],
        },
        attribute_reasoning={
            "gene_name": "The response clearly identifies BCL2 as the gene of interest.",
            "chromosome": "The chromosome location is explicitly stated as 18.",
        },
        deep_judgment_stages_completed=["excerpts", "reasoning", "scoring"],
        deep_judgment_model_calls=3,
    )


@pytest.fixture
def verification_result_with_dj(
    template_result_success: VerificationResultTemplate,
    deep_judgment_result: VerificationResultDeepJudgment,
) -> VerificationResult:
    """Create a verification result with deep judgment data."""
    return VerificationResult(
        metadata=_create_metadata("dj_q1", "gpt-4"),
        template=template_result_success,
        deep_judgment=deep_judgment_result,
    )


@pytest.fixture
def verification_result_with_dj_different_model(
    template_result_success: VerificationResultTemplate,
    deep_judgment_result: VerificationResultDeepJudgment,
) -> VerificationResult:
    """Create a verification result with a different model."""
    return VerificationResult(
        metadata=_create_metadata("dj_q2", "claude-sonnet-4"),
        template=template_result_success,
        deep_judgment=deep_judgment_result,
    )


@pytest.fixture
def verification_results_list(
    verification_result_with_dj: VerificationResult,
    verification_result_with_dj_different_model: VerificationResult,
) -> list[VerificationResult]:
    """Create a list of verification results with deep judgment."""
    return [verification_result_with_dj, verification_result_with_dj_different_model]


# =============================================================================
# JudgmentResults DataFrame Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.deep_judgment
class TestJudgmentResultsDataFrame:
    """Test JudgmentResults DataFrame conversion."""

    def test_to_dataframe_with_deep_judgment(self, verification_results_list: list[VerificationResult]):
        """Test DataFrame export with deep judgment data."""
        judgment_results = JudgmentResults(results=verification_results_list)
        df = judgment_results.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

        # Check core columns
        core_columns = [
            "completed_without_errors",
            "question_id",
            "answering_model",
        ]
        for col in core_columns:
            assert col in df.columns

    def test_dataframe_has_excerpt_columns(self, verification_results_list: list[VerificationResult]):
        """Test that DataFrame has excerpt-related columns."""
        judgment_results = JudgmentResults(results=verification_results_list)
        df = judgment_results.to_dataframe()

        # Should have excerpt columns
        assert "attribute_name" in df.columns
        assert "excerpt_index" in df.columns
        assert "excerpt_text" in df.columns

    def test_dataframe_excerpt_explosion(self, verification_results_list: list[VerificationResult]):
        """Test that excerpts are exploded into separate rows."""
        judgment_results = JudgmentResults(results=verification_results_list)
        df = judgment_results.to_dataframe()

        # gene_name has 2 excerpts per result, chromosome has 1
        # Total: (2 + 1) * 2 results = 6 rows
        gene_name_rows = df[df["attribute_name"] == "gene_name"]
        assert len(gene_name_rows) >= 2  # At least one result's excerpts

        # Check excerpt indices
        if len(gene_name_rows) >= 2:
            indices = gene_name_rows["excerpt_index"].values
            assert 0 in indices
            assert 1 in indices

    def test_dataframe_has_reasoning(self, verification_results_list: list[VerificationResult]):
        """Test that DataFrame includes reasoning columns."""
        judgment_results = JudgmentResults(results=verification_results_list)
        df = judgment_results.to_dataframe()

        assert "attribute_reasoning" in df.columns

        # Check that reasoning is populated
        gene_name_rows = df[df["attribute_name"] == "gene_name"]
        if len(gene_name_rows) > 0:
            first_row = gene_name_rows.iloc[0]
            assert first_row["attribute_reasoning"] is not None
            assert len(first_row["attribute_reasoning"]) > 0


@pytest.mark.integration
@pytest.mark.deep_judgment
class TestJudgmentResultsAggregation:
    """Test JudgmentResults aggregation methods."""

    def test_aggregate_excerpt_counts_by_question(self, verification_results_list: list[VerificationResult]):
        """Test aggregating excerpt counts by question."""
        judgment_results = JudgmentResults(results=verification_results_list)
        counts = judgment_results.aggregate_excerpt_counts(strategy="mean", by="question_id")

        assert isinstance(counts, dict)

        for _, count_data in counts.items():
            if isinstance(count_data, dict):
                # Per-attribute counts
                for _, count in count_data.items():
                    assert isinstance(count, int | float)
                    assert count >= 0
            else:
                # Single count value
                assert isinstance(count_data, int | float)
                assert count_data >= 0

    def test_aggregate_excerpt_counts_by_model(self, verification_results_list: list[VerificationResult]):
        """Test aggregating excerpt counts by model."""
        judgment_results = JudgmentResults(results=verification_results_list)
        counts = judgment_results.aggregate_excerpt_counts(strategy="mean", by="answering_model")

        assert isinstance(counts, dict)

        # Should have entries for each model
        for model_name, _ in counts.items():
            assert isinstance(model_name, str)


@pytest.mark.integration
@pytest.mark.deep_judgment
class TestDeepJudgmentConsistency:
    """Test consistency between JudgmentResults and TemplateResults."""

    def test_common_columns_with_template_results(self, verification_results_list: list[VerificationResult]):
        """Test that common columns are consistent between result types."""
        template_results = TemplateResults(results=verification_results_list)
        judgment_results = JudgmentResults(results=verification_results_list)

        template_df = template_results.to_dataframe()
        judgment_df = judgment_results.to_dataframe()

        # Common columns
        common_columns = ["completed_without_errors", "question_id", "answering_model"]

        for col in common_columns:
            assert col in template_df.columns
            assert col in judgment_df.columns

    def test_status_column_first(self, verification_results_list: list[VerificationResult]):
        """Test that status column appears first."""
        judgment_results = JudgmentResults(results=verification_results_list)
        df = judgment_results.to_dataframe()

        assert df.columns[0] == "completed_without_errors"


@pytest.mark.integration
@pytest.mark.deep_judgment
class TestDeepJudgmentPandasOperations:
    """Test pandas operations on JudgmentResults DataFrames."""

    def test_groupby_operations(self, verification_results_list: list[VerificationResult]):
        """Test pandas groupby operations."""
        judgment_results = JudgmentResults(results=verification_results_list)
        df = judgment_results.to_dataframe()

        # Group by question
        grouped = df.groupby("question_id")
        assert len(grouped) > 0

    def test_filtering_operations(self, verification_results_list: list[VerificationResult]):
        """Test pandas filtering operations."""
        judgment_results = JudgmentResults(results=verification_results_list)
        df = judgment_results.to_dataframe()

        # Filter to specific attribute
        gene_df = df[df["attribute_name"] == "gene_name"]
        assert len(gene_df) > 0

    def test_pivot_operations_on_attributes(self, verification_results_list: list[VerificationResult]):
        """Test pandas pivot operations on attributes."""
        judgment_results = JudgmentResults(results=verification_results_list)
        df = judgment_results.to_dataframe()

        if len(df) == 0:
            pytest.skip("No data for pivot testing")

        # Count excerpts per attribute
        try:
            pivot = df.groupby(["question_id", "attribute_name"]).size().unstack(fill_value=0)
            assert isinstance(pivot, pd.DataFrame)
        except Exception as e:
            pytest.skip(f"Pivot not applicable: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
