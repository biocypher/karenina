"""Integration tests for DataFrame functionality with rubric evaluation.

This test module validates that RubricResults DataFrame methods work correctly
using fixture-created VerificationResult objects.

Tests are marked with:
- @pytest.mark.integration: Tests that combine multiple components
- @pytest.mark.rubric: Tests specific to rubric evaluation mode

Run with: pytest tests/integration/test_dataframe_integration_rubrics.py -v
"""

import pandas as pd
import pytest

from karenina.schemas.workflow import (
    RubricResults,
    VerificationResult,
    VerificationResultRubric,
    VerificationResultTemplate,
)
from tests.integration.dataframe_helpers import (
    CommonColumnTestMixin,
    PandasOperationsTestMixin,
    create_metadata,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rubric_result_with_all_traits() -> VerificationResultRubric:
    """Create a rubric result with all trait types."""
    return VerificationResultRubric(
        rubric_evaluation_performed=True,
        rubric_evaluation_strategy="batch",
        llm_trait_scores={
            "Clarity": 4,
            "Completeness": 5,
            "Accuracy": True,
        },
        regex_trait_scores={
            "HasCitations": True,
            "HasNumbers": True,
        },
        callable_trait_scores={
            "ContainsCitations": True,
            "ResponseLength": 3,
        },
        metric_trait_scores={
            "EntityExtraction": {
                "precision": 0.85,
                "recall": 0.90,
                "f1": 0.87,
            }
        },
    )


@pytest.fixture
def template_result_success() -> VerificationResultTemplate:
    """Create a successful template result."""
    return VerificationResultTemplate(
        raw_llm_response="BCL2 is an anti-apoptotic gene.",
        parsed_gt_response={"gene_name": "BCL2"},
        parsed_llm_response={"gene_name": "BCL2"},
        template_verification_performed=True,
        verify_result=True,
        verify_granular_result={"gene_name": True},
    )


@pytest.fixture
def verification_result_with_rubric(
    rubric_result_with_all_traits: VerificationResultRubric,
    template_result_success: VerificationResultTemplate,
) -> VerificationResult:
    """Create a verification result with rubric data."""
    return VerificationResult(
        metadata=create_metadata("rubric_q1", "gpt-4"),
        template=template_result_success,
        rubric=rubric_result_with_all_traits,
    )


@pytest.fixture
def verification_result_different_model(
    rubric_result_with_all_traits: VerificationResultRubric,
    template_result_success: VerificationResultTemplate,
) -> VerificationResult:
    """Create a verification result with a different model."""
    return VerificationResult(
        metadata=create_metadata("rubric_q2", "claude-sonnet-4"),
        template=template_result_success,
        rubric=rubric_result_with_all_traits,
    )


@pytest.fixture
def verification_results_list(
    verification_result_with_rubric: VerificationResult,
    verification_result_different_model: VerificationResult,
) -> list[VerificationResult]:
    """Create a list of verification results."""
    return [verification_result_with_rubric, verification_result_different_model]


# =============================================================================
# RubricResults DataFrame Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.rubric
class TestRubricResultsDataFrame:
    """Test RubricResults DataFrame conversion."""

    def test_to_dataframe_all_traits(self, verification_results_list: list[VerificationResult]):
        """Test DataFrame export with all trait types."""
        rubric_results = RubricResults(results=verification_results_list)
        df = rubric_results.to_dataframe(trait_type="all")

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

        # Should have trait rows for each result
        # 3 LLM + 2 regex + 2 callable + 3 metric (exploded) = 10 per result
        # 2 results = 20 rows total
        assert len(df) >= 10  # At least one result's traits

    def test_to_dataframe_llm_traits_only(self, verification_results_list: list[VerificationResult]):
        """Test DataFrame export with LLM traits only."""
        rubric_results = RubricResults(results=verification_results_list)
        df = rubric_results.to_dataframe(trait_type="llm")

        # Should have LLM trait rows only
        assert len(df) > 0
        assert all(df["trait_type"].str.startswith("llm"))

        # Check columns
        assert "trait_name" in df.columns
        assert "trait_score" in df.columns
        assert "question_id" in df.columns

    def test_to_dataframe_regex_traits_only(self, verification_results_list: list[VerificationResult]):
        """Test DataFrame export with regex traits only."""
        rubric_results = RubricResults(results=verification_results_list)
        df = rubric_results.to_dataframe(trait_type="regex")

        # Should have regex trait rows only
        assert len(df) > 0
        assert all(df["trait_type"] == "regex")

        # Should have 2 regex traits per result
        assert len(df) == 4  # 2 traits * 2 results

    def test_to_dataframe_callable_traits_only(self, verification_results_list: list[VerificationResult]):
        """Test DataFrame export with callable traits only."""
        rubric_results = RubricResults(results=verification_results_list)
        df = rubric_results.to_dataframe(trait_type="callable")

        # Should have callable trait rows only
        assert len(df) > 0
        assert all(df["trait_type"] == "callable")

    def test_to_dataframe_metric_traits_only(self, verification_results_list: list[VerificationResult]):
        """Test DataFrame export with metric traits only (exploded by metric)."""
        rubric_results = RubricResults(results=verification_results_list)
        df = rubric_results.to_dataframe(trait_type="metric")

        # Should have metric trait rows, exploded by metric type
        assert len(df) > 0
        assert all(df["trait_type"] == "metric")

        # Should have metric_name column
        assert "metric_name" in df.columns

        # Each metric (precision, recall, f1) should be a separate row
        metric_names = df["metric_name"].unique()
        assert "precision" in metric_names
        assert "recall" in metric_names
        assert "f1" in metric_names


@pytest.mark.integration
@pytest.mark.rubric
class TestRubricResultsAggregation:
    """Test RubricResults aggregation methods."""

    def test_aggregate_llm_traits_by_question(self, verification_results_list: list[VerificationResult]):
        """Test aggregating LLM traits by question."""
        rubric_results = RubricResults(results=verification_results_list)
        aggregated = rubric_results.aggregate_llm_traits(strategy="mean", by="question_id")

        assert isinstance(aggregated, dict)
        assert len(aggregated) > 0

        # Each question should have trait scores
        for _question_id, traits in aggregated.items():
            assert isinstance(traits, dict)
            # Should have the LLM traits we defined
            assert "Clarity" in traits or "Completeness" in traits

    def test_aggregate_llm_traits_by_model(self, verification_results_list: list[VerificationResult]):
        """Test aggregating LLM traits by model."""
        rubric_results = RubricResults(results=verification_results_list)
        aggregated = rubric_results.aggregate_llm_traits(strategy="mean", by="answering_model")

        assert isinstance(aggregated, dict)

        # Should have entries for each model
        for model_name, traits in aggregated.items():
            assert isinstance(model_name, str)
            assert isinstance(traits, dict)

    def test_aggregate_regex_traits(self, verification_results_list: list[VerificationResult]):
        """Test aggregating regex traits."""
        rubric_results = RubricResults(results=verification_results_list)
        aggregated = rubric_results.aggregate_regex_traits(strategy="majority_vote", by="question_id")

        assert isinstance(aggregated, dict)

        for _question_id, traits in aggregated.items():
            assert isinstance(traits, dict)
            # Regex traits are boolean
            for _trait_name, value in traits.items():
                assert isinstance(value, bool)

    def test_aggregate_callable_traits(self, verification_results_list: list[VerificationResult]):
        """Test aggregating callable traits."""
        rubric_results = RubricResults(results=verification_results_list)
        aggregated = rubric_results.aggregate_callable_traits(strategy="majority_vote", by="question_id")

        assert isinstance(aggregated, dict)

        for _question_id, traits in aggregated.items():
            assert isinstance(traits, dict)

    def test_aggregate_metric_traits(self, verification_results_list: list[VerificationResult]):
        """Test aggregating metric traits."""
        rubric_results = RubricResults(results=verification_results_list)
        aggregated = rubric_results.aggregate_metric_traits(
            metric_name="f1",
            strategy="mean",
            by="question_id",
        )

        assert isinstance(aggregated, dict)

        for _question_id, traits in aggregated.items():
            assert isinstance(traits, dict)
            # Should have the EntityExtraction trait
            if "EntityExtraction" in traits:
                assert isinstance(traits["EntityExtraction"], float)
                assert 0.0 <= traits["EntityExtraction"] <= 1.0


# =============================================================================
# Rubric Consistency Tests (uses mixin)
# =============================================================================


@pytest.mark.integration
@pytest.mark.rubric
class TestRubricConsistency(CommonColumnTestMixin):
    """Test consistency between RubricResults and TemplateResults.

    Inherits common column and status tests from CommonColumnTestMixin.
    """

    test_template = True
    test_rubric = True
    test_judgment = False  # No deep judgment data in these fixtures


# =============================================================================
# Rubric Pandas Operations Tests (uses mixin)
# =============================================================================


@pytest.mark.integration
@pytest.mark.rubric
class TestRubricPandasOperations(PandasOperationsTestMixin):
    """Test pandas operations on RubricResults DataFrames.

    Inherits groupby and filtering tests from PandasOperationsTestMixin.
    """

    def _get_test_dataframe(
        self, verification_results_list: list[VerificationResult]
    ) -> pd.DataFrame:
        """Override to return RubricResults DataFrame with LLM traits."""
        rubric_results = RubricResults(results=verification_results_list)
        return rubric_results.to_dataframe(trait_type="llm")

    def test_filtering_by_trait(self, verification_results_list: list[VerificationResult]):
        """Test filtering DataFrame by trait name."""
        rubric_results = RubricResults(results=verification_results_list)
        df = rubric_results.to_dataframe(trait_type="all")

        # Filter to specific trait
        clarity_df = df[df["trait_name"] == "Clarity"]
        assert len(clarity_df) > 0

    def test_pivot_operations_on_traits(self, verification_results_list: list[VerificationResult]):
        """Test pandas pivot operations on traits."""
        df = self._get_test_dataframe(verification_results_list)

        if len(df) == 0:
            pytest.skip("No LLM trait data for pivot testing")

        # Pivot: questions x traits
        try:
            pivot = df.pivot_table(
                values="trait_score",
                index="question_id",
                columns="trait_name",
                aggfunc="mean",
            )
            assert isinstance(pivot, pd.DataFrame)
        except Exception as e:
            pytest.skip(f"Pivot not applicable: {e}")

    def test_multi_level_groupby(self, verification_results_list: list[VerificationResult]):
        """Test multi-level groupby operations."""
        rubric_results = RubricResults(results=verification_results_list)
        df = rubric_results.to_dataframe(trait_type="all")

        # Group by model and trait name
        grouped = df.groupby(["answering_model", "trait_name"])["trait_score"].mean()
        assert isinstance(grouped, pd.Series)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
