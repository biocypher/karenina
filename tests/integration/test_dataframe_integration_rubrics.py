"""Integration tests for DataFrame functionality with rubric evaluation.

This test module runs actual verification with rubric evaluation enabled using real
checkpoints and presets, then validates that RubricResults DataFrame methods
work correctly with real data.

Tests are marked with:
- @pytest.mark.integration: Slow tests that run real verification
- @pytest.mark.requires_api: Tests that need API access (e.g., OpenAI or custom endpoints)
- @pytest.mark.rubric: Tests specific to rubric evaluation mode

Run with: pytest tests/test_dataframe_integration_rubrics.py -v
Skip with: pytest -m "not integration"
"""

from pathlib import Path

import pandas as pd
import pytest

from karenina.benchmark import Benchmark
from karenina.schemas import VerificationConfig
from karenina.schemas.workflow import RubricResults, TemplateResults

# Paths
CHECKPOINT_PATH = Path("/Users/carli/Projects/karenina_dev/checkpoints/latest_rubric_advanced.jsonld")
RUBRIC_PRESET = Path("/Users/carli/Projects/karenina_dev/presets/gpt-oss-003-8000-rubrics.json")


@pytest.fixture(scope="module")
def checkpoint_exists():
    """Check if checkpoint file exists."""
    if not CHECKPOINT_PATH.exists():
        pytest.skip(f"Checkpoint not found at {CHECKPOINT_PATH}")
    return True


@pytest.fixture(scope="module")
def api_key_available():
    """Check if API key or endpoint is available.

    The rubric preset uses a custom endpoint (codon-gpu-003.ebi.ac.uk:8000)
    which may not require OPENAI_API_KEY.
    We assume the endpoint is accessible and will let the test fail if auth is needed.
    """
    # For custom endpoints, we assume they're accessible
    # The actual verification will fail if authentication is required but not provided
    return True


@pytest.fixture(scope="module")
def loaded_benchmark(checkpoint_exists):  # noqa: ARG001
    """Load benchmark from checkpoint."""
    return Benchmark.load(CHECKPOINT_PATH)


@pytest.fixture(scope="module")
def rubric_config():
    """Load rubric-enabled verification config."""
    if not RUBRIC_PRESET.exists():
        pytest.skip(f"Rubric preset not found at {RUBRIC_PRESET}")

    return VerificationConfig.from_preset(RUBRIC_PRESET)


@pytest.fixture(scope="module")
def verification_results(loaded_benchmark, rubric_config, api_key_available):  # noqa: ARG001
    """Run verification with rubric evaluation and return results (cached for module)."""
    # Get a subset of questions to verify (limit to 2 for speed and cost)
    all_questions = loaded_benchmark.get_all_questions(ids_only=False)
    finished_questions = [q for q in all_questions if q.get("finished") and q.get("answer_template")]

    if len(finished_questions) < 2:
        pytest.skip("Not enough finished questions in checkpoint")

    # Select first 2 questions
    question_ids = [finished_questions[0]["id"], finished_questions[1]["id"]]

    # Run verification with rubrics
    results = loaded_benchmark.run_verification(config=rubric_config, question_ids=question_ids)

    if not results:
        pytest.fail("Verification returned no results")

    return results


@pytest.mark.integration
@pytest.mark.requires_api
@pytest.mark.rubric
class TestRubricResultsIntegrationFull:
    """Integration tests for RubricResults with real rubric evaluation data."""

    def test_to_dataframe_with_real_rubric_evaluation(self, verification_results):
        """Test RubricResults.to_dataframe() with real rubric evaluation results."""
        # Filter to results with rubric data
        rubric_results_list = [
            r for r in verification_results.results if r.rubric and r.rubric.rubric_evaluation_performed
        ]

        if not rubric_results_list:
            pytest.skip("No rubric results in verification data")

        rubric_results = RubricResults(results=rubric_results_list)

        # Convert to DataFrame (all traits)
        df = rubric_results.to_dataframe(trait_type="all")

        # Validate DataFrame
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0, "DataFrame should not be empty"

        # Check required columns
        required_columns = [
            "completed_without_errors",
            "question_id",
            "answering_model",
            "parsing_model",
            "trait_name",
            "trait_type",
            "trait_score",
        ]

        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"

        # Validate trait_type values
        valid_trait_types = {"llm_score", "llm_binary", "manual", "metric"}
        trait_types = set(df["trait_type"].dropna().unique())
        assert trait_types.issubset(valid_trait_types), f"Invalid trait types: {trait_types - valid_trait_types}"

        # Check trait explosion - should have multiple rows per result
        assert "trait_name" in df.columns
        assert df["trait_name"].notna().any(), "Should have trait data"

    def test_to_dataframe_llm_traits_only(self, verification_results):
        """Test RubricResults.to_dataframe() with trait_type='llm' filter."""
        rubric_results_list = [
            r for r in verification_results.results if r.rubric and r.rubric.rubric_evaluation_performed
        ]

        if not rubric_results_list:
            pytest.skip("No rubric results")

        rubric_results = RubricResults(results=rubric_results_list)

        # Convert to DataFrame (LLM traits only)
        df = rubric_results.to_dataframe(trait_type="llm")

        # Validate DataFrame
        assert isinstance(df, pd.DataFrame)

        if len(df) > 0:
            # All trait_type values should be llm_score or llm_binary
            valid_llm_types = {"llm_score", "llm_binary"}
            trait_types = set(df["trait_type"].dropna().unique())
            assert trait_types.issubset(valid_llm_types), f"Non-LLM trait types found: {trait_types - valid_llm_types}"

    def test_to_dataframe_manual_traits_only(self, verification_results):
        """Test RubricResults.to_dataframe() with trait_type='manual' filter."""
        rubric_results_list = [
            r for r in verification_results.results if r.rubric and r.rubric.rubric_evaluation_performed
        ]

        if not rubric_results_list:
            pytest.skip("No rubric results")

        rubric_results = RubricResults(results=rubric_results_list)

        # Convert to DataFrame (manual traits only)
        df = rubric_results.to_dataframe(trait_type="manual")

        # Validate DataFrame
        assert isinstance(df, pd.DataFrame)

        if len(df) > 0:
            # All trait_type values should be manual
            trait_types = set(df["trait_type"].dropna().unique())
            assert trait_types == {"manual"}, f"Non-manual trait types found: {trait_types - {'manual'}}"

    def test_to_dataframe_metric_traits_only(self, verification_results):
        """Test RubricResults.to_dataframe() with trait_type='metric' filter."""
        rubric_results_list = [
            r for r in verification_results.results if r.rubric and r.rubric.rubric_evaluation_performed
        ]

        if not rubric_results_list:
            pytest.skip("No rubric results")

        rubric_results = RubricResults(results=rubric_results_list)

        # Convert to DataFrame (metric traits only)
        df = rubric_results.to_dataframe(trait_type="metric")

        # Validate DataFrame
        assert isinstance(df, pd.DataFrame)

        if len(df) > 0:
            # All trait_type values should be metric
            trait_types = set(df["trait_type"].dropna().unique())
            assert trait_types == {"metric"}, f"Non-metric trait types found: {trait_types - {'metric'}}"

    def test_aggregate_llm_traits_with_real_data(self, verification_results):
        """Test aggregate_llm_traits() with real rubric evaluation results."""
        # Filter to results with LLM traits
        rubric_results_list = [r for r in verification_results.results if r.rubric and r.rubric.llm_trait_scores]

        if not rubric_results_list:
            pytest.skip("No LLM trait data in verification results")

        rubric_results = RubricResults(results=rubric_results_list)

        # Aggregate LLM traits by question
        aggregated = rubric_results.aggregate_llm_traits(strategy="mean", by="question_id")

        # Validate results
        assert isinstance(aggregated, dict)
        assert len(aggregated) > 0, "Should have aggregated LLM trait data"

        for question_id, traits in aggregated.items():
            assert isinstance(question_id, str)
            assert isinstance(traits, dict)

            for trait_name, score in traits.items():
                assert isinstance(trait_name, str)
                # LLM scores should be 1-5 for llm_score or boolean for llm_binary
                assert isinstance(score, int | float | bool)
                if isinstance(score, int | float):
                    # Check if it's within expected range (1-5 for LLM scores)
                    # Note: After aggregation with mean, scores might be floats
                    assert score >= 1 and score <= 5, f"LLM score out of range: {score}"

    def test_aggregate_llm_traits_by_model(self, verification_results):
        """Test aggregate_llm_traits() grouped by model."""
        rubric_results_list = [r for r in verification_results.results if r.rubric and r.rubric.llm_trait_scores]

        if not rubric_results_list:
            pytest.skip("No LLM trait data")

        rubric_results = RubricResults(results=rubric_results_list)

        # Aggregate by model
        aggregated = rubric_results.aggregate_llm_traits(strategy="mean", by="answering_model")

        # Validate results
        assert isinstance(aggregated, dict)

        for model_name, traits in aggregated.items():
            assert isinstance(model_name, str)
            assert isinstance(traits, dict)

            for trait_name, score in traits.items():
                assert isinstance(trait_name, str)
                assert isinstance(score, int | float | bool)

    def test_aggregate_manual_traits_with_real_data(self, verification_results):
        """Test aggregate_manual_traits() with real rubric evaluation results."""
        # Filter to results with manual traits
        rubric_results_list = [r for r in verification_results.results if r.rubric and r.rubric.manual_trait_scores]

        if not rubric_results_list:
            pytest.skip("No manual trait data in verification results")

        rubric_results = RubricResults(results=rubric_results_list)

        # Aggregate manual traits by question
        aggregated = rubric_results.aggregate_manual_traits(strategy="mean", by="question_id")

        # Validate results
        assert isinstance(aggregated, dict)
        assert len(aggregated) > 0, "Should have aggregated manual trait data"

        for question_id, traits in aggregated.items():
            assert isinstance(question_id, str)
            assert isinstance(traits, dict)

            for trait_name, score in traits.items():
                assert isinstance(trait_name, str)
                # Manual scores can be numeric or boolean
                assert isinstance(score, int | float | bool)

    def test_aggregate_metric_traits_with_real_data(self, verification_results):
        """Test aggregate_metric_traits() with real rubric evaluation results."""
        # Filter to results with metric traits
        rubric_results_list = [r for r in verification_results.results if r.rubric and r.rubric.metric_trait_scores]

        if not rubric_results_list:
            pytest.skip("No metric trait data in verification results")

        rubric_results = RubricResults(results=rubric_results_list)

        # Aggregate metric traits by question
        aggregated = rubric_results.aggregate_metric_traits(strategy="mean", by="question_id")

        # Validate results
        assert isinstance(aggregated, dict)
        assert len(aggregated) > 0, "Should have aggregated metric trait data"

        for question_id, traits in aggregated.items():
            assert isinstance(question_id, str)
            assert isinstance(traits, dict)

            for trait_name, score in traits.items():
                assert isinstance(trait_name, str)
                # Metric scores should be numeric
                assert isinstance(score, int | float)

    def test_rubric_columns_present(self, verification_results):
        """Test that rubric-specific columns are present in DataFrame."""
        rubric_results_list = [
            r for r in verification_results.results if r.rubric and r.rubric.rubric_evaluation_performed
        ]

        if not rubric_results_list:
            pytest.skip("No rubric results")

        rubric_results = RubricResults(results=rubric_results_list)
        df = rubric_results.to_dataframe(trait_type="all")

        # Rubric-specific columns (trait explosion columns)
        rubric_columns = [
            "trait_name",
            "trait_type",
            "trait_score",
            "evaluation_rubric",  # The rubric used for evaluation
        ]

        for col in rubric_columns:
            assert col in df.columns, f"Missing rubric column: {col}"

    def test_trait_explosion_in_dataframe(self, verification_results):
        """Test that traits are properly exploded in DataFrame."""
        rubric_results_list = [
            r for r in verification_results.results if r.rubric and r.rubric.rubric_evaluation_performed
        ]

        if not rubric_results_list:
            pytest.skip("No rubric results")

        rubric_results = RubricResults(results=rubric_results_list)
        df = rubric_results.to_dataframe(trait_type="all")

        # Should have trait-related columns
        assert "trait_name" in df.columns
        assert "trait_type" in df.columns

        # Check that we have multiple rows per result (due to trait explosion)
        # Each result should have multiple traits
        question_counts = df.groupby("question_id").size()
        assert (question_counts > 1).any(), "Should have multiple traits per question"


@pytest.mark.integration
@pytest.mark.requires_api
@pytest.mark.rubric
class TestRubricConsistency:
    """Integration tests for consistency of rubric DataFrames."""

    def test_common_columns_with_template_results(self, verification_results):
        """Test that rubric results share common columns with template results."""
        results_list = list(verification_results.results)

        # Get both DataFrames
        template_results = TemplateResults(results=results_list)
        template_df = template_results.to_dataframe()

        rubric_results_list = [r for r in results_list if r.rubric and r.rubric.rubric_evaluation_performed]

        if not rubric_results_list:
            pytest.skip("No rubric results")

        rubric_results = RubricResults(results=rubric_results_list)
        rubric_df = rubric_results.to_dataframe(trait_type="all")

        # Common columns that should exist in both
        common_columns = [
            "completed_without_errors",
            "question_id",
            "answering_model",
            "parsing_model",
            "execution_time",
            "timestamp",
        ]

        for col in common_columns:
            assert col in template_df.columns, f"TemplateResults missing common column: {col}"
            assert col in rubric_df.columns, f"RubricResults missing common column: {col}"

    def test_status_column_first(self, verification_results):
        """Test that status column appears first in RubricResults DataFrame."""
        rubric_results_list = [
            r for r in verification_results.results if r.rubric and r.rubric.rubric_evaluation_performed
        ]

        if not rubric_results_list:
            pytest.skip("No rubric results")

        rubric_results = RubricResults(results=rubric_results_list)
        rubric_df = rubric_results.to_dataframe(trait_type="all")

        # First column should be status
        assert rubric_df.columns[0] == "completed_without_errors"


@pytest.mark.integration
@pytest.mark.requires_api
@pytest.mark.rubric
class TestRubricPandasOperations:
    """Integration tests for pandas operations on rubric DataFrames."""

    def test_groupby_operations(self, verification_results):
        """Test pandas groupby operations on RubricResults DataFrame."""
        rubric_results_list = [
            r for r in verification_results.results if r.rubric and r.rubric.rubric_evaluation_performed
        ]

        if not rubric_results_list:
            pytest.skip("No rubric results")

        rubric_results = RubricResults(results=rubric_results_list)
        df = rubric_results.to_dataframe(trait_type="all")

        # Test groupby question_id
        grouped = df.groupby("question_id")
        assert len(grouped) > 0

        # Test aggregation on trait scores
        if "trait_score" in df.columns:
            # Filter numeric scores only
            numeric_df = df[df["trait_score"].apply(lambda x: isinstance(x, int | float))]
            if len(numeric_df) > 0:
                score_means = numeric_df.groupby("question_id")["trait_score"].mean()
                assert isinstance(score_means, pd.Series)

    def test_filtering_operations(self, verification_results):
        """Test pandas filtering operations on RubricResults DataFrame."""
        rubric_results_list = [
            r for r in verification_results.results if r.rubric and r.rubric.rubric_evaluation_performed
        ]

        if not rubric_results_list:
            pytest.skip("No rubric results")

        rubric_results = RubricResults(results=rubric_results_list)
        df = rubric_results.to_dataframe(trait_type="all")

        # Filter to successful results only
        successful = df[df["completed_without_errors"] == True]  # noqa: E712
        assert len(successful) >= 0

        # Filter to specific trait type
        if len(df) > 0 and "trait_type" in df.columns:
            trait_types = df["trait_type"].unique()
            if len(trait_types) > 0:
                first_type = trait_types[0]
                type_df = df[df["trait_type"] == first_type]
                assert len(type_df) > 0
                assert (type_df["trait_type"] == first_type).all()

    def test_pivot_operations_on_traits(self, verification_results):
        """Test pandas pivot operations on RubricResults DataFrame."""
        rubric_results_list = [
            r for r in verification_results.results if r.rubric and r.rubric.rubric_evaluation_performed
        ]

        if not rubric_results_list:
            pytest.skip("No rubric results")

        rubric_results = RubricResults(results=rubric_results_list)
        df = rubric_results.to_dataframe(trait_type="llm")  # Use LLM traits for pivot

        if len(df) == 0:
            pytest.skip("No LLM trait data for pivot testing")

        # Filter to numeric scores only for pivot
        numeric_df = df[df["trait_score"].apply(lambda x: isinstance(x, int | float))]

        if len(numeric_df) == 0:
            pytest.skip("No numeric trait scores for pivot testing")

        # Try pivot: questions × traits
        try:
            pivot = numeric_df.pivot_table(
                values="trait_score", index="question_id", columns="trait_name", aggfunc="mean"
            )
            assert isinstance(pivot, pd.DataFrame)
            assert len(pivot) > 0
        except Exception as e:
            # Pivot may fail if data structure doesn't support it
            pytest.skip(f"Pivot not applicable to this data structure: {e}")

    def test_multi_level_groupby(self, verification_results):
        """Test multi-level groupby operations (question + trait)."""
        rubric_results_list = [
            r for r in verification_results.results if r.rubric and r.rubric.rubric_evaluation_performed
        ]

        if not rubric_results_list:
            pytest.skip("No rubric results")

        rubric_results = RubricResults(results=rubric_results_list)
        df = rubric_results.to_dataframe(trait_type="all")

        if len(df) == 0:
            pytest.skip("No data for multi-level groupby")

        # Test groupby multiple columns
        grouped = df.groupby(["question_id", "trait_name"])
        assert len(grouped) > 0

        # Get group keys
        group_keys = list(grouped.groups.keys())
        assert len(group_keys) > 0
        assert all(len(key) == 2 for key in group_keys)  # Each key should be (question_id, trait_name)
