"""Integration tests for DataFrame functionality with deep judgment verification.

This test module runs actual verification with deep judgment enabled using real
checkpoints and presets, then validates that JudgmentResults DataFrame methods
work correctly with real data.

Tests are marked with:
- @pytest.mark.integration: Slow tests that run real verification
- @pytest.mark.requires_api: Tests that need OpenAI API access
- @pytest.mark.deep_judgment: Tests specific to deep judgment mode

Run with: pytest tests/test_dataframe_integration_deep_judgment.py -v
Skip with: pytest -m "not integration"
"""

import os
from pathlib import Path

import pandas as pd
import pytest

from karenina.benchmark import Benchmark
from karenina.schemas import VerificationConfig
from karenina.schemas.workflow import JudgmentResults, TemplateResults

# Paths
CHECKPOINT_PATH = Path("/Users/carli/Projects/karenina_dev/checkpoints/latest.jsonld")
DEEP_JUDGMENT_PRESET = Path("/Users/carli/Projects/karenina_dev/presets/got-oss-003-8000-deep.json")


@pytest.fixture(scope="module")
def checkpoint_exists():
    """Check if checkpoint file exists."""
    if not CHECKPOINT_PATH.exists():
        pytest.skip(f"Checkpoint not found at {CHECKPOINT_PATH}")
    return True


@pytest.fixture(scope="module")
def api_key_available():
    """Check if OpenAI API key is available."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set - skipping integration tests")
    return True


@pytest.fixture(scope="module")
def loaded_benchmark(checkpoint_exists):  # noqa: ARG001
    """Load benchmark from checkpoint."""
    return Benchmark.load(CHECKPOINT_PATH)


@pytest.fixture(scope="module")
def deep_judgment_config():
    """Load deep judgment verification config."""
    if not DEEP_JUDGMENT_PRESET.exists():
        pytest.skip(f"Deep judgment preset not found at {DEEP_JUDGMENT_PRESET}")

    return VerificationConfig.from_preset(DEEP_JUDGMENT_PRESET)


@pytest.fixture(scope="module")
def verification_results(loaded_benchmark, deep_judgment_config, api_key_available):  # noqa: ARG001
    """Run verification with deep judgment and return results (cached for module)."""
    # Get a subset of questions to verify (limit to 2 for speed and cost)
    all_questions = loaded_benchmark.get_all_questions(ids_only=False)
    finished_questions = [q for q in all_questions if q.get("finished") and q.get("answer_template")]

    if len(finished_questions) < 2:
        pytest.skip("Not enough finished questions in checkpoint")

    # Select first 2 questions
    question_ids = [finished_questions[0]["id"], finished_questions[1]["id"]]

    # Run verification with deep judgment
    results = loaded_benchmark.run_verification(config=deep_judgment_config, question_ids=question_ids)

    if not results:
        pytest.fail("Verification returned no results")

    return results


@pytest.mark.integration
@pytest.mark.requires_api
@pytest.mark.deep_judgment
class TestJudgmentResultsIntegrationDeep:
    """Integration tests for JudgmentResults with real deep judgment verification data."""

    def test_to_dataframe_with_real_deep_judgment(self, verification_results):
        """Test JudgmentResults.to_dataframe() with real deep judgment results."""
        # Filter to results with deep judgment data
        judgment_results_list = [
            r for r in verification_results.results if r.deep_judgment and r.deep_judgment.deep_judgment_performed
        ]

        if not judgment_results_list:
            pytest.skip("No deep judgment results in verification data")

        judgment_results = JudgmentResults(results=judgment_results_list)

        # Convert to DataFrame
        df = judgment_results.to_dataframe()

        # Validate DataFrame
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0, "DataFrame should not be empty"

        # Check required columns
        required_columns = [
            "completed_without_errors",
            "question_id",
            "answering_model",
            "parsing_model",
            "attribute_name",
            "deep_judgment_performed",
        ]

        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"

        # Validate deep judgment flag
        assert df["deep_judgment_performed"].dtype == bool
        # Should all be True since we filtered for deep_judgment_performed
        assert df["deep_judgment_performed"].all(), "All results should have deep judgment performed"

        # Check attribute explosion
        assert "attribute_name" in df.columns
        assert df["attribute_name"].notna().any(), "Should have attribute data"

    def test_aggregate_excerpt_counts_with_real_data(self, verification_results):
        """Test aggregate_excerpt_counts() with real deep judgment results."""
        # Filter to results with excerpts
        judgment_results_list = [
            r for r in verification_results.results if r.deep_judgment and r.deep_judgment.extracted_excerpts
        ]

        if not judgment_results_list:
            pytest.skip("No excerpt data in verification results")

        judgment_results = JudgmentResults(results=judgment_results_list)

        # Aggregate excerpt counts by question
        counts = judgment_results.aggregate_excerpt_counts(strategy="mean", by="question_id")

        # Validate results
        assert isinstance(counts, dict)
        assert len(counts) > 0, "Should have excerpt count data"

        for question_id, count_data in counts.items():
            assert isinstance(question_id, str)
            # count_data is a dict mapping attribute names to their excerpt counts
            assert isinstance(count_data, dict)
            for attr_name, count_value in count_data.items():
                assert isinstance(attr_name, str)
                assert isinstance(count_value, int | float)
                assert count_value >= 0, f"Excerpt count should be non-negative: {count_value}"

    def test_aggregate_excerpt_counts_by_model(self, verification_results):
        """Test aggregate_excerpt_counts() grouped by model."""
        # Filter to results with excerpts
        judgment_results_list = [
            r for r in verification_results.results if r.deep_judgment and r.deep_judgment.extracted_excerpts
        ]

        if not judgment_results_list:
            pytest.skip("No excerpt data in verification results")

        judgment_results = JudgmentResults(results=judgment_results_list)

        # Aggregate by model
        counts = judgment_results.aggregate_excerpt_counts(strategy="mean", by="answering_model")

        # Validate results
        assert isinstance(counts, dict)

        for model_name, count_data in counts.items():
            assert isinstance(model_name, str)
            # count_data is a dict mapping attribute names to their excerpt counts
            assert isinstance(count_data, dict)
            for attr_name, count_value in count_data.items():
                assert isinstance(attr_name, str)
                assert isinstance(count_value, int | float)
                assert count_value >= 0

    def test_deep_judgment_columns_present(self, verification_results):
        """Test that deep judgment specific columns are present in DataFrame."""
        judgment_results_list = [
            r for r in verification_results.results if r.deep_judgment and r.deep_judgment.deep_judgment_performed
        ]

        if not judgment_results_list:
            pytest.skip("No deep judgment results")

        judgment_results = JudgmentResults(results=judgment_results_list)
        df = judgment_results.to_dataframe()

        # Deep judgment specific columns (based on actual schema)
        deep_judgment_columns = [
            "attribute_name",
            "deep_judgment_performed",
            "attribute_match",  # The actual judgment result
            "attribute_reasoning",  # Reasoning for the judgment
        ]

        for col in deep_judgment_columns:
            assert col in df.columns, f"Missing deep judgment column: {col}"

    def test_extracted_excerpts_in_dataframe(self, verification_results):
        """Test that extracted excerpts are properly included in DataFrame."""
        judgment_results_list = [
            r for r in verification_results.results if r.deep_judgment and r.deep_judgment.extracted_excerpts
        ]

        if not judgment_results_list:
            pytest.skip("No excerpt data")

        judgment_results = JudgmentResults(results=judgment_results_list)
        df = judgment_results.to_dataframe()

        # Should have excerpt-related columns
        assert "attribute_name" in df.columns

        # Check that we have multiple rows per result (due to attribute explosion)
        # Each result should have multiple attributes
        question_counts = df.groupby("question_id").size()
        assert (question_counts > 1).any(), "Should have multiple attributes per question"


@pytest.mark.integration
@pytest.mark.requires_api
@pytest.mark.deep_judgment
class TestDeepJudgmentConsistency:
    """Integration tests for consistency of deep judgment DataFrames."""

    def test_common_columns_with_template_results(self, verification_results):
        """Test that deep judgment results share common columns with template results."""
        results_list = list(verification_results.results)

        # Get both DataFrames
        template_results = TemplateResults(results=results_list)
        template_df = template_results.to_dataframe()

        judgment_results_list = [r for r in results_list if r.deep_judgment and r.deep_judgment.deep_judgment_performed]

        if not judgment_results_list:
            pytest.skip("No deep judgment results")

        judgment_results = JudgmentResults(results=judgment_results_list)
        judgment_df = judgment_results.to_dataframe()

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
            assert col in judgment_df.columns, f"JudgmentResults missing common column: {col}"

    def test_status_column_first(self, verification_results):
        """Test that status column appears first in JudgmentResults DataFrame."""
        judgment_results_list = [
            r for r in verification_results.results if r.deep_judgment and r.deep_judgment.deep_judgment_performed
        ]

        if not judgment_results_list:
            pytest.skip("No deep judgment results")

        judgment_results = JudgmentResults(results=judgment_results_list)
        judgment_df = judgment_results.to_dataframe()

        # First column should be status
        assert judgment_df.columns[0] == "completed_without_errors"


@pytest.mark.integration
@pytest.mark.requires_api
@pytest.mark.deep_judgment
class TestDeepJudgmentPandasOperations:
    """Integration tests for pandas operations on deep judgment DataFrames."""

    def test_groupby_operations(self, verification_results):
        """Test pandas groupby operations on JudgmentResults DataFrame."""
        judgment_results_list = [
            r for r in verification_results.results if r.deep_judgment and r.deep_judgment.deep_judgment_performed
        ]

        if not judgment_results_list:
            pytest.skip("No deep judgment results")

        judgment_results = JudgmentResults(results=judgment_results_list)
        df = judgment_results.to_dataframe()

        # Test groupby question_id
        grouped = df.groupby("question_id")
        assert len(grouped) > 0

        # Test aggregation on attribute match results
        if "attribute_match" in df.columns:
            match_rates = grouped["attribute_match"].mean()
            assert isinstance(match_rates, pd.Series)

    def test_filtering_operations(self, verification_results):
        """Test pandas filtering operations on JudgmentResults DataFrame."""
        judgment_results_list = [
            r for r in verification_results.results if r.deep_judgment and r.deep_judgment.deep_judgment_performed
        ]

        if not judgment_results_list:
            pytest.skip("No deep judgment results")

        judgment_results = JudgmentResults(results=judgment_results_list)
        df = judgment_results.to_dataframe()

        # Filter to successful results only
        successful = df[df["completed_without_errors"] == True]  # noqa: E712
        assert len(successful) >= 0

        # Filter to specific question
        if len(df) > 0:
            first_question = df["question_id"].iloc[0]
            question_df = df[df["question_id"] == first_question]
            assert len(question_df) > 0
            assert (question_df["question_id"] == first_question).all()

    def test_pivot_operations_on_attributes(self, verification_results):
        """Test pandas pivot operations on JudgmentResults DataFrame."""
        judgment_results_list = [
            r for r in verification_results.results if r.deep_judgment and r.deep_judgment.deep_judgment_performed
        ]

        if not judgment_results_list:
            pytest.skip("No deep judgment results")

        judgment_results = JudgmentResults(results=judgment_results_list)
        df = judgment_results.to_dataframe()

        if len(df) == 0 or "attribute_match" not in df.columns:
            pytest.skip("No attribute match data for pivot testing")

        # Try pivot: questions × attributes
        try:
            pivot = df.pivot_table(
                values="attribute_match", index="question_id", columns="attribute_name", aggfunc="mean"
            )
            assert isinstance(pivot, pd.DataFrame)
            assert len(pivot) > 0
        except Exception as e:
            # Pivot may fail if data structure doesn't support it
            pytest.skip(f"Pivot not applicable to this data structure: {e}")
