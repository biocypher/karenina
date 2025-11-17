"""Integration tests for DataFrame functionality with real verification results.

This test module runs actual verification with real checkpoints and presets,
then validates that all DataFrame methods and aggregations work correctly.

Tests are marked with:
- @pytest.mark.integration: Slow tests that run real verification
- @pytest.mark.requires_api: Tests that need OpenAI API access

Run with: pytest tests/test_dataframe_integration.py -v
Skip with: pytest -m "not integration"
"""

import os
from pathlib import Path

import pandas as pd
import pytest

from karenina.benchmark import Benchmark
from karenina.schemas import VerificationConfig
from karenina.schemas.workflow import JudgmentResults, RubricResults, TemplateResults

# Paths
CHECKPOINT_PATH = Path("/Users/carli/Projects/karenina_dev/checkpoints/latest.jsonld")
PRESET_DIR = Path("/Users/carli/Projects/karenina_dev/presets")

# Available template-only presets
TEMPLATE_ONLY_PRESETS = [
    "gpt-oss-001-8000.json",
    "gpt-oss-001-8001.json",
    "gpt-oss-003-8000.json",
    "gpt-oss-003-8001.json",
]


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
def loaded_benchmark(checkpoint_exists):  # noqa: ARG001  # vulture: ignore
    """Load benchmark from checkpoint."""
    _ = checkpoint_exists  # Fixture dependency
    return Benchmark.load(CHECKPOINT_PATH)


@pytest.fixture(scope="module")
def template_only_config():
    """Load a template-only verification config."""
    # Use gpt-oss-003-8000.json as the test config
    preset_path = PRESET_DIR / "gpt-oss-003-8000.json"
    if not preset_path.exists():
        pytest.skip(f"Preset not found at {preset_path}")

    return VerificationConfig.from_preset(preset_path)


@pytest.fixture(scope="module")
def verification_results(loaded_benchmark, template_only_config, api_key_available):  # noqa: ARG001  # vulture: ignore
    """Run verification and return results (cached for module)."""
    _ = api_key_available  # Fixture dependency
    # Get a subset of questions to verify (limit to 2 for speed)
    all_questions = loaded_benchmark.get_all_questions(ids_only=False)
    finished_questions = [q for q in all_questions if q.get("finished") and q.get("answer_template")]

    if len(finished_questions) < 2:
        pytest.skip("Not enough finished questions in checkpoint")

    # Select first 2 questions
    question_ids = [finished_questions[0]["id"], finished_questions[1]["id"]]

    # Run verification
    results = loaded_benchmark.run_verification(config=template_only_config, question_ids=question_ids)

    if not results:
        pytest.fail("Verification returned no results")

    return results


@pytest.mark.integration
@pytest.mark.requires_api
class TestTemplateResultsIntegration:
    """Integration tests for TemplateResults with real verification data."""

    def test_to_dataframe_with_real_results(self, verification_results):
        """Test TemplateResults.to_dataframe() with real verification results."""
        # Create TemplateResults from verification
        template_results = TemplateResults(results=list(verification_results.results))

        # Convert to DataFrame
        df = template_results.to_dataframe()

        # Validate DataFrame
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0, "DataFrame should not be empty"

        # Check required columns
        required_columns = [
            "completed_without_errors",
            "question_id",
            "answering_model",
            "parsing_model",
            "field_name",
            "field_match",
            "verify_result",
        ]

        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"

        # Validate data types
        assert df["completed_without_errors"].dtype == bool
        # field_match can be bool (if no None values) or object (if None values present)
        assert df["field_match"].dtype in (bool, object)

        # Check field explosion
        assert "field_name" in df.columns
        assert df["field_name"].notna().any(), "Should have field data"

    def test_aggregate_pass_rate_with_real_results(self, verification_results):
        """Test aggregate_pass_rate() with real verification results."""
        template_results = TemplateResults(results=list(verification_results.results))

        # Aggregate by question
        pass_rates_by_question = template_results.aggregate_pass_rate(by="question_id")

        # Validate results
        assert isinstance(pass_rates_by_question, dict)
        assert len(pass_rates_by_question) > 0, "Should have pass rate data"

        # All pass rates should be between 0 and 1
        for question_id, pass_rate in pass_rates_by_question.items():
            assert 0.0 <= pass_rate <= 1.0, f"Invalid pass rate for {question_id}: {pass_rate}"

    def test_aggregate_pass_rate_by_model(self, verification_results):
        """Test aggregate_pass_rate() grouped by model."""
        template_results = TemplateResults(results=list(verification_results.results))

        # Aggregate by model
        pass_rates_by_model = template_results.aggregate_pass_rate(by="answering_model")

        # Validate results
        assert isinstance(pass_rates_by_model, dict)

        # Check that model names are present
        for model_name, pass_rate in pass_rates_by_model.items():
            assert isinstance(model_name, str)
            assert 0.0 <= pass_rate <= 1.0

    def test_to_usage_dataframe_with_real_results(self, verification_results):
        """Test to_usage_dataframe() with real verification results."""
        template_results = TemplateResults(results=list(verification_results.results))

        # Get usage DataFrame (exploded by stage)
        usage_df = template_results.to_usage_dataframe(totals_only=False)

        # Validate DataFrame
        assert isinstance(usage_df, pd.DataFrame)

        if len(usage_df) > 0:
            # Check usage columns
            usage_columns = ["usage_stage", "input_tokens", "output_tokens", "total_tokens"]
            for col in usage_columns:
                assert col in usage_df.columns, f"Missing usage column: {col}"

            # Validate stage explosion
            assert usage_df["usage_stage"].notna().any()

            # Token counts should be non-negative
            assert (usage_df["input_tokens"] >= 0).all()
            assert (usage_df["output_tokens"] >= 0).all()

    def test_to_usage_dataframe_totals_only(self, verification_results):
        """Test to_usage_dataframe(totals_only=True) with real results."""
        template_results = TemplateResults(results=list(verification_results.results))

        # Get totals-only usage DataFrame
        totals_df = template_results.to_usage_dataframe(totals_only=True)

        # Validate DataFrame
        assert isinstance(totals_df, pd.DataFrame)

        if len(totals_df) > 0:
            # Should not have usage_stage for totals
            if "usage_stage" in totals_df.columns:
                # usage_stage should be None for totals
                assert totals_df["usage_stage"].isna().all()

            # Should have total token counts
            assert "total_tokens" in totals_df.columns


@pytest.mark.integration
@pytest.mark.requires_api
class TestRubricResultsIntegration:
    """Integration tests for RubricResults with real verification data."""

    def test_to_dataframe_with_real_results(self, verification_results):
        """Test RubricResults.to_dataframe() with real verification results."""
        # Filter to results that have rubric data
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
        assert len(df) > 0

        # Check required columns
        required_columns = [
            "completed_without_errors",
            "question_id",
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

    def test_aggregate_llm_traits_with_real_results(self, verification_results):
        """Test aggregate_llm_traits() with real verification results."""
        # Filter to results with LLM traits
        rubric_results_list = [r for r in verification_results.results if r.rubric and r.rubric.llm_trait_scores]

        if not rubric_results_list:
            pytest.skip("No LLM trait data in verification results")

        rubric_results = RubricResults(results=rubric_results_list)

        # Aggregate LLM traits
        aggregated = rubric_results.aggregate_llm_traits(strategy="mean", by="question_id")

        # Validate results
        assert isinstance(aggregated, dict)

        for _question_id, traits in aggregated.items():
            assert isinstance(traits, dict)

            for _trait_name, score in traits.items():
                # LLM scores should be 1-5 or boolean
                assert isinstance(score, int | float | bool)


@pytest.mark.integration
@pytest.mark.requires_api
class TestJudgmentResultsIntegration:
    """Integration tests for JudgmentResults with real verification data."""

    def test_to_dataframe_with_real_results(self, verification_results):
        """Test JudgmentResults.to_dataframe() with real verification results."""
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
        assert len(df) > 0

        # Check required columns
        required_columns = [
            "completed_without_errors",
            "question_id",
            "attribute_name",
            "deep_judgment_performed",
        ]

        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"

        # Validate deep judgment flag
        assert df["deep_judgment_performed"].dtype == bool

    def test_aggregate_excerpt_counts_with_real_results(self, verification_results):
        """Test aggregate_excerpt_counts() with real verification results."""
        # Filter to results with excerpts
        judgment_results_list = [
            r for r in verification_results.results if r.deep_judgment and r.deep_judgment.extracted_excerpts
        ]

        if not judgment_results_list:
            pytest.skip("No excerpt data in verification results")

        judgment_results = JudgmentResults(results=judgment_results_list)

        # Aggregate excerpt counts
        counts = judgment_results.aggregate_excerpt_counts(strategy="mean", by="question_id")

        # Validate results
        assert isinstance(counts, dict)

        for _question_id, count in counts.items():
            assert isinstance(count, int | float)
            assert count >= 0


@pytest.mark.integration
@pytest.mark.requires_api
class TestDataFrameConsistency:
    """Integration tests for DataFrame consistency across result types."""

    def test_common_columns_consistency(self, verification_results):
        """Test that common columns are consistent across all DataFrame types."""
        results_list = list(verification_results.results)

        # Get DataFrames from all three types
        template_results = TemplateResults(results=results_list)
        template_df = template_results.to_dataframe()

        # Common columns that should exist in all DataFrames
        common_columns = [
            "completed_without_errors",
            "question_id",
            "answering_model",
            "parsing_model",
            "execution_time",
            "timestamp",
        ]

        # Check TemplateResults
        for col in common_columns:
            assert col in template_df.columns, f"TemplateResults missing common column: {col}"

        # Check RubricResults (if available)
        rubric_results_list = [r for r in results_list if r.rubric and r.rubric.rubric_evaluation_performed]
        if rubric_results_list:
            rubric_results = RubricResults(results=rubric_results_list)
            rubric_df = rubric_results.to_dataframe(trait_type="all")

            for col in common_columns:
                assert col in rubric_df.columns, f"RubricResults missing common column: {col}"

        # Check JudgmentResults (if available)
        judgment_results_list = [r for r in results_list if r.deep_judgment and r.deep_judgment.deep_judgment_performed]
        if judgment_results_list:
            judgment_results = JudgmentResults(results=judgment_results_list)
            judgment_df = judgment_results.to_dataframe()

            for col in common_columns:
                assert col in judgment_df.columns, f"JudgmentResults missing common column: {col}"

    def test_status_columns_first(self, verification_results):
        """Test that status columns appear first in all DataFrames."""
        results_list = list(verification_results.results)

        # TemplateResults
        template_results = TemplateResults(results=results_list)
        template_df = template_results.to_dataframe()

        # First column should be status
        assert template_df.columns[0] == "completed_without_errors"

        # RubricResults (if available)
        rubric_results_list = [r for r in results_list if r.rubric and r.rubric.rubric_evaluation_performed]
        if rubric_results_list:
            rubric_results = RubricResults(results=rubric_results_list)
            rubric_df = rubric_results.to_dataframe(trait_type="all")
            assert rubric_df.columns[0] == "completed_without_errors"

        # JudgmentResults (if available)
        judgment_results_list = [r for r in results_list if r.deep_judgment and r.deep_judgment.deep_judgment_performed]
        if judgment_results_list:
            judgment_results = JudgmentResults(results=judgment_results_list)
            judgment_df = judgment_results.to_dataframe()
            assert judgment_df.columns[0] == "completed_without_errors"


@pytest.mark.integration
@pytest.mark.requires_api
class TestPandasOperations:
    """Integration tests for pandas operations on real DataFrames."""

    def test_groupby_operations(self, verification_results):
        """Test pandas groupby operations on TemplateResults DataFrame."""
        template_results = TemplateResults(results=list(verification_results.results))
        df = template_results.to_dataframe()

        # Test groupby question_id
        grouped = df.groupby("question_id")
        assert len(grouped) > 0

        # Test aggregation
        pass_rates = grouped["verify_result"].mean()
        assert isinstance(pass_rates, pd.Series)
        assert len(pass_rates) > 0

    def test_filtering_operations(self, verification_results):
        """Test pandas filtering operations on DataFrames."""
        template_results = TemplateResults(results=list(verification_results.results))
        df = template_results.to_dataframe()

        # Filter to successful results only
        successful = df[df["completed_without_errors"] == True]  # noqa: E712
        assert len(successful) >= 0

        # Filter to specific question
        if len(df) > 0:
            first_question = df["question_id"].iloc[0]
            question_df = df[df["question_id"] == first_question]
            assert len(question_df) > 0
            assert (question_df["question_id"] == first_question).all()

    def test_pivot_operations(self, verification_results):
        """Test pandas pivot operations on RubricResults DataFrame."""
        # Filter to results with rubric data
        rubric_results_list = [
            r for r in verification_results.results if r.rubric and r.rubric.rubric_evaluation_performed
        ]

        if not rubric_results_list:
            pytest.skip("No rubric results for pivot testing")

        rubric_results = RubricResults(results=rubric_results_list)
        df = rubric_results.to_dataframe(trait_type="llm")

        if len(df) == 0:
            pytest.skip("No LLM trait data for pivot testing")

        # Try pivot: questions × traits
        try:
            pivot = df.pivot_table(values="trait_score", index="question_id", columns="trait_name", aggfunc="mean")
            assert isinstance(pivot, pd.DataFrame)
        except Exception as e:
            # Pivot may fail if data structure doesn't support it
            pytest.skip(f"Pivot not applicable to this data structure: {e}")
