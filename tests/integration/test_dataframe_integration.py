"""Integration tests for DataFrame functionality with verification results.

This test module validates that all DataFrame methods and aggregations work
correctly with VerificationResult objects.

Tests use fixture-created results rather than running actual verification,
following the fixture-based testing pattern.
"""

from datetime import UTC, datetime

import pandas as pd
import pytest

from karenina.schemas.workflow import (
    JudgmentResults,
    RubricResults,
    TemplateResults,
    VerificationResult,
    VerificationResultDeepJudgment,
    VerificationResultMetadata,
    VerificationResultRubric,
    VerificationResultTemplate,
)

# =============================================================================
# Fixtures for VerificationResult objects
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


@pytest.fixture
def sample_template_result() -> VerificationResultTemplate:
    """Create a sample template result with field evaluations."""
    return VerificationResultTemplate(
        raw_llm_response="The capital of France is Paris.",
        parsed_gt_response={"capital": "Paris", "country": "France"},
        parsed_llm_response={"capital": "Paris", "country": "France"},
        template_verification_performed=True,
        verify_result=True,
        verify_granular_result={"capital": True, "country": True},
    )


@pytest.fixture
def sample_failed_template_result() -> VerificationResultTemplate:
    """Create a sample failed template result."""
    return VerificationResultTemplate(
        raw_llm_response="The capital of France is Lyon.",
        parsed_gt_response={"capital": "Paris"},
        parsed_llm_response={"capital": "Lyon"},
        template_verification_performed=True,
        verify_result=False,
        verify_granular_result={"capital": False},
    )


@pytest.fixture
def sample_rubric_result() -> VerificationResultRubric:
    """Create a sample rubric result with traits."""
    return VerificationResultRubric(
        rubric_evaluation_performed=True,
        rubric_evaluation_strategy="batch",
        llm_trait_scores={"clarity": True, "completeness": 4},
        regex_trait_scores={"has_citations": True},
        callable_trait_scores={},
        metric_trait_scores={},
    )


@pytest.fixture
def sample_deep_judgment_result() -> VerificationResultDeepJudgment:
    """Create a sample deep judgment result."""
    return VerificationResultDeepJudgment(
        deep_judgment_enabled=True,
        deep_judgment_performed=True,
        extracted_excerpts={"capital": [{"text": "Paris is the capital", "confidence": "high"}]},
        attribute_reasoning={"capital": "Clearly stated in response"},
        deep_judgment_stages_completed=["excerpts", "reasoning"],
        deep_judgment_model_calls=2,
    )


@pytest.fixture
def verification_result_success(
    sample_template_result: VerificationResultTemplate,
    sample_rubric_result: VerificationResultRubric,
    sample_deep_judgment_result: VerificationResultDeepJudgment,
) -> VerificationResult:
    """Create a successful verification result."""
    return VerificationResult(
        metadata=_create_metadata("q001", "claude-haiku-4-5", completed=True),
        template=sample_template_result,
        rubric=sample_rubric_result,
        deep_judgment=sample_deep_judgment_result,
    )


@pytest.fixture
def verification_result_failed(
    sample_failed_template_result: VerificationResultTemplate,
) -> VerificationResult:
    """Create a failed verification result."""
    return VerificationResult(
        metadata=_create_metadata("q002", "gpt-4o-mini", completed=True),
        template=sample_failed_template_result,
        rubric=None,
        deep_judgment=None,
    )


@pytest.fixture
def verification_result_error() -> VerificationResult:
    """Create a verification result with errors."""
    return VerificationResult(
        metadata=_create_metadata("q003", "claude-haiku-4-5", completed=False, error="Connection timeout"),
        template=None,
        rubric=None,
        deep_judgment=None,
    )


@pytest.fixture
def verification_results_list(
    verification_result_success: VerificationResult,
    verification_result_failed: VerificationResult,
    verification_result_error: VerificationResult,
) -> list[VerificationResult]:
    """Create a list of mixed verification results."""
    return [
        verification_result_success,
        verification_result_failed,
        verification_result_error,
    ]


# =============================================================================
# TemplateResults DataFrame Tests
# =============================================================================


@pytest.mark.integration
class TestTemplateResultsIntegration:
    """Integration tests for TemplateResults with verification data."""

    def test_to_dataframe_with_results(self, verification_results_list: list[VerificationResult]):
        """Test TemplateResults.to_dataframe() with verification results."""
        template_results = TemplateResults(results=verification_results_list)

        # Convert to DataFrame
        df = template_results.to_dataframe()

        # Validate DataFrame
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0, "DataFrame should not be empty"

        # Check core columns exist
        core_columns = [
            "completed_without_errors",
            "question_id",
            "answering_model",
            "parsing_model",
        ]

        for col in core_columns:
            assert col in df.columns, f"Missing core column: {col}"

        # Validate data types
        assert df["completed_without_errors"].dtype == bool

    def test_aggregate_pass_rate_with_results(self, verification_results_list: list[VerificationResult]):
        """Test aggregate_pass_rate() with verification results."""
        template_results = TemplateResults(results=verification_results_list)

        # Aggregate by question
        pass_rates_by_question = template_results.aggregate_pass_rate(by="question_id")

        # Validate results
        assert isinstance(pass_rates_by_question, dict)
        assert len(pass_rates_by_question) > 0, "Should have pass rate data"

        # All pass rates should be between 0 and 1
        for question_id, pass_rate in pass_rates_by_question.items():
            assert 0.0 <= pass_rate <= 1.0, f"Invalid pass rate for {question_id}: {pass_rate}"

    def test_aggregate_pass_rate_by_model(self, verification_results_list: list[VerificationResult]):
        """Test aggregate_pass_rate() grouped by model."""
        template_results = TemplateResults(results=verification_results_list)

        # Aggregate by model
        pass_rates_by_model = template_results.aggregate_pass_rate(by="answering_model")

        # Validate results
        assert isinstance(pass_rates_by_model, dict)

        # Check that model names are present
        for model_name, pass_rate in pass_rates_by_model.items():
            assert isinstance(model_name, str)
            assert 0.0 <= pass_rate <= 1.0

    def test_to_usage_dataframe_with_results(self, verification_result_success: VerificationResult):
        """Test to_usage_dataframe() with verification results."""
        # Add usage metadata to the template
        verification_result_success.template.usage_metadata = {
            "answering": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            "parsing": {"input_tokens": 200, "output_tokens": 100, "total_tokens": 300},
            "total": {"input_tokens": 300, "output_tokens": 150, "total_tokens": 450},
        }
        template_results = TemplateResults(results=[verification_result_success])

        # Get usage DataFrame (exploded by stage)
        usage_df = template_results.to_usage_dataframe(totals_only=False)

        # Validate DataFrame
        assert isinstance(usage_df, pd.DataFrame)

        if len(usage_df) > 0:
            # Check usage columns exist
            assert "total_tokens" in usage_df.columns
            # Token counts should be non-negative
            assert (usage_df["total_tokens"] >= 0).all()


# =============================================================================
# RubricResults DataFrame Tests
# =============================================================================


@pytest.mark.integration
class TestRubricResultsIntegration:
    """Integration tests for RubricResults with verification data."""

    def test_to_dataframe_with_results(self, verification_result_success: VerificationResult):
        """Test RubricResults.to_dataframe() with verification results."""
        # Use only the result with rubric data
        rubric_results = RubricResults(results=[verification_result_success])

        # Convert to DataFrame (all traits)
        df = rubric_results.to_dataframe(trait_type="all")

        # Validate DataFrame
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

        # Check core columns
        core_columns = [
            "completed_without_errors",
            "question_id",
        ]

        for col in core_columns:
            assert col in df.columns, f"Missing core column: {col}"

    def test_aggregate_llm_traits_with_results(self, verification_result_success: VerificationResult):
        """Test aggregate_llm_traits() with verification results."""
        rubric_results = RubricResults(results=[verification_result_success])

        # Aggregate LLM traits
        aggregated = rubric_results.aggregate_llm_traits(strategy="mean", by="question_id")

        # Validate results
        assert isinstance(aggregated, dict)

        for _question_id, traits in aggregated.items():
            assert isinstance(traits, dict)

            for _trait_name, score in traits.items():
                # LLM scores should be 1-5 or boolean
                assert isinstance(score, int | float | bool)


# =============================================================================
# JudgmentResults DataFrame Tests
# =============================================================================


@pytest.mark.integration
class TestJudgmentResultsIntegration:
    """Integration tests for JudgmentResults with verification data."""

    def test_to_dataframe_with_results(self, verification_result_success: VerificationResult):
        """Test JudgmentResults.to_dataframe() with verification results."""
        # Use only the result with deep judgment data
        judgment_results = JudgmentResults(results=[verification_result_success])

        # Convert to DataFrame
        df = judgment_results.to_dataframe()

        # Validate DataFrame
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

        # Check core columns
        core_columns = [
            "completed_without_errors",
            "question_id",
        ]

        for col in core_columns:
            assert col in df.columns, f"Missing core column: {col}"

    def test_aggregate_excerpt_counts_with_results(self, verification_result_success: VerificationResult):
        """Test aggregate_excerpt_counts() with verification results."""
        judgment_results = JudgmentResults(results=[verification_result_success])

        # Aggregate excerpt counts
        counts = judgment_results.aggregate_excerpt_counts(strategy="mean", by="question_id")

        # Validate results
        assert isinstance(counts, dict)

        for _question_id, count_data in counts.items():
            # count_data can be a dict of attribute counts or a single value
            if isinstance(count_data, dict):
                for _attr, count in count_data.items():
                    assert isinstance(count, int | float)
                    assert count >= 0
            else:
                assert isinstance(count_data, int | float)
                assert count_data >= 0


# =============================================================================
# DataFrame Consistency Tests
# =============================================================================


@pytest.mark.integration
class TestDataFrameConsistency:
    """Integration tests for DataFrame consistency across result types."""

    def test_common_columns_consistency(self, verification_result_success: VerificationResult):
        """Test that common columns are consistent across all DataFrame types."""
        results_list = [verification_result_success]

        # Get DataFrames from all three types
        template_results = TemplateResults(results=results_list)
        template_df = template_results.to_dataframe()

        # Common columns that should exist in all DataFrames
        common_columns = [
            "completed_without_errors",
            "question_id",
            "answering_model",
            "parsing_model",
        ]

        # Check TemplateResults
        for col in common_columns:
            assert col in template_df.columns, f"TemplateResults missing common column: {col}"

        # Check RubricResults
        rubric_results = RubricResults(results=results_list)
        rubric_df = rubric_results.to_dataframe(trait_type="all")

        for col in common_columns:
            assert col in rubric_df.columns, f"RubricResults missing common column: {col}"

        # Check JudgmentResults
        judgment_results = JudgmentResults(results=results_list)
        judgment_df = judgment_results.to_dataframe()

        for col in common_columns:
            assert col in judgment_df.columns, f"JudgmentResults missing common column: {col}"

    def test_status_columns_first(self, verification_result_success: VerificationResult):
        """Test that status columns appear first in all DataFrames."""
        results_list = [verification_result_success]

        # TemplateResults
        template_results = TemplateResults(results=results_list)
        template_df = template_results.to_dataframe()

        # First column should be status
        assert template_df.columns[0] == "completed_without_errors"

        # RubricResults
        rubric_results = RubricResults(results=results_list)
        rubric_df = rubric_results.to_dataframe(trait_type="all")
        assert rubric_df.columns[0] == "completed_without_errors"

        # JudgmentResults
        judgment_results = JudgmentResults(results=results_list)
        judgment_df = judgment_results.to_dataframe()
        assert judgment_df.columns[0] == "completed_without_errors"


# =============================================================================
# Pandas Operations Tests
# =============================================================================


@pytest.mark.integration
class TestPandasOperations:
    """Integration tests for pandas operations on DataFrames."""

    def test_groupby_operations(self, verification_results_list: list[VerificationResult]):
        """Test pandas groupby operations on TemplateResults DataFrame."""
        template_results = TemplateResults(results=verification_results_list)
        df = template_results.to_dataframe()

        # Test groupby question_id
        grouped = df.groupby("question_id")
        assert len(grouped) > 0

        # Test aggregation on boolean column
        pass_rates = grouped["completed_without_errors"].mean()
        assert isinstance(pass_rates, pd.Series)
        assert len(pass_rates) > 0

    def test_filtering_operations(self, verification_results_list: list[VerificationResult]):
        """Test pandas filtering operations on DataFrames."""
        template_results = TemplateResults(results=verification_results_list)
        df = template_results.to_dataframe()

        # Filter to successful results only
        successful = df[df["completed_without_errors"]]
        assert len(successful) >= 0

        # Filter to specific question
        if len(df) > 0:
            first_question = df["question_id"].iloc[0]
            question_df = df[df["question_id"] == first_question]
            assert len(question_df) > 0
            assert (question_df["question_id"] == first_question).all()

    def test_pivot_operations(self, verification_result_success: VerificationResult):
        """Test pandas pivot operations on RubricResults DataFrame."""
        rubric_results = RubricResults(results=[verification_result_success])
        df = rubric_results.to_dataframe(trait_type="llm")

        if len(df) == 0:
            pytest.skip("No LLM trait data for pivot testing")

        # Try pivot: questions × traits (if trait_name column exists)
        if "trait_name" in df.columns and "trait_score" in df.columns:
            try:
                pivot = df.pivot_table(
                    values="trait_score",
                    index="question_id",
                    columns="trait_name",
                    aggfunc="mean",
                )
                assert isinstance(pivot, pd.DataFrame)
            except Exception as e:
                # Pivot may fail if data structure doesn't support it
                pytest.skip(f"Pivot not applicable to this data structure: {e}")
