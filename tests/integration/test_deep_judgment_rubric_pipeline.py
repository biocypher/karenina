"""Integration tests for deep judgment rubric DataFrame exports.

This module tests the DataFrame export methods for deep judgment rubric results
using fixture-created VerificationResult objects.

Tests are marked with:
- @pytest.mark.integration: Tests that combine multiple components
- @pytest.mark.deep_judgment: Tests specific to deep judgment rubrics

Run with: pytest tests/integration/test_deep_judgment_rubric_pipeline.py -v
"""

from datetime import UTC, datetime

import pytest

from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.schemas.workflow import (
    VerificationResult,
    VerificationResultDeepJudgmentRubric,
    VerificationResultMetadata,
    VerificationResultRubric,
)
from karenina.schemas.workflow.rubric_judgment_results import RubricJudgmentResults
from karenina.schemas.workflow.rubric_results import RubricResults

RubricJudgmentResults.model_rebuild()


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
    _answering = ModelIdentity(interface="langchain", model_name=answering_model)
    _parsing = ModelIdentity(interface="langchain", model_name="claude-haiku-4-5")
    return VerificationResultMetadata(
        question_id=question_id,
        template_id="test-template-id",
        completed_without_errors=completed,
        error=error,
        question_text=f"Question text for {question_id}",
        raw_answer="Expected answer",
        answering=_answering,
        parsing=_parsing,
        execution_time=1.5,
        timestamp=timestamp,
        result_id=VerificationResultMetadata.compute_result_id(
            question_id=question_id,
            answering=_answering,
            parsing=_parsing,
            timestamp=timestamp,
        ),
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rubric_result_with_llm_traits() -> VerificationResultRubric:
    """Create a rubric result with LLM trait scores."""
    return VerificationResultRubric(
        rubric_evaluation_performed=True,
        llm_trait_scores={
            "scientific_accuracy": 5,
            "completeness": 4,
        },
    )


@pytest.fixture
def deep_judgment_rubric_result() -> VerificationResultDeepJudgmentRubric:
    """Create a deep judgment rubric result with excerpts and reasoning."""
    return VerificationResultDeepJudgmentRubric(
        deep_judgment_rubric_performed=True,
        deep_judgment_rubric_scores={"scientific_accuracy": 5},
        standard_rubric_scores={"completeness": 4},
        rubric_trait_reasoning={"scientific_accuracy": "The answer provides accurate scientific information."},
        extracted_rubric_excerpts={
            "scientific_accuracy": [
                {"text": "Excerpt 1", "confidence": "high", "similarity_score": 0.95},
                {"text": "Excerpt 2", "confidence": "high", "similarity_score": 0.92},
            ]
        },
        trait_metadata={
            "scientific_accuracy": {
                "stages_completed": ["excerpt_extraction", "reasoning_generation", "score_extraction"],
                "model_calls": 3,
                "had_excerpts": True,
                "excerpt_retry_count": 0,
            }
        },
    )


@pytest.fixture
def verification_result_with_dj_rubric(
    rubric_result_with_llm_traits: VerificationResultRubric,
    deep_judgment_rubric_result: VerificationResultDeepJudgmentRubric,
) -> VerificationResult:
    """Create a verification result with deep judgment rubric data."""
    return VerificationResult(
        metadata=_create_metadata("dj_q1", "gpt-4"),
        rubric=rubric_result_with_llm_traits,
        deep_judgment_rubric=deep_judgment_rubric_result,
    )


@pytest.fixture
def verification_results_list(
    verification_result_with_dj_rubric: VerificationResult,
) -> list[VerificationResult]:
    """Create a list of verification results."""
    return [verification_result_with_dj_rubric]


# =============================================================================
# DataFrame Export Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.deep_judgment
class TestBothDataframeExportMethods:
    """Test both standard and detailed dataframe export methods."""

    def test_both_dataframe_export_methods(self, verification_results_list: list[VerificationResult]):
        """Test both standard (RubricResults) and detailed (RubricJudgmentResults) exports."""
        results = verification_results_list

        # Standard export (RubricResults)
        rubric_results = RubricResults(results=results)
        df_standard = rubric_results.to_dataframe(trait_type="llm_score")

        # Should have 2 rows (2 traits)
        assert len(df_standard) == 2
        assert set(df_standard["trait_name"].values) == {"scientific_accuracy", "completeness"}

        # Standard columns
        assert "trait_name" in df_standard.columns
        assert "trait_score" in df_standard.columns
        assert "trait_type" in df_standard.columns

        # Detailed export (RubricJudgmentResults)
        judgment_results = RubricJudgmentResults(results=results)
        df_detailed = judgment_results.to_dataframe()

        # Should have 2 rows (2 excerpts for scientific_accuracy)
        assert len(df_detailed) == 2
        assert all(df_detailed["trait_name"] == "scientific_accuracy")

        # Detailed columns
        assert "excerpt_index" in df_detailed.columns
        assert "excerpt_text" in df_detailed.columns
        assert "excerpt_confidence" in df_detailed.columns
        assert "trait_reasoning" in df_detailed.columns

        # Check excerpt explosion
        assert df_detailed.iloc[0]["excerpt_index"] == 0
        assert df_detailed.iloc[1]["excerpt_index"] == 1
        assert df_detailed.iloc[0]["excerpt_text"] == "Excerpt 1"
        assert df_detailed.iloc[1]["excerpt_text"] == "Excerpt 2"


@pytest.mark.integration
@pytest.mark.deep_judgment
class TestDataframeStructureValidation:
    """Validate structure and content of exported dataframes."""

    def test_standard_export_columns(self):
        """Test that standard export has expected columns."""
        metadata = _create_metadata("col_test_q1", "gpt-4")

        rubric = VerificationResultRubric(
            rubric_evaluation_performed=True,
            llm_trait_scores={"clarity": 4},
        )

        result = VerificationResult(metadata=metadata, rubric=rubric)
        rubric_results = RubricResults(results=[result])

        df = rubric_results.to_dataframe(trait_type="llm_score")

        # Expected standard columns
        expected_columns = [
            "completed_without_errors",
            "question_id",
            "trait_name",
            "trait_score",
            "trait_type",
            "answering_model",
            "parsing_model",
        ]

        for col in expected_columns:
            assert col in df.columns, f"Missing expected column: {col}"

    def test_detailed_export_columns(self):
        """Test that detailed export has expected columns."""
        metadata = _create_metadata("detail_col_q1", "gpt-4")

        rubric = VerificationResultRubric(
            rubric_evaluation_performed=True,
            llm_trait_scores={"clarity": 4},
        )

        dj_rubric = VerificationResultDeepJudgmentRubric(
            deep_judgment_rubric_performed=True,
            deep_judgment_rubric_scores={"clarity": 4},
            rubric_trait_reasoning={"clarity": "Clear."},
            extracted_rubric_excerpts={"clarity": [{"text": "Test", "confidence": "high", "similarity_score": 0.9}]},
            trait_metadata={
                "clarity": {
                    "stages_completed": ["excerpt_extraction"],
                    "model_calls": 1,
                    "had_excerpts": True,
                    "excerpt_retry_count": 0,
                }
            },
        )

        result = VerificationResult(metadata=metadata, rubric=rubric, deep_judgment_rubric=dj_rubric)
        judgment_results = RubricJudgmentResults(results=[result])

        df = judgment_results.to_dataframe()

        # Expected detailed columns
        expected_detailed_columns = [
            "completed_without_errors",
            "question_id",
            "trait_name",
            "trait_score",
            "trait_reasoning",
            "excerpt_index",
            "excerpt_text",
            "excerpt_confidence",
        ]

        for col in expected_detailed_columns:
            assert col in df.columns, f"Missing expected detailed column: {col}"


@pytest.mark.integration
@pytest.mark.deep_judgment
class TestDeepJudgmentRubricDataFrame:
    """Test DataFrame creation for deep judgment rubric results."""

    def test_dataframe_has_dj_specific_columns(self, verification_results_list: list[VerificationResult]):
        """Test that DataFrame has deep judgment specific columns."""
        judgment_results = RubricJudgmentResults(results=verification_results_list)

        df = judgment_results.to_dataframe()

        # Should have excerpt-specific columns
        assert "excerpt_text" in df.columns
        assert "excerpt_confidence" in df.columns
        assert "trait_reasoning" in df.columns

    def test_dataframe_excerpt_explosion(self, verification_results_list: list[VerificationResult]):
        """Test that excerpts are exploded into separate rows."""
        judgment_results = RubricJudgmentResults(results=verification_results_list)

        df = judgment_results.to_dataframe()

        # Each excerpt should be a separate row
        # We have 2 excerpts for scientific_accuracy
        scientific_accuracy_rows = df[df["trait_name"] == "scientific_accuracy"]
        assert len(scientific_accuracy_rows) == 2
