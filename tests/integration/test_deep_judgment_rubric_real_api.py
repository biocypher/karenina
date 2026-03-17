"""Integration tests for Deep Judgment Rubrics using fixtures.

These tests validate deep judgment rubric functionality using fixture-created
VerificationResult objects, following the fixture-based testing pattern.

Tests are marked with:
- @pytest.mark.integration: Tests that combine multiple components
- @pytest.mark.deep_judgment: Tests specific to deep judgment rubrics

Run with: pytest tests/integration/test_deep_judgment_rubric_real_api.py -v
"""

from datetime import UTC, datetime

import pytest

from karenina.schemas.results import RubricJudgmentResults, RubricResults
from karenina.schemas.verification import (
    VerificationResult,
    VerificationResultDeepJudgmentRubric,
    VerificationResultMetadata,
    VerificationResultRubric,
)
from karenina.schemas.verification.model_identity import ModelIdentity

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
def rubric_result_with_multiple_traits() -> VerificationResultRubric:
    """Create a rubric result with multiple LLM traits."""
    return VerificationResultRubric(
        rubric_evaluation_performed=True,
        llm_trait_scores={
            "Clarity": 4,
            "Completeness": 5,
            "Accuracy": 4,
        },
    )


@pytest.fixture
def deep_judgment_rubric_full() -> VerificationResultDeepJudgmentRubric:
    """Create a comprehensive deep judgment rubric result.

    This fixture simulates what would be returned from a full deep judgment
    evaluation with enable_all mode, including excerpts, reasoning, scores,
    and metadata for all traits.
    """
    return VerificationResultDeepJudgmentRubric(
        deep_judgment_rubric_performed=True,
        deep_judgment_rubric_scores={
            "Clarity": 4,
            "Completeness": 5,
            "Accuracy": 4,
        },
        standard_rubric_scores={},  # Empty when enable_all mode is used
        rubric_trait_reasoning={
            "Clarity": "The response is well-organized with clear paragraph structure and logical flow.",
            "Completeness": "All aspects of the question were thoroughly addressed with supporting details.",
            "Accuracy": "The information provided is factually correct and well-supported.",
        },
        extracted_rubric_excerpts={
            "Clarity": [
                {
                    "text": "The answer clearly explains the concept step by step",
                    "confidence": "high",
                    "similarity_score": 0.95,
                },
                {
                    "text": "Each point is presented in a logical sequence",
                    "confidence": "high",
                    "similarity_score": 0.92,
                },
            ],
            "Completeness": [
                {
                    "text": "The response covers all key aspects including background, mechanism, and applications",
                    "confidence": "high",
                    "similarity_score": 0.94,
                },
            ],
            "Accuracy": [
                {
                    "text": "BCL-2 is correctly identified as an anti-apoptotic protein",
                    "confidence": "high",
                    "similarity_score": 0.97,
                },
                {
                    "text": "The mechanism of blocking BAX and BAK is accurately described",
                    "confidence": "high",
                    "similarity_score": 0.93,
                },
            ],
        },
        trait_metadata={
            "Clarity": {
                "stages_completed": ["excerpt_extraction", "reasoning_generation", "score_extraction"],
                "model_calls": 3,
                "had_excerpts": True,
                "excerpt_retry_count": 0,
            },
            "Completeness": {
                "stages_completed": ["excerpt_extraction", "reasoning_generation", "score_extraction"],
                "model_calls": 3,
                "had_excerpts": True,
                "excerpt_retry_count": 0,
            },
            "Accuracy": {
                "stages_completed": ["excerpt_extraction", "reasoning_generation", "score_extraction"],
                "model_calls": 3,
                "had_excerpts": True,
                "excerpt_retry_count": 0,
            },
        },
        traits_without_valid_excerpts=[],
    )


@pytest.fixture
def verification_result_full_dj(
    rubric_result_with_multiple_traits: VerificationResultRubric,
    deep_judgment_rubric_full: VerificationResultDeepJudgmentRubric,
) -> VerificationResult:
    """Create a verification result with full deep judgment data."""
    return VerificationResult(
        metadata=_create_metadata("dj_full_q1", "gpt-4"),
        rubric=rubric_result_with_multiple_traits,
        deep_judgment_rubric=deep_judgment_rubric_full,
    )


@pytest.fixture
def deep_judgment_rubric_for_export() -> VerificationResultDeepJudgmentRubric:
    """Create a deep judgment rubric result for export testing."""
    return VerificationResultDeepJudgmentRubric(
        deep_judgment_rubric_performed=True,
        deep_judgment_rubric_scores={"MentionsAntiApoptotic": True},
        rubric_trait_reasoning={"MentionsAntiApoptotic": "The answer explicitly mentions BCL-2's anti-apoptotic role."},
        extracted_rubric_excerpts={
            "MentionsAntiApoptotic": [
                {
                    "text": "BCL-2 is an anti-apoptotic protein",
                    "confidence": "high",
                    "similarity_score": 0.98,
                },
                {
                    "text": "prevents programmed cell death",
                    "confidence": "high",
                    "similarity_score": 0.95,
                },
            ]
        },
        trait_metadata={
            "MentionsAntiApoptotic": {
                "stages_completed": ["excerpt_extraction", "reasoning_generation", "score_extraction"],
                "model_calls": 3,
                "had_excerpts": True,
                "excerpt_retry_count": 0,
            }
        },
    )


@pytest.fixture
def verification_result_for_export(
    deep_judgment_rubric_for_export: VerificationResultDeepJudgmentRubric,
) -> VerificationResult:
    """Create a verification result for export testing."""
    rubric = VerificationResultRubric(
        rubric_evaluation_performed=True,
        llm_trait_scores={"MentionsAntiApoptotic": True},
    )
    return VerificationResult(
        metadata=_create_metadata("export_q1", "gpt-4"),
        rubric=rubric,
        deep_judgment_rubric=deep_judgment_rubric_for_export,
    )


# =============================================================================
# Deep Judgment Rubric Structure Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.deep_judgment
class TestDeepJudgmentRubricStructure:
    """Test the structure and content of deep judgment rubric results."""

    def test_deep_judgment_performed_flag(self, verification_result_full_dj: VerificationResult):
        """Test that deep judgment performed flag is set correctly."""
        dj_rubric = verification_result_full_dj.deep_judgment_rubric
        assert dj_rubric is not None, "Deep judgment rubric data missing"
        assert dj_rubric.deep_judgment_rubric_performed is True, "Deep judgment not performed"

    def test_all_traits_have_scores(self, verification_result_full_dj: VerificationResult):
        """Test that all LLM traits have deep judgment scores."""
        dj_rubric = verification_result_full_dj.deep_judgment_rubric
        expected_traits = ["Clarity", "Completeness", "Accuracy"]

        assert dj_rubric.deep_judgment_rubric_scores is not None, "No scores generated"

        for trait_name in expected_traits:
            assert trait_name in dj_rubric.deep_judgment_rubric_scores, f"Score missing for {trait_name}"
            score = dj_rubric.deep_judgment_rubric_scores[trait_name]
            assert isinstance(score, int | bool), f"Expected int/bool score for {trait_name}, got {type(score)}"

    def test_all_traits_have_reasoning(self, verification_result_full_dj: VerificationResult):
        """Test that all traits have reasoning generated."""
        dj_rubric = verification_result_full_dj.deep_judgment_rubric
        expected_traits = ["Clarity", "Completeness", "Accuracy"]

        assert dj_rubric.rubric_trait_reasoning is not None, "No reasoning generated"

        for trait_name in expected_traits:
            assert trait_name in dj_rubric.rubric_trait_reasoning, f"Reasoning missing for {trait_name}"
            reasoning = dj_rubric.rubric_trait_reasoning[trait_name]
            assert len(reasoning) > 0, f"Empty reasoning for {trait_name}"

    def test_excerpts_structure(self, verification_result_full_dj: VerificationResult):
        """Test that excerpts have the expected structure."""
        dj_rubric = verification_result_full_dj.deep_judgment_rubric

        assert dj_rubric.extracted_rubric_excerpts is not None, "No excerpts extracted"
        assert len(dj_rubric.extracted_rubric_excerpts) > 0, "No traits have excerpts"

        # Test Clarity trait excerpts in detail
        if "Clarity" in dj_rubric.extracted_rubric_excerpts:
            excerpts = dj_rubric.extracted_rubric_excerpts["Clarity"]
            assert len(excerpts) > 0, "No excerpts for Clarity"

            for excerpt in excerpts:
                assert "text" in excerpt, "Excerpt missing 'text' field"
                assert "confidence" in excerpt, "Excerpt missing 'confidence' field"
                assert "similarity_score" in excerpt, "Excerpt missing 'similarity_score' field"

    def test_trait_metadata_structure(self, verification_result_full_dj: VerificationResult):
        """Test that trait metadata has expected fields."""
        dj_rubric = verification_result_full_dj.deep_judgment_rubric

        assert dj_rubric.trait_metadata is not None, "No trait metadata"
        assert "Clarity" in dj_rubric.trait_metadata, "Metadata missing for Clarity"

        metadata = dj_rubric.trait_metadata["Clarity"]
        assert "model_calls" in metadata, "model_calls missing from metadata"
        assert "stages_completed" in metadata, "stages_completed missing from metadata"
        assert metadata["model_calls"] >= 1, f"Expected >= 1 model calls, got {metadata['model_calls']}"

    def test_no_standard_scores_in_enable_all_mode(self, verification_result_full_dj: VerificationResult):
        """Test that enable_all mode produces no standard scores."""
        dj_rubric = verification_result_full_dj.deep_judgment_rubric

        # In enable_all mode, all traits use deep judgment
        assert dj_rubric.standard_rubric_scores is None or len(dj_rubric.standard_rubric_scores) == 0, (
            "Expected no standard scores with enable_all mode"
        )

    def test_no_auto_fail_for_valid_excerpts(self, verification_result_full_dj: VerificationResult):
        """Test that traits with valid excerpts don't trigger auto-fail."""
        dj_rubric = verification_result_full_dj.deep_judgment_rubric

        if dj_rubric.traits_without_valid_excerpts:
            assert "Clarity" not in dj_rubric.traits_without_valid_excerpts, "Clarity should not have failed validation"


# =============================================================================
# DataFrame Export Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.deep_judgment
class TestDeepJudgmentDataframeExport:
    """Test DataFrame export with deep judgment rubric data."""

    def test_standard_export_with_results(self, verification_result_for_export: VerificationResult):
        """Test standard RubricResults export with deep judgment data."""
        rubric_results = RubricResults(results=[verification_result_for_export])
        # Use "llm" to include both score and binary traits (fixture has boolean trait)
        df = rubric_results.to_dataframe(trait_type="llm")

        # Should have rows for traits
        assert len(df) > 0, "Empty dataframe"

        # Check core columns
        assert "trait_name" in df.columns
        assert "trait_score" in df.columns
        assert "question_id" in df.columns

    def test_detailed_export_structure(self, verification_result_for_export: VerificationResult):
        """Test detailed RubricJudgmentResults export structure."""
        judgment_results = RubricJudgmentResults(results=[verification_result_for_export])
        df = judgment_results.to_dataframe()

        # Should have excerpt-level columns
        expected_cols = [
            "trait_name",
            "trait_score",
            "excerpt_index",
            "excerpt_text",
            "excerpt_confidence",
            "trait_reasoning",
        ]

        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_excerpt_explosion(self, verification_result_for_export: VerificationResult):
        """Test that excerpts are exploded into separate rows."""
        judgment_results = RubricJudgmentResults(results=[verification_result_for_export])
        df = judgment_results.to_dataframe()

        # We have 2 excerpts for MentionsAntiApoptotic
        trait_rows = df[df["trait_name"] == "MentionsAntiApoptotic"]
        assert len(trait_rows) == 2, f"Expected 2 rows (one per excerpt), got {len(trait_rows)}"

        # Check excerpt indices
        assert trait_rows.iloc[0]["excerpt_index"] == 0
        assert trait_rows.iloc[1]["excerpt_index"] == 1

    def test_excerpt_content_preserved(self, verification_result_for_export: VerificationResult):
        """Test that excerpt content is preserved in export."""
        judgment_results = RubricJudgmentResults(results=[verification_result_for_export])
        df = judgment_results.to_dataframe()

        # Filter to our trait
        trait_rows = df[df["trait_name"] == "MentionsAntiApoptotic"]

        if len(trait_rows) > 0:
            first_row = trait_rows.iloc[0]

            # Check that excerpt data is populated
            assert first_row["excerpt_text"] is not None, "Missing excerpt_text"
            assert first_row["excerpt_confidence"] is not None, "Missing excerpt_confidence"

            # Check specific content
            assert "anti-apoptotic" in first_row["excerpt_text"].lower()

    def test_reasoning_preserved_in_export(self, verification_result_for_export: VerificationResult):
        """Test that reasoning is preserved in detailed export."""
        judgment_results = RubricJudgmentResults(results=[verification_result_for_export])
        df = judgment_results.to_dataframe()

        # Each row should have reasoning
        trait_rows = df[df["trait_name"] == "MentionsAntiApoptotic"]

        if len(trait_rows) > 0:
            for _, row in trait_rows.iterrows():
                assert row["trait_reasoning"] is not None, "No reasoning in row"
                assert len(row["trait_reasoning"]) > 0, "Empty reasoning"


# =============================================================================
# Multi-trait Deep Judgment Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.deep_judgment
class TestMultiTraitDeepJudgment:
    """Test deep judgment with multiple traits."""

    def test_all_traits_evaluated(self, verification_result_full_dj: VerificationResult):
        """Test that all traits are evaluated with deep judgment."""
        dj_rubric = verification_result_full_dj.deep_judgment_rubric

        expected_traits = ["Clarity", "Completeness", "Accuracy"]

        # All traits should have scores
        for trait in expected_traits:
            assert trait in dj_rubric.deep_judgment_rubric_scores

        # All traits should have reasoning
        for trait in expected_traits:
            assert trait in dj_rubric.rubric_trait_reasoning

        # All traits should have metadata
        for trait in expected_traits:
            assert trait in dj_rubric.trait_metadata

    def test_multiple_excerpts_per_trait(self, verification_result_full_dj: VerificationResult):
        """Test that traits can have multiple excerpts."""
        dj_rubric = verification_result_full_dj.deep_judgment_rubric

        # Clarity has 2 excerpts in our fixture
        clarity_excerpts = dj_rubric.extracted_rubric_excerpts.get("Clarity", [])
        assert len(clarity_excerpts) == 2, f"Expected 2 excerpts for Clarity, got {len(clarity_excerpts)}"

        # Accuracy has 2 excerpts
        accuracy_excerpts = dj_rubric.extracted_rubric_excerpts.get("Accuracy", [])
        assert len(accuracy_excerpts) == 2, f"Expected 2 excerpts for Accuracy, got {len(accuracy_excerpts)}"

    def test_trait_count_in_export(self, verification_result_full_dj: VerificationResult):
        """Test that export has correct number of trait entries."""
        rubric_results = RubricResults(results=[verification_result_full_dj])
        df = rubric_results.to_dataframe(trait_type="llm_score")

        # Should have 3 traits
        unique_traits = df["trait_name"].unique()
        assert len(unique_traits) == 3, f"Expected 3 traits, got {len(unique_traits)}"


if __name__ == "__main__":
    # For manual testing
    print("Running Deep Judgment Rubrics Integration Tests...")
    print("=" * 70)
    pytest.main([__file__, "-v"])
