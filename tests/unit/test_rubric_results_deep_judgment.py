"""Unit tests for RubricResults deep judgment dataframe export.

This module tests the dataframe export extensions for deep judgment:
- Standard export with include_deep_judgment flag
- Detailed export with RubricJudgmentResults
- Column structure and content validation
- Excerpt explosion logic
"""

import json

import pytest

# Import and rebuild RubricJudgmentResults to resolve forward references
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.schemas.workflow.rubric_judgment_results import RubricJudgmentResults
from karenina.schemas.workflow.rubric_results import RubricResults
from karenina.schemas.workflow.verification import (
    VerificationResult,
    VerificationResultDeepJudgmentRubric,
    VerificationResultMetadata,
    VerificationResultRubric,
)

RubricJudgmentResults.model_rebuild()


class TestRubricResultsStandardExport:
    """Test standard RubricResults.get_rubrics() with deep judgment flag."""

    @pytest.fixture
    def basic_metadata(self) -> VerificationResultMetadata:
        """Create basic metadata for testing."""
        _answering = ModelIdentity(interface="langchain", model_name="gpt-4")
        _parsing = ModelIdentity(interface="langchain", model_name="gpt-4-mini")
        result_id = VerificationResultMetadata.compute_result_id(
            question_id="q1",
            answering=_answering,
            parsing=_parsing,
            timestamp="2024-01-15T10:30:00",
            replicate=1,
        )
        return VerificationResultMetadata(
            question_id="q1",
            template_id="template1",
            completed_without_errors=True,
            error=None,
            question_text="What is photosynthesis?",
            keywords=["biology"],
            answering=_answering,
            parsing=_parsing,
            execution_time=2.5,
            timestamp="2024-01-15T10:30:00",
            result_id=result_id,
            replicate=1,
        )

    def test_get_rubrics_default_no_deep_judgment(self, basic_metadata):
        """Test that default to_dataframe() does not include deep judgment columns."""
        rubric = VerificationResultRubric(
            rubric_evaluation_performed=True,
            llm_trait_scores={"clarity": 4, "accuracy": 5},
        )

        # Add deep judgment data
        dj_rubric = VerificationResultDeepJudgmentRubric(
            deep_judgment_rubric_performed=True,
            deep_judgment_rubric_scores={"clarity": 4},
            rubric_trait_reasoning={"clarity": "The answer is clear and well-structured."},
            extracted_rubric_excerpts={
                "clarity": [{"text": "Photosynthesis is the process", "confidence": "high", "similarity_score": 0.95}]
            },
        )

        result = VerificationResult(metadata=basic_metadata, rubric=rubric, deep_judgment_rubric=dj_rubric)

        # Default: include_deep_judgment=False (backward compatible)
        rubric_results = RubricResults(results=[result], include_deep_judgment=False)
        df = rubric_results.to_dataframe(trait_type="llm_score")

        # Should NOT have deep judgment columns
        assert "trait_reasoning" not in df.columns
        assert "trait_excerpts" not in df.columns
        assert "trait_hallucination_risk" not in df.columns

        # Should still have standard columns
        assert "trait_name" in df.columns
        assert "trait_score" in df.columns

    def test_get_rubrics_with_deep_judgment_flag(self, basic_metadata):
        """Test that include_deep_judgment=True adds 3 deep judgment columns."""
        rubric = VerificationResultRubric(
            rubric_evaluation_performed=True,
            llm_trait_scores={"clarity": 4},
        )

        dj_rubric = VerificationResultDeepJudgmentRubric(
            deep_judgment_rubric_performed=True,
            deep_judgment_rubric_scores={"clarity": 4},
            rubric_trait_reasoning={"clarity": "The answer is well-organized with clear explanations."},
            extracted_rubric_excerpts={
                "clarity": [{"text": "Photosynthesis is the process", "confidence": "high", "similarity_score": 0.95}]
            },
        )

        result = VerificationResult(metadata=basic_metadata, rubric=rubric, deep_judgment_rubric=dj_rubric)

        # With flag: should have deep judgment columns
        rubric_results = RubricResults(results=[result], include_deep_judgment=True)
        df = rubric_results.to_dataframe(trait_type="llm_score")

        # Should have the 3 deep judgment columns
        assert "trait_reasoning" in df.columns
        assert "trait_excerpts" in df.columns
        assert "trait_hallucination_risk" in df.columns

        # Check column names are correct
        expected_dj_columns = ["trait_reasoning", "trait_excerpts", "trait_hallucination_risk"]
        for col in expected_dj_columns:
            assert col in df.columns, f"Missing deep judgment column: {col}"

    def test_deep_judgment_column_content(self, basic_metadata):
        """Test deep judgment column data types and content format."""
        rubric = VerificationResultRubric(
            rubric_evaluation_performed=True,
            llm_trait_scores={"scientific_accuracy": 5},
        )

        dj_rubric = VerificationResultDeepJudgmentRubric(
            deep_judgment_rubric_performed=True,
            deep_judgment_rubric_scores={"scientific_accuracy": 5},
            rubric_trait_reasoning={
                "scientific_accuracy": "The answer provides accurate scientific information with proper terminology."
            },
            extracted_rubric_excerpts={
                "scientific_accuracy": [
                    {"text": "Photosynthesis converts light energy", "confidence": "high", "similarity_score": 0.92},
                    {"text": "occurs in chloroplasts", "confidence": "medium", "similarity_score": 0.88},
                ]
            },
            rubric_hallucination_risk_assessment={
                "scientific_accuracy": {"overall_risk": "low", "per_excerpt_risks": ["none", "low"]}
            },
        )

        result = VerificationResult(metadata=basic_metadata, rubric=rubric, deep_judgment_rubric=dj_rubric)

        rubric_results = RubricResults(results=[result], include_deep_judgment=True)
        df = rubric_results.to_dataframe(trait_type="llm_score")

        row = df.iloc[0]

        # trait_reasoning should be a string
        assert isinstance(row["trait_reasoning"], str)
        assert len(row["trait_reasoning"]) > 0
        assert "accurate" in row["trait_reasoning"].lower()

        # trait_excerpts should be JSON list (or list)
        excerpts = row["trait_excerpts"]
        if isinstance(excerpts, str):
            excerpts = json.loads(excerpts)
        assert isinstance(excerpts, list)
        assert len(excerpts) == 2

        # trait_hallucination_risk should be JSON dict (or dict)
        risk = row["trait_hallucination_risk"]
        if isinstance(risk, str):
            risk = json.loads(risk)
        assert isinstance(risk, dict)
        assert "overall_risk" in risk
        assert risk["overall_risk"] == "low"


class TestRubricJudgmentResultsCreation:
    """Test RubricJudgmentResults instantiation and basic methods."""

    @pytest.fixture
    def basic_metadata(self) -> VerificationResultMetadata:
        """Create basic metadata."""
        _answering = ModelIdentity(interface="langchain", model_name="gpt-4")
        _parsing = ModelIdentity(interface="langchain", model_name="gpt-4-mini")
        result_id = VerificationResultMetadata.compute_result_id(
            question_id="q1",
            answering=_answering,
            parsing=_parsing,
            timestamp="2024-01-15T10:30:00",
            replicate=1,
        )
        return VerificationResultMetadata(
            question_id="q1",
            template_id="template1",
            completed_without_errors=True,
            error=None,
            question_text="Test question",
            keywords=None,
            answering=_answering,
            parsing=_parsing,
            execution_time=1.0,
            timestamp="2024-01-15T10:30:00",
            result_id=result_id,
            replicate=1,
        )

    def test_rubric_judgment_results_creation(self, basic_metadata):
        """Test that RubricJudgmentResults can be instantiated."""
        rubric = VerificationResultRubric(
            rubric_evaluation_performed=True,
            llm_trait_scores={"clarity": 4},
        )

        dj_rubric = VerificationResultDeepJudgmentRubric(
            deep_judgment_rubric_performed=True,
            deep_judgment_rubric_scores={"clarity": 4},
            rubric_trait_reasoning={"clarity": "Clear and well-structured."},
            extracted_rubric_excerpts={"clarity": [{"text": "Test excerpt", "confidence": "high"}]},
        )

        result = VerificationResult(metadata=basic_metadata, rubric=rubric, deep_judgment_rubric=dj_rubric)

        # Should be able to create RubricJudgmentResults
        judgment_results = RubricJudgmentResults(results=[result])

        assert judgment_results is not None
        assert len(judgment_results.results) == 1


class TestRubricJudgmentResultsExcerptExplosion:
    """Test excerpt explosion logic in RubricJudgmentResults.to_dataframe()."""

    @pytest.fixture
    def basic_metadata(self) -> VerificationResultMetadata:
        """Create basic metadata."""
        _answering = ModelIdentity(interface="langchain", model_name="gpt-4")
        _parsing = ModelIdentity(interface="langchain", model_name="gpt-4-mini")
        result_id = VerificationResultMetadata.compute_result_id(
            question_id="q1",
            answering=_answering,
            parsing=_parsing,
            timestamp="2024-01-15T10:30:00",
            replicate=1,
        )
        return VerificationResultMetadata(
            question_id="q1",
            template_id="template1",
            completed_without_errors=True,
            error=None,
            question_text="Test question",
            keywords=None,
            answering=_answering,
            parsing=_parsing,
            execution_time=1.0,
            timestamp="2024-01-15T10:30:00",
            result_id=result_id,
            replicate=1,
        )

    def test_rubric_judgment_results_excerpt_explosion(self, basic_metadata):
        """Test that excerpts are exploded into separate rows."""
        rubric = VerificationResultRubric(
            rubric_evaluation_performed=True,
            llm_trait_scores={"scientific_accuracy": 5},
        )

        # Trait with 3 excerpts
        dj_rubric = VerificationResultDeepJudgmentRubric(
            deep_judgment_rubric_performed=True,
            deep_judgment_rubric_scores={"scientific_accuracy": 5},
            rubric_trait_reasoning={"scientific_accuracy": "Accurate and comprehensive."},
            extracted_rubric_excerpts={
                "scientific_accuracy": [
                    {"text": "Excerpt 1", "confidence": "high", "similarity_score": 0.95},
                    {"text": "Excerpt 2", "confidence": "high", "similarity_score": 0.92},
                    {"text": "Excerpt 3", "confidence": "medium", "similarity_score": 0.88},
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

        result = VerificationResult(metadata=basic_metadata, rubric=rubric, deep_judgment_rubric=dj_rubric)
        judgment_results = RubricJudgmentResults(results=[result])

        df = judgment_results.to_dataframe()

        # Should have 3 rows (one per excerpt)
        assert len(df) == 3

        # Each row should have the same trait_name
        assert all(df["trait_name"] == "scientific_accuracy")

        # excerpt_index should be 0, 1, 2
        assert list(df["excerpt_index"].values) == [0, 1, 2]

        # excerpt_text should be different
        excerpt_texts = df["excerpt_text"].values
        assert "Excerpt 1" in excerpt_texts
        assert "Excerpt 2" in excerpt_texts
        assert "Excerpt 3" in excerpt_texts

        # All rows should share the same trait_reasoning
        assert all(df["trait_reasoning"] == "Accurate and comprehensive.")

    def test_rubric_judgment_results_no_excerpts(self, basic_metadata):
        """Test traits without excerpts get a single row."""
        rubric = VerificationResultRubric(
            rubric_evaluation_performed=True,
            llm_trait_scores={"overall_clarity": 4},
        )

        # Trait without excerpts (excerpt_enabled=False)
        dj_rubric = VerificationResultDeepJudgmentRubric(
            deep_judgment_rubric_performed=True,
            deep_judgment_rubric_scores={"overall_clarity": 4},
            rubric_trait_reasoning={"overall_clarity": "Overall response is clear."},
            extracted_rubric_excerpts={},  # No excerpts
            trait_metadata={
                "overall_clarity": {
                    "stages_completed": ["reasoning_generation", "score_extraction"],
                    "model_calls": 2,
                    "had_excerpts": False,
                    "excerpt_retry_count": 0,
                }
            },
        )

        result = VerificationResult(metadata=basic_metadata, rubric=rubric, deep_judgment_rubric=dj_rubric)
        judgment_results = RubricJudgmentResults(results=[result])

        df = judgment_results.to_dataframe()

        # Should have 1 row (trait without excerpts)
        assert len(df) == 1

        # Excerpt fields should be None
        row = df.iloc[0]
        assert row["excerpt_index"] is None or row["excerpt_index"] == 0
        assert row["excerpt_text"] is None
        assert row["excerpt_confidence"] is None
        assert row["excerpt_similarity_score"] is None

        # trait_had_excerpts should be False
        assert not row["trait_had_excerpts"]


class TestRubricJudgmentResultsMetadata:
    """Test metadata columns in RubricJudgmentResults."""

    @pytest.fixture
    def basic_metadata(self) -> VerificationResultMetadata:
        """Create basic metadata."""
        _answering = ModelIdentity(interface="langchain", model_name="gpt-4")
        _parsing = ModelIdentity(interface="langchain", model_name="gpt-4-mini")
        result_id = VerificationResultMetadata.compute_result_id(
            question_id="q1",
            answering=_answering,
            parsing=_parsing,
            timestamp="2024-01-15T10:30:00",
            replicate=1,
        )
        return VerificationResultMetadata(
            question_id="q1",
            template_id="template1",
            completed_without_errors=True,
            error=None,
            question_text="Test question",
            keywords=None,
            answering=_answering,
            parsing=_parsing,
            execution_time=1.0,
            timestamp="2024-01-15T10:30:00",
            result_id=result_id,
            replicate=1,
        )

    def test_rubric_judgment_results_metadata(self, basic_metadata):
        """Test that metadata columns are present and correct."""
        rubric = VerificationResultRubric(
            rubric_evaluation_performed=True,
            llm_trait_scores={"clarity": 4},
        )

        dj_rubric = VerificationResultDeepJudgmentRubric(
            deep_judgment_rubric_performed=True,
            deep_judgment_rubric_scores={"clarity": 4},
            rubric_trait_reasoning={"clarity": "Clear."},
            extracted_rubric_excerpts={"clarity": [{"text": "Test", "confidence": "high"}]},
            trait_metadata={
                "clarity": {
                    "stages_completed": ["excerpt_extraction", "reasoning_generation", "score_extraction"],
                    "model_calls": 3,
                    "had_excerpts": True,
                    "excerpt_retry_count": 2,
                    "excerpt_validation_failed": False,
                }
            },
        )

        result = VerificationResult(metadata=basic_metadata, rubric=rubric, deep_judgment_rubric=dj_rubric)
        judgment_results = RubricJudgmentResults(results=[result])

        df = judgment_results.to_dataframe()

        # Check metadata columns exist
        assert "trait_model_calls" in df.columns
        assert "trait_excerpt_retries" in df.columns
        assert "trait_stages_completed" in df.columns

        row = df.iloc[0]

        # Check values
        assert row["trait_model_calls"] == 3
        assert row["trait_excerpt_retries"] == 2

        # trait_stages_completed should be JSON serialized list
        stages = row["trait_stages_completed"]
        if isinstance(stages, str):
            stages = json.loads(stages)
        assert isinstance(stages, list)
        assert "excerpt_extraction" in stages
        assert "reasoning_generation" in stages
        assert "score_extraction" in stages


class TestRubricJudgmentResultsMixedTraits:
    """Test handling of mixed deep judgment and standard traits."""

    @pytest.fixture
    def basic_metadata(self) -> VerificationResultMetadata:
        """Create basic metadata."""
        _answering = ModelIdentity(interface="langchain", model_name="gpt-4")
        _parsing = ModelIdentity(interface="langchain", model_name="gpt-4-mini")
        result_id = VerificationResultMetadata.compute_result_id(
            question_id="q1",
            answering=_answering,
            parsing=_parsing,
            timestamp="2024-01-15T10:30:00",
            replicate=1,
        )
        return VerificationResultMetadata(
            question_id="q1",
            template_id="template1",
            completed_without_errors=True,
            error=None,
            question_text="Test question",
            keywords=None,
            answering=_answering,
            parsing=_parsing,
            execution_time=1.0,
            timestamp="2024-01-15T10:30:00",
            result_id=result_id,
            replicate=1,
        )

    def test_mixed_dj_and_standard_traits(self, basic_metadata):
        """Test rubric with both deep judgment and standard traits."""
        # Both DJ and standard traits
        rubric = VerificationResultRubric(
            rubric_evaluation_performed=True,
            llm_trait_scores={
                "clarity": 4,  # Deep judgment trait
                "completeness": 5,  # Standard trait
            },
        )

        dj_rubric = VerificationResultDeepJudgmentRubric(
            deep_judgment_rubric_performed=True,
            deep_judgment_rubric_scores={"clarity": 4},  # Only DJ trait
            standard_rubric_scores={"completeness": 5},  # Standard trait
            rubric_trait_reasoning={"clarity": "Clear and organized."},
            extracted_rubric_excerpts={"clarity": [{"text": "Test excerpt", "confidence": "high"}]},
        )

        result = VerificationResult(metadata=basic_metadata, rubric=rubric, deep_judgment_rubric=dj_rubric)

        # RubricResults should include both traits
        rubric_results = RubricResults(results=[result])
        df_rubric = rubric_results.to_dataframe(trait_type="llm_score")
        assert len(df_rubric) == 2
        assert set(df_rubric["trait_name"].values) == {"clarity", "completeness"}

        # RubricJudgmentResults should only include DJ trait
        judgment_results = RubricJudgmentResults(results=[result])
        df_judgment = judgment_results.to_dataframe()

        # Should only have the deep judgment trait
        assert len(df_judgment) >= 1
        assert all(df_judgment["trait_name"] == "clarity")

        # Standard trait should not appear in detailed export
        assert "completeness" not in df_judgment["trait_name"].values


class TestJSONSerializationComplexFields:
    """Test JSON serialization of complex fields."""

    @pytest.fixture
    def basic_metadata(self) -> VerificationResultMetadata:
        """Create basic metadata."""
        _answering = ModelIdentity(interface="langchain", model_name="gpt-4")
        _parsing = ModelIdentity(interface="langchain", model_name="gpt-4-mini")
        result_id = VerificationResultMetadata.compute_result_id(
            question_id="q1",
            answering=_answering,
            parsing=_parsing,
            timestamp="2024-01-15T10:30:00",
            replicate=1,
        )
        return VerificationResultMetadata(
            question_id="q1",
            template_id="template1",
            completed_without_errors=True,
            error=None,
            question_text="Test question",
            keywords=None,
            answering=_answering,
            parsing=_parsing,
            execution_time=1.0,
            timestamp="2024-01-15T10:30:00",
            result_id=result_id,
            replicate=1,
        )

    def test_json_serialization_excerpts_list(self, basic_metadata):
        """Test that excerpts list is properly JSON serialized."""
        rubric = VerificationResultRubric(
            rubric_evaluation_performed=True,
            llm_trait_scores={"clarity": 4},
        )

        dj_rubric = VerificationResultDeepJudgmentRubric(
            deep_judgment_rubric_performed=True,
            deep_judgment_rubric_scores={"clarity": 4},
            rubric_trait_reasoning={"clarity": "Clear."},
            extracted_rubric_excerpts={
                "clarity": [
                    {"text": "Excerpt 1", "confidence": "high", "similarity_score": 0.95},
                    {"text": "Excerpt 2", "confidence": "medium", "similarity_score": 0.88},
                ]
            },
        )

        result = VerificationResult(metadata=basic_metadata, rubric=rubric, deep_judgment_rubric=dj_rubric)
        # include_deep_judgment is passed to RubricResults() init, not to_dataframe()
        rubric_results = RubricResults(results=[result], include_deep_judgment=True)

        df = rubric_results.to_dataframe(trait_type="llm_score")

        # excerpts should be JSON serialized
        excerpts = df.iloc[0]["trait_excerpts"]
        if isinstance(excerpts, str):
            # Should be valid JSON
            parsed = json.loads(excerpts)
            assert isinstance(parsed, list)
            assert len(parsed) == 2
        else:
            # Or already a list
            assert isinstance(excerpts, list)

    def test_json_serialization_search_results(self, basic_metadata):
        """Test that search results are properly JSON serialized."""
        rubric = VerificationResultRubric(
            rubric_evaluation_performed=True,
            llm_trait_scores={"scientific_accuracy": 5},
        )

        dj_rubric = VerificationResultDeepJudgmentRubric(
            deep_judgment_rubric_performed=True,
            deep_judgment_rubric_scores={"scientific_accuracy": 5},
            rubric_trait_reasoning={"scientific_accuracy": "Accurate."},
            extracted_rubric_excerpts={
                "scientific_accuracy": [
                    {
                        "text": "Photosynthesis converts light energy",
                        "confidence": "high",
                        "similarity_score": 0.95,
                        "search_results": ["Result 1 from search", "Result 2 from search"],
                        "hallucination_risk": "low",
                        "hallucination_justification": "Strong external evidence found.",
                    }
                ]
            },
            rubric_hallucination_risk_assessment={
                "scientific_accuracy": {"overall_risk": "low", "per_excerpt_risks": ["low"]}
            },
        )

        result = VerificationResult(metadata=basic_metadata, rubric=rubric, deep_judgment_rubric=dj_rubric)
        judgment_results = RubricJudgmentResults(results=[result])

        df = judgment_results.to_dataframe()

        row = df.iloc[0]

        # Check that search_results is present (if column exists)
        if "excerpt_search_results" in df.columns:
            search_results = row["excerpt_search_results"]
            if isinstance(search_results, str):
                parsed = json.loads(search_results)
                assert isinstance(parsed, list)
            else:
                assert isinstance(search_results, list) or search_results is None
