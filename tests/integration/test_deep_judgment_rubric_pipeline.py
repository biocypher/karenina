"""Integration tests for deep judgment rubric evaluation pipeline.

This module tests the end-to-end deep judgment rubric evaluation flow:
- Full verification pipeline with deep judgment-enabled rubrics
- Auto-fail mechanism after retry exhaustion
- Mixed standard and deep judgment traits
- Both dataframe export methods (standard + detailed)

Tests are marked with:
- @pytest.mark.integration: Slow tests that run real verification
- @pytest.mark.requires_api: Tests that need API access
- @pytest.mark.deep_judgment: Tests specific to deep judgment rubrics

Run with: pytest tests/integration/test_deep_judgment_rubric_pipeline.py -v
Skip with: pytest -m "not integration"
"""

from unittest.mock import Mock, patch

import pytest

from karenina.benchmark import Benchmark
from karenina.schemas import ModelConfig, VerificationConfig
from karenina.schemas.domain import LLMRubricTrait, Rubric

# Import and rebuild RubricJudgmentResults to resolve forward references
from karenina.schemas.workflow.rubric_judgment_results import RubricJudgmentResults

RubricJudgmentResults.model_rebuild()


@pytest.fixture
def test_model_config() -> ModelConfig:
    """Create test model configuration."""
    return ModelConfig(
        id="test-dj-model",
        model_provider="openai",
        model_name="gpt-4.1-mini",
        temperature=0.0,
        interface="langchain",
        system_prompt="You are a helpful assistant.",
    )


@pytest.fixture
def dj_verification_config(test_model_config: ModelConfig) -> VerificationConfig:
    """Create verification config with deep judgment enabled."""
    return VerificationConfig(
        answering_models=[test_model_config],
        parsing_models=[test_model_config],
        evaluation_mode="rubric_only",
        rubric_enabled=True,
        deep_judgment_enabled=True,
        deep_judgment_fuzzy_match_threshold=0.8,
        deep_judgment_excerpt_retry_attempts=2,
        deep_judgment_max_excerpts=3,
    )


@pytest.fixture
def sample_rubric_with_dj() -> Rubric:
    """Create a rubric with deep judgment-enabled traits."""
    return Rubric(
        llm_traits=[
            LLMRubricTrait(
                name="scientific_accuracy",
                description="Does the answer provide scientifically accurate information?",
                kind="score",
                min_score=1,
                max_score=5,
                deep_judgment_enabled=True,
                deep_judgment_excerpt_enabled=True,
                deep_judgment_max_excerpts=3,
                deep_judgment_fuzzy_match_threshold=0.8,
            ),
            LLMRubricTrait(
                name="overall_clarity",
                description="Is the overall response clear and well-organized?",
                kind="score",
                min_score=1,
                max_score=5,
                deep_judgment_enabled=True,
                deep_judgment_excerpt_enabled=False,  # No excerpts
            ),
        ]
    )


@pytest.fixture
def mixed_rubric() -> Rubric:
    """Create a rubric with both deep judgment and standard traits."""
    return Rubric(
        llm_traits=[
            LLMRubricTrait(
                name="factual_accuracy",
                description="Is the response factually accurate?",
                kind="score",
                min_score=1,
                max_score=5,
                deep_judgment_enabled=True,
                deep_judgment_excerpt_enabled=True,
            ),
            LLMRubricTrait(
                name="completeness",
                description="Is the response complete?",
                kind="score",
                min_score=1,
                max_score=5,
                deep_judgment_enabled=False,  # Standard evaluation
            ),
            LLMRubricTrait(
                name="mentions_examples",
                description="Does the response include examples?",
                kind="boolean",
                deep_judgment_enabled=False,  # Standard evaluation
            ),
        ]
    )


@pytest.mark.integration
@pytest.mark.requires_api
@pytest.mark.deep_judgment
class TestEndToEndDeepJudgmentPipeline:
    """Test end-to-end verification with deep judgment rubrics."""

    @pytest.fixture
    def simple_benchmark(self, sample_rubric_with_dj: Rubric):
        """Create a simple benchmark with one question."""
        benchmark = Benchmark(name="DJ Test Benchmark")

        # Add a question with answer template
        question_data = {
            "id": "dj_test_q1",
            "question": "What is photosynthesis?",
            "answer_template": {
                "id": "template_photosynthesis",
                "fields": {
                    "process": "conversion of light energy to chemical energy",
                    "location": "chloroplasts",
                    "equation": "6CO2 + 6H2O + light → C6H12O6 + 6O2",
                },
                "correct": True,
            },
            "finished": True,
        }

        benchmark.questions = {"dj_test_q1": question_data}
        benchmark.rubric = sample_rubric_with_dj

        return benchmark

    @patch("karenina.benchmark.verification.evaluators.rubric_evaluator.init_chat_model_unified")
    def test_end_to_end_verification_with_dj_rubrics(self, mock_init_model, simple_benchmark, dj_verification_config):
        """Test full verification pipeline with deep judgment rubrics."""
        # Mock LLM for all stages
        mock_llm = Mock()
        mock_init_model.return_value = mock_llm

        # Mock response sequence for verification
        mock_llm.invoke.side_effect = [
            # Answer generation
            Mock(
                content=(
                    "Photosynthesis is the process by which plants convert light energy into chemical energy. "
                    "This occurs in the chloroplasts. The equation is: 6CO2 + 6H2O + light → C6H12O6 + 6O2."
                )
            ),
            # Parsing (extract fields)
            Mock(
                content='{"process": "conversion of light energy to chemical energy", '
                '"location": "chloroplasts", '
                '"equation": "6CO2 + 6H2O + light → C6H12O6 + 6O2"}'
            ),
            # Deep judgment trait 1: scientific_accuracy (with excerpts)
            # - Excerpt extraction
            Mock(
                content='{"excerpts": [{"text": "Photosynthesis is the process by which plants convert light energy", '
                '"confidence": "high"}]}'
            ),
            # - Reasoning
            Mock(content="The answer provides scientifically accurate information with correct terminology."),
            # - Score
            Mock(content='{"scientific_accuracy": 5}'),
            # Deep judgment trait 2: overall_clarity (without excerpts)
            # - Reasoning only
            Mock(content="The response is clear and well-organized."),
            # - Score
            Mock(content='{"overall_clarity": 4}'),
        ]

        # Run verification
        results = simple_benchmark.run_verification(config=dj_verification_config, question_ids=["dj_test_q1"])

        # Verify results exist
        assert len(results) > 0

        # Get the result
        result = next(iter(results.values()))

        # Check deep judgment was performed
        assert result.deep_judgment_rubric is not None
        assert result.deep_judgment_rubric.deep_judgment_rubric_performed is True

        # Check trait scores
        assert result.deep_judgment_rubric.deep_judgment_rubric_scores is not None
        assert "scientific_accuracy" in result.deep_judgment_rubric.deep_judgment_rubric_scores
        assert "overall_clarity" in result.deep_judgment_rubric.deep_judgment_rubric_scores

        # Check reasoning exists for both traits
        assert result.deep_judgment_rubric.rubric_trait_reasoning is not None
        assert "scientific_accuracy" in result.deep_judgment_rubric.rubric_trait_reasoning
        assert "overall_clarity" in result.deep_judgment_rubric.rubric_trait_reasoning

        # Check excerpts exist for first trait only
        assert result.deep_judgment_rubric.extracted_rubric_excerpts is not None
        assert "scientific_accuracy" in result.deep_judgment_rubric.extracted_rubric_excerpts
        assert "overall_clarity" not in result.deep_judgment_rubric.extracted_rubric_excerpts


@pytest.mark.integration
@pytest.mark.requires_api
@pytest.mark.deep_judgment
class TestAutoFailMechanism:
    """Test auto-fail mechanism when retries are exhausted."""

    @patch("karenina.benchmark.verification.evaluators.rubric_evaluator.init_chat_model_unified")
    def test_auto_fail_triggers_after_retries(self, mock_init_model, test_model_config):
        """Test that auto-fail triggers when all retry attempts are exhausted."""
        mock_llm = Mock()
        mock_init_model.return_value = mock_llm

        # Create benchmark with question
        benchmark = Benchmark(name="Auto-fail Test")
        question_data = {
            "id": "autofail_q1",
            "question": "What is ATP?",
            "answer_template": {
                "id": "template_atp",
                "fields": {"molecule": "adenosine triphosphate", "function": "energy carrier"},
                "correct": True,
            },
            "finished": True,
        }
        benchmark.questions = {"autofail_q1": question_data}

        # Rubric with DJ trait
        trait = LLMRubricTrait(
            name="biochemical_accuracy",
            description="Biochemically accurate?",
            kind="score",
            min_score=1,
            max_score=5,
            deep_judgment_enabled=True,
            deep_judgment_excerpt_enabled=True,
            deep_judgment_excerpt_retry_attempts=2,  # 2 retries max
        )
        benchmark.rubric = Rubric(llm_traits=[trait])

        # Config with DJ enabled
        config = VerificationConfig(
            answering_models=[test_model_config],
            parsing_models=[test_model_config],
            evaluation_mode="rubric_only",
            rubric_enabled=True,
            deep_judgment_enabled=True,
            deep_judgment_fuzzy_match_threshold=0.8,
        )

        # Mock all excerpt extraction attempts to return invalid excerpts
        mock_llm.invoke.side_effect = [
            # Answer generation
            Mock(content="ATP is adenosine triphosphate, the energy carrier molecule."),
            # Parsing
            Mock(content='{"molecule": "adenosine triphosphate", "function": "energy carrier"}'),
            # First excerpt extraction (invalid)
            Mock(content='{"excerpts": [{"text": "Completely wrong text", "confidence": "low"}]}'),
            # Second excerpt extraction (invalid)
            Mock(content='{"excerpts": [{"text": "Still wrong text", "confidence": "low"}]}'),
            # Third excerpt extraction (invalid)
            Mock(content='{"excerpts": [{"text": "Wrong again", "confidence": "low"}]}'),
            # Auto-fail stage should trigger here
        ]

        # Run verification
        results = benchmark.run_verification(config=config, question_ids=["autofail_q1"])

        # Result should exist but may have auto-fail indicator
        result = next(iter(results.values()))

        # Check if auto-fail was triggered
        if result.deep_judgment_rubric and result.deep_judgment_rubric.traits_without_valid_excerpts:
            assert "biochemical_accuracy" in result.deep_judgment_rubric.traits_without_valid_excerpts


@pytest.mark.integration
@pytest.mark.deep_judgment
class TestMixedStandardAndDJTraits:
    """Test rubrics with both standard and deep judgment traits."""

    @pytest.fixture
    def mixed_benchmark(self, mixed_rubric: Rubric):
        """Create benchmark with mixed rubric."""
        benchmark = Benchmark(name="Mixed Rubric Test")
        question_data = {
            "id": "mixed_q1",
            "question": "Explain cellular respiration.",
            "answer_template": {
                "id": "template_respiration",
                "fields": {"process": "energy production", "location": "mitochondria"},
                "correct": True,
            },
            "finished": True,
        }
        benchmark.questions = {"mixed_q1": question_data}
        benchmark.rubric = mixed_rubric
        return benchmark

    @patch("karenina.benchmark.verification.evaluators.rubric_evaluator.init_chat_model_unified")
    def test_mixed_standard_and_dj_traits(self, mock_init_model, mixed_benchmark, test_model_config):
        """Test evaluation with both standard and deep judgment traits."""
        mock_llm = Mock()
        mock_init_model.return_value = mock_llm

        # Mock response sequence
        mock_llm.invoke.side_effect = [
            # Answer generation
            Mock(
                content=(
                    "Cellular respiration is the process of energy production in mitochondria. "
                    "For example, glucose is broken down to produce ATP."
                )
            ),
            # Parsing
            Mock(content='{"process": "energy production", "location": "mitochondria"}'),
            # Deep judgment trait: factual_accuracy
            # - Excerpts
            Mock(content='{"excerpts": [{"text": "energy production in mitochondria", "confidence": "high"}]}'),
            # - Reasoning
            Mock(content="Factually accurate."),
            # - Score
            Mock(content='{"factual_accuracy": 5}'),
            # Standard traits evaluated in batch
            Mock(content='{"completeness": 4, "mentions_examples": true}'),
        ]

        config = VerificationConfig(
            answering_models=[test_model_config],
            parsing_models=[test_model_config],
            evaluation_mode="rubric_only",
            rubric_enabled=True,
            deep_judgment_enabled=True,
        )

        results = mixed_benchmark.run_verification(config=config, question_ids=["mixed_q1"])

        result = next(iter(results.values()))

        # Check standard rubric scores
        if result.rubric:
            assert result.rubric.llm_trait_scores is not None

        # Check deep judgment scores
        if result.deep_judgment_rubric:
            assert result.deep_judgment_rubric.deep_judgment_rubric_scores is not None
            # factual_accuracy should be in DJ scores
            assert "factual_accuracy" in result.deep_judgment_rubric.deep_judgment_rubric_scores

            # completeness and mentions_examples should be in standard scores
            if result.deep_judgment_rubric.standard_rubric_scores:
                assert "completeness" in result.deep_judgment_rubric.standard_rubric_scores
                assert "mentions_examples" in result.deep_judgment_rubric.standard_rubric_scores


@pytest.mark.integration
@pytest.mark.deep_judgment
class TestBothDataframeExportMethods:
    """Test both standard and detailed dataframe export methods."""

    @pytest.fixture
    def verification_results_fixture(self):
        """Create mock verification results for testing."""
        from karenina.schemas.workflow.verification import (
            VerificationResult,
            VerificationResultDeepJudgmentRubric,
            VerificationResultMetadata,
            VerificationResultRubric,
        )

        metadata = VerificationResultMetadata(
            question_id="export_q1",
            template_id="template1",
            completed_without_errors=True,
            error=None,
            question_text="Test question for export",
            keywords=["test"],
            answering_model="gpt-4",
            parsing_model="gpt-4-mini",
            execution_time=3.5,
            timestamp="2024-01-15T10:30:00",
            replicate=1,
        )

        rubric = VerificationResultRubric(
            rubric_evaluation_performed=True,
            llm_trait_scores={
                "scientific_accuracy": 5,
                "completeness": 4,
            },
        )

        dj_rubric = VerificationResultDeepJudgmentRubric(
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

        result = VerificationResult(metadata=metadata, rubric=rubric, deep_judgment_rubric=dj_rubric)

        return [result]

    def test_both_dataframe_export_methods(self, verification_results_fixture):
        """Test both standard (RubricResults) and detailed (RubricJudgmentResults) exports."""
        from karenina.schemas.workflow.rubric_judgment_results import RubricJudgmentResults
        from karenina.schemas.workflow.rubric_results import RubricResults

        results = verification_results_fixture

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

        # Without include_deep_judgment flag, no DJ columns
        assert "trait_reasoning" not in df_standard.columns

        # With include_deep_judgment flag
        df_standard_dj = rubric_results.to_dataframe(trait_type="llm_score", include_deep_judgment=True)
        assert "trait_reasoning" in df_standard_dj.columns
        assert "trait_excerpts" in df_standard_dj.columns

        # Detailed export (RubricJudgmentResults)
        judgment_results = RubricJudgmentResults(results=results)
        df_detailed = judgment_results.to_dataframe()

        # Should have 2 rows (2 excerpts for scientific_accuracy)
        # completeness is standard trait, should not appear
        assert len(df_detailed) == 2
        assert all(df_detailed["trait_name"] == "scientific_accuracy")

        # Detailed columns
        assert "excerpt_index" in df_detailed.columns
        assert "excerpt_text" in df_detailed.columns
        assert "excerpt_confidence" in df_detailed.columns
        assert "excerpt_similarity_score" in df_detailed.columns
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
        from karenina.schemas.workflow.rubric_results import RubricResults
        from karenina.schemas.workflow.verification import (
            VerificationResult,
            VerificationResultMetadata,
            VerificationResultRubric,
        )

        metadata = VerificationResultMetadata(
            question_id="col_test_q1",
            template_id="template1",
            completed_without_errors=True,
            error=None,
            question_text="Column test",
            keywords=None,
            answering_model="gpt-4",
            parsing_model="gpt-4-mini",
            execution_time=1.0,
            timestamp="2024-01-15T10:30:00",
            replicate=1,
        )

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
            "template_id",
            "trait_name",
            "trait_score",
            "trait_type",
            "answering_model",
            "parsing_model",
            "execution_time",
            "timestamp",
        ]

        for col in expected_columns:
            assert col in df.columns, f"Missing expected column: {col}"

    def test_detailed_export_columns(self):
        """Test that detailed export has expected columns."""
        from karenina.schemas.workflow.rubric_judgment_results import RubricJudgmentResults
        from karenina.schemas.workflow.verification import (
            VerificationResult,
            VerificationResultDeepJudgmentRubric,
            VerificationResultMetadata,
            VerificationResultRubric,
        )

        metadata = VerificationResultMetadata(
            question_id="detail_col_q1",
            template_id="template1",
            completed_without_errors=True,
            error=None,
            question_text="Detail column test",
            keywords=None,
            answering_model="gpt-4",
            parsing_model="gpt-4-mini",
            execution_time=1.0,
            timestamp="2024-01-15T10:30:00",
            replicate=1,
        )

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
                "clarity": {"stages_completed": ["excerpt_extraction"], "model_calls": 1, "had_excerpts": True}
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
            "excerpt_similarity_score",
            "trait_model_calls",
            "trait_excerpt_retries",
            "trait_stages_completed",
            "trait_had_excerpts",
        ]

        for col in expected_detailed_columns:
            assert col in df.columns, f"Missing expected detailed column: {col}"
