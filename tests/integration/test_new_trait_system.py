"""Integration test for the new trait system (RegexTrait + CallableTrait).

This test verifies that the refactored trait system works correctly:
- Rubric trait evaluation (LLM, Regex, Callable)
- Results properly split into regex_trait_scores and callable_trait_scores
- Export functionality handles all trait types

Tests are marked with:
- @pytest.mark.integration: Tests that combine multiple components
- @pytest.mark.rubric: Tests specific to rubric trait evaluation

Run with: pytest tests/integration/test_new_trait_system.py -v
"""

import csv
import re
from datetime import UTC, datetime

import pandas as pd
import pytest

from karenina.schemas.domain.rubric import CallableTrait, LLMRubricTrait, RegexTrait, Rubric
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.schemas.workflow import (
    RubricResults,
    VerificationResult,
    VerificationResultMetadata,
    VerificationResultRubric,
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
# Rubric Trait Fixtures
# =============================================================================


@pytest.fixture
def llm_traits() -> list[LLMRubricTrait]:
    """Create LLM rubric traits."""
    return [
        LLMRubricTrait(
            name="Clarity",
            description="The response is clear and well-organized",
            kind="score",
            min_score=1,
            max_score=5,
            higher_is_better=True,
        ),
        LLMRubricTrait(
            name="Accuracy",
            description="The response is factually correct",
            kind="boolean",
            higher_is_better=True,
        ),
        LLMRubricTrait(
            name="Completeness",
            description="The response addresses all aspects",
            kind="score",
            min_score=1,
            max_score=5,
            higher_is_better=True,
        ),
    ]


@pytest.fixture
def regex_traits() -> list[RegexTrait]:
    """Create regex rubric traits."""
    return [
        RegexTrait(
            name="HasCitations",
            description="Response includes citations like [1], [2]",
            pattern=r"\[\d+\]",
            higher_is_better=True,
        ),
        RegexTrait(
            name="HasNumbers",
            description="Response includes numeric values",
            pattern=r"\d+",
            higher_is_better=True,
        ),
    ]


@pytest.fixture
def callable_traits() -> list[CallableTrait]:
    """Create callable rubric traits."""

    def contains_citations(text: str) -> bool:
        return bool(re.search(r"\[\d+\]", text))

    def response_length_score(text: str) -> int:
        words = len(text.split())
        if words < 20:
            return 1
        elif words < 50:
            return 2
        elif words < 100:
            return 3
        elif words < 200:
            return 4
        else:
            return 5

    return [
        CallableTrait.from_callable(
            name="ContainsCitations",
            func=contains_citations,
            kind="boolean",
            description="Check if response contains citation markers",
            higher_is_better=True,
        ),
        CallableTrait.from_callable(
            name="ResponseLength",
            func=response_length_score,
            kind="score",
            min_score=1,
            max_score=5,
            description="Score based on response length",
            higher_is_better=True,
        ),
    ]


@pytest.fixture
def rubric(
    llm_traits: list[LLMRubricTrait],
    regex_traits: list[RegexTrait],
    callable_traits: list[CallableTrait],
) -> Rubric:
    """Create a complete rubric with all trait types."""
    return Rubric(
        llm_traits=llm_traits,
        regex_traits=regex_traits,
        callable_traits=callable_traits,
        metric_traits=[],
    )


# =============================================================================
# VerificationResult Fixtures
# =============================================================================


@pytest.fixture
def template_result_success() -> VerificationResultTemplate:
    """Create a successful template result."""
    return VerificationResultTemplate(
        raw_llm_response="BCL2 is an anti-apoptotic gene located on chromosome 18.",
        parsed_gt_response={"gene_name": "BCL2", "gene_function": "anti-apoptotic"},
        parsed_llm_response={"gene_name": "BCL2", "gene_function": "anti-apoptotic"},
        template_verification_performed=True,
        verify_result=True,
        verify_granular_result={"gene_name": True, "gene_function": True},
    )


@pytest.fixture
def rubric_result_with_all_traits() -> VerificationResultRubric:
    """Create a rubric result with all trait types."""
    return VerificationResultRubric(
        rubric_evaluation_performed=True,
        rubric_evaluation_strategy="batch",
        llm_trait_scores={
            "Clarity": 4,
            "Accuracy": True,
            "Completeness": 5,
        },
        regex_trait_scores={
            "HasCitations": True,
            "HasNumbers": True,
        },
        callable_trait_scores={
            "ContainsCitations": True,
            "ResponseLength": 3,
        },
        metric_trait_scores={},
    )


@pytest.fixture
def verification_result_with_all_traits(
    template_result_success: VerificationResultTemplate,
    rubric_result_with_all_traits: VerificationResultRubric,
) -> VerificationResult:
    """Create a verification result with all trait types."""
    return VerificationResult(
        metadata=_create_metadata("rq001", "gpt-4"),
        template=template_result_success,
        rubric=rubric_result_with_all_traits,
    )


@pytest.fixture
def verification_results_list(
    verification_result_with_all_traits: VerificationResult,
) -> list[VerificationResult]:
    """Create a list of verification results."""
    second_result = VerificationResult(
        metadata=_create_metadata("rq002", "claude-sonnet-4"),
        template=VerificationResultTemplate(
            raw_llm_response="100 degrees Celsius",
            parsed_gt_response={"celsius": 100, "fahrenheit": 212},
            parsed_llm_response={"celsius": 100, "fahrenheit": 212},
            template_verification_performed=True,
            verify_result=True,
            verify_granular_result={"celsius": True, "fahrenheit": True},
        ),
        rubric=VerificationResultRubric(
            rubric_evaluation_performed=True,
            rubric_evaluation_strategy="batch",
            llm_trait_scores={
                "Clarity": 5,
                "Accuracy": True,
                "Completeness": 4,
            },
            regex_trait_scores={
                "HasCitations": False,
                "HasNumbers": True,
            },
            callable_trait_scores={
                "ContainsCitations": False,
                "ResponseLength": 2,
            },
            metric_trait_scores={},
        ),
    )
    return [verification_result_with_all_traits, second_result]


# =============================================================================
# Rubric Structure Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.rubric
class TestRubricStructure:
    """Test rubric structure and trait organization."""

    def test_rubric_has_all_trait_types(self, rubric: Rubric):
        """Verify rubric contains all trait types."""
        assert len(rubric.llm_traits) == 3, "Should have 3 LLM traits"
        assert len(rubric.regex_traits) == 2, "Should have 2 Regex traits"
        assert len(rubric.callable_traits) == 2, "Should have 2 Callable traits"
        assert len(rubric.metric_traits) == 0, "Should have 0 Metric traits"

    def test_llm_trait_properties(self, llm_traits: list[LLMRubricTrait]):
        """Verify LLM traits have correct properties."""
        trait_names = {t.name for t in llm_traits}
        assert "Clarity" in trait_names
        assert "Accuracy" in trait_names
        assert "Completeness" in trait_names

        # Check score vs boolean kinds
        clarity_trait = next(t for t in llm_traits if t.name == "Clarity")
        accuracy_trait = next(t for t in llm_traits if t.name == "Accuracy")

        assert clarity_trait.kind == "score"
        assert clarity_trait.min_score == 1
        assert clarity_trait.max_score == 5

        assert accuracy_trait.kind == "boolean"

    def test_regex_trait_properties(self, regex_traits: list[RegexTrait]):
        """Verify regex traits have correct properties."""
        trait_names = {t.name for t in regex_traits}
        assert "HasCitations" in trait_names
        assert "HasNumbers" in trait_names

        # Verify pattern exists
        citation_trait = next(t for t in regex_traits if t.name == "HasCitations")
        assert citation_trait.pattern is not None
        assert len(citation_trait.pattern) > 0

    def test_callable_trait_properties(self, callable_traits: list[CallableTrait]):
        """Verify callable traits have correct properties."""
        trait_names = {t.name for t in callable_traits}
        assert "ContainsCitations" in trait_names
        assert "ResponseLength" in trait_names

        # Check boolean vs score kinds
        citation_trait = next(t for t in callable_traits if t.name == "ContainsCitations")
        length_trait = next(t for t in callable_traits if t.name == "ResponseLength")

        assert citation_trait.kind == "boolean"
        assert length_trait.kind == "score"
        assert length_trait.min_score == 1
        assert length_trait.max_score == 5


# =============================================================================
# Trait Evaluation Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.rubric
class TestTraitEvaluation:
    """Test trait evaluation functionality."""

    def test_regex_trait_evaluation(self, regex_traits: list[RegexTrait]):
        """Verify regex traits evaluate correctly."""
        citation_trait = next(t for t in regex_traits if t.name == "HasCitations")
        numbers_trait = next(t for t in regex_traits if t.name == "HasNumbers")

        # Test citation detection
        assert citation_trait.evaluate("This is a fact [1].") is True
        assert citation_trait.evaluate("No citations here.") is False

        # Test number detection
        assert numbers_trait.evaluate("There are 42 items.") is True
        assert numbers_trait.evaluate("No numbers here.") is False

    def test_boolean_callable_evaluation(self, callable_traits: list[CallableTrait]):
        """Verify boolean callable traits evaluate correctly."""
        citation_trait = next(t for t in callable_traits if t.name == "ContainsCitations")

        # Test with citation
        assert citation_trait.evaluate("This is a fact [1].") is True

        # Test without citation
        assert citation_trait.evaluate("No citations here.") is False

    def test_score_callable_evaluation(self, callable_traits: list[CallableTrait]):
        """Verify score callable traits evaluate correctly."""
        length_trait = next(t for t in callable_traits if t.name == "ResponseLength")

        short_score = length_trait.evaluate("Short.")
        long_score = length_trait.evaluate(
            "This is a comprehensive answer that provides extensive detail. "
            "It covers multiple aspects thoroughly. Each point is clear. "
            "The response demonstrates understanding. Multiple perspectives are considered."
        )

        assert isinstance(short_score, int)
        assert isinstance(long_score, int)
        assert 1 <= short_score <= 5
        assert 1 <= long_score <= 5
        assert long_score > short_score, "Longer answer should score higher"


# =============================================================================
# Trait Score Separation Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.rubric
class TestTraitScoreSeparation:
    """Test that trait scores are properly separated by type."""

    def test_trait_scores_separated_by_type(self, verification_result_with_all_traits: VerificationResult):
        """Verify trait scores are split correctly into separate dictionaries."""
        result = verification_result_with_all_traits

        assert result.rubric is not None
        assert result.rubric.rubric_evaluation_performed is True

        # Check LLM trait scores
        assert result.rubric.llm_trait_scores is not None
        assert len(result.rubric.llm_trait_scores) == 3
        assert "Clarity" in result.rubric.llm_trait_scores
        assert "Accuracy" in result.rubric.llm_trait_scores
        assert "Completeness" in result.rubric.llm_trait_scores

        # Check Regex trait scores
        assert result.rubric.regex_trait_scores is not None
        assert len(result.rubric.regex_trait_scores) == 2
        for trait_name, score in result.rubric.regex_trait_scores.items():
            assert isinstance(score, bool), f"Regex trait {trait_name} should return bool"

        # Check Callable trait scores
        assert result.rubric.callable_trait_scores is not None
        assert len(result.rubric.callable_trait_scores) == 2
        for trait_name, score in result.rubric.callable_trait_scores.items():
            assert isinstance(score, bool | int), f"Callable {trait_name} should return bool/int"

    def test_get_all_trait_scores_includes_all_types(self, verification_result_with_all_traits: VerificationResult):
        """Verify get_all_trait_scores() includes all trait types."""
        result = verification_result_with_all_traits

        all_scores = result.rubric.get_all_trait_scores()

        # Should have all traits combined
        expected_traits = {
            "Clarity",
            "Accuracy",
            "Completeness",  # LLM
            "HasCitations",
            "HasNumbers",  # Regex
            "ContainsCitations",
            "ResponseLength",  # Callable
        }

        actual_traits = set(all_scores.keys())
        assert expected_traits == actual_traits, f"Missing traits: {expected_traits - actual_traits}"


# =============================================================================
# DataFrame Export Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.rubric
class TestRubricDataFrameExport:
    """Test RubricResults DataFrame export with all trait types."""

    def test_dataframe_export_all_traits(self, verification_results_list: list[VerificationResult]):
        """Test DataFrame export includes all trait types."""
        rubric_results = RubricResults(results=verification_results_list)
        df = rubric_results.to_dataframe(trait_type="all")

        assert len(df) > 0

        # Should have rows for each trait type
        trait_types = df["trait_type"].unique()
        assert "llm_score" in trait_types or "llm_boolean" in trait_types
        assert "regex" in trait_types
        assert "callable" in trait_types

    def test_dataframe_export_llm_traits(self, verification_results_list: list[VerificationResult]):
        """Test DataFrame export with LLM traits only."""
        rubric_results = RubricResults(results=verification_results_list)
        df = rubric_results.to_dataframe(trait_type="llm")

        # Should only have LLM trait rows
        assert all(df["trait_type"].str.startswith("llm"))

        # Check LLM trait names present
        trait_names = set(df["trait_name"].unique())
        assert "Clarity" in trait_names
        assert "Accuracy" in trait_names
        assert "Completeness" in trait_names

    def test_dataframe_export_regex_traits(self, verification_results_list: list[VerificationResult]):
        """Test DataFrame export with regex traits only."""
        rubric_results = RubricResults(results=verification_results_list)
        df = rubric_results.to_dataframe(trait_type="regex")

        # Should only have regex trait rows
        assert all(df["trait_type"] == "regex")

        # Check regex trait names
        trait_names = set(df["trait_name"].unique())
        assert "HasCitations" in trait_names
        assert "HasNumbers" in trait_names

        # Regex scores should be boolean
        assert all(df["trait_score"].isin([True, False]))

    def test_dataframe_export_callable_traits(self, verification_results_list: list[VerificationResult]):
        """Test DataFrame export with callable traits only."""
        rubric_results = RubricResults(results=verification_results_list)
        df = rubric_results.to_dataframe(trait_type="callable")

        # Should only have callable trait rows
        assert all(df["trait_type"] == "callable")

        # Check callable trait names
        trait_names = set(df["trait_name"].unique())
        assert "ContainsCitations" in trait_names
        assert "ResponseLength" in trait_names


# =============================================================================
# CSV Export Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.rubric
class TestCSVExport:
    """Test CSV export functionality with all trait types."""

    def test_csv_export_with_all_traits(self, verification_results_list: list[VerificationResult], tmp_path):
        """Test CSV export includes all trait type columns."""
        rubric_results = RubricResults(results=verification_results_list)
        df = rubric_results.to_dataframe(trait_type="all")

        # Export to CSV
        csv_path = tmp_path / "results.csv"
        df.to_csv(csv_path, index=False)

        assert csv_path.exists(), "CSV file should be created"

        # Read CSV and verify structure
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            assert len(rows) > 0, "Should have at least one row"

            # Check column presence
            headers = list(rows[0].keys()) if rows else []
            assert "trait_name" in headers
            assert "trait_score" in headers
            assert "trait_type" in headers
            assert "question_id" in headers

    def test_csv_roundtrip_preserves_data(self, verification_results_list: list[VerificationResult], tmp_path):
        """Test that CSV export and import preserves data."""
        rubric_results = RubricResults(results=verification_results_list)
        df_original = rubric_results.to_dataframe(trait_type="all")

        # Export to CSV
        csv_path = tmp_path / "results.csv"
        df_original.to_csv(csv_path, index=False)

        # Read back
        df_loaded = pd.read_csv(csv_path)

        # Compare row counts
        assert len(df_loaded) == len(df_original)

        # Compare columns
        assert set(df_loaded.columns) == set(df_original.columns)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
