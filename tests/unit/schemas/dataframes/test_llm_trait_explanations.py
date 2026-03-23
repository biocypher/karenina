"""Unit tests for LLM trait explanations plumbing (issue 154).

The LLM judge generates structured output with both score and explanation,
but only the score is extracted today. This module tests the storage field
on VerificationResultRubric and the DataFrame column in
RubricDataFrameBuilder so that explanations can flow through once the
evaluator is updated.

Covers:
- VerificationResultRubric accepts llm_trait_explanations field
- llm_trait_explanations defaults to None
- Rubric DataFrame includes trait_explanation column for LLM traits
- trait_explanation is populated correctly when explanations are provided
- trait_explanation is None when no explanations are stored
- trait_explanation column appears in correct position (after trait_label)
- Non-LLM traits (regex, callable, metric, agentic) do NOT have trait_explanation
"""

import pytest

from karenina.schemas.dataframes.rubric import RubricDataFrameBuilder
from karenina.schemas.verification import (
    VerificationResult,
    VerificationResultRubric,
    VerificationResultTemplate,
)
from tests.integration.dataframe_helpers import create_metadata

# =============================================================================
# Helpers
# =============================================================================


def _make_template() -> VerificationResultTemplate:
    """Build a minimal successful template result."""
    return VerificationResultTemplate(
        raw_llm_response="test response",
        parsed_gt_response={"answer": "Paris"},
        parsed_llm_response={"answer": "Paris"},
        template_verification_performed=True,
        verify_result=True,
    )


def _make_rubric(**overrides) -> VerificationResultRubric:
    """Build a rubric result with LLM traits (and optional overrides)."""
    defaults = {
        "rubric_evaluation_performed": True,
        "rubric_evaluation_strategy": "batch",
        "llm_trait_scores": {"Clarity": 4, "Relevance": 3},
    }
    defaults.update(overrides)
    return VerificationResultRubric(**defaults)


def _make_result(
    question_id: str = "q001",
    rubric: VerificationResultRubric | None = None,
) -> VerificationResult:
    """Build a VerificationResult with optional rubric."""
    metadata = create_metadata(question_id)
    return VerificationResult(
        metadata=metadata,
        template=_make_template(),
        rubric=rubric,
    )


def _build_df(results: list[VerificationResult], trait_type: str = "all"):
    """Build a rubric DataFrame from a list of results."""
    builder = RubricDataFrameBuilder(results)
    return builder.build_dataframe(trait_type=trait_type)


# =============================================================================
# Schema: VerificationResultRubric field
# =============================================================================


@pytest.mark.unit
class TestVerificationResultRubricField:
    """VerificationResultRubric must accept and default llm_trait_explanations."""

    def test_field_defaults_to_none(self):
        """llm_trait_explanations defaults to None when not provided."""
        rubric = VerificationResultRubric()
        assert rubric.llm_trait_explanations is None

    def test_field_accepts_dict(self):
        """llm_trait_explanations accepts a dict of trait name to explanation."""
        explanations = {
            "Clarity": "The response was well structured with clear paragraphs.",
            "Relevance": "The answer directly addressed the question asked.",
        }
        rubric = VerificationResultRubric(llm_trait_explanations=explanations)
        assert rubric.llm_trait_explanations == explanations

    def test_field_round_trips_through_model_dump(self):
        """llm_trait_explanations survives model_dump/model_validate cycle."""
        explanations = {"Clarity": "Clear and concise."}
        rubric = VerificationResultRubric(
            rubric_evaluation_performed=True,
            llm_trait_scores={"Clarity": 5},
            llm_trait_explanations=explanations,
        )
        dumped = rubric.model_dump()
        restored = VerificationResultRubric.model_validate(dumped)
        assert restored.llm_trait_explanations == explanations

    def test_field_coexists_with_labels(self):
        """llm_trait_explanations can coexist with llm_trait_labels."""
        rubric = VerificationResultRubric(
            rubric_evaluation_performed=True,
            llm_trait_scores={"Clarity": 4, "tone": 1},
            llm_trait_labels={"tone": "Professional"},
            llm_trait_explanations={
                "Clarity": "Good structure.",
                "tone": "Formal language throughout.",
            },
        )
        assert rubric.llm_trait_labels == {"tone": "Professional"}
        assert rubric.llm_trait_explanations["Clarity"] == "Good structure."
        assert rubric.llm_trait_explanations["tone"] == "Formal language throughout."


# =============================================================================
# DataFrame: trait_explanation column for LLM traits
# =============================================================================


@pytest.mark.unit
class TestTraitExplanationColumn:
    """Rubric DataFrame must include trait_explanation for LLM traits."""

    def test_column_present_when_explanations_provided(self):
        """trait_explanation column appears when explanations are populated."""
        rubric = _make_rubric(
            llm_trait_explanations={
                "Clarity": "Well organized response.",
                "Relevance": "Directly addressed the question.",
            },
        )
        result = _make_result(rubric=rubric)
        df = _build_df([result], trait_type="llm")

        assert "trait_explanation" in df.columns

    def test_column_present_when_explanations_none(self):
        """trait_explanation column appears even when explanations are None."""
        rubric = _make_rubric(llm_trait_explanations=None)
        result = _make_result(rubric=rubric)
        df = _build_df([result], trait_type="llm")

        assert "trait_explanation" in df.columns

    def test_values_populated_correctly(self):
        """trait_explanation values match the provided explanations per trait."""
        rubric = _make_rubric(
            llm_trait_explanations={
                "Clarity": "Well organized response.",
                "Relevance": "Directly addressed the question.",
            },
        )
        result = _make_result(rubric=rubric)
        df = _build_df([result], trait_type="llm")

        clarity_row = df[df["trait_name"] == "Clarity"]
        assert clarity_row["trait_explanation"].iloc[0] == "Well organized response."

        relevance_row = df[df["trait_name"] == "Relevance"]
        assert relevance_row["trait_explanation"].iloc[0] == "Directly addressed the question."

    def test_values_none_when_no_explanations(self):
        """trait_explanation is None when llm_trait_explanations is None."""
        rubric = _make_rubric(llm_trait_explanations=None)
        result = _make_result(rubric=rubric)
        df = _build_df([result], trait_type="llm")

        assert df["trait_explanation"].isna().all()

    def test_partial_explanations(self):
        """trait_explanation is None for traits not in the explanations dict."""
        rubric = _make_rubric(
            llm_trait_explanations={"Clarity": "Good structure."},
        )
        result = _make_result(rubric=rubric)
        df = _build_df([result], trait_type="llm")

        clarity_row = df[df["trait_name"] == "Clarity"]
        assert clarity_row["trait_explanation"].iloc[0] == "Good structure."

        relevance_row = df[df["trait_name"] == "Relevance"]
        assert relevance_row["trait_explanation"].iloc[0] is None


# =============================================================================
# Column ordering
# =============================================================================


@pytest.mark.unit
class TestTraitExplanationColumnOrder:
    """trait_explanation must appear after trait_label in column ordering."""

    def test_after_trait_label(self):
        """trait_explanation comes immediately after trait_label."""
        rubric = _make_rubric(
            llm_trait_explanations={"Clarity": "Good."},
        )
        result = _make_result(rubric=rubric)
        df = _build_df([result], trait_type="llm")

        cols = list(df.columns)
        label_idx = cols.index("trait_label")
        explanation_idx = cols.index("trait_explanation")
        assert explanation_idx == label_idx + 1, (
            f"trait_explanation at {explanation_idx} should be right after trait_label at {label_idx}"
        )

    def test_before_trait_type(self):
        """trait_explanation comes before trait_type in column ordering."""
        rubric = _make_rubric(
            llm_trait_explanations={"Clarity": "Good."},
        )
        result = _make_result(rubric=rubric)
        df = _build_df([result], trait_type="llm")

        cols = list(df.columns)
        explanation_idx = cols.index("trait_explanation")
        type_idx = cols.index("trait_type")
        assert explanation_idx < type_idx


# =============================================================================
# Non-LLM traits must NOT have trait_explanation
# =============================================================================


@pytest.mark.unit
class TestNonLLMTraitsNoExplanation:
    """Regex, callable, metric, and agentic traits must not carry trait_explanation."""

    def test_regex_trait_no_explanation(self):
        """Regex trait rows do not include trait_explanation."""
        rubric = _make_rubric(regex_trait_scores={"HasCitations": True})
        result = _make_result(rubric=rubric)
        df = _build_df([result], trait_type="regex")

        assert "trait_explanation" not in df.columns

    def test_callable_trait_no_explanation(self):
        """Callable trait rows do not include trait_explanation."""
        rubric = _make_rubric(callable_trait_scores={"LenCheck": True})
        result = _make_result(rubric=rubric)
        df = _build_df([result], trait_type="callable")

        assert "trait_explanation" not in df.columns

    def test_metric_trait_no_explanation(self):
        """Metric trait rows do not include trait_explanation."""
        rubric = _make_rubric(
            metric_trait_scores={"Entity": {"precision": 0.9, "recall": 0.8, "f1": 0.85}},
        )
        result = _make_result(rubric=rubric)
        df = _build_df([result], trait_type="metric")

        assert "trait_explanation" not in df.columns

    def test_agentic_trait_no_explanation(self):
        """Agentic trait rows do not include trait_explanation."""
        rubric = _make_rubric(agentic_trait_scores={"Quality": 4})
        result = _make_result(rubric=rubric)
        df = _build_df([result], trait_type="agentic")

        assert "trait_explanation" not in df.columns
