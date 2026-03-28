"""Unit tests for trait_provenance column in RubricDataFrameBuilder.

Issue 020, Task 3: RubricDataFrameBuilder must include a trait_provenance
column that carries per-trait provenance metadata ("global",
"question_specific", or "dynamic") from merge_rubrics() and dynamic rubric
resolution through the pipeline into the DataFrame output.
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
# Fixtures
# =============================================================================


def _make_template() -> VerificationResultTemplate:
    return VerificationResultTemplate(
        raw_llm_response="The answer is BCL2.",
        template_verification_performed=True,
        verify_result=True,
    )


def _make_result(
    question_id: str,
    trait_provenance: dict[str, str] | None = None,
    llm_trait_scores: dict | None = None,
    regex_trait_scores: dict | None = None,
    callable_trait_scores: dict | None = None,
    metric_trait_scores: dict | None = None,
    agentic_trait_scores: dict | None = None,
) -> VerificationResult:
    rubric = VerificationResultRubric(
        rubric_evaluation_performed=True,
        rubric_evaluation_strategy="batch",
        llm_trait_scores=llm_trait_scores,
        regex_trait_scores=regex_trait_scores,
        callable_trait_scores=callable_trait_scores,
        metric_trait_scores=metric_trait_scores,
        agentic_trait_scores=agentic_trait_scores,
        trait_provenance=trait_provenance,
    )
    return VerificationResult(
        metadata=create_metadata(question_id),
        template=_make_template(),
        rubric=rubric,
    )


# =============================================================================
# Tests: trait_provenance column presence
# =============================================================================


@pytest.mark.unit
class TestTraitProvenanceColumnPresence:
    """trait_provenance column must appear for all trait row types."""

    def test_column_present_for_llm_traits(self):
        """trait_provenance column exists for LLM trait rows."""
        result = _make_result(
            "q1",
            trait_provenance={"Clarity": "global"},
            llm_trait_scores={"Clarity": 4},
        )
        builder = RubricDataFrameBuilder(results=[result])
        df = builder.build_dataframe(trait_type="llm")

        assert "trait_provenance" in df.columns

    def test_column_present_for_regex_traits(self):
        """trait_provenance column exists for regex trait rows."""
        result = _make_result(
            "q1",
            trait_provenance={"HasCitations": "question_specific"},
            regex_trait_scores={"HasCitations": True},
        )
        builder = RubricDataFrameBuilder(results=[result])
        df = builder.build_dataframe(trait_type="regex")

        assert "trait_provenance" in df.columns

    def test_column_present_for_callable_traits(self):
        """trait_provenance column exists for callable trait rows."""
        result = _make_result(
            "q1",
            trait_provenance={"ContainsCitations": "global"},
            callable_trait_scores={"ContainsCitations": True},
        )
        builder = RubricDataFrameBuilder(results=[result])
        df = builder.build_dataframe(trait_type="callable")

        assert "trait_provenance" in df.columns

    def test_column_present_for_metric_traits(self):
        """trait_provenance column exists for metric trait rows."""
        result = _make_result(
            "q1",
            trait_provenance={"EntityExtraction": "global"},
            metric_trait_scores={"EntityExtraction": {"precision": 0.85, "recall": 0.90, "f1": 0.87}},
        )
        builder = RubricDataFrameBuilder(results=[result])
        df = builder.build_dataframe(trait_type="metric")

        assert "trait_provenance" in df.columns

    def test_column_present_for_agentic_traits(self):
        """trait_provenance column exists for agentic trait rows."""
        result = _make_result(
            "q1",
            trait_provenance={"CodeQuality": "dynamic"},
            agentic_trait_scores={"CodeQuality": 4},
        )
        builder = RubricDataFrameBuilder(results=[result])
        df = builder.build_dataframe(trait_type="agentic")

        assert "trait_provenance" in df.columns

    def test_column_present_in_all_trait_type(self):
        """trait_provenance column exists when trait_type='all'."""
        result = _make_result(
            "q1",
            trait_provenance={"Clarity": "global", "HasNumbers": "question_specific"},
            llm_trait_scores={"Clarity": 3},
            regex_trait_scores={"HasNumbers": False},
        )
        builder = RubricDataFrameBuilder(results=[result])
        df = builder.build_dataframe(trait_type="all")

        assert "trait_provenance" in df.columns


# =============================================================================
# Tests: trait_provenance column values
# =============================================================================


@pytest.mark.unit
class TestTraitProvenanceColumnValues:
    """trait_provenance column must carry correct per-trait provenance values."""

    def test_global_provenance_value(self):
        """Traits from global rubric have provenance 'global'."""
        result = _make_result(
            "q1",
            trait_provenance={"Clarity": "global"},
            llm_trait_scores={"Clarity": 4},
        )
        builder = RubricDataFrameBuilder(results=[result])
        df = builder.build_dataframe(trait_type="llm")

        assert df["trait_provenance"].iloc[0] == "global"

    def test_question_specific_provenance_value(self):
        """Traits from question rubric have provenance 'question_specific'."""
        result = _make_result(
            "q1",
            trait_provenance={"HasCitations": "question_specific"},
            regex_trait_scores={"HasCitations": True},
        )
        builder = RubricDataFrameBuilder(results=[result])
        df = builder.build_dataframe(trait_type="regex")

        assert df["trait_provenance"].iloc[0] == "question_specific"

    def test_dynamic_provenance_value(self):
        """Traits promoted from dynamic rubric have provenance 'dynamic'."""
        result = _make_result(
            "q1",
            trait_provenance={"CodeQuality": "dynamic"},
            agentic_trait_scores={"CodeQuality": 4},
        )
        builder = RubricDataFrameBuilder(results=[result])
        df = builder.build_dataframe(trait_type="agentic")

        assert df["trait_provenance"].iloc[0] == "dynamic"

    def test_mixed_provenance_values(self):
        """Multiple traits with different provenances are correctly mapped."""
        result = _make_result(
            "q1",
            trait_provenance={
                "Clarity": "global",
                "Relevance": "question_specific",
                "Engagement": "dynamic",
            },
            llm_trait_scores={"Clarity": 4, "Relevance": 3, "Engagement": 5},
        )
        builder = RubricDataFrameBuilder(results=[result])
        df = builder.build_dataframe(trait_type="llm")

        provenance_map = dict(zip(df["trait_name"], df["trait_provenance"], strict=False))
        assert provenance_map["Clarity"] == "global"
        assert provenance_map["Relevance"] == "question_specific"
        assert provenance_map["Engagement"] == "dynamic"

    def test_none_provenance_when_not_set(self):
        """trait_provenance is None when no provenance metadata is set."""
        result = _make_result(
            "q1",
            trait_provenance=None,
            llm_trait_scores={"Clarity": 4},
        )
        builder = RubricDataFrameBuilder(results=[result])
        df = builder.build_dataframe(trait_type="llm")

        assert "trait_provenance" in df.columns
        assert df["trait_provenance"].iloc[0] is None

    def test_none_provenance_for_unknown_trait(self):
        """trait_provenance is None when trait is not in provenance dict."""
        result = _make_result(
            "q1",
            trait_provenance={"OtherTrait": "global"},
            llm_trait_scores={"Clarity": 4},
        )
        builder = RubricDataFrameBuilder(results=[result])
        df = builder.build_dataframe(trait_type="llm")

        assert df["trait_provenance"].iloc[0] is None

    def test_metric_trait_provenance_per_exploded_row(self):
        """Each exploded metric row carries the same provenance for its parent trait."""
        result = _make_result(
            "q1",
            trait_provenance={"EntityExtraction": "global"},
            metric_trait_scores={"EntityExtraction": {"precision": 0.85, "recall": 0.90, "f1": 0.87}},
        )
        builder = RubricDataFrameBuilder(results=[result])
        df = builder.build_dataframe(trait_type="metric")

        # All three exploded rows should have the same provenance
        assert len(df) == 3
        assert all(df["trait_provenance"] == "global")


# =============================================================================
# Tests: column ordering
# =============================================================================


@pytest.mark.unit
class TestTraitProvenanceColumnOrdering:
    """trait_provenance must appear after rubric_evaluation_strategy in column order."""

    def test_column_after_rubric_evaluation_strategy(self):
        """trait_provenance appears after rubric_evaluation_strategy."""
        result = _make_result(
            "q1",
            trait_provenance={"Clarity": "global"},
            llm_trait_scores={"Clarity": 4},
        )
        builder = RubricDataFrameBuilder(results=[result])
        df = builder.build_dataframe(trait_type="llm")

        cols = list(df.columns)
        strategy_idx = cols.index("rubric_evaluation_strategy")
        provenance_idx = cols.index("trait_provenance")

        assert provenance_idx > strategy_idx, "trait_provenance must come after rubric_evaluation_strategy"

    def test_column_before_deep_judgment_when_included(self):
        """trait_provenance appears before deep judgment columns when included."""
        result = _make_result(
            "q1",
            trait_provenance={"Clarity": "global"},
            llm_trait_scores={"Clarity": 4},
        )
        builder = RubricDataFrameBuilder(results=[result], include_deep_judgment=True)
        df = builder.build_dataframe(trait_type="llm")

        cols = list(df.columns)
        provenance_idx = cols.index("trait_provenance")

        if "trait_reasoning" in cols:
            reasoning_idx = cols.index("trait_reasoning")
            assert provenance_idx < reasoning_idx


# =============================================================================
# Tests: VerificationResultRubric field
# =============================================================================


@pytest.mark.unit
class TestVerificationResultRubricProvenanceField:
    """VerificationResultRubric must accept trait_provenance."""

    def test_field_exists_and_defaults_to_none(self):
        """trait_provenance field exists and defaults to None."""
        rubric = VerificationResultRubric()
        assert rubric.trait_provenance is None

    def test_field_accepts_dict(self):
        """trait_provenance field accepts dict[str, str]."""
        provenance = {"Clarity": "global", "Safety": "question_specific"}
        rubric = VerificationResultRubric(trait_provenance=provenance)
        assert rubric.trait_provenance == provenance

    def test_field_round_trips_through_model_dump(self):
        """trait_provenance survives model_dump() and model_validate()."""
        provenance = {"Clarity": "global", "Safety": "dynamic"}
        rubric = VerificationResultRubric(trait_provenance=provenance)

        dumped = rubric.model_dump()
        restored = VerificationResultRubric.model_validate(dumped)
        assert restored.trait_provenance == provenance
