"""Unit tests for rubric_evaluation_performed and rubric_evaluation_strategy columns.

Issue 156: RubricDataFrameBuilder does not export rubric_evaluation_performed
and rubric_evaluation_strategy as DataFrame columns, causing data loss.

VerificationResultRubric already has both fields; this tests that
RubricDataFrameBuilder passes them through correctly.
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
    rubric_evaluation_performed: bool,
    rubric_evaluation_strategy: str | None,
    llm_trait_scores: dict | None = None,
    regex_trait_scores: dict | None = None,
    callable_trait_scores: dict | None = None,
    metric_trait_scores: dict | None = None,
    agentic_trait_scores: dict | None = None,
) -> VerificationResult:
    rubric = VerificationResultRubric(
        rubric_evaluation_performed=rubric_evaluation_performed,
        rubric_evaluation_strategy=rubric_evaluation_strategy,
        llm_trait_scores=llm_trait_scores,
        regex_trait_scores=regex_trait_scores,
        callable_trait_scores=callable_trait_scores,
        metric_trait_scores=metric_trait_scores,
        agentic_trait_scores=agentic_trait_scores,
    )
    return VerificationResult(
        metadata=create_metadata(question_id),
        template=_make_template(),
        rubric=rubric,
    )


# =============================================================================
# Tests: rubric_evaluation_performed column
# =============================================================================


@pytest.mark.unit
class TestRubricEvaluationPerformedColumn:
    """rubric_evaluation_performed must appear in all trait row types."""

    def test_column_present_for_llm_traits(self):
        """rubric_evaluation_performed column exists for LLM trait rows."""
        result = _make_result(
            "q1",
            rubric_evaluation_performed=True,
            rubric_evaluation_strategy="batch",
            llm_trait_scores={"Clarity": 4},
        )
        builder = RubricDataFrameBuilder(results=[result])
        df = builder.build_dataframe(trait_type="llm")

        assert "rubric_evaluation_performed" in df.columns

    def test_column_value_for_llm_traits(self):
        """rubric_evaluation_performed value is correctly set for LLM trait rows."""
        result = _make_result(
            "q1",
            rubric_evaluation_performed=True,
            rubric_evaluation_strategy="batch",
            llm_trait_scores={"Clarity": 4},
        )
        builder = RubricDataFrameBuilder(results=[result])
        df = builder.build_dataframe(trait_type="llm")

        assert df["rubric_evaluation_performed"].iloc[0] == True  # noqa: E712

    def test_column_present_for_regex_traits(self):
        """rubric_evaluation_performed column exists for regex trait rows."""
        result = _make_result(
            "q1",
            rubric_evaluation_performed=True,
            rubric_evaluation_strategy="sequential",
            regex_trait_scores={"HasCitations": True},
        )
        builder = RubricDataFrameBuilder(results=[result])
        df = builder.build_dataframe(trait_type="regex")

        assert "rubric_evaluation_performed" in df.columns
        assert df["rubric_evaluation_performed"].iloc[0] == True  # noqa: E712

    def test_column_present_for_callable_traits(self):
        """rubric_evaluation_performed column exists for callable trait rows."""
        result = _make_result(
            "q1",
            rubric_evaluation_performed=True,
            rubric_evaluation_strategy="batch",
            callable_trait_scores={"ContainsCitations": True},
        )
        builder = RubricDataFrameBuilder(results=[result])
        df = builder.build_dataframe(trait_type="callable")

        assert "rubric_evaluation_performed" in df.columns
        assert df["rubric_evaluation_performed"].iloc[0] == True  # noqa: E712

    def test_column_present_for_metric_traits(self):
        """rubric_evaluation_performed column exists for metric trait rows."""
        result = _make_result(
            "q1",
            rubric_evaluation_performed=True,
            rubric_evaluation_strategy="batch",
            metric_trait_scores={"EntityExtraction": {"precision": 0.85, "recall": 0.90, "f1": 0.87}},
        )
        builder = RubricDataFrameBuilder(results=[result])
        df = builder.build_dataframe(trait_type="metric")

        assert "rubric_evaluation_performed" in df.columns
        assert all(df["rubric_evaluation_performed"] == True)  # noqa: E712

    def test_column_present_for_agentic_traits(self):
        """rubric_evaluation_performed column exists for agentic trait rows."""
        result = _make_result(
            "q1",
            rubric_evaluation_performed=True,
            rubric_evaluation_strategy="batch",
            agentic_trait_scores={"CodeQuality": 4},
        )
        builder = RubricDataFrameBuilder(results=[result])
        df = builder.build_dataframe(trait_type="agentic")

        assert "rubric_evaluation_performed" in df.columns
        assert df["rubric_evaluation_performed"].iloc[0] == True  # noqa: E712

    def test_column_present_in_all_trait_type(self):
        """rubric_evaluation_performed column exists when trait_type='all'."""
        result = _make_result(
            "q1",
            rubric_evaluation_performed=True,
            rubric_evaluation_strategy="batch",
            llm_trait_scores={"Clarity": 3},
            regex_trait_scores={"HasNumbers": False},
        )
        builder = RubricDataFrameBuilder(results=[result])
        df = builder.build_dataframe(trait_type="all")

        assert "rubric_evaluation_performed" in df.columns
        assert all(df["rubric_evaluation_performed"] == True)  # noqa: E712

    def test_zero_rows_when_not_performed(self):
        """Results with rubric_evaluation_performed=False produce zero rows (issue 186)."""
        result = _make_result(
            "q1",
            rubric_evaluation_performed=False,
            rubric_evaluation_strategy=None,
        )
        # Ghost row fix (issue 186): results where rubric evaluation was not
        # performed are skipped entirely to prevent ghost rows with trait_name=None.
        builder = RubricDataFrameBuilder(results=[result])
        df = builder.build_dataframe(trait_type="all")

        assert len(df) == 0


# =============================================================================
# Tests: rubric_evaluation_strategy column
# =============================================================================


@pytest.mark.unit
class TestRubricEvaluationStrategyColumn:
    """rubric_evaluation_strategy must appear in all trait row types."""

    def test_column_present_for_llm_traits(self):
        """rubric_evaluation_strategy column exists for LLM trait rows."""
        result = _make_result(
            "q1",
            rubric_evaluation_performed=True,
            rubric_evaluation_strategy="batch",
            llm_trait_scores={"Clarity": 4},
        )
        builder = RubricDataFrameBuilder(results=[result])
        df = builder.build_dataframe(trait_type="llm")

        assert "rubric_evaluation_strategy" in df.columns

    def test_column_value_batch_strategy(self):
        """rubric_evaluation_strategy contains 'batch' when strategy is batch."""
        result = _make_result(
            "q1",
            rubric_evaluation_performed=True,
            rubric_evaluation_strategy="batch",
            llm_trait_scores={"Clarity": 4},
        )
        builder = RubricDataFrameBuilder(results=[result])
        df = builder.build_dataframe(trait_type="llm")

        assert df["rubric_evaluation_strategy"].iloc[0] == "batch"

    def test_column_value_sequential_strategy(self):
        """rubric_evaluation_strategy contains 'sequential' when strategy is sequential."""
        result = _make_result(
            "q1",
            rubric_evaluation_performed=True,
            rubric_evaluation_strategy="sequential",
            llm_trait_scores={"Clarity": 4},
        )
        builder = RubricDataFrameBuilder(results=[result])
        df = builder.build_dataframe(trait_type="llm")

        assert df["rubric_evaluation_strategy"].iloc[0] == "sequential"

    def test_column_none_when_strategy_is_none(self):
        """rubric_evaluation_strategy is None when no strategy was set."""
        # A result that has rubric data but strategy is None
        rubric = VerificationResultRubric(
            rubric_evaluation_performed=True,
            rubric_evaluation_strategy=None,
            llm_trait_scores={"Clarity": 4},
        )
        result = VerificationResult(
            metadata=create_metadata("q_none_strategy"),
            template=_make_template(),
            rubric=rubric,
        )
        builder = RubricDataFrameBuilder(results=[result])
        df = builder.build_dataframe(trait_type="llm")

        assert "rubric_evaluation_strategy" in df.columns
        assert df["rubric_evaluation_strategy"].iloc[0] is None

    def test_column_present_for_regex_traits(self):
        """rubric_evaluation_strategy column exists for regex trait rows."""
        result = _make_result(
            "q1",
            rubric_evaluation_performed=True,
            rubric_evaluation_strategy="sequential",
            regex_trait_scores={"HasCitations": True},
        )
        builder = RubricDataFrameBuilder(results=[result])
        df = builder.build_dataframe(trait_type="regex")

        assert "rubric_evaluation_strategy" in df.columns
        assert df["rubric_evaluation_strategy"].iloc[0] == "sequential"

    def test_column_present_for_callable_traits(self):
        """rubric_evaluation_strategy column exists for callable trait rows."""
        result = _make_result(
            "q1",
            rubric_evaluation_performed=True,
            rubric_evaluation_strategy="batch",
            callable_trait_scores={"ContainsCitations": True},
        )
        builder = RubricDataFrameBuilder(results=[result])
        df = builder.build_dataframe(trait_type="callable")

        assert "rubric_evaluation_strategy" in df.columns
        assert df["rubric_evaluation_strategy"].iloc[0] == "batch"

    def test_column_present_for_metric_traits(self):
        """rubric_evaluation_strategy column exists for metric trait rows."""
        result = _make_result(
            "q1",
            rubric_evaluation_performed=True,
            rubric_evaluation_strategy="batch",
            metric_trait_scores={"EntityExtraction": {"precision": 0.85, "recall": 0.90, "f1": 0.87}},
        )
        builder = RubricDataFrameBuilder(results=[result])
        df = builder.build_dataframe(trait_type="metric")

        assert "rubric_evaluation_strategy" in df.columns
        assert all(df["rubric_evaluation_strategy"] == "batch")

    def test_column_present_for_agentic_traits(self):
        """rubric_evaluation_strategy column exists for agentic trait rows."""
        result = _make_result(
            "q1",
            rubric_evaluation_performed=True,
            rubric_evaluation_strategy="batch",
            agentic_trait_scores={"CodeQuality": 4},
        )
        builder = RubricDataFrameBuilder(results=[result])
        df = builder.build_dataframe(trait_type="agentic")

        assert "rubric_evaluation_strategy" in df.columns
        assert df["rubric_evaluation_strategy"].iloc[0] == "batch"

    def test_column_present_in_all_trait_type(self):
        """rubric_evaluation_strategy column exists when trait_type='all'."""
        result = _make_result(
            "q1",
            rubric_evaluation_performed=True,
            rubric_evaluation_strategy="sequential",
            llm_trait_scores={"Clarity": 3},
            regex_trait_scores={"HasNumbers": False},
        )
        builder = RubricDataFrameBuilder(results=[result])
        df = builder.build_dataframe(trait_type="all")

        assert "rubric_evaluation_strategy" in df.columns
        assert all(df["rubric_evaluation_strategy"] == "sequential")


# =============================================================================
# Tests: column ordering
# =============================================================================


@pytest.mark.unit
class TestMetadataColumnOrdering:
    """rubric_evaluation_performed and rubric_evaluation_strategy must appear
    after run_name and before deep judgment columns."""

    def test_columns_appear_before_trait_name(self):
        """Both columns appear before trait_name in column order."""
        result = _make_result(
            "q1",
            rubric_evaluation_performed=True,
            rubric_evaluation_strategy="batch",
            llm_trait_scores={"Clarity": 4},
        )
        builder = RubricDataFrameBuilder(results=[result])
        df = builder.build_dataframe(trait_type="llm")

        cols = list(df.columns)
        trait_name_idx = cols.index("trait_name")
        performed_idx = cols.index("rubric_evaluation_performed")
        strategy_idx = cols.index("rubric_evaluation_strategy")

        assert performed_idx < trait_name_idx, "rubric_evaluation_performed must come before trait_name"
        assert strategy_idx < trait_name_idx, "rubric_evaluation_strategy must come before trait_name"

    def test_performed_before_strategy(self):
        """rubric_evaluation_performed appears before rubric_evaluation_strategy."""
        result = _make_result(
            "q1",
            rubric_evaluation_performed=True,
            rubric_evaluation_strategy="batch",
            llm_trait_scores={"Clarity": 4},
        )
        builder = RubricDataFrameBuilder(results=[result])
        df = builder.build_dataframe(trait_type="llm")

        cols = list(df.columns)
        performed_idx = cols.index("rubric_evaluation_performed")
        strategy_idx = cols.index("rubric_evaluation_strategy")

        assert performed_idx < strategy_idx

    def test_columns_before_deep_judgment_when_included(self):
        """New columns appear before deep judgment columns when deep_judgment is enabled."""
        result = _make_result(
            "q1",
            rubric_evaluation_performed=True,
            rubric_evaluation_strategy="batch",
            llm_trait_scores={"Clarity": 4},
        )
        builder = RubricDataFrameBuilder(results=[result], include_deep_judgment=True)
        df = builder.build_dataframe(trait_type="llm")

        cols = list(df.columns)
        performed_idx = cols.index("rubric_evaluation_performed")
        strategy_idx = cols.index("rubric_evaluation_strategy")

        # trait_reasoning is the first deep judgment column
        if "trait_reasoning" in cols:
            reasoning_idx = cols.index("trait_reasoning")
            assert performed_idx < reasoning_idx
            assert strategy_idx < reasoning_idx


# =============================================================================
# Tests: multiple results with different strategies
# =============================================================================


@pytest.mark.unit
class TestMetadataColumnsMultipleResults:
    """Columns are correctly populated across multiple results."""

    def test_different_strategies_preserved(self):
        """Different strategies across results are all preserved correctly."""
        result_batch = _make_result(
            "q_batch",
            rubric_evaluation_performed=True,
            rubric_evaluation_strategy="batch",
            llm_trait_scores={"Clarity": 4},
        )
        result_sequential = _make_result(
            "q_sequential",
            rubric_evaluation_performed=True,
            rubric_evaluation_strategy="sequential",
            llm_trait_scores={"Clarity": 3},
        )
        builder = RubricDataFrameBuilder(results=[result_batch, result_sequential])
        df = builder.build_dataframe(trait_type="llm")

        assert "rubric_evaluation_strategy" in df.columns
        strategies = set(df["rubric_evaluation_strategy"].unique())
        assert strategies == {"batch", "sequential"}

    def test_performed_true_for_all_evaluated_results(self):
        """rubric_evaluation_performed is True for all results that had evaluation."""
        result1 = _make_result(
            "q1",
            rubric_evaluation_performed=True,
            rubric_evaluation_strategy="batch",
            llm_trait_scores={"Clarity": 4},
        )
        result2 = _make_result(
            "q2",
            rubric_evaluation_performed=True,
            rubric_evaluation_strategy="batch",
            llm_trait_scores={"Clarity": 5},
        )
        builder = RubricDataFrameBuilder(results=[result1, result2])
        df = builder.build_dataframe(trait_type="llm")

        assert all(df["rubric_evaluation_performed"] == True)  # noqa: E712
