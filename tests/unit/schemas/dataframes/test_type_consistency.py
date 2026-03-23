"""Unit tests for DataFrame type consistency and usage scoping.

Covers three issues:
- Issue 162: replicate column Int64 dtype consistency across all builders
- Issue 123: trait_hallucination_risk stores string, not dict
- Issue 126: agent metrics scoped to answer_generation stage only
"""

import pandas as pd
import pytest

from karenina.schemas.dataframes.judgment import JudgmentDataFrameBuilder
from karenina.schemas.dataframes.rubric import RubricDataFrameBuilder
from karenina.schemas.dataframes.template import TemplateDataFrameBuilder
from karenina.schemas.verification import (
    VerificationResult,
    VerificationResultDeepJudgment,
    VerificationResultDeepJudgmentRubric,
    VerificationResultRubric,
    VerificationResultTemplate,
)
from tests.integration.dataframe_helpers import create_metadata

# =============================================================================
# Helpers
# =============================================================================


def _make_result(
    question_id: str = "q001",
    replicate: int | None = None,
    template: VerificationResultTemplate | None = None,
    rubric: VerificationResultRubric | None = None,
    deep_judgment: VerificationResultDeepJudgment | None = None,
    deep_judgment_rubric: VerificationResultDeepJudgmentRubric | None = None,
) -> VerificationResult:
    """Build a VerificationResult with configurable replicate and components."""
    metadata = create_metadata(question_id)
    metadata.replicate = replicate
    return VerificationResult(
        metadata=metadata,
        template=template,
        rubric=rubric,
        deep_judgment=deep_judgment,
        deep_judgment_rubric=deep_judgment_rubric,
    )


# =============================================================================
# Issue 162: Replicate column Int64 dtype
# =============================================================================


@pytest.mark.unit
class TestReplicateInt64Dtype:
    """Issue 162: replicate column must use pd.Int64Dtype() in all three builders."""

    def test_template_field_df_mixed_replicates_uses_int64(self):
        """TemplateDataFrameBuilder.build_field_dataframe() coerces replicate to Int64
        even when some values are None."""
        template = VerificationResultTemplate(
            raw_llm_response="answer",
            parsed_gt_response={"answer": "Paris"},
            parsed_llm_response={"answer": "Paris"},
        )
        r1 = _make_result(question_id="q001", replicate=1, template=template)
        r2 = _make_result(question_id="q002", replicate=None, template=template)

        builder = TemplateDataFrameBuilder([r1, r2])
        df = builder.build_field_dataframe()

        assert "replicate" in df.columns
        assert df["replicate"].dtype == pd.Int64Dtype()

    def test_template_field_df_all_replicates_set_uses_int64(self):
        """When all replicates are set, dtype is Int64 (not int64)."""
        template = VerificationResultTemplate(
            raw_llm_response="answer",
            parsed_gt_response={"answer": "Paris"},
            parsed_llm_response={"answer": "Paris"},
        )
        r1 = _make_result(question_id="q001", replicate=1, template=template)
        r2 = _make_result(question_id="q002", replicate=2, template=template)

        builder = TemplateDataFrameBuilder([r1, r2])
        df = builder.build_field_dataframe()

        assert df["replicate"].dtype == pd.Int64Dtype()

    def test_rubric_df_mixed_replicates_uses_int64(self):
        """RubricDataFrameBuilder.build_dataframe() coerces replicate to Int64."""
        rubric = VerificationResultRubric(
            rubric_evaluation_performed=True,
            llm_trait_scores={"clarity": 4},
        )
        r1 = _make_result(question_id="q001", replicate=1, rubric=rubric)
        r2 = _make_result(question_id="q002", replicate=None, rubric=rubric)

        builder = RubricDataFrameBuilder([r1, r2])
        df = builder.build_dataframe()

        assert "replicate" in df.columns
        assert df["replicate"].dtype == pd.Int64Dtype()

    def test_rubric_df_all_replicates_set_uses_int64(self):
        """When all replicates are set, rubric dtype is Int64 (not int64)."""
        rubric = VerificationResultRubric(
            rubric_evaluation_performed=True,
            llm_trait_scores={"clarity": 4},
        )
        r1 = _make_result(question_id="q001", replicate=1, rubric=rubric)
        r2 = _make_result(question_id="q002", replicate=2, rubric=rubric)

        builder = RubricDataFrameBuilder([r1, r2])
        df = builder.build_dataframe()

        assert df["replicate"].dtype == pd.Int64Dtype()

    def test_judgment_df_mixed_replicates_uses_int64(self):
        """JudgmentDataFrameBuilder.build_dataframe() coerces replicate to Int64."""
        dj = VerificationResultDeepJudgment(
            deep_judgment_performed=True,
            extracted_excerpts={
                "answer": [{"text": "excerpt", "confidence": "high"}],
            },
        )
        template = VerificationResultTemplate(
            raw_llm_response="answer",
            parsed_gt_response={"answer": "Paris"},
            parsed_llm_response={"answer": "Paris"},
        )
        r1 = _make_result(question_id="q001", replicate=1, template=template, deep_judgment=dj)
        r2 = _make_result(question_id="q002", replicate=None, template=template, deep_judgment=dj)

        builder = JudgmentDataFrameBuilder([r1, r2])
        df = builder.build_dataframe()

        assert "replicate" in df.columns
        assert df["replicate"].dtype == pd.Int64Dtype()

    def test_judgment_df_all_replicates_set_uses_int64(self):
        """When all replicates are set, judgment dtype is Int64 (not int64)."""
        dj = VerificationResultDeepJudgment(
            deep_judgment_performed=True,
            extracted_excerpts={
                "answer": [{"text": "excerpt", "confidence": "high"}],
            },
        )
        template = VerificationResultTemplate(
            raw_llm_response="answer",
            parsed_gt_response={"answer": "Paris"},
            parsed_llm_response={"answer": "Paris"},
        )
        r1 = _make_result(question_id="q001", replicate=1, template=template, deep_judgment=dj)
        r2 = _make_result(question_id="q002", replicate=2, template=template, deep_judgment=dj)

        builder = JudgmentDataFrameBuilder([r1, r2])
        df = builder.build_dataframe()

        assert df["replicate"].dtype == pd.Int64Dtype()


# =============================================================================
# Issue 123: Hallucination risk type inconsistency
# =============================================================================


@pytest.mark.unit
class TestHallucinationRiskType:
    """Issue 123: trait_hallucination_risk must be a string, not a dict."""

    def test_rubric_dj_hallucination_risk_is_string(self):
        """When rubric deep judgment has hallucination risk as a dict,
        the DataFrame column should contain just the overall_risk string."""
        rubric = VerificationResultRubric(
            rubric_evaluation_performed=True,
            llm_trait_scores={"clarity": 4},
        )
        rubric_dj = VerificationResultDeepJudgmentRubric(
            deep_judgment_rubric_performed=True,
            rubric_trait_reasoning={"clarity": "Good reasoning"},
            extracted_rubric_excerpts={"clarity": [{"text": "excerpt", "confidence": "high"}]},
            rubric_hallucination_risk_assessment={
                "clarity": {
                    "overall_risk": "low",
                    "per_excerpt_risks": ["low", "medium"],
                },
            },
        )

        result = _make_result(
            question_id="q001",
            replicate=1,
            rubric=rubric,
            deep_judgment_rubric=rubric_dj,
        )

        builder = RubricDataFrameBuilder([result], include_deep_judgment=True)
        df = builder.build_dataframe()

        risk_value = df["trait_hallucination_risk"].iloc[0]
        assert isinstance(risk_value, str), (
            f"trait_hallucination_risk should be a string, got {type(risk_value).__name__}"
        )
        assert risk_value == "low"

    def test_rubric_dj_hallucination_risk_none_when_absent(self):
        """When no hallucination risk data exists, value is None."""
        rubric = VerificationResultRubric(
            rubric_evaluation_performed=True,
            llm_trait_scores={"clarity": 4},
        )
        rubric_dj = VerificationResultDeepJudgmentRubric(
            deep_judgment_rubric_performed=True,
            rubric_trait_reasoning={"clarity": "Good reasoning"},
            extracted_rubric_excerpts={"clarity": []},
        )

        result = _make_result(
            question_id="q001",
            replicate=1,
            rubric=rubric,
            deep_judgment_rubric=rubric_dj,
        )

        builder = RubricDataFrameBuilder([result], include_deep_judgment=True)
        df = builder.build_dataframe()

        assert pd.isna(df["trait_hallucination_risk"].iloc[0])

    def test_rubric_dj_hallucination_risk_missing_trait(self):
        """If a trait has no entry in hallucination_risk_assessment, value is None."""
        rubric = VerificationResultRubric(
            rubric_evaluation_performed=True,
            llm_trait_scores={"clarity": 4, "tone": 3},
        )
        rubric_dj = VerificationResultDeepJudgmentRubric(
            deep_judgment_rubric_performed=True,
            rubric_trait_reasoning={"clarity": "reasoning", "tone": "reasoning"},
            extracted_rubric_excerpts={"clarity": [], "tone": []},
            rubric_hallucination_risk_assessment={
                "clarity": {
                    "overall_risk": "medium",
                    "per_excerpt_risks": [],
                },
                # "tone" intentionally missing
            },
        )

        result = _make_result(
            question_id="q001",
            replicate=1,
            rubric=rubric,
            deep_judgment_rubric=rubric_dj,
        )

        builder = RubricDataFrameBuilder([result], include_deep_judgment=True)
        df = builder.build_dataframe()

        clarity_row = df[df["trait_name"] == "clarity"]
        tone_row = df[df["trait_name"] == "tone"]

        assert clarity_row["trait_hallucination_risk"].iloc[0] == "medium"
        assert pd.isna(tone_row["trait_hallucination_risk"].iloc[0])


# =============================================================================
# Issue 126: Usage DataFrame agent metrics scoped to answer_generation
# =============================================================================


@pytest.mark.unit
class TestUsageAgentMetricsScoping:
    """Issue 126: agent metrics only populated for answer_generation stage."""

    def _make_usage_result(self) -> VerificationResult:
        """Create a result with multi-stage usage data and agent metrics."""
        template = VerificationResultTemplate(
            raw_llm_response="answer",
            parsed_gt_response={"answer": "Paris"},
            parsed_llm_response={"answer": "Paris"},
            usage_metadata={
                "answer_generation": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "total_tokens": 150,
                    "model": "claude-haiku-4-5",
                },
                "parsing": {
                    "input_tokens": 200,
                    "output_tokens": 100,
                    "total_tokens": 300,
                    "model": "claude-haiku-4-5",
                },
                "rubric_evaluation": {
                    "input_tokens": 150,
                    "output_tokens": 75,
                    "total_tokens": 225,
                    "model": "claude-haiku-4-5",
                },
                "total": {
                    "input_tokens": 450,
                    "output_tokens": 225,
                    "total_tokens": 675,
                },
            },
            agent_metrics={
                "iterations": 3,
                "tool_calls": 5,
                "tools_used": ["mcp__brave_search", "mcp__read_resource"],
                "suspect_failed_tool_calls": 1,
            },
        )
        return _make_result(question_id="q001", replicate=1, template=template)

    def test_agent_metrics_populated_for_answer_generation(self):
        """Agent metrics should be populated for the answer_generation stage."""
        result = self._make_usage_result()
        builder = TemplateDataFrameBuilder([result])
        df = builder.build_usage_dataframe()

        answer_rows = df[df["usage_stage"] == "answer_generation"]
        assert len(answer_rows) == 1

        row = answer_rows.iloc[0]
        assert row["agent_iterations"] == 3
        assert row["agent_tool_calls"] == 5
        assert row["agent_tools_used"] == ["mcp__brave_search", "mcp__read_resource"]
        assert row["agent_suspected_failures"] == 1

    def test_agent_metrics_none_for_other_stages(self):
        """Agent metrics should be None for non-answer_generation stages."""
        result = self._make_usage_result()
        builder = TemplateDataFrameBuilder([result])
        df = builder.build_usage_dataframe()

        other_rows = df[df["usage_stage"] != "answer_generation"]
        assert len(other_rows) > 0, "Expected rows for parsing and rubric_evaluation stages"

        for _, row in other_rows.iterrows():
            stage_name = row["usage_stage"]
            assert pd.isna(row["agent_iterations"]), f"agent_iterations should be None/NaN for stage {stage_name}"
            assert pd.isna(row["agent_tool_calls"]), f"agent_tool_calls should be None/NaN for stage {stage_name}"
            assert row["agent_tools_used"] is None, f"agent_tools_used should be None for stage {stage_name}"
            assert pd.isna(row["agent_suspected_failures"]), (
                f"agent_suspected_failures should be None/NaN for stage {stage_name}"
            )

    def test_totals_only_excludes_agent_metrics_detail(self):
        """In totals_only mode, the single row has no usage_stage;
        agent metrics should be None since there is no answer_generation stage row."""
        result = self._make_usage_result()
        builder = TemplateDataFrameBuilder([result])
        df = builder.build_usage_dataframe(totals_only=True)

        assert len(df) == 1
        row = df.iloc[0]
        # totals_only generates a "total" stage row, which is not answer_generation
        assert row["agent_iterations"] is None
        assert row["agent_tool_calls"] is None
