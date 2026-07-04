"""Unit tests for RubricDataFrameBuilder column completeness and ghost row fix.

Covers issues 147, 156, 164, 186, 122, and 155:
- Issue 147: failed_stage column in all row creators
- Issue 156: rubric_evaluation_performed and rubric_evaluation_strategy columns
- Issue 164: scenario columns in all row creators
- Issue 186: ghost rows eliminated when rubric is None or evaluation not performed
- Issue 122: deep judgment columns documented as LLM-only
- Issue 155: confusion data duplication documented in metric trait row
"""

import pytest

from karenina.schemas.dataframes.rubric import RubricDataFrameBuilder
from karenina.schemas.results.failure import Failure, FailureCategory
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
        "llm_trait_scores": {"Clarity": 4},
    }
    defaults.update(overrides)
    return VerificationResultRubric(**defaults)


def _make_result(
    question_id: str = "q001",
    rubric: VerificationResultRubric | None = None,
    failed_stage: str | None = None,
    scenario_id: str | None = None,
    scenario_node: str | None = None,
    scenario_turn: int | None = None,
    scenario_path: list[str] | None = None,
) -> VerificationResult:
    """Build a VerificationResult with optional metadata and rubric overrides."""
    metadata = create_metadata(question_id)
    if failed_stage is not None:
        metadata.failure = Failure(
            category=FailureCategory.UNEXPECTED_ERROR,
            stage=failed_stage,
            reason="test failure",
        )
    metadata.scenario_id = scenario_id
    metadata.scenario_node = scenario_node
    metadata.scenario_turn = scenario_turn
    metadata.scenario_path = scenario_path
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
# Issue 147: failed_stage column
# =============================================================================


@pytest.mark.unit
class TestFailedStageColumn:
    """Issue 147: All row creators must include failed_stage."""

    def test_llm_trait_row_has_failed_stage(self):
        """LLM trait rows must contain failed_stage."""
        result = _make_result(rubric=_make_rubric(), failed_stage="abstention_check")
        df = _build_df([result], trait_type="llm")

        assert "failure_stage" in df.columns
        assert df["failure_stage"].iloc[0] == "abstention_check"

    def test_regex_trait_row_has_failed_stage(self):
        """Regex trait rows must contain failed_stage."""
        rubric = _make_rubric(regex_trait_scores={"HasCitations": True})
        result = _make_result(rubric=rubric, failed_stage="parse_template")
        df = _build_df([result], trait_type="regex")

        assert "failure_stage" in df.columns
        assert df["failure_stage"].iloc[0] == "parse_template"

    def test_callable_trait_row_has_failed_stage(self):
        """Callable trait rows must contain failed_stage."""
        rubric = _make_rubric(callable_trait_scores={"LenCheck": True})
        result = _make_result(rubric=rubric, failed_stage="verify_template")
        df = _build_df([result], trait_type="callable")

        assert "failure_stage" in df.columns
        assert df["failure_stage"].iloc[0] == "verify_template"

    def test_metric_trait_row_has_failed_stage(self):
        """Metric trait rows must contain failed_stage."""
        rubric = _make_rubric(
            metric_trait_scores={"Entity": {"precision": 0.9, "recall": 0.8, "f1": 0.85}},
        )
        result = _make_result(rubric=rubric, failed_stage="embedding_check")
        df = _build_df([result], trait_type="metric")

        assert "failure_stage" in df.columns
        assert (df["failure_stage"] == "embedding_check").all()

    def test_agentic_trait_row_has_failed_stage(self):
        """Agentic trait rows must contain failed_stage."""
        rubric = _make_rubric(agentic_trait_scores={"Quality": 4})
        result = _make_result(rubric=rubric, failed_stage="generate_answer")
        df = _build_df([result], trait_type="agentic")

        assert "failure_stage" in df.columns
        assert df["failure_stage"].iloc[0] == "generate_answer"

    def test_failed_stage_none_when_not_set(self):
        """failed_stage is None when no stage failed."""
        result = _make_result(rubric=_make_rubric(), failed_stage=None)
        df = _build_df([result], trait_type="llm")

        assert "failure_stage" in df.columns
        assert df["failure_stage"].iloc[0] is None


# =============================================================================
# Issue 156: rubric evaluation metadata columns
# =============================================================================


@pytest.mark.unit
class TestRubricEvaluationMetadataColumns:
    """Issue 156: All row creators must include rubric_evaluation_performed
    and rubric_evaluation_strategy."""

    def test_llm_trait_row_has_rubric_eval_columns(self):
        """LLM trait rows must have rubric evaluation metadata."""
        rubric = _make_rubric(rubric_evaluation_strategy="sequential")
        result = _make_result(rubric=rubric)
        df = _build_df([result], trait_type="llm")

        assert "rubric_evaluation_performed" in df.columns
        assert "rubric_evaluation_strategy" in df.columns
        assert bool(df["rubric_evaluation_performed"].iloc[0]) is True
        assert df["rubric_evaluation_strategy"].iloc[0] == "sequential"

    def test_regex_trait_row_has_rubric_eval_columns(self):
        """Regex trait rows must have rubric evaluation metadata."""
        rubric = _make_rubric(regex_trait_scores={"HasCitations": True})
        result = _make_result(rubric=rubric)
        df = _build_df([result], trait_type="regex")

        assert "rubric_evaluation_performed" in df.columns
        assert "rubric_evaluation_strategy" in df.columns
        assert bool(df["rubric_evaluation_performed"].iloc[0]) is True

    def test_callable_trait_row_has_rubric_eval_columns(self):
        """Callable trait rows must have rubric evaluation metadata."""
        rubric = _make_rubric(callable_trait_scores={"LenCheck": True})
        result = _make_result(rubric=rubric)
        df = _build_df([result], trait_type="callable")

        assert "rubric_evaluation_performed" in df.columns
        assert "rubric_evaluation_strategy" in df.columns

    def test_metric_trait_row_has_rubric_eval_columns(self):
        """Metric trait rows must have rubric evaluation metadata."""
        rubric = _make_rubric(
            metric_trait_scores={"Entity": {"precision": 0.9}},
        )
        result = _make_result(rubric=rubric)
        df = _build_df([result], trait_type="metric")

        assert "rubric_evaluation_performed" in df.columns
        assert "rubric_evaluation_strategy" in df.columns

    def test_agentic_trait_row_has_rubric_eval_columns(self):
        """Agentic trait rows must have rubric evaluation metadata."""
        rubric = _make_rubric(agentic_trait_scores={"Quality": 4})
        result = _make_result(rubric=rubric)
        df = _build_df([result], trait_type="agentic")

        assert "rubric_evaluation_performed" in df.columns
        assert "rubric_evaluation_strategy" in df.columns


# =============================================================================
# Issue 164: scenario columns
# =============================================================================


@pytest.mark.unit
class TestScenarioColumns:
    """Issue 164: All row creators must include 4 scenario columns."""

    SCENARIO_COLUMNS = [
        "scenario_id",
        "scenario_node",
        "scenario_turn",
        "scenario_path",
    ]

    def test_llm_trait_row_has_scenario_columns(self):
        """LLM trait rows must contain scenario columns."""
        result = _make_result(
            rubric=_make_rubric(),
            scenario_id="sc-001",
            scenario_node="greeting",
            scenario_turn=2,
            scenario_path=["greeting", "followup"],
        )
        df = _build_df([result], trait_type="llm")

        for col in self.SCENARIO_COLUMNS:
            assert col in df.columns, f"Missing scenario column: {col}"
        assert df["scenario_id"].iloc[0] == "sc-001"
        assert df["scenario_node"].iloc[0] == "greeting"
        assert df["scenario_turn"].iloc[0] == 2
        assert df["scenario_path"].iloc[0] == ["greeting", "followup"]

    def test_regex_trait_row_has_scenario_columns(self):
        """Regex trait rows must contain scenario columns."""
        rubric = _make_rubric(regex_trait_scores={"HasCitations": True})
        result = _make_result(
            rubric=rubric,
            scenario_id="sc-002",
            scenario_node="node-a",
            scenario_turn=1,
            scenario_path=["node-a"],
        )
        df = _build_df([result], trait_type="regex")

        for col in self.SCENARIO_COLUMNS:
            assert col in df.columns, f"Missing scenario column: {col}"
        assert df["scenario_id"].iloc[0] == "sc-002"

    def test_callable_trait_row_has_scenario_columns(self):
        """Callable trait rows must contain scenario columns."""
        rubric = _make_rubric(callable_trait_scores={"LenCheck": True})
        result = _make_result(rubric=rubric, scenario_id="sc-003")
        df = _build_df([result], trait_type="callable")

        for col in self.SCENARIO_COLUMNS:
            assert col in df.columns, f"Missing scenario column: {col}"

    def test_metric_trait_row_has_scenario_columns(self):
        """Metric trait rows must contain scenario columns."""
        rubric = _make_rubric(
            metric_trait_scores={"Entity": {"f1": 0.85}},
        )
        result = _make_result(rubric=rubric, scenario_id="sc-004")
        df = _build_df([result], trait_type="metric")

        for col in self.SCENARIO_COLUMNS:
            assert col in df.columns, f"Missing scenario column: {col}"

    def test_agentic_trait_row_has_scenario_columns(self):
        """Agentic trait rows must contain scenario columns."""
        rubric = _make_rubric(agentic_trait_scores={"Quality": 4})
        result = _make_result(rubric=rubric, scenario_id="sc-005")
        df = _build_df([result], trait_type="agentic")

        for col in self.SCENARIO_COLUMNS:
            assert col in df.columns, f"Missing scenario column: {col}"

    def test_scenario_none_for_standalone(self):
        """Scenario columns are None for standalone (non-scenario) questions."""
        result = _make_result(rubric=_make_rubric())
        df = _build_df([result], trait_type="llm")

        for col in self.SCENARIO_COLUMNS:
            assert df[col].iloc[0] is None, f"Expected None for standalone {col}"


# =============================================================================
# Issue 186: ghost row fix
# =============================================================================


@pytest.mark.unit
class TestGhostRowFix:
    """Issue 186: build_dataframe() must emit zero rows when rubric is None
    or rubric_evaluation_performed is False, not a ghost row with trait_name=None."""

    def test_no_row_when_rubric_is_none(self):
        """A result with rubric=None should produce zero rows."""
        result = _make_result(rubric=None)
        df = _build_df([result])

        assert len(df) == 0, f"Expected zero rows for result with rubric=None, got {len(df)} rows"

    def test_no_row_when_evaluation_not_performed(self):
        """A result with rubric_evaluation_performed=False should produce zero rows."""
        rubric = VerificationResultRubric(
            rubric_evaluation_performed=False,
            rubric_evaluation_strategy=None,
        )
        result = _make_result(rubric=rubric)
        df = _build_df([result])

        assert len(df) == 0, f"Expected zero rows for result with rubric_evaluation_performed=False, got {len(df)} rows"

    def test_mix_rubric_and_no_rubric(self):
        """Mixed results: only results with rubric data produce rows."""
        result_with = _make_result(question_id="q_with", rubric=_make_rubric())
        result_without = _make_result(question_id="q_without", rubric=None)
        df = _build_df([result_with, result_without])

        # Only the result with rubric should produce rows
        assert (df["question_id"] == "q_with").all()
        assert "q_without" not in df["question_id"].values

    def test_no_ghost_rows_pollute_value_counts(self):
        """Ghost rows with trait_name=None must not appear in value_counts."""
        result_with = _make_result(question_id="q_with", rubric=_make_rubric())
        result_none = _make_result(question_id="q_none", rubric=None)
        df = _build_df([result_with, result_none])

        # No None in trait_name
        assert df["trait_name"].isna().sum() == 0

    def test_empty_df_when_all_rubrics_none(self):
        """When all results lack rubric data, the DataFrame is empty."""
        r1 = _make_result(question_id="q1", rubric=None)
        r2 = _make_result(question_id="q2", rubric=None)
        df = _build_df([r1, r2])

        assert df.empty


# =============================================================================
# Column ordering
# =============================================================================


@pytest.mark.unit
class TestColumnOrdering:
    """New columns must appear in the correct positions in _get_column_order."""

    def test_failure_stage_after_category(self):
        """failure_stage must come after failure_category/group in column order."""
        result = _make_result(rubric=_make_rubric(), failed_stage="abstention_check")
        df = _build_df([result])

        cols = list(df.columns)
        group_idx = cols.index("failure_group")
        stage_idx = cols.index("failure_stage")
        assert stage_idx == group_idx + 1, (
            f"failure_stage at {stage_idx} should be right after failure_group at {group_idx}"
        )

    def test_rubric_eval_columns_after_system_prompts(self):
        """rubric_evaluation_performed/strategy must come after parsing_system_prompt."""
        result = _make_result(rubric=_make_rubric())
        df = _build_df([result])

        cols = list(df.columns)
        parsing_sp_idx = cols.index("parsing_system_prompt")
        eval_performed_idx = cols.index("rubric_evaluation_performed")
        eval_strategy_idx = cols.index("rubric_evaluation_strategy")
        assert eval_performed_idx == parsing_sp_idx + 1
        assert eval_strategy_idx == parsing_sp_idx + 2

    def test_scenario_columns_after_replicate(self):
        """Scenario columns must come after replicate."""
        result = _make_result(
            rubric=_make_rubric(),
            scenario_id="sc-001",
            scenario_node="node",
            scenario_turn=1,
            scenario_path=["node"],
        )
        df = _build_df([result])

        cols = list(df.columns)
        replicate_idx = cols.index("replicate")
        scenario_id_idx = cols.index("scenario_id")
        assert scenario_id_idx == replicate_idx + 1

    def test_all_new_columns_in_order(self):
        """Verify the full ordering of new columns relative to neighbors."""
        rubric = _make_rubric(
            regex_trait_scores={"HasCitations": True},
            metric_trait_scores={"Entity": {"precision": 0.9}},
        )
        result = _make_result(
            rubric=rubric,
            failed_stage="parse_template",
            scenario_id="sc-001",
            scenario_node="node",
            scenario_turn=1,
            scenario_path=["node"],
        )
        df = _build_df([result])

        cols = list(df.columns)

        # Status section order
        assert cols.index("success") < cols.index("failure_category")
        assert cols.index("failure_category") < cols.index("failure_group")
        assert cols.index("failure_group") < cols.index("failure_stage")
        assert cols.index("failure_stage") < cols.index("failure_reason")
        assert cols.index("failure_reason") < cols.index("caveats")

        # Identification section: replicate before scenario columns
        assert cols.index("replicate") < cols.index("scenario_id")
        assert cols.index("scenario_id") < cols.index("scenario_node")
        assert cols.index("scenario_node") < cols.index("scenario_turn")
        assert cols.index("scenario_turn") < cols.index("scenario_path")

        # Scenario columns before model config
        assert cols.index("scenario_path") < cols.index("answering_model")

        # Rubric eval metadata after system prompts, before rubric data
        assert cols.index("parsing_system_prompt") < cols.index("rubric_evaluation_performed")
        assert cols.index("rubric_evaluation_strategy") < cols.index("trait_name")
