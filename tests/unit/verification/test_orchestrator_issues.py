"""Tests for StageOrchestrator bug fixes.

Issue 157: rubric_only mode silently excludes sufficiency check.
Issue 158: DeepJudgmentRubricAutoFailStage always included in rubric_only.
"""

import logging

import pytest

from karenina.benchmark.verification.stages.core.orchestrator import StageOrchestrator
from karenina.benchmark.verification.stages.pipeline.deep_judgment_rubric_auto_fail import (
    DeepJudgmentRubricAutoFailStage,
)
from karenina.benchmark.verification.stages.pipeline.sufficiency_check import SufficiencyCheckStage
from karenina.schemas.entities.rubric import LLMRubricTrait, Rubric


def _make_rubric_with_llm_trait() -> Rubric:
    """Create a minimal rubric with one LLM trait for testing."""
    return Rubric(
        llm_traits=[
            LLMRubricTrait(
                name="clarity",
                description="Is the response clear?",
                kind="boolean",
            )
        ]
    )


def _get_stage_names(orchestrator: StageOrchestrator) -> list[str]:
    """Extract stage names from orchestrator."""
    return [stage.name for stage in orchestrator.stages]


@pytest.mark.unit
class TestRubricOnlySufficiencyLog:
    """Issue 157: rubric_only with sufficiency_enabled should log a message."""

    def test_rubric_only_with_sufficiency_enabled_logs_info(self, caplog):
        """When rubric_only and sufficiency_enabled=True, a log message should be emitted."""
        rubric = _make_rubric_with_llm_trait()

        with caplog.at_level(logging.INFO):
            orchestrator = StageOrchestrator.from_config(
                rubric=rubric,
                evaluation_mode="rubric_only",
                sufficiency_enabled=True,
            )

        # Sufficiency stage should NOT be in the pipeline
        assert not any(isinstance(s, SufficiencyCheckStage) for s in orchestrator.stages), (
            "SufficiencyCheckStage should not be in rubric_only pipeline"
        )

        # But an info log should have been emitted explaining the skip
        assert any(
            "sufficiency" in record.message.lower() and "rubric_only" in record.message.lower()
            for record in caplog.records
        ), (
            f"Expected info log about sufficiency being skipped in rubric_only mode, "
            f"got: {[r.message for r in caplog.records]}"
        )

    def test_rubric_only_without_sufficiency_no_log(self, caplog):
        """When rubric_only and sufficiency_enabled=False, no sufficiency log should be emitted."""
        rubric = _make_rubric_with_llm_trait()

        with caplog.at_level(logging.INFO):
            StageOrchestrator.from_config(
                rubric=rubric,
                evaluation_mode="rubric_only",
                sufficiency_enabled=False,
            )

        # No sufficiency-related log expected
        assert not any(
            "sufficiency" in record.message.lower() and "rubric_only" in record.message.lower()
            for record in caplog.records
        )


@pytest.mark.unit
class TestRubricOnlyDeepJudgmentGating:
    """Issue 158: DeepJudgmentRubricAutoFailStage should be gated on deep_judgment_enabled."""

    def test_rubric_only_includes_deep_judgment_rubric_without_flag(self):
        """Before fix: DeepJudgmentRubricAutoFailStage is always included in rubric_only."""
        rubric = _make_rubric_with_llm_trait()

        orchestrator = StageOrchestrator.from_config(
            rubric=rubric,
            evaluation_mode="rubric_only",
            deep_judgment_enabled=False,
        )

        has_dj_rubric = any(isinstance(s, DeepJudgmentRubricAutoFailStage) for s in orchestrator.stages)
        # After fix, it should NOT be present when deep_judgment_enabled=False
        assert not has_dj_rubric, (
            "DeepJudgmentRubricAutoFailStage should not be in pipeline when deep_judgment_enabled=False"
        )

    def test_rubric_only_includes_deep_judgment_rubric_with_flag(self):
        """When deep_judgment_enabled=True, DeepJudgmentRubricAutoFailStage should be present."""
        rubric = _make_rubric_with_llm_trait()

        orchestrator = StageOrchestrator.from_config(
            rubric=rubric,
            evaluation_mode="rubric_only",
            deep_judgment_enabled=True,
        )

        has_dj_rubric = any(isinstance(s, DeepJudgmentRubricAutoFailStage) for s in orchestrator.stages)
        assert has_dj_rubric, "DeepJudgmentRubricAutoFailStage should be in pipeline when deep_judgment_enabled=True"

    def test_template_and_rubric_deep_judgment_gating_consistent(self):
        """template_and_rubric mode should also gate DeepJudgmentRubricAutoFailStage.

        This ensures the fix is consistent across both modes.
        """
        rubric = _make_rubric_with_llm_trait()

        # Without deep_judgment_enabled
        orchestrator_off = StageOrchestrator.from_config(
            rubric=rubric,
            evaluation_mode="template_and_rubric",
            deep_judgment_enabled=False,
        )
        has_dj_rubric_off = any(isinstance(s, DeepJudgmentRubricAutoFailStage) for s in orchestrator_off.stages)

        # With deep_judgment_enabled
        orchestrator_on = StageOrchestrator.from_config(
            rubric=rubric,
            evaluation_mode="template_and_rubric",
            deep_judgment_enabled=True,
        )
        has_dj_rubric_on = any(isinstance(s, DeepJudgmentRubricAutoFailStage) for s in orchestrator_on.stages)

        assert not has_dj_rubric_off, (
            "template_and_rubric should not include DeepJudgmentRubricAutoFailStage when deep_judgment_enabled=False"
        )
        assert has_dj_rubric_on, (
            "template_and_rubric should include DeepJudgmentRubricAutoFailStage when deep_judgment_enabled=True"
        )
