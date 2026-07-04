"""Tests for StageOrchestrator.from_config() with dynamic_rubric parameter.

Verifies that the orchestrator includes RubricEvaluationStage and
AgenticRubricEvaluationStage when a DynamicRubric contributes traits,
even when no regular Rubric is provided.
"""

import pytest

from karenina.benchmark.verification.stages import StageOrchestrator
from karenina.schemas.entities.rubric import (
    AgenticRubricTrait,
    DynamicRubric,
    LLMRubricTrait,
    RegexRubricTrait,
)


def _make_llm_dynamic_rubric() -> DynamicRubric:
    """Create a DynamicRubric with a single LLM trait."""
    return DynamicRubric(
        llm_traits=[
            LLMRubricTrait(
                name="clarity",
                summary="Response is clear and well structured",
                kind="boolean",
            )
        ]
    )


def _make_regex_dynamic_rubric() -> DynamicRubric:
    """Create a DynamicRubric with a single regex trait."""
    return DynamicRubric(
        regex_traits=[
            RegexRubricTrait(
                name="has_citations",
                summary="Response includes bracket citations",
                pattern=r"\[\d+\]",
            )
        ]
    )


def _make_agentic_dynamic_rubric() -> DynamicRubric:
    """Create a DynamicRubric with a single agentic trait."""
    return DynamicRubric(
        agentic_traits=[
            AgenticRubricTrait(
                name="code_quality",
                description="Investigate code quality using tools.",
                summary="Code quality assessment",
                kind="boolean",
            )
        ]
    )


def _make_mixed_dynamic_rubric() -> DynamicRubric:
    """Create a DynamicRubric with both LLM and agentic traits."""
    return DynamicRubric(
        llm_traits=[
            LLMRubricTrait(
                name="safety",
                summary="Response is safe for general audiences",
                kind="boolean",
            )
        ],
        agentic_traits=[
            AgenticRubricTrait(
                name="workspace_check",
                description="Check workspace files.",
                summary="Workspace file check",
                kind="boolean",
            )
        ],
    )


def _stage_names(orchestrator: StageOrchestrator) -> list[str]:
    """Extract stage names from an orchestrator instance."""
    return [s.name for s in orchestrator.stages]


# =============================================================================
# RubricEvaluationStage gating on dynamic_rubric (non-agentic traits)
# =============================================================================


@pytest.mark.unit
class TestDynamicRubricNonAgenticStageGating:
    """RubricEvaluationStage should be included when dynamic_rubric has non-agentic traits."""

    def test_llm_trait_triggers_rubric_stage_template_and_rubric(self):
        """LLM traits in dynamic_rubric include RubricEvaluationStage in template_and_rubric mode."""
        orch = StageOrchestrator.from_config(
            evaluation_mode="template_and_rubric",
            dynamic_rubric=_make_llm_dynamic_rubric(),
        )
        assert "RubricEvaluation" in _stage_names(orch)

    def test_llm_trait_triggers_rubric_stage_rubric_only(self):
        """LLM traits in dynamic_rubric include RubricEvaluationStage in rubric_only mode."""
        orch = StageOrchestrator.from_config(
            evaluation_mode="rubric_only",
            dynamic_rubric=_make_llm_dynamic_rubric(),
        )
        assert "RubricEvaluation" in _stage_names(orch)

    def test_regex_trait_triggers_rubric_stage_template_and_rubric(self):
        """Regex traits in dynamic_rubric include RubricEvaluationStage in template_and_rubric mode."""
        orch = StageOrchestrator.from_config(
            evaluation_mode="template_and_rubric",
            dynamic_rubric=_make_regex_dynamic_rubric(),
        )
        assert "RubricEvaluation" in _stage_names(orch)

    def test_regex_trait_triggers_rubric_stage_rubric_only(self):
        """Regex traits in dynamic_rubric include RubricEvaluationStage in rubric_only mode."""
        orch = StageOrchestrator.from_config(
            evaluation_mode="rubric_only",
            dynamic_rubric=_make_regex_dynamic_rubric(),
        )
        assert "RubricEvaluation" in _stage_names(orch)


# =============================================================================
# AgenticRubricEvaluationStage gating on dynamic_rubric (agentic traits)
# =============================================================================


@pytest.mark.unit
class TestDynamicRubricAgenticStageGating:
    """AgenticRubricEvaluationStage should be included when dynamic_rubric has agentic traits."""

    def test_agentic_trait_triggers_agentic_stage_template_and_rubric(self):
        """Agentic traits in dynamic_rubric include AgenticRubricEvaluationStage."""
        orch = StageOrchestrator.from_config(
            evaluation_mode="template_and_rubric",
            dynamic_rubric=_make_agentic_dynamic_rubric(),
        )
        assert "AgenticRubricEvaluation" in _stage_names(orch)

    def test_agentic_trait_triggers_agentic_stage_rubric_only(self):
        """Agentic traits in dynamic_rubric include AgenticRubricEvaluationStage in rubric_only mode."""
        orch = StageOrchestrator.from_config(
            evaluation_mode="rubric_only",
            dynamic_rubric=_make_agentic_dynamic_rubric(),
        )
        assert "AgenticRubricEvaluation" in _stage_names(orch)

    def test_mixed_dynamic_rubric_includes_both_stages(self):
        """A dynamic rubric with LLM and agentic traits includes both evaluation stages."""
        orch = StageOrchestrator.from_config(
            evaluation_mode="template_and_rubric",
            dynamic_rubric=_make_mixed_dynamic_rubric(),
        )
        names = _stage_names(orch)
        assert "RubricEvaluation" in names
        assert "AgenticRubricEvaluation" in names


# =============================================================================
# Negative cases: no rubric and no dynamic_rubric
# =============================================================================


@pytest.mark.unit
class TestNoRubricNoStages:
    """Neither rubric stage should appear when no rubric and no dynamic_rubric are provided."""

    def test_no_rubric_no_dynamic_rubric_template_and_rubric(self):
        """No rubric stages when both rubric and dynamic_rubric are None."""
        orch = StageOrchestrator.from_config(
            evaluation_mode="template_and_rubric",
        )
        names = _stage_names(orch)
        assert "RubricEvaluation" not in names
        assert "AgenticRubricEvaluation" not in names

    def test_no_rubric_no_dynamic_rubric_rubric_only(self):
        """No rubric stages when both rubric and dynamic_rubric are None in rubric_only mode."""
        orch = StageOrchestrator.from_config(
            evaluation_mode="rubric_only",
        )
        names = _stage_names(orch)
        assert "RubricEvaluation" not in names
        assert "AgenticRubricEvaluation" not in names

    def test_empty_dynamic_rubric_no_stages(self):
        """An empty DynamicRubric (no traits) does not trigger rubric stages."""
        orch = StageOrchestrator.from_config(
            evaluation_mode="template_and_rubric",
            dynamic_rubric=DynamicRubric(),
        )
        names = _stage_names(orch)
        assert "RubricEvaluation" not in names
        assert "AgenticRubricEvaluation" not in names


# =============================================================================
# template_only mode: dynamic_rubric has no effect
# =============================================================================


@pytest.mark.unit
class TestTemplateOnlyIgnoresDynamicRubric:
    """In template_only mode, rubric stages are excluded regardless of dynamic_rubric."""

    def test_template_only_excludes_rubric_stage(self):
        """RubricEvaluationStage excluded in template_only mode even with dynamic_rubric."""
        orch = StageOrchestrator.from_config(
            evaluation_mode="template_only",
            dynamic_rubric=_make_llm_dynamic_rubric(),
        )
        names = _stage_names(orch)
        assert "RubricEvaluation" not in names

    def test_template_only_excludes_agentic_stage(self):
        """AgenticRubricEvaluationStage excluded in template_only mode even with dynamic_rubric."""
        orch = StageOrchestrator.from_config(
            evaluation_mode="template_only",
            dynamic_rubric=_make_agentic_dynamic_rubric(),
        )
        names = _stage_names(orch)
        assert "AgenticRubricEvaluation" not in names

    def test_template_only_excludes_mixed_dynamic_rubric(self):
        """Both rubric stages excluded in template_only mode with mixed dynamic_rubric."""
        orch = StageOrchestrator.from_config(
            evaluation_mode="template_only",
            dynamic_rubric=_make_mixed_dynamic_rubric(),
        )
        names = _stage_names(orch)
        assert "RubricEvaluation" not in names
        assert "AgenticRubricEvaluation" not in names
