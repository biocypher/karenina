"""Tests for dynamic rubric presence check in RubricEvaluationStage.

Covers:
- Promotes present traits into rubric
- Skips absent traits with reason
- Creates rubric if context.rubric is None
- Name conflict raises ValueError
- Empty dynamic rubric: no LLM call
- rubric_trait_names filter: present but filtered = skipped
- should_run() returns True with only dynamic rubric
- should_run() returns False with empty everything
- should_run() returns False with only agentic dynamic traits
"""

from unittest.mock import patch

import pytest

from karenina.benchmark.verification.stages.core.base import (
    ArtifactKeys,
    VerificationContext,
)
from karenina.benchmark.verification.stages.pipeline.rubric_evaluation import (
    RubricEvaluationStage,
)
from karenina.schemas.config.models import ModelConfig
from karenina.schemas.entities.rubric import (
    AgenticRubricTrait,
    DynamicRubric,
    LLMRubricTrait,
    MetricRubricTrait,
    RegexRubricTrait,
    Rubric,
)

# =============================================================================
# Helpers
# =============================================================================


def _model() -> ModelConfig:
    """Return a minimal ModelConfig."""
    return ModelConfig(
        id="test",
        model_name="test-model",
        model_provider="anthropic",
    )


def _llm_trait(name: str, summary: str | None = None) -> LLMRubricTrait:
    """Create a minimal LLMRubricTrait for testing."""
    return LLMRubricTrait(
        name=name,
        kind="boolean",
        higher_is_better=True,
        summary=summary or f"Concept for {name}",
    )


def _regex_trait(name: str, summary: str | None = None) -> RegexRubricTrait:
    """Create a minimal RegexRubricTrait for testing."""
    return RegexRubricTrait(
        name=name,
        pattern=r"\btest\b",
        higher_is_better=True,
        summary=summary or f"Concept for {name}",
    )


def _metric_trait(name: str, summary: str | None = None) -> MetricRubricTrait:
    """Create a minimal MetricRubricTrait for testing."""
    return MetricRubricTrait(
        name=name,
        evaluation_mode="tp_only",
        metrics=["precision"],
        tp_instructions=["instruction"],
        summary=summary or f"Concept for {name}",
    )


def _agentic_trait(name: str, summary: str | None = None) -> AgenticRubricTrait:
    """Create a minimal AgenticRubricTrait for testing."""
    return AgenticRubricTrait(
        name=name,
        kind="boolean",
        higher_is_better=True,
        description="Agentic placeholder",
        summary=summary or f"Concept for {name}",
    )


def _make_context(
    rubric: Rubric | None = None,
    dynamic_rubric: DynamicRubric | None = None,
    rubric_trait_names: list[str] | None = None,
    raw_llm_response: str = "BCL2 is the primary target of venetoclax.",
    error: str | None = None,
) -> VerificationContext:
    """Build a VerificationContext for testing."""
    model = _model()
    ctx = VerificationContext(
        question_id="q1",
        template_id="t1",
        question_text="What is the target of venetoclax?",
        template_code="class Answer: pass",
        answering_model=model,
        parsing_model=model,
        rubric=rubric,
        dynamic_rubric=dynamic_rubric,
        rubric_trait_names=rubric_trait_names,
    )
    ctx.set_artifact(ArtifactKeys.RAW_LLM_RESPONSE, raw_llm_response)
    if error:
        ctx.mark_error(error)
    return ctx


# =============================================================================
# should_run() tests
# =============================================================================


@pytest.mark.unit
class TestShouldRunWithDynamicRubric:
    """Tests for should_run() behavior with dynamic rubric."""

    def test_returns_true_with_only_dynamic_rubric_llm_traits(self) -> None:
        """should_run returns True when dynamic rubric has LLM traits, even if static rubric is None."""
        stage = RubricEvaluationStage()
        dr = DynamicRubric(llm_traits=[_llm_trait("safety")])
        ctx = _make_context(dynamic_rubric=dr)
        assert stage.should_run(ctx) is True

    def test_returns_true_with_only_dynamic_rubric_regex_traits(self) -> None:
        """should_run returns True when dynamic rubric has regex traits."""
        stage = RubricEvaluationStage()
        dr = DynamicRubric(regex_traits=[_regex_trait("citations")])
        ctx = _make_context(dynamic_rubric=dr)
        assert stage.should_run(ctx) is True

    def test_returns_true_with_only_dynamic_rubric_metric_traits(self) -> None:
        """should_run returns True when dynamic rubric has metric traits."""
        stage = RubricEvaluationStage()
        dr = DynamicRubric(metric_traits=[_metric_trait("coverage")])
        ctx = _make_context(dynamic_rubric=dr)
        assert stage.should_run(ctx) is True

    def test_returns_false_with_empty_everything(self) -> None:
        """should_run returns False when both rubric and dynamic rubric are empty/None."""
        stage = RubricEvaluationStage()
        ctx = _make_context()
        assert stage.should_run(ctx) is False

    def test_returns_false_with_only_agentic_dynamic_traits(self) -> None:
        """should_run returns False when dynamic rubric has only agentic traits."""
        stage = RubricEvaluationStage()
        dr = DynamicRubric(agentic_traits=[_agentic_trait("code_quality")])
        ctx = _make_context(dynamic_rubric=dr)
        assert stage.should_run(ctx) is False

    def test_returns_false_when_error_set(self) -> None:
        """should_run returns False when context has an error, even with dynamic traits."""
        stage = RubricEvaluationStage()
        dr = DynamicRubric(llm_traits=[_llm_trait("safety")])
        ctx = _make_context(dynamic_rubric=dr, error="prior failure")
        assert stage.should_run(ctx) is False

    def test_returns_true_with_static_rubric_only(self) -> None:
        """should_run still works with just a static rubric (existing behavior)."""
        stage = RubricEvaluationStage()
        rubric = Rubric(llm_traits=[_llm_trait("clarity")])
        ctx = _make_context(rubric=rubric)
        assert stage.should_run(ctx) is True

    def test_returns_false_with_empty_dynamic_rubric(self) -> None:
        """should_run returns False when dynamic rubric exists but is empty."""
        stage = RubricEvaluationStage()
        dr = DynamicRubric()
        ctx = _make_context(dynamic_rubric=dr)
        assert stage.should_run(ctx) is False


# =============================================================================
# _resolve_dynamic_rubric() tests
# =============================================================================


@pytest.mark.unit
class TestResolveDynamicRubric:
    """Tests for _resolve_dynamic_rubric behavior."""

    def test_promotes_present_traits_into_rubric(self) -> None:
        """Traits flagged as present by the LLM are added to context.rubric."""
        stage = RubricEvaluationStage()
        trait_a = _llm_trait("safety")
        trait_b = _llm_trait("clarity")
        dr = DynamicRubric(llm_traits=[trait_a, trait_b])
        ctx = _make_context(dynamic_rubric=dr)

        presence_map = {"safety": True, "clarity": True}
        with patch.object(stage, "_call_presence_check", return_value=presence_map):
            stage._resolve_dynamic_rubric(ctx)

        assert ctx.rubric is not None
        names = ctx.rubric.get_trait_names()
        assert "safety" in names
        assert "clarity" in names
        assert ctx.get_artifact(ArtifactKeys.DYNAMIC_RUBRIC_PROMOTED_TRAITS) == ["safety", "clarity"]

    def test_skips_absent_traits_with_reason(self) -> None:
        """Traits flagged as absent are recorded in skipped artifacts."""
        stage = RubricEvaluationStage()
        trait_a = _llm_trait("safety")
        trait_b = _llm_trait("clarity")
        dr = DynamicRubric(llm_traits=[trait_a, trait_b])
        ctx = _make_context(dynamic_rubric=dr)

        presence_map = {"safety": True, "clarity": False}
        with patch.object(stage, "_call_presence_check", return_value=presence_map):
            stage._resolve_dynamic_rubric(ctx)

        assert ctx.rubric is not None
        assert "safety" in ctx.rubric.get_trait_names()
        assert "clarity" not in ctx.rubric.get_trait_names()

        skipped = ctx.get_artifact(ArtifactKeys.DYNAMIC_RUBRIC_SKIPPED_TRAITS)
        assert skipped is not None
        assert "clarity" in skipped
        assert skipped["clarity"] == "concept not present in response"

    def test_creates_rubric_if_context_rubric_is_none(self) -> None:
        """If context.rubric is None and traits are promoted, a new Rubric is created."""
        stage = RubricEvaluationStage()
        trait = _llm_trait("safety")
        dr = DynamicRubric(llm_traits=[trait])
        ctx = _make_context(dynamic_rubric=dr)
        assert ctx.rubric is None

        presence_map = {"safety": True}
        with patch.object(stage, "_call_presence_check", return_value=presence_map):
            stage._resolve_dynamic_rubric(ctx)

        assert ctx.rubric is not None
        assert "safety" in ctx.rubric.get_trait_names()

    def test_name_conflict_raises_value_error(self) -> None:
        """Dynamic trait with same name as existing static trait raises ValueError."""
        stage = RubricEvaluationStage()
        static_rubric = Rubric(llm_traits=[_llm_trait("safety")])
        dr = DynamicRubric(llm_traits=[_llm_trait("safety")])
        ctx = _make_context(rubric=static_rubric, dynamic_rubric=dr)

        with pytest.raises(ValueError, match="safety"):
            stage._resolve_dynamic_rubric(ctx)

    def test_empty_dynamic_rubric_no_llm_call(self) -> None:
        """Empty dynamic rubric returns early without calling LLM."""
        stage = RubricEvaluationStage()
        dr = DynamicRubric()
        ctx = _make_context(dynamic_rubric=dr)

        with patch.object(stage, "_call_presence_check") as mock_call:
            stage._resolve_dynamic_rubric(ctx)
            mock_call.assert_not_called()

    def test_none_dynamic_rubric_no_llm_call(self) -> None:
        """None dynamic rubric returns early without calling LLM."""
        stage = RubricEvaluationStage()
        ctx = _make_context()

        with patch.object(stage, "_call_presence_check") as mock_call:
            stage._resolve_dynamic_rubric(ctx)
            mock_call.assert_not_called()

    def test_rubric_trait_names_filter_skips_present_trait(self) -> None:
        """A trait present in response but excluded by rubric_trait_names is skipped."""
        stage = RubricEvaluationStage()
        trait_a = _llm_trait("safety")
        trait_b = _llm_trait("clarity")
        dr = DynamicRubric(llm_traits=[trait_a, trait_b])
        ctx = _make_context(dynamic_rubric=dr, rubric_trait_names=["safety"])

        presence_map = {"safety": True, "clarity": True}
        with patch.object(stage, "_call_presence_check", return_value=presence_map):
            stage._resolve_dynamic_rubric(ctx)

        assert ctx.rubric is not None
        assert "safety" in ctx.rubric.get_trait_names()
        assert "clarity" not in ctx.rubric.get_trait_names()

        skipped = ctx.get_artifact(ArtifactKeys.DYNAMIC_RUBRIC_SKIPPED_TRAITS)
        assert skipped is not None
        assert skipped["clarity"] == "excluded by rubric_trait_names filter"

    def test_promotes_regex_traits(self) -> None:
        """Regex traits are promoted into the correct list on the rubric."""
        stage = RubricEvaluationStage()
        trait = _regex_trait("has_citations")
        dr = DynamicRubric(regex_traits=[trait])
        ctx = _make_context(dynamic_rubric=dr)

        presence_map = {"has_citations": True}
        with patch.object(stage, "_call_presence_check", return_value=presence_map):
            stage._resolve_dynamic_rubric(ctx)

        assert ctx.rubric is not None
        assert len(ctx.rubric.regex_traits) == 1
        assert ctx.rubric.regex_traits[0].name == "has_citations"

    def test_promotes_metric_traits(self) -> None:
        """Metric traits are promoted into the correct list on the rubric."""
        stage = RubricEvaluationStage()
        trait = _metric_trait("coverage")
        dr = DynamicRubric(metric_traits=[trait])
        ctx = _make_context(dynamic_rubric=dr)

        presence_map = {"coverage": True}
        with patch.object(stage, "_call_presence_check", return_value=presence_map):
            stage._resolve_dynamic_rubric(ctx)

        assert ctx.rubric is not None
        assert len(ctx.rubric.metric_traits) == 1
        assert ctx.rubric.metric_traits[0].name == "coverage"

    def test_merges_with_existing_static_rubric(self) -> None:
        """Promoted traits are appended to an existing static rubric."""
        stage = RubricEvaluationStage()
        static_rubric = Rubric(llm_traits=[_llm_trait("accuracy")])
        trait = _llm_trait("safety")
        dr = DynamicRubric(llm_traits=[trait])
        ctx = _make_context(rubric=static_rubric, dynamic_rubric=dr)

        presence_map = {"safety": True}
        with patch.object(stage, "_call_presence_check", return_value=presence_map):
            stage._resolve_dynamic_rubric(ctx)

        assert ctx.rubric is not None
        names = ctx.rubric.get_trait_names()
        assert "accuracy" in names
        assert "safety" in names

    def test_all_traits_absent_produces_none_promoted(self) -> None:
        """When all traits are absent, promoted list is None and skipped has entries."""
        stage = RubricEvaluationStage()
        dr = DynamicRubric(llm_traits=[_llm_trait("safety"), _llm_trait("clarity")])
        ctx = _make_context(dynamic_rubric=dr)

        presence_map = {"safety": False, "clarity": False}
        with patch.object(stage, "_call_presence_check", return_value=presence_map):
            stage._resolve_dynamic_rubric(ctx)

        assert ctx.get_artifact(ArtifactKeys.DYNAMIC_RUBRIC_PROMOTED_TRAITS) is None
        skipped = ctx.get_artifact(ArtifactKeys.DYNAMIC_RUBRIC_SKIPPED_TRAITS)
        assert len(skipped) == 2

    def test_only_agentic_traits_no_llm_call(self) -> None:
        """Dynamic rubric with only agentic traits does not trigger presence check."""
        stage = RubricEvaluationStage()
        dr = DynamicRubric(agentic_traits=[_agentic_trait("code_quality")])
        ctx = _make_context(dynamic_rubric=dr)

        with patch.object(stage, "_call_presence_check") as mock_call:
            stage._resolve_dynamic_rubric(ctx)
            mock_call.assert_not_called()

    def test_mixed_types_promoted_correctly(self) -> None:
        """LLM and regex traits are both promoted to correct lists."""
        stage = RubricEvaluationStage()
        llm = _llm_trait("safety")
        regex = _regex_trait("format")
        dr = DynamicRubric(llm_traits=[llm], regex_traits=[regex])
        ctx = _make_context(dynamic_rubric=dr)

        presence_map = {"safety": True, "format": True}
        with patch.object(stage, "_call_presence_check", return_value=presence_map):
            stage._resolve_dynamic_rubric(ctx)

        assert ctx.rubric is not None
        assert len(ctx.rubric.llm_traits) == 1
        assert len(ctx.rubric.regex_traits) == 1
        assert ctx.rubric.llm_traits[0].name == "safety"
        assert ctx.rubric.regex_traits[0].name == "format"
