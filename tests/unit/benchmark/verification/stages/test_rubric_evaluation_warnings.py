"""Tests for rubric evaluation stage non-fatal failure warnings."""

from __future__ import annotations

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
from karenina.schemas.entities.rubric import LLMRubricTrait, Rubric


def _make_context_with_rubric() -> VerificationContext:
    """Create a VerificationContext with a rubric containing one LLM trait."""
    model = ModelConfig(id="test", model_name="test-model")
    rubric = Rubric(
        llm_traits=[
            LLMRubricTrait(
                name="clarity",
                description="Is the response clear?",
                kind="boolean",
                higher_is_better=True,
            ),
        ],
    )
    ctx = VerificationContext(
        question_id="q1",
        question_text="What is Python?",
        raw_answer="A programming language.",
        template_code="class Answer: pass",
        answering_model=model,
        parsing_model=model,
        template_id="tpl1",
        rubric=rubric,
    )
    ctx.set_artifact(ArtifactKeys.RAW_LLM_RESPONSE, "A programming language.")
    return ctx


@pytest.mark.unit
class TestRubricEvaluationWarnings:
    """Verify that non-fatal rubric evaluation failures append context warnings."""

    def test_value_error_appends_warning(self) -> None:
        """A ValueError during evaluation adds a warning and sets rubric_result to None."""
        stage = RubricEvaluationStage()
        ctx = _make_context_with_rubric()

        with patch(
            "karenina.benchmark.verification.stages.pipeline.rubric_evaluation.RubricEvaluator",
            side_effect=ValueError("bad config"),
        ):
            stage.execute(ctx)

        assert ctx.error is None, "Non-fatal error should not set context.error"
        assert any("Rubric evaluation failed" in w for w in ctx.warnings)
        assert any("bad config" in w for w in ctx.warnings)
        assert ctx.get_artifact(ArtifactKeys.RUBRIC_RESULT) is None

    def test_runtime_error_appends_warning(self) -> None:
        """A RuntimeError during evaluation adds a warning and sets rubric_result to None."""
        stage = RubricEvaluationStage()
        ctx = _make_context_with_rubric()

        with patch(
            "karenina.benchmark.verification.stages.pipeline.rubric_evaluation.RubricEvaluator",
            side_effect=RuntimeError("initialization failed"),
        ):
            stage.execute(ctx)

        assert ctx.error is None
        assert any("Rubric evaluation failed" in w for w in ctx.warnings)
        assert any("initialization failed" in w for w in ctx.warnings)
        assert ctx.get_artifact(ArtifactKeys.RUBRIC_RESULT) is None

    def test_generic_exception_appends_warning(self) -> None:
        """A generic Exception during evaluation adds a warning and sets rubric_result to None."""
        stage = RubricEvaluationStage()
        ctx = _make_context_with_rubric()

        with patch(
            "karenina.benchmark.verification.stages.pipeline.rubric_evaluation.RubricEvaluator",
            side_effect=TypeError("unexpected type"),
        ):
            stage.execute(ctx)

        assert ctx.error is None
        assert any("Rubric evaluation failed" in w for w in ctx.warnings)
        assert any("unexpected type" in w for w in ctx.warnings)
        assert ctx.get_artifact(ArtifactKeys.RUBRIC_RESULT) is None

    def test_no_warning_on_success(self) -> None:
        """Successful rubric evaluation does not add any warnings."""
        stage = RubricEvaluationStage()
        ctx = _make_context_with_rubric()

        mock_evaluator_instance = type(
            "MockEval",
            (),
            {
                "evaluate_rubric": lambda _self, **_kw: ({"clarity": 1.0}, {"clarity": "yes"}, []),
            },
        )()
        with patch(
            "karenina.benchmark.verification.stages.pipeline.rubric_evaluation.RubricEvaluator",
            return_value=mock_evaluator_instance,
        ):
            stage.execute(ctx)

        assert ctx.warnings == []
        assert ctx.get_artifact(ArtifactKeys.RUBRIC_RESULT) == {"clarity": 1.0}
