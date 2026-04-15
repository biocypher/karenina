"""Unit tests for FinalizeResultStage wiring failure + caveats.

Covers Task 7 of the failure-state harmonization plan:

1. A passing pipeline yields ``metadata.failure is None`` and no caveats.
2. A content failure (``verify_template`` returned False) populates
   ``metadata.failure`` with ``FailureCategory.CONTENT``.
3. Orthogonal caveats (e.g. partial streaming content) land on
   ``metadata.caveats`` regardless of verdict.
4. A stage-1 template validation error surfaces as
   ``FailureCategory.TEMPLATE_VALIDATION`` on the finalized metadata.
5. An orchestrator-level guarantee: when a stage (in particular
   ``ValidateTemplate``) raises an exception, ``FinalizeResultStage`` still
   runs exactly once and the raised error is reflected as a
   ``TEMPLATE_VALIDATION`` failure on the returned result.
"""

from __future__ import annotations

import pytest

from karenina.benchmark.verification.stages.core.base import (
    ArtifactKeys,
    BaseVerificationStage,
    VerificationContext,
)
from karenina.benchmark.verification.stages.core.orchestrator import StageOrchestrator
from karenina.benchmark.verification.stages.pipeline.finalize_result import (
    FinalizeResultStage,
)
from karenina.benchmark.verification.stages.pipeline.validate_template import (
    ValidateTemplateStage,
)
from karenina.schemas.config import ModelConfig
from karenina.schemas.results.caveat import Caveat
from karenina.schemas.results.failure import FailureCategory
from karenina.schemas.verification import VerificationResult
from karenina.schemas.verification.model_identity import ModelIdentity
from tests.unit.benchmark.verification.stages.core._context_factory import make_context


def _seed_identities(ctx: VerificationContext) -> None:
    """Attach minimal ModelIdentity artifacts so FinalizeResultStage can build metadata."""
    identity = ModelIdentity(model_name="test-model", interface="openai")
    ctx.set_artifact(ArtifactKeys.ANSWERING_MODEL_IDENTITY, identity)
    ctx.set_artifact(ArtifactKeys.PARSING_MODEL_IDENTITY, identity)


def _run_finalize(ctx: VerificationContext) -> VerificationResult:
    """Run FinalizeResultStage against ``ctx`` and return the produced result."""
    _seed_identities(ctx)
    FinalizeResultStage().execute(ctx)
    result = ctx.get_artifact(ArtifactKeys.FINAL_RESULT)
    assert isinstance(result, VerificationResult), (
        f"FinalizeResultStage did not produce VerificationResult: {type(result).__name__}"
    )
    return result


@pytest.mark.unit
class TestFinalizeWithFailure:
    """FinalizeResultStage wires classify_failure + collect_caveats into metadata."""

    def test_pass_yields_no_failure(self) -> None:
        """A successful pipeline produces ``failure=None`` and no caveats."""
        ctx = make_context(verify_result=True)
        result = _run_finalize(ctx)
        assert result.metadata.failure is None
        assert result.metadata.caveats == []

    def test_content_fail_populated(self) -> None:
        """verify_template returning False lands as FailureCategory.CONTENT."""
        ctx = make_context(verify_result=False)
        # Set field verification result so template_verification_performed is derived True
        ctx.set_artifact(ArtifactKeys.FIELD_VERIFICATION_RESULT, {"field": False})
        result = _run_finalize(ctx)
        assert result.metadata.failure is not None
        assert result.metadata.failure.category is FailureCategory.CONTENT
        assert result.metadata.failure.stage == "verify_template"

    def test_caveats_attached(self) -> None:
        """Partial-content flag surfaces as ``Caveat.PARTIAL_CONTENT`` on metadata."""
        ctx = make_context(verify_result=True, response_timeout_partial=True)
        result = _run_finalize(ctx)
        assert Caveat.PARTIAL_CONTENT in result.metadata.caveats

    def test_stage_one_raise_still_finalizes(self) -> None:
        """A pre-set ``template_validation_error`` artifact yields TEMPLATE_VALIDATION."""
        ctx = make_context(template_validation_error="missing BaseAnswer")
        # Simulate the orchestrator's stage-1 raise path by marking the stage.
        ctx.mark_error("template validation raised", stage="ValidateTemplate")
        result = _run_finalize(ctx)
        assert result.metadata.failure is not None
        assert result.metadata.failure.category is FailureCategory.TEMPLATE_VALIDATION


class _RaisingValidateTemplate(ValidateTemplateStage):
    """ValidateTemplate subclass that unconditionally raises in ``execute``.

    Used by the orchestrator-level test to verify that a stage-1 raise still
    reaches FinalizeResultStage and that the template-validation artifact is
    populated so the classifier can attribute the failure.
    """

    def execute(self, context: VerificationContext) -> None:  # noqa: ARG002
        raise RuntimeError("simulated validate_template failure")


def _orchestrator_context() -> VerificationContext:
    """Build a context suitable for an end-to-end orchestrator run."""
    model = ModelConfig(id="test", model_name="test-model")
    ctx = VerificationContext(
        question_id="q1",
        question_text="What?",
        raw_answer="Y",
        template_code="class Answer: pass",
        answering_model=model,
        parsing_model=model,
        template_id="tpl1",
    )
    identity = ModelIdentity(model_name="test-model", interface="openai")
    ctx.set_artifact(ArtifactKeys.ANSWERING_MODEL_IDENTITY, identity)
    ctx.set_artifact(ArtifactKeys.PARSING_MODEL_IDENTITY, identity)
    return ctx


@pytest.mark.unit
class TestOrchestratorStageOneRaiseGuarantee:
    """The orchestrator must guarantee finalize runs even when stage 1 raises."""

    def test_validate_template_raise_reaches_finalize(self) -> None:
        """A raising ValidateTemplate still yields a populated VerificationResult."""
        ctx = _orchestrator_context()
        orchestrator = StageOrchestrator(stages=[_RaisingValidateTemplate(), FinalizeResultStage()])

        result = orchestrator.execute(ctx)

        assert isinstance(result, VerificationResult)
        assert result.metadata.failure is not None
        assert result.metadata.failure.category is FailureCategory.TEMPLATE_VALIDATION

    def test_validate_template_raise_invokes_finalize_once(self) -> None:
        """FinalizeResultStage.execute is invoked exactly once on stage-1 raise."""
        call_counter = {"count": 0}

        class _CountingFinalize(FinalizeResultStage):
            def execute(self, context: VerificationContext) -> None:
                call_counter["count"] += 1
                super().execute(context)

        ctx = _orchestrator_context()
        orchestrator = StageOrchestrator(stages=[_RaisingValidateTemplate(), _CountingFinalize()])
        orchestrator.execute(ctx)
        assert call_counter["count"] == 1


class _RaisingStage(BaseVerificationStage):
    """Generic stage that raises a caller-supplied exception."""

    def __init__(self, name: str, exc: Exception) -> None:
        self._name = name
        self._exc = exc

    @property
    def name(self) -> str:
        return self._name

    def execute(self, context: VerificationContext) -> None:  # noqa: ARG002
        raise self._exc


@pytest.mark.unit
class TestOrchestratorPropagatesContextToClassifier:
    """Non-validate stage raises should fall through to ``UNEXPECTED_ERROR``."""

    def test_unknown_stage_raise_becomes_unexpected_error(self) -> None:
        ctx = _orchestrator_context()
        orchestrator = StageOrchestrator(
            stages=[_RaisingStage("MysteryStage", ValueError("boom")), FinalizeResultStage()]
        )
        result = orchestrator.execute(ctx)
        assert result.metadata.failure is not None
        assert result.metadata.failure.category is FailureCategory.UNEXPECTED_ERROR
