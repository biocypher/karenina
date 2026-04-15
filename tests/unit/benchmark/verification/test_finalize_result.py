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
6. A retry-exhausted scenario orchestrated end-to-end surfaces the correct
   ``FailureCategory`` plus ``Caveat.RETRIES_USED`` (C1 regression guard).
7. The orchestrator rejects mid-list ``FinalizeResultStage`` at construction
   time (I1 regression guard).
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
from karenina.utils.errors import ErrorCategory, ErrorRegistry
from karenina.utils.retry_policy import CategoryRetryConfig, RetryExecutor, RetryPolicy
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


class _TimeoutExhaustingStage(BaseVerificationStage):
    """Stage that drives a RetryExecutor until its timeout budget is spent.

    Uses the real retry machinery so the orchestrator's ``track_retries``
    contextvar sees each increment; this is how the ``RETRY_COUNTS``
    artifact ends up reflecting exhausted state at finalize time. Mirrors
    the production shape where adapter retries are mediated by
    ``RetryExecutor`` and bubble out as ``TimeoutError`` after exhaustion.
    """

    def __init__(self, policy: RetryPolicy) -> None:
        self._policy = policy

    @property
    def name(self) -> str:
        return "timeout_exhausting_stage"

    def execute(self, context: VerificationContext) -> None:
        executor = RetryExecutor(self._policy, ErrorRegistry())

        def _always_timeout() -> None:
            raise TimeoutError("simulated request timed out")

        try:
            executor.execute(_always_timeout)
        except TimeoutError as exc:
            context.mark_error(
                str(exc),
                category=ErrorCategory.TIMEOUT,
                stage=self.name,
            )


def _retry_policy_with_timeout_budget(budget: int) -> RetryPolicy:
    """Build a RetryPolicy whose timeout budget matches ``budget``.

    Other categories keep defaults; only the timeout slot is customised so
    the exhaustion math in classify_failure is unambiguous.
    """
    return RetryPolicy(
        timeout=CategoryRetryConfig(max_attempts=budget, backoff_min=0, backoff_max=0),
    )


@pytest.mark.unit
class TestOrchestratorRetryExhaustionEndToEnd:
    """C1 regression: retry exhaustion must classify correctly end-to-end.

    The orchestrator writes ``RETRY_COUNTS`` via ``set_artifact``; the
    classifier must read via ``get_artifact`` too. A prior implementation
    read via ``get_result_field`` and silently fell through to
    ``UNEXPECTED_ERROR``. These tests guard against that regression by
    running the orchestrator with a real retry policy and asserting the
    classified failure matches the simulated retry-exhausted state.
    """

    def test_timeout_exhaustion_classifies_as_timeout(self) -> None:
        policy = _retry_policy_with_timeout_budget(budget=2)
        model = ModelConfig(
            id="test",
            model_name="test-model",
            retry_policy=policy,
        )
        ctx = VerificationContext(
            question_id="q1",
            template_id="tpl1",
            question_text="What?",
            raw_answer="Y",
            template_code="class Answer: pass",
            answering_model=model,
            parsing_model=model,
        )
        _seed_identities(ctx)
        orchestrator = StageOrchestrator(stages=[_TimeoutExhaustingStage(policy), FinalizeResultStage()])
        result = orchestrator.execute(ctx)

        assert result.metadata.failure is not None
        assert result.metadata.failure.category is FailureCategory.TIMEOUT
        assert Caveat.RETRIES_USED in result.metadata.caveats
        assert result.metadata.retry_counts is not None
        timeout_entry = result.metadata.retry_counts["timeout"]
        assert timeout_entry["used"] == timeout_entry["budget"]

    def test_retries_used_caveat_fires_when_orchestrator_writes_artifact(self) -> None:
        """Caveat collector reads the artifact written by the orchestrator.

        Even without exhaustion, any ``used > 0`` observation should produce
        the ``RETRIES_USED`` caveat on the finalized metadata.
        """
        policy = _retry_policy_with_timeout_budget(budget=3)
        model = ModelConfig(
            id="test",
            model_name="test-model",
            retry_policy=policy,
        )
        ctx = VerificationContext(
            question_id="q1",
            template_id="tpl1",
            question_text="What?",
            raw_answer="Y",
            template_code="class Answer: pass",
            answering_model=model,
            parsing_model=model,
        )
        _seed_identities(ctx)

        class _PartialRetryStage(BaseVerificationStage):
            """Stage that retries once, then succeeds.

            Exercises the ``RETRIES_USED`` caveat without tripping rule 3
            of the classifier: one retry counted, two remaining.
            """

            def __init__(self, policy: RetryPolicy) -> None:
                self._policy = policy
                self._attempts = 0

            @property
            def name(self) -> str:
                return "partial_retry_stage"

            def execute(self, context: VerificationContext) -> None:  # noqa: ARG002
                executor = RetryExecutor(self._policy, ErrorRegistry())

                def _succeed_after_one_timeout() -> str:
                    self._attempts += 1
                    if self._attempts == 1:
                        raise TimeoutError("transient")
                    return "ok"

                executor.execute(_succeed_after_one_timeout)

        orchestrator = StageOrchestrator(stages=[_PartialRetryStage(policy), FinalizeResultStage()])
        result = orchestrator.execute(ctx)

        assert result.metadata.failure is None
        assert Caveat.RETRIES_USED in result.metadata.caveats


@pytest.mark.unit
class TestOrchestratorRejectsMidListFinalize:
    """I1 regression: the orchestrator must reject invalid finalize placement.

    A stage list like ``[A(), FinalizeResultStage(), B()]`` previously
    slipped past the LAST-only check, leaving finalize in the main loop AND
    synthesising a trailing one so finalize ran twice. The constructor now
    rejects such layouts at build time with a ValueError.
    """

    def test_mid_list_finalize_raises(self) -> None:
        class _NoopStage(BaseVerificationStage):
            @property
            def name(self) -> str:
                return "noop_stage"

            def execute(self, context: VerificationContext) -> None:  # noqa: ARG002
                return None

        class _OtherStage(BaseVerificationStage):
            @property
            def name(self) -> str:
                return "other_stage"

            def execute(self, context: VerificationContext) -> None:  # noqa: ARG002
                return None

        with pytest.raises(ValueError, match="FinalizeResultStage"):
            StageOrchestrator(stages=[_NoopStage(), FinalizeResultStage(), _OtherStage()])

    def test_multiple_finalize_stages_raises(self) -> None:
        with pytest.raises(ValueError, match="FinalizeResultStage"):
            StageOrchestrator(stages=[FinalizeResultStage(), FinalizeResultStage()])

    def test_empty_stage_list_accepted(self) -> None:
        # An empty list is unusual but legal: execute() will simply invoke
        # the synthesized finalize stage once and return. No ValueError.
        StageOrchestrator(stages=[])

    def test_trailing_finalize_accepted(self) -> None:
        class _NoopStage(BaseVerificationStage):
            @property
            def name(self) -> str:
                return "noop_stage"

            def execute(self, context: VerificationContext) -> None:  # noqa: ARG002
                return None

        StageOrchestrator(stages=[_NoopStage(), FinalizeResultStage()])

    def test_no_finalize_accepted_and_synthesized(self) -> None:
        class _NoopStage(BaseVerificationStage):
            @property
            def name(self) -> str:
                return "noop_stage"

            def execute(self, context: VerificationContext) -> None:  # noqa: ARG002
                return None

        StageOrchestrator(stages=[_NoopStage()])
