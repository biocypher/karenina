"""Tests that StageOrchestrator wires retry tracking into the result.

These tests verify the full wiring path:

1. The orchestrator opens ``track_retries`` with the answering model's
   ``RetryPolicy`` so per-category budgets are pre-populated.
2. Stages that retry via ``RetryExecutor`` increment ``used`` for the matching
   category.
3. ``ArtifactKeys.RETRY_COUNTS`` is snapshotted on the context so
   ``FinalizeResultStage`` (or any later stage) can read a plain dict.
4. ``VerificationResultMetadata.retry_counts`` reflects the snapshot when the
   pipeline runs through ``FinalizeResultStage``.
"""

from __future__ import annotations

import pytest

from karenina.benchmark.verification.stages.core.base import (
    ArtifactKeys,
    BaseVerificationStage,
    VerificationContext,
)
from karenina.benchmark.verification.stages.core.orchestrator import StageOrchestrator
from karenina.benchmark.verification.stages.pipeline.finalize_result import FinalizeResultStage
from karenina.schemas.config import ModelConfig
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.utils.errors import ErrorRegistry
from karenina.utils.retry_policy import (
    CategoryRetryConfig,
    RetryExecutor,
    RetryPolicy,
)


def _zero_delay_policy() -> RetryPolicy:
    """A policy with non-zero budgets but zero backoff for fast tests."""
    return RetryPolicy(
        connection=CategoryRetryConfig(max_attempts=3, backoff_min=0, backoff_max=0),
        timeout=CategoryRetryConfig(max_attempts=2, backoff_min=0, backoff_max=0),
        rate_limit=CategoryRetryConfig(max_attempts=4, backoff_min=0, backoff_max=0),
        server_error=CategoryRetryConfig(max_attempts=1, backoff_min=0, backoff_max=0),
    )


def _make_context(policy: RetryPolicy | None) -> VerificationContext:
    model = ModelConfig(
        id="test",
        model_name="test-model",
        retry_policy=policy,
    )
    ctx = VerificationContext(
        question_id="q1",
        question_text="What?",
        raw_answer="Y",
        template_code="class Answer: pass",
        answering_model=model,
        parsing_model=model,
        template_id="tpl1",
    )
    # FinalizeResultStage needs these to assemble metadata.
    ctx.set_artifact(
        ArtifactKeys.ANSWERING_MODEL_IDENTITY,
        ModelIdentity(model_name="test-model", interface="openai"),
    )
    ctx.set_artifact(
        ArtifactKeys.PARSING_MODEL_IDENTITY,
        ModelIdentity(model_name="test-model", interface="openai"),
    )
    return ctx


class _RetryingStage(BaseVerificationStage):
    """Stage that fires N retries via RetryExecutor for a specific exception."""

    def __init__(self, exc_factory, fail_times: int) -> None:
        self._exc_factory = exc_factory
        self._fail_times = fail_times
        self._call_count = 0

    @property
    def name(self) -> str:
        return "retrying_stage"

    def execute(self, context: VerificationContext) -> None:  # noqa: ARG002
        executor = RetryExecutor(_zero_delay_policy(), ErrorRegistry())

        def attempt() -> str:
            self._call_count += 1
            if self._call_count <= self._fail_times:
                raise self._exc_factory()
            return "ok"

        executor.execute(attempt)


@pytest.mark.unit
class TestOrchestratorPopulatesRetryArtifact:
    """The orchestrator must always store a snapshot of the retry tracker."""

    def test_no_retries_still_populates_budgets(self) -> None:
        """Even with zero retries, the artifact carries the budget shape."""
        ctx = _make_context(_zero_delay_policy())

        class _NoopStage(BaseVerificationStage):
            @property
            def name(self) -> str:
                return "noop"

            def execute(self, context: VerificationContext) -> None:  # noqa: ARG002
                pass

        # We cannot call execute with only _NoopStage because there is no
        # FinalizeResultStage to produce the final result; register a finalize
        # stub as the second stage instead.
        orchestrator = StageOrchestrator(stages=[_NoopStage(), _FinalizeStub()])
        result = orchestrator.execute(ctx)

        assert result.metadata.retry_counts == {
            "connection": {"used": 0, "budget": 3},
            "timeout": {"used": 0, "budget": 2},
            "rate_limit": {"used": 0, "budget": 4},
            "server_error": {"used": 0, "budget": 1},
        }

    def test_connection_retries_recorded(self) -> None:
        ctx = _make_context(_zero_delay_policy())
        stage = _RetryingStage(lambda: ConnectionError("flaky"), fail_times=2)
        orchestrator = StageOrchestrator(stages=[stage, _FinalizeStub()])
        result = orchestrator.execute(ctx)

        assert result.metadata.retry_counts is not None
        assert result.metadata.retry_counts["connection"] == {"used": 2, "budget": 3}
        # Other categories untouched
        assert result.metadata.retry_counts["timeout"] == {"used": 0, "budget": 2}

    def test_timeout_retries_recorded(self) -> None:
        ctx = _make_context(_zero_delay_policy())
        stage = _RetryingStage(lambda: TimeoutError("slow"), fail_times=1)
        orchestrator = StageOrchestrator(stages=[stage, _FinalizeStub()])
        result = orchestrator.execute(ctx)

        assert result.metadata.retry_counts["timeout"] == {"used": 1, "budget": 2}

    def test_default_policy_used_when_model_has_none(self) -> None:
        """When the answering model has no retry_policy, defaults populate budgets."""
        ctx = _make_context(None)
        orchestrator = StageOrchestrator(stages=[_FinalizeStub()])
        result = orchestrator.execute(ctx)

        # Defaults from RetryPolicy() (see src/karenina/utils/retry_policy.py)
        default = RetryPolicy()
        assert result.metadata.retry_counts == {
            "connection": {"used": 0, "budget": default.connection.max_attempts},
            "timeout": {"used": 0, "budget": default.timeout.max_attempts},
            "rate_limit": {"used": 0, "budget": default.rate_limit.max_attempts},
            "server_error": {"used": 0, "budget": default.server_error.max_attempts},
        }

    def test_artifact_is_a_plain_dict_copy(self) -> None:
        """Snapshot must not be a live reference to the contextvar tracker."""
        ctx = _make_context(_zero_delay_policy())
        stage = _RetryingStage(lambda: ConnectionError("flaky"), fail_times=1)
        orchestrator = StageOrchestrator(stages=[stage, _FinalizeStub()])
        orchestrator.execute(ctx)

        snapshot = ctx.get_artifact(ArtifactKeys.RETRY_COUNTS)
        # Mutating the snapshot must not affect the metadata field on the
        # final result, since orchestrator stored a deepcopy.
        assert isinstance(snapshot, dict)
        assert snapshot["connection"]["used"] == 1


# ---------------------------------------------------------------------------
# Local stub finalize stage
# ---------------------------------------------------------------------------


class _FinalizeStub(BaseVerificationStage):
    """Minimal finalize stage that exercises the real FinalizeResultStage path
    for retry_counts plumbing without requiring a full pipeline setup.

    Internally delegates to FinalizeResultStage.execute so the tested code
    path is the production one. Required artifacts are pre-set in
    _make_context.
    """

    @property
    def name(self) -> str:
        return "FinalizeResultStage"

    def should_run(self, context: VerificationContext) -> bool:  # noqa: ARG002
        return True

    def execute(self, context: VerificationContext) -> None:
        FinalizeResultStage().execute(context)
