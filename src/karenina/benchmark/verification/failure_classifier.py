"""Central classifier that maps a finalized VerificationContext to a Failure or None.

The classifier encodes the single source of truth for how a completed
verification pipeline is summarised as a structured non-pass verdict. Stages
populate the context during execution (artifacts, result fields, error
attribution), and this classifier inspects the final state to decide which
failure category (if any) applies.

Rule priority is intentional: autofail stages are decisive, content failures
take precedence over transient retry exhaustion, and system-level errors fall
through to a catchall so no verdict is ever silently dropped.
"""

from __future__ import annotations

import logging

from karenina.benchmark.verification.stages.core.base import ArtifactKeys, VerificationContext
from karenina.schemas.results.caveat import Caveat
from karenina.schemas.results.failure import Failure, FailureCategory
from karenina.utils.errors import ErrorCategory

logger = logging.getLogger(__name__)

_AUTOFAIL_STAGE_TO_CATEGORY: dict[str, FailureCategory] = {
    "RecursionLimitAutoFail": FailureCategory.RECURSION_LIMIT,
    "TraceValidationAutoFail": FailureCategory.TRACE_VALIDATION,
    "DeepJudgmentAutoFail": FailureCategory.DEEP_JUDGMENT,
    "DeepJudgmentRubricAutoFail": FailureCategory.DEEP_JUDGMENT_RUBRIC,
    # PlaceholderRetryAutoFail fires when LangChain's ModelRetryMiddleware
    # emits its exhaustion placeholder ("Model call failed after ..."). The
    # underlying cause is always an infrastructure failure that exhausted the
    # in-agent retry budget, so we map it to CONNECTION (which lives in the
    # RETRY_EXHAUSTED FailureGroup).
    "PlaceholderRetryAutoFail": FailureCategory.CONNECTION,
}

_ERROR_CATEGORY_TO_FAILURE: dict[ErrorCategory, FailureCategory] = {
    ErrorCategory.TIMEOUT: FailureCategory.TIMEOUT,
    ErrorCategory.CONNECTION: FailureCategory.CONNECTION,
    ErrorCategory.RATE_LIMIT: FailureCategory.RATE_LIMIT,
    ErrorCategory.SERVER_ERROR: FailureCategory.SERVER_ERROR,
}


def _retry_exhausted(ctx: VerificationContext) -> bool:
    """Return True when the retry budget for the current error category is spent.

    The ``RETRY_COUNTS`` snapshot is written to ``ctx.artifacts`` by the
    orchestrator (inside ``track_retries``); reads here must use
    ``get_artifact`` to hit the same dict. Reading via ``get_result_field``
    silently returns ``None`` because artifacts and result_builder are
    separate dicts on VerificationContext.

    Args:
        ctx: The finalized verification context.

    Returns:
        True if ``used >= budget`` for the context's error category, else False.
    """
    rc: dict[str, dict[str, int]] = ctx.get_artifact(ArtifactKeys.RETRY_COUNTS) or {}
    cat = ctx.error_category
    if cat is None:
        return False
    entry = rc.get(cat.value)
    if not entry:
        return False
    return int(entry.get("used") or 0) >= int(entry.get("budget") or 0)


def _trim(s: str | None) -> str:
    """Return a stripped ``reason`` string capped at 500 characters."""
    return (s or "").strip()[:500]


def classify_failure(ctx: VerificationContext) -> Failure | None:
    """Classify a finalized VerificationContext into a Failure, or None if it passed.

    Rule priority:
        1. Autofail stage (decisive).
        2. Content fail (verify_template returned False) wins over retry exhaustion.
        3. Retry exhausted, only when verify_template did not complete.
        4. Abstention / sufficiency (observability flags from earlier guards).
        5. Template validation error (from stage 1).
        6. Parsing failure (parse_template raised).
        7. Unexpected error catchall.
        8. Pass.

    Args:
        ctx: The finalized verification context produced by the pipeline.

    Returns:
        A :class:`Failure` describing the non-pass verdict, or ``None`` if the
        pipeline completed successfully.
    """
    # Rule 1: autofail stage
    failed_stage = ctx.get_result_field(ArtifactKeys.FAILED_STAGE)
    if failed_stage in _AUTOFAIL_STAGE_TO_CATEGORY:
        return Failure(
            category=_AUTOFAIL_STAGE_TO_CATEGORY[failed_stage],
            stage=failed_stage,
            reason=f"auto-fail at stage {failed_stage}",
        )

    # Rule 2: content fail wins over retry exhaustion
    performed = bool(ctx.get_result_field(ArtifactKeys.TEMPLATE_VERIFICATION_PERFORMED))
    verify_result = ctx.get_result_field(ArtifactKeys.VERIFY_RESULT)
    if performed and verify_result is False:
        return Failure(
            category=FailureCategory.CONTENT,
            stage="verify_template",
            reason="verify_template returned False",
        )

    # Rule 3: retry exhausted (only when verify_template did not complete).
    # RETRY_COUNTS is stored in ``ctx.artifacts`` by the orchestrator; use
    # ``get_artifact`` so the details payload matches what ``_retry_exhausted``
    # read a moment earlier.
    if ctx.error_category in _ERROR_CATEGORY_TO_FAILURE and _retry_exhausted(ctx):
        cat = _ERROR_CATEGORY_TO_FAILURE[ctx.error_category]
        return Failure(
            category=cat,
            stage=ctx.error_stage or ctx.last_run_stage or "unknown",
            reason=f"{cat.value} retries exhausted",
            details={
                "error_message": ctx.error,
                "retry_counts": ctx.get_artifact(ArtifactKeys.RETRY_COUNTS),
            },
        )

    # Rule 4: abstention / sufficiency
    if ctx.get_result_field(ArtifactKeys.ABSTENTION_DETECTED):
        return Failure(
            category=FailureCategory.ABSTENTION,
            stage="abstention_check",
            reason=_trim(ctx.get_result_field(ArtifactKeys.ABSTENTION_REASONING)) or "model abstained",
        )
    if ctx.get_result_field(ArtifactKeys.SUFFICIENCY_DETECTED):
        return Failure(
            category=FailureCategory.SUFFICIENCY,
            stage="sufficiency_check",
            reason=_trim(ctx.get_result_field(ArtifactKeys.SUFFICIENCY_REASONING)) or "insufficient response",
        )

    # Rule 5: template validation
    tve = ctx.get_artifact(ArtifactKeys.TEMPLATE_VALIDATION_ERROR)
    if tve:
        return Failure(
            category=FailureCategory.TEMPLATE_VALIDATION,
            stage="validate_template",
            reason=_trim(str(tve)),
        )

    # Rule 6: parsing failure
    if ctx.error_stage == "parse_template" and ctx.error:
        return Failure(
            category=FailureCategory.PARSING,
            stage="parse_template",
            reason=_trim(ctx.error) or "parse failed",
            details={"error_message": ctx.error},
        )

    # Rule 7: unexpected error catchall
    if ctx.error:
        return Failure(
            category=FailureCategory.UNEXPECTED_ERROR,
            stage=ctx.error_stage or ctx.last_run_stage or "unknown",
            reason=_trim(ctx.error) or "unexpected error",
            details={"error_message": ctx.error},
        )

    # Rule 8: pass
    return None


def collect_caveats(ctx: VerificationContext) -> list[Caveat]:
    """Collect informational caveats from a finalized VerificationContext.

    Caveats are orthogonal to the pass/fail verdict: they report conditions
    worth flagging regardless of whether the run succeeded.

    Args:
        ctx: Finalized VerificationContext produced by the pipeline.

    Returns:
        A list of Caveat values in emission order. Empty when nothing fires.
    """
    caveats: list[Caveat] = []
    if ctx.get_result_field(ArtifactKeys.RESPONSE_TIMEOUT_PARTIAL):
        caveats.append(Caveat.PARTIAL_CONTENT)
    if ctx.get_result_field(ArtifactKeys.EMBEDDING_OVERRIDE_APPLIED):
        caveats.append(Caveat.EMBEDDING_OVERRIDE)
    if ctx.get_result_field(ArtifactKeys.PARSE_DECISION_MALFORMED):
        caveats.append(Caveat.PARSE_DECISION_MALFORMED)
    # RETRY_COUNTS lives in ``ctx.artifacts`` (orchestrator.set_artifact), not
    # in ``ctx.result_builder``. Reading via ``get_result_field`` returns None
    # silently and suppresses this caveat in production.
    rc: dict[str, dict[str, int]] = ctx.get_artifact(ArtifactKeys.RETRY_COUNTS) or {}
    if any(int(entry.get("used") or 0) > 0 for entry in rc.values()):
        caveats.append(Caveat.RETRIES_USED)
    return caveats


__all__ = ["classify_failure", "collect_caveats"]
