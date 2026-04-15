"""Shared factory for VerificationContext fixtures across verification tests.

Provides a dependency-free ``make_context`` helper that builds a minimal
``VerificationContext`` suitable for stage-level unit tests. Callers may
override any field via keyword arguments.

Artifact and result-field keys are routed to the appropriate internal dict
after construction so tests can express context state declaratively without
caring which storage slot a given key lives in.
"""

from __future__ import annotations

from typing import Any

from karenina.benchmark.verification.stages.core.base import ArtifactKeys, VerificationContext
from karenina.schemas.config import ModelConfig

# Kwargs routed to ``context.set_result_field`` after construction.
_RESULT_FIELD_KWARGS: dict[str, str] = {
    "verify_result": ArtifactKeys.VERIFY_RESULT,
    "failed_stage": ArtifactKeys.FAILED_STAGE,
    "template_verification_performed": ArtifactKeys.TEMPLATE_VERIFICATION_PERFORMED,
    "retry_counts": ArtifactKeys.RETRY_COUNTS,
    "response_timeout_partial": ArtifactKeys.RESPONSE_TIMEOUT_PARTIAL,
    "recursion_limit_reached": ArtifactKeys.RECURSION_LIMIT_REACHED,
    "embedding_override_applied": ArtifactKeys.EMBEDDING_OVERRIDE_APPLIED,
    "abstention_detected": ArtifactKeys.ABSTENTION_DETECTED,
    "abstention_reasoning": ArtifactKeys.ABSTENTION_REASONING,
    "sufficiency_detected": ArtifactKeys.SUFFICIENCY_DETECTED,
    "sufficiency_reasoning": ArtifactKeys.SUFFICIENCY_REASONING,
}

# Kwargs routed to ``context.set_artifact`` after construction.
_ARTIFACT_KWARGS: dict[str, str] = {
    "template_validation_error": ArtifactKeys.TEMPLATE_VALIDATION_ERROR,
}


def make_context(**overrides: Any) -> VerificationContext:
    """Build a minimal ``VerificationContext`` for unit tests.

    The default configuration uses a placeholder ``ModelConfig`` for both the
    answering and parsing models; callers should override those when a stage
    under test touches model identity.

    Kwargs listed in ``_RESULT_FIELD_KWARGS`` or ``_ARTIFACT_KWARGS`` are
    routed to ``set_result_field`` / ``set_artifact`` after the context is
    constructed. All other kwargs are forwarded to ``VerificationContext`` as
    constructor arguments.

    Args:
        **overrides: Keyword arguments; may be ``VerificationContext`` fields
            or recognised artifact/result-field aliases.

    Returns:
        A ``VerificationContext`` instance ready to be passed to stages.
    """
    model = ModelConfig(id="test", model_name="test-model")
    base: dict[str, Any] = {
        "question_id": "q1",
        "template_id": "tpl1",
        "question_text": "What?",
        "template_code": "class Answer: pass",
        "answering_model": model,
        "parsing_model": model,
        "raw_answer": "Y",
    }
    result_fields: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    for key, value in overrides.items():
        if key in _RESULT_FIELD_KWARGS:
            result_fields[_RESULT_FIELD_KWARGS[key]] = value
        elif key in _ARTIFACT_KWARGS:
            artifacts[_ARTIFACT_KWARGS[key]] = value
        else:
            base[key] = value

    ctx = VerificationContext(**base)
    for key, value in result_fields.items():
        ctx.set_result_field(key, value)
    for key, value in artifacts.items():
        ctx.set_artifact(key, value)
    return ctx


__all__ = ["make_context"]
