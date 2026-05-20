"""Placeholder model-retry auto-fail stage.

LangChain's ``ModelRetryMiddleware``, when configured with
``on_failure="continue"``, swallows exceptions on retry exhaustion and emits
a placeholder ``AIMessage`` whose content begins with
``"Model call failed after "``. Without an early guard the placeholder reaches
``parse_template``, the judge dutifully extracts default-False fields, and the
row is recorded as a content failure when the root cause is an infrastructure
connection failure. See daily note
``2026-04-25-mcp-connection-error-misclassification.md``.

With the karenina adapter mapping ``ModelRetryConfig.on_failure="raise"`` to
LangChain's ``"error"``, the underlying exception propagates and this stage
never fires. This stage exists as a structural guarantee against future
regressions: leftover ``"continue"`` callers, alternative middleware paths, or
upstream changes to the placeholder format will still be caught here.
"""

import logging
from typing import Any

from karenina.utils.errors import ErrorCategory

from ..core.base import ArtifactKeys, BaseVerificationStage, VerificationContext

logger = logging.getLogger(__name__)

PLACEHOLDER_PREFIX = "Model call failed after "


class PlaceholderRetryAutoFailStage(BaseVerificationStage):
    """Auto-fails when the agent trace ends with a model-retry placeholder.

    The placeholder fingerprint is: the LAST assistant message in the trace
    has text content starting with ``"Model call failed after "``. This catches
    both:
    - The trivial case (trace = [single placeholder]) when the very first
      model call exhausts retries.
    - The mid-trace case (trace = [tool_call, tool_result, ..., placeholder])
      when the model successfully runs tools but the final synthesis call
      hits a connection error.

    Either way the model never produced a real final answer; parsing the
    placeholder against ground truth produces meaningless content failures.

    On match, the stage marks the context as a connection-category error
    (so downstream stages skip via the default ``should_run`` check) and
    sets ``FAILED_STAGE`` so ``classify_failure`` resolves the result to
    ``Failure(category=CONNECTION, ...)`` via the autofail rule.
    """

    @property
    def name(self) -> str:
        return "PlaceholderRetryAutoFail"

    @property
    def requires(self) -> list[str]:
        return [ArtifactKeys.TRACE_MESSAGES]

    def should_run(self, context: VerificationContext) -> bool:
        if not super().should_run(context):
            return False
        return context.has_artifact(ArtifactKeys.TRACE_MESSAGES)

    def execute(self, context: VerificationContext) -> None:
        msgs = context.get_artifact(ArtifactKeys.TRACE_MESSAGES) or []
        if not msgs:
            return

        last = msgs[-1]
        if not _is_assistant(last):
            return

        text = _extract_text(last)
        if not text.startswith(PLACEHOLDER_PREFIX):
            return

        truncated = text[:500]
        logger.warning(
            "PlaceholderRetryAutoFail for question %s: detected ModelRetryMiddleware "
            "exhaustion placeholder at end of %d-message trace; reclassifying as "
            "connection failure. Placeholder: %s",
            context.question_id,
            len(msgs),
            truncated,
        )

        context.set_artifact(ArtifactKeys.VERIFY_RESULT, False)
        context.set_artifact(ArtifactKeys.FIELD_VERIFICATION_RESULT, False)
        context.set_result_field(ArtifactKeys.VERIFY_RESULT, False)

        if not context.get_result_field(ArtifactKeys.FAILED_STAGE):
            context.set_result_field(ArtifactKeys.FAILED_STAGE, self.name)

        context.mark_error(
            f"Model retry exhausted in agent loop: {truncated}",
            category=ErrorCategory.CONNECTION,
            stage="generate_answer",
        )


def _is_assistant(message: Any) -> bool:
    role = getattr(message, "role", None)
    if role is not None:
        role_value = role.value if hasattr(role, "value") else role
        return bool(role_value == "assistant")
    if isinstance(message, dict):
        return message.get("role") == "assistant"
    return False


def _extract_text(message: Any) -> str:
    """Extract textual content from a Message (port type) or trace dict."""
    if hasattr(message, "text"):
        text = message.text
        return text if isinstance(text, str) else ""
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(block.get("text", "") for block in content if isinstance(block, dict))
    return ""
