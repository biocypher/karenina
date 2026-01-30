"""Deep-judgment auto-fail stage.

Auto-fails verification when excerpts are missing for attributes.
"""

import logging

from ..core.autofail_stage_base import BaseAutoFailStage
from ..core.base import ArtifactKeys, VerificationContext

logger = logging.getLogger(__name__)


class DeepJudgmentAutoFailStage(BaseAutoFailStage):
    """
    Auto-fails verification for missing deep-judgment excerpts.

    This stage:
    1. Only runs if deep-judgment was performed
    2. Checks if any attributes are missing corroborating excerpts
    3. If missing excerpts found AND abstention NOT detected, auto-fails
    4. Does NOT auto-fail if abstention was detected (already handled)

    Requires:
        - "deep_judgment_performed": Whether deep-judgment was used (bool)
        - "attributes_without_excerpts": List of attributes missing excerpts
        - "verify_result": Current verification result
        - "field_verification_result": Field verification result

    Optional (used if available):
        - "abstention_detected": Whether abstention was detected (bool or None)
        - "abstention_check_performed": Whether abstention check ran (bool)

    Produces:
        - None (only modifies existing verification results)

    Side Effects:
        - May set verify_result and field_verification_result to False
        - Logs auto-fail reason

    Note:
        This stage runs after abstention check (if enabled). If abstention was detected,
        the verification is already failed and we don't need to auto-fail again.
        The auto-fail is specifically for cases where the LLM generated an answer
        but couldn't provide excerpts to support it. When abstention detection is disabled,
        the stage proceeds with auto-fail logic without checking abstention artifacts.
    """

    @property
    def name(self) -> str:
        """Stage name."""
        return "DeepJudgmentAutoFail"

    @property
    def requires(self) -> list[str]:
        """Artifacts required by this stage."""
        return [
            ArtifactKeys.DEEP_JUDGMENT_PERFORMED,
            ArtifactKeys.ATTRIBUTES_WITHOUT_EXCERPTS,
            ArtifactKeys.VERIFY_RESULT,
            ArtifactKeys.FIELD_VERIFICATION_RESULT,
        ]

    def should_run(self, context: VerificationContext) -> bool:
        """
        Run only if deep-judgment was performed.

        Also requires that we have attributes_without_excerpts data,
        which is set by deep-judgment parsing.
        Inherits error-checking from BaseVerificationStage.
        """
        if not super().should_run(context):
            return False

        deep_judgment_performed = context.get_artifact(ArtifactKeys.DEEP_JUDGMENT_PERFORMED, False)
        attributes_without_excerpts = context.get_artifact(ArtifactKeys.ATTRIBUTES_WITHOUT_EXCERPTS)

        return (
            deep_judgment_performed and attributes_without_excerpts is not None and len(attributes_without_excerpts) > 0
        )

    def _should_skip_due_to_prior_failure(self, context: VerificationContext) -> bool:
        """
        Skip if abstention was detected.

        If abstention was detected, verification is already failed for a better reason.
        """
        abstention_detected = context.get_artifact(ArtifactKeys.ABSTENTION_DETECTED)
        abstention_check_performed = context.get_artifact(ArtifactKeys.ABSTENTION_CHECK_PERFORMED)
        return bool(abstention_detected and abstention_check_performed)

    def _get_autofail_reason(self, context: VerificationContext) -> str:
        """Get the auto-fail reason message."""
        attributes_without_excerpts = context.get_artifact(ArtifactKeys.ATTRIBUTES_WITHOUT_EXCERPTS)
        return (
            f"{len(attributes_without_excerpts)} attributes without excerpts: {', '.join(attributes_without_excerpts)}"
        )

    def _get_log_level(self) -> int:
        """Use WARNING level for auto-fail (matches logging convention)."""
        return logging.WARNING
