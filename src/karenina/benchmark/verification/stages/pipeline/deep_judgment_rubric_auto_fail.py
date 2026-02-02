"""Deep judgment rubric auto-fail stage.

Auto-fails verification if deep-judgment-enabled traits failed to extract valid excerpts
after all retry attempts.
"""

import logging

from ..core.autofail_stage_base import BaseAutoFailStage
from ..core.base import ArtifactKeys, VerificationContext

logger = logging.getLogger(__name__)


class DeepJudgmentRubricAutoFailStage(BaseAutoFailStage):
    """
    Auto-fails verification if deep-judgment rubric traits lack valid excerpts.

    This stage:
    1. Checks if any deep-judgment-enabled traits failed excerpt extraction
    2. Auto-fails verification if traits_without_valid_excerpts is non-empty
    3. Skips auto-fail if abstention was already detected (abstention takes priority)

    Requires:
        - Deep judgment rubric evaluation must have run
        - traits_without_valid_excerpts field must be set

    Produces:
        - Sets verify_result to False if auto-fail triggered

    Error Handling:
        Non-fatal: If required fields are missing, stage logs warning and skips.

    Pipeline Position:
        After RubricEvaluationStage, before FinalizeResultStage
    """

    @property
    def name(self) -> str:
        """Stage name."""
        return "DeepJudgmentRubricAutoFail"

    @property
    def requires(self) -> list[str]:
        """Artifacts required by this stage."""
        return [ArtifactKeys.RAW_LLM_RESPONSE]  # Minimal requirement

    def should_run(self, context: VerificationContext) -> bool:
        """
        Run only if deep judgment rubric evaluation was performed.

        Inherits error-checking from BaseVerificationStage.
        """
        if not super().should_run(context):
            return False

        # Check if deep judgment rubric was performed
        deep_judgment_performed = context.get_result_field(ArtifactKeys.DEEP_JUDGMENT_RUBRIC_PERFORMED)
        if not deep_judgment_performed:
            return False

        # Check if there are traits without excerpts
        traits_without_excerpts = context.get_result_field(ArtifactKeys.TRAITS_WITHOUT_VALID_EXCERPTS)
        return bool(traits_without_excerpts)

    def _should_skip_due_to_prior_failure(self, context: VerificationContext) -> bool:
        """
        Skip if abstention was detected.

        Abstention takes priority over deep judgment rubric auto-fail.
        """
        abstention_detected = context.get_result_field(ArtifactKeys.ABSTENTION_DETECTED)
        if abstention_detected:
            traits_without_excerpts = context.get_result_field(ArtifactKeys.TRAITS_WITHOUT_VALID_EXCERPTS)
            logger.debug(
                f"Abstention detected - skipping deep judgment rubric auto-fail "
                f"for {len(traits_without_excerpts)} trait(s): {', '.join(traits_without_excerpts)}"
            )
            return True
        return False

    def _get_autofail_reason(self, context: VerificationContext) -> str:
        """Get the auto-fail reason message."""
        traits_without_excerpts = context.get_result_field(ArtifactKeys.TRAITS_WITHOUT_VALID_EXCERPTS)
        trait_names = ", ".join(traits_without_excerpts)
        return (
            f"{len(traits_without_excerpts)} trait(s) failed to extract valid excerpts "
            f"after retry attempts: {trait_names}"
        )

    def _set_additional_failure_fields(self, context: VerificationContext) -> None:
        """
        Log retry metadata for transparency.

        Note: failure_reason is captured in the auto-fail log message via _get_autofail_reason(),
        not stored as a separate result field since it's not read by finalize_result.
        """
        traits_without_excerpts = context.get_result_field(ArtifactKeys.TRAITS_WITHOUT_VALID_EXCERPTS)

        # Log retry counts for transparency
        trait_metadata = context.get_result_field(ArtifactKeys.TRAIT_METADATA) or {}
        for trait_name in traits_without_excerpts:
            metadata = trait_metadata.get(trait_name, {})
            retry_count = metadata.get("excerpt_retry_count", 0)
            logger.debug(f"  - Trait '{trait_name}' failed after {retry_count} retry attempt(s)")
