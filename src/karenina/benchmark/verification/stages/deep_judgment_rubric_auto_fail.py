"""Deep judgment rubric auto-fail stage.

Auto-fails verification if deep-judgment-enabled traits failed to extract valid excerpts
after all retry attempts.
"""

import logging

from ..stage import BaseVerificationStage, VerificationContext

logger = logging.getLogger(__name__)


class DeepJudgmentRubricAutoFailStage(BaseVerificationStage):
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
        - Sets failure_reason explaining which traits failed

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
        return ["raw_llm_response"]  # Minimal requirement

    @property
    def produces(self) -> list[str]:
        """Artifacts produced by this stage."""
        return []  # Modifies existing verify_result

    def should_run(self, context: VerificationContext) -> bool:
        """
        Run only if deep judgment rubric evaluation was performed.

        This stage only runs when:
        1. No error has occurred
        2. Deep judgment rubric evaluation was performed
        3. There might be traits without valid excerpts
        """
        if context.error:
            return False

        # Check if deep judgment rubric was performed
        deep_judgment_performed = context.get_result_field("deep_judgment_rubric_performed")
        return bool(deep_judgment_performed)

    def execute(self, context: VerificationContext) -> None:
        """
        Check for auto-fail condition and update verification result if needed.

        Args:
            context: Verification context

        Side Effects:
            - May set verify_result to False
            - May set failure_reason explaining the failure
            - Logs warning for each trait that failed
        """
        # Get traits without valid excerpts
        traits_without_excerpts = context.get_result_field("traits_without_valid_excerpts")

        # Check if abstention was detected (takes priority over auto-fail)
        abstention_detected = context.get_result_field("abstention_detected")

        # Check if there are traits that failed excerpt extraction
        if not traits_without_excerpts:
            logger.debug("No traits without excerpts, skipping auto-fail")
            return

        # Skip auto-fail if abstention was detected
        if abstention_detected:
            logger.info(
                f"Abstention detected - skipping deep judgment rubric auto-fail "
                f"for {len(traits_without_excerpts)} trait(s): {', '.join(traits_without_excerpts)}"
            )
            return

        # Auto-fail verification
        trait_names = ", ".join(traits_without_excerpts)
        failure_reason = (
            f"Deep judgment rubric auto-fail: {len(traits_without_excerpts)} trait(s) "
            f"failed to extract valid excerpts after retry attempts: {trait_names}"
        )

        logger.warning(f"Auto-failing verification for question {context.question_id}: {failure_reason}")

        # Log retry counts for transparency
        trait_metadata = context.get_result_field("trait_metadata") or {}
        for trait_name in traits_without_excerpts:
            metadata = trait_metadata.get(trait_name, {})
            retry_count = metadata.get("excerpt_retry_count", 0)
            logger.info(f"  - Trait '{trait_name}' failed after {retry_count} retry attempt(s)")

        # Set verification result to False
        context.set_result_field("verify_result", False)
        context.set_result_field("failure_reason", failure_reason)
