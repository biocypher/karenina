"""Deep-judgment auto-fail stage.

Auto-fails verification when excerpts are missing for attributes.
"""

import logging

from ..stage import BaseVerificationStage, VerificationContext

# Set up logger
logger = logging.getLogger(__name__)


class DeepJudgmentAutoFailStage(BaseVerificationStage):
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
        - "abstention_detected": Whether abstention was detected (bool or None)
        - "abstention_check_performed": Whether abstention check ran (bool)
        - "verification_result": Current verification result
        - "field_verification_result": Field verification result

    Produces:
        - None (only modifies existing verification results)

    Side Effects:
        - May set verification_result and field_verification_result to False
        - Logs auto-fail reason

    Note:
        This stage runs after abstention check. If abstention was detected,
        the verification is already failed and we don't need to auto-fail again.
        The auto-fail is specifically for cases where the LLM generated an answer
        but couldn't provide excerpts to support it.
    """

    @property
    def name(self) -> str:
        """Stage name."""
        return "DeepJudgmentAutoFail"

    @property
    def requires(self) -> list[str]:
        """Artifacts required by this stage."""
        return [
            "deep_judgment_performed",
            "attributes_without_excerpts",
            "abstention_detected",
            "abstention_check_performed",
            "verification_result",
            "field_verification_result",
        ]

    @property
    def produces(self) -> list[str]:
        """Artifacts produced by this stage."""
        return []  # Only modifies existing results

    def should_run(self, context: VerificationContext) -> bool:
        """
        Run only if deep-judgment was performed and no errors.

        Also requires that we have attributes_without_excerpts data,
        which is set by deep-judgment parsing.
        """
        if context.error:
            return False

        deep_judgment_performed = context.get_artifact("deep_judgment_performed", False)
        attributes_without_excerpts = context.get_artifact("attributes_without_excerpts")

        return (
            deep_judgment_performed and attributes_without_excerpts is not None and len(attributes_without_excerpts) > 0
        )

    def execute(self, context: VerificationContext) -> None:
        """
        Apply auto-fail if conditions are met.

        Args:
            context: Verification context

        Side Effects:
            - May set verification_result to False
            - May set field_verification_result to False
            - Logs auto-fail reason
        """
        abstention_detected = context.get_artifact("abstention_detected")
        abstention_check_performed = context.get_artifact("abstention_check_performed")
        attributes_without_excerpts = context.get_artifact("attributes_without_excerpts")

        # Only auto-fail if abstention was NOT detected
        # If abstention was detected, verification is already failed for a better reason
        if abstention_detected and abstention_check_performed:
            # Abstention already handled, don't override
            return

        # Auto-fail: Set verification to False
        verification_result = False
        field_verification_result = False

        # Update stored results
        context.set_artifact("verification_result", verification_result)
        context.set_artifact("field_verification_result", field_verification_result)
        context.set_result_field("verify_result", verification_result)

        logger.info(
            f"Deep-judgment auto-fail for question {context.question_id}: "
            f"{len(attributes_without_excerpts)} attributes without excerpts: "
            f"{', '.join(attributes_without_excerpts)}"
        )
