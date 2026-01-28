"""Recursion limit auto-fail stage.

Auto-fails verification when agent hits recursion limit.
"""

import logging

from .base import BaseVerificationStage, VerificationContext

# Set up logger
logger = logging.getLogger(__name__)


class RecursionLimitAutoFailStage(BaseVerificationStage):
    """
    Auto-fails verification when recursion limit is reached.

    This stage:
    1. Only runs if recursion_limit_reached is True
    2. Immediately fails the verification
    3. Sets verify_result to False
    4. Keeps completed_without_errors as True (we want trace and tokens)
    5. Short-circuits remaining verification stages

    Requires:
        - "recursion_limit_reached": Whether agent hit recursion limit (bool)

    Produces:
        - None (only modifies existing verification results)

    Side Effects:
        - Sets verify_result to False
        - Sets field_verification_result to False
        - Logs auto-fail reason

    Note:
        When recursion limit is hit, the test is not valid as the agent
        couldn't complete its reasoning. We still preserve the trace and
        token tracking for analysis purposes.
    """

    @property
    def name(self) -> str:
        """Stage name."""
        return "RecursionLimitAutoFail"

    @property
    def requires(self) -> list[str]:
        """Artifacts required by this stage."""
        return ["recursion_limit_reached"]

    @property
    def produces(self) -> list[str]:
        """Artifacts produced by this stage."""
        return []  # Only modifies existing results

    def should_run(self, context: VerificationContext) -> bool:
        """
        Run only if recursion limit was reached.

        Inherits error-checking from BaseVerificationStage.
        """
        if not super().should_run(context):
            return False

        recursion_limit_reached = context.get_artifact("recursion_limit_reached", False)
        return recursion_limit_reached is True

    def execute(self, context: VerificationContext) -> None:
        """
        Apply auto-fail for recursion limit.

        Args:
            context: Verification context

        Side Effects:
            - Sets verify_result to False
            - Sets verification_result to False
            - Sets field_verification_result to False (if it exists)
            - Logs auto-fail reason
        """
        # Auto-fail: Set all verification results to False
        verification_result = False
        field_verification_result = False

        # Update stored results
        context.set_artifact("verification_result", verification_result)
        context.set_artifact("field_verification_result", field_verification_result)
        context.set_result_field("verify_result", verification_result)

        logger.warning(
            f"Recursion limit auto-fail for question {context.question_id}: "
            f"Agent hit maximum recursion depth. Verification marked as failed. "
            f"Trace and token usage preserved for analysis."
        )
