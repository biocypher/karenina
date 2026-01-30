"""Recursion limit auto-fail stage.

Auto-fails verification when agent hits recursion limit.
"""

from ..core.autofail_stage_base import BaseAutoFailStage
from ..core.base import ArtifactKeys, VerificationContext


class RecursionLimitAutoFailStage(BaseAutoFailStage):
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
        return [ArtifactKeys.RECURSION_LIMIT_REACHED]

    def should_run(self, context: VerificationContext) -> bool:
        """
        Run only if recursion limit was reached.

        Inherits error-checking from BaseVerificationStage.
        """
        if not super().should_run(context):
            return False

        recursion_limit_reached = context.get_artifact(ArtifactKeys.RECURSION_LIMIT_REACHED, False)
        return recursion_limit_reached is True

    def _should_skip_due_to_prior_failure(self, context: VerificationContext) -> bool:  # noqa: ARG002
        """
        Recursion limit auto-fail never skips due to prior failure.

        This is one of the first checks in the pipeline.
        """
        return False

    def _get_autofail_reason(self, context: VerificationContext) -> str:  # noqa: ARG002
        """Get the auto-fail reason message."""
        return (
            "Agent hit maximum recursion depth. Verification marked as failed. "
            "Trace and token usage preserved for analysis."
        )
