"""Trace validation auto-fail stage.

Auto-fails verification when agent trace doesn't end with an AI message.
"""

import logging

from ....infrastructure.llm.mcp_utils import extract_final_ai_message
from ..stage import BaseVerificationStage, VerificationContext

# Set up logger
logger = logging.getLogger(__name__)


class TraceValidationAutoFailStage(BaseVerificationStage):
    """
    Auto-fails verification when trace doesn't end with an AI message.

    This stage:
    1. Always runs after GenerateAnswerStage (if raw_llm_response exists)
    2. Checks if the trace ends with a valid AI message
    3. If not, auto-fails the verification (verify_result=False)
    4. Keeps completed_without_errors as True (we want trace and tokens)
    5. Subsequent parsing stages will skip their extraction logic

    Requires:
        - "raw_llm_response": Raw LLM response text (agent trace)

    Produces:
        - "trace_validation_failed": Whether trace validation failed (bool)
        - "trace_validation_error": Error message if validation failed

    Side Effects:
        - Sets verify_result to False if trace doesn't end with AI message
        - Stores trace validation error for diagnostics
        - Logs auto-fail reason

    Note:
        A valid agent trace should always end with an AI message containing
        the final answer. Traces ending with tool messages or having no
        AI message content indicate incomplete or failed agent execution.
    """

    @property
    def name(self) -> str:
        """Stage name."""
        return "TraceValidationAutoFail"

    @property
    def requires(self) -> list[str]:
        """Artifacts required by this stage."""
        return ["raw_llm_response"]

    @property
    def produces(self) -> list[str]:
        """Artifacts produced by this stage."""
        return ["trace_validation_failed", "trace_validation_error"]

    def should_run(self, context: VerificationContext) -> bool:
        """
        Run if we have raw_llm_response and no errors yet.

        Skip if there's already an error (to preserve error state).
        """
        if context.error:
            return False

        return context.has_artifact("raw_llm_response")

    def execute(self, context: VerificationContext) -> None:
        """
        Validate that trace ends with an AI message.

        Args:
            context: Verification context

        Side Effects:
            - Sets trace_validation_failed artifact
            - Sets trace_validation_error artifact (if failed)
            - Sets verify_result to False (if failed)
            - Logs validation result
        """
        raw_llm_response = context.get_artifact("raw_llm_response")

        # Try to extract the final AI message
        # This validates that the trace ends with a valid AI message
        _, error = extract_final_ai_message(raw_llm_response)

        if error is not None:
            # Trace validation failed - auto-fail the test
            logger.warning(
                f"Trace validation auto-fail for question {context.question_id}: "
                f"{error}. Verification marked as failed. "
                f"Trace and token usage preserved for analysis."
            )

            # Store validation failure info
            context.set_artifact("trace_validation_failed", True)
            context.set_artifact("trace_validation_error", error)

            # Auto-fail: Set verification result to False
            context.set_artifact("verification_result", False)
            context.set_artifact("field_verification_result", False)
            context.set_result_field("verify_result", False)

            # Store error in result for diagnostics
            context.set_result_field("trace_extraction_error", error)
            context.set_result_field("trace_validation_failed", True)
        else:
            # Trace validation passed
            context.set_artifact("trace_validation_failed", False)
            context.set_artifact("trace_validation_error", None)
            context.set_result_field("trace_validation_failed", False)

            logger.debug(f"Trace validation passed for question {context.question_id}")
