"""Abstention detection stage.

Detects when LLMs refuse to answer or abstain from responding.
"""

import logging

from ..evaluators import detect_abstention
from ..utils import UsageTracker
from .base import BaseVerificationStage, VerificationContext

# Set up logger
logger = logging.getLogger(__name__)


class AbstentionCheckStage(BaseVerificationStage):
    """
    Detects LLM abstention or refusal to answer.

    This stage:
    1. Only runs if abstention detection is enabled
    2. Analyzes raw LLM response for refusal patterns
    3. Uses parsing model to judge if response is an abstention
    4. If abstention detected, sets verification result to False
    5. Provides reasoning for the abstention determination

    Requires:
        - "raw_llm_response": Raw LLM response text

    Produces:
        - "abstention_check_performed": Whether check was attempted (bool)
        - "abstention_detected": Whether abstention was detected (bool or None)
        - "abstention_override_applied": Whether override was applied (bool)
        - "abstention_reasoning": LLM's reasoning for determination (str or None)

    Side Effects:
        - Sets "verification_result" to False if abstention detected
        - In rubric_only mode, this may be the first stage to set verification_result

    Note:
        Abstention detection runs before template parsing. If abstention is detected,
        subsequent parsing and verification stages are skipped. This prevents wasted
        LLM calls when the model refused to answer.
    """

    @property
    def name(self) -> str:
        """Stage name."""
        return "AbstentionCheck"

    @property
    def requires(self) -> list[str]:
        """Artifacts required by this stage."""
        # Only requires raw_llm_response - verification_result is optional
        # In rubric_only mode, verification_result may not exist yet
        return ["raw_llm_response"]

    @property
    def produces(self) -> list[str]:
        """Artifacts produced by this stage."""
        return [
            "abstention_check_performed",
            "abstention_detected",
            "abstention_override_applied",
            "abstention_reasoning",
        ]

    def should_run(self, context: VerificationContext) -> bool:
        """Run only if abstention detection is enabled, no errors, and no recursion limit hit."""
        # Skip if recursion limit was reached (response is truncated/unreliable)
        if context.get_artifact("recursion_limit_reached", False):
            return False
        return context.abstention_enabled and not context.error

    def execute(self, context: VerificationContext) -> None:
        """
        Detect abstention and apply override if needed.

        Args:
            context: Verification context

        Side Effects:
            - Sets abstention metadata artifacts
            - May override verification_result to False if abstention detected
            - Sets result fields for abstention metadata
        """
        raw_llm_response = context.get_artifact("raw_llm_response")

        # Retrieve usage tracker from previous stage or initialize new one
        usage_tracker = context.get_artifact("usage_tracker")
        if usage_tracker is None:
            usage_tracker = UsageTracker()
            logger.warning("No usage tracker found in context, initializing new one")

        # Build model string for tracking (centralized via adapter registry)
        from karenina.adapters import format_model_string

        parsing_model = context.parsing_model
        parsing_model_str = format_model_string(parsing_model)

        # Detect abstention
        abstention_detected, abstention_check_performed, abstention_reasoning, usage_metadata = detect_abstention(
            raw_llm_response=raw_llm_response,
            parsing_model=context.parsing_model,
            question_text=context.question_text,
        )

        # Track the abstention check call
        if usage_metadata:
            usage_tracker.track_call("abstention_check", parsing_model_str, usage_metadata)

        abstention_override_applied = False

        # Apply override if abstention detected
        if abstention_detected and abstention_check_performed:
            # Mark as failed since model didn't provide a real answer
            verification_result = False
            abstention_override_applied = True

            # Update stored result
            context.set_artifact("verification_result", verification_result)
            context.set_result_field("verify_result", verification_result)

            logger.info(f"Abstention detected for question {context.question_id} - overriding result to False")

        # Store abstention metadata
        context.set_artifact("abstention_check_performed", abstention_check_performed)
        context.set_artifact("abstention_detected", abstention_detected)
        context.set_artifact("abstention_override_applied", abstention_override_applied)
        context.set_artifact("abstention_reasoning", abstention_reasoning)

        # Store updated usage tracker for next stages
        context.set_artifact("usage_tracker", usage_tracker)

        # Store in result builder
        context.set_result_field("abstention_check_performed", abstention_check_performed)
        context.set_result_field("abstention_detected", abstention_detected)
        context.set_result_field("abstention_override_applied", abstention_override_applied)
        context.set_result_field("abstention_reasoning", abstention_reasoning)
