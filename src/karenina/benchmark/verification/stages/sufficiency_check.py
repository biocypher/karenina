"""Sufficiency detection stage.

Detects when LLM responses lack sufficient information to populate a template.
"""

import logging
from typing import Any

from ..evaluators import detect_sufficiency
from .base import BaseVerificationStage, VerificationContext

# Set up logger
logger = logging.getLogger(__name__)


class SufficiencyCheckStage(BaseVerificationStage):
    """
    Detects if LLM response has sufficient information for template population.

    This stage:
    1. Only runs if sufficiency detection is enabled and a template exists
    2. Analyzes raw LLM response against the template schema
    3. Uses parsing model to judge if response has information for all fields
    4. If insufficient, sets verification result to False
    5. Provides reasoning for the sufficiency determination

    Requires:
        - "raw_llm_response": Raw LLM response text
        - "Answer": Validated Answer class (Pydantic model)

    Produces:
        - "sufficiency_check_performed": Whether check was attempted (bool)
        - "sufficiency_detected": Whether response is sufficient (bool or None)
        - "sufficiency_override_applied": Whether override was applied (bool)
        - "sufficiency_reasoning": LLM's reasoning for determination (str or None)

    Side Effects:
        - Sets "verification_result" to False if response is insufficient
        - Subsequent parsing stages are skipped if insufficient

    Note:
        Sufficiency detection runs before template parsing. If insufficient,
        subsequent parsing and verification stages are skipped. This provides
        early detection of responses that cannot populate the template.
    """

    @property
    def name(self) -> str:
        """Stage name."""
        return "SufficiencyCheck"

    @property
    def requires(self) -> list[str]:
        """Artifacts required by this stage."""
        # Requires both raw_llm_response and Answer (the template class)
        return ["raw_llm_response", "Answer"]

    @property
    def produces(self) -> list[str]:
        """Artifacts produced by this stage."""
        return [
            "sufficiency_check_performed",
            "sufficiency_detected",
            "sufficiency_override_applied",
            "sufficiency_reasoning",
        ]

    def should_run(self, context: VerificationContext) -> bool:
        """Run only if sufficiency detection is enabled, no recursion limit hit, and no prior failures.

        Inherits error-checking from BaseVerificationStage.
        """
        if not super().should_run(context):
            return False
        # Skip if recursion limit was reached (response is truncated/unreliable)
        if context.get_artifact("recursion_limit_reached", False):
            return False
        # Skip if trace validation failed (trace doesn't end with AI message)
        if context.get_artifact("trace_validation_failed", False):
            return False
        # Skip if abstention was detected (already handled by abstention check)
        if context.get_artifact("abstention_detected", False):
            return False
        return context.sufficiency_enabled

    def execute(self, context: VerificationContext) -> None:
        """
        Detect sufficiency and apply override if needed.

        Args:
            context: Verification context

        Side Effects:
            - Sets sufficiency metadata artifacts
            - May override verification_result to False if insufficient
            - Sets result fields for sufficiency metadata
        """
        raw_llm_response = context.get_artifact("raw_llm_response")
        Answer = context.get_artifact("Answer")

        # Retrieve usage tracker from previous stage or create new one
        usage_tracker = self.get_or_create_usage_tracker(context)

        # Build model string for tracking (centralized via adapter registry)
        from karenina.adapters import format_model_string

        parsing_model = context.parsing_model
        parsing_model_str = format_model_string(parsing_model)

        # Get the JSON schema from the Answer class
        try:
            template_schema: dict[str, Any] = Answer.model_json_schema()
        except Exception as e:
            logger.warning(f"Failed to get JSON schema from Answer class: {e}")
            # Cannot perform check without schema, mark as not performed
            self.set_artifact_and_result(context, "sufficiency_check_performed", False)
            self.set_artifact_and_result(context, "sufficiency_detected", None)
            self.set_artifact_and_result(context, "sufficiency_override_applied", False)
            self.set_artifact_and_result(context, "sufficiency_reasoning", None)
            return

        # Detect sufficiency
        sufficient, sufficiency_check_performed, sufficiency_reasoning, usage_metadata = detect_sufficiency(
            raw_llm_response=raw_llm_response,
            parsing_model=context.parsing_model,
            question_text=context.question_text,
            template_schema=template_schema,
        )

        # Track the sufficiency check call
        if usage_metadata:
            usage_tracker.track_call("sufficiency_check", parsing_model_str, usage_metadata)

        sufficiency_override_applied = False

        # Apply override if insufficient (sufficient=False means we need to fail)
        if not sufficient and sufficiency_check_performed:
            # Mark as failed since response lacks information for template
            verification_result = False
            sufficiency_override_applied = True

            # Update stored result
            context.set_artifact("verification_result", verification_result)
            context.set_result_field("verify_result", verification_result)

            logger.info(f"Insufficient response for question {context.question_id} - overriding result to False")

        # Store sufficiency metadata (both artifact and result field)
        self.set_artifact_and_result(context, "sufficiency_check_performed", sufficiency_check_performed)
        self.set_artifact_and_result(
            context, "sufficiency_detected", sufficient
        )  # True = sufficient, False = insufficient
        self.set_artifact_and_result(context, "sufficiency_override_applied", sufficiency_override_applied)
        self.set_artifact_and_result(context, "sufficiency_reasoning", sufficiency_reasoning)

        # Store updated usage tracker for next stages (artifact only)
        context.set_artifact("usage_tracker", usage_tracker)
