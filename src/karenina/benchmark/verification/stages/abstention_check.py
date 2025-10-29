"""Abstention detection stage.

Detects when LLMs refuse to answer or abstain from responding.
"""

import logging

from ..abstention_checker import detect_abstention
from ..stage import BaseVerificationStage, VerificationContext

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
        In template modes, abstention detection runs after template verification
        and can override the result. In rubric_only mode, it may run before any
        verification result is set.
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
        """Run only if abstention detection is enabled and no errors."""
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

        # Detect abstention
        abstention_detected, abstention_check_performed, abstention_reasoning = detect_abstention(
            raw_llm_response=raw_llm_response,
            parsing_model=context.parsing_model,
            question_text=context.question_text,
        )

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

        # Store in result builder
        context.set_result_field("abstention_check_performed", abstention_check_performed)
        context.set_result_field("abstention_detected", abstention_detected)
        context.set_result_field("abstention_override_applied", abstention_override_applied)
        context.set_result_field("abstention_reasoning", abstention_reasoning)
