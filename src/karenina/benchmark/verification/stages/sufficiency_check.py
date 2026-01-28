"""Sufficiency detection stage.

Detects when LLM responses lack sufficient information to populate a template.
"""

import logging
from typing import Any

from ..evaluators import detect_sufficiency
from .base import ArtifactKeys, VerificationContext
from .check_stage_base import BaseCheckStage

# Set up logger
logger = logging.getLogger(__name__)


class SufficiencyCheckStage(BaseCheckStage):
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
        - Sets "verify_result" to False if response is insufficient
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
    def _artifact_prefix(self) -> str:
        """Prefix for artifact/result field names."""
        return "sufficiency"

    @property
    def requires(self) -> list[str]:
        """Artifacts required by this stage."""
        # Requires both raw_llm_response and Answer (the template class)
        return [ArtifactKeys.RAW_LLM_RESPONSE, ArtifactKeys.ANSWER]

    def should_run(self, context: VerificationContext) -> bool:
        """Run only if sufficiency detection is enabled, no recursion limit hit, and no prior failures.

        Inherits error-checking from BaseVerificationStage.
        """
        if not super().should_run(context):
            return False
        # Skip if recursion limit was reached (response is truncated/unreliable)
        if context.get_artifact(ArtifactKeys.RECURSION_LIMIT_REACHED, False):
            return False
        # Skip if trace validation failed (trace doesn't end with AI message)
        if context.get_artifact(ArtifactKeys.TRACE_VALIDATION_FAILED, False):
            return False
        # Skip if abstention was detected (already handled by abstention check)
        if context.get_artifact(ArtifactKeys.ABSTENTION_DETECTED, False):
            return False
        return context.sufficiency_enabled

    def _should_trigger_override(self, detected: bool | None, check_performed: bool) -> bool:
        """Trigger override if response is insufficient (detected=False means insufficient)."""
        # Note: For sufficiency, detected=True means sufficient (good)
        # and detected=False means insufficient (trigger override)
        return not detected and check_performed

    def _detect(
        self,
        context: VerificationContext,
    ) -> tuple[bool | None, bool, str | None, dict[str, Any] | None]:
        """
        Detect sufficiency of the raw LLM response for populating the template.

        Returns early with check_performed=False if the template schema cannot be obtained.
        """
        raw_llm_response = context.get_artifact(ArtifactKeys.RAW_LLM_RESPONSE)
        Answer = context.get_artifact(ArtifactKeys.ANSWER)

        # Get the JSON schema from the Answer class
        try:
            template_schema: dict[str, Any] = Answer.model_json_schema()
        except Exception as e:
            logger.warning(f"Failed to get JSON schema from Answer class: {e}")
            # Cannot perform check without schema, return not performed
            return (
                None,  # detected (None since check wasn't performed)
                False,  # check_performed
                None,  # reasoning
                None,  # usage_metadata
            )

        return detect_sufficiency(
            raw_llm_response=raw_llm_response,
            parsing_model=context.parsing_model,
            question_text=context.question_text,
            template_schema=template_schema,
        )
