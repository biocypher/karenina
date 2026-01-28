"""Abstention detection stage.

Detects when LLMs refuse to answer or abstain from responding.
"""

from typing import Any

from ..evaluators import detect_abstention
from .base import ArtifactKeys, VerificationContext
from .check_stage_base import BaseCheckStage


class AbstentionCheckStage(BaseCheckStage):
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
        - Sets "verify_result" to False if abstention detected
        - In rubric_only mode, this may be the first stage to set verify_result

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
    def _artifact_prefix(self) -> str:
        """Prefix for artifact/result field names."""
        return "abstention"

    @property
    def requires(self) -> list[str]:
        """Artifacts required by this stage."""
        # Only requires raw_llm_response - verify_result is optional
        # In rubric_only mode, verify_result may not exist yet
        return [ArtifactKeys.RAW_LLM_RESPONSE]

    def should_run(self, context: VerificationContext) -> bool:
        """Run only if abstention detection is enabled and no recursion limit hit.

        Inherits error-checking from BaseVerificationStage.
        """
        if not super().should_run(context):
            return False
        # Skip if recursion limit was reached (response is truncated/unreliable)
        if context.get_artifact(ArtifactKeys.RECURSION_LIMIT_REACHED, False):
            return False
        return context.abstention_enabled

    def _should_trigger_override(self, detected: bool | None, check_performed: bool) -> bool:
        """Trigger override if abstention was detected."""
        return bool(detected) and check_performed

    def _detect(
        self,
        context: VerificationContext,
    ) -> tuple[bool | None, bool, str | None, dict[str, Any] | None]:
        """Detect abstention in the raw LLM response."""
        raw_llm_response = context.get_artifact(ArtifactKeys.RAW_LLM_RESPONSE)

        return detect_abstention(
            raw_llm_response=raw_llm_response,
            parsing_model=context.parsing_model,
            question_text=context.question_text,
        )
