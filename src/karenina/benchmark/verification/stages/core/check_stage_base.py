"""Base class for check stages (abstention and sufficiency).

This module provides a base class that consolidates the common patterns
shared between AbstentionCheckStage and SufficiencyCheckStage, which are
~80% structurally identical.
"""

import logging
from abc import abstractmethod
from typing import Any

from .base import ArtifactKeys, BaseVerificationStage, VerificationContext

logger = logging.getLogger(__name__)


class BaseCheckStage(BaseVerificationStage):
    """
    Abstract base class for check stages (abstention, sufficiency).

    Check stages share a common pattern:
    1. Retrieve raw_llm_response and usage tracker from context
    2. Get model string for tracking
    3. Call a detection function that returns (detected, check_performed, reasoning, usage_metadata)
    4. Track usage if metadata exists
    5. Apply override if condition triggers (set verify_result to False)
    6. Store 4 metadata fields using set_artifact_and_result()
    7. Store usage tracker back to context

    Subclasses must implement:
    - name: Stage name property
    - _get_artifact_prefix(): Returns "abstention" or "sufficiency"
    - _should_trigger_override(detected, check_performed): Whether to apply override
    - _detect(context, usage_tracker): The actual detection logic
    - should_run(): Override with check-specific conditions
    """

    @property
    @abstractmethod
    def _artifact_prefix(self) -> str:
        """
        Prefix for artifact/result field names.

        Returns:
            "abstention" or "sufficiency"
        """
        pass

    @abstractmethod
    def _should_trigger_override(self, detected: bool | None, check_performed: bool) -> bool:
        """
        Determine if the override should be triggered.

        This accounts for the semantic difference between checks:
        - Abstention: override if abstention_detected is True
        - Sufficiency: override if sufficient is False (i.e., insufficient)

        Args:
            detected: The detection result from _detect()
            check_performed: Whether the check was performed

        Returns:
            True if verify_result should be overridden to False
        """
        pass

    @abstractmethod
    def _detect(
        self,
        context: VerificationContext,
    ) -> tuple[bool | None, bool, str | None, dict[str, Any] | None]:
        """
        Perform the detection logic.

        Subclasses implement this to call the appropriate detection function
        (detect_abstention or detect_sufficiency).

        Args:
            context: Verification context with raw_llm_response and other data

        Returns:
            Tuple of (detected, check_performed, reasoning, usage_metadata)
            - detected: The primary result (abstention_detected or sufficient).
                        Should be None if check_performed is False.
            - check_performed: Whether the check was performed successfully
            - reasoning: LLM's reasoning for the determination (or None)
            - usage_metadata: Token usage data (or None)
        """
        pass

    @property
    def produces(self) -> list[str]:
        """Artifacts produced by this stage."""
        prefix = self._artifact_prefix
        return [
            f"{prefix}_check_performed",
            f"{prefix}_detected",
            f"{prefix}_override_applied",
            f"{prefix}_reasoning",
        ]

    def execute(self, context: VerificationContext) -> None:
        """
        Execute the check stage logic.

        This implements the common pattern shared by abstention and sufficiency checks:
        1. Retrieve raw_llm_response and usage tracker
        2. Call the detection function via _detect()
        3. Track usage
        4. Apply override if triggered
        5. Store all metadata

        Args:
            context: Verification context

        Side Effects:
            - Sets check metadata artifacts
            - May override verify_result to False if triggered
            - Sets result fields for check metadata
        """
        # Get the artifact prefix for consistent naming
        prefix = self._artifact_prefix

        # Retrieve usage tracker from previous stage or create new one
        usage_tracker = self.get_or_create_usage_tracker(context)

        # Build model string for tracking (centralized via adapter registry)
        parsing_model = context.parsing_model
        parsing_model_str = self.get_model_string(parsing_model)

        # Perform the detection
        detected, check_performed, reasoning, usage_metadata = self._detect(context)

        # Track the check call
        if usage_metadata:
            usage_tracker.track_call(f"{prefix}_check", parsing_model_str, usage_metadata)

        override_applied = False

        # Apply override if triggered
        if self._should_trigger_override(detected, check_performed):
            # Mark as failed
            verification_result = False
            override_applied = True

            # Update stored result
            context.set_artifact(ArtifactKeys.VERIFY_RESULT, verification_result)
            context.set_result_field(ArtifactKeys.VERIFY_RESULT, verification_result)

            logger.warning(f"{self.name} triggered for question {context.question_id} - overriding result to False")

        # Store check metadata (both artifact and result field)
        self.set_artifact_and_result(context, f"{prefix}_check_performed", check_performed)
        self.set_artifact_and_result(context, f"{prefix}_detected", detected)
        self.set_artifact_and_result(context, f"{prefix}_override_applied", override_applied)
        self.set_artifact_and_result(context, f"{prefix}_reasoning", reasoning)

        # Store updated usage tracker for next stages (artifact only)
        context.set_artifact(ArtifactKeys.USAGE_TRACKER, usage_tracker)
