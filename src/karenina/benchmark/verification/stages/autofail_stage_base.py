"""Base class for auto-fail stages.

This module provides a base class that consolidates the common patterns
shared between auto-fail stages (recursion_limit, deep_judgment, deep_judgment_rubric).

Auto-fail stages follow a common pattern:
1. Check if auto-fail condition is met
2. Skip if a higher-priority failure already occurred (e.g., abstention)
3. Set verify_result to False
4. Log the auto-fail reason
"""

import logging
from abc import abstractmethod

from .base import BaseVerificationStage, VerificationContext

logger = logging.getLogger(__name__)


class BaseAutoFailStage(BaseVerificationStage):
    """
    Abstract base class for auto-fail stages.

    Auto-fail stages share a common pattern:
    1. Check if auto-fail should trigger (via should_run)
    2. Check if a higher-priority failure should skip auto-fail (e.g., abstention)
    3. Set verify_result to False (both artifact and result field)
    4. Optionally set field_verification_result to False
    5. Log the auto-fail reason

    Subclasses must implement:
    - name: Stage name property
    - requires: Artifacts required by this stage
    - should_run: When the stage should execute
    - _should_skip_due_to_prior_failure(context): Whether to skip due to abstention etc.
    - _get_autofail_reason(context): The reason message for the auto-fail
    - _set_additional_failure_fields(context): Optional additional fields to set

    The base class handles:
    - Setting verify_result to False (artifact and result field)
    - Setting field_verification_result to False (artifact only)
    - Logging the auto-fail message
    """

    @property
    def produces(self) -> list[str]:
        """Artifacts produced by this stage."""
        return []  # Only modifies existing results

    @abstractmethod
    def _should_skip_due_to_prior_failure(self, context: VerificationContext) -> bool:
        """
        Check if auto-fail should be skipped due to a prior failure.

        For example, if abstention was detected, we don't need to auto-fail
        since the verification is already failed for a more descriptive reason.

        Args:
            context: Verification context

        Returns:
            True if auto-fail should be skipped
        """
        pass

    @abstractmethod
    def _get_autofail_reason(self, context: VerificationContext) -> str:
        """
        Get the reason message for the auto-fail.

        Args:
            context: Verification context

        Returns:
            Human-readable reason for the auto-fail
        """
        pass

    def _set_additional_failure_fields(self, context: VerificationContext) -> None:
        """
        Set additional failure-related fields.

        Override this in subclasses to set stage-specific fields
        (e.g., failure_reason for DeepJudgmentRubricAutoFail).

        Args:
            context: Verification context
        """
        pass

    def _get_log_level(self) -> int:
        """
        Get the logging level for the auto-fail message.

        By convention, auto-fails should use WARNING level since they
        change verify_result due to a detected condition. This follows
        the logging convention documented in base.py.

        Returns:
            Logging level constant (default: logging.WARNING)
        """
        return logging.WARNING

    def execute(self, context: VerificationContext) -> None:
        """
        Execute the auto-fail logic.

        This implements the common pattern:
        1. Check for prior failure (skip if exists)
        2. Set verify_result to False
        3. Set field_verification_result to False
        4. Call _set_additional_failure_fields() for subclass-specific fields
        5. Log the auto-fail reason

        Args:
            context: Verification context

        Side Effects:
            - Sets verify_result to False (artifact and result field)
            - Sets field_verification_result to False (artifact)
            - May set additional fields via _set_additional_failure_fields()
            - Logs auto-fail reason
        """
        # Check if we should skip due to a prior failure
        if self._should_skip_due_to_prior_failure(context):
            return

        # Apply auto-fail
        verification_result = False
        field_verification_result = False

        # Update stored results
        context.set_artifact("verify_result", verification_result)
        context.set_artifact("field_verification_result", field_verification_result)
        context.set_result_field("verify_result", verification_result)

        # Set any additional fields (subclass-specific)
        self._set_additional_failure_fields(context)

        # Log the auto-fail reason
        reason = self._get_autofail_reason(context)
        log_level = self._get_log_level()
        logger.log(log_level, f"{self.name} auto-fail for question {context.question_id}: {reason}")
