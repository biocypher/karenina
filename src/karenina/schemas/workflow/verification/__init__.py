"""Verification configuration and result models.

This module re-exports all verification-related classes for backward compatibility.
"""

# Re-export configuration and constants
# Re-export API models
from .api_models import (
    FinishedTemplate,
    VerificationRequest,
    VerificationStartResponse,
    VerificationStatusResponse,
)
from .config import (
    DEFAULT_ANSWERING_SYSTEM_PROMPT,
    DEFAULT_PARSING_SYSTEM_PROMPT,
    VerificationConfig,
)

# Re-export job
from .job import VerificationJob

# Re-export result
from .result import VerificationResult

# Re-export result components
from .result_components import (
    VerificationResultDeepJudgment,
    VerificationResultDeepJudgmentRubric,
    VerificationResultMetadata,
    VerificationResultRubric,
    VerificationResultTemplate,
)

__all__ = [
    # Constants
    "DEFAULT_ANSWERING_SYSTEM_PROMPT",
    "DEFAULT_PARSING_SYSTEM_PROMPT",
    # Configuration
    "VerificationConfig",
    # Result components
    "VerificationResultMetadata",
    "VerificationResultTemplate",
    "VerificationResultRubric",
    "VerificationResultDeepJudgment",
    "VerificationResultDeepJudgmentRubric",
    # Result
    "VerificationResult",
    # Job
    "VerificationJob",
    # API models
    "FinishedTemplate",
    "VerificationRequest",
    "VerificationStatusResponse",
    "VerificationStartResponse",
]
