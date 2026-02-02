"""Verification configuration and result models.

DEPRECATED: Import from `karenina.schemas.verification` instead.

This module re-exports all verification-related classes for backward compatibility.
"""

# Re-export from new location for backward compatibility
from ...verification import (
    DEFAULT_ANSWERING_SYSTEM_PROMPT,
    DEFAULT_PARSING_SYSTEM_PROMPT,
    DeepJudgmentTraitConfig,
    FinishedTemplate,
    VerificationConfig,
    VerificationJob,
    VerificationRequest,
    VerificationResult,
    VerificationResultDeepJudgment,
    VerificationResultDeepJudgmentRubric,
    VerificationResultMetadata,
    VerificationResultRubric,
    VerificationResultTemplate,
    VerificationStartResponse,
    VerificationStatusResponse,
    create_preset_structure,
    load_preset,
    sanitize_model_config,
    sanitize_preset_name,
    save_preset,
    validate_preset_metadata,
)

__all__ = [
    # Constants
    "DEFAULT_ANSWERING_SYSTEM_PROMPT",
    "DEFAULT_PARSING_SYSTEM_PROMPT",
    # Configuration
    "VerificationConfig",
    "DeepJudgmentTraitConfig",
    # Preset utilities
    "sanitize_model_config",
    "sanitize_preset_name",
    "validate_preset_metadata",
    "create_preset_structure",
    "save_preset",
    "load_preset",
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
