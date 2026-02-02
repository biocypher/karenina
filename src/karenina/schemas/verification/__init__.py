"""Verification configuration and result models.

This module contains models for verification workflow:
- VerificationConfig: Pipeline configuration
- VerificationResult: Single question result
- VerificationJob: Batch job tracking
"""

# Re-export configuration and constants
from .api_models import (
    FinishedTemplate,
    VerificationRequest,
    VerificationStartResponse,
    VerificationStatusResponse,
)
from .config import (
    DEFAULT_ANSWERING_SYSTEM_PROMPT,
    DEFAULT_PARSING_SYSTEM_PROMPT,
    DeepJudgmentTraitConfig,
    VerificationConfig,
)
from .config_presets import (
    create_preset_structure,
    load_preset,
    sanitize_model_config,
    sanitize_preset_name,
    save_preset,
    validate_preset_metadata,
)

# Re-export job
from .job import VerificationJob

# Re-export result components
from .model_identity import ModelIdentity
from .prompt_config import PromptConfig

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
    "DeepJudgmentTraitConfig",
    "PromptConfig",
    # Preset utilities
    "sanitize_model_config",
    "sanitize_preset_name",
    "validate_preset_metadata",
    "create_preset_structure",
    "save_preset",
    "load_preset",
    # Model identity
    "ModelIdentity",
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
