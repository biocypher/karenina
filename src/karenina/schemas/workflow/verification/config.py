"""Verification configuration model.

DEPRECATED: Import from `karenina.schemas.verification` instead.
"""

# Re-export from new location for backward compatibility
from ...verification.config import (
    DEFAULT_ANSWERING_SYSTEM_PROMPT,
    DEFAULT_PARSING_SYSTEM_PROMPT,
    DeepJudgmentTraitConfig,
    VerificationConfig,
)

__all__ = [
    "DEFAULT_ANSWERING_SYSTEM_PROMPT",
    "DEFAULT_PARSING_SYSTEM_PROMPT",
    "DeepJudgmentTraitConfig",
    "VerificationConfig",
]
