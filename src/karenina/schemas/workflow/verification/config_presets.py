"""Preset utility functions for VerificationConfig.

DEPRECATED: Import from `karenina.schemas.verification` instead.
"""

# Re-export from new location for backward compatibility
from ...verification.config_presets import (
    create_preset_structure,
    load_preset,
    sanitize_model_config,
    sanitize_preset_name,
    save_preset,
    validate_preset_metadata,
)

__all__ = [
    "sanitize_model_config",
    "sanitize_preset_name",
    "validate_preset_metadata",
    "create_preset_structure",
    "save_preset",
    "load_preset",
]
