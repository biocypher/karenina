"""Preset utility functions for VerificationConfig.

DEPRECATED: Import from `karenina.schemas.verification` instead.
"""

# Re-export from new location for backward compatibility
from ...verification.config_presets import (
    create_preset_structure,
    get_default_presets_dir,
    list_preset_files,
    load_preset,
    resolve_preset_path,
    sanitize_model_config,
    sanitize_preset_name,
    save_preset,
    validate_preset_metadata,
)

__all__ = [
    "get_default_presets_dir",
    "list_preset_files",
    "resolve_preset_path",
    "sanitize_model_config",
    "sanitize_preset_name",
    "validate_preset_metadata",
    "create_preset_structure",
    "save_preset",
    "load_preset",
]
