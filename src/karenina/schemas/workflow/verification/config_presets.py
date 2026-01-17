"""Preset utility functions for VerificationConfig.

This module provides utilities for managing verification configuration presets,
including sanitization, validation, serialization, and file I/O operations.
"""

import json
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import VerificationConfig


def sanitize_model_config(model: dict[str, Any]) -> dict[str, Any]:
    """
    Sanitize model configuration to remove interface-specific fields.

    This function removes fields that don't apply to a model's interface type,
    ensuring only relevant configuration is saved in presets.

    Note: manual_traces field is automatically excluded (not saved in presets).
    Manual traces must be uploaded separately when using a preset with manual interface.

    Args:
        model: Model configuration dictionary

    Returns:
        Sanitized model configuration with only applicable fields

    Example:
        >>> config = {"interface": "langchain", "endpoint_base_url": "http://..."}
        >>> sanitized = sanitize_model_config(config)
        >>> # endpoint_base_url removed since interface is not openai_endpoint
    """
    sanitized: dict[str, Any] = {
        "id": model["id"],
        "model_provider": model["model_provider"],
        "model_name": model["model_name"],
        "temperature": model["temperature"],
        "interface": model["interface"],
        "system_prompt": model["system_prompt"],
    }

    # Include max_retries if present
    if "max_retries" in model:
        sanitized["max_retries"] = model["max_retries"]

    # Only include endpoint fields for openai_endpoint interface
    if model["interface"] == "openai_endpoint":
        if "endpoint_base_url" in model and model["endpoint_base_url"]:
            sanitized["endpoint_base_url"] = model["endpoint_base_url"]
        if "endpoint_api_key" in model and model["endpoint_api_key"]:
            sanitized["endpoint_api_key"] = model["endpoint_api_key"]

    # Only include MCP fields if they have values
    if "mcp_urls_dict" in model and model["mcp_urls_dict"]:
        sanitized["mcp_urls_dict"] = model["mcp_urls_dict"]
    if "mcp_tool_filter" in model and model["mcp_tool_filter"]:
        sanitized["mcp_tool_filter"] = model["mcp_tool_filter"]

    # Include extra_kwargs if present (vendor-specific API keys, custom parameters, etc.)
    if "extra_kwargs" in model and model["extra_kwargs"]:
        sanitized["extra_kwargs"] = model["extra_kwargs"]

    # Include agent_middleware if present (for MCP-enabled agents)
    if "agent_middleware" in model and model["agent_middleware"]:
        sanitized["agent_middleware"] = model["agent_middleware"]

    # Include max_context_tokens if specified (for summarization middleware)
    # Relevant for openrouter and openai_endpoint interfaces
    if "max_context_tokens" in model and model["max_context_tokens"] is not None:
        sanitized["max_context_tokens"] = model["max_context_tokens"]

    return sanitized


def sanitize_preset_name(name: str) -> str:
    """
    Convert preset name to safe filename.

    Transforms a preset name into a sanitized filename by:
    - Converting to lowercase
    - Replacing spaces with hyphens
    - Removing non-alphanumeric characters (except hyphens)
    - Removing consecutive hyphens
    - Limiting length to 96 characters
    - Adding .json extension

    Args:
        name: Preset name

    Returns:
        Sanitized filename (e.g., "Quick Test" -> "quick-test.json")

    Example:
        >>> sanitize_preset_name("My Test Config!")
        "my-test-config.json"
    """
    sanitized = name.lower()
    sanitized = sanitized.replace(" ", "-")
    sanitized = re.sub(r"[^a-z0-9-]", "", sanitized)
    sanitized = re.sub(r"-+", "-", sanitized)
    sanitized = sanitized.strip("-")

    if not sanitized:
        sanitized = "preset"

    if len(sanitized) > 96:
        sanitized = sanitized[:96]

    return f"{sanitized}.json"


def validate_preset_metadata(name: str, description: str | None = None) -> None:
    """
    Validate preset name and description length limits.

    Note: This function only validates basic metadata constraints (length limits).
    Name uniqueness must be checked separately by the caller (server has the list).

    Args:
        name: Preset name
        description: Optional preset description

    Raises:
        ValueError: If validation fails

    Example:
        >>> validate_preset_metadata("Test", "A test preset")
        >>> # Passes validation
        >>> validate_preset_metadata("", "Description")
        ValueError: Preset name cannot be empty
    """
    # Validate name
    if not name or not isinstance(name, str) or len(name.strip()) == 0:
        raise ValueError("Preset name cannot be empty")

    if len(name) > 100:
        raise ValueError("Preset name cannot exceed 100 characters")

    # Validate description if provided
    if description is not None and len(description) > 500:
        raise ValueError("Description cannot exceed 500 characters")


def create_preset_structure(
    preset_id: str,
    name: str,
    description: str | None,
    config_dict: dict[str, Any],
    created_at: str,
    updated_at: str,
) -> dict[str, Any]:
    """
    Create standardized preset data structure.

    This function provides a consistent format for preset metadata across
    all preset operations.

    Args:
        preset_id: UUID for the preset
        name: Preset name
        description: Optional preset description
        config_dict: VerificationConfig as dictionary
        created_at: ISO format timestamp
        updated_at: ISO format timestamp

    Returns:
        Preset dictionary with standardized structure

    Example:
        >>> preset = create_preset_structure(
        ...     preset_id="abc-123",
        ...     name="Test",
        ...     description="A test preset",
        ...     config_dict={...},
        ...     created_at="2025-11-03T12:00:00Z",
        ...     updated_at="2025-11-03T12:00:00Z"
        ... )
    """
    return {
        "id": preset_id,
        "name": name,
        "description": description,
        "config": config_dict,
        "created_at": created_at,
        "updated_at": updated_at,
    }


def save_preset(
    config: "VerificationConfig",
    name: str,
    description: str | None = None,
    presets_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Save a VerificationConfig as a preset file.

    Args:
        config: The VerificationConfig instance to save
        name: Preset name
        description: Optional preset description
        presets_dir: Optional path to presets directory.
                    If None, uses KARENINA_PRESETS_DIR env or default location.

    Returns:
        Dictionary with preset metadata (id, name, description, filepath, created_at, updated_at)

    Raises:
        ValueError: If name is invalid
        IOError: If file operations fail

    Example:
        >>> config = VerificationConfig(...)
        >>> metadata = save_preset(config, "Quick Test", "Fast testing configuration")
        >>> print(f"Saved to {metadata['filepath']}")
    """
    # Determine presets directory
    if presets_dir is None:
        # Check environment variable first
        env_presets_dir = os.getenv("KARENINA_PRESETS_DIR")
        if env_presets_dir:
            presets_dir = Path(env_presets_dir)
        else:
            # Default to presets/ directory in project root
            project_root = Path(__file__).parent.parent.parent.parent.parent.parent
            presets_dir = project_root / "presets"

    presets_dir = presets_dir.resolve()

    # Validate metadata
    validate_preset_metadata(name, description)

    # Ensure directory exists
    presets_dir.mkdir(parents=True, exist_ok=True)

    # Generate preset ID and timestamps
    preset_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()

    # Convert config to dict and sanitize models
    config_dict = config.model_dump(mode="json")

    # Sanitize answering and parsing models
    if "answering_models" in config_dict:
        config_dict["answering_models"] = [sanitize_model_config(m) for m in config_dict["answering_models"]]
    if "parsing_models" in config_dict:
        config_dict["parsing_models"] = [sanitize_model_config(m) for m in config_dict["parsing_models"]]

    # Create preset structure
    preset = create_preset_structure(
        preset_id=preset_id,
        name=name,
        description=description,
        config_dict=config_dict,
        created_at=now,
        updated_at=now,
    )

    # Generate safe filename
    filename = sanitize_preset_name(name)
    filepath = presets_dir / filename

    # Check if file already exists
    if filepath.exists():
        raise ValueError(f"A preset file already exists at {filepath}. Please use a different name.")

    # Write preset to file
    with open(filepath, "w") as f:
        json.dump(preset, f, indent=2)

    return {
        "id": preset_id,
        "name": name,
        "description": description,
        "filepath": str(filepath),
        "created_at": now,
        "updated_at": now,
    }


def load_preset(filepath: Path) -> "VerificationConfig":
    """
    Load a VerificationConfig from a preset file.

    Args:
        filepath: Path to the preset JSON file

    Returns:
        VerificationConfig instance loaded from the preset

    Raises:
        FileNotFoundError: If the preset file doesn't exist
        json.JSONDecodeError: If the preset file is corrupted
        ValueError: If the config is invalid

    Example:
        >>> config = load_preset(Path("presets/gpt-oss.json"))
        >>> results = verify_questions(checkpoint, config)
    """
    # Import here to avoid circular imports
    from .config import VerificationConfig

    filepath = filepath.resolve()

    # Check if file exists
    if not filepath.exists():
        raise FileNotFoundError(f"Preset file not found at {filepath}")

    # Load preset file
    try:
        with open(filepath) as f:
            preset = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Preset file at {filepath} is corrupted: {e.msg}",
            e.doc,
            e.pos,
        ) from e

    # Extract config from preset wrapper
    config_data = preset.get("config")
    if not config_data:
        raise ValueError(f"Preset file '{filepath}' has no configuration data")

    # Create VerificationConfig instance from the config dict
    try:
        return VerificationConfig(**config_data)
    except Exception as e:
        preset_name = preset.get("name", "unknown")
        raise ValueError(f"Failed to load preset '{preset_name}' from {filepath}: {e}") from e
