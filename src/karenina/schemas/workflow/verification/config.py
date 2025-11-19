"""Verification configuration model."""

import contextlib
import json
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..models import (
    INTERFACE_LANGCHAIN,
    INTERFACES_NO_PROVIDER_REQUIRED,
    FewShotConfig,
    ModelConfig,
)

# Default system prompts for answering and parsing models
DEFAULT_ANSWERING_SYSTEM_PROMPT = "You are an expert assistant. Answer the question accurately and concisely."
DEFAULT_PARSING_SYSTEM_PROMPT = (
    "You are a validation assistant. Parse and validate responses against the given Pydantic template."
)


class DeepJudgmentTraitConfig(BaseModel):
    """
    Configuration for deep judgment evaluation of a single rubric trait.

    This model validates trait-level deep judgment settings used in custom mode.
    """

    enabled: bool = True
    excerpt_enabled: bool = True
    max_excerpts: int | None = None
    fuzzy_match_threshold: float | None = None
    excerpt_retry_attempts: int | None = None
    search_enabled: bool = False


class VerificationConfig(BaseModel):
    """Configuration for verification run with multiple models."""

    answering_models: list[ModelConfig] = Field(default_factory=list)
    parsing_models: list[ModelConfig]
    replicate_count: int = 1  # Number of times to run each test combination

    # Parsing-only mode (for TaskEval and similar use cases)
    parsing_only: bool = False  # When True, only parsing models are required

    # Rubric evaluation settings
    rubric_enabled: bool = False
    rubric_trait_names: list[str] | None = None  # Optional filter for specific traits
    rubric_evaluation_strategy: Literal["batch", "sequential"] | None = "batch"
    # - "batch": Evaluate all LLM traits in a single call (efficient, requires JSON output)
    # - "sequential": Evaluate traits one-by-one (reliable, more expensive)

    # Evaluation mode: determines which stages run in the verification pipeline
    evaluation_mode: Literal["template_only", "template_and_rubric", "rubric_only"] = "template_only"
    # - "template_only": Run template verification only (default behavior)
    # - "template_and_rubric": Run both template verification AND rubric evaluation
    # - "rubric_only": Skip template verification, only evaluate rubrics on raw LLM response

    # Abstention detection settings
    abstention_enabled: bool = False  # Enable abstention/refusal detection

    # Embedding check settings (semantic similarity fallback)
    embedding_check_enabled: bool = False  # Enable semantic similarity fallback
    embedding_check_model: str = "all-MiniLM-L6-v2"  # SentenceTransformer model for embeddings
    embedding_check_threshold: float = 0.85  # Similarity threshold (0.0-1.0)

    # Async execution settings
    async_enabled: bool = True  # Enable parallel execution
    async_max_workers: int = 2  # Number of parallel workers

    # Deep-judgment settings (multi-stage parsing with excerpts and reasoning)
    deep_judgment_enabled: bool = False  # Enable deep-judgment analysis (default: disabled)
    deep_judgment_max_excerpts_per_attribute: int = 3  # Max excerpts to extract per attribute
    deep_judgment_fuzzy_match_threshold: float = 0.80  # Similarity threshold for excerpt validation
    deep_judgment_excerpt_retry_attempts: int = 2  # Additional retry attempts for excerpt validation

    # Search-enhanced deep-judgment settings (validate excerpts against external evidence)
    deep_judgment_search_enabled: bool = False  # Enable search validation for excerpts
    deep_judgment_search_tool: str | Any = "tavily"  # Search tool name or callable instance
    # Supported built-in tools: "tavily"
    # Can also pass any callable: (str | list[str]) -> (str | list[str])
    # Examples: langchain tools, MCP tools, custom functions

    # Deep-judgment rubric settings (global defaults for per-trait configuration)
    deep_judgment_rubric_max_excerpts_default: int = 7  # Default max excerpts per trait (higher than templates)
    deep_judgment_rubric_fuzzy_match_threshold_default: float = 0.80  # Default fuzzy match threshold for traits
    deep_judgment_rubric_excerpt_retry_attempts_default: int = 2  # Default retry attempts for trait excerpts
    deep_judgment_rubric_search_tool: str | Any = "tavily"  # Search tool for rubric hallucination detection

    # Deep-judgment rubric configuration modes (NEW - runtime control of deep judgment)
    deep_judgment_rubric_mode: Literal["disabled", "enable_all", "use_checkpoint", "custom"] = "disabled"
    # - "disabled": Deep judgment is OFF (default, explicit)
    # - "enable_all": Apply deep judgment to all LLM traits (respects excerpt toggle)
    # - "use_checkpoint": Use deep judgment settings saved in checkpoint (if available)
    # - "custom": Use per-trait configuration from deep_judgment_rubric_config

    deep_judgment_rubric_global_excerpts: bool = True  # For enable_all mode: enable/disable excerpts globally
    deep_judgment_rubric_config: dict[str, Any] | None = None  # For custom mode: nested trait config
    # Expected structure for custom mode:
    # {
    #   "global": {
    #     "TraitName": {"enabled": True, "excerpt_enabled": True, ...}
    #   },
    #   "question_specific": {
    #     "question-id": {
    #       "TraitName": {"enabled": True, ...}
    #     }
    #   }
    # }

    # Few-shot prompting settings
    few_shot_config: FewShotConfig | None = None  # New flexible configuration

    # Database storage settings
    db_config: Any | None = None  # DBConfig instance for automatic result persistence

    # Legacy few-shot fields for backward compatibility (deprecated)
    few_shot_enabled: bool | None = None
    few_shot_mode: Literal["all", "k-shot", "custom"] | None = None
    few_shot_k: int | None = None

    # Legacy fields for backward compatibility (deprecated)
    answering_model_provider: str | None = None
    answering_model_name: str | None = None
    answering_temperature: float | None = None
    answering_interface: str | None = None
    answering_system_prompt: str | None = None

    parsing_model_provider: str | None = None
    parsing_model_name: str | None = None
    parsing_temperature: float | None = None
    parsing_interface: str | None = None
    parsing_system_prompt: str | None = None

    def __init__(self, **data: Any) -> None:
        """
        Initialize with backward compatibility for single model configs.

        Configuration precedence (highest to lowest):
        1. Explicit arguments (including preset values)
        2. Environment variables (only if set)
        3. Field defaults
        """
        # Read environment variables for embedding check settings (only if not explicitly provided AND env var is set)
        if "embedding_check_enabled" not in data:
            env_val = os.getenv("EMBEDDING_CHECK")
            if env_val is not None:
                data["embedding_check_enabled"] = env_val.lower() in ("true", "1", "yes")
            # else: let Pydantic use field default (False)

        if "embedding_check_model" not in data:
            env_val = os.getenv("EMBEDDING_CHECK_MODEL")
            if env_val is not None:
                data["embedding_check_model"] = env_val
            # else: let Pydantic use field default ("all-MiniLM-L6-v2")

        if "embedding_check_threshold" not in data:
            env_val = os.getenv("EMBEDDING_CHECK_THRESHOLD")
            if env_val is not None:
                # Invalid env var value will let Pydantic use field default (0.85)
                with contextlib.suppress(ValueError):
                    data["embedding_check_threshold"] = float(env_val)
            # else: let Pydantic use field default (0.85)

        # Read environment variables for async execution settings (only if not explicitly provided AND env var is set)
        if "async_enabled" not in data:
            env_val = os.getenv("KARENINA_ASYNC_ENABLED")
            if env_val is not None:
                data["async_enabled"] = env_val.lower() in ("true", "1", "yes")
            # else: let Pydantic use field default (True)

        if "async_max_workers" not in data:
            env_val = os.getenv("KARENINA_ASYNC_MAX_WORKERS")
            if env_val is not None:
                # Invalid env var value will let Pydantic use field default (2)
                with contextlib.suppress(ValueError):
                    data["async_max_workers"] = int(env_val)
            # else: let Pydantic use field default (2)

        # If legacy single model fields are provided, convert to arrays
        if "answering_models" not in data and any(k.startswith("answering_") for k in data):
            answering_model = ModelConfig(
                id="answering-legacy",
                model_provider=data.get("answering_model_provider") or None,
                model_name=data.get("answering_model_name", ""),
                temperature=data.get("answering_temperature", 0.1),
                interface=data.get("answering_interface", INTERFACE_LANGCHAIN),
                system_prompt=data.get("answering_system_prompt", DEFAULT_ANSWERING_SYSTEM_PROMPT),
            )
            data["answering_models"] = [answering_model]

        if "parsing_models" not in data and any(k.startswith("parsing_") for k in data):
            parsing_model = ModelConfig(
                id="parsing-legacy",
                model_provider=data.get("parsing_model_provider") or None,
                model_name=data.get("parsing_model_name", ""),
                temperature=data.get("parsing_temperature", 0.1),
                interface=data.get("parsing_interface", INTERFACE_LANGCHAIN),
                system_prompt=data.get("parsing_system_prompt", DEFAULT_PARSING_SYSTEM_PROMPT),
            )
            data["parsing_models"] = [parsing_model]

        # Convert legacy few-shot fields to new FewShotConfig
        if "few_shot_config" not in data and any(
            k in data for k in ["few_shot_enabled", "few_shot_mode", "few_shot_k"]
        ):
            few_shot_enabled = data.get("few_shot_enabled", False)
            few_shot_mode = data.get("few_shot_mode", "all")
            few_shot_k = data.get("few_shot_k", 3)

            # Create FewShotConfig from legacy settings
            data["few_shot_config"] = FewShotConfig(
                enabled=few_shot_enabled,
                global_mode=few_shot_mode,
                global_k=few_shot_k,
            )

        # Apply default system prompts to models that don't have one
        if "answering_models" in data:
            for model in data["answering_models"]:
                if isinstance(model, ModelConfig) and not model.system_prompt:
                    model.system_prompt = DEFAULT_ANSWERING_SYSTEM_PROMPT
                elif isinstance(model, dict) and not model.get("system_prompt"):
                    model["system_prompt"] = DEFAULT_ANSWERING_SYSTEM_PROMPT

        if "parsing_models" in data:
            for model in data["parsing_models"]:
                if isinstance(model, ModelConfig) and not model.system_prompt:
                    model.system_prompt = DEFAULT_PARSING_SYSTEM_PROMPT
                elif isinstance(model, dict) and not model.get("system_prompt"):
                    model["system_prompt"] = DEFAULT_PARSING_SYSTEM_PROMPT

        super().__init__(**data)

        # Validate configuration after initialization
        self._validate_config()

    def _validate_config(self) -> None:
        """
        Validate configuration, especially for rubric-enabled scenarios.

        Validates that:
        - At least one parsing model is configured
        - At least one answering model is configured (unless parsing_only=True)
        - Required fields are present for each model
        - Model provider is provided for interfaces that require it
        - Rubric-specific requirements are met when enabled

        Raises:
            ValueError: If any validation rule fails
        """
        # Check that we have at least one parsing model (always required)
        if not self.parsing_models:
            raise ValueError("At least one parsing model must be configured")

        # Check answering models only if not in parsing-only mode
        if not self.parsing_only and not self.answering_models:
            raise ValueError("At least one answering model must be configured (unless parsing_only=True)")

        # Validate model configurations
        for model in self.answering_models + self.parsing_models:
            # Model provider is optional for OpenRouter and manual interfaces
            if model.interface not in INTERFACES_NO_PROVIDER_REQUIRED and not model.model_provider:
                raise ValueError(
                    f"Model provider is required for model {model.id} "
                    f"(interface: {model.interface}). Only {INTERFACES_NO_PROVIDER_REQUIRED} "
                    f"interfaces allow empty providers."
                )
            if not model.model_name:
                raise ValueError(f"Model name is required for model {model.id}")
            if not model.system_prompt:
                raise ValueError(f"System prompt is required for model {model.id}")

        # Additional validation for rubric-enabled scenarios
        if self.rubric_enabled:
            # Ensure parsing models are configured since they're needed for rubric evaluation
            if not self.parsing_models:
                raise ValueError("Parsing models are required when rubric evaluation is enabled")

            # Check that replicate count is valid
            if self.replicate_count < 1:
                raise ValueError("Replicate count must be at least 1")

        # Validation for evaluation_mode
        if self.evaluation_mode == "rubric_only":
            # Rubric-only mode requires rubric to be enabled
            if not self.rubric_enabled:
                raise ValueError(
                    "evaluation_mode='rubric_only' requires rubric_enabled=True. "
                    "Rubric-only mode evaluates rubric traits on the raw LLM response without template verification."
                )
        elif self.evaluation_mode == "template_and_rubric":
            # Template + rubric mode requires rubric to be enabled
            if not self.rubric_enabled:
                raise ValueError(
                    "evaluation_mode='template_and_rubric' requires rubric_enabled=True. "
                    "This mode runs both template verification AND rubric evaluation."
                )
        elif self.evaluation_mode == "template_only" and self.rubric_enabled:
            # Template-only mode should not have rubric enabled (warn about configuration mismatch)
            raise ValueError(
                "evaluation_mode='template_only' is incompatible with rubric_enabled=True. "
                "Use evaluation_mode='template_and_rubric' to run both template verification and rubric evaluation, "
                "or set rubric_enabled=False for template-only verification."
            )

        # Additional validation for few-shot prompting scenarios
        # Handle both legacy and new few-shot configurations
        few_shot_config = self.few_shot_config

        # Check legacy fields for backward compatibility
        if (
            few_shot_config is None
            and self.few_shot_enabled
            and self.few_shot_mode == "k-shot"
            and (self.few_shot_k is None or self.few_shot_k < 1)
        ):
            raise ValueError("Few-shot k value must be at least 1 when using k-shot mode")

        # Validate new FewShotConfig if present
        if few_shot_config is not None and few_shot_config.enabled:
            if few_shot_config.global_mode == "k-shot" and few_shot_config.global_k < 1:
                raise ValueError("Global few-shot k value must be at least 1 when using k-shot mode")

            # Validate question-specific k values
            for question_id, question_config in few_shot_config.question_configs.items():
                if question_config.mode == "k-shot" and question_config.k is not None and question_config.k < 1:
                    raise ValueError(
                        f"Question {question_id} few-shot k value must be at least 1 when using k-shot mode"
                    )

        # Additional validation for search-enhanced deep-judgment
        if self.deep_judgment_search_enabled:
            # Validate search tool
            if isinstance(self.deep_judgment_search_tool, str):
                # Check if it's a supported built-in tool
                supported_tools = ["tavily"]
                if self.deep_judgment_search_tool.lower() not in supported_tools:
                    raise ValueError(
                        f"Unknown search tool: '{self.deep_judgment_search_tool}'. Supported tools: {supported_tools}"
                    )
            elif not callable(self.deep_judgment_search_tool):
                raise ValueError(
                    "Search tool must be either a supported tool name string "
                    "or a callable with signature (str | list[str]) -> (str | list[str])"
                )

    def get_few_shot_config(self) -> FewShotConfig | None:
        """
        Get the effective FewShotConfig, handling backward compatibility.

        Returns:
            The FewShotConfig to use, or None if few-shot is disabled
        """
        # Return new config if present
        if self.few_shot_config is not None:
            return self.few_shot_config

        # Handle legacy configuration
        if self.few_shot_enabled:
            few_shot_mode = self.few_shot_mode or "all"
            few_shot_k = self.few_shot_k or 3

            return FewShotConfig(
                enabled=True,
                global_mode=few_shot_mode,
                global_k=few_shot_k,
            )

        return None

    def is_few_shot_enabled(self) -> bool:
        """
        Check if few-shot prompting is enabled (backward compatible).

        Returns:
            True if few-shot is enabled in any format
        """
        config = self.get_few_shot_config()
        return config is not None and config.enabled

    # ===== Preset Utility Class Methods =====
    # These methods provide reusable utilities for preset management
    # and can be used by both core and server implementations.

    @classmethod
    def sanitize_model_config(cls, model: dict[str, Any]) -> dict[str, Any]:
        """
        Sanitize model configuration to remove interface-specific fields.

        This method removes fields that don't apply to a model's interface type,
        ensuring only relevant configuration is saved in presets.

        Note: manual_traces field is automatically excluded (not saved in presets).
        Manual traces must be uploaded separately when using a preset with manual interface.

        Args:
            model: Model configuration dictionary

        Returns:
            Sanitized model configuration with only applicable fields

        Example:
            >>> config = {"interface": "langchain", "endpoint_base_url": "http://..."}
            >>> sanitized = VerificationConfig.sanitize_model_config(config)
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

        return sanitized

    @classmethod
    def sanitize_preset_name(cls, name: str) -> str:
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
            >>> VerificationConfig.sanitize_preset_name("My Test Config!")
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

    @classmethod
    def validate_preset_metadata(cls, name: str, description: str | None = None) -> None:
        """
        Validate preset name and description length limits.

        Note: This method only validates basic metadata constraints (length limits).
        Name uniqueness must be checked separately by the caller (server has the list).

        Args:
            name: Preset name
            description: Optional preset description

        Raises:
            ValueError: If validation fails

        Example:
            >>> VerificationConfig.validate_preset_metadata("Test", "A test preset")
            >>> # Passes validation
            >>> VerificationConfig.validate_preset_metadata("", "Description")
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

    @classmethod
    def create_preset_structure(
        cls,
        preset_id: str,
        name: str,
        description: str | None,
        config_dict: dict[str, Any],
        created_at: str,
        updated_at: str,
    ) -> dict[str, Any]:
        """
        Create standardized preset data structure.

        This method provides a consistent format for preset metadata across
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
            >>> preset = VerificationConfig.create_preset_structure(
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
        self,
        name: str,
        description: str | None = None,
        presets_dir: Path | None = None,
    ) -> dict[str, Any]:
        """
        Save this VerificationConfig as a preset file.

        Args:
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
            >>> metadata = config.save_preset("Quick Test", "Fast testing configuration")
            >>> print(f"Saved to {metadata['filepath']}")
        """
        # Determine presets directory
        if presets_dir is None:
            # Check environment variable first
            env_presets_dir = os.getenv("KARENINA_PRESETS_DIR")
            if env_presets_dir:
                presets_dir = Path(env_presets_dir)
            else:
                # Default to benchmark_presets/ directory in project root
                project_root = Path(__file__).parent.parent.parent.parent.parent.parent
                presets_dir = project_root / "benchmark_presets"

        presets_dir = presets_dir.resolve()

        # Validate metadata using class method
        self.validate_preset_metadata(name, description)

        # Ensure directory exists
        presets_dir.mkdir(parents=True, exist_ok=True)

        # Generate preset ID and timestamps
        preset_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()

        # Convert config to dict and sanitize models
        config_dict = self.model_dump(mode="json")

        # Sanitize answering and parsing models using class method
        if "answering_models" in config_dict:
            config_dict["answering_models"] = [self.sanitize_model_config(m) for m in config_dict["answering_models"]]
        if "parsing_models" in config_dict:
            config_dict["parsing_models"] = [self.sanitize_model_config(m) for m in config_dict["parsing_models"]]

        # Create preset structure using class method
        preset = self.create_preset_structure(
            preset_id=preset_id,
            name=name,
            description=description,
            config_dict=config_dict,
            created_at=now,
            updated_at=now,
        )

        # Generate safe filename using class method
        filename = self.sanitize_preset_name(name)
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

    @classmethod
    def from_preset(cls, filepath: Path) -> "VerificationConfig":
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
            >>> config = VerificationConfig.from_preset(Path("presets/gpt-oss.json"))
            >>> results = verify_questions(checkpoint, config)
        """
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
            return cls(**config_data)
        except Exception as e:
            preset_name = preset.get("name", "unknown")
            raise ValueError(f"Failed to load preset '{preset_name}' from {filepath}: {e}") from e
