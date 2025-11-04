"""Verification configuration and result models."""

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .models import (
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

    # Evaluation mode: determines which stages run in the verification pipeline
    evaluation_mode: Literal["template_only", "template_and_rubric", "rubric_only"] = "template_only"
    # - "template_only": Run template verification only (default behavior)
    # - "template_and_rubric": Run both template verification AND rubric evaluation
    # - "rubric_only": Skip template verification, only evaluate rubrics on raw LLM response

    # Abstention detection settings
    abstention_enabled: bool = False  # Enable abstention/refusal detection

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
        """Initialize with backward compatibility for single model configs."""
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
        import re

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
        import os
        import uuid
        from datetime import UTC, datetime

        # Determine presets directory
        if presets_dir is None:
            # Check environment variable first
            env_presets_dir = os.getenv("KARENINA_PRESETS_DIR")
            if env_presets_dir:
                presets_dir = Path(env_presets_dir)
            else:
                # Default to benchmark_presets/ directory in project root
                project_root = Path(__file__).parent.parent.parent.parent.parent
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
            >>> config = VerificationConfig.from_preset(Path("benchmark_presets/quick-test.json"))
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

        # Extract config
        config_data = preset.get("config")
        if not config_data:
            raise ValueError(f"Preset file '{filepath}' has no configuration data")

        # Create VerificationConfig instance from the config dict
        try:
            return cls(**config_data)
        except Exception as e:
            preset_name = preset.get("name", "unknown")
            raise ValueError(f"Failed to load preset '{preset_name}' from {filepath}: {e}") from e


class VerificationResult(BaseModel):
    """Result of verifying a single question."""

    question_id: str
    template_id: str  # MD5 of template or "no_template" (composite key component)
    completed_without_errors: bool
    error: str | None = None

    # Raw data
    question_text: str
    raw_llm_response: str
    parsed_gt_response: dict[str, Any] | None = None  # Ground truth from 'correct' field
    parsed_llm_response: dict[str, Any] | None = None  # LLM extracted fields (excluding 'id' and 'correct')

    # Verification outcomes
    template_verification_performed: bool = False  # Whether template verification was executed
    verify_result: Any | None = None  # Template verification result (None if template verification skipped)
    verify_granular_result: Any | None = None
    rubric_evaluation_performed: bool = False  # Whether rubric evaluation was executed
    verify_rubric: dict[str, int | bool] | None = None  # Rubric trait scores (LLM and manual)
    evaluation_rubric: dict[str, Any] | None = None  # The merged rubric used for evaluation

    # Question metadata
    keywords: list[str] | None = None  # Keywords associated with the question

    # Metadata
    answering_model: str
    parsing_model: str
    execution_time: float
    timestamp: str
    answering_system_prompt: str | None = None
    parsing_system_prompt: str | None = None

    # Run identification
    run_name: str | None = None
    job_id: str | None = None

    # Replicate tracking
    answering_replicate: int | None = None  # Replicate number for answering model (1, 2, 3, ...)
    parsing_replicate: int | None = None  # Replicate number for parsing model (1, 2, 3, ...)

    # Embedding check metadata
    embedding_check_performed: bool = False  # Whether embedding check was attempted
    embedding_similarity_score: float | None = None  # Similarity score (0.0 to 1.0)
    embedding_override_applied: bool = False  # Whether embedding check overrode the result
    embedding_model_used: str | None = None  # Name of the embedding model used

    # Regex validation metadata
    regex_validations_performed: bool = False  # Whether regex validation was attempted
    regex_validation_results: dict[str, bool] | None = None  # Individual regex pattern results
    regex_validation_details: dict[str, dict[str, Any]] | None = None  # Detailed regex match information
    regex_overall_success: bool | None = None  # Overall regex validation result
    regex_extraction_results: dict[str, Any] | None = None  # What the regex patterns actually extracted
    # Recursion limit metadata
    recursion_limit_reached: bool = False  # Whether agent hit recursion limit

    # Abstention detection metadata
    abstention_check_performed: bool = False  # Whether abstention check was attempted
    abstention_detected: bool | None = None  # Whether model refused/abstained from answering
    abstention_override_applied: bool = False  # Whether abstention check overrode the result
    abstention_reasoning: str | None = None  # LLM's reasoning for abstention determination

    # MCP server metadata
    answering_mcp_servers: list[str] | None = None  # Names of MCP servers attached to answering model

    # Deep-judgment metadata (multi-stage parsing with excerpts and reasoning)
    deep_judgment_enabled: bool = False  # Whether deep-judgment was configured
    deep_judgment_performed: bool = False  # Whether deep-judgment was successfully executed
    extracted_excerpts: dict[str, list[dict[str, Any]]] | None = None  # Extracted excerpts per attribute
    # Structure: {"attribute_name": [{"text": str, "confidence": "low|medium|high", "similarity_score": float,
    #   "search_results"?: str, "hallucination_risk"?: "none|low|medium|high", "hallucination_justification"?: str}]}
    # Empty list [] indicates no excerpts found for that attribute (e.g., refusals, no corroborating evidence)
    # When search is enabled, each excerpt includes:
    #   - "search_results": External validation text from search tool
    #   - "hallucination_risk": Per-excerpt risk assessment (none/low/medium/high)
    #   - "hallucination_justification": Explanation for the risk level
    attribute_reasoning: dict[str, str] | None = None  # Reasoning traces per attribute
    # Structure: {"attribute_name": "reasoning text"}
    # Reasoning can exist even when excerpts are empty (explains why no excerpts found)
    deep_judgment_stages_completed: list[str] | None = None  # Stages completed: ["excerpts", "reasoning", "parameters"]
    deep_judgment_model_calls: int = 0  # Number of LLM invocations for deep-judgment
    deep_judgment_excerpt_retry_count: int = 0  # Number of retries for excerpt validation
    attributes_without_excerpts: list[str] | None = None  # Attributes with no corroborating excerpts

    # Search-enhanced deep-judgment metadata
    deep_judgment_search_enabled: bool = False  # Whether search enhancement was enabled for this verification
    hallucination_risk_assessment: dict[str, str] | None = None  # Hallucination risk per attribute
    # Structure: {"attribute_name": "none" | "low" | "medium" | "high"}
    # Scale: "none" = lowest risk (strong external evidence), "high" = highest risk (weak/no evidence)
    # Only populated when deep_judgment_search_enabled=True

    # LLM usage tracking metadata
    usage_metadata: dict[str, dict[str, Any]] | None = None  # Token usage breakdown by verification stage
    # Structure: {
    #   "answer_generation": {
    #     "input_tokens": 150, "output_tokens": 200, "total_tokens": 350,
    #     "model": "gpt-4.1-mini-2025-04-14",
    #     "input_token_details": {"audio": 0, "cache_read": 0},
    #     "output_token_details": {"audio": 0, "reasoning": 0}
    #   },
    #   "parsing": {...}, "rubric_evaluation": {...}, "abstention_check": {...},
    #   "total": {"input_tokens": 600, "output_tokens": 360, "total_tokens": 960}
    # }
    agent_metrics: dict[str, Any] | None = None  # MCP agent execution metrics (only if agent used)
    # Structure: {
    #   "iterations": 3,  # Number of agent think-act cycles
    #   "tool_calls": 5,  # Total tool invocations
    #   "tools_used": ["mcp__brave_search", "mcp__read_resource"],  # Unique tool names used
    #   "suspect_failed_tool_calls": 2,  # Count of tool calls with error-like output patterns
    #   "suspect_failed_tools": ["mcp__brave_search"]  # List of tools with suspected failures
    # }

    # Metric trait evaluation metadata (confusion-matrix analysis)
    metric_trait_confusion_lists: dict[str, dict[str, list[str]]] | None = None  # Confusion lists per metric trait
    # Structure: {"trait_name": {"tp": ["excerpt1", ...], "tn": [...], "fp": [...], "fn": [...]}}
    # Each trait has four lists (TP/TN/FP/FN) containing extracted excerpts from the answer
    metric_trait_metrics: dict[str, dict[str, float]] | None = None  # Computed metrics per metric trait
    # Structure: {"trait_name": {"precision": 0.85, "recall": 0.92, "f1": 0.88, ...}}
    # Only metrics requested by the trait are included in the inner dictionary


class VerificationJob(BaseModel):
    """Represents a verification job."""

    job_id: str
    run_name: str  # User-defined or auto-generated run name
    status: Literal["pending", "running", "completed", "failed", "cancelled"]
    config: VerificationConfig

    # Database storage
    storage_url: str | None = None  # Database URL for auto-save functionality
    benchmark_name: str | None = None  # Benchmark name for auto-save functionality

    # Progress tracking
    total_questions: int
    processed_count: int = 0
    successful_count: int = 0
    failed_count: int = 0
    percentage: float = 0.0
    current_question: str = ""
    last_task_duration: float | None = None  # Execution time of last completed task

    # WebSocket streaming progress fields
    in_progress_questions: list[str] = Field(default_factory=list)

    # Task timing tracking (maps question_id to start time)
    task_start_times: dict[str, float] = Field(default_factory=dict)

    # Timing
    start_time: float | None = None
    end_time: float | None = None

    # Results
    results: dict[str, VerificationResult] = Field(default_factory=dict)
    error_message: str | None = None

    def task_started(self, question_id: str) -> None:
        """Mark a task as started and record start time."""
        import time

        if question_id not in self.in_progress_questions:
            self.in_progress_questions.append(question_id)

        # Record task start time
        self.task_start_times[question_id] = time.time()

    def task_finished(self, question_id: str, success: bool) -> None:
        """Mark a task as finished, calculate duration, and update counts."""
        import time

        # Calculate task duration from recorded start time
        task_duration = 0.0
        if question_id in self.task_start_times:
            task_duration = time.time() - self.task_start_times[question_id]
            # Clean up start time
            del self.task_start_times[question_id]

        # Remove from in-progress list
        if question_id in self.in_progress_questions:
            self.in_progress_questions.remove(question_id)

        # Update counts
        self.processed_count += 1
        if success:
            self.successful_count += 1
        else:
            self.failed_count += 1

        # Update percentage
        self.percentage = (self.processed_count / self.total_questions) * 100 if self.total_questions > 0 else 0.0

        # Track last task duration
        self.last_task_duration = task_duration

    def to_dict(self) -> dict[str, Any]:
        """Convert job to dictionary for API response."""
        # Calculate duration if job has started
        duration = None
        if self.start_time:
            if self.end_time:
                duration = self.end_time - self.start_time  # Completed
            else:
                import time

                duration = time.time() - self.start_time  # In progress

        return {
            "job_id": self.job_id,
            "run_name": self.run_name,
            "status": self.status,
            "total_questions": self.total_questions,
            "processed_count": self.processed_count,
            "successful_count": self.successful_count,
            "failed_count": self.failed_count,
            "percentage": self.percentage,
            "current_question": self.current_question,
            "duration_seconds": duration,
            "last_task_duration": self.last_task_duration,
            "error_message": self.error_message,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "in_progress_questions": self.in_progress_questions,
        }


class FinishedTemplate(BaseModel):
    """Metadata for a finished answer template."""

    question_id: str
    question_text: str
    question_preview: str  # Truncated version for UI
    template_code: str
    last_modified: str
    finished: bool = True
    question_rubric: dict[str, Any] | None = None  # Question-specific rubric as dict
    keywords: list[str] | None = None  # Keywords associated with the question
    few_shot_examples: list[dict[str, str]] | None = None  # Few-shot examples for this question


class VerificationRequest(BaseModel):
    """Request to start verification."""

    config: VerificationConfig
    question_ids: list[str] | None = None  # If None, verify all finished templates
    run_name: str | None = None  # Optional user-defined run name


class VerificationStatusResponse(BaseModel):
    """Response for verification status."""

    job_id: str
    run_name: str
    status: str
    percentage: float
    current_question: str
    processed_count: int
    total_count: int
    successful_count: int
    failed_count: int
    duration_seconds: float | None = None
    last_task_duration: float | None = None
    error: str | None = None
    results: dict[str, VerificationResult] | None = None


class VerificationStartResponse(BaseModel):
    """Response when starting verification."""

    job_id: str
    run_name: str
    status: str
    message: str
