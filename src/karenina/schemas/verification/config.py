"""Verification configuration model."""

import contextlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..config.models import (
    INTERFACE_LANGCHAIN,
    INTERFACES_NO_PROVIDER_REQUIRED,
    FewShotConfig,
    ModelConfig,
)
from .config_presets import (
    create_preset_structure,
    load_preset,
    sanitize_model_config,
    sanitize_preset_name,
    save_preset,
    validate_preset_metadata,
)
from .prompt_config import PromptConfig

# Default system prompts for answering and parsing models
DEFAULT_ANSWERING_SYSTEM_PROMPT = "You are an expert assistant. Answer the question accurately and concisely."
DEFAULT_PARSING_SYSTEM_PROMPT = (
    "You are a validation assistant. Parse and validate responses against the given Pydantic template."
)

# Default embedding check settings
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_THRESHOLD = 0.85

# Default async execution settings
DEFAULT_ASYNC_ENABLED = True
DEFAULT_ASYNC_MAX_WORKERS = 2

# Default deep-judgment settings
DEFAULT_DEEP_JUDGMENT_MAX_EXCERPTS = 3
DEFAULT_DEEP_JUDGMENT_FUZZY_THRESHOLD = 0.80
DEFAULT_DEEP_JUDGMENT_RETRY_ATTEMPTS = 2

# Default deep-judgment rubric settings
DEFAULT_RUBRIC_MAX_EXCERPTS = 7


class DeepJudgmentTraitConfig(BaseModel):
    """
    Configuration for deep judgment evaluation of a single rubric trait.

    This model validates trait-level deep judgment settings used in custom mode.
    """

    model_config = ConfigDict(extra="forbid")

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

    # Trace input control: determines what portion of MCP agent trace is passed to evaluation
    use_full_trace_for_template: bool = (
        False  # If True, pass full agent trace to template parsing; if False, extract only final AI message
    )
    use_full_trace_for_rubric: bool = (
        True  # If True, pass full agent trace to rubric evaluation; if False, extract only final AI message
    )
    # Note: The full trace is ALWAYS captured and stored in raw_llm_response regardless of these settings.
    # These flags only control what input is provided to the parsing/evaluation models.
    # If False and the trace doesn't end with an AI message, verification stage will fail with error.

    # Abstention detection settings
    abstention_enabled: bool = False  # Enable abstention/refusal detection

    # Sufficiency detection settings
    sufficiency_enabled: bool = False  # Enable trace sufficiency detection

    # Embedding check settings (semantic similarity fallback)
    embedding_check_enabled: bool = False  # Enable semantic similarity fallback
    embedding_check_model: str = DEFAULT_EMBEDDING_MODEL  # SentenceTransformer model for embeddings
    embedding_check_threshold: float = DEFAULT_EMBEDDING_THRESHOLD  # Similarity threshold (0.0-1.0)

    # Async execution settings
    async_enabled: bool = DEFAULT_ASYNC_ENABLED  # Enable parallel execution
    async_max_workers: int = DEFAULT_ASYNC_MAX_WORKERS  # Number of parallel workers

    # Deep-judgment settings (multi-stage parsing with excerpts and reasoning)
    deep_judgment_enabled: bool = False  # Enable deep-judgment analysis (default: disabled)
    deep_judgment_max_excerpts_per_attribute: int = DEFAULT_DEEP_JUDGMENT_MAX_EXCERPTS  # Max excerpts per attribute
    deep_judgment_fuzzy_match_threshold: float = DEFAULT_DEEP_JUDGMENT_FUZZY_THRESHOLD  # Similarity threshold
    deep_judgment_excerpt_retry_attempts: int = DEFAULT_DEEP_JUDGMENT_RETRY_ATTEMPTS  # Retry attempts

    # Search-enhanced deep-judgment settings (validate excerpts against external evidence)
    deep_judgment_search_enabled: bool = False  # Enable search validation for excerpts
    deep_judgment_search_tool: str | Callable[..., Any] = "tavily"  # Search tool name or callable instance
    # Supported built-in tools: "tavily"
    # Can also pass any callable: (str | list[str]) -> (str | list[str])
    # Examples: langchain tools, MCP tools, custom functions

    # Deep-judgment rubric settings (global defaults for per-trait configuration)
    deep_judgment_rubric_max_excerpts_default: int = DEFAULT_RUBRIC_MAX_EXCERPTS  # Max excerpts per trait
    deep_judgment_rubric_fuzzy_match_threshold_default: float = DEFAULT_DEEP_JUDGMENT_FUZZY_THRESHOLD  # Fuzzy match
    deep_judgment_rubric_excerpt_retry_attempts_default: int = DEFAULT_DEEP_JUDGMENT_RETRY_ATTEMPTS  # Retry attempts
    deep_judgment_rubric_search_tool: str | Callable[..., Any] = (
        "tavily"  # Search tool for rubric hallucination detection
    )

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

    # Per-task-type prompt instructions (optional user-injected instructions for each pipeline stage)
    prompt_config: PromptConfig | None = None

    # Database storage settings
    db_config: Any | None = None  # DBConfig instance for automatic result persistence

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
            # else: let Pydantic use field default (DEFAULT_EMBEDDING_MODEL)

        if "embedding_check_threshold" not in data:
            env_val = os.getenv("EMBEDDING_CHECK_THRESHOLD")
            if env_val is not None:
                # Invalid env var value will let Pydantic use field default (0.85)
                with contextlib.suppress(ValueError):
                    data["embedding_check_threshold"] = float(env_val)
            # else: let Pydantic use field default (DEFAULT_EMBEDDING_THRESHOLD)

        # Read environment variables for async execution settings (only if not explicitly provided AND env var is set)
        if "async_enabled" not in data:
            env_val = os.getenv("KARENINA_ASYNC_ENABLED")
            if env_val is not None:
                data["async_enabled"] = env_val.lower() in ("true", "1", "yes")
            # else: let Pydantic use field default (DEFAULT_ASYNC_ENABLED)

        if "async_max_workers" not in data:
            env_val = os.getenv("KARENINA_ASYNC_MAX_WORKERS")
            if env_val is not None:
                # Invalid env var value will let Pydantic use field default (2)
                with contextlib.suppress(ValueError):
                    data["async_max_workers"] = int(env_val)
            # else: let Pydantic use field default (DEFAULT_ASYNC_MAX_WORKERS)

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
        # Note: Basic model validation (model_name, model_provider) is also done by
        # the adapter factory at runtime, but we validate here too for early failure.
        for model in self.answering_models + self.parsing_models:
            if not model.model_name:
                raise ValueError(f"Model name is required in model configuration (model: {model.id})")
            # Model provider is optional for OpenRouter, manual, and openai_endpoint interfaces
            if model.interface not in INTERFACES_NO_PROVIDER_REQUIRED and not model.model_provider:
                raise ValueError(
                    f"Model provider is required for interface '{model.interface}'. "
                    f"Only {list(INTERFACES_NO_PROVIDER_REQUIRED)} interfaces allow empty providers. "
                    f"(model: {model.id})"
                )
            # System prompt is required for verification (not validated by factory)
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
        if self.few_shot_config is not None and self.few_shot_config.enabled:
            if self.few_shot_config.global_mode == "k-shot" and self.few_shot_config.global_k < 1:
                raise ValueError("Global few-shot k value must be at least 1 when using k-shot mode")

            # Validate question-specific k values
            for question_id, question_config in self.few_shot_config.question_configs.items():
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

    def __repr__(self) -> str:
        """
        Return detailed string representation for debugging/inspection.

        Shows key configuration settings including models, execution parameters,
        and enabled features in a human-readable multi-line format.
        """
        lines = ["VerificationConfig("]

        # === MODELS ===
        lines.append("  === MODELS ===")

        # Answering models
        if self.answering_models:
            lines.append(f"  Answering ({len(self.answering_models)}):")
            for model in self.answering_models:
                provider = model.model_provider or "none"
                lines.append(
                    f"    - {model.model_name} ({provider}) [temp={model.temperature}, interface={model.interface}]"
                )
        else:
            lines.append("  Answering: none")

        # Parsing models
        lines.append(f"  Parsing ({len(self.parsing_models)}):")
        for model in self.parsing_models:
            provider = model.model_provider or "none"
            lines.append(
                f"    - {model.model_name} ({provider}) [temp={model.temperature}, interface={model.interface}]"
            )

        # === EXECUTION ===
        lines.append("")
        lines.append("  === EXECUTION ===")
        lines.append(f"  Replicates: {self.replicate_count}")
        lines.append(f"  Async: {self.async_enabled}")
        if self.async_enabled:
            lines.append(f"    └─ workers: {self.async_max_workers}")
        if self.parsing_only:
            lines.append("  Parsing Only: True")
        lines.append(f"  Evaluation Mode: {self.evaluation_mode}")
        lines.append(f"  Rubric Evaluation Strategy: {self.rubric_evaluation_strategy}")

        # === FEATURES ===
        lines.append("")
        lines.append("  === FEATURES ===")
        features_shown = False

        # Rubric - just enabled/disabled status with optional trait selection
        if self.rubric_enabled:
            features_shown = True
            trait_info = ""
            if self.rubric_trait_names:
                trait_info = f" ({len(self.rubric_trait_names)} traits selected)"
            lines.append(f"  Rubric: enabled{trait_info}")
        else:
            lines.append("  Rubric: disabled")

        # Deep Judgment - Template
        if self.deep_judgment_enabled:
            features_shown = True
            lines.append(
                f"  Deep Judgment (Template): "
                f"max_excerpts={self.deep_judgment_max_excerpts_per_attribute}, "
                f"fuzzy_threshold={self.deep_judgment_fuzzy_match_threshold}"
            )
            if self.deep_judgment_search_enabled:
                search_tool = self.deep_judgment_search_tool
                if callable(search_tool):
                    search_tool = "<custom_callable>"
                lines.append(f"    └─ search: {search_tool}")

        # Deep Judgment - Rubric
        if self.deep_judgment_rubric_mode != "disabled":
            features_shown = True
            lines.append(
                f"  Deep Judgment (Rubric): mode={self.deep_judgment_rubric_mode}, "
                f"global_excerpts={self.deep_judgment_rubric_global_excerpts}"
            )
            # Warning about sequential evaluation
            lines.append("    ⚠️  Deep judgment traits are ALWAYS evaluated sequentially (one-by-one)")
            if self.deep_judgment_rubric_mode == "custom" and self.deep_judgment_rubric_config:
                global_traits = self.deep_judgment_rubric_config.get("global", {})
                question_configs = self.deep_judgment_rubric_config.get("question_specific", {})
                lines.append(f"    └─ {len(global_traits)} global traits, {len(question_configs)} question configs")

        # Abstention
        if self.abstention_enabled:
            features_shown = True
            lines.append("  Abstention: enabled")

        # Sufficiency
        if self.sufficiency_enabled:
            features_shown = True
            lines.append("  Sufficiency: enabled")

        # Embedding Check
        if self.embedding_check_enabled:
            features_shown = True
            lines.append(
                f"  Embedding Check: model={self.embedding_check_model}, threshold={self.embedding_check_threshold}"
            )

        # Few-Shot
        few_shot_config = self.get_few_shot_config()
        if few_shot_config and few_shot_config.enabled:
            features_shown = True
            lines.append(f"  Few-Shot: mode={few_shot_config.global_mode}")
            if few_shot_config.global_mode == "k-shot":
                lines.append(f"    └─ k={few_shot_config.global_k}")
            if few_shot_config.question_configs:
                lines.append(f"    └─ {len(few_shot_config.question_configs)} question configs")

        if not features_shown:
            lines.append("  (none enabled)")

        lines.append(")")

        return "\n".join(lines)

    def __str__(self) -> str:
        """String representation (same as repr for developer-friendly output)."""
        return self.__repr__()

    def get_few_shot_config(self) -> FewShotConfig | None:
        """
        Get the FewShotConfig for this verification run.

        Returns:
            The FewShotConfig to use, or None if few-shot is disabled
        """
        return self.few_shot_config

    def is_few_shot_enabled(self) -> bool:
        """
        Check if few-shot prompting is enabled (backward compatible).

        Returns:
            True if few-shot is enabled in any format
        """
        config = self.get_few_shot_config()
        return config is not None and config.enabled

    # ===== Preset Utility Class Methods =====
    # These methods delegate to config_presets module for backward compatibility.

    @classmethod
    def sanitize_model_config(cls, model: dict[str, Any]) -> dict[str, Any]:
        """Sanitize model configuration. Delegates to config_presets.sanitize_model_config."""
        return sanitize_model_config(model)

    @classmethod
    def sanitize_preset_name(cls, name: str) -> str:
        """Convert preset name to safe filename. Delegates to config_presets.sanitize_preset_name."""
        return sanitize_preset_name(name)

    @classmethod
    def validate_preset_metadata(cls, name: str, description: str | None = None) -> None:
        """Validate preset metadata. Delegates to config_presets.validate_preset_metadata."""
        return validate_preset_metadata(name, description)

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
        """Create preset structure. Delegates to config_presets.create_preset_structure."""
        return create_preset_structure(preset_id, name, description, config_dict, created_at, updated_at)

    def save_preset(
        self,
        name: str,
        description: str | None = None,
        presets_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Save this config as a preset file. Delegates to config_presets.save_preset."""
        return save_preset(self, name, description, presets_dir)

    @classmethod
    def from_preset(cls, filepath: Path) -> "VerificationConfig":
        """Load a VerificationConfig from a preset file. Delegates to config_presets.load_preset."""
        return load_preset(filepath)

    @classmethod
    def from_overrides(
        cls,
        base: "VerificationConfig | None" = None,
        *,
        # Model configuration
        answering_model: str | None = None,
        answering_provider: str | None = None,
        answering_id: str | None = None,
        answering_interface: str | None = None,
        parsing_model: str | None = None,
        parsing_provider: str | None = None,
        parsing_id: str | None = None,
        parsing_interface: str | None = None,
        temperature: float | None = None,
        manual_traces: Any | None = None,
        # Execution settings
        replicate_count: int | None = None,
        # Feature flags
        abstention: bool | None = None,
        sufficiency: bool | None = None,
        embedding_check: bool | None = None,
        deep_judgment: bool | None = None,
        # Evaluation settings
        evaluation_mode: str | None = None,
        embedding_threshold: float | None = None,
        embedding_model: str | None = None,
        async_execution: bool | None = None,
        async_workers: int | None = None,
        # Trace filtering
        use_full_trace_for_template: bool | None = None,
        use_full_trace_for_rubric: bool | None = None,
        # Deep judgment rubric settings
        deep_judgment_rubric_mode: str | None = None,
        deep_judgment_rubric_excerpts: bool | None = None,
        deep_judgment_rubric_max_excerpts: int | None = None,
        deep_judgment_rubric_fuzzy_threshold: float | None = None,
        deep_judgment_rubric_retry_attempts: int | None = None,
        deep_judgment_rubric_search: bool | None = None,
        deep_judgment_rubric_search_tool: str | None = None,
        deep_judgment_rubric_config: dict[str, Any] | None = None,
    ) -> "VerificationConfig":
        """
        Create a VerificationConfig by applying overrides to an optional base config.

        Implements the hierarchy: overrides > base config > defaults.
        Parameters set to None are not applied (base or default value is preserved).

        This is the canonical way to construct a VerificationConfig with selective
        overrides, usable by CLI, server, and programmatic callers.

        Args:
            base: Optional base config (e.g., from a preset). If None, starts from defaults.
            **kwargs: Override values. None means "don't override".

        Returns:
            A new VerificationConfig with overrides applied.
        """
        # Start with base config dump or empty dict
        config_dict: dict[str, Any] = base.model_dump() if base else {}

        # --- Scalar overrides (None = don't override) ---

        # Replicate count
        if replicate_count is not None:
            config_dict["replicate_count"] = replicate_count
        elif not base:
            config_dict["replicate_count"] = 1

        # Feature flags
        if abstention is not None:
            config_dict["abstention_enabled"] = abstention
        if sufficiency is not None:
            config_dict["sufficiency_enabled"] = sufficiency
        if embedding_check is not None:
            config_dict["embedding_check_enabled"] = embedding_check
        if deep_judgment is not None:
            config_dict["deep_judgment_enabled"] = deep_judgment

        # Evaluation settings
        if evaluation_mode is not None:
            config_dict["evaluation_mode"] = evaluation_mode
            config_dict["rubric_enabled"] = evaluation_mode in ["template_and_rubric", "rubric_only"]
        if embedding_threshold is not None:
            config_dict["embedding_check_threshold"] = embedding_threshold
        if embedding_model is not None:
            config_dict["embedding_check_model"] = embedding_model
        if async_execution is not None:
            config_dict["async_enabled"] = async_execution
        if async_workers is not None:
            config_dict["async_max_workers"] = async_workers

        # Trace filtering
        if use_full_trace_for_template is not None:
            config_dict["use_full_trace_for_template"] = use_full_trace_for_template
        if use_full_trace_for_rubric is not None:
            config_dict["use_full_trace_for_rubric"] = use_full_trace_for_rubric

        # Deep judgment rubric settings
        if deep_judgment_rubric_mode is not None:
            config_dict["deep_judgment_rubric_mode"] = deep_judgment_rubric_mode
        if deep_judgment_rubric_excerpts is not None:
            config_dict["deep_judgment_rubric_global_excerpts"] = deep_judgment_rubric_excerpts
        if deep_judgment_rubric_max_excerpts is not None:
            config_dict["deep_judgment_rubric_max_excerpts_default"] = deep_judgment_rubric_max_excerpts
        if deep_judgment_rubric_fuzzy_threshold is not None:
            config_dict["deep_judgment_rubric_fuzzy_match_threshold_default"] = deep_judgment_rubric_fuzzy_threshold
        if deep_judgment_rubric_retry_attempts is not None:
            config_dict["deep_judgment_rubric_excerpt_retry_attempts_default"] = deep_judgment_rubric_retry_attempts
        if deep_judgment_rubric_search is not None:
            config_dict["deep_judgment_rubric_search_enabled"] = deep_judgment_rubric_search
        if deep_judgment_rubric_search_tool is not None:
            config_dict["deep_judgment_rubric_search_tool"] = deep_judgment_rubric_search_tool
        if deep_judgment_rubric_config is not None:
            config_dict["deep_judgment_rubric_config"] = deep_judgment_rubric_config

        # --- Model configuration ---
        # Determine the unified interface (answering and parsing may differ)
        ans_interface = answering_interface
        par_interface = parsing_interface
        # If only a single 'interface' concept was provided via answering_interface,
        # it's already split by the caller. No implicit sharing here.

        answering_has_overrides = any(
            [
                answering_model is not None,
                answering_provider is not None,
                ans_interface is not None,
            ]
        )

        parsing_has_overrides = any(
            [
                parsing_model is not None,
                parsing_provider is not None,
                par_interface is not None,
            ]
        )

        if answering_has_overrides:
            config_dict["answering_models"] = [
                cls._build_model_config_dict(
                    base_models=base.answering_models if base else None,
                    model_name=answering_model,
                    provider=answering_provider,
                    model_id=answering_id,
                    temperature=temperature,
                    interface=ans_interface,
                    manual_traces=manual_traces,
                    default_model="gpt-4.1-mini",
                    default_provider="openai",
                    default_interface="langchain",
                )
            ]
        elif manual_traces is not None:
            # Manual interface requested via manual_traces without explicit model overrides
            config_dict["answering_models"] = [ModelConfig(interface="manual", manual_traces=manual_traces)]

        if parsing_has_overrides:
            config_dict["parsing_models"] = [
                cls._build_model_config_dict(
                    base_models=base.parsing_models if base else None,
                    model_name=parsing_model,
                    provider=parsing_provider,
                    model_id=parsing_id,
                    temperature=temperature,
                    interface=par_interface,
                    manual_traces=None,  # Parsing model never uses manual interface
                    default_model="gpt-4.1-mini",
                    default_provider="openai",
                    default_interface="langchain",
                )
            ]

        return cls(**config_dict)

    @classmethod
    def _build_model_config_dict(
        cls,
        *,
        base_models: list[ModelConfig] | None,
        model_name: str | None,
        provider: str | None,
        model_id: str | None,
        temperature: float | None,
        interface: str | None,
        manual_traces: Any | None,
        default_model: str,
        default_provider: str,
        default_interface: str,
    ) -> ModelConfig:
        """
        Build a ModelConfig by applying overrides to an optional base model.

        If base_models is provided, uses the first model as the starting point and
        applies only non-None overrides. If no base, constructs from scratch with defaults.

        Returns:
            A new ModelConfig instance.
        """
        if interface == "manual" and manual_traces is not None:
            return ModelConfig(interface="manual", manual_traces=manual_traces)

        if base_models:
            # Start from base model, apply overrides
            base_model = base_models[0].model_dump()
            if model_name is not None:
                base_model["model_name"] = model_name
            if provider is not None:
                base_model["model_provider"] = provider
            if model_id is not None:
                base_model["id"] = model_id
            if temperature is not None:
                base_model["temperature"] = temperature
            if interface is not None:
                base_model["interface"] = interface
            return ModelConfig(**base_model)

        # No base — build from scratch
        from typing import Literal, cast

        InterfaceType = Literal[
            "langchain", "openrouter", "manual", "openai_endpoint", "claude_agent_sdk", "claude_tool"
        ]
        final_interface = cast(InterfaceType, interface or default_interface)

        return ModelConfig(
            model_name=model_name or default_model,
            model_provider=provider or default_provider,
            interface=final_interface,
            temperature=temperature if temperature is not None else 0.1,
            id=model_id,
        )
