"""Data models for benchmark verification system."""

from typing import Any, Literal, cast

from pydantic import BaseModel, Field

from ..utils.async_utils import AsyncConfig

# Interface constants
INTERFACE_OPENROUTER = "openrouter"
INTERFACE_MANUAL = "manual"
INTERFACE_LANGCHAIN = "langchain"
INTERFACES_NO_PROVIDER_REQUIRED = [INTERFACE_OPENROUTER, INTERFACE_MANUAL]


class QuestionFewShotConfig(BaseModel):
    """Per-question few-shot configuration."""

    mode: Literal["all", "k-shot", "custom", "none", "inherit"] = "inherit"
    k: int | None = None  # Override global k for this question
    selected_examples: list[str | int] | None = None  # Hash (MD5) or index selection
    external_examples: list[dict[str, str]] | None = None  # Question-specific external examples
    excluded_examples: list[str | int] | None = None  # Examples to exclude


class FewShotConfig(BaseModel):
    """Flexible configuration for few-shot prompting with convenient bulk setup interface."""

    # Global fallback settings
    global_mode: Literal["all", "k-shot", "custom", "none"] = "all"
    global_k: int = 3  # Default number of examples for k-shot mode
    enabled: bool = True  # Master enable/disable switch

    # Per-question configurations (auto-generated from convenient interfaces)
    question_configs: dict[str, QuestionFewShotConfig] = Field(default_factory=dict)

    # Global external examples that can be used across all questions
    global_external_examples: list[dict[str, str]] = Field(default_factory=list)

    @classmethod
    def from_index_selections(
        cls,
        selections: dict[str, list[int]],
        global_mode: Literal["all", "k-shot", "custom", "none"] = "custom",
        **kwargs: Any,
    ) -> "FewShotConfig":
        """
        Create FewShotConfig from question_id -> [example_indices] mapping.

        Args:
            selections: Dictionary mapping question IDs to lists of example indices
            global_mode: Global mode to use (defaults to "custom")
            **kwargs: Additional arguments passed to FewShotConfig constructor

        Returns:
            Configured FewShotConfig instance

        Example:
            config = FewShotConfig.from_index_selections({
                "question_1": [0, 2, 4],  # Use examples at indices 0, 2, 4
                "question_2": [1, 3],     # Use examples at indices 1, 3
            })
        """
        question_configs = {}
        for question_id, indices in selections.items():
            question_configs[question_id] = QuestionFewShotConfig(
                mode="custom", selected_examples=cast(list[str | int], indices)
            )

        return cls(global_mode=global_mode, question_configs=question_configs, **kwargs)

    @classmethod
    def from_hash_selections(
        cls,
        selections: dict[str, list[str]],
        global_mode: Literal["all", "k-shot", "custom", "none"] = "custom",
        **kwargs: Any,
    ) -> "FewShotConfig":
        """
        Create FewShotConfig from question_id -> [example_hashes] mapping.

        Args:
            selections: Dictionary mapping question IDs to lists of MD5 hashes
            global_mode: Global mode to use (defaults to "custom")
            **kwargs: Additional arguments passed to FewShotConfig constructor

        Returns:
            Configured FewShotConfig instance

        Example:
            config = FewShotConfig.from_hash_selections({
                "question_1": ["abc123def456...", "ghi789jkl012..."],
                "question_2": ["mno345pqr678..."],
            })
        """
        question_configs = {}
        for question_id, hashes in selections.items():
            question_configs[question_id] = QuestionFewShotConfig(
                mode="custom", selected_examples=cast(list[str | int], hashes)
            )

        return cls(global_mode=global_mode, question_configs=question_configs, **kwargs)

    @classmethod
    def k_shot_for_questions(
        cls,
        question_k_mapping: dict[str, int],
        global_k: int = 3,
        **kwargs: Any,
    ) -> "FewShotConfig":
        """
        Create FewShotConfig with different k values per question.

        Args:
            question_k_mapping: Dictionary mapping question IDs to k values
            global_k: Global k value for questions not in mapping
            **kwargs: Additional arguments passed to FewShotConfig constructor

        Returns:
            Configured FewShotConfig instance

        Example:
            config = FewShotConfig.k_shot_for_questions({
                "question_1": 5,  # Use 5 examples
                "question_2": 2,  # Use 2 examples
            })
        """
        question_configs = {}
        for question_id, k_value in question_k_mapping.items():
            question_configs[question_id] = QuestionFewShotConfig(mode="k-shot", k=k_value)

        return cls(global_mode="k-shot", global_k=global_k, question_configs=question_configs, **kwargs)

    def add_selections_by_index(self, selections: dict[str, list[int]]) -> None:
        """
        Add/update question selections using indices.

        Args:
            selections: Dictionary mapping question IDs to lists of example indices
        """
        for question_id, indices in selections.items():
            self.question_configs[question_id] = QuestionFewShotConfig(
                mode="custom", selected_examples=cast(list[str | int], indices)
            )

    def add_selections_by_hash(self, selections: dict[str, list[str]]) -> None:
        """
        Add/update question selections using hashes.

        Args:
            selections: Dictionary mapping question IDs to lists of MD5 hashes
        """
        for question_id, hashes in selections.items():
            self.question_configs[question_id] = QuestionFewShotConfig(
                mode="custom", selected_examples=cast(list[str | int], hashes)
            )

    def add_k_shot_configs(self, question_k_mapping: dict[str, int]) -> None:
        """
        Add/update k-shot configurations for specific questions.

        Args:
            question_k_mapping: Dictionary mapping question IDs to k values
        """
        for question_id, k_value in question_k_mapping.items():
            self.question_configs[question_id] = QuestionFewShotConfig(mode="k-shot", k=k_value)

    def get_effective_config(self, question_id: str) -> QuestionFewShotConfig:
        """
        Get the effective configuration for a specific question, resolving inheritance.

        Args:
            question_id: The question ID to get configuration for

        Returns:
            Effective QuestionFewShotConfig with inheritance resolved
        """
        question_config = self.question_configs.get(question_id, QuestionFewShotConfig())

        if question_config.mode == "inherit":
            # Inherit from global settings
            return QuestionFewShotConfig(
                mode=self.global_mode,
                k=self.global_k,
                selected_examples=question_config.selected_examples,
                external_examples=question_config.external_examples,
                excluded_examples=question_config.excluded_examples,
            )

        # Use question-specific config, filling in defaults where needed
        return QuestionFewShotConfig(
            mode=question_config.mode,
            k=question_config.k if question_config.k is not None else self.global_k,
            selected_examples=question_config.selected_examples,
            external_examples=question_config.external_examples,
            excluded_examples=question_config.excluded_examples,
        )

    def resolve_examples_for_question(
        self,
        question_id: str,
        available_examples: list[dict[str, str]] | None = None,
        question_text: str | None = None,
    ) -> list[dict[str, str]]:
        """
        Resolve the final list of examples to use for a specific question.

        Args:
            question_id: The question ID to resolve examples for
            available_examples: Available examples from the question (can be None)
            question_text: Question text (used for hash validation, optional)

        Returns:
            List of resolved examples to use for few-shot prompting
        """
        import hashlib
        import random

        if not self.enabled:
            return []

        effective_config = self.get_effective_config(question_id)

        if effective_config.mode == "none":
            return []

        # Start with available examples or empty list
        if available_examples is None:
            available_examples = []

        # Create lookup for hash-to-example mapping if needed
        example_hash_map = {}
        if question_text is not None and available_examples:
            for i, example in enumerate(available_examples):
                example_question = example.get("question", "")
                example_hash = hashlib.md5(example_question.encode("utf-8")).hexdigest()
                example_hash_map[example_hash] = i

        resolved_examples = []

        # Handle different modes
        if effective_config.mode == "all":
            resolved_examples = available_examples.copy()

        elif effective_config.mode == "k-shot":
            k = effective_config.k if effective_config.k is not None else self.global_k
            if len(available_examples) <= k:
                resolved_examples = available_examples.copy()
            else:
                # Randomly sample k examples for consistency
                # Use question_id as seed for reproducible results
                random.seed(hash(question_id) & 0x7FFFFFFF)
                resolved_examples = random.sample(available_examples, k)

        elif effective_config.mode == "custom" and effective_config.selected_examples:
            for selection in effective_config.selected_examples:
                if isinstance(selection, int):
                    # Index-based selection
                    if 0 <= selection < len(available_examples):
                        resolved_examples.append(available_examples[selection])
                elif isinstance(selection, str) and selection in example_hash_map:
                    # Hash-based selection
                    idx = example_hash_map[selection]
                    resolved_examples.append(available_examples[idx])

        # Apply exclusions if specified
        if effective_config.excluded_examples:
            exclusion_indices = set()
            for exclusion in effective_config.excluded_examples:
                if isinstance(exclusion, int):
                    exclusion_indices.add(exclusion)
                elif isinstance(exclusion, str) and exclusion in example_hash_map:
                    exclusion_indices.add(example_hash_map[exclusion])

            # Filter out excluded examples
            resolved_examples = [ex for i, ex in enumerate(resolved_examples) if i not in exclusion_indices]

        # Add external examples (question-specific first, then global)
        final_examples = resolved_examples.copy()

        if effective_config.external_examples:
            final_examples.extend(effective_config.external_examples)

        if self.global_external_examples:
            final_examples.extend(self.global_external_examples)

        return final_examples

    @staticmethod
    def generate_example_hash(question_text: str) -> str:
        """
        Generate MD5 hash for example question text.

        Args:
            question_text: The question text to hash

        Returns:
            32-character MD5 hash string
        """
        import hashlib

        return hashlib.md5(question_text.encode("utf-8")).hexdigest()

    def validate_selections(self, question_examples: dict[str, list[dict[str, str]]]) -> list[str]:
        """
        Validate that all selections reference valid examples.

        Args:
            question_examples: Dictionary mapping question IDs to their available examples

        Returns:
            List of validation error messages (empty if all valid)
        """
        errors = []

        for question_id, config in self.question_configs.items():
            available_examples = question_examples.get(question_id, [])
            max_index = len(available_examples) - 1

            # Create hash lookup for this question
            example_hashes = set()
            for example in available_examples:
                example_question = example.get("question", "")
                hash_value = self.generate_example_hash(example_question)
                example_hashes.add(hash_value)

            # Check selected_examples
            if config.selected_examples:
                for selection in config.selected_examples:
                    if isinstance(selection, int):
                        if selection < 0 or selection > max_index:
                            errors.append(
                                f"Question {question_id}: Index {selection} out of range (available: 0-{max_index})"
                            )
                    elif isinstance(selection, str) and selection not in example_hashes:
                        errors.append(f"Question {question_id}: Hash {selection} not found in available examples")

            # Check excluded_examples
            if config.excluded_examples:
                for exclusion in config.excluded_examples:
                    if isinstance(exclusion, int):
                        if exclusion < 0 or exclusion > max_index:
                            errors.append(
                                f"Question {question_id}: Exclusion index {exclusion} out of range "
                                f"(available: 0-{max_index})"
                            )
                    elif isinstance(exclusion, str) and exclusion not in example_hashes:
                        errors.append(
                            f"Question {question_id}: Exclusion hash {exclusion} not found in available examples"
                        )

        return errors


class ModelConfig(BaseModel):
    """Configuration for a single model."""

    id: str
    model_provider: str
    model_name: str
    temperature: float = 0.1
    interface: Literal["langchain", "openrouter", "manual"] = "langchain"
    system_prompt: str
    max_retries: int = 2  # Optional max retries for template generation
    mcp_urls_dict: dict[str, str] | None = None  # Optional MCP server URLs
    mcp_tool_filter: list[str] | None = None  # Optional list of MCP tools to include


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
                model_provider=data.get("answering_model_provider", ""),
                model_name=data.get("answering_model_name", ""),
                temperature=data.get("answering_temperature", 0.1),
                interface=data.get("answering_interface", INTERFACE_LANGCHAIN),
                system_prompt=data.get(
                    "answering_system_prompt",
                    "You are an expert assistant. Answer the question accurately and concisely.",
                ),
            )
            data["answering_models"] = [answering_model]

        if "parsing_models" not in data and any(k.startswith("parsing_") for k in data):
            parsing_model = ModelConfig(
                id="parsing-legacy",
                model_provider=data.get("parsing_model_provider", ""),
                model_name=data.get("parsing_model_name", ""),
                temperature=data.get("parsing_temperature", 0.1),
                interface=data.get("parsing_interface", INTERFACE_LANGCHAIN),
                system_prompt=data.get(
                    "parsing_system_prompt",
                    "You are a validation assistant. Parse and validate responses against the given Pydantic template.",
                ),
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
    verify_result: Any | None = None
    verify_granular_result: Any | None = None
    verify_rubric: dict[str, int | bool] | None = None  # Rubric trait scores
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


class VerificationJob(BaseModel):
    """Represents a verification job."""

    job_id: str
    run_name: str  # User-defined or auto-generated run name
    status: Literal["pending", "running", "completed", "failed", "cancelled"]
    config: VerificationConfig
    async_config: AsyncConfig | None = None

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
    estimated_time_remaining: float | None = None

    # Timing
    start_time: float | None = None
    end_time: float | None = None

    # Results
    results: dict[str, VerificationResult] = Field(default_factory=dict)
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert job to dictionary for API response."""
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
            "estimated_time_remaining": self.estimated_time_remaining,
            "error_message": self.error_message,
            "start_time": self.start_time,
            "end_time": self.end_time,
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
    estimated_time_remaining: float | None = None
    error: str | None = None
    results: dict[str, VerificationResult] | None = None


class VerificationStartResponse(BaseModel):
    """Response when starting verification."""

    job_id: str
    run_name: str
    status: str
    message: str
