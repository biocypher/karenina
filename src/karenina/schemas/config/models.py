"""Model configuration and few-shot configuration models."""

from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel, Field, SecretStr, model_validator

if TYPE_CHECKING:
    pass


# ============================================================================
# Agent Middleware Configuration Models
# ============================================================================


class ModelRetryConfig(BaseModel):
    """Configuration for LangChain ModelRetryMiddleware.

    Controls automatic retry behavior for failed model calls with exponential backoff.
    """

    max_retries: int = Field(
        default=2,
        description="Maximum retry attempts (total calls = max_retries + 1 initial)",
    )
    backoff_factor: float = Field(
        default=2.0,
        description="Multiplier for exponential backoff between retries",
    )
    initial_delay: float = Field(
        default=2.0,
        description="Initial delay in seconds before first retry",
    )
    max_delay: float = Field(
        default=10.0,
        description="Maximum delay in seconds between retries",
    )
    jitter: bool = Field(
        default=True,
        description="Add random jitter (±25%) to retry delays",
    )
    on_failure: Literal["continue", "raise"] = Field(
        default="continue",
        description="Behavior when all retries exhausted: 'continue' returns partial response, 'raise' raises exception",
    )


class ToolRetryConfig(BaseModel):
    """Configuration for LangChain ToolRetryMiddleware.

    Controls automatic retry behavior for failed tool calls with exponential backoff.
    """

    max_retries: int = Field(
        default=3,
        description="Maximum retry attempts for tool calls",
    )
    backoff_factor: float = Field(
        default=2.0,
        description="Multiplier for exponential backoff between retries",
    )
    initial_delay: float = Field(
        default=1.0,
        description="Initial delay in seconds before first retry",
    )
    on_failure: Literal["return_message", "raise"] = Field(
        default="return_message",
        description="Behavior when all retries exhausted: 'return_message' returns error as message, 'raise' raises exception",
    )


class SummarizationConfig(BaseModel):
    """Configuration for LangChain SummarizationMiddleware.

    Automatically summarizes conversation history when approaching token limits.
    """

    enabled: bool = Field(
        default=True,
        description="Enable automatic summarization of conversation history (default: True for MCP agents)",
    )
    model: str | None = Field(
        default=None,
        description="Model to use for summarization (defaults to a lightweight model like gpt-4o-mini)",
    )
    trigger_fraction: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Fraction of context window that triggers summarization (0.0-1.0)",
    )

    trigger_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Number of tokens that triggers summarization (optional, overrides trigger_fraction)",
    )

    keep_messages: int = Field(
        default=20,
        ge=1,
        description="Number of recent messages to preserve after summarization",
    )


class PromptCachingConfig(BaseModel):
    """Configuration for Anthropic prompt caching middleware.

    Reduces costs and latency by caching static or repetitive prompt content
    (like system prompts, tool definitions, and conversation history) on Anthropic's servers.

    Only applies when using Anthropic models with the langchain interface.
    See: https://docs.langchain.com/oss/python/integrations/middleware/anthropic#prompt-caching
    """

    enabled: bool = Field(
        default=True,
        description="Enable Anthropic prompt caching (default: True for Anthropic models with MCP tools)",
    )
    ttl: Literal["5m", "1h"] = Field(
        default="5m",
        description="Time to live for cached content. Valid values: '5m' (5 minutes) or '1h' (1 hour)",
    )
    min_messages_to_cache: int = Field(
        default=0,
        ge=0,
        description="Minimum number of messages before caching starts",
    )
    unsupported_model_behavior: Literal["ignore", "warn", "raise"] = Field(
        default="warn",
        description="Behavior when using non-Anthropic models. Options: 'ignore', 'warn', or 'raise'",
    )


class AgentLimitConfig(BaseModel):
    """Configuration for agent execution limits.

    Controls maximum model and tool calls to prevent infinite loops or excessive costs.
    """

    model_call_limit: int = Field(
        default=25,
        ge=1,
        description="Maximum number of LLM calls per agent invocation",
    )
    tool_call_limit: int = Field(
        default=50,
        ge=1,
        description="Maximum number of tool calls per agent invocation",
    )
    exit_behavior: Literal["end", "continue"] = Field(
        default="end",
        description="Behavior when limit reached: 'end' returns partial response gracefully, 'continue' blocks exceeded calls but continues",
    )


class AgentMiddlewareConfig(BaseModel):
    """Complete middleware configuration for MCP-enabled agents.

    Only applies when mcp_urls_dict is provided in ModelConfig.
    Configures retry logic, execution limits, summarization, and prompt caching for agent workflows.
    """

    limits: AgentLimitConfig = Field(
        default_factory=AgentLimitConfig,
        description="Agent execution limits (model/tool call caps)",
    )
    model_retry: ModelRetryConfig = Field(
        default_factory=ModelRetryConfig,
        description="Model call retry configuration",
    )
    tool_retry: ToolRetryConfig = Field(
        default_factory=ToolRetryConfig,
        description="Tool call retry configuration",
    )
    summarization: SummarizationConfig = Field(
        default_factory=SummarizationConfig,
        description="Conversation summarization configuration",
    )
    prompt_caching: PromptCachingConfig = Field(
        default_factory=PromptCachingConfig,
        description="Anthropic prompt caching configuration (only applies to Anthropic models)",
    )


# Interface constants
INTERFACE_OPENROUTER = "openrouter"
INTERFACE_MANUAL = "manual"
INTERFACE_LANGCHAIN = "langchain"
INTERFACE_OPENAI_ENDPOINT = "openai_endpoint"
INTERFACES_NO_PROVIDER_REQUIRED = [INTERFACE_OPENROUTER, INTERFACE_MANUAL, INTERFACE_OPENAI_ENDPOINT]


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

    id: str | None = None  # Optional - defaults to "manual" for manual interface
    model_provider: str | None = None  # Optional - only required for langchain interface
    model_name: str | None = None  # Optional - defaults to "manual" for manual interface
    temperature: float = 0.1
    max_tokens: int = 8192  # Maximum tokens for model response
    interface: Literal["langchain", "openrouter", "manual", "openai_endpoint", "claude_agent_sdk", "claude_tool"] = (
        "langchain"
    )
    system_prompt: str | None = None  # Optional - defaults applied based on context (answering/parsing)
    max_retries: int = 2  # Optional max retries for template generation
    mcp_urls_dict: dict[str, str] | None = None  # Optional MCP server URLs
    mcp_tool_filter: list[str] | None = None  # Optional list of MCP tools to include
    mcp_tool_description_overrides: dict[str, str] | None = (
        None  # Optional tool description overrides for GEPA optimization
    )
    # OpenAI Endpoint configuration (for openai_endpoint interface)
    endpoint_base_url: str | None = None  # Custom endpoint base URL
    endpoint_api_key: SecretStr | None = None  # User-provided API key
    # Extra keyword arguments to pass to the underlying model interface
    # Useful for passing vendor-specific API keys, custom parameters, etc.
    extra_kwargs: dict[str, Any] | None = None
    # Manual interface configuration
    manual_traces: Any = Field(default=None, exclude=True)  # Excluded from serialization; type: ManualTraces | None
    # Agent middleware configuration (only used when mcp_urls_dict is provided)
    # Controls retry behavior, execution limits, and summarization for MCP-enabled agents
    agent_middleware: AgentMiddlewareConfig | None = None
    # Token threshold for triggering summarization middleware.
    # When specified, summarization triggers at exactly this token count.
    # For langchain interface without this value, fraction-based triggering is used (auto-detected from model).
    # For openai_endpoint interface without this value, auto-detected from /v1/models API if available.
    # For openrouter interface without this value, defaults to 100000 * trigger_fraction.
    max_context_tokens: int | None = None

    @model_validator(mode="after")
    def validate_manual_interface(self) -> "ModelConfig":
        """Validate manual interface configuration and set defaults."""
        if self.interface == INTERFACE_MANUAL:
            # Manual interface requires manual_traces
            if self.manual_traces is None:
                raise ValueError(
                    "manual_traces is required when interface='manual'. "
                    "Create a ManualTraces instance and pass it to ModelConfig."
                )

            # Set defaults for manual interface
            if self.id is None:
                self.id = "manual"
            if self.model_name is None:
                self.model_name = "manual"

            # MCP not supported with manual interface
            if self.mcp_urls_dict is not None:
                raise ValueError(
                    "MCP tools are not supported with manual interface. "
                    "Manual traces are precomputed and cannot use dynamic tools."
                )
        else:
            # Non-manual interfaces require id and model_name
            if self.id is None:
                raise ValueError("id is required for non-manual interfaces")
            if self.model_name is None:
                raise ValueError("model_name is required for non-manual interfaces")

        return self
