"""Stage-based verification pipeline architecture.

This module provides the core abstractions for the modular verification pipeline:
- VerificationContext: Shared state and artifacts across stages
- VerificationStage: Protocol defining stage interface
- StageRegistry: Manages stage instances and dependencies
- ArtifactKeys: Type-safe constants for artifact and result field keys

Logging Convention:
    All stages should follow this logging level convention:
    - ERROR: System/code errors (exceptions, failures, unexpected states)
    - WARNING: Auto-fails and overrides (anything that changes verify_result due
               to a detected condition like abstention, recursion limit, etc.)
    - INFO: Important flow events (stage completion, feature enabled, etc.)
    - DEBUG: Normal flow decisions (skipping stages, validation passed, etc.)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from .....schemas.domain import Rubric
from .....schemas.verification import PromptConfig
from .....schemas.workflow import ModelConfig

if TYPE_CHECKING:
    from ...utils.trace_usage_tracker import UsageTracker

logger = logging.getLogger(__name__)


class ArtifactKeys:
    """
    Type-safe constants for artifact and result field keys.

    Using these constants instead of raw strings prevents typos and enables
    IDE autocomplete. All stage implementations should use these constants
    when calling context.set_artifact(), context.get_artifact(),
    context.set_result_field(), and context.get_result_field().

    Naming conventions:
    - SCREAMING_SNAKE_CASE for constants
    - Group by functional area (template, rubric, deep judgment, etc.)
    """

    # ==========================================================================
    # Core Pipeline Artifacts
    # ==========================================================================

    # LLM Response & Parsing
    RAW_LLM_RESPONSE = "raw_llm_response"
    PARSED_ANSWER = "parsed_answer"
    USAGE_TRACKER = "usage_tracker"
    TRACE_MESSAGES = "trace_messages"

    # Template Classes (from validate_template stage)
    ANSWER = "Answer"
    RAW_ANSWER = "RawAnswer"
    TEMPLATE_VALIDATION_ERROR = "template_validation_error"
    TEMPLATE_EVALUATOR = "template_evaluator"

    # Model Identification
    ANSWERING_MODEL_STR = "answering_model_str"
    PARSING_MODEL_STR = "parsing_model_str"
    ANSWERING_MCP_SERVERS = "answering_mcp_servers"

    # ==========================================================================
    # Verification Results
    # ==========================================================================

    # Core Verification
    VERIFY_RESULT = "verify_result"
    VERIFY_GRANULAR_RESULT = "verify_granular_result"
    FIELD_VERIFICATION_RESULT = "field_verification_result"
    FINAL_RESULT = "final_result"

    # Regex Verification
    REGEX_VERIFICATION_RESULTS = "regex_verification_results"
    REGEX_EXTRACTION_RESULTS = "regex_extraction_results"
    REGEX_VALIDATIONS_PERFORMED = "regex_validations_performed"
    REGEX_VALIDATION_RESULTS = "regex_validation_results"
    REGEX_VALIDATION_DETAILS = "regex_validation_details"
    REGEX_OVERALL_SUCCESS = "regex_overall_success"

    # ==========================================================================
    # Auto-Fail & Trace Validation
    # ==========================================================================

    # Recursion Limit
    RECURSION_LIMIT_REACHED = "recursion_limit_reached"

    # Trace Validation
    MCP_ENABLED = "mcp_enabled"
    TRACE_VALIDATION_FAILED = "trace_validation_failed"
    TRACE_VALIDATION_ERROR = "trace_validation_error"
    TRACE_EXTRACTION_ERROR = "trace_extraction_error"

    # ==========================================================================
    # Abstention & Sufficiency Detection
    # ==========================================================================

    # Abstention
    ABSTENTION_CHECK_PERFORMED = "abstention_check_performed"
    ABSTENTION_DETECTED = "abstention_detected"
    ABSTENTION_OVERRIDE_APPLIED = "abstention_override_applied"
    ABSTENTION_REASONING = "abstention_reasoning"

    # Sufficiency
    SUFFICIENCY_CHECK_PERFORMED = "sufficiency_check_performed"
    SUFFICIENCY_DETECTED = "sufficiency_detected"
    SUFFICIENCY_OVERRIDE_APPLIED = "sufficiency_override_applied"
    SUFFICIENCY_REASONING = "sufficiency_reasoning"

    # ==========================================================================
    # Embedding Check
    # ==========================================================================

    EMBEDDING_CHECK_PERFORMED = "embedding_check_performed"
    EMBEDDING_SIMILARITY_SCORE = "embedding_similarity_score"
    EMBEDDING_OVERRIDE_APPLIED = "embedding_override_applied"
    EMBEDDING_MODEL_USED = "embedding_model_used"

    # ==========================================================================
    # Deep Judgment (Template)
    # ==========================================================================

    DEEP_JUDGMENT_ENABLED = "deep_judgment_enabled"
    DEEP_JUDGMENT_PERFORMED = "deep_judgment_performed"
    EXTRACTED_EXCERPTS = "extracted_excerpts"
    ATTRIBUTE_REASONING = "attribute_reasoning"
    DEEP_JUDGMENT_STAGES_COMPLETED = "deep_judgment_stages_completed"
    DEEP_JUDGMENT_MODEL_CALLS = "deep_judgment_model_calls"
    DEEP_JUDGMENT_EXCERPT_RETRY_COUNT = "deep_judgment_excerpt_retry_count"
    ATTRIBUTES_WITHOUT_EXCERPTS = "attributes_without_excerpts"
    DEEP_JUDGMENT_SEARCH_ENABLED = "deep_judgment_search_enabled"
    HALLUCINATION_RISK_ASSESSMENT = "hallucination_risk_assessment"

    # ==========================================================================
    # Rubric Evaluation
    # ==========================================================================

    RUBRIC_RESULT = "rubric_result"
    VERIFY_RUBRIC = "verify_rubric"
    LLM_TRAIT_LABELS = "llm_trait_labels"
    METRIC_CONFUSION_LISTS = "metric_confusion_lists"
    METRIC_RESULTS = "metric_results"
    METRIC_TRAIT_CONFUSION_LISTS = "metric_trait_confusion_lists"
    METRIC_TRAIT_METRICS = "metric_trait_metrics"
    RUBRIC_EVALUATION_STRATEGY = "rubric_evaluation_strategy"

    # ==========================================================================
    # Deep Judgment (Rubric)
    # ==========================================================================

    DEEP_JUDGMENT_RUBRIC_PERFORMED = "deep_judgment_rubric_performed"
    EXTRACTED_RUBRIC_EXCERPTS = "extracted_rubric_excerpts"
    RUBRIC_TRAIT_REASONING = "rubric_trait_reasoning"
    DEEP_JUDGMENT_RUBRIC_SCORES = "deep_judgment_rubric_scores"
    STANDARD_RUBRIC_SCORES = "standard_rubric_scores"
    TRAIT_METADATA = "trait_metadata"
    TRAITS_WITHOUT_VALID_EXCERPTS = "traits_without_valid_excerpts"
    RUBRIC_HALLUCINATION_RISK_ASSESSMENT = "rubric_hallucination_risk_assessment"
    TOTAL_DEEP_JUDGMENT_MODEL_CALLS = "total_deep_judgment_model_calls"
    TOTAL_TRAITS_EVALUATED = "total_traits_evaluated"
    TOTAL_EXCERPT_RETRIES = "total_excerpt_retries"

    # ==========================================================================
    # Trace Filtering & Evaluation Input
    # ==========================================================================

    USED_FULL_TRACE = "used_full_trace"
    EVALUATION_INPUT = "evaluation_input"

    # ==========================================================================
    # Result Metadata
    # ==========================================================================

    TIMESTAMP = "timestamp"
    EXECUTION_TIME = "execution_time"


@dataclass
class VerificationContext:
    """
    Shared context and state for verification pipeline stages.

    This context flows through all stages, accumulating artifacts and building
    the final VerificationResult. Stages read from and write to this context.

    Attributes:
        # Identity & Metadata
        question_id: Unique identifier for the question
        template_id: MD5 hash of template code
        question_text: The question to verify
        template_code: Python code defining the Answer class

        # Configuration
        answering_model: Configuration for the answering LLM
        parsing_model: Configuration for the parsing LLM
        rubric: Optional rubric for evaluation
        keywords: Keywords associated with the question

        # Run Metadata
        run_name: Optional run name for tracking
        replicate: Replicate number (1, 2, 3, ...) for repeated runs of the same question

        # Feature Flags
        few_shot_enabled: Whether few-shot prompting is enabled
        abstention_enabled: Whether abstention detection is enabled
        sufficiency_enabled: Whether trace sufficiency detection is enabled
        deep_judgment_enabled: Whether deep-judgment parsing is enabled

        # Rubric Configuration
        rubric_evaluation_strategy: Strategy for evaluating LLM rubric traits ("batch" or "sequential")

        # Deep-Judgment Configuration
        deep_judgment_max_excerpts_per_attribute: Max excerpts per attribute
        deep_judgment_fuzzy_match_threshold: Similarity threshold for excerpts
        deep_judgment_excerpt_retry_attempts: Retry attempts for excerpt validation
        deep_judgment_search_enabled: Whether search enhancement is enabled
        deep_judgment_search_tool: Search tool name or callable

        # Few-Shot Configuration
        few_shot_examples: List of question-answer pairs for few-shot prompting

        # Answer Caching
        cached_answer_data: Optional cached answer data from previous generation.
            If provided, GenerateAnswerStage will use this instead of calling LLM.
            Used to share answers across multiple judges.

        # Artifacts (populated by stages)
        artifacts: Dictionary storing stage outputs (raw_answer, parsed_answer, etc.)

        # Result Builder (accumulates VerificationResult fields)
        result_builder: Dictionary accumulating result fields

        # Error Tracking
        error: Optional error message if pipeline fails
        completed_without_errors: Whether pipeline completed successfully
    """

    # Identity & Metadata
    question_id: str
    template_id: str
    question_text: str
    template_code: str

    # Configuration
    answering_model: ModelConfig
    parsing_model: ModelConfig
    rubric: Rubric | None = None
    keywords: list[str] | None = None
    raw_answer: str | None = None

    # Run Metadata
    run_name: str | None = None
    replicate: int | None = None

    # Feature Flags
    few_shot_enabled: bool = False
    abstention_enabled: bool = False
    sufficiency_enabled: bool = False
    deep_judgment_enabled: bool = False

    # Rubric Configuration
    rubric_evaluation_strategy: str = "batch"  # "batch" or "sequential"

    # Deep-Judgment Configuration
    deep_judgment_max_excerpts_per_attribute: int = 3
    deep_judgment_fuzzy_match_threshold: float = 0.80
    deep_judgment_excerpt_retry_attempts: int = 2
    deep_judgment_search_enabled: bool = False
    deep_judgment_search_tool: str | Any = "tavily"

    # Deep-Judgment Rubric Configuration (NEW - runtime control of deep judgment for rubrics)
    deep_judgment_rubric_mode: str = "disabled"  # Mode: disabled, enable_all, use_checkpoint, custom
    deep_judgment_rubric_global_excerpts: bool = True  # For enable_all mode: enable/disable excerpts
    deep_judgment_rubric_config: dict[str, Any] | None = None  # For custom mode: nested trait config
    deep_judgment_rubric_max_excerpts_default: int = 7
    deep_judgment_rubric_fuzzy_match_threshold_default: float = 0.80
    deep_judgment_rubric_excerpt_retry_attempts_default: int = 2
    deep_judgment_rubric_search_enabled: bool = False
    deep_judgment_rubric_search_tool: str | Any = "tavily"

    # Prompt Configuration
    prompt_config: PromptConfig | None = None

    # Few-Shot Configuration
    few_shot_examples: list[dict[str, str]] | None = None

    # Trace Filtering Configuration (MCP Agent Evaluation)
    use_full_trace_for_template: bool = False  # Whether to use full trace for template parsing
    use_full_trace_for_rubric: bool = True  # Whether to use full trace for rubric evaluation

    # Answer Caching
    cached_answer_data: dict[str, Any] | None = None

    # Artifacts (populated by stages)
    artifacts: dict[str, Any] = field(default_factory=dict)

    # Result Builder (accumulates VerificationResult fields)
    result_builder: dict[str, Any] = field(default_factory=dict)

    # Error Tracking
    error: str | None = None
    completed_without_errors: bool = True

    def set_artifact(self, key: str, value: Any) -> None:
        """Store an artifact produced by a stage."""
        self.artifacts[key] = value

    def get_artifact(self, key: str, default: Any = None) -> Any:
        """Retrieve an artifact produced by a previous stage."""
        return self.artifacts.get(key, default)

    def has_artifact(self, key: str) -> bool:
        """Check if an artifact exists."""
        return key in self.artifacts

    def set_result_field(self, key: str, value: Any) -> None:
        """Set a field in the result builder."""
        self.result_builder[key] = value

    def get_result_field(self, key: str, default: Any = None) -> Any:
        """Get a field from the result builder."""
        return self.result_builder.get(key, default)

    def mark_error(self, error_message: str) -> None:
        """Mark the context as failed with an error message."""
        self.error = error_message
        self.completed_without_errors = False


class VerificationStage(Protocol):
    """
    Protocol defining the interface for verification pipeline stages.

    Each stage is a self-contained unit that:
    1. Declares its required input artifacts
    2. Declares what artifacts it produces
    3. Decides whether it should run (conditional execution)
    4. Executes its logic, reading from and writing to the context

    Stages are composable and can be enabled/disabled based on configuration.
    """

    @property
    def name(self) -> str:
        """
        Human-readable name for this stage.

        Returns:
            Stage name (e.g., "ValidateTemplate", "GenerateAnswer")
        """
        ...

    @property
    def requires(self) -> list[str]:
        """
        List of artifact keys this stage requires from the context.

        The orchestrator validates that all required artifacts are present
        before executing this stage.

        Returns:
            List of artifact keys (e.g., ["raw_llm_response", "parsed_answer"])
        """
        ...

    @property
    def produces(self) -> list[str]:
        """
        List of NEW artifact keys this stage produces.

        These artifacts will be available to subsequent stages via
        StageRegistry.validate_dependencies().

        Important: Only list artifacts that this stage CREATES. Do not list
        artifacts that this stage merely MODIFIES (e.g., auto-fail stages
        that update verify_result should return [] since verify_result is
        produced by VerifyTemplateStage).

        Returns:
            List of artifact keys (e.g., ["verify_result", "regex_metadata"])
        """
        ...

    def should_run(self, context: VerificationContext) -> bool:
        """
        Determine whether this stage should execute based on context/config.

        This enables conditional execution (e.g., embedding check only runs
        if enabled and field verification failed).

        Args:
            context: Current verification context

        Returns:
            True if stage should execute, False to skip
        """
        ...

    def execute(self, context: VerificationContext) -> None:
        """
        Execute this stage's logic.

        Stages should:
        - Read required artifacts from context.artifacts
        - Perform their processing
        - Write produced artifacts to context.artifacts
        - Accumulate result fields in context.result_builder
        - Handle errors gracefully (set context.error if fatal)

        Args:
            context: Current verification context (modified in-place)

        Raises:
            Exception: If stage encounters a fatal error
        """
        ...


class BaseVerificationStage(ABC):
    """
    Abstract base class for verification stages.

    Provides a convenient implementation of the VerificationStage protocol
    with common functionality and clear contracts.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stage name (must be implemented by subclasses)."""
        pass

    @property
    def requires(self) -> list[str]:
        """
        Artifacts required by this stage.

        Override in subclasses to declare dependencies.
        Default: No requirements.
        """
        return []

    @property
    def produces(self) -> list[str]:
        """
        NEW artifacts produced by this stage.

        Override in subclasses to declare outputs.
        Default: No outputs (correct for stages that only modify existing artifacts).

        Important: Only list artifacts that this stage CREATES. Do not list
        artifacts that this stage merely MODIFIES. For example:
        - VerifyTemplateStage produces "verify_result" (creates it)
        - RecursionLimitAutoFailStage returns [] (only modifies verify_result)
        """
        return []

    def should_run(self, context: VerificationContext) -> bool:
        """
        Whether this stage should execute.

        Default behavior: Skip if context has an error.
        This makes error-checking the default for all stages.

        Override in subclasses for additional conditional execution.
        When overriding, call super().should_run(context) first to preserve
        the error-checking behavior:

            def should_run(self, context: VerificationContext) -> bool:
                if not super().should_run(context):
                    return False
                # ... additional conditions ...
                return True

        Args:
            context: Verification context

        Returns:
            False if context has an error, True otherwise
        """
        return not context.error

    @abstractmethod
    def execute(self, context: VerificationContext) -> None:
        """
        Execute stage logic (must be implemented by subclasses).

        Note: Error checking is handled by should_run() before execute() is called.
        Stages should not check `context.error` at the start of execute() since
        the orchestrator only calls execute() if should_run() returns True.

        Args:
            context: Verification context (modified in-place)
        """
        pass

    def get_or_create_usage_tracker(self, context: VerificationContext) -> "UsageTracker":
        """
        Retrieve UsageTracker from context or create a new one.

        This helper consolidates the common pattern of retrieving the usage tracker
        from the 'usage_tracker' artifact, or creating a new one if it doesn't exist.

        Args:
            context: Verification context

        Returns:
            UsageTracker instance (from context or newly created)
        """
        from ...utils.trace_usage_tracker import UsageTracker

        usage_tracker: UsageTracker | None = context.get_artifact("usage_tracker")
        if usage_tracker is None:
            usage_tracker = UsageTracker()
            logger.warning("No usage tracker found in context, initializing new one")
        return usage_tracker

    def set_artifact_and_result(self, context: VerificationContext, key: str, value: Any) -> None:
        """
        Set both artifact and result field with the same key/value.

        This helper consolidates the common pattern of setting both
        context.set_artifact(key, value) and context.set_result_field(key, value)
        with identical key and value parameters.

        Use this when the same data needs to be available as both:
        - An artifact (for subsequent stages to read)
        - A result field (for inclusion in the final VerificationResult)

        Args:
            context: Verification context
            key: The key for both artifact and result field
            value: The value to store
        """
        context.set_artifact(key, value)
        context.set_result_field(key, value)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"{self.__class__.__name__}(name={self.name})"


class StageRegistry:
    """
    Registry for managing verification stages.

    Provides:
    - Stage registration and lookup
    - Dependency validation
    - Stage ordering
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._stages: dict[str, VerificationStage] = {}

    def register(self, stage: VerificationStage) -> None:
        """
        Register a stage.

        Args:
            stage: Stage to register

        Raises:
            ValueError: If stage name already registered
        """
        if stage.name in self._stages:
            raise ValueError(f"Stage '{stage.name}' is already registered")
        self._stages[stage.name] = stage

    def validate_dependencies(self, stages: list[VerificationStage]) -> list[str]:
        """
        Validate that all stage dependencies can be satisfied.

        Checks that for each stage, all required artifacts are produced
        by earlier stages in the list.

        Args:
            stages: Ordered list of stages to validate

        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        available_artifacts: set[str] = set()

        for stage in stages:
            # Check if all required artifacts are available
            missing = set(stage.requires) - available_artifacts
            if missing:
                errors.append(
                    f"Stage '{stage.name}' requires artifacts {missing} which are not produced by any earlier stage"
                )

            # Add this stage's outputs to available artifacts
            available_artifacts.update(stage.produces)

        return errors

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"StageRegistry(stages={list(self._stages.keys())})"


# Convenience type alias for stage lists
StageList = list[VerificationStage]
