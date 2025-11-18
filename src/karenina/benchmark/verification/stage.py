"""Stage-based verification pipeline architecture.

This module provides the core abstractions for the modular verification pipeline:
- VerificationContext: Shared state and artifacts across stages
- VerificationStage: Protocol defining stage interface
- StageRegistry: Manages stage instances and dependencies
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol

from ...schemas.domain import Rubric
from ...schemas.workflow import ModelConfig


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
        job_id: Optional job ID for tracking
        answering_replicate: Replicate number for answering (1, 2, 3, ...)
        parsing_replicate: Replicate number for parsing (1, 2, 3, ...)

        # Feature Flags
        few_shot_enabled: Whether few-shot prompting is enabled
        abstention_enabled: Whether abstention detection is enabled
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

    # Run Metadata
    run_name: str | None = None
    job_id: str | None = None
    answering_replicate: int | None = None
    parsing_replicate: int | None = None

    # Feature Flags
    few_shot_enabled: bool = False
    abstention_enabled: bool = False
    deep_judgment_enabled: bool = False

    # Rubric Configuration
    rubric_evaluation_strategy: str = "batch"  # "batch" or "sequential"

    # Deep-Judgment Configuration
    deep_judgment_max_excerpts_per_attribute: int = 3
    deep_judgment_fuzzy_match_threshold: float = 0.80
    deep_judgment_excerpt_retry_attempts: int = 2
    deep_judgment_search_enabled: bool = False
    deep_judgment_search_tool: str | Any = "tavily"

    # Few-Shot Configuration
    few_shot_examples: list[dict[str, str]] | None = None

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
        List of artifact keys this stage produces.

        These artifacts will be available to subsequent stages.

        Returns:
            List of artifact keys (e.g., ["verification_result", "regex_metadata"])
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
        Artifacts produced by this stage.

        Override in subclasses to declare outputs.
        Default: No outputs.
        """
        return []

    def should_run(self, context: VerificationContext) -> bool:  # noqa: ARG002
        """
        Whether this stage should execute.

        Override in subclasses for conditional execution.
        Default: Always run.

        Args:
            context: Verification context (may be unused in simple stages)
        """
        return True

    @abstractmethod
    def execute(self, context: VerificationContext) -> None:
        """
        Execute stage logic (must be implemented by subclasses).

        Args:
            context: Verification context (modified in-place)
        """
        pass

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

    def get(self, name: str) -> VerificationStage | None:
        """
        Get a stage by name.

        Args:
            name: Stage name

        Returns:
            Stage instance or None if not found
        """
        return self._stages.get(name)

    def has(self, name: str) -> bool:
        """
        Check if stage is registered.

        Args:
            name: Stage name

        Returns:
            True if stage registered, False otherwise
        """
        return name in self._stages

    def list_stages(self) -> list[str]:
        """
        List all registered stage names.

        Returns:
            List of stage names
        """
        return list(self._stages.keys())

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
        return f"StageRegistry(stages={self.list_stages()})"


# Convenience type alias for stage lists
StageList = list[VerificationStage]
