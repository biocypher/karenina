"""Configuration models for GEPA optimization."""

from enum import Enum
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, model_validator

from karenina.integrations.gepa.prompts.defaults import (
    DEFAULT_ANSWERING_SYSTEM_PROMPT,
    DEFAULT_PARSING_INSTRUCTIONS,
)

if TYPE_CHECKING:
    from karenina.schemas.config import ModelConfig

# Type alias for GEPA frontier tracking strategies
FrontierType = Literal["instance", "objective", "hybrid", "cartesian"]


class TraitSelectionMode(str, Enum):
    """Mode for selecting which rubric traits become optimization objectives."""

    ALL = "all"
    """Include all rubric traits as separate objectives."""

    NONE = "none"
    """Do not include rubric traits as objectives."""

    CUSTOM = "custom"
    """Include only specified traits as objectives."""


class MetricObjectiveConfig(BaseModel):
    """Configuration for how metric traits are converted to objectives.

    Metric traits (e.g., entity extraction) produce precision, recall, and f1.
    This config controls which of those metrics become optimization objectives.
    """

    include_precision: bool = Field(
        default=False,
        description="Include precision as a separate objective",
    )
    include_recall: bool = Field(
        default=False,
        description="Include recall as a separate objective",
    )
    include_f1: bool = Field(
        default=True,
        description="Include f1 as a separate objective",
    )

    def get_enabled_metrics(self) -> list[str]:
        """Return list of metric names to include as objectives."""
        metrics = []
        if self.include_precision:
            metrics.append("precision")
        if self.include_recall:
            metrics.append("recall")
        if self.include_f1:
            metrics.append("f1")
        return metrics


class ObjectiveConfig(BaseModel):
    """Configuration for multi-objective optimization dimensions.

    Controls which verification dimensions become separate Pareto objectives.
    Each objective is keyed as 'model:dimension' for cross-model optimization.

    Example:
        >>> config = ObjectiveConfig(
        ...     include_template=True,
        ...     trait_mode=TraitSelectionMode.CUSTOM,
        ...     selected_traits=["clarity", "safety"],
        ... )
        # Produces keys: "claude-haiku:template", "claude-haiku:clarity", etc.
    """

    include_template: bool = Field(
        default=True,
        description="Include template verification as an objective dimension",
    )
    trait_mode: TraitSelectionMode = Field(
        default=TraitSelectionMode.ALL,
        description="Mode for rubric trait selection",
    )
    selected_traits: list[str] | None = Field(
        default=None,
        description="Trait names to include (required when trait_mode=CUSTOM)",
    )
    metric_config: MetricObjectiveConfig = Field(
        default_factory=MetricObjectiveConfig,
        description="Configuration for metric trait objectives",
    )

    @model_validator(mode="after")
    def validate_custom_requires_traits(self) -> "ObjectiveConfig":
        """Ensure selected_traits is provided when mode is CUSTOM."""
        if self.trait_mode == TraitSelectionMode.CUSTOM and not self.selected_traits:
            raise ValueError("selected_traits must be provided when trait_mode is CUSTOM")
        return self

    @model_validator(mode="after")
    def validate_has_objectives(self) -> "ObjectiveConfig":
        """Warn if configuration would result in no objectives."""
        if not self.include_template and self.trait_mode == TraitSelectionMode.NONE:
            raise ValueError(
                "ObjectiveConfig must include at least one objective dimension. "
                "Set include_template=True or trait_mode to ALL/CUSTOM."
            )
        return self

    def should_include_trait(self, trait_name: str) -> bool:
        """Check if a specific trait should be included as an objective."""
        if self.trait_mode == TraitSelectionMode.NONE:
            return False
        if self.trait_mode == TraitSelectionMode.ALL:
            return True
        return trait_name in (self.selected_traits or [])


class OptimizationTarget(str, Enum):
    """Text components that can be optimized by GEPA."""

    ANSWERING_SYSTEM_PROMPT = "answering_system_prompt"
    """System prompt for the answering model (generates responses to questions)."""

    PARSING_INSTRUCTIONS = "parsing_instructions"
    """Instructions for the parsing/judge model (extracts structured data from responses)."""

    MCP_TOOL_DESCRIPTIONS = "mcp_tool_descriptions"
    """Descriptions for MCP tools (guides model tool selection and usage)."""


class OptimizationConfig(BaseModel):
    """Configuration for a GEPA optimization run.

    This configures how GEPA optimizes text components used in karenina's
    verification pipeline with multi-objective Pareto optimization.

    Example:
        >>> config = OptimizationConfig(
        ...     targets=[OptimizationTarget.ANSWERING_SYSTEM_PROMPT],
        ...     seed_answering_prompt="You are a helpful assistant.",
        ...     objective_config=ObjectiveConfig(
        ...         include_template=True,
        ...         trait_mode=TraitSelectionMode.ALL,
        ...     ),
        ...     frontier_type="objective",  # Use multi-objective Pareto
        ...     reflection_model="openai/gpt-4o",
        ...     max_metric_calls=150,
        ... )
    """

    # What to optimize
    targets: list[OptimizationTarget] = Field(description="Text components to optimize")

    # Seed text (initial candidates)
    seed_answering_prompt: str | None = Field(
        default=None,
        description="Initial system prompt for answering model",
    )
    seed_parsing_instructions: str | None = Field(
        default=None,
        description="Initial instructions for parsing model",
    )
    seed_mcp_tool_descriptions: dict[str, str] | None = Field(
        default=None,
        description="Initial MCP tool descriptions (tool_name -> description)",
    )

    # Multi-objective configuration
    objective_config: ObjectiveConfig = Field(
        default_factory=ObjectiveConfig,
        description="Configuration for multi-objective optimization dimensions",
    )
    frontier_type: FrontierType = Field(
        default="objective",
        description=(
            "GEPA Pareto frontier tracking strategy: "
            "'instance' (per validation example), "
            "'objective' (per objective metric - recommended for multi-objective), "
            "'hybrid' (both), or 'cartesian' (per example × objective)"
        ),
    )

    # GEPA parameters
    reflection_model: str = Field(
        default="openai/gpt-4o",
        description="LiteLLM model string for GEPA's reflection LLM",
    )
    max_metric_calls: int = Field(
        default=150,
        ge=1,
        description="Maximum number of evaluation calls (budget)",
    )
    candidate_selection_strategy: str = Field(
        default="pareto",
        description="GEPA candidate selection strategy: 'pareto', 'current_best', 'epsilon_greedy'",
    )

    # Feedback generation
    feedback_model: "ModelConfig | None" = Field(
        default=None,
        description="Model config for generating LLM feedback. If None, uses programmatic feedback.",
    )
    enable_differential_analysis: bool = Field(
        default=True,
        description="When True, performs differential analysis between successful and failed traces.",
    )

    # Data splitting (used when not providing explicit train/val sets)
    train_ratio: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Fraction of questions for training (default: 80%)",
    )
    val_ratio: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Fraction of questions for validation (default: 20%)",
    )
    test_ratio: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional fraction for test set. If None, no test set.",
    )
    split_seed: int | None = Field(
        default=None,
        description="Random seed for reproducible data splitting",
    )

    # Explicit question ID lists (override ratio-based splitting)
    train_question_ids: list[str] | None = Field(
        default=None,
        description="Explicit list of question IDs for training",
    )
    val_question_ids: list[str] | None = Field(
        default=None,
        description="Explicit list of question IDs for validation",
    )
    test_question_ids: list[str] | None = Field(
        default=None,
        description="Explicit list of question IDs for testing",
    )

    @model_validator(mode="after")
    def validate_split_ratios(self) -> "OptimizationConfig":
        """Ensure split ratios sum to 1.0."""
        if self.test_ratio is not None:
            total = self.train_ratio + self.val_ratio + self.test_ratio
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"train_ratio + val_ratio + test_ratio must equal 1.0, got {total}")
        else:
            total = self.train_ratio + self.val_ratio
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"train_ratio + val_ratio must equal 1.0 when test_ratio is None, got {total}")
        return self

    @model_validator(mode="after")
    def validate_seeds_for_targets(self) -> "OptimizationConfig":
        """Warn if optimizing a target without providing a seed."""
        if OptimizationTarget.ANSWERING_SYSTEM_PROMPT in self.targets and self.seed_answering_prompt is None:
            # Use a default seed prompt
            self.seed_answering_prompt = DEFAULT_ANSWERING_SYSTEM_PROMPT
        if OptimizationTarget.PARSING_INSTRUCTIONS in self.targets and self.seed_parsing_instructions is None:
            self.seed_parsing_instructions = DEFAULT_PARSING_INSTRUCTIONS
        return self

    def get_seed_candidate(self) -> dict[str, str]:
        """Build the initial candidate dict for GEPA optimization.

        Returns:
            Dict mapping component names to their seed text.
        """
        candidate: dict[str, str] = {}
        if OptimizationTarget.ANSWERING_SYSTEM_PROMPT in self.targets:
            candidate["answering_system_prompt"] = self.seed_answering_prompt or ""
        if OptimizationTarget.PARSING_INSTRUCTIONS in self.targets:
            candidate["parsing_instructions"] = self.seed_parsing_instructions or ""
        if OptimizationTarget.MCP_TOOL_DESCRIPTIONS in self.targets and self.seed_mcp_tool_descriptions:
            # Flatten tool descriptions into the candidate
            for tool_name, desc in self.seed_mcp_tool_descriptions.items():
                candidate[f"mcp_tool_{tool_name}"] = desc
        return candidate
