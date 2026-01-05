"""Configuration models for GEPA optimization."""

from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    pass


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
    verification pipeline.

    Example:
        >>> config = OptimizationConfig(
        ...     targets=[OptimizationTarget.ANSWERING_SYSTEM_PROMPT],
        ...     seed_answering_prompt="You are a helpful assistant.",
        ...     template_weight=0.7,
        ...     rubric_weight=0.3,
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

    # Scoring weights
    template_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for template verification result (pass/fail)",
    )
    rubric_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for rubric evaluation scores",
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
    def validate_weights_sum(self) -> "OptimizationConfig":
        """Ensure template_weight + rubric_weight == 1.0."""
        total = self.template_weight + self.rubric_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"template_weight + rubric_weight must equal 1.0, got {total}")
        return self

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
            self.seed_answering_prompt = "You are a helpful assistant."
        if OptimizationTarget.PARSING_INSTRUCTIONS in self.targets and self.seed_parsing_instructions is None:
            self.seed_parsing_instructions = "Extract the answer from the response following the schema."
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
