"""Data types for GEPA-Karenina integration.

These types map karenina's verification concepts to GEPA's adapter interface:
- KareninaDataInst: Input data instance (question + ground truth)
- KareninaTrajectory: Execution trace (verification result + metadata)
- KareninaOutput: Optimized components output
- BenchmarkSplit: Train/val/test split result
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from karenina.schemas.config import ModelConfig
    from karenina.schemas.workflow.verification.result import VerificationResult


@dataclass
class KareninaDataInst:
    """Single evaluation instance for GEPA.

    Represents one question from a karenina benchmark with all information
    needed for verification.
    """

    question_id: str
    """Unique identifier for the question."""

    question_text: str
    """The question text to be answered by the model."""

    raw_answer: str
    """Ground truth answer for verification."""

    template_code: str
    """Python code defining the Answer template class."""

    rubric: dict[str, Any] | None = None
    """Optional rubric configuration for quality evaluation."""

    few_shot_examples: list[dict[str, str]] | None = None
    """Optional few-shot examples for the question."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata (author, tags, etc.)."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict for GEPA."""
        return {
            "question_id": self.question_id,
            "question_text": self.question_text,
            "raw_answer": self.raw_answer,
            "template_code": self.template_code,
            "rubric": self.rubric,
            "few_shot_examples": self.few_shot_examples,
            "metadata": self.metadata,
        }


@dataclass
class KareninaTrajectory:
    """Execution trace from one verification run.

    Captures detailed information about a single verification attempt
    for use in GEPA's reflective feedback mechanism.

    Each trajectory represents one (question, model) combination.
    """

    data_inst: KareninaDataInst
    """The input question instance."""

    model_name: str
    """Name/identifier of the model that produced this result."""

    model_config: "ModelConfig"
    """Full model configuration used."""

    optimized_components: dict[str, str]
    """The optimized text components that were injected."""

    verification_result: "VerificationResult"
    """Complete verification result from karenina."""

    # Detailed feedback for reflection
    raw_llm_response: str | None = None
    """Raw text response from the answering model."""

    parsing_error: str | None = None
    """Error message if parsing failed."""

    failed_fields: list[str] | None = None
    """List of template fields that failed verification."""

    rubric_scores: dict[str, float] | None = None
    """Per-trait rubric scores."""

    def passed(self) -> bool:
        """Check if verification passed."""
        if self.verification_result.template:
            return self.verification_result.template.verify_result or False
        return False

    def to_feedback_dict(self) -> dict[str, Any]:
        """Convert to feedback dict for GEPA reflection.

        Returns dict with "Inputs", "Generated Outputs", "Feedback" keys
        as expected by GEPA's reflective proposer.
        """
        feedback_parts = []

        if self.parsing_error:
            feedback_parts.append(f"Parsing error: {self.parsing_error}")

        if self.failed_fields:
            feedback_parts.append(f"Failed fields: {', '.join(self.failed_fields)}")

        feedback_parts.append(f"Expected answer: {self.data_inst.raw_answer}")

        return {
            "Inputs": {
                "question": self.data_inst.question_text,
                "model": self.model_name,
            },
            "Generated Outputs": self.raw_llm_response or "(no response)",
            "Feedback": "\n".join(feedback_parts),
        }


@dataclass
class KareninaOutput:
    """Optimized text components from a GEPA run.

    Contains the best-performing text components discovered during
    optimization, along with metrics.
    """

    # Optimized components
    answering_system_prompt: str | None = None
    """Optimized system prompt for answering model."""

    parsing_instructions: str | None = None
    """Optimized instructions for parsing model."""

    mcp_tool_descriptions: dict[str, str] | None = None
    """Optimized MCP tool descriptions (tool_name -> description)."""

    # Metrics
    train_score: float = 0.0
    """Final score on training set."""

    val_score: float = 0.0
    """Final score on validation set."""

    test_score: float | None = None
    """Optional score on test set (if test set was used)."""

    baseline_score: float = 0.0
    """Score before optimization (seed candidate)."""

    improvement: float = 0.0
    """Relative improvement: (val_score - baseline_score) / baseline_score."""

    # Optimization metadata
    total_generations: int = 0
    """Number of GEPA generations run."""

    total_metric_calls: int = 0
    """Total number of evaluation calls used."""

    best_generation: int = 0
    """Generation number where best candidate was found."""

    def get_optimized_prompts(self) -> dict[str, str]:
        """Get all optimized prompts as a dict."""
        result: dict[str, str] = {}
        if self.answering_system_prompt:
            result["answering_system_prompt"] = self.answering_system_prompt
        if self.parsing_instructions:
            result["parsing_instructions"] = self.parsing_instructions
        if self.mcp_tool_descriptions:
            for name, desc in self.mcp_tool_descriptions.items():
                result[f"mcp_tool_{name}"] = desc
        return result


@dataclass
class BenchmarkSplit:
    """Result of splitting a benchmark for optimization.

    Contains train, validation, and optionally test sets of
    KareninaDataInst objects.
    """

    train: list[KareninaDataInst]
    """Training set instances."""

    val: list[KareninaDataInst]
    """Validation set instances."""

    test: list[KareninaDataInst] | None = None
    """Optional test set instances."""

    seed: int | None = None
    """Random seed used for splitting (for reproducibility)."""

    def __post_init__(self) -> None:
        """Validate the split."""
        if not self.train:
            raise ValueError("Training set cannot be empty")
        if not self.val:
            raise ValueError("Validation set cannot be empty")

    @property
    def train_ids(self) -> list[str]:
        """Get question IDs in training set."""
        return [inst.question_id for inst in self.train]

    @property
    def val_ids(self) -> list[str]:
        """Get question IDs in validation set."""
        return [inst.question_id for inst in self.val]

    @property
    def test_ids(self) -> list[str] | None:
        """Get question IDs in test set (if exists)."""
        if self.test is None:
            return None
        return [inst.question_id for inst in self.test]

    def summary(self) -> str:
        """Get a summary string of the split."""
        parts = [
            f"Train: {len(self.train)} questions",
            f"Val: {len(self.val)} questions",
        ]
        if self.test:
            parts.append(f"Test: {len(self.test)} questions")
        if self.seed is not None:
            parts.append(f"Seed: {self.seed}")
        return ", ".join(parts)
