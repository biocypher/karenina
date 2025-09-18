"""Data models for TaskEval."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class LogEvent(BaseModel):
    """Single log event in TaskEval."""

    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    level: Literal["debug", "info", "warn", "error"]
    text: str
    tags: list[str] | None = None
    payload: dict[str, Any] | None = None
    # New fields for agent output logging
    question_id: str | None = Field(default=None, description="Question this log answers")
    is_agent_output: bool = Field(default=False, description="Whether this is agent output to be evaluated")
    output_type: str | None = Field(default=None, description="Type of output: answer, reasoning, analysis, etc.")


class StepEval(BaseModel):
    """Evaluation results for a single step or global evaluation."""

    rubric_scores: dict[str, int | bool] = Field(
        default_factory=dict, description="Rubric trait evaluations with same structure as verification"
    )
    question_verification: dict[str, list[dict[str, Any]]] = Field(
        default_factory=dict,
        description="Question verification results: question_id -> list of results for multiple responses",
    )


class TaskEvalResult(BaseModel):
    """Complete task evaluation results."""

    task_id: str | None = None
    metadata: dict[str, Any] | None = None
    global_eval: StepEval | None = None
    per_step: dict[str, StepEval] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    logs: dict[str, list[LogEvent]] = Field(default_factory=dict, description="Optional: include logs in results")
