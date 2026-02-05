"""API models for verification endpoints."""

from typing import Any

from pydantic import BaseModel, ConfigDict

from .config import VerificationConfig
from .result import VerificationResult


class FinishedTemplate(BaseModel):
    """Metadata for a finished answer template."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    question_text: str
    question_preview: str  # Truncated version for UI
    raw_answer: str | None = None  # Ground truth answer from checkpoint
    template_code: str
    last_modified: str
    finished: bool = True
    question_rubric: dict[str, Any] | None = None  # Question-specific rubric as dict
    keywords: list[str] | None = None  # Keywords associated with the question
    few_shot_examples: list[dict[str, str]] | None = None  # Few-shot examples for this question


class VerificationRequest(BaseModel):
    """Request to start verification."""

    model_config = ConfigDict(extra="forbid")

    config: VerificationConfig
    question_ids: list[str] | None = None  # If None, verify all finished templates
    run_name: str | None = None  # Optional user-defined run name


class VerificationStatusResponse(BaseModel):
    """Response for verification status."""

    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")

    job_id: str
    run_name: str
    status: str
    message: str
