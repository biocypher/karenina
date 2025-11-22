"""Verification job model."""

import time
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..verification_result_set import VerificationResultSet
    from .config import VerificationConfig


class VerificationJob(BaseModel):
    """Represents a verification job."""

    job_id: str
    run_name: str  # User-defined or auto-generated run name
    status: Literal["pending", "running", "completed", "failed", "cancelled"]
    config: "VerificationConfig"

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
    last_task_duration: float | None = None  # Execution time of last completed task

    # WebSocket streaming progress fields
    in_progress_questions: list[str] = Field(default_factory=list)

    # Task timing tracking (maps question_id to start time)
    task_start_times: dict[str, float] = Field(default_factory=dict)

    # Timing
    start_time: float | None = None
    end_time: float | None = None

    # Results
    result_set: "VerificationResultSet | None" = None  # Unified verification result container
    error_message: str | None = None

    def task_started(self, question_id: str) -> None:
        """Mark a task as started and record start time."""
        if question_id not in self.in_progress_questions:
            self.in_progress_questions.append(question_id)

        # Record task start time
        self.task_start_times[question_id] = time.time()

    def task_finished(self, question_id: str, success: bool) -> None:
        """Mark a task as finished, calculate duration, and update counts."""
        # Calculate task duration from recorded start time
        task_duration = 0.0
        if question_id in self.task_start_times:
            task_duration = time.time() - self.task_start_times[question_id]
            # Clean up start time
            del self.task_start_times[question_id]

        # Remove from in-progress list
        if question_id in self.in_progress_questions:
            self.in_progress_questions.remove(question_id)

        # Update counts
        self.processed_count += 1
        if success:
            self.successful_count += 1
        else:
            self.failed_count += 1

        # Update percentage
        self.percentage = (self.processed_count / self.total_questions) * 100 if self.total_questions > 0 else 0.0

        # Track last task duration
        self.last_task_duration = task_duration

    def to_dict(self) -> dict[str, Any]:
        """Convert job to dictionary for API response."""
        # Calculate duration if job has started
        duration = None
        if self.start_time:
            duration = self.end_time - self.start_time if self.end_time else time.time() - self.start_time

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
            "duration_seconds": duration,
            "last_task_duration": self.last_task_duration,
            "error_message": self.error_message,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "in_progress_questions": self.in_progress_questions,
        }
