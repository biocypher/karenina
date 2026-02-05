"""Verification job model."""

import time
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from ..results.verification_result_set import VerificationResultSet
    from .config import VerificationConfig


class VerificationJob(BaseModel):
    """Represents a verification job."""

    model_config = ConfigDict(extra="forbid")

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

    @staticmethod
    def _make_task_key(question_id: str, replicate: int | None = None) -> str:
        """Create a unique task key from question_id and optional replicate.

        For single-replicate runs, returns just the question_id.
        For multi-replicate runs, appends _rep{N} to distinguish tasks.
        """
        if replicate is None:
            return question_id
        return f"{question_id}_rep{replicate}"

    def task_started(self, question_id: str, replicate: int | None = None) -> None:
        """Mark a task as started and record start time.

        Args:
            question_id: The question identifier
            replicate: Optional replicate number (for multi-replicate runs)
        """
        task_key = self._make_task_key(question_id, replicate)
        if task_key not in self.in_progress_questions:
            self.in_progress_questions.append(task_key)

        # Record task start time
        self.task_start_times[task_key] = time.time()

    def task_finished(self, question_id: str, success: bool, replicate: int | None = None) -> None:
        """Mark a task as finished, calculate duration, and update counts.

        Args:
            question_id: The question identifier
            success: Whether the task completed successfully
            replicate: Optional replicate number (for multi-replicate runs)
        """
        task_key = self._make_task_key(question_id, replicate)

        # Calculate task duration from recorded start time
        task_duration = 0.0
        if task_key in self.task_start_times:
            task_duration = time.time() - self.task_start_times[task_key]
            # Clean up start time
            del self.task_start_times[task_key]

        # Remove from in-progress list
        if task_key in self.in_progress_questions:
            self.in_progress_questions.remove(task_key)

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
