"""Progressive save functionality for CLI verification with resume support.

This module provides incremental saving of verification results and the ability
to resume interrupted verification runs.

Key Components:
- TaskIdentifier: Unique identifier for a verification task
- ProgressiveSaveManager: Manages .tmp and .state files for progressive saving
"""

import contextlib
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..schemas import VerificationConfig, VerificationResult
from ..schemas.config import ModelConfig
from ..schemas.entities import Rubric
from ..schemas.results import VerificationResultSet
from ..schemas.verification.model_identity import ModelIdentity

logger = logging.getLogger(__name__)


def get_karenina_version() -> str:
    """Get the current Karenina version."""
    try:
        import karenina

        return getattr(karenina, "__version__", "unknown")
    except ImportError:
        return "unknown"


@dataclass
class TaskIdentifier:
    """Unique identifier for a verification task.

    A task is uniquely identified by:
    - question_id: The question being verified
    - answering_canonical_key: ModelIdentity canonical_key for the answering model
      (encodes interface, model_name, and tools)
    - parsing_canonical_key: ModelIdentity canonical_key for the parsing model
    - replicate: Replicate number (None for single replicate)
    """

    question_id: str
    answering_canonical_key: str
    parsing_canonical_key: str
    replicate: int | None

    # Separator between key parts — chosen to avoid conflicts with canonical_key
    # characters (:, |) and common ID characters (-, _)
    _SEP = "\t"

    def to_key(self) -> str:
        """Generate unique string key for this task.

        Format: {question_id}\\t{answering_canonical_key}\\t{parsing_canonical_key}[\\trepN]
        Uses tab separator to avoid conflicts with canonical_key's : and | characters.
        """
        parts = [
            self.question_id,
            self.answering_canonical_key,
            self.parsing_canonical_key,
        ]
        if self.replicate is not None:
            parts.append(f"rep{self.replicate}")
        return self._SEP.join(parts)

    @classmethod
    def from_key(cls, key: str) -> "TaskIdentifier":
        """Parse a task key back into a TaskIdentifier."""
        parts = key.split(cls._SEP)

        # Check if last part is a replicate marker
        replicate = None
        if parts[-1].startswith("rep"):
            replicate = int(parts[-1][3:])
            parts = parts[:-1]

        if len(parts) != 3:
            raise ValueError(f"Invalid task key format: {key}")

        return cls(
            question_id=parts[0],
            answering_canonical_key=parts[1],
            parsing_canonical_key=parts[2],
            replicate=replicate,
        )

    @classmethod
    def from_task_dict(cls, task: dict[str, Any]) -> "TaskIdentifier":
        """Create TaskIdentifier from a batch_runner task dictionary."""
        answering_model: ModelConfig = task["answering_model"]
        parsing_model: ModelConfig = task["parsing_model"]

        return cls(
            question_id=task["question_id"],
            answering_canonical_key=ModelIdentity.from_model_config(answering_model, role="answering").canonical_key,
            parsing_canonical_key=ModelIdentity.from_model_config(parsing_model, role="parsing").canonical_key,
            replicate=task.get("replicate"),
        )

    @classmethod
    def from_result(cls, result: VerificationResult) -> "TaskIdentifier":
        """Create TaskIdentifier from a VerificationResult."""
        return cls(
            question_id=result.metadata.question_id,
            answering_canonical_key=result.metadata.answering.canonical_key,
            parsing_canonical_key=result.metadata.parsing.canonical_key,
            replicate=result.metadata.replicate,
        )


class ProgressiveSaveManager:
    """Manages progressive save state with two files.

    Files:
    - .tmp: Results in standard export format (frontend-readable)
    - .state: Task manifest and progress tracking

    Usage:
        # New job
        manager = ProgressiveSaveManager(output, config, benchmark_path)
        manager.initialize(task_manifest)

        # Resume existing
        manager = ProgressiveSaveManager.load_for_resume(state_path)

        # During verification
        manager.add_result(result)

        # After completion
        manager.finalize()
    """

    STATE_FORMAT_VERSION = "1.0"
    RESULTS_FORMAT_VERSION = "2.1"

    def __init__(
        self,
        output_path: Path,
        config: VerificationConfig,
        benchmark_path: str,
        global_rubric: Rubric | None = None,
    ):
        """Initialize a new ProgressiveSaveManager.

        Args:
            output_path: Final output path (e.g., results.json)
            config: Verification configuration
            benchmark_path: Path to the benchmark file
            global_rubric: Optional global rubric for rubric definition in exports
        """
        self.output_path = output_path
        self.tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        self.state_path = output_path.with_suffix(output_path.suffix + ".state")
        self.config = config
        self.benchmark_path = benchmark_path
        self.global_rubric = global_rubric

        # State tracking
        self._task_manifest: list[str] = []
        self._completed_task_ids: set[str] = set()
        self._results: list[VerificationResult] = []
        self._start_time: float | None = None
        self._config_hash: str = ""

    @classmethod
    def can_resume(cls, output_path: Path) -> bool:
        """Check if state file exists for resumption."""
        state_path = output_path.with_suffix(output_path.suffix + ".state")
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        return state_path.exists() and tmp_path.exists()

    @classmethod
    def load_for_resume(cls, state_path: Path) -> "ProgressiveSaveManager":
        """Load existing state and results for resume.

        Args:
            state_path: Path to the .state file

        Returns:
            Initialized ProgressiveSaveManager ready for resume
        """
        if not state_path.exists():
            raise FileNotFoundError(f"State file not found: {state_path}")

        # Derive paths
        # State path is like results.json.state, so we need results.json
        output_path = Path(str(state_path).rsplit(".state", 1)[0])
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

        if not tmp_path.exists():
            raise FileNotFoundError(f"Results file not found: {tmp_path} (state file exists but results missing)")

        # Load state
        with open(state_path) as f:
            state_data = json.load(f)

        # Validate format version
        if state_data.get("format_version") != cls.STATE_FORMAT_VERSION:
            raise ValueError(
                f"Incompatible state format version: {state_data.get('format_version')} "
                f"(expected {cls.STATE_FORMAT_VERSION})"
            )

        # Load config from state
        config = VerificationConfig(**state_data["config"])
        benchmark_path = state_data["benchmark_path"]

        # Create manager
        manager = cls(output_path, config, benchmark_path)
        manager._task_manifest = state_data["task_manifest"]
        manager._completed_task_ids = set(state_data["completed_task_ids"])
        manager._start_time = state_data.get("start_time")
        manager._config_hash = state_data.get("config_hash", "")

        # Load existing results
        with open(tmp_path) as f:
            tmp_data = json.load(f)

        for result_dict in tmp_data.get("results", []):
            result = VerificationResult.model_validate(result_dict)
            manager._results.append(result)

        logger.info(
            f"Loaded progressive save state: {len(manager._completed_task_ids)}/{len(manager._task_manifest)} "
            f"tasks completed"
        )

        return manager

    def is_compatible(self, config: VerificationConfig, benchmark_path: str) -> tuple[bool, str]:
        """Verify config/benchmark match for safe resume.

        Returns:
            Tuple of (is_compatible, reason_if_not)
        """
        # Check benchmark path
        if self.benchmark_path != benchmark_path:
            return False, f"Benchmark path changed: {self.benchmark_path} -> {benchmark_path}"

        # Check config hash
        new_config_hash = self._compute_config_hash(config)
        if self._config_hash and self._config_hash != new_config_hash:
            return False, "Configuration has changed since the job started"

        return True, ""

    def initialize(self, task_manifest: list[str]) -> None:
        """Initialize fresh state and tmp files with task manifest.

        Args:
            task_manifest: List of task IDs (from TaskIdentifier.to_key())
        """
        self._task_manifest = task_manifest
        self._completed_task_ids = set()
        self._results = []
        self._start_time = time.time()
        self._config_hash = self._compute_config_hash(self.config)

        # Create initial files
        self._save_state()
        self._save_results()

        logger.info(f"Initialized progressive save with {len(task_manifest)} tasks")

    def get_pending_task_ids(self) -> set[str]:
        """Get task IDs not yet completed."""
        return set(self._task_manifest) - self._completed_task_ids

    def set_global_rubric(self, global_rubric: Rubric | None) -> None:
        """Set the global rubric for export formatting.

        This is typically called after loading from resume state,
        once the benchmark is loaded and the rubric is available.

        Args:
            global_rubric: The global rubric from the benchmark
        """
        self.global_rubric = global_rubric

    def add_result(self, result: VerificationResult) -> None:
        """Add result to .tmp and mark complete in .state.

        Uses atomic write pattern for crash safety.
        """
        # Generate task ID from result
        task_id = TaskIdentifier.from_result(result).to_key()

        # Add to results
        self._results.append(result)
        self._completed_task_ids.add(task_id)

        # Save both files atomically
        self._save_results()
        self._save_state()

        logger.debug(f"Saved result for task: {task_id}")

    def get_all_results(self) -> list[VerificationResult]:
        """Load all results from memory."""
        return self._results.copy()

    def get_result_set(self) -> VerificationResultSet:
        """Get all results as a VerificationResultSet."""
        return VerificationResultSet(results=self._results)

    def finalize(self) -> None:
        """Delete .tmp and .state files after successful completion."""
        try:
            if self.tmp_path.exists():
                self.tmp_path.unlink()
                logger.info(f"Removed temporary file: {self.tmp_path}")
        except OSError as e:
            logger.warning(f"Failed to remove {self.tmp_path}: {e}")

        try:
            if self.state_path.exists():
                self.state_path.unlink()
                logger.info(f"Removed state file: {self.state_path}")
        except OSError as e:
            logger.warning(f"Failed to remove {self.state_path}: {e}")

    @property
    def completed_count(self) -> int:
        """Number of completed tasks."""
        return len(self._completed_task_ids)

    @property
    def total_tasks(self) -> int:
        """Total number of tasks in manifest."""
        return len(self._task_manifest)

    def _compute_config_hash(self, config: VerificationConfig) -> str:
        """Compute hash of configuration for compatibility checking."""
        config_json = config.model_dump_json(exclude={"manual_traces"})
        return hashlib.md5(config_json.encode()).hexdigest()

    def _save_state(self) -> None:
        """Save state file with atomic write."""
        state_data = {
            "format_version": self.STATE_FORMAT_VERSION,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._start_time or time.time())),
            "last_updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "benchmark_path": self.benchmark_path,
            "output_path": str(self.output_path),
            "config_hash": self._config_hash,
            "config": self.config.model_dump(mode="json", exclude={"manual_traces": True}),
            "task_manifest": self._task_manifest,
            "completed_task_ids": list(self._completed_task_ids),
            "total_tasks": len(self._task_manifest),
            "completed_count": len(self._completed_task_ids),
            "start_time": self._start_time,
        }

        self._atomic_write(self.state_path, json.dumps(state_data, indent=2))

    def _save_results(self) -> None:
        """Save results file in standard export format with atomic write.

        Uses the same v2.0 format as export_verification_results_json for consistency:
        - metadata.verification_config: answering_model and parsing_model info
        - metadata.job_summary: total_questions, successful_count, etc.
        - shared_data.rubric_definition: global rubric traits (if available)
        """
        # Build rubric definition from global_rubric if provided
        # This is stored once in shared_data instead of per-result
        rubric_definition = None
        if self.global_rubric is not None:
            if hasattr(self.global_rubric, "model_dump"):
                rubric_definition = self.global_rubric.model_dump(mode="json", exclude_unset=True)
            elif hasattr(self.global_rubric, "get_trait_names"):
                rubric_definition = {"trait_names": self.global_rubric.get_trait_names()}

        # Get model info for verification_config
        answering_model = self.config.answering_models[0] if self.config.answering_models else None
        parsing_model = self.config.parsing_models[0] if self.config.parsing_models else None

        # Build export data in standard v2.0 format (same as export_verification_results_json)
        export_data: dict[str, Any] = {
            "format_version": self.RESULTS_FORMAT_VERSION,
            "metadata": {
                "export_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "karenina_version": get_karenina_version(),
                "job_id": f"progressive-{int(self._start_time or time.time())}",
                "verification_config": {
                    "answering_model": {
                        "provider": answering_model.model_provider if answering_model else None,
                        "name": answering_model.model_name if answering_model else None,
                        "temperature": answering_model.temperature if answering_model else None,
                        "interface": answering_model.interface if answering_model else None,
                    },
                    "parsing_model": {
                        "provider": parsing_model.model_provider if parsing_model else None,
                        "name": parsing_model.model_name if parsing_model else None,
                        "temperature": parsing_model.temperature if parsing_model else None,
                        "interface": parsing_model.interface if parsing_model else None,
                    },
                },
                "job_summary": {
                    "total_questions": self.total_tasks,
                    "successful_count": self.completed_count,
                    "failed_count": 0,  # Not tracked during progressive save
                    "start_time": self._start_time,
                    "end_time": None,  # Not complete yet
                    "total_duration": None,  # Not complete yet
                },
            },
            "shared_data": {
                "rubric_definition": rubric_definition,
            },
            "results": [],
        }

        # Add results
        for result in self._results:
            result_dict = result.model_dump(mode="json")
            export_data["results"].append(result_dict)

        self._atomic_write(self.tmp_path, json.dumps(export_data, indent=2, ensure_ascii=False))

    def _atomic_write(self, path: Path, content: str) -> None:
        """Write content to file atomically using write-rename pattern."""
        partial_path = path.with_suffix(path.suffix + ".partial")

        try:
            # Write to partial file
            with open(partial_path, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())

            # Atomic rename
            partial_path.replace(path)

        except Exception as e:
            # Clean up partial file on error
            if partial_path.exists():
                with contextlib.suppress(OSError):
                    partial_path.unlink()
            raise e


def generate_task_manifest(tasks: list[dict[str, Any]]) -> list[str]:
    """Generate task manifest (list of task IDs) from task queue.

    Args:
        tasks: Task queue from generate_task_queue()

    Returns:
        List of task ID strings
    """
    return [TaskIdentifier.from_task_dict(task).to_key() for task in tasks]


@dataclass
class ProgressiveJobStatus:
    """Status summary for a progressive save job."""

    # Basic info
    state_file_path: Path
    output_path: Path
    benchmark_path: str

    # Progress
    total_tasks: int
    completed_count: int
    pending_count: int

    # Task details
    completed_task_ids: list[str]
    pending_task_ids: list[str]

    # Timing
    created_at: str
    last_updated_at: str
    start_time: float | None

    # Config summary
    answering_models: list[str]
    parsing_models: list[str]
    replicate_count: int

    # File status
    tmp_file_exists: bool
    tmp_file_size: int | None

    @property
    def progress_percent(self) -> float:
        """Get progress as a percentage."""
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_count / self.total_tasks) * 100

    @property
    def elapsed_time(self) -> float | None:
        """Get elapsed time in seconds since job started."""
        if self.start_time is None:
            return None
        return time.time() - self.start_time

    def get_unique_question_ids(self, task_ids: list[str]) -> list[str]:
        """Extract unique question IDs from task IDs."""
        question_ids = set()
        for task_id in task_ids:
            # Task ID format: {question_id}\t{answering_key}\t{parsing_key}[\trepN]
            parts = task_id.split("\t")
            if parts:
                question_ids.add(parts[0])
        return sorted(question_ids)

    @property
    def completed_question_ids(self) -> list[str]:
        """Get unique question IDs that have been completed."""
        return self.get_unique_question_ids(self.completed_task_ids)

    @property
    def pending_question_ids(self) -> list[str]:
        """Get unique question IDs that are still pending."""
        return self.get_unique_question_ids(self.pending_task_ids)


def inspect_state_file(state_path: Path) -> ProgressiveJobStatus:
    """Inspect a progressive save state file and return status summary.

    Args:
        state_path: Path to the .state file

    Returns:
        ProgressiveJobStatus with job summary

    Raises:
        FileNotFoundError: If state file doesn't exist
        ValueError: If state file is invalid
    """
    if not state_path.exists():
        raise FileNotFoundError(f"State file not found: {state_path}")

    with open(state_path) as f:
        state_data = json.load(f)

    # Validate format version
    format_version = state_data.get("format_version")
    if format_version != ProgressiveSaveManager.STATE_FORMAT_VERSION:
        raise ValueError(
            f"Incompatible state format version: {format_version} "
            f"(expected {ProgressiveSaveManager.STATE_FORMAT_VERSION})"
        )

    # Extract config info
    config = state_data.get("config", {})
    answering_models = []
    for model in config.get("answering_models", []):
        model_name = model.get("model_name", "unknown")
        model_id = model.get("id", "")
        if model_id:
            answering_models.append(f"{model_id} ({model_name})")
        else:
            answering_models.append(model_name)

    parsing_models = []
    for model in config.get("parsing_models", []):
        model_name = model.get("model_name", "unknown")
        model_id = model.get("id", "")
        if model_id:
            parsing_models.append(f"{model_id} ({model_name})")
        else:
            parsing_models.append(model_name)

    # Get task lists
    task_manifest = state_data.get("task_manifest", [])
    completed_task_ids = state_data.get("completed_task_ids", [])
    pending_task_ids = list(set(task_manifest) - set(completed_task_ids))

    # Check tmp file
    output_path = Path(state_data.get("output_path", ""))
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_file_exists = tmp_path.exists()
    tmp_file_size = tmp_path.stat().st_size if tmp_file_exists else None

    return ProgressiveJobStatus(
        state_file_path=state_path,
        output_path=output_path,
        benchmark_path=state_data.get("benchmark_path", ""),
        total_tasks=state_data.get("total_tasks", len(task_manifest)),
        completed_count=state_data.get("completed_count", len(completed_task_ids)),
        pending_count=len(pending_task_ids),
        completed_task_ids=completed_task_ids,
        pending_task_ids=pending_task_ids,
        created_at=state_data.get("created_at", ""),
        last_updated_at=state_data.get("last_updated_at", ""),
        start_time=state_data.get("start_time"),
        answering_models=answering_models,
        parsing_models=parsing_models,
        replicate_count=config.get("replicate_count", 1),
        tmp_file_exists=tmp_file_exists,
        tmp_file_size=tmp_file_size,
    )
