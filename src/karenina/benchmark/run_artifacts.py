"""Standard run directories and manifests for verification workflows."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from karenina.schemas.verification import ModelIdentity, VerificationConfig
from karenina.utils.file_ops import atomic_write

RUN_MANIFEST_SCHEMA_VERSION = "1.0"
RUN_MANIFEST_FILENAME = "run_manifest.json"
RESULTS_FILENAME = "results.json"
_SAFE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SECRET_FRAGMENTS = ("api_key", "apikey", "password", "secret", "token")
_MASKED_VALUE = "**********"

RunStatus = Literal["running", "completed", "interrupted", "failed"]


class RunManifest(BaseModel):
    """Machine-readable description of one verification run."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = RUN_MANIFEST_SCHEMA_VERSION
    status: RunStatus
    label: str
    run_name: str
    created_at: str
    updated_at: str
    completed_at: str | None = None
    benchmark_path: str
    selected_question_count: int | None = Field(default=None, ge=0)
    replicate_count: int = Field(ge=1)
    answering_models: list[ModelIdentity]
    parsing_models: list[ModelIdentity]
    configuration: dict[str, JsonValue]
    results_path: str = RESULTS_FILENAME
    traces_path: str = "traces"
    workspaces_path: str = "workspaces"
    failure_type: str | None = None
    failure_message: str | None = None


class RunDirectory(BaseModel):
    """Paths and mutable manifest lifecycle for one run directory."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    path: Path
    manifest: RunManifest

    @property
    def manifest_path(self) -> Path:
        """Return the manifest file path."""
        return self.path / RUN_MANIFEST_FILENAME

    @property
    def results_path(self) -> Path:
        """Return the standard result export path."""
        return self.path / self.manifest.results_path

    @property
    def traces_path(self) -> Path:
        """Return the standard readable-trace directory."""
        return self.path / self.manifest.traces_path

    @property
    def workspaces_path(self) -> Path:
        """Return the standard workspace-sidecar directory."""
        return self.path / self.manifest.workspaces_path

    def update_status(
        self,
        status: RunStatus,
        *,
        now: datetime | None = None,
        error: BaseException | None = None,
    ) -> None:
        """Update and persist the lifecycle status.

        Args:
            status: New run status.
            now: Timestamp override for deterministic callers and tests.
            error: Failure recorded when status is ``failed``.
        """
        timestamp = _format_time(now or datetime.now(UTC))
        updates: dict[str, JsonValue | None] = {
            "status": status,
            "updated_at": timestamp,
            "completed_at": timestamp if status != "running" else None,
            "failure_type": type(error).__name__ if error is not None else None,
            "failure_message": str(error) if error is not None else None,
        }
        self.manifest = self.manifest.model_copy(update=updates)
        self.write_manifest()

    def write_manifest(self) -> None:
        """Persist the current manifest atomically."""
        atomic_write(
            self.manifest_path,
            json.dumps(self.manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        )


def _format_time(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _mask_configuration(value: JsonValue, *, key: str = "") -> JsonValue:
    lowered = key.lower()
    if any(fragment in lowered for fragment in _SECRET_FRAGMENTS):
        return _MASKED_VALUE if value is not None else None
    if isinstance(value, dict):
        return {str(child_key): _mask_configuration(child, key=str(child_key)) for child_key, child in value.items()}
    if isinstance(value, list):
        return [_mask_configuration(item) for item in value]
    return value


def mask_run_configuration(configuration: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """Return a JSON configuration snapshot with credential fields masked.

    Args:
        configuration: Configuration mapping to sanitize.

    Returns:
        A deep copy safe to write into a run manifest.
    """
    return {str(key): _mask_configuration(value, key=str(key)) for key, value in configuration.items()}


def create_run_directory(
    root: Path,
    label: str,
    *,
    benchmark_path: str | Path,
    config: VerificationConfig,
    run_name: str | None = None,
    selected_question_count: int | None = None,
    now: datetime | None = None,
    resume: bool = False,
) -> RunDirectory:
    """Create a timestamped run directory and running manifest.

    Args:
        root: Parent directory for verification runs.
        label: Filesystem-safe reviewer-facing run label.
        benchmark_path: Benchmark source recorded for provenance.
        config: Verification configuration to describe and sanitize.
        run_name: Result metadata run name. Defaults to ``label``.
        selected_question_count: Number of questions selected for this run.
        now: Timestamp override for deterministic callers and tests.
        resume: Reopen the timestamped directory when it already exists.

    Returns:
        Newly created run directory descriptor.

    Raises:
        ValueError: If the label is unsafe.
        FileExistsError: If the timestamped directory exists without resume.
    """
    if not _SAFE_LABEL.fullmatch(label):
        raise ValueError("Run label must contain only letters, digits, dots, underscores, and hyphens")
    created = now or datetime.now(UTC)
    directory_name = f"{label}_{created.astimezone(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    path = Path(root) / directory_name
    if path.exists():
        if not resume:
            raise FileExistsError(path)
        manifest_path = path / RUN_MANIFEST_FILENAME
        manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        run_directory = RunDirectory(path=path, manifest=manifest)
        run_directory.traces_path.mkdir(exist_ok=True)
        run_directory.workspaces_path.mkdir(exist_ok=True)
        run_directory.update_status("running", now=created)
        return run_directory
    path.mkdir(parents=True, exist_ok=False)
    traces_path = path / "traces"
    workspaces_path = path / "workspaces"
    traces_path.mkdir()
    workspaces_path.mkdir()

    raw_configuration = config.model_dump(mode="json", exclude={"manual_traces": True})
    manifest = RunManifest(
        status="running",
        label=label,
        run_name=run_name or label,
        created_at=_format_time(created),
        updated_at=_format_time(created),
        benchmark_path=str(benchmark_path),
        selected_question_count=selected_question_count,
        replicate_count=config.replicate_count,
        answering_models=[ModelIdentity.from_model_config(model) for model in config.answering_models],
        parsing_models=[ModelIdentity.from_model_config(model, role="parsing") for model in config.parsing_models],
        configuration=mask_run_configuration(raw_configuration),
    )
    run_directory = RunDirectory(path=path, manifest=manifest)
    run_directory.write_manifest()
    return run_directory


@contextmanager
def managed_run_directory(
    root: Path,
    label: str,
    *,
    benchmark_path: str | Path,
    config: VerificationConfig,
    run_name: str | None = None,
    selected_question_count: int | None = None,
    now: datetime | None = None,
    resume: bool = False,
) -> Iterator[RunDirectory]:
    """Create a run directory and maintain its terminal manifest status.

    Args:
        root: Parent directory for verification runs.
        label: Filesystem-safe reviewer-facing run label.
        benchmark_path: Benchmark source recorded for provenance.
        config: Verification configuration to describe and sanitize.
        run_name: Result metadata run name. Defaults to ``label``.
        selected_question_count: Number of selected questions.
        now: Creation timestamp override.
        resume: Reopen the timestamped directory when it already exists.

    Yields:
        The running directory descriptor.
    """
    run = create_run_directory(
        root,
        label,
        benchmark_path=benchmark_path,
        config=config,
        run_name=run_name,
        selected_question_count=selected_question_count,
        now=now,
        resume=resume,
    )
    try:
        yield run
    except KeyboardInterrupt:
        run.update_status("interrupted")
        raise
    except BaseException as exc:
        run.update_status("failed", error=exc)
        raise
    else:
        run.update_status("completed")
