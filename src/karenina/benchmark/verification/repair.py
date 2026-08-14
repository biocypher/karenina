"""Repair selected rows of a completed verification result export."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field

from karenina.benchmark.core.results_io import ResultsIOManager
from karenina.benchmark.verification.stages.helpers.results_exporter import export_verification_results_json_stream
from karenina.replay import capture_from_result_set
from karenina.schemas.results import ResultRowKey, VerificationResultSet
from karenina.schemas.verification import ModelIdentity, VerificationConfig, VerificationJob, VerificationResult
from karenina.utils.file_ops import atomic_write

RepairMode = Literal["replay", "live"]


class _RepairBenchmark(Protocol):
    def run_verification(
        self,
        *,
        config: VerificationConfig,
        question_ids: list[str],
        run_name: str,
    ) -> VerificationResultSet: ...


class RepairSelection(BaseModel):
    """Declarative row filters for completed-run repair."""

    model_config = ConfigDict(extra="forbid")

    question_ids: set[str] = Field(default_factory=set)
    answerer_keys: set[str] = Field(default_factory=set)
    parser_keys: set[str] = Field(default_factory=set)
    replicates: set[int | None] = Field(default_factory=set)
    failure_groups: set[str] = Field(default_factory=set)
    failure_categories: set[str] = Field(default_factory=set)
    failure_stages: set[str] = Field(default_factory=set)
    select_all: bool = False

    def has_filters(self) -> bool:
        """Return whether at least one explicit selection constraint exists."""
        return bool(
            self.select_all
            or self.question_ids
            or self.answerer_keys
            or self.parser_keys
            or self.replicates
            or self.failure_groups
            or self.failure_categories
            or self.failure_stages
        )

    def matches(self, result: VerificationResult) -> bool:
        """Return whether a result satisfies every configured filter."""
        if not self.has_filters():
            raise ValueError("Repair selection requires at least one filter or select_all=True")
        key = ResultRowKey.from_result(result)
        failure = result.metadata.failure
        failure_group = failure.group.value if failure is not None else None
        failure_category = failure.category.value if failure is not None else None
        failure_stage = failure.stage if failure is not None else None
        return bool(
            (not self.question_ids or key.question_id in self.question_ids)
            and (not self.answerer_keys or key.answering_key in self.answerer_keys)
            and (not self.parser_keys or key.parsing_key in self.parser_keys)
            and (not self.replicates or key.replicate in self.replicates)
            and (not self.failure_groups or failure_group in self.failure_groups)
            and (not self.failure_categories or failure_category in self.failure_categories)
            and (not self.failure_stages or failure_stage in self.failure_stages)
        )


class RepairOutcome(BaseModel):
    """Paths and counts produced by a completed repair."""

    model_config = ConfigDict(extra="forbid")

    source_path: Path
    output_path: Path
    backup_path: Path | None
    provenance_path: Path | None
    selected_count: int
    replaced_count: int
    mode: RepairMode
    dry_run: bool
    selected_keys: list[str]


def _key_string(key: ResultRowKey) -> str:
    replicate = "none" if key.replicate is None else str(key.replicate)
    return "|".join((key.question_id, key.answering_key, key.parsing_key, replicate, key.run_name or ""))


def _index_unique(results: Iterable[VerificationResult]) -> dict[ResultRowKey, VerificationResult]:
    indexed: dict[ResultRowKey, VerificationResult] = {}
    for result in results:
        key = ResultRowKey.from_result(result)
        if key in indexed:
            raise ValueError(f"Duplicate result row identity: {_key_string(key)}")
        indexed[key] = result
    return indexed


def select_repair_rows(
    results: Iterable[VerificationResult],
    selection: RepairSelection | Callable[[VerificationResult], bool],
) -> list[VerificationResult]:
    """Select completed rows for repair.

    Args:
        results: Validated verification results.
        selection: Declarative filters or a Python predicate.

    Returns:
        Selected rows in source order.
    """
    predicate = selection.matches if isinstance(selection, RepairSelection) else selection
    return [result for result in results if predicate(result)]


def splice_repaired_rows(
    original: Iterable[VerificationResult],
    replacements: Iterable[VerificationResult],
) -> list[VerificationResult]:
    """Replace exact row identities while preserving source order.

    Args:
        original: Complete original result rows.
        replacements: Newly evaluated rows with matching identities.

    Returns:
        Original rows with exact matching replacements spliced in.

    Raises:
        ValueError: If either input contains duplicates or a replacement has no
            corresponding original row.
    """
    original_rows = list(original)
    original_index = _index_unique(original_rows)
    replacement_index = _index_unique(replacements)
    unknown = set(replacement_index) - set(original_index)
    if unknown:
        key = sorted(unknown, key=_key_string)[0]
        raise ValueError(f"Replacement has no original row: {_key_string(key)}")
    return [replacement_index.get(ResultRowKey.from_result(row), row) for row in original_rows]


def _model_keys(config: VerificationConfig) -> tuple[list[str], list[str]]:
    answerers = [
        ModelIdentity.from_model_config(model, role="answering").canonical_key for model in config.answering_models
    ]
    parsers = [ModelIdentity.from_model_config(model, role="parsing").canonical_key for model in config.parsing_models]
    return answerers, parsers


def _targeted_config(
    config: VerificationConfig,
    selected: list[VerificationResult],
    mode: RepairMode,
) -> VerificationConfig:
    selected_keys = {ResultRowKey.from_result(row) for row in selected}
    selected_triples = {(key.question_id, key.answering_key, key.parsing_key, key.replicate) for key in selected_keys}
    answerer_keys, parser_keys = _model_keys(config)
    missing_answerers = {key.answering_key for key in selected_keys} - set(answerer_keys)
    missing_parsers = {key.parsing_key for key in selected_keys} - set(parser_keys)
    if missing_answerers or missing_parsers:
        raise ValueError(
            "Repair config does not contain every selected model identity: "
            f"answerers={sorted(missing_answerers)}, parsers={sorted(missing_parsers)}"
        )
    question_ids = {key.question_id for key in selected_keys}
    replicates: list[int | None] = [None] if config.replicate_count == 1 else list(range(1, config.replicate_count + 1))
    skip_triples = frozenset(
        (question_id, answerer, parser, replicate)
        for question_id in question_ids
        for answerer in answerer_keys
        for parser in parser_keys
        for replicate in replicates
        if (question_id, answerer, parser, replicate) not in selected_triples
    )
    updates: dict[str, object] = {"skip_triples": skip_triples}
    if mode == "replay":
        store = capture_from_result_set(
            VerificationResultSet(results=selected),
            include_parsed=False,
            include_agent_traces=True,
            only_successful=False,
            replicate_selector="all",
        )
        store.miss_policy = "strict"
        updates["replay_store"] = store
    else:
        updates["replay_store"] = None
    return config.model_copy(update=updates)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _export(path: Path, rows: list[VerificationResult], config: VerificationConfig, run_name: str) -> None:
    successful = sum(row.metadata.failure is None for row in rows)
    job = VerificationJob(
        job_id="repair",
        run_name=run_name,
        status="completed",
        config=config,
        total_questions=len({row.metadata.question_id for row in rows}),
        processed_count=len(rows),
        successful_count=successful,
        failed_count=len(rows) - successful,
        percentage=100.0,
    )
    export_verification_results_json_stream(job, rows, is_complete=True, out_path=path)


def repair_results_export(
    benchmark: _RepairBenchmark,
    source_path: Path,
    config: VerificationConfig,
    selection: RepairSelection | Callable[[VerificationResult], bool],
    *,
    mode: RepairMode = "replay",
    output_path: Path | None = None,
    dry_run: bool = False,
    now: datetime | None = None,
) -> RepairOutcome:
    """Rerun and splice selected rows of a completed JSON export.

    Args:
        benchmark: Loaded ``Benchmark`` exposing ``run_verification``.
        source_path: Completed result export to load with ``ResultsIOManager``.
        config: Full model configuration corresponding to the source run.
        selection: Declarative filters or a Python predicate.
        mode: ``replay`` reuses answer traces. ``live`` regenerates answers.
        output_path: Destination. Defaults to replacing ``source_path`` safely.
        dry_run: Select and report rows without model calls or file changes.
        now: Timestamp override for backup and provenance names.

    Returns:
        Repair paths, counts, mode, and selected identities.
    """
    source_path = Path(source_path)
    original_iter = cast(Iterator[VerificationResult], ResultsIOManager.iter_from_json(source_path))
    original = list(original_iter)
    _index_unique(original)
    selected = select_repair_rows(original, selection)
    if not selected:
        raise ValueError("Repair selection matched no result rows")
    selected_keys = [_key_string(ResultRowKey.from_result(row)) for row in selected]
    destination = Path(output_path) if output_path is not None else source_path
    if dry_run:
        return RepairOutcome(
            source_path=source_path,
            output_path=destination,
            backup_path=None,
            provenance_path=None,
            selected_count=len(selected),
            replaced_count=0,
            mode=mode,
            dry_run=True,
            selected_keys=selected_keys,
        )

    run_names = {row.metadata.run_name for row in selected}
    if len(run_names) != 1:
        raise ValueError("Selected repair rows must share one run name")
    run_name = next(iter(run_names)) or "repaired_run"
    targeted_config = _targeted_config(config, selected, mode)
    question_ids = sorted({row.metadata.question_id for row in selected})
    repaired = benchmark.run_verification(
        config=targeted_config,
        question_ids=question_ids,
        run_name=run_name,
    )
    repaired_rows = list(repaired.results)
    if len(repaired_rows) != len(selected):
        raise ValueError(f"Repair produced {len(repaired_rows)} rows for {len(selected)} selected rows")
    merged = splice_repaired_rows(original, repaired_rows)

    timestamp = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_path: Path | None = None
    if destination.resolve() == source_path.resolve():
        backup_path = source_path.with_name(f"{source_path.name}.backup_{timestamp}")
        if backup_path.exists():
            raise FileExistsError(backup_path)
        shutil.copy2(source_path, backup_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_checksum = _sha256(source_path)
    _export(destination, merged, config, run_name)
    provenance_path = destination.with_name(f"{destination.name}.repair_{timestamp}.json")
    provenance = {
        "schema_version": "1.0",
        "timestamp": timestamp,
        "mode": mode,
        "source_path": str(source_path),
        "source_sha256": source_checksum,
        "output_path": str(destination),
        "backup_path": str(backup_path) if backup_path is not None else None,
        "selected_count": len(selected),
        "replaced_count": len(repaired_rows),
        "selected_keys": selected_keys,
    }
    atomic_write(provenance_path, json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return RepairOutcome(
        source_path=source_path,
        output_path=destination,
        backup_path=backup_path,
        provenance_path=provenance_path,
        selected_count=len(selected),
        replaced_count=len(repaired_rows),
        mode=mode,
        dry_run=False,
        selected_keys=selected_keys,
    )
