"""Deterministic tables for the BixBench harness comparison."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import cast

from karenina.benchmark import ResultsIOManager
from karenina.schemas.verification import VerificationResult
from paper.bixbench_harness_comparison.config import (
    ExperimentCondition,
    HarnessLabel,
    ModelLabel,
)

TOOL_BUCKETS = {
    "Bash": "shell",
    "execute": "shell",
    "Write": "write",
    "write_file": "write",
    "Read": "read",
    "read_file": "read",
    "Edit": "edit",
    "edit_file": "edit",
    "TaskCreate": "todo",
    "TaskUpdate": "todo",
    "TaskOutput": "todo",
    "TaskStop": "todo",
    "write_todos": "todo",
    "update_todos": "todo",
    "update_todo": "todo",
    "task": "todo",
    "Glob": "glob",
    "glob": "glob",
    "ls": "glob",
    "Grep": "grep",
    "grep": "grep",
    "WebSearch": "search",
}
TOOL_BUCKET_NAMES = (
    "shell",
    "write",
    "read",
    "edit",
    "todo",
    "glob",
    "grep",
    "search",
    "other",
)


@dataclass(frozen=True)
class ResultSource:
    """One validated result export and its comparison condition."""

    condition: ExperimentCondition
    path: Path
    replicate_override: int | None = None


def iter_source_results(source: ResultSource) -> Iterator[VerificationResult]:
    """Load validated rows through ``ResultsIOManager``."""

    return cast(
        Iterator[VerificationResult],
        ResultsIOManager.iter_from_json(source.path),
    )


def _canonical_fields(
    loaded: list[tuple[ResultSource, VerificationResult]],
) -> dict[str, dict[str, str]]:
    fields: dict[str, dict[str, str]] = defaultdict(dict)
    for _source, result in loaded:
        if result.template is None:
            continue
        ground_truth = result.template.parsed_gt_response or {}
        for name in result.template.field_results or {}:
            value = ground_truth.get(name)
            fields[result.metadata.question_id].setdefault(
                name,
                "boolean" if isinstance(value, bool) else "numeric",
            )
    return dict(fields)


def _tool_bucket_counts(metrics: dict[str, object]) -> dict[str, int]:
    counts = dict.fromkeys(TOOL_BUCKET_NAMES, 0)
    raw_counts = metrics.get("tool_call_counts")
    if not isinstance(raw_counts, dict):
        return counts
    for raw_name, count in raw_counts.items():
        bucket = TOOL_BUCKETS.get(str(raw_name), "other")
        counts[bucket] += int(cast(int | float | str, count))
    return counts


def build_result_tables(
    sources: Iterable[ResultSource],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Build per-field and per-task rows, scoring missing timeout fields as zero."""

    loaded = [
        (source, result)
        for source in sources
        for result in iter_source_results(source)
    ]
    canonical = _canonical_fields(loaded)
    field_rows: list[dict[str, object]] = []
    task_rows: list[dict[str, object]] = []
    for source, result in loaded:
        template = result.template
        question_id = result.metadata.question_id
        replicate = source.replicate_override or result.metadata.replicate or 1
        results = template.field_results if template and template.field_results else {}
        scores = template.field_scores if template and template.field_scores else {}
        task_scores: list[float] = []
        task_correct: list[bool] = []
        for field, field_type in canonical.get(question_id, {}).items():
            correct = results.get(field) is True
            raw_score = scores.get(field)
            score = float(raw_score) if raw_score is not None else float(correct)
            task_scores.append(score)
            task_correct.append(correct)
            field_rows.append(
                {
                    "model": source.condition.model,
                    "harness": source.condition.harness,
                    "replicate": replicate,
                    "task": question_id,
                    "field": field,
                    "field_type": field_type,
                    "correct": correct,
                    "field_score": score,
                    "failure_category": (
                        result.metadata.failure.category
                        if result.metadata.failure is not None
                        else "pass"
                    ),
                }
            )
        metrics: dict[str, object] = (
            template.agent_metrics if template and template.agent_metrics else {}
        )
        task_row: dict[str, object] = {
                "model": source.condition.model,
                "harness": source.condition.harness,
                "replicate": replicate,
                "task": question_id,
                "task_pass": result.metadata.failure is None,
                "field_count": len(task_correct),
                "fields_correct": sum(task_correct),
                "graded_accuracy": mean(task_scores) if task_scores else 0.0,
                "execution_time": result.metadata.execution_time,
                "iterations": metrics.get("iterations"),
                "tool_calls": metrics.get("tool_calls"),
                "timed_out": (
                    result.metadata.failure is not None
                    and result.metadata.failure.category == "timeout"
                ),
                "result_id": result.metadata.result_id,
        }
        task_row.update(_tool_bucket_counts(metrics))
        task_rows.append(task_row)
    return field_rows, task_rows


def build_comparison_summary(
    field_rows: list[dict[str, object]],
    task_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Summarize each model, harness, and replicate cell."""

    fields_by_cell: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    tasks_by_cell: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in field_rows:
        fields_by_cell[(row["model"], row["harness"], row["replicate"])].append(row)
    for row in task_rows:
        tasks_by_cell[(row["model"], row["harness"], row["replicate"])].append(row)
    output: list[dict[str, object]] = []
    for cell in sorted(fields_by_cell, key=lambda item: tuple(map(str, item))):
        fields = fields_by_cell[cell]
        tasks = tasks_by_cell[cell]
        output.append(
            {
                "model": cell[0],
                "harness": cell[1],
                "replicate": cell[2],
                "tasks": len(tasks),
                "fields": len(fields),
                "graded_accuracy": mean(
                    float(cast(int | float | str, row["field_score"])) for row in fields
                ),
                "binary_accuracy": mean(bool(row["correct"]) for row in fields),
                "task_pass_rate": mean(bool(row["task_pass"]) for row in tasks),
                "timeouts": sum(bool(row["timed_out"]) for row in tasks),
            }
        )
    return output


def normalize_archive_model(value: str) -> ModelLabel:
    """Translate immutable archive labels to configured model labels."""

    lowered = value.lower()
    if lowered.startswith("qwen"):
        return "Qwen 3.5 122B A10B"
    if lowered.startswith("glm"):
        return "GLM-5.1"
    if lowered.startswith("opus") or lowered.startswith("claude"):
        return "Claude Opus 4.6"
    raise ValueError(f"Unknown archived BixBench model label: {value}")


def normalize_archive_harness(value: str) -> HarnessLabel:
    """Translate immutable archive labels to configured harness labels."""

    lowered = value.lower()
    if lowered.startswith("csdk") or "claude" in lowered:
        return "Claude Code"
    if lowered.startswith("da") or "deep" in lowered:
        return "DeepAgents"
    raise ValueError(f"Unknown archived BixBench harness label: {value}")


def load_stored_burdens(path: Path) -> list[dict[str, object]]:
    """Load and normalize the curator-approved archived GLM judgments."""

    with path.open(encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows: list[dict[str, object]] = []
    for row in raw_rows:
        raw_replicate = row.get("rep") or row.get("replicate") or ""
        replicate = int(raw_replicate.removeprefix("rep"))
        rows.append(
            {
                **row,
                "model": normalize_archive_model(row["model"]),
                "harness": normalize_archive_harness(row["harness"]),
                "replicate": replicate,
                "task": row.get("task_id") or row.get("task"),
            }
        )
    return rows


def validate_burden_alignment(
    task_rows: list[dict[str, object]],
    burden_rows: list[dict[str, object]],
) -> None:
    """Require exactly one burden judgment for every selected task instance."""

    key_fields = ("model", "harness", "replicate", "task")
    task_keys = {tuple(row[field] for field in key_fields) for row in task_rows}
    burden_keys = {tuple(row[field] for field in key_fields) for row in burden_rows}
    if task_keys != burden_keys or len(burden_rows) != len(burden_keys):
        missing = sorted(task_keys - burden_keys, key=lambda key: tuple(map(str, key)))
        orphan = sorted(burden_keys - task_keys, key=lambda key: tuple(map(str, key)))
        raise ValueError(
            "Failure-burden judgments do not align one-to-one with task results. "
            f"Missing sample: {missing[:3]}. Orphan sample: {orphan[:3]}."
        )


def build_burden_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Average each ordinal failure burden by model and harness."""

    axes = (
        "environment_failure_burden",
        "data_ingestion_failure_burden",
        "tool_api_failure_burden",
        "analysis_code_failure_burden",
        "repetition_churn_burden",
        "incompletion_burden",
    )
    grouped: dict[tuple[object, object], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["harness"])].append(row)
    output: list[dict[str, object]] = []
    for (model, harness), group in sorted(
        grouped.items(), key=lambda item: tuple(map(str, item[0]))
    ):
        for axis in axes:
            values = [
                float(cast(int | float | str, row[axis]))
                for row in group
                if row.get(axis) not in {None, ""}
            ]
            output.append(
                {
                    "model": model,
                    "harness": harness,
                    "axis": axis.removesuffix("_failure_burden").removesuffix("_burden"),
                    "mean_burden": mean(values) if values else None,
                    "judgments": len(values),
                }
            )
    return output


def _write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty analysis table: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_analysis(
    output_dir: Path,
    field_rows: list[dict[str, object]],
    task_rows: list[dict[str, object]],
    burden_rows: list[dict[str, object]],
) -> None:
    """Write all deterministic analysis tables."""

    validate_burden_alignment(task_rows, burden_rows)
    _write_tsv(output_dir / "fields.tsv", field_rows)
    _write_tsv(output_dir / "tasks.tsv", task_rows)
    _write_tsv(
        output_dir / "comparison_summary.tsv",
        build_comparison_summary(field_rows, task_rows),
    )
    _write_tsv(output_dir / "failure_burdens.tsv", burden_rows)
    _write_tsv(
        output_dir / "failure_burden_summary.tsv",
        build_burden_summary(burden_rows),
    )
