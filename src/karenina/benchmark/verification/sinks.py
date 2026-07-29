"""Result sinks for the verification pipeline.

A :class:`ResultSink` is a side-effect handler invoked by the batch runner
as each task completes. It is the single extension point for progressive
save, crash recovery, database auto-save, and any other per-result
persistence the caller may need.

Four implementations live here:

- :class:`ProgressiveFileSink`: appends each completed result to a JSONL
  sidecar and maintains a :file:`.state` manifest, so a crashed run can be
  resumed via :func:`VerificationConfig.skip_triples`. Replaces the legacy
  `ProgressiveSaveManager` full-rewrite strategy.
- :class:`AgenticProgressiveFileSink`: extends the progressive file sink with
  readable trace sidecars and parser-failure retry semantics for agentic
  workspace runs.
- :class:`CompositeSink`: fans out to several sinks in order; used when a
  single run wants both file progressive-save and database auto-save.
- :class:`InMemorySink`: test-friendly sink that just buffers results.

The protocol itself is duck-typed (see CLAUDE.md §8): adapters do not
inherit from :class:`ResultSink`.

Thread safety and lock hierarchy (T17): every sink whose lifecycle methods
mutate shared state holds an internal lock, so sink correctness does not
depend on the executors' progress-lock serialization (which remains in place
as belt and suspenders). The sink lock is INNERMOST: code holding a sink
lock must never call back into executor code or acquire another sink's lock,
with one sanctioned exception: :class:`CompositeSink` holds its own RLock
while fanning out to its children, which then take their own locks. That
one-level ordering (composite lock, then child lock) is the only permitted
nesting; children never call back into the composite.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from karenina.benchmark.verification.stages.helpers.results_exporter import (
    export_verification_results_json_stream,
)
from karenina.benchmark.verification.workspace_capture import safe_path_component
from karenina.schemas import VerificationConfig, VerificationResult
from karenina.schemas.entities import Rubric
from karenina.schemas.results import VerificationResultSet
from karenina.schemas.results.failure import FailureCategory
from karenina.schemas.verification import VerificationJob
from karenina.utils.file_ops import atomic_write
from karenina.utils.progressive_save import TaskIdentifier

logger = logging.getLogger(__name__)

# State file schema version for ProgressiveFileSink. Bump when on-disk shape
# changes in an incompatible way.
STATE_FORMAT_VERSION = "2.0"

# Canonical 4-tuple used by VerificationConfig.skip_triples.
TripleKey = tuple[str, str, str, int | None]


def is_completed_result(result: VerificationResult | None) -> bool:
    """Return True if ``result`` represents a finished task, not a preview.

    The executor emits a preview ``VerificationResult`` (with an empty
    ``metadata.timestamp``) when a task is about to start, and a real one
    once the task has finished. Sinks should act only on the latter.
    """
    if result is None:
        return False
    timestamp = getattr(result.metadata, "timestamp", "")
    return bool(timestamp)


def _failure_attr(failure: Any, name: str) -> Any:
    value = getattr(failure, name, None)
    return getattr(value, "value", value)


def _is_content_failure(failure: Any) -> bool:
    return bool(_failure_attr(failure, "group") == "content" or _failure_attr(failure, "category") == "content")


def _is_parser_stage(stage: str | None) -> bool:
    return stage in {"parse_template", "ParseTemplate", "AgenticParseTemplate", "DynamicParseTemplate"}


def _has_usable_answer_trace(result: VerificationResult) -> bool:
    template = result.template
    if template is None:
        return False
    return bool((template.raw_llm_response or "").strip())


def _is_retryable_parser_failure(result: VerificationResult) -> bool:
    """Return True when a result should checkpoint answer output, not skip resume.

    Agentic workspace runs can fail after answer generation, during the
    investigation or structured extraction/parser phase. Those rows contain a
    useful answer trace and workspace artifacts, so resume should rehydrate the
    answer cache and retry parsing/verification instead of rerunning the
    answerer or treating the task as terminal.

    Timeout rows with a captured partial trace are handled the same way when a
    later resume enables partial trace scoring: the answer trace is usable
    parser input, so the task must not be skipped just because the timeout row
    already exists in the JSONL sidecar.
    """

    failure = result.metadata.failure
    if failure is None or not _has_usable_answer_trace(result):
        return False

    category = _failure_attr(failure, "category")
    stage = _failure_attr(failure, "stage")
    if stage == "GenerateAnswer" and category == FailureCategory.TIMEOUT.value:
        return True
    return bool(category == FailureCategory.PARSING.value or _is_parser_stage(stage))


def _export_scenario_turn(row: VerificationResult) -> dict[str, Any]:
    template = row.template
    return {
        "node_id": row.metadata.scenario_node,
        "question_text": row.metadata.question_text,
        "raw_response": template.raw_llm_response if template is not None else "",
        "parsed_fields": dict((template.parsed_llm_response if template is not None else None) or {}),
        "verify_result": template.verify_result if template is not None else None,
        "verification_result_id": row.metadata.result_id,
    }


def _scenario_manifest_key(result: VerificationResult, manifest: Iterable[str]) -> str:
    fallback = TaskIdentifier.from_result(result).to_key()
    scenario_id = result.metadata.scenario_id
    if not scenario_id:
        return fallback

    candidates: list[str] = []
    for key in manifest:
        try:
            ident = TaskIdentifier.from_key(key)
        except ValueError:
            continue
        if ident.question_id == scenario_id and ident.replicate == result.metadata.replicate:
            candidates.append(key)
    if len(candidates) == 1:
        return candidates[0]
    if fallback in candidates:
        return fallback
    return fallback


def _dedupe_latest_by_manifest_key(
    rows: Iterable[VerificationResult],
    *,
    manifest: Iterable[str] = (),
) -> list[VerificationResult]:
    """Keep the latest row for each manifest key while preserving first-key order."""

    latest: dict[str, VerificationResult] = {}
    order: list[str] = []
    for row in rows:
        key = _scenario_manifest_key(row, manifest)
        if key not in latest:
            order.append(key)
        latest[key] = row
    return [latest[key] for key in order]


def _count_success_failure(rows: Iterable[VerificationResult]) -> tuple[int, int]:
    """Count passing and failing rows using the structured failure marker."""

    successful = 0
    failed = 0
    for row in rows:
        if row.metadata.failure is None:
            successful += 1
        else:
            failed += 1
    return successful, failed


def _reconstruct_scenario_results_for_export(
    rows: Iterable[VerificationResult],
    *,
    manifest: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Build top-level scenario summaries from per-node sink rows.

    The progressive sink persists only completed ``VerificationResult`` rows.
    For scenario runs those rows are node turns, so the final sink export must
    rebuild the compact scenario view that callers get from
    ``Benchmark.run_verification``.
    """
    grouped: dict[str, list[VerificationResult]] = {}
    first_seen: list[str] = []
    for row in rows:
        scenario_id = row.metadata.scenario_id
        if not scenario_id:
            continue
        key = _scenario_manifest_key(row, manifest)
        if key not in grouped:
            first_seen.append(key)
            grouped[key] = []
        grouped[key].append(row)

    scenario_results: list[dict[str, Any]] = []
    for key in first_seen:
        turn_rows = sorted(grouped[key], key=lambda row: row.metadata.scenario_turn or 0)
        if not turn_rows:
            continue
        last_row = turn_rows[-1]
        last_failure = last_row.metadata.failure
        status = "error" if last_failure is not None and not _is_content_failure(last_failure) else "completed"
        path = [row.metadata.scenario_node or "" for row in turn_rows]
        terminal_failure = None
        if status == "error" and last_failure is not None:
            terminal_failure = {
                "node_id": last_row.metadata.scenario_node,
                "category": _failure_attr(last_failure, "category"),
                "stage": getattr(last_failure, "stage", None),
                "reason": getattr(last_failure, "reason", None),
            }
        ident = TaskIdentifier.from_key(key)
        scenario_results.append(
            {
                "scenario_id": last_row.metadata.scenario_id,
                "status": status,
                "path": path,
                "turn_count": len(turn_rows),
                "replicate": ident.replicate,
                "terminal_failure": terminal_failure,
                "outcome_results": {},
                "history": [_export_scenario_turn(row) for row in turn_rows],
            }
        )
    return scenario_results


@runtime_checkable
class ResultSink(Protocol):
    """Side-effect handler invoked by the batch runner.

    Lifecycle:

    1. ``seed_prior_results(prior_results)`` is optional and only used by the
       ``extend_template`` / ``extend_rubric`` flows. It tells the sink about
       rows that exist outside this batch so the sink's persisted state and
       final export reflect the merged (prior + new) shape rather than only
       the new work. Called before ``on_start``.
    2. ``on_start(manifest, config)`` is called once, before any task runs.
       The sink may advertise already-completed triples via
       :meth:`completed_triples` before this call so the caller can populate
       ``config.skip_triples`` and avoid resubmitting work.
    3. ``on_result(result)`` is called once per completed task, in completion
       order. The executor may also emit preview results (task started,
       no timestamp); sinks must ignore those via :func:`is_completed_result`.
    4. ``on_finalize(all_complete)`` is called exactly once when the batch
       returns. ``all_complete=True`` means every task in the manifest
       produced a successful result; ``False`` means something failed or
       was cancelled. Sinks use this to decide whether to keep or delete
       persistent state.
    """

    def completed_triples(self) -> set[TripleKey]:
        """Return the set of (question_id, ans_key, parse_key, replicate)
        triples already persisted by this sink, to be fed into
        :attr:`VerificationConfig.skip_triples` on resume."""
        ...

    def seed_prior_results(self, prior_results: VerificationResultSet) -> None:
        """Pre-populate the sink with already-completed rows.

        Used by the ``extend_*`` facades so prior rows land in the sink's
        buffer / persisted state and the final export reflects the merged
        shape. Implementations must be idempotent (safe to call with rows
        whose triples are already tracked) and must not require ``on_start``
        to have run first.
        """
        ...

    def on_start(self, manifest: list[str], config: VerificationConfig) -> None:
        """Called once before task execution begins."""
        ...

    def on_result(self, result: VerificationResult) -> None:
        """Called once per completed task. Preview results are filtered out."""
        ...

    def on_finalize(self, *, all_complete: bool) -> None:
        """Called exactly once when the batch returns."""
        ...


# ---------------------------------------------------------------------------
# ProgressiveFileSink
# ---------------------------------------------------------------------------


class ProgressiveFileSink:
    """Append-only JSONL sidecar + ``.state`` manifest for crash recovery.

    On disk during a run (paths relative to ``output_path``):

    - ``<output>.state``: JSON manifest. Contains the task manifest, the
      completed / failed triple sets, a config snapshot, and a config hash.
      Rewritten atomically on every completion.
    - ``<output>.results.jsonl``: one JSON-serialized :class:`VerificationResult`
      per line, appended as tasks complete.

    On successful completion (:meth:`on_finalize` with ``all_complete=True``):

    - The JSONL sidecar is reassembled into the final frontend-export JSON
      at ``output_path`` via :func:`export_verification_results_json_stream`.
    - The ``.state`` and ``.results.jsonl`` files are deleted.

    On partial completion (:meth:`on_finalize` with ``all_complete=False``):

    - Both sidecars are kept so ``--resume`` can continue.

    Resume path: :meth:`load_for_resume` reads the state and JSONL and
    returns a sink populated with the already-completed triples.
    """

    def __init__(
        self,
        output_path: Path,
        config: VerificationConfig,
        benchmark_path: str,
        global_rubric: Rubric | None = None,
    ) -> None:
        """Initialize a fresh sink targeting ``output_path``.

        Args:
            output_path: Final export target (e.g. ``results.json``). The
                sidecars live next to it at ``results.json.state`` and
                ``results.json.results.jsonl``.
            config: Verification configuration to snapshot into the state.
            benchmark_path: Path of the benchmark file, stored in the state
                for reference and diagnostics.
            global_rubric: Optional rubric folded into the final export.
        """
        self.output_path = output_path
        self.state_path = output_path.with_suffix(output_path.suffix + ".state")
        self.jsonl_path = output_path.with_suffix(output_path.suffix + ".results.jsonl")
        self.config = config
        self.benchmark_path = benchmark_path
        self.global_rubric = global_rubric

        self._manifest: list[str] = []
        self._completed: set[str] = set()
        self._failed: set[str] = set()
        self._start_time: float | None = None
        self._config_hash: str = ""
        # In-memory mirror of JSONL content, built from appends and from
        # load_for_resume so finalize() can reassemble without re-reading.
        self._results: list[VerificationResult] = []
        # Internal sink lock guarding the JSONL append, the state rewrite,
        # and the completed/failed bookkeeping. Reentrant so subclass
        # overrides (AgenticProgressiveFileSink) can hold it around a
        # super() call. Innermost lock: never call executor code while held.
        self._lock = threading.RLock()

    # -- Resume ------------------------------------------------------------

    @classmethod
    def load_for_resume(
        cls,
        state_path: Path,
        *,
        global_rubric: Rubric | None = None,
    ) -> ProgressiveFileSink:
        """Reconstruct a sink from an existing ``.state`` + JSONL pair.

        Args:
            state_path: Path to the ``.state`` file on disk.
            global_rubric: Rubric to attach to the sink for final export.

        Returns:
            A ``ProgressiveFileSink`` whose :meth:`completed_triples` returns
            the set of triples already persisted, and whose internal
            ``_results`` buffer is pre-populated.

        Raises:
            FileNotFoundError: If ``state_path`` does not exist.
            ValueError: If the state format version is incompatible.
        """
        if not state_path.exists():
            raise FileNotFoundError(f"State file not found: {state_path}")

        with open(state_path) as f:
            state_data = json.load(f)

        fmt = state_data.get("format_version")
        if fmt != STATE_FORMAT_VERSION:
            raise ValueError(f"Incompatible state format version: {fmt} (expected {STATE_FORMAT_VERSION})")

        output_path = Path(state_data["output_path"])
        config = VerificationConfig(**state_data["config"])
        benchmark_path = state_data["benchmark_path"]

        sink = cls(output_path, config, benchmark_path, global_rubric=global_rubric)
        sink._manifest = list(state_data.get("task_manifest", []))
        sink._completed = set(state_data.get("completed_task_ids", []))
        sink._failed = set(state_data.get("failed_task_ids", []))
        sink._start_time = state_data.get("start_time")
        sink._config_hash = state_data.get("config_hash", "")

        if sink.jsonl_path.exists():
            sink._results = list(_read_jsonl_results(sink.jsonl_path))
            # Cross-check: _completed should match the JSONL contents. If the
            # JSONL has more entries than the state tracked (e.g. a crash
            # right after append, before state rewrite), trust the JSONL.
            for result in sink._results:
                sink._completed.add(_scenario_manifest_key(result, sink._manifest))

        logger.info(
            "Loaded progressive sink: %d/%d tasks completed (%d failed)",
            len(sink._completed),
            len(sink._manifest),
            len(sink._failed),
        )
        return sink

    # -- Seeding ----------------------------------------------------------

    def seed_prior_results(self, prior_results: VerificationResultSet) -> None:
        """Pre-populate JSONL, ``_results``, ``_completed``, ``_manifest``.

        Called by the ``extend_*`` facades before ``run_verification`` so
        the sink's final export covers prior rows in addition to the new
        rows produced by the pipeline. Idempotent: rows whose triple is
        already tracked are skipped. Does not require :meth:`on_start` to
        have run first; the JSONL file is created on demand.
        """
        with self._lock:
            added = 0
            manifest_set = set(self._manifest)
            for row in prior_results.results:
                if not is_completed_result(row):
                    continue
                task_id = _scenario_manifest_key(row, self._manifest)
                if task_id in self._completed:
                    continue
                _append_jsonl_line(self.jsonl_path, row.model_dump_json())
                self._results.append(row)
                self._completed.add(task_id)
                self._failed.discard(task_id)
                if task_id not in manifest_set:
                    self._manifest.append(task_id)
                    manifest_set.add(task_id)
                added += 1
            if added:
                self._save_state()
                logger.info("ProgressiveFileSink seeded with %d prior rows", added)

    # -- ResultSink protocol ----------------------------------------------

    def completed_triples(self) -> set[TripleKey]:
        """Return already-persisted triples for ``config.skip_triples``."""
        with self._lock:
            completed_snapshot = set(self._completed)
        out: set[TripleKey] = set()
        for key in completed_snapshot:
            try:
                ident = TaskIdentifier.from_key(key)
            except ValueError:
                logger.warning("Skipping malformed task key in sink: %r", key)
                continue
            out.add(
                (
                    ident.question_id,
                    ident.answering_canonical_key,
                    ident.parsing_canonical_key,
                    ident.replicate,
                )
            )
        return out

    def on_start(self, manifest: list[str], config: VerificationConfig) -> None:
        """Record the manifest and write the initial state file.

        Merge semantics: the stored manifest is unioned with the provided
        one, preserving insertion order. This handles both the resume path
        (provided manifest is a subset of the stored full manifest) and
        the seeded-extend path (stored holds prior keys from
        :meth:`seed_prior_results`, provided holds the new-task keys
        from :func:`generate_task_queue`). Repeated keys are dropped.
        """
        with self._lock:
            if not self._manifest:
                self._manifest = list(manifest)
            else:
                existing = set(self._manifest)
                for key in manifest:
                    if key not in existing:
                        self._manifest.append(key)
                        existing.add(key)

            # Always refresh config + hash: a resume may pass a config that
            # was rehydrated from the state, but we keep whatever the caller
            # handed us so later callers get the current object identity.
            self.config = config
            if not self._config_hash:
                self._config_hash = _compute_config_hash(config)
            if self._start_time is None:
                self._start_time = time.time()

            self._save_state()
            # Touch the JSONL so readers can rely on its existence.
            if not self.jsonl_path.exists():
                self.jsonl_path.touch()

    def on_result(self, result: VerificationResult) -> None:
        """Append a completed result to the JSONL and refresh the state.

        The JSONL line is always written. The ``_completed`` set is
        idempotent: scenario runs emit one ``on_result`` per turn, all
        sharing the same combo-level task key (scenario_id-keyed), and the
        combo is only marked complete once even though N turn lines are
        appended. QA runs emit exactly one result per task so the guard
        is a no-op.
        """
        if not is_completed_result(result):
            return

        with self._lock:
            task_id = _scenario_manifest_key(result, self._manifest)
            _append_jsonl_line(self.jsonl_path, result.model_dump_json())
            self._results.append(result)
            if task_id not in self._completed:
                self._completed.add(task_id)
                self._failed.discard(task_id)
            self._save_state()

    def mark_failed(self, task_id: str) -> None:
        """Record a task that was attempted but did not produce a result.

        Optional helper for the batch runner so partial-failure state can
        be preserved across resumes. Not part of the :class:`ResultSink`
        protocol.
        """
        with self._lock:
            if task_id in self._completed:
                return
            self._failed.add(task_id)
            self._save_state()

    def on_finalize(self, *, all_complete: bool) -> None:
        """Handle terminal state.

        On ``all_complete=True``, write the final frontend-export JSON at
        :attr:`output_path` and remove the sidecars. On partial completion,
        keep the sidecars intact so ``resume`` can continue.
        """
        with self._lock:
            if all_complete:
                self.write_final_export()
                self._delete_sidecars()
                logger.info("ProgressiveFileSink finalized: full export at %s", self.output_path)
            else:
                # Preserve sidecars so a later `--resume` can continue.
                # Refresh the state one last time so the last batch's
                # activity lands.
                self._save_state()
                logger.info(
                    "ProgressiveFileSink left state in place for resume: %d/%d complete",
                    len(self._completed),
                    len(self._manifest),
                )

    # -- Accessors used by CLI / tests ------------------------------------

    @property
    def completed_count(self) -> int:
        """Number of completed tasks."""
        with self._lock:
            return len(self._completed)

    @property
    def total_tasks(self) -> int:
        """Total tasks in the manifest."""
        with self._lock:
            return len(self._manifest)

    @property
    def task_manifest(self) -> list[str]:
        """Task manifest keys recorded for this run."""
        with self._lock:
            return list(self._manifest)

    def get_all_results(self) -> list[VerificationResult]:
        """Return a shallow copy of buffered completed results."""
        with self._lock:
            return list(self._results)

    def iter_results(self) -> Iterable[VerificationResult]:
        """Iterate over a snapshot of buffered completed results.

        Used by the batch runner to hydrate the workspace
        :class:`AnswerTraceCache` on resume so new parser variants for
        already-completed answerer triples reuse the prior trace instead
        of regenerating it. The snapshot is taken under the sink lock so
        the iterator stays valid even if results land concurrently.
        """
        with self._lock:
            return iter(list(self._results))

    def get_result_set(self) -> VerificationResultSet:
        """Return buffered results as a :class:`VerificationResultSet`."""
        with self._lock:
            return VerificationResultSet(results=list(self._results))

    def set_global_rubric(self, rubric: Rubric | None) -> None:
        """Attach a global rubric used by the final export."""
        self.global_rubric = rubric

    # -- Internals --------------------------------------------------------

    def _save_state(self) -> None:
        """Atomically rewrite the ``.state`` file."""
        state = {
            "format_version": STATE_FORMAT_VERSION,
            "created_at": _fmt_utc(self._start_time or time.time()),
            "last_updated_at": _fmt_utc(time.time()),
            "benchmark_path": self.benchmark_path,
            "output_path": str(self.output_path),
            "config_hash": self._config_hash,
            "config": self.config.model_dump(mode="json", exclude={"manual_traces": True}),
            "task_manifest": self._manifest,
            "completed_task_ids": sorted(self._completed),
            "failed_task_ids": sorted(self._failed),
            "total_tasks": len(self._manifest),
            "completed_count": len(self._completed),
            "failed_count": len(self._failed),
            "start_time": self._start_time,
        }
        atomic_write(self.state_path, json.dumps(state, indent=2))

    def write_final_export(self, output_path: Path | None = None) -> Path:
        """Assemble and atomically write the frontend-export JSON.

        Args:
            output_path: Destination. Defaults to :attr:`output_path`.

        Returns:
            The path the export was written to.
        """
        with self._lock:
            target = output_path or self.output_path
            now = time.time()
            results = list(_read_jsonl_results(self.jsonl_path)) if self.jsonl_path.exists() else []
            latest_rows = _dedupe_latest_by_manifest_key(results, manifest=self._manifest)
            successful_count, failed_count = _count_success_failure(latest_rows)
            job = VerificationJob(
                job_id=f"progressive-{int(self._start_time or now)}",
                run_name="progressive",
                status="completed",
                config=self.config,
                total_questions=self.total_tasks,
                successful_count=successful_count,
                failed_count=failed_count,
                start_time=self._start_time,
                end_time=now,
            )
            # Stream from the JSONL sidecar if it exists. write_final_export
            # may be called before any on_result (e.g. the empty-run case),
            # in which case jsonl_path was never created; guard is mandatory
            # because _read_jsonl_results opens the path unconditionally.
            scenario_results = _reconstruct_scenario_results_for_export(results, manifest=self._manifest)
            export_verification_results_json_stream(
                job,
                iter(results),
                self.global_rubric,
                is_complete=True,
                scenario_results=scenario_results or None,
                out_path=target,
            )
            return target

    def _delete_sidecars(self) -> None:
        """Remove the ``.state`` and ``.results.jsonl`` files."""
        for path in (self.state_path, self.jsonl_path):
            try:
                if path.exists():
                    path.unlink()
            except OSError as e:
                logger.warning("Failed to remove %s: %s", path, e)


# ---------------------------------------------------------------------------
# AgenticProgressiveFileSink
# ---------------------------------------------------------------------------


class AgenticProgressiveFileSink(ProgressiveFileSink):
    """Progressive sink for agentic workspace benchmarks.

    This sink keeps the standard JSONL + state resume machinery, adds readable
    trace sidecars, and treats parser-stage failures with a usable answer trace
    as resumable checkpoints. On resume those rows are exposed through
    :meth:`iter_results` for answer-cache hydration, but omitted from
    :meth:`completed_triples` so parsing can be retried.
    """

    def __init__(
        self,
        output_path: Path,
        config: VerificationConfig,
        benchmark_path: str,
        global_rubric: Rubric | None = None,
        *,
        trace_output_dir: Path | None = None,
        trace_layout: str = "question",
        keep_progress_sidecars: bool = False,
        write_partial_export: bool = True,
    ) -> None:
        super().__init__(
            output_path=output_path,
            config=config,
            benchmark_path=benchmark_path,
            global_rubric=global_rubric,
        )
        if trace_layout not in {"question", "result"}:
            raise ValueError("trace_layout must be 'question' or 'result'")
        self.trace_output_dir = trace_output_dir
        self.trace_layout = trace_layout
        self.keep_progress_sidecars = keep_progress_sidecars
        self.write_partial_export = write_partial_export

    @classmethod
    def load_for_resume(
        cls,
        state_path: Path,
        *,
        global_rubric: Rubric | None = None,
        trace_output_dir: Path | None = None,
        trace_layout: str = "question",
        keep_progress_sidecars: bool = False,
        write_partial_export: bool = True,
    ) -> AgenticProgressiveFileSink:
        base = ProgressiveFileSink.load_for_resume(state_path, global_rubric=global_rubric)
        sink = cls(
            output_path=base.output_path,
            config=base.config,
            benchmark_path=base.benchmark_path,
            global_rubric=base.global_rubric,
            trace_output_dir=trace_output_dir,
            trace_layout=trace_layout,
            keep_progress_sidecars=keep_progress_sidecars,
            write_partial_export=write_partial_export,
        )
        sink._manifest = base.task_manifest
        sink._completed = set(base._completed)
        sink._failed = set(base._failed)
        sink._start_time = base._start_time
        sink._config_hash = base._config_hash
        sink._results = base.get_all_results()
        return sink

    def completed_triples(self) -> set[TripleKey]:
        """Return only terminal completed triples.

        Parser-stage failures are intentionally excluded so resume retries
        parsing while :meth:`iter_results` still exposes their answer traces
        for cache hydration.
        """

        with self._lock:
            latest_rows = _dedupe_latest_by_manifest_key(self._results, manifest=self._manifest)
            retryable = {
                _scenario_manifest_key(result, self._manifest)
                for result in latest_rows
                if _is_retryable_parser_failure(result)
            }
            remaining = self._completed - retryable
        out: set[TripleKey] = set()
        for key in remaining:
            try:
                ident = TaskIdentifier.from_key(key)
            except ValueError:
                logger.warning("Skipping malformed task key in sink: %r", key)
                continue
            out.add(
                (
                    ident.question_id,
                    ident.answering_canonical_key,
                    ident.parsing_canonical_key,
                    ident.replicate,
                )
            )
        return out

    def on_result(self, result: VerificationResult) -> None:
        if not is_completed_result(result):
            return
        # The inherited RLock is held across both the base bookkeeping and
        # the trace sidecar write so concurrent on_result calls cannot
        # interleave between the two.
        with self._lock:
            super().on_result(result)
            self._write_trace_sidecars(result)

    def on_finalize(self, *, all_complete: bool) -> None:
        with self._lock:
            effective_all_complete = all_complete and not self._has_retryable_parser_failures()
            if effective_all_complete:
                self.write_final_export(is_complete=True)
                if not self.keep_progress_sidecars:
                    self._delete_sidecars()
                logger.info("AgenticProgressiveFileSink finalized: full export at %s", self.output_path)
                return

            if self.write_partial_export:
                self.write_final_export(is_complete=False)
            self._save_state()
            logger.info(
                "AgenticProgressiveFileSink left state in place for resume: %d/%d complete",
                len(self._completed),
                len(self._manifest),
            )

    def get_latest_result_set(self) -> VerificationResultSet:
        """Return one latest result per task key, matching final export rows."""

        with self._lock:
            return VerificationResultSet(results=_dedupe_latest_by_manifest_key(self._results, manifest=self._manifest))

    def write_final_export(self, output_path: Path | None = None, *, is_complete: bool | None = None) -> Path:
        with self._lock:
            target = output_path or self.output_path
            now = time.time()
            rows = list(_read_jsonl_results(self.jsonl_path)) if self.jsonl_path.exists() else []
            latest_rows = _dedupe_latest_by_manifest_key(rows, manifest=self._manifest)
            successful_count, failed_count = _count_success_failure(latest_rows)
            job = VerificationJob(
                job_id=f"progressive-{int(self._start_time or now)}",
                run_name="progressive",
                status="completed",
                config=self.config,
                total_questions=self.total_tasks,
                successful_count=successful_count,
                failed_count=failed_count,
                start_time=self._start_time,
                end_time=now,
            )
            scenario_results = _reconstruct_scenario_results_for_export(latest_rows, manifest=self._manifest)
            export_verification_results_json_stream(
                job,
                iter(latest_rows),
                self.global_rubric,
                is_complete=is_complete
                if is_complete is not None
                else not self._has_retryable_parser_failures(latest_rows),
                scenario_results=scenario_results or None,
                out_path=target,
            )
            return target

    def _has_retryable_parser_failures(self, rows: Iterable[VerificationResult] | None = None) -> bool:
        source = (
            list(rows) if rows is not None else _dedupe_latest_by_manifest_key(self._results, manifest=self._manifest)
        )
        return any(_is_retryable_parser_failure(result) for result in source)

    def _trace_dir_for_result(self, result: VerificationResult) -> Path | None:
        if self.trace_output_dir is None:
            return None
        question = safe_path_component(result.metadata.question_id)
        if self.trace_layout == "result":
            result_id = safe_path_component(result.metadata.result_id)
            return self.trace_output_dir / f"{question}__{result_id}"
        return self.trace_output_dir / question

    def _write_trace_sidecars(self, result: VerificationResult) -> None:
        trace_dir = self._trace_dir_for_result(result)
        template = result.template
        if trace_dir is None or template is None:
            return
        trace_dir.mkdir(parents=True, exist_ok=True)
        if template.raw_llm_response:
            atomic_write(trace_dir / "answering_trace.txt", template.raw_llm_response)
        if template.investigation_trace:
            atomic_write(trace_dir / "investigation_trace.txt", template.investigation_trace)


# ---------------------------------------------------------------------------
# CompositeSink
# ---------------------------------------------------------------------------


class CompositeSink:
    """Fan-out wrapper that forwards every lifecycle call to its children.

    :meth:`completed_triples` returns the union across all children; this
    is conservative (if any sink reports a triple as complete, it is
    skipped) and matches the semantics callers expect: resuming with
    partial persistence should not redo work either store already has.

    Thread safety: every fan-out holds the composite's RLock for its full
    iteration, so two threads cannot interleave their child dispatches.
    Children take their own internal locks underneath. This composite-then-
    child nesting is the one sanctioned lock ordering (see the module
    docstring) and children never call back into the composite. A child
    exception propagates to the caller (per-child isolation is deferred to
    the clean-core redesign), but the ``with`` block guarantees the RLock is
    released, so the composite stays usable for subsequent events.
    """

    def __init__(self, sinks: Iterable[ResultSink]) -> None:
        """Initialize with an ordered iterable of child sinks.

        Args:
            sinks: Child sinks, called in iteration order on every event.
        """
        self._sinks: list[ResultSink] = list(sinks)
        # Reentrant: a fan-out method may be invoked while another composite
        # method on the same thread already holds the lock.
        self._lock = threading.RLock()

    @property
    def sinks(self) -> list[ResultSink]:
        """Child sinks in invocation order."""
        return list(self._sinks)

    def completed_triples(self) -> set[TripleKey]:
        """Union of completed triples across all children."""
        out: set[TripleKey] = set()
        with self._lock:
            for sink in self._sinks:
                out.update(sink.completed_triples())
        return out

    def iter_results(self) -> Iterable[VerificationResult]:
        """Iterate buffered results across children that expose them.

        Iteration order: first child to expose ``iter_results`` is drained
        first, then the next, and so on. The fan-out is used by the batch
        runner to hydrate the workspace cache on resume; duplicate rows
        across children are tolerated because cache hydration is idempotent
        per cache key.

        The child list is snapshotted under the lock, but the lock is NOT
        held while yielding: holding the RLock across consumer code would
        invert the sanctioned lock ordering whenever the consumer touches
        sinks.
        """
        with self._lock:
            sinks = list(self._sinks)
        for sink in sinks:
            iterator = getattr(sink, "iter_results", None)
            if callable(iterator):
                yield from iterator()

    def seed_prior_results(self, prior_results: VerificationResultSet) -> None:
        """Forward ``seed_prior_results`` to each child that implements it."""
        with self._lock:
            for sink in self._sinks:
                seeder = getattr(sink, "seed_prior_results", None)
                if callable(seeder):
                    seeder(prior_results)

    def on_start(self, manifest: list[str], config: VerificationConfig) -> None:
        """Forward ``on_start`` to each child."""
        with self._lock:
            for sink in self._sinks:
                sink.on_start(manifest, config)

    def on_result(self, result: VerificationResult) -> None:
        """Forward ``on_result`` to each child."""
        with self._lock:
            for sink in self._sinks:
                sink.on_result(result)

    def on_finalize(self, *, all_complete: bool) -> None:
        """Forward ``on_finalize`` to each child."""
        with self._lock:
            for sink in self._sinks:
                sink.on_finalize(all_complete=all_complete)


# ---------------------------------------------------------------------------
# DBSink
# ---------------------------------------------------------------------------


class DBSink:
    """Incremental database writer.

    Wraps :func:`karenina.storage.save_verification_results` so each
    completed task writes a row immediately instead of waiting for the
    end-of-batch :func:`auto_save_results` hook. Lets crashed runs retain
    durable rows for everything that finished.

    Resume support via the database is not implemented: :meth:`completed_triples`
    returns an empty set. Pair with :class:`ProgressiveFileSink` through
    :class:`CompositeSink` if resume is needed.
    """

    def __init__(
        self,
        storage_url: str,
        benchmark_name: str,
        run_name: str,
        run_id: str | None = None,
        templates: list[Any] | None = None,
    ) -> None:
        """Configure the sink.

        Args:
            storage_url: SQLAlchemy URL accepted by :class:`DBConfig`.
            benchmark_name: Benchmark name to register rows under.
            run_name: Human-readable run label.
            run_id: Unique run identifier; defaults to ``run_name``.
            templates: Finished templates; used to seed the benchmark row on
                ``on_start`` when the benchmark is new.
        """
        self.storage_url = storage_url
        self.benchmark_name = benchmark_name
        self.run_name = run_name
        self.run_id = run_id or run_name
        self._templates = list(templates or [])
        self._config_dict: dict[str, object] = {}
        self._initialized = False
        # Buffer of prior rows accepted before on_start. Flushed on on_start
        # once _initialized flips and the config snapshot is available.
        self._pending_seed: list[VerificationResult] = []
        # Internal sink lock guarding the seed buffer and init bookkeeping.
        # on_result deliberately stays outside it: it only reads the
        # _initialized flag and each DB write is independent. Innermost
        # lock: never call executor code while held.
        self._lock = threading.Lock()

    def completed_triples(self) -> set[TripleKey]:
        """Always empty; DB-based resume is out of scope for this sink."""
        return set()

    def seed_prior_results(self, prior_results: VerificationResultSet) -> None:
        """Buffer prior rows for durable DB write during :meth:`on_start`.

        DBSink requires a config snapshot (populated in ``on_start``) before
        it can persist rows, so seeding is deferred rather than written
        immediately. The buffer is flushed exactly once when ``on_start``
        runs, using the same :func:`save_verification_results` path as
        ``on_result``.
        """
        with self._lock:
            for row in prior_results.results:
                if is_completed_result(row):
                    self._pending_seed.append(row)

    def on_start(self, manifest: list[str], config: VerificationConfig) -> None:  # noqa: ARG002
        """Seed the benchmark row if missing and cache the config snapshot."""
        with self._lock:
            self._on_start_locked(config)

    def _on_start_locked(self, config: VerificationConfig) -> None:
        """Body of :meth:`on_start`. The caller holds ``self._lock``."""
        self._config_dict = config.model_dump(mode="json")
        try:
            from karenina.benchmark import Benchmark
            from karenina.storage import DBConfig, get_benchmark_summary, save_benchmark
        except Exception:  # noqa: BLE001
            logger.warning("DBSink disabled: karenina.storage import failed", exc_info=True)
            return

        db_config = DBConfig(storage_url=self.storage_url)
        try:
            existing = get_benchmark_summary(db_config, benchmark_name=self.benchmark_name)
            if not existing and self._templates:
                bench = Benchmark.create(
                    name=self.benchmark_name,
                    description=f"Auto-created for verification run: {self.run_name}",
                    version="1.0.0",
                )
                for template in self._templates:
                    bench.add_question(
                        question=template.question_text,
                        raw_answer="[Placeholder - see template]",
                        answer_template=template.template_code,
                        question_id=template.question_id,
                    )
                save_benchmark(bench, db_config)
            self._initialized = True
        except Exception:  # noqa: BLE001
            logger.warning("DBSink benchmark bootstrap failed; results will still be attempted", exc_info=True)
            self._initialized = True

        if self._pending_seed:
            flushed = 0
            for row in self._pending_seed:
                if self._persist(row):
                    flushed += 1
            self._pending_seed.clear()
            logger.info("DBSink flushed %d seeded prior rows", flushed)

    def on_result(self, result: VerificationResult) -> None:
        """Persist a single completed result."""
        if not is_completed_result(result) or not self._initialized:
            return
        self._persist(result)

    def _persist(self, result: VerificationResult) -> bool:
        """Write a single result to the DB. Returns True on success."""
        try:
            from karenina.storage import DBConfig, save_verification_results
        except Exception:  # noqa: BLE001
            logger.warning("DBSink _persist: import failed", exc_info=True)
            return False

        task_id = TaskIdentifier.from_result(result).to_key()
        try:
            save_verification_results(
                results={task_id: result},
                db_config=DBConfig(storage_url=self.storage_url),
                run_id=self.run_id,
                benchmark_name=self.benchmark_name,
                run_name=self.run_name,
                config=self._config_dict,
            )
            return True
        except Exception:  # noqa: BLE001
            logger.warning("DBSink failed to persist result %s; continuing", task_id, exc_info=True)
            return False

    def on_finalize(self, *, all_complete: bool) -> None:
        """Log a summary. Rows are already durable."""
        logger.info(
            "DBSink finalized (run=%s benchmark=%s all_complete=%s)",
            self.run_name,
            self.benchmark_name,
            all_complete,
        )


# ---------------------------------------------------------------------------
# InMemorySink (test helper)
# ---------------------------------------------------------------------------


class InMemorySink:
    """Buffers completed results in RAM. Intended for tests."""

    def __init__(self) -> None:
        self.results: list[VerificationResult] = []
        self.manifest: list[str] = []
        self.finalized_all_complete: bool | None = None
        self.started: bool = False
        # Internal sink lock guarding the results/manifest mutation.
        # Innermost lock: never call executor code while held.
        self._lock = threading.Lock()

    def completed_triples(self) -> set[TripleKey]:
        """Derive triples from buffered results."""
        with self._lock:
            snapshot = list(self.results)
        out: set[TripleKey] = set()
        for result in snapshot:
            ident = TaskIdentifier.from_result(result)
            out.add(
                (
                    ident.question_id,
                    ident.answering_canonical_key,
                    ident.parsing_canonical_key,
                    ident.replicate,
                )
            )
        return out

    def iter_results(self) -> Iterable[VerificationResult]:
        """Iterate buffered results without copying.

        Symmetry with :class:`ProgressiveFileSink` so the batch runner can
        hydrate the workspace cache from any sink type.
        """
        return iter(self.results)

    def seed_prior_results(self, prior_results: VerificationResultSet) -> None:
        """Append prior rows and their task keys, skipping duplicates."""
        with self._lock:
            existing_keys: set[str] = {TaskIdentifier.from_result(r).to_key() for r in self.results}
            manifest_set = set(self.manifest)
            for row in prior_results.results:
                if not is_completed_result(row):
                    continue
                task_id = TaskIdentifier.from_result(row).to_key()
                if task_id in existing_keys:
                    continue
                self.results.append(row)
                existing_keys.add(task_id)
                if task_id not in manifest_set:
                    self.manifest.append(task_id)
                    manifest_set.add(task_id)

    def on_start(self, manifest: list[str], config: VerificationConfig) -> None:  # noqa: ARG002
        """Record the manifest and mark started. Unions with any seeded keys."""
        with self._lock:
            if not self.manifest:
                self.manifest = list(manifest)
            else:
                existing = set(self.manifest)
                for key in manifest:
                    if key not in existing:
                        self.manifest.append(key)
                        existing.add(key)
            self.started = True

    def on_result(self, result: VerificationResult) -> None:
        """Append completed results, skipping previews."""
        if is_completed_result(result):
            with self._lock:
                self.results.append(result)

    def on_finalize(self, *, all_complete: bool) -> None:
        """Record the terminal state."""
        with self._lock:
            self.finalized_all_complete = all_complete


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_config_hash(config: VerificationConfig) -> str:
    """MD5 of the config JSON, excluding transient ``manual_traces``."""
    payload = config.model_dump_json(exclude={"manual_traces"})
    return hashlib.md5(payload.encode()).hexdigest()


def _fmt_utc(ts: float) -> str:
    """Format a POSIX timestamp as an ISO-8601 UTC string."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def _append_jsonl_line(path: Path, line: str) -> None:
    """Append one JSON line to ``path`` and fsync the file.

    The call is synchronous and not atomic across lines (a crash mid-write
    can leave a partial trailing line). :func:`_read_jsonl_results`
    tolerates that by skipping malformed trailing entries.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.write("\n")
        f.flush()
        import os

        os.fsync(f.fileno())


def _read_jsonl_results(path: Path) -> Iterator[VerificationResult]:
    """Yield :class:`VerificationResult` entries from a JSONL file.

    Malformed lines (e.g. a partial write after a crash) are logged and
    skipped. A trailing empty line is tolerated.
    """
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed JSONL line %d in %s: %s", lineno, path, e)
                continue
            try:
                yield VerificationResult.model_validate(data)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Skipping un-validatable JSONL entry at line %d in %s: %s",
                    lineno,
                    path,
                    e,
                )
