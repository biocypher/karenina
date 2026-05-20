"""Result sinks for the verification pipeline.

A :class:`ResultSink` is a side-effect handler invoked by the batch runner
as each task completes. It is the single extension point for progressive
save, crash recovery, database auto-save, and any other per-result
persistence the caller may need.

Three implementations live here:

- :class:`ProgressiveFileSink`: appends each completed result to a JSONL
  sidecar and maintains a :file:`.state` manifest, so a crashed run can be
  resumed via :func:`VerificationConfig.skip_triples`. Replaces the legacy
  `ProgressiveSaveManager` full-rewrite strategy.
- :class:`CompositeSink`: fans out to several sinks in order; used when a
  single run wants both file progressive-save and database auto-save.
- :class:`InMemorySink`: test-friendly sink that just buffers results.

The protocol itself is duck-typed (see CLAUDE.md §8): adapters do not
inherit from :class:`ResultSink`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from karenina.benchmark.verification.stages.helpers.results_exporter import (
    export_verification_results_json_stream,
)
from karenina.schemas import VerificationConfig, VerificationResult
from karenina.schemas.entities import Rubric
from karenina.schemas.results import VerificationResultSet
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
                sink._completed.add(TaskIdentifier.from_result(result).to_key())

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
        added = 0
        manifest_set = set(self._manifest)
        for row in prior_results.results:
            if not is_completed_result(row):
                continue
            task_id = TaskIdentifier.from_result(row).to_key()
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
        out: set[TripleKey] = set()
        for key in self._completed:
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
        if not self._manifest:
            self._manifest = list(manifest)
        else:
            existing = set(self._manifest)
            for key in manifest:
                if key not in existing:
                    self._manifest.append(key)
                    existing.add(key)

        # Always refresh config + hash: a resume may pass a config that was
        # rehydrated from the state, but we keep whatever the caller handed
        # us so later callers get the current object identity.
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

        task_id = TaskIdentifier.from_result(result).to_key()
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
        if all_complete:
            self.write_final_export()
            self._delete_sidecars()
            logger.info("ProgressiveFileSink finalized: full export at %s", self.output_path)
        else:
            # Preserve sidecars so a later `--resume` can continue. Refresh
            # the state one last time so the last batch's activity lands.
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
        return len(self._completed)

    @property
    def total_tasks(self) -> int:
        """Total tasks in the manifest."""
        return len(self._manifest)

    def get_all_results(self) -> list[VerificationResult]:
        """Return a shallow copy of buffered completed results."""
        return list(self._results)

    def get_result_set(self) -> VerificationResultSet:
        """Return buffered results as a :class:`VerificationResultSet`."""
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
        target = output_path or self.output_path
        now = time.time()
        job = VerificationJob(
            job_id=f"progressive-{int(self._start_time or now)}",
            run_name="progressive",
            status="completed",
            config=self.config,
            total_questions=self.total_tasks,
            successful_count=self.completed_count,
            start_time=self._start_time,
            end_time=now,
        )
        # Stream from the JSONL sidecar if it exists. write_final_export may be
        # called before any on_result (e.g. the empty-run case), in which case
        # jsonl_path was never created; guard is mandatory because
        # _read_jsonl_results opens the path unconditionally.
        results_iter = _read_jsonl_results(self.jsonl_path) if self.jsonl_path.exists() else iter(())
        export_verification_results_json_stream(
            job,
            results_iter,
            self.global_rubric,
            is_complete=True,
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
# CompositeSink
# ---------------------------------------------------------------------------


class CompositeSink:
    """Fan-out wrapper that forwards every lifecycle call to its children.

    :meth:`completed_triples` returns the union across all children; this
    is conservative (if any sink reports a triple as complete, it is
    skipped) and matches the semantics callers expect: resuming with
    partial persistence should not redo work either store already has.
    """

    def __init__(self, sinks: Iterable[ResultSink]) -> None:
        """Initialize with an ordered iterable of child sinks.

        Args:
            sinks: Child sinks, called in iteration order on every event.
        """
        self._sinks: list[ResultSink] = list(sinks)

    @property
    def sinks(self) -> list[ResultSink]:
        """Child sinks in invocation order."""
        return list(self._sinks)

    def completed_triples(self) -> set[TripleKey]:
        """Union of completed triples across all children."""
        out: set[TripleKey] = set()
        for sink in self._sinks:
            out.update(sink.completed_triples())
        return out

    def seed_prior_results(self, prior_results: VerificationResultSet) -> None:
        """Forward ``seed_prior_results`` to each child that implements it."""
        for sink in self._sinks:
            seeder = getattr(sink, "seed_prior_results", None)
            if callable(seeder):
                seeder(prior_results)

    def on_start(self, manifest: list[str], config: VerificationConfig) -> None:
        """Forward ``on_start`` to each child."""
        for sink in self._sinks:
            sink.on_start(manifest, config)

    def on_result(self, result: VerificationResult) -> None:
        """Forward ``on_result`` to each child."""
        for sink in self._sinks:
            sink.on_result(result)

    def on_finalize(self, *, all_complete: bool) -> None:
        """Forward ``on_finalize`` to each child."""
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
        for row in prior_results.results:
            if is_completed_result(row):
                self._pending_seed.append(row)

    def on_start(self, manifest: list[str], config: VerificationConfig) -> None:  # noqa: ARG002
        """Seed the benchmark row if missing; cache the config snapshot."""
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

    def completed_triples(self) -> set[TripleKey]:
        """Derive triples from buffered results."""
        out: set[TripleKey] = set()
        for result in self.results:
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

    def seed_prior_results(self, prior_results: VerificationResultSet) -> None:
        """Append prior rows and their task keys, skipping duplicates."""
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
            self.results.append(result)

    def on_finalize(self, *, all_complete: bool) -> None:
        """Record the terminal state."""
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
