# Result Sinks

`ResultSink` is the Protocol that streams [`VerificationResult`](../../core_concepts/results-and-scoring.md) instances out of the executor as a verification run progresses. Sinks are the single extension point for progressive save, crash recovery, incremental database writes, and any other per-result persistence the caller may need. Karenina ships four implementations covering the common cases, plus `AgenticProgressiveFileSink`, a `ProgressiveFileSink` subclass specialized for agentic workspace benchmarks (see section 3.5).

For the user-facing tutorial covering `--progressive-save`, `--resume`, `verify-status`, and the typical CLI flow, see [Progressive Save and Resume](../../notebooks/running-verification/progressive-save.ipynb). For how sinks interact with `extend_template` / `extend_rubric`, see [Extending Runs](../../core_concepts/extending-runs.md). For inspecting `.state` sidecars from the terminal, see the [`verify-status` CLI reference](../cli/verify-status.md).

## 1. Canonical Imports

```python
from karenina.benchmark.verification.sinks import (
    ResultSink,
    ProgressiveFileSink,
    CompositeSink,
    DBSink,
    InMemorySink,
    STATE_FORMAT_VERSION,
)
from karenina.utils.progressive_save import inspect_state_file, TaskIdentifier
```

`STATE_FORMAT_VERSION` lives next to the sinks because it describes the on-disk schema written by `ProgressiveFileSink`. Its current value is `"2.0"`. The legacy `ProgressiveSaveManager` (format `"1.0"`) is still readable by `inspect_state_file` for backward compatibility, but new runs always write `"2.0"`.

## 2. The `ResultSink` Protocol

`ResultSink` is a runtime-checkable [Protocol](https://peps.python.org/pep-0544/). Implementations are duck-typed: classes do not inherit from it, they just provide the methods. The batch runner ([`batch_runner.py`](https://github.com/codeforliberty/karenina/blob/main/src/karenina/benchmark/verification/batch_runner.py)) drives the lifecycle.

| Method | When the batch runner calls it | What the sink should do |
|--------|--------------------------------|--------------------------|
| `seed_prior_results(prior_results: VerificationResultSet) -> None` | Optional. Called by `extend_template` / `extend_rubric` before `on_start` to inject already-completed rows that live outside this batch. | Track the prior triples so `completed_triples()` includes them and the merged final export contains them. Implementations must be idempotent. |
| `completed_triples() -> set[tuple[str, str, str, int \| None]]` | Once at the top of `run_batch`, before task-queue generation. | Return every `(question_id, ans_canonical_key, parse_canonical_key, replicate)` triple already persisted. The runner unions this into `config.skip_triples`. |
| `on_start(manifest: list[str], config: VerificationConfig) -> None` | Once, after the task queue is generated, before any task runs. | Record the full task manifest and snapshot the config. |
| `on_result(result: VerificationResult) -> None` | Once per completed task, in completion order. The executor also emits preview results (no `metadata.timestamp`); sinks must filter those via `is_completed_result`. | Persist the result. |
| `on_finalize(*, all_complete: bool) -> None` | Exactly once when the batch returns, including the `VerificationBatchError` partial-failure path when a sink is attached. | Decide what to do with persistent state: write a final export and clean up sidecars on `True`, retain them for resume on `False`. |

`seed_prior_results` is the bridge from the [extension engine](../../core_concepts/extending-runs.md) to the sink layer: it lets `extend_template` / `extend_rubric` persist prior rows progressively rather than only carrying them in memory.

## 3. Implementations

### 3.1 `ProgressiveFileSink`

Append-only JSONL sidecar plus a `.state` manifest, written next to the final export. This is the implementation `--progressive-save` and `Benchmark.resume_verification` use.

```python
from pathlib import Path
from karenina.benchmark.verification.sinks import ProgressiveFileSink

sink = ProgressiveFileSink(
    output_path=Path("results.json"),       # final export target
    config=verification_config,             # snapshotted into .state
    benchmark_path="checkpoint.jsonld",     # stored for diagnostics
    global_rubric=None,                     # optional, folded into export
)
```

While a run is active, `ProgressiveFileSink` maintains two transient companion files alongside `output_path`:

- `<output>.results.jsonl`: append-only, one `VerificationResult.model_dump_json()` per line.
- `<output>.state`: JSON manifest with task list, completion / failure sets, config snapshot, config hash, format version.

These two files are referred to as the sink's "sidecars" throughout `sinks.py`. On `on_finalize(all_complete=True)`, both are deleted and the canonical export is written to `output_path`. On `on_finalize(all_complete=False)`, both are preserved so a later resume can continue.

Disambiguation: the unrelated [`atomic_write`](https://github.com/codeforliberty/karenina/blob/main/src/karenina/utils/file_ops.py) helper in `karenina.utils.file_ops` also uses the word "sidecar" to refer to its `.partial` temp file. Those are lower-level atomic-write companion files, written and deleted within a single call. They never coexist with progressive-save sidecars.

Notable instance methods beyond the protocol:

| Method | Signature | Purpose |
|--------|-----------|---------|
| `load_for_resume` (classmethod) | `(state_path: Path, *, global_rubric: Rubric \| None = None) -> ProgressiveFileSink` | Reconstruct a sink from an existing `.state` + JSONL pair. Raises `ValueError` if the state's `format_version` does not match `STATE_FORMAT_VERSION`. Hydrates `_completed`, `_manifest`, `_failed`, `_results` from disk. |
| `write_final_export` | `(output_path: Path \| None = None) -> Path` | Stream the JSONL into the canonical frontend-export JSON at `output_path` (or `self.output_path`). Called automatically by `on_finalize(all_complete=True)`. |
| `mark_failed` | `(task_id: str) -> None` | Record a task that was attempted but never produced a result. Optional helper used by the batch runner; not part of `ResultSink`. |
| `iter_results` | `() -> Iterable[VerificationResult]` | Iterate buffered results without copying. Used by the executor to hydrate its workspace cache on resume (see section 7). |
| `seed_prior_results` | `(prior_results: VerificationResultSet) -> None` | Pre-populate the JSONL, the buffered list, the completed set, and the manifest with rows from a prior run. Idempotent. Does not require `on_start` to have run first. |

### 3.2 `DBSink`

Incremental writer for the karenina storage layer. Wraps `save_verification_results` so each completed task writes a row immediately rather than waiting for the end-of-batch `auto_save_results` hook. Lets a crashed run keep durable rows for everything that finished.

```python
from karenina.benchmark.verification.sinks import DBSink

db_sink = DBSink(
    storage_url="sqlite:///runs.sqlite",   # SQLAlchemy URL (DBConfig)
    benchmark_name="my-benchmark",
    run_name="2026-05-02-baseline",
    run_id=None,                            # defaults to run_name
    templates=None,                         # finished templates; seeds the
                                            # benchmark row on on_start when
                                            # the benchmark does not exist yet
)
```

`DBSink.completed_triples()` always returns the empty set: DB-based resume is out of scope for this sink. To get resume on top of `DBSink`, pair it with `ProgressiveFileSink` inside a `CompositeSink` (the file sink supplies the resume state; both sinks see every result).

### 3.3 `CompositeSink`

Fan-out wrapper that forwards every lifecycle call to its children in order. `completed_triples()` returns the union across all children, which is the conservative choice for resume: if any sink reports a triple as complete, the runner skips it.

```python
from karenina.benchmark.verification.sinks import (
    CompositeSink, DBSink, ProgressiveFileSink,
)

sink = CompositeSink([
    ProgressiveFileSink(
        output_path=Path("results.json"),
        config=verification_config,
        benchmark_path="checkpoint.jsonld",
    ),
    DBSink(
        storage_url="sqlite:///runs.sqlite",
        benchmark_name="my-benchmark",
        run_name="2026-05-02-baseline",
    ),
])
```

`seed_prior_results` is forwarded to every child that implements it. `iter_results` drains children in iteration order.

### 3.4 `InMemorySink`

Buffers results in RAM. Useful as an in-process testing harness and as a canonical reference implementation showing the minimum work each method must do.

```python
from karenina.benchmark.verification.sinks import InMemorySink

sink = InMemorySink()
# After run_verification:
#   sink.results              -> list[VerificationResult]
#   sink.manifest             -> list[str]
#   sink.started              -> bool
#   sink.finalized_all_complete -> bool | None
```

`InMemorySink` implements `iter_results`, `seed_prior_results`, and the full Protocol; it never touches disk.

### 3.5 `AgenticProgressiveFileSink`

A `ProgressiveFileSink` subclass used by agentic workspace benchmarks. It keeps the standard JSONL plus `.state` resume machinery, and adds two behaviors on top:

- **Readable trace sidecars.** For every completed result it writes a human-readable trace file under `trace_output_dir`, laid out per question (`trace_layout="question"`) or per result (`trace_layout="result"`).
- **Parser-failure retry semantics.** A parser-stage failure that still carries a usable answer trace is treated as a resumable checkpoint rather than a terminal result. Those rows are exposed through `iter_results` for answer-cache hydration but excluded from `completed_triples()`, so a resume retries only the parsing step instead of regenerating the answer. Correspondingly, `on_finalize(all_complete=True)` writes the final export and deletes the sidecars only when no retryable parser failures remain. Otherwise it retains state for resume.

```python
from pathlib import Path
from karenina.benchmark.verification.sinks import AgenticProgressiveFileSink

sink = AgenticProgressiveFileSink(
    output_path=Path("results.json"),
    config=verification_config,
    benchmark_path="checkpoint.jsonld",
    global_rubric=None,
    trace_output_dir=Path("traces/"),      # where readable trace sidecars go
    trace_layout="question",               # "question" or "result"
    keep_progress_sidecars=False,          # keep JSONL/.state after a full export
    write_partial_export=True,             # write a partial export when resuming
)
```

`load_for_resume` accepts the same extra keyword arguments so a resumed agentic run keeps its trace-sidecar configuration.

## 4. JSONL Line Format

Each line in `<output>.results.jsonl` is the output of `VerificationResult.model_dump_json()`: a single self-contained JSON object with no surrounding array or wrapper. Append is atomic (`write` + `flush` + `os.fsync`) but a crash mid-write can leave a partial trailing line. `_read_jsonl_results` tolerates that: malformed final lines are logged and skipped, the rest of the file is still loadable.

A representative line (truncated) looks like:

```json
{"question_id": "q-123", "answering_model": "openai/gpt-4", ..., "metadata": {"timestamp": "2026-05-02T18:31:04Z", "replicate": null, "answering": {...}, "parsing": {...}}}
```

Reading is via the private helper `_read_jsonl_results(path)` which yields validated `VerificationResult` instances; consumers should normally use `ProgressiveFileSink.load_for_resume` rather than reading the JSONL directly.

## 5. State File Format

The `.state` sidecar is a single JSON document. The current schema is version `"2.0"` (`STATE_FORMAT_VERSION`). Top-level keys: `format_version`, `created_at`, `last_updated_at`, `benchmark_path`, `output_path`, `config_hash`, `config`, `task_manifest`, `completed_task_ids`, `failed_task_ids`, `total_tasks`, `completed_count`, `failed_count`, `start_time`. Every write goes through `atomic_write` (rename-after-temp-write), so partial writes never reach the visible file.

`ProgressiveFileSink.load_for_resume` raises `ValueError` if `format_version` does not match `STATE_FORMAT_VERSION` exactly. `inspect_state_file` is more lenient: it accepts both `"1.0"` (legacy `ProgressiveSaveManager` layout) and `"2.0"` so older runs remain inspectable from `karenina verify-status`.

## 6. `karenina.utils.progressive_save` Helpers

These helpers underlie the file-sidecar machinery and are useful when you want to inspect or build state files outside a sink.

### `inspect_state_file(state_path: Path) -> ProgressiveJobStatus`

Read-only view of a `.state` file. Returns a `ProgressiveJobStatus` dataclass with `state_file_path`, `output_path`, `benchmark_path`, `total_tasks`, `completed_count`, `pending_count`, `completed_task_ids`, `pending_task_ids`, `created_at`, `last_updated_at`, `start_time`, `answering_models`, `parsing_models`, `replicate_count`, `tmp_file_exists`, `tmp_file_size`, plus computed `progress_percent`, `elapsed_time`, `completed_question_ids`, `pending_question_ids`. Used by [`karenina verify-status`](../cli/verify-status.md).

### `TaskIdentifier`

Dataclass that maps between the four-tuple the executor uses and the tab-separated string keys persisted in the `.state` manifest:

| Method | Returns | Purpose |
|--------|---------|---------|
| `from_result(result: VerificationResult)` (classmethod) | `TaskIdentifier` | Build the identifier from a completed result. For scenario turn results (`metadata.scenario_id` non-`None`), slot 0 holds the scenario id so every turn in the same combo shares one task key. For QA results it holds `metadata.question_id`. |
| `from_task_dict(task: dict)` (classmethod) | `TaskIdentifier` | Build the identifier from a `generate_task_queue` dict. |
| `from_key(key: str)` (classmethod) | `TaskIdentifier` | Parse a tab-separated key string back into the dataclass; raises `ValueError` on malformed input. |
| `to_key()` (instance method) | `str` | Serialize to the canonical tab-separated form used in `.state` manifests. |

The four-tuple shape persisted by `completed_triples()` and consumed by `VerificationConfig.skip_triples` is:

```python
(question_id, answering_canonical_key, parsing_canonical_key, replicate)
```

with `replicate` either an `int` (one-based) or `None` for single-replicate runs.

## 7. Wiring Sinks Into `Benchmark`

Two facade methods on [`Benchmark`](../../notebooks/core_concepts/questions-and-benchmarks/benchmarks.ipynb) accept or construct sinks.

### `Benchmark.run_verification(config, *, sink=None, ...)`

Full signature:

```python
def run_verification(
    self,
    config: VerificationConfig,
    question_ids: list[str] | None = None,
    run_name: str | None = None,
    async_enabled: bool | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    sink: Any = None,
) -> VerificationResultSet
```

When `sink` is provided, the batch runner unions `sink.completed_triples()` into `config.skip_triples`, materializes any prior results via `sink.iter_results()` for cache hydration, calls `sink.on_start(manifest, config)`, dispatches `sink.on_result(result)` from inside the progress adapter, and calls `sink.on_finalize(all_complete=...)` exactly once before returning. `sink` is supported for standalone QA benchmarks and for scenario benchmarks (the scenario executor uses combo-atomic persistence: each `(scenario_id, ans, parse, replicate)` combo is persisted as a single completed unit once all its turn results are emitted, and an interrupted combo re-runs from turn 1 on resume).

### `Benchmark.resume_verification(state_path, *, config=None, ...)`

Full signature:

```python
def resume_verification(
    self,
    state_path: "str | Path",
    *,
    config: VerificationConfig | None = None,
    question_ids: list[str] | None = None,
    run_name: str | None = None,
    async_enabled: bool | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> VerificationResultSet
```

Reconstructs a `ProgressiveFileSink` via `ProgressiveFileSink.load_for_resume(state_path)`, attaches the benchmark's global rubric, and delegates to `run_verification(sink=...)`. The `prior_config` semantics are: if `config=None`, the config snapshotted into the `.state` file is used unchanged; if `config` is provided explicitly, the override completely replaces the stored one (useful for bumping `request_timeout`, switching `async_enabled`, or relaxing retry limits before a retry pass). The override is not merged field-by-field: any field absent from the override falls back to the override object's defaults, not to the stored config.

## 8. Cache Hydration on Resume

When a sink is attached, the batch runner calls `sink.iter_results()` once at the top of `run_batch` and passes the materialized list to the executor as `prior_results`. The executor's `_hydrate_cache_from_results` uses each prior row to populate the workspace `AnswerTraceCache` (see [Scenario Execution](../../core_concepts/scenarios/execution.md#answertracecache-sharing) for the broader cache picture), keyed by `(question_id, answering_canonical_key)`. The effect: on a resume that adds a new parsing variant for an already-completed answerer triple, the answer is reused from the cache instead of being regenerated at the original (possibly non-zero) temperature. Hydration is a no-op when the executor's cache is disabled, so it is safe to always materialize.

## 9. Partial-Failure Semantics

`on_finalize(all_complete=False)` is invoked when at least one of two conditions holds:

1. The executor raised `VerificationBatchError`. The runner catches it (only when a sink is attached; without a sink the exception still propagates, preserving back-compat) and uses `exc.partial_results` for the merged result set.
2. After the batch returns, `len(results) != len(task_queue)`. This catches silent shortfalls: tasks that completed but never produced a stored result, or queues that were truncated mid-flight.

Custom-sink authors should treat `all_complete` as "every task in the manifest produced a successful result"; a `False` value can mean either an explicit failure path or a quiet shortfall. Persistent storage should be retained on `False` so a later resume can continue from the durable manifest. See [Progressive Save and Resume](../../notebooks/running-verification/progressive-save.ipynb#partial-failure-keep-state-retry-only-failures) for the user-facing description of this behavior.
