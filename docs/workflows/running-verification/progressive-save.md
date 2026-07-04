---
jupyter:
  jupytext:
    formats: docs/workflows/running-verification//md,docs/notebooks/running-verification//ipynb
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Progressive Save and Resume

This tutorial shows how to checkpoint verification progress so you can resume interrupted runs. For long verification runs (many questions, expensive models, multiple replicates), progressive save writes results incrementally. If the run stops for any reason (crash, Ctrl+C, a subset of tasks failing), you resume from the last checkpoint instead of re-evaluating everything.

Progressive save is available both from the CLI (`--progressive-save` / `--resume`) and from the Python API (via the `ResultSink` protocol). Both paths share the same on-disk layout, so a run started on the CLI can be resumed from Python and vice versa.

For the API reference of every sink (`ResultSink` Protocol, `ProgressiveFileSink`, `CompositeSink`, `DBSink`, `InMemorySink`) plus the helpers in `karenina.utils.progressive_save`, see the [Sinks reference](../../reference/api/sinks.md).

**What you'll learn:**

- Enable progressive save with `--progressive-save` (CLI)
- Resume an interrupted run with `--resume` (CLI) or `Benchmark.resume_verification()` (Python)
- Pass a `ProgressiveFileSink` to `Benchmark.run_verification(sink=...)` from Python
- Check run status with `karenina verify-status`
- Understand the `.results.jsonl` and `.state` file pair
- Compose sinks with `CompositeSink` (file + database)
- Resume is **triple-level**: completed (question, answering-model, parsing-model, replicate) tuples are skipped

```python tags=["hide-cell"]
# Setup cell: creates mock objects for progressive save demonstration.
# This cell is hidden in the rendered documentation.
import tempfile
from pathlib import Path

_tmp_dir = Path(tempfile.mkdtemp())
_output_path = _tmp_dir / "results.json"
```

---

## CLI Workflow

Three commands cover the full progressive save lifecycle:

```python
# 1. Start a verification run with progressive save enabled:
# karenina verify checkpoint.jsonld --preset default.json \
#   --output results.json --progressive-save

# 2. Check progress while a run is active (or after interruption):
# karenina verify-status results.json.state

# 3. Resume from where the run stopped:
# karenina verify --resume results.json.state

print("CLI commands shown in comments above")
```

The `--progressive-save` flag tells the runner to write each completed result to a `.results.jsonl` sidecar and track the task manifest in a `.state` file. If the process is interrupted, `--resume` picks up exactly where it stopped. The `verify-status` command reads the `.state` file and reports how many triples are pending.

---

## How It Works

Progressive save maintains two sidecar files alongside your output path:

```
verify --progressive-save
    │
    ├── results.json.results.jsonl   (append-only, one result per line)
    ├── results.json.state           (manifest + config + completion set)
    │
    ▼ (on full completion)
    results.json                     (final export)
```

The `.results.jsonl` file is append-only, so each completed result is an O(1) write. The `.state` file tracks the task manifest (every triple that needs to run), the set of completed triple keys, a config snapshot, and a config hash. Both files use atomic writes to prevent corruption if the process terminates mid-write.

These two files are referred to as the sink's "sidecars" throughout the codebase. The unrelated `.partial` files in `karenina.utils.file_ops.atomic_write` are also called sidecars, but those are short-lived atomic-write companion files; see the [Sinks reference](../../reference/api/sinks.md#31-progressivefilesink) for the disambiguation.

On successful completion, `on_finalize(all_complete=True)` assembles the final export from the JSONL, writes it to the output path, and removes the sidecars. If a run finishes with some tasks still failed, `on_finalize(all_complete=False)` retains the sidecars so `--resume` works next time.

**Resume is triple-level.** The unit of work is a `(question_id, answering_canonical_key, parsing_canonical_key, replicate)` tuple, not a question. If you have 10 questions × 2 answering models × 3 replicates, the state tracks 60 triples. Resuming after a crash that completed 35 triples runs only the remaining 25, even when those 25 cover questions that already had *some* triples completed.

**Scenarios are combo-atomic.** Scenario benchmarks use the same machinery with one adjustment: slot 0 of the triple holds `scenario_id` instead of `question_id`. Each scenario combo `(scenario_id, ans, parse, replicate)` is persisted as a single unit once all its turns complete; an interrupted combo re-runs from turn 1 on resume (no turn-level checkpointing). `Benchmark.resume_verification()` auto-detects QA vs scenario.

---

## Python API: Fresh Run with a Sink

Pass a `ProgressiveFileSink` to `Benchmark.run_verification(sink=...)`. The sink is threaded through the batch runner, which writes one JSONL line per completed result and drops completed triples from the task queue.

```python
# Real usage (not executed in this doc):
#
# from karenina.benchmark import Benchmark
# from karenina.benchmark.verification.sinks import ProgressiveFileSink
# from karenina.schemas.verification import VerificationConfig
#
# benchmark = Benchmark.load("checkpoint.jsonld")
# config = VerificationConfig(...)
#
# sink = ProgressiveFileSink(
#     output_path=Path("results.json"),
#     config=config,
#     benchmark_path="checkpoint.jsonld",
# )
#
# result_set = benchmark.run_verification(config=config, sink=sink)
# # On clean completion: results.json exists; sidecars are gone.
# # On partial failure: sidecars remain so resume_verification() works.
print("Fresh-run pattern shown in comments above")
```

The sink owns the full progressive-save lifecycle: it emits `on_start` (writing the manifest), `on_result` (appending JSONL lines), and `on_finalize` (clean export on success, retain sidecars on failure).

---

## Python API: Resume

Use `Benchmark.resume_verification(state_path)` to pick up where a prior run stopped. It reconstructs a `ProgressiveFileSink` from the `.state` file, populates `skip_triples` from the set of already-completed triples, and delegates to `run_verification`. The config stored in the state is used by default; pass `config=` to override (useful for bumping `request_timeout` or switching async workers on retry).

```python
# Real usage (not executed in this doc):
#
# benchmark = Benchmark.load("checkpoint.jsonld")
# result_set = benchmark.resume_verification("results.json.state")
#
# # To override config (e.g. raise request_timeout for flaky endpoints):
# tweaked = sink_config.model_copy(update={"request_timeout": 300.0})
# result_set = benchmark.resume_verification(
#     "results.json.state",
#     config=tweaked,
# )
print("Resume pattern shown in comments above")
```

Previously-completed results remain in the sink's buffer; the returned `VerificationResultSet` reflects whatever the executor produced during the resume pass. After a clean full completion, the `.state` and `.results.jsonl` sidecars are deleted and only the final export remains.

---

## Python API: Inspect State

Use `inspect_state_file()` to check progress without loading the full sink:

```python
# from karenina.utils.progressive_save import inspect_state_file
#
# status = inspect_state_file(Path("results.json.state"))
# print(f"{status.completed_count}/{status.total_tasks} done "
#       f"({status.progress_percent:.1f}%)")
print("Inspection pattern shown in comments above")
```

`inspect_state_file` handles both the v1 (`.tmp`) and v2 (`.results.jsonl`) layouts, so it works on runs started with older karenina versions too.

---

## Composing Sinks: File + Database

`CompositeSink` fans `on_start` / `on_result` / `on_finalize` across children and returns the union of their `completed_triples()`. Pair a `ProgressiveFileSink` with a `DBSink` to write to both the filesystem and a SQLite database incrementally:

```python
# from karenina.benchmark.verification.sinks import (
#     CompositeSink, DBSink, ProgressiveFileSink,
# )
#
# sink = CompositeSink([
#     ProgressiveFileSink(output_path=Path("results.json"),
#                         config=config, benchmark_path="checkpoint.jsonld"),
#     DBSink(storage_url="sqlite:///runs.sqlite",
#            benchmark_name="my-benchmark", run_name="my-run"),
# ])
# benchmark.run_verification(config=config, sink=sink)
print("CompositeSink pattern shown in comments above")
```

`DBSink` writes one row per completed result (replacing the end-of-batch `auto_save_results` pattern). Both children see the same stream of results, so the JSON export and the SQLite rows stay consistent.

When a sink is used by [`extend_template` / `extend_rubric`](../../core_concepts/extending-runs.md), prior rows are persisted progressively via `ResultSink.seed_prior_results(prior_results)`, which is forwarded by `CompositeSink` to every child that implements it. This is how an extension run's final export ends up containing both the prior rows and the new ones without holding everything in memory.

---

## Partial Failure: Keep State, Retry Only Failures

When the executor raises `VerificationBatchError` (some tasks failed, others succeeded), the batch runner catches it, flushes partial results through the sink, and calls `on_finalize(all_complete=False)`. The sidecars are kept. A subsequent `resume_verification()` will skip the completed triples and re-run only the failed ones.

Without a sink, `VerificationBatchError` still propagates for back-compat with existing code.

---

## Extending a Prior Run with Progressive Save

Both [`extend_template` and `extend_rubric`](../../core_concepts/extending-runs.md) accept the same `sink=` argument as `Benchmark.run_verification`, so the resumability story carries over verbatim to extension runs. This matters when an extension is itself long-running: a multi-judge `extend_template` on a 10k-question prior, or an `extend_rubric` over many traits, can take hours and is exactly the kind of workload progressive save was built for.

The mechanics:

1. The extension engine calls `sink.seed_prior_results(prior_results)` (when the sink implements it) before the pipeline starts, pre-populating the `.results.jsonl` and the `.state` manifest with the prior rows. This is how the merged final export ends up containing both prior and new rows even though only the new rows flow through `on_result`.
2. The new rows stream through `on_result` as the pipeline produces them, written incrementally to the JSONL.
3. On clean completion, `on_finalize(all_complete=True)` collapses the JSONL into a final JSON export at `output_path` and removes the sidecars.
4. On crash or partial failure, the sidecars are kept; `Benchmark.resume_verification(state_path)` resumes the extension, seeded prior rows included.

```python
# Extending a long prior run with a resumable sink:
#
# from pathlib import Path
# from karenina.benchmark.verification.sinks import ProgressiveFileSink
#
# sink = ProgressiveFileSink(
#     output_path=Path("merged.json"),
#     config=phase_b,
#     benchmark_path="checkpoint.jsonld",
# )
# merged = bench.extend_template(prior_results=prior, config=phase_b, sink=sink)
#
# # On crash, resume from the .state file:
# # bench.resume_verification("merged.json.state")
print("Extension-with-sink pattern shown in comments above")
```

The pattern composes with `CompositeSink`: pair a `ProgressiveFileSink` with a `DBSink` to get both an on-disk extension and a SQLite extension that survive a crash mid-run. All four in-tree sinks (`ProgressiveFileSink`, `DBSink`, `CompositeSink`, `InMemorySink`) implement `seed_prior_results`, so the prior rows make it into every persistence path. Third-party sinks that omit `seed_prior_results` are silently tolerated; the extension still completes, but the sink's final export contains only the new rows.

For the full extension surface, see the [Extending a Prior Run tutorial](./extending-a-prior-run.ipynb).

---

## Cleanup

```python tags=["hide-cell"]
import shutil

shutil.rmtree(_tmp_dir, ignore_errors=True)
```

---

## Next Steps

- [Basic Verification](../../notebooks/running-verification/basic-verification.ipynb): Single-model template-only evaluation
- [Full Evaluation](../../notebooks/running-verification/full-evaluation.ipynb): Template and rubric evaluation with quality checks
- [CLI Reference: verify](../../reference/cli/verify.md): All `karenina verify` options
- [CLI Reference: verify-status](../../reference/cli/verify-status.md): Inspect progressive save state
