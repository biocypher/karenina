# SPEC-006: Results Durability and Interchange

**Status:** Draft for review
**Charter:** [README.md](README.md) §3, SPEC-006
**Principles:** P6 (results are durable artifacts), P4, P7 exception clause
**Destination:** ADR (sink v2 + format versioning), `benchmark/verification/sinks.py` successor, `replay/`, checkpoint schema
**Decisions inherited:** Auto-retry transient failures on resume. Replay capture is opt-in (P6 wording amended accordingly in SPEC-001).

---

## 1. Resume Semantics (the F01 fix)

Definitions, keyed by `TaskId.as_string()` (SPEC-003 §1.1, which also fixes F03's key mismatch):

- **Completed**: a recorded result with no `Failure`, or with `failure_class` in `{guard_autofail, user_config}` (those are final verdicts about the task, re-running cannot change them honestly).
- **Retryable**: a recorded result with `failure_class == "transient"`. On resume these are re-run **by default**: resume means finish the run.
- **Stuck**: `failure_class` in `{permanent_infra, internal_bug}`. Skipped on resume by default (re-running cannot help until something is fixed), re-run with `retry_failed=True` (also re-runs transient, never re-runs completed).

The sink's recovery state is never deleted while any task is non-completed. On a fully clean completion the state file is finalized in place (marked complete), not removed: a resumed-then-extended run must still know what it contains.

The run summary (SPEC-005 §3) reports what resume did: n hydrated, n retried, n skipped-stuck, so the default is visible, never silent.

## 2. Sink Protocol v2

```python
class ResultSink(Protocol):
    async def on_start(self, run: RunMeta) -> None: ...          # no total-count parameter
    async def on_result(self, result: TaskResult) -> None: ...
    async def on_run_complete(self, summary: RunSummary) -> None: ...

class ResumableSink(ResultSink, Protocol):
    def completed(self) -> set[str]: ...                          # TaskId strings, per §1 definition
    def retryable(self) -> set[str]: ...
    def load(self, task_id: str) -> TaskResult | None: ...        # hydration for dependents and reports
```

- **Safety ownership:** the driver calls sink hooks from a single consumer, serialized. Sinks need no internal locking, and the protocol documents this (the current borrowed-`progress_lock` arrangement is deleted).
- **Streaming-friendly:** `on_start` carries run metadata but no finite task count. Progress totals are an optional hint (`RunMeta.expected_total: int | None`), absent for streaming sources, per the recorded design pressure.
- **Implementations:** `ProgressiveFileSink` (JSONL + state, the resumable default for CLI and facade runs), `InMemorySink` (default when no path given), `CompositeSink`, `DBSink`, and the server's progress broadcaster wrapped as a sink. The agentic sidecar variant folds into `ProgressiveFileSink` behind a flag rather than a parallel class, pending Track A evidence.

Every entry point always has a sink (P5, P6): the no-sink path ceases to exist.

## 3. Replay Store Contract

- **Opt-in** via `ReplayConfig(capture=True, store=...)`. Default off.
- **Keying:** `TaskId.as_string()` plus the interaction index within the task. Replicates are distinct keys by construction (fixes the cross-replicate contamination risk flagged in the original audit).
- **Capture guarantee when enabled:** every adapter interaction (LLM, agent, parser) flowing through the gateway is captured, including failed attempts, because the hook lives in the gateway (SPEC-003 §3), not in adapters.
- **Miss behavior:** `extend_template` / `extend_rubric` on a task without captured trace raises `ReplayMissError` with the SPEC-005 §4 anatomy (what is missing, why, the exact config to enable capture). It never silently regenerates.

## 4. The Result Artifact

`VerificationResult` is self-describing. Embedded, as schema fields:

- `task_id` (string form), `karenina_version`, `result_schema_version`, timestamps.
- The resolved `VerificationConfig` (model identities in canonical form) plus its hash, so any result can be traced to the exact configuration that produced it without external context.
- `Failure` and `Caveats` per SPEC-005.
- Whether replay capture was on (so downstream tooling knows if extension is possible without trying).

## 5. Interchange Format and Versioning

Two surfaces with different promises (resolving tension T1):

| Surface | Promise |
|---|---|
| Checkpoint / JSON-LD benchmark format and exported results | **Interchange**: carries `formatVersion` (semver). Stability commitment begins when the harmonized format ships. Breaking changes after that require a major bump and a documented migration |
| JSONL sink files, `.state` recovery files, replay store layout | **Internal**: no stability promise, readable only by the karenina version that wrote them (a `writer_version` field enables a helpful error, per SPEC-005 §4, instead of a parse failure) |

During this program, both surfaces may break freely (master design §8). The version fields are added now so the commitment is enforceable later.

## 6. Open Questions

1. `DBSink` resume support (currently absent): decided by Phase 0 Track B evidence on `storage/` usage, leaning ResumableSink for parity with files.
2. Replay store on-disk layout under the new keying: implementation-plan detail, the contract in §3 is the binding part.
3. Whether `RunSummary` is itself persisted in the checkpoint or only in sink output: leaning both, finalized with SPEC-002's results API.
