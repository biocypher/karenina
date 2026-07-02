# SPEC-005: Failure and Observability

**Status:** Draft for review
**Charter:** [README.md](README.md) §3, SPEC-005
**Principles:** P4 (no silent degradation), P8
**Destination:** `schemas/results/failure.py` (taxonomy), `utils/errors.py` (registry), adapter conformance suite (SPEC-007), contributing docs (error anatomy)
**Decisions inherited:** Agent-actionable error anatomy is binding with a review checklist. Errors must be harmonized across adapters and modalities (user directive, 2026-06-10).

---

## 1. Failure Taxonomy

A **Failure** means the evaluation of a task could not be completed or trusted. A wrong answer is a result, never a Failure. The closed set:

```python
FailureClass = Literal[
    "transient",        # timeouts, rate limits, connection resets: retryable
    "permanent_infra",  # auth errors, invalid model id, provider 4xx: not retryable
    "guard_autofail",   # abstention, insufficiency, recursion limit, trace invalid, placeholder retry exhausted
    "user_config",      # bad template code, invalid config caught mid-run
    "internal_bug",     # unexpected exception inside karenina code, traceback attached
]
```

`retryable` is derived: `transient` only. The existing `Failure` and `Caveats` schemas remain the carrier, gaining `failure_class`. The mapping from current failure types to these classes is produced by Phase 0 Track A and recorded in the backlog.

**Cross-adapter and cross-modality harmonization.** Every adapter maps its SDK's exceptions into this taxonomy through the `ErrorRegistry`, and the same taxonomy applies on every modality (direct LLM calls, agent runs, MCP tool use, parsing calls). Concretely:

- Each adapter ships an error-mapping table (SDK exception type or status code → `FailureClass`) registered with `ErrorRegistry`, no ad hoc `except` clauses in adapter code.
- The conformance suite (SPEC-007) includes an error battery: each adapter is driven with simulated SDK failures (timeout, 429, 401, malformed response) and must produce identical classifications. Two adapters disagreeing on the same wire condition is a conformance failure.
- The retry policy (`RetryExecutor`) consumes only `FailureClass`, so retry behavior is uniform by construction.

## 2. Raise vs Record

| Error site | Behavior | Rationale |
|---|---|---|
| Config and setup errors (invalid template, unknown model, malformed config) | **Raise** before any LLM call | Fail fast, nothing to preserve yet |
| Adapter/provider error inside a task | **Record**: task gets a classified Failure, run continues | Resilience is the point of a batch harness, but nothing is silent: the failure is in the result |
| Stage raises unexpectedly | **Record** as `internal_bug` with traceback in the result, log at error | Replaces the orchestrator's current per-stage swallow. Never lost, never fatal to the batch |
| Dependent of a failed task | **Record**: skipped with a caveat pointing at the failed dependency | DAG semantics from SPEC-003 §2 |
| Sink write error | **Raise** (infrastructure) | Continuing while losing results would violate P6 silently |
| Teardown/cleanup error | **Log** at warning with `exc_info`, attach a run-level caveat, never raise | Cleanup must not destroy a completed run, but is never silent either |

The QA batch path's current swallow-and-discard of `exc.errors` is eliminated by this table: there is no row whose behavior is "discard".

## 3. The Visibility Guarantee

A user or agent must be able to discover every absorbed failure from the returned objects alone, without logs:

1. Every `VerificationResult` carries its `Failure` (if any) and `Caveats`.
2. Every run returns a **run summary**: counts by `FailureClass`, the list of failed and skipped `TaskId`s, and run-level caveats (including teardown warnings). The driver builds it, every entry point returns it, the sink persists it (`on_run_complete`, SPEC-006).
3. The tier-1 happy path prints nothing on success and a one-line warning with the failure count when the summary is non-clean. This is the joint rule resolving tension T4: quiet by default, but a degraded run is never indistinguishable from a clean one.

## 4. Error Anatomy (binding)

Every error raised at a user boundary states, in order: **what** failed, **why** (the observed condition, with values), and the **next action** (concrete: the flag, method, or fix). Example wording the rule requires:

> `ReplayMissError: extend_template could not find a captured trace for task qa/q42/anthropic:claude-haiku-4-5/0. This run was executed without replay capture. Re-run with VerificationConfig(replay=ReplayConfig(capture=True)) to enable extension.`

Enforcement: a review-checklist item (SPEC-008 development skill), plus tests asserting message content for the main boundary errors (config validation, resume mismatches, replay misses, adapter construction). Deep internals are exempt from the anatomy but never from classification.

## 5. Logging Policy

- Module-level `logger = logging.getLogger(__name__)`, lazy `%` formatting, no `print()` (existing convention, restated as binding).
- Levels: `info` = run lifecycle milestones, `warning` = absorbed failure or degraded behavior (every `warning` has a corresponding caveat per §3), `error` = task failure recorded, `debug` = internals.
- The no-swallow rule: any `except` that does not re-raise must log at warning or above with `exc_info` and produce a caveat. Bare `except: pass` is a review reject and a lint target.

## 6. Open Questions

1. Mapping table from current failure types to `FailureClass`: Phase 0 Track A output.
2. Whether `guard_autofail` subdivides further in the schema (per guard) or stays one class with the guard name as detail: decided when Track A lists the guards, leaning detail-field.
