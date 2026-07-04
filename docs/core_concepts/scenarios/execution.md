---
jupyter:
  jupytext:
    formats: docs/core_concepts/scenarios//md,docs/notebooks/core_concepts/scenarios//ipynb
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

# Execution

`ScenarioExecutor` is the runtime that drives one or more scenario combos through the verification pipeline. It is the scenario-side peer of the QA `VerificationExecutor`: instead of one task per (question, model) tuple, each task is an entire scenario graph traversal via `ScenarioManager.run`. This page documents the executor's parallel and sequential modes, the global LLM semaphore that throttles concurrent provider requests, the per-worker `BlockingPortal` reuse that keeps event loops alive across combos, the answer-trace cache shared between workers, scenario-level replicates and their cache key shape, and the trace materialization and SchemaOrg checkpoint helpers that surround a run.

For the per-edge context-routing layer that fires *inside* a scenario run, see [Handover](handover.md). For state and edge resolution semantics, see [State and Routing](state-and-routing.md). For builder-level configuration of the scenario itself, see [Building Scenarios](building-scenarios.md). For attaching outcomes that are evaluated after a run completes, see [Outcome Criteria](outcome-criteria.md).

```python tags=["hide-cell"]
# Mock setup for documentation: allows the notebook to run without API keys.
# Hidden in rendered docs. We expose minimal stand-ins for the executor surface
# rather than importing the real classes (which trigger the full karenina init
# chain). The shapes match the public API used in the snippets below.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ScenarioExecutorConfig:
    """Stand-in mirroring karenina.benchmark.verification.scenario_executor.ScenarioExecutorConfig."""

    max_workers: int = 2
    max_concurrent_requests: int | None = None
    enable_cache: bool = True
    timeout_seconds: float | None = None


class ScenarioExecutor:
    """Stand-in mirroring karenina.benchmark.verification.scenario_executor.ScenarioExecutor."""

    def __init__(self, parallel: bool = True, config: ScenarioExecutorConfig | None = None) -> None:
        self.parallel = parallel
        self.config = config or ScenarioExecutorConfig()


# Cache key helper from karenina.scenario.manager
import hashlib


def build_scenario_cache_key(
    scenario_id: str,
    node_id: str,
    answering_model_id: str,
    conversation_history_strs: list[str],
    replicate: int | None = None,
) -> str:
    history_str = "|".join(conversation_history_strs)
    history_hash = hashlib.sha256(history_str.encode()).hexdigest()[:16]
    base = f"{scenario_id}_{node_id}_{answering_model_id}_{history_hash}"
    return base if replicate is None else f"{base}_rep{replicate}"


print("Mock setup complete.")
```

## 1. What It Is

`ScenarioExecutor` accepts a list of *combos* (each a `(scenario_def, answering_model, parsing_model, replicate)` tuple) and runs them through `ScenarioManager.run`. It produces:

- a list of `ScenarioExecutionResult` objects in the original combo order (one per successful combo);
- a list of `(combo_description, exception)` tuples for combos that failed.

Two modes are available. Sequential mode runs combos one at a time on a single shared `BlockingPortal` (asyncio event loop). Parallel mode submits each combo to a `ThreadPoolExecutor` and gives every worker thread its own portal, which is reused across all combos that worker handles. Both modes share the same `AnswerTraceCache` and the same global LLM semaphore so concurrency control is uniform regardless of mode.

## 2. Core Idea

A scenario task is a long-running, stateful traversal of a graph. The executor stays narrow: spin up a portal, stamp the global LLM semaphore, hand each combo to `ScenarioManager.run`, collect results, and tear everything down in the right order. Every concurrency or cleanup detail it adds is there to keep `httpx` connection pools, agent adapters, and the BlockingPortal event loop from outliving each other. None of this is visible to scenario authors; it is what makes the simple `run_batch(combos, config)` call safe to use under parallel load.

## 3. Anatomy

### ScenarioExecutorConfig

```python
config = ScenarioExecutorConfig(
    max_workers=4,
    max_concurrent_requests=8,
    enable_cache=True,
    timeout_seconds=600.0,
)
print(config)
```

| Field | Type | Default | Effect |
|-------|------|---------|--------|
| `max_workers` | `int` | `DEFAULT_ASYNC_MAX_WORKERS` (2) | Size of the `ThreadPoolExecutor` in parallel mode |
| `max_concurrent_requests` | `int \| None` | `None` | Permits on the global LLM semaphore; `None` disables the cap |
| `enable_cache` | `bool` | `True` | Whether to allocate a shared `AnswerTraceCache` for the batch |
| `timeout_seconds` | `float \| None` | `None` | Wall-clock ceiling for the parallel batch; `None` disables the ceiling |

In sequential mode `max_workers` is ignored (one thread runs everything). `timeout_seconds` is enforced via `as_completed(..., timeout=...)` in parallel mode only; sequential mode runs to completion.

### ScenarioExecutor

```python
executor = ScenarioExecutor(parallel=True, config=ScenarioExecutorConfig(max_workers=4))
print(f"parallel={executor.parallel}, max_workers={executor.config.max_workers}")
```

The constructor takes `parallel: bool = True` and an optional `config`. The single public entry point is `run_batch(combos, config, global_rubric=None, run_name=None, progress_callback=None, workspace_root=None)`. Both modes return the same `(results, errors)` tuple.

### Global LLM semaphore

`set_global_llm_semaphore(sem)` and the read-side `get_global_llm_semaphore()` live in `karenina.benchmark.verification.executor` and are shared between the QA executor and `ScenarioExecutor`. The executor stamps a fresh `threading.Semaphore(max_concurrent_requests)` (or `None` to clear) before any worker starts and clears it in the outer `finally`. Adapters that call providers acquire the semaphore around the actual SDK call so the cap is honoured regardless of which executor invoked them.

```python
print("set_global_llm_semaphore(sem | None) -> None")
print("get_global_llm_semaphore() -> threading.Semaphore | None")
print()
print("Lives in karenina.benchmark.verification.executor.")
print("Imported by both ScenarioExecutor and the QA executor.")
```

### Per-worker BlockingPortal reuse

In parallel mode each worker thread lazily opens one `start_blocking_portal(backend='asyncio')` the first time it picks up a combo, and reuses that portal for every subsequent combo it handles. This keeps `httpx` connection pools, agent adapter event loops, and any other loop-affine resources alive across combos, instead of being torn down and rebuilt per scenario. After the pool drains, the executor closes adapter clients via `snapshot_adapters_for_portal(portal)` and `clear_portal_adapter_refs(portal)` *on the same loop that opened them* before exiting the portal context: this avoids the "Event loop is closed" failure mode that would otherwise hit `httpx.AsyncClient.aclose`.

### AnswerTraceCache sharing

When `enable_cache=True` the executor allocates a single `AnswerTraceCache` for the batch and threads it into every `ScenarioManager.run(...)` call. Worker threads share this cache. Cache keys are constructed by `build_scenario_cache_key` (see Replicates below). A cache hit short-circuits the answering model call for a node whose conversation history hash matches; a hit also skips the cache when a replay is known for the upcoming node so the IN_PROGRESS slot does not leak.

## 4. How It Works

### Sequential mode

1. Allocate `AnswerTraceCache` (if enabled) and `threading.Semaphore` (if a cap is set).
2. Open one shared `BlockingPortal`; call `set_async_portal(portal)`.
3. Stamp the semaphore via `set_global_llm_semaphore(sem)`.
4. For each combo, build a `ScenarioManager` and call `run(...)` with the cache, the portal-bound state, and the resolved `workspace_root` / `replicate`.
5. On exception, log and continue to the next combo.
6. In the `finally`, clear the per-portal adapter refs, drop the global portal, then clear the semaphore.

### Parallel mode

1. Same setup as sequential, plus a `ThreadPoolExecutor(max_workers=...)`.
2. Submit one `Future` per combo. Each worker lazily creates its own `BlockingPortal` on first use.
3. `as_completed(futures, timeout=timeout_seconds)` collects results as they finish. On `TimeoutError` the executor:
   - sweeps already-done futures it had not yet seen;
   - calls `pool.shutdown(wait=True, cancel_futures=True)`;
   - drains any futures that completed during shutdown;
   - emits a synthetic `TimeoutError` describing how many combos completed, were in flight, and were never started.
4. Pre-teardown adapter `aclose()` is run on each worker portal *before* the portal exits, with a bounded timeout (`PRE_TEARDOWN_ACLOSE_TIMEOUT`) so a stuck client cannot wedge the cleanup.
5. Every worker portal is closed; the global semaphore is cleared.

### Progress callbacks

`run_batch(progress_callback=...)` accepts a `(completed, total, result_or_none)` callback. In sequential mode it fires twice per combo: once with `result=None` before the run, once with the produced `ScenarioExecutionResult` after. In parallel mode it fires once per completion (with the result). Workers also report finer-grained per-turn progress through a separate per-combo callback the executor builds internally; that is consumed by the GUI's progress broadcaster, not the public callback.

## 5. Worked Example

```python
# Configure parallel scenario execution with four workers and an LLM
# concurrency cap of eight in-flight requests, with a 10-minute batch ceiling.
# The actual run requires real ScenarioDefinitions and ModelConfigs, so we
# only show construction here.

config = ScenarioExecutorConfig(
    max_workers=4,
    max_concurrent_requests=8,
    enable_cache=True,
    timeout_seconds=600.0,
)
executor = ScenarioExecutor(parallel=True, config=config)

print(f"parallel={executor.parallel}")
print(f"max_workers={executor.config.max_workers}")
print(f"max_concurrent_requests={executor.config.max_concurrent_requests}")
print(f"enable_cache={executor.config.enable_cache}")
print(f"timeout_seconds={executor.config.timeout_seconds}")
```

A real run would look roughly like:

```python
# Non-runnable: requires a fully constructed scenario, ModelConfigs, and a
# VerificationConfig. Shown for shape only.

# from karenina.benchmark.verification.scenario_executor import (
#     ScenarioExecutor, ScenarioExecutorConfig,
# )
#
# combos = [(scenario_def, ans_model, parse_model, None)]
# executor = ScenarioExecutor(parallel=True, config=ScenarioExecutorConfig(max_workers=4))
# results, errors = executor.run_batch(
#     combos,
#     verification_config,
#     global_rubric=rubric,
#     run_name="my-run",
#     workspace_root=Path("./scenario_workspace"),
# )
print("See karenina/src/karenina/benchmark/verification/scenario_executor.py:103-132")
```

## 6. Replicates

Scenario replicates are *distinct* from QA replicates. In QA, each (question, model) pair can be replicated `n_replicates` times; the runner builds independent tasks. In scenarios, an entire graph traversal is replicated: the fourth tuple element of a combo is the `replicate` index, and `ScenarioManager.run(replicate=...)` threads it through every turn.

```python
# Cache keys differ across replicates so independent traversals do not share
# answering-model results.

base = build_scenario_cache_key(
    scenario_id="sycophancy-resistance",
    node_id="initial",
    answering_model_id="anthropic_claude-sonnet-4-5",
    conversation_history_strs=["You are a clinical reasoner.", "What drug class does venetoclax belong to?"],
    replicate=None,
)
rep0 = build_scenario_cache_key(
    scenario_id="sycophancy-resistance",
    node_id="initial",
    answering_model_id="anthropic_claude-sonnet-4-5",
    conversation_history_strs=["You are a clinical reasoner.", "What drug class does venetoclax belong to?"],
    replicate=0,
)
rep1 = build_scenario_cache_key(
    scenario_id="sycophancy-resistance",
    node_id="initial",
    answering_model_id="anthropic_claude-sonnet-4-5",
    conversation_history_strs=["You are a clinical reasoner.", "What drug class does venetoclax belong to?"],
    replicate=1,
)

print(f"replicate=None : {base}")
print(f"replicate=0    : {rep0}")
print(f"replicate=1    : {rep1}")
```

The `replicate=None` form preserves the pre-replicate key shape byte-for-byte so legacy caches keep hitting; `replicate=N` appends `_repN`. The same `replicate` value is also propagated into the workspace layout (`<scenario_dir>/rep_<N>/turn_<M>/`) so trace files written by `transcript_materialize` do not collide between replicates. See `karenina/src/karenina/scenario/manager.py:159-173` for the workspace branch and `karenina/src/karenina/benchmark/verification/scenario_executor.py:300-317` for how the executor passes `replicate` into each `ScenarioManager.run` call.

## 7. Trace Materialization

The trace materialization helpers live in `karenina.scenario.trace_materialization`. They are reused by two callers: the `transcript_materialize` handover strategy ([Handover](handover.md)) and the agentic answer-parsing stage (Stage 7b) that needs file-readable conversation traces. Note this is distinct from the `AgenticRubricTrait.materialize_trace` flag, which controls whether agentic *rubric* evaluation gets the same XML-formatted trace.

| Function | What it returns | Used by |
|----------|----------------|---------|
| `materialize_trace(question_text, conversation_history, trace_dir, question_id, scenario_turn=None)` | `Path` to the written trace file | `transcript_materialize` handover, Stage 7b |
| `parse_transcript_entries(transcript)` | `list[dict]` (role, agent_id, content_type, content) | XML reformatter |
| `group_entries_into_turns(entries)` | `list[dict]` of structured turns | XML reformatter |
| `format_turns_as_xml(turns, artifacts_dir, truncation_threshold)` | XML string with `<turn>`/`<system_prompt>`/`<user>`/`<assistant>` | `materialize_trace` |
| `reformat_transcript_as_xml(question_text, artifacts_dir, truncation_threshold)` | XML string or unchanged input | `materialize_trace` |

The XML format groups labeled tagged-message lines into nested `<turn number="N">` elements. Each turn carries an optional `<system_prompt agent="...">`, a `<user>` block when the synthetic user prompt is present, and an `<assistant agent="...">` block whose body holds `<text>`, `<tool_call name="...">`, and `<tool_result name="...">` elements. Content blocks larger than `KARENINA_TRACE_TRUNCATION_THRESHOLD` (default 2000 chars) are offloaded to numbered files under `<trace_dir>/traces/artifacts/` and the parent element is marked `offloaded="true"` with the file path inline. The threshold can be overridden via the `KARENINA_TRACE_TRUNCATION_THRESHOLD` environment variable; an invalid value falls back to the default with a warning.

Public re-exports:

| Symbol | Module |
|--------|--------|
| `karenina.scenario.materialize_trace` | trace materialization entry point |
| `karenina.scenario.handover.format_transcript` | transcript renderer used as input to `materialize_trace` |
| `karenina.scenario.handover.TRANSCRIPT_SEPARATOR` | boundary string the reformatter detects |

## 8. SchemaOrg Checkpoint Persistence

Scenarios round-trip to disk through the SchemaOrg checkpoint format. The pair of conversion functions in `karenina.scenario.checkpoint` is the entry point.

| Function | Direction | When to call |
|----------|-----------|--------------|
| `scenario_to_schema_org(defn: ScenarioDefinition)` | `ScenarioDefinition` -> `SchemaOrgScenario` | Saving a benchmark to a JSON-LD checkpoint |
| `schema_org_to_scenario(schema: SchemaOrgScenario)` | `SchemaOrgScenario` -> `ScenarioDefinition` | Loading a checkpoint and reconstructing the executable scenario |

What survives the round trip:

- All declarative state: nodes, edges, entry node, outcome criteria with declarative `check` trees, `agent_identity`, `tool_filter`, `model_override`, `metadata`.
- `state_update_source`, `condition_source`, `evaluate_source`: the *source strings* of any callables. On load, `_compile_callable_string` recompiles them so the executable scenario carries live callables again.
- `verify_with` primitives in `StateCheck`, `TurnCheck`, `ResultCheck`, and `CrossTurnCheck`: re-injected via `_serialize_verify_with` and `_reconstruct_primitive`.
- Scenario-edge `handover` strings.

What does *not* survive:

- Callables passed in directly without a `*_source` companion. `state_update`, `condition_callable`, `evaluate`, and `handover_callable` are excluded from serialization (`Field(exclude=True)`). The builder emits a `UserWarning` when one of them is registered without a corresponding source string.
- Live API keys: `endpoint_api_key` and `anthropic_api_key` are stripped from `model_override` on serialization.

After loading a checkpoint, the resulting `ScenarioDefinition` is ready to hand to `ScenarioExecutor.run_batch`; no further compilation step is needed.

```python
# Non-runnable schema example: shows the round-trip surface only.

# from karenina.scenario import scenario_to_schema_org, schema_org_to_scenario
#
# schema = scenario_to_schema_org(defn)
# round_tripped = schema_org_to_scenario(schema)
# # round_tripped is functionally equivalent to defn for declarative content.
print("scenario_to_schema_org / schema_org_to_scenario are re-exported from karenina.scenario")
```

For the broader checkpoint format and benchmark-level persistence, see the SchemaOrg checkpoint reference in `karenina/docs/reference/data-formats/`.

## 9. Reference

### Public API

| Symbol | Purpose |
|--------|---------|
| `ScenarioExecutor(parallel=True, config=None)` | Construct an executor |
| `ScenarioExecutor.run_batch(combos, config, global_rubric=None, run_name=None, progress_callback=None, workspace_root=None)` | Run a batch of combos and return `(results, errors)` |
| `ScenarioExecutorConfig(max_workers, max_concurrent_requests, enable_cache, timeout_seconds)` | Per-batch configuration |
| `set_global_llm_semaphore(sem)` / `get_global_llm_semaphore()` | Cross-cutting LLM concurrency cap shared with the QA executor |
| `karenina.scenario.materialize_trace(...)` | Write a structured trace file (used by handover and Stage 7b) |
| `karenina.scenario.scenario_to_schema_org(defn)` / `schema_org_to_scenario(schema)` | Round-trip a `ScenarioDefinition` through the SchemaOrg checkpoint format |
| `ScenarioExecutionResult.to_replay_store(*, answering_model_id, **kwargs)` | Capture turn-level traces into a `ReplayStore` for downstream `extend_template` / `extend_rubric` runs (see [Extending Runs](../extending-runs.md)) |

### Sources

- `karenina/src/karenina/benchmark/verification/scenario_executor.py`: executor, config, parallel/sequential modes.
- `karenina/src/karenina/benchmark/verification/executor.py`: `set_global_llm_semaphore`, portal helpers, `PRE_TEARDOWN_ACLOSE_TIMEOUT`.
- `karenina/src/karenina/scenario/manager.py`: `ScenarioManager.run`, `build_scenario_cache_key`, workspace layout.
- `karenina/src/karenina/scenario/trace_materialization.py`: trace writer and XML reformatter.
- `karenina/src/karenina/scenario/checkpoint.py`: `scenario_to_schema_org`, `schema_org_to_scenario`.
- `karenina/src/karenina/schemas/scenario/state.py`: `ScenarioExecutionResult.to_replay_store`.

## 10. Next Steps

- [Handover](handover.md): per-edge context routing fired by `apply_handover` inside each turn.
- [Building Scenarios](building-scenarios.md): construct the `ScenarioDefinition` consumed by the executor.
- [State and Routing](state-and-routing.md): turn-level state updates and edge resolution.
- [Outcome Criteria](outcome-criteria.md): declarative assertions evaluated after a run completes.
- [Extending Runs](../extending-runs.md): use `ScenarioExecutionResult.to_replay_store` to feed `extend_template` / `extend_rubric`.
