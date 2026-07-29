# SPEC-003: Execution Model

**Status:** Draft for review
**Charter:** [README.md](README.md) §3, SPEC-003
**Principles:** P5 (one engine, many entries), P2, P4, P6
**Destination:** ADR-001 (decision), `src/karenina/engine/` (code), extension guide for custom tasks (SPEC-008)
**Decisions inherited:** DAG task model, streaming-shaped core, TaskEval full projection, driver public-but-experimental, async-native core gated on the SPEC-009 §5 spike.

---

## 1. The Task Abstraction

### 1.1 Identity

`TaskId` replaces the current result_key 4-tuple and the scenario slot-0 convention with one explicit, serializable identity:

```python
@dataclass(frozen=True, slots=True)
class TaskId:
    kind: Literal["qa", "scenario_turn", "task_eval", "custom"]
    subject: str          # question id; "scenario_id/node_id" for turns; step id for task_eval
    answering: str        # canonical model key (fixes the F03 raw-id/canonical-key split)
    parsing: str          # canonical model key
    replicate: int = 0

    def as_string(self) -> str:
        """Stable, order-fixed serialization used by sinks, resume, and logs."""
```

All persistence (sink records, skip sets, replay keys) uses `as_string()`. There is exactly one identity, used everywhere: the cache-vs-hydration key mismatch class of bugs (F03) becomes structurally impossible.

### 1.2 The Task

```python
class Task(Protocol):
    id: TaskId
    dependencies: frozenset[TaskId]

    async def run(
        self,
        runtime: TaskRuntime,
        upstream: Mapping[TaskId, TaskResult],
    ) -> TaskResult: ...
```

`TaskRuntime` is the driver-provided handle: the adapter gateway (§3), the config projection for this task (SPEC-004), and the replay capture hook (SPEC-006). Tasks never construct adapters or touch semaphores themselves.

A `Task` may return new tasks for the driver to schedule (`TaskResult.spawned: Sequence[Task]`). This is how dynamic graphs work: a scenario turn that resolves its outgoing edge spawns the next turn-task. The DAG is therefore not known upfront, which the streaming core (§2) supports natively.

### 1.3 Projections per workload

| Workload | Tasks | Dependencies |
|---|---|---|
| QA | one task per (question, answering, parsing, replicate) | none (flat) |
| Scenario | one task per traversed node turn, per combo and replicate | each turn depends on the previous turn's task; successors are spawned dynamically after edge resolution |
| TaskEval | one task per logged item and replicate, generation stage fed by `cached_answer_data` | none |
| Tournament (future) | comparison tasks | depend on the compared runs' tasks, fits without driver changes |

Each projection is one module (`engine/projections/qa.py`, `scenario.py`, `task_eval.py`) that turns the user-facing input (benchmark, scenario graph, logged outputs) into a task stream. Projections contain no scheduling, concurrency, sink, or retry logic. That is the P5 enforcement boundary, and it is what deletes the executor twins.

## 2. The Driver

```python
class Driver:
    def __init__(
        self,
        tuning: ExecutionTuning,
        sink: ResultSink,                 # always present; InMemorySink is the default
        gateway: AdapterGateway,
    ): ...

    async def run(self, tasks: AsyncIterable[Task]) -> AsyncIterator[TaskResult]: ...
```

Semantics:

- **Ready-set scheduling.** A task launches when all dependencies have results and the backpressure limit (`max_in_flight_tasks`) allows. Results are yielded as they complete, never batched at the end.
- **Streaming-shaped.** The input is an async iterable and the output an async iterator. Finite batches are the degenerate case (`async for` over a list). Online evaluation later plugs in an unbounded source, per the recorded design pressure.
- **Resume.** Before launching a task, the driver consults the sink's completed set (semantics owned by SPEC-006, including the F01 fix: a transient failure is not "completed"). Completed tasks short-circuit and emit their stored result so dependents can run.
- **Failure.** A task failure produces a `TaskResult` carrying the classified `Failure` (P4), is recorded by the sink, and unblocks nothing: dependents are cancelled with a recorded skip caveat. The driver itself raises only on infrastructure errors (sink unwritable, cancelled scope).
- **Cancellation.** One `asyncio.TaskGroup` owns all in-flight work. Ctrl-C or scope cancellation flushes the sink before propagating, so interruption never loses completed work (P6).

**Exposure:** public but experimental. The driver, `Task`, `TaskId`, and `TaskResult` are importable from `karenina.experimental.engine`, documented with an explicit no-stability-promise banner. Entry points import from the internal path. Promotion to a stable tier-2 surface is a later ADR.

**Entry-point projection.** All four entries become thin:

- `Benchmark.run_verification` (QA and scenario branches) builds the projection, the sink, and the tuning, then drives.
- TaskEval projects fully onto the driver (decision: full projection). It gains parallelism, sinks, resume, and caps. Its sequential behavior remains available as `ExecutionTuning(max_concurrent_llm_calls=1)`.
- karenina-server calls the same facade and supplies its progress broadcaster as a sink (composite with the file sink), removing the fourth sink-less path.

## 3. Concurrency

One model, two knobs, one enforcement point:

```python
class ExecutionTuning(BaseModel):
    max_concurrent_llm_calls: int = 8                 # global cap, process-wide per Driver
    per_model_limits: dict[str, int] = {}             # canonical model key -> cap
    max_in_flight_tasks: int | None = None            # backpressure for streaming sources
```

Enforcement lives in the **AdapterGateway**, the single async chokepoint through which every LLM and agent invocation passes:

```python
class AdapterGateway:
    async def invoke(self, model: ModelConfig, call: AdapterCall) -> AdapterResult:
        async with self._global_sem, self._model_sem(model.canonical_key):
            ...
```

Guarantee statement: any provider call made outside the gateway is a conformance-suite failure. Agent adapters declare `concurrency_granularity: "call" | "run"` in their capabilities: `"call"` adapters acquire a permit per internal LLM call (via the adapter's hook), `"run"` adapters hold one permit for the whole agent run. This closes the F06 agent bypass with an explicit, per-adapter, tested contract instead of a module-global semaphore nothing sets.

The current three uncoordinated mechanisms (ThreadPoolExecutor width, module-global `_global_llm_semaphore`, per-answerer `threading.Semaphore`) are all deleted.

## 4. Async Boundary

- **Async-native below the facade.** Driver, gateway, adapters, and stages run on one event loop. `Stage.execute` becomes `async def`. Pure-CPU stages stay trivially async (no awaits), LLM-touching stages await the gateway.
- **Sync facade.** Public sync methods (`run_verification` and friends) submit the coroutine to a dedicated background event-loop thread owned by a single bridge module (`engine/bridge.py`). Not `asyncio.run`: the background loop works inside Jupyter (where a loop is already running, the tier-1 environment), and it lets connection pools persist across calls. Async users get `arun_verification` directly.
- **Adapter requirements** are the SPEC-009 §5 spike checklist: native async invocation, single-loop resource lifecycle, cap enforceable at the leaf, MCP on the driver loop.
- **Fallback (tension T3).** The driver, task model, gateway, and projections are independent of the bridging choice. If the spike fails, only `bridge.py` changes (portal-based bridging, the current `async_lifecycle.py` work becomes its core) and the stages keep a sync execution path. The fallback is contained in one module by construction.

## 5. Lifecycle and Leak Detection

- The gateway owns adapter instances and their clients: created lazily on the driver loop, closed by `async with AdapterGateway(...)` in `Driver.run`'s exit path. MCP sessions are gateway-owned, per model, attached and torn down on the same loop.
- No module-global adapter registries, no thread-local portals, no teardown ordering rituals: everything lives and dies inside the driver's scope. `cleanup_resources()` and `_adapter_portal_refs` are deleted.
- Leak detection is a contract test (SPEC-007): after `Driver.run` exits, a debug registry of created clients must be empty. The conformance suite runs every adapter through it.

## 6. Survives vs Replaced (provisional, finalized by Phase 0 Track A)

| Current | Fate |
|---|---|
| `stages/` core, pipeline, helpers | Survives, `execute` becomes async, context slimmed per SPEC-004 |
| `failure_classifier.py`, `prompts/`, `evaluators/` | Survives |
| `sinks.py` | Protocol amended to v2 (SPEC-006), implementations survive |
| `replay/` | Survives, keyed by `TaskId.as_string()` (SPEC-006) |
| `executor.py`, `scenario_executor.py` | Deleted, replaced by `engine/driver.py` + projections |
| `batch_runner.py` | Deleted: queue generation moves to the QA projection, sink lifecycle into the driver |
| `benchmark.py` `_run_scenario_verification` sink logic | Deleted, scenario projection + sink v2 |
| `task_eval.py` `_run_evaluation_loop` | Deleted, TaskEval projection |
| `async_lifecycle.py` (in-flight branch work) | Becomes the fallback bridge core, or is deleted if the spike passes |
| `runner.py` | Slims to the per-task stage invocation inside `Task.run` |

New package layout: `src/karenina/engine/{driver.py, gateway.py, tuning.py, bridge.py, tasks.py, projections/{qa,scenario,task_eval}.py}`, each well under the 800-line cap.

## 7. Open Questions

1. `TaskRuntime` exact contents: blocked on SPEC-004 (config projection shape).
2. Sink v2 hook set consumed by the driver: owned by SPEC-006.
3. Stage async migration mechanics (signature change touches all 16 stages): sequencing detail for the implementation plan, not a design unknown.
4. Scenario edge resolution inside `Task.run` vs in the projection controller: decided during Phase 3 design review with Phase 0 Track A evidence on `manager.py`.
