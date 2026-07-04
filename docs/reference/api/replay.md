# Replay Layer

`karenina.replay` is the user-facing surface for capturing, persisting, and replaying LLM responses. The replay path short-circuits the answering call (and, when `parsed_answer_fields` is captured, the parsing judge) so a fully captured run replays in milliseconds. This page is the canonical reference; for the conceptual walkthrough with worked examples see [Replay Store](../../advanced-pipeline/replay-store.md), and for how the replay store powers `extend_template` / `extend_rubric` see [Extending Runs](../../core_concepts/extending-runs.md).

## 1. Canonical Imports

```python
from karenina.replay import (
    # Value types
    ReplayKey,
    ReplayEntry,
    ReplayStore,
    ReplayMissPolicy,
    # Capture entry points
    capture_from_result_set,
    capture_from_scenario_result,
    # Persistence free functions
    dump,
    load,
    # Scenario projection
    ScenarioReplayBuilder,
    ProjectionReport,
    OrphanEntry,
    UnmatchedTarget,
    # Exceptions
    ProjectionError,
    ReplayMissError,
    ReplayHydrationError,
    ReplayPersistenceError,
    ReplayError,
)
from karenina.replay.ports_message_hydration import hydrate_trace_messages
```

## 2. Value Types

### 2.1 `ReplayKey`

Frozen Pydantic v2 model identifying *where* a captured turn lives in the evaluation graph. Two keys with the same field values are equal and hashable, which is what makes the lookup index O(1).

| Field | Type | Description |
|---|---|---|
| `question_id` | `str` | Question id (URN, derived from question text via `generate_question_id`). Required. |
| `scenario_id` | `str \| None` | When set, the key matches a turn inside a scenario; lookup is keyed by `(scenario_id, scenario_node)` and `question_id` is informational. |
| `scenario_node` | `str \| None` | Node id inside the scenario; used together with `scenario_id`. |
| `answering_model_id` | `str \| None` | Canonical model display string (`"interface:model_name"`). `None` is a wildcard matching any answering model. |
| `visit_index` | `int \| None` | Zero-based count of how many times the same node was visited inside one scenario run. `None` matches any visit. |
| `replicate` | `int \| None` | Replicate index. `None` is a wildcard matching any replicate. |

Lookups walk a four-step specificity ladder over `(answering_model_id, visit_index, replicate)`: most-specific first, wildcards last.

### 2.2 `ReplayEntry`

The captured payload itself. Pydantic v2 with `extra="forbid"`.

| Field | Type | Description |
|---|---|---|
| `raw_trace` | `str` | Captured response text. Always present. May be empty. |
| `parsed_answer_fields` | `dict \| None` | Pre-parsed Answer fields. When present, the parsing judge is bypassed entirely. |
| `trace_messages` | `list[dict] \| None` | Captured agent trace as `Message.to_dict()` outputs. Rehydrated via `hydrate_trace_messages` before being threaded into downstream stages. |
| `verify_result` | `bool \| None` | Captured template verdict. Enables verdict-only replay for rows whose raw trace cannot be reparsed. |
| `usage_metadata` | `dict \| None` | Captured usage summary, rehydrated into the replayed result so token accounting survives replay. |
| `agent_metrics` | `dict \| None` | Per-turn agent telemetry. The replay path only consumes `limit_reached`, which propagates to `recursion_limit_reached` and feeds the recursion-limit autofail stage. |
| `captured_model_id` | `str \| None` | Canonical display string of the model that produced the entry. Provenance only. |
| `captured_at` | `str \| None` | ISO timestamp of capture. Provenance only. |

### 2.3 `ReplayMissPolicy`

```python
ReplayMissPolicy = Literal["fall_through", "strict"]
```

`fall_through` (default): a miss returns `None` from `lookup()` and the live pipeline takes over. `strict`: a miss raises [`ReplayMissError`](#7-exception-hierarchy). Use `strict` for frozen captures in CI and `fall_through` for hybrid replay-plus-live runs.

## 3. `ReplayStore`

Keyed store of `(key, entry)` pairs with specificity-based lookup. The `entries` list is the source of truth; internal scenario / QA indexes are rebuilt from `entries` after every mutation.

### 3.1 Constructors

| Symbol | Signature | Purpose |
|---|---|---|
| `ReplayStore` | `(*, miss_policy: ReplayMissPolicy = "fall_through", entries: list[tuple[ReplayKey, ReplayEntry]] = [])` | Default constructor. |
| `ReplayStore.load` (classmethod) | `(path: str \| Path, *, miss_policy: ReplayMissPolicy \| None = None) -> ReplayStore` | Load from JSON; thin wrapper around `karenina.replay.persistence.load`. |
| `ReplayStore.from_manual_traces` (classmethod) | `(manual_traces: Any, benchmark: Any, *, miss_policy: ReplayMissPolicy = "strict") -> ReplayStore` | Build from a legacy `ManualTraces` instance. Emits one wildcard entry per registered trace (`answering_model_id=None`, `visit_index=None`, `replicate=None`). The classmethod `Benchmark.run_verification` uses internally when `interface="manual"` is in play. |

### 3.2 Mutation

| Method | Signature | Behavior |
|---|---|---|
| `register` | `(key: ReplayKey, entry: ReplayEntry) -> None` | Add or overwrite an entry. Overwrites emit a `WARNING` log. **Not safe to call concurrently with pipeline execution.** |

### 3.3 Lookup

| Method | Signature | Behavior |
|---|---|---|
| `lookup` | `(*, question_id: str, scenario_id: str \| None = None, scenario_node: str \| None = None, answering_model_id: str \| None = None, visit_index: int \| None = None, replicate: int \| None = None) -> ReplayEntry \| None` | Walk the specificity ladder and return the matching entry, or `None` (or raise `ReplayMissError` when `miss_policy="strict"`). |
| `has_any_for` | `(*, question_id: str, scenario_id: str \| None = None, scenario_node: str \| None = None) -> bool` | Cheap outer-key probe; does NOT walk the ladder. |

### 3.4 Persistence

| Method | Signature | Behavior |
|---|---|---|
| `save` | `(path: str \| Path) -> None` | Serialize to JSON; thin wrapper around `dump`. |

The JSON file is a versioned wrapper around a list of `(key, entry)` pairs emitted in store order (the order in which they were registered). Diff-friendliness comes from `json.dump(..., sort_keys=True, indent=2)`, which sorts the keys inside each JSON object and pretty-prints, not from sorting the entries list by key. Writes go through tempfile plus `os.replace` rename, so partial writes never reach the visible file.

## 4. Capture Entry Points

### 4.1 `capture_from_result_set`

```python
capture_from_result_set(
    result_set: VerificationResultSet,
    *,
    include_parsed: bool = True,
    include_agent_traces: bool = True,
    only_successful: bool = True,
    answering_model_ids: set[str] | None = None,
    scenario_ids: set[str] | None = None,
    replicate_selector: Literal["all", "first", "last"] = "all",
) -> ReplayStore
```

Walk a `VerificationResultSet` and emit a populated store. The default emits one entry per successful turn with parsed fields and structured trace messages included.

| Argument | Effect |
|---|---|
| `include_parsed` | When `True`, store `parsed_answer_fields` so the parsing judge is bypassed on replay. |
| `include_agent_traces` | When `True`, store the structured `trace_messages`. Set to `False` to drop tool-call detail and reduce file size. |
| `only_successful` | When `True`, drop turns whose pipeline did not produce a verifiable output (retry exhaustion, autofail, abstention, parsing or system errors). Content failures (clean pipeline, `verify()` returned False) are retained. |
| `answering_model_ids` | Allow-list by canonical model display string. |
| `scenario_ids` | Allow-list by scenario id. Applies only to scenario turns; QA turns pass through unaffected. |
| `replicate_selector` | `"all"` preserves replicate keys; `"first"` and `"last"` collapse to `replicate=None` wildcards (required by `ScenarioReplayBuilder`). |

`VerificationResultSet.to_replay_store(**kwargs)` is a forwarding wrapper around this function.

### 4.2 `capture_from_scenario_result`

```python
capture_from_scenario_result(
    scenario_result: ScenarioExecutionResult,
    *,
    answering_model_id: str,
    scenario_id: str | None = None,
    nodes: set[str] | None = None,
    include_parsed: bool = True,
    include_agent_traces: bool = True,
    replicate: int | None = None,
) -> ReplayStore
```

Walk a single `ScenarioExecutionResult`. `answering_model_id` is required because a `ScenarioExecutionResult` does not carry per-turn model identity. `scenario_id` defaults to `scenario_result.scenario_id` when not supplied. `nodes` is an optional allow-list of node ids.

## 5. Persistence Free Functions

```python
dump(store: ReplayStore, path: str | Path) -> None
load(path: str | Path, *, miss_policy: ReplayMissPolicy | None = None) -> ReplayStore
```

`ReplayStore.save` and `ReplayStore.load` are thin wrappers around these. Both routes are bit-for-bit equivalent. The methods are the canonical entry points; the free functions exist for symmetry and discoverability.

## 6. Scenario Projection

`ScenarioReplayBuilder` projects QA-mode `ReplayStore` entries onto scenario-mode keys, turning one `(question, model)` QA run into N scenario-mode replay entries that serve a shared "ask" turn across many dialogue framings.

### 6.1 `ScenarioReplayBuilder`

| Method | Signature | Behavior |
|---|---|---|
| `__init__` | `(benchmark: Benchmark, *, config: VerificationConfig, miss_policy: ReplayMissPolicy = "strict")` | Bind the builder to a benchmark and a default config (used for projections that omit `config=`). `miss_policy` is written onto the produced `ReplayStore`, so the default raises `ReplayMissError` on a miss rather than falling through. |
| `add_qa` | `(qa_store: ReplayStore, *, target_nodes: list[str], scenarios: list[str] \| None = None, config: VerificationConfig \| None = None) -> ScenarioReplayBuilder` | Stage a projection and return `self` for chaining. Requires `qa_store` captured with `replicate_selector="first"` or `"last"` (which emit `replicate=None`); `"all"` is rejected. |
| `validate` | `() -> ProjectionReport` | Walk all staged projections without mutation. |
| `build` | `(*, strict: bool = False) -> ReplayStore` | Run `validate`, then register matched entries into a fresh store. `strict=True` raises `ProjectionError` on any unmatched or duplicate target. `strict=False` emits a warning and returns whatever it matched. |

### 6.2 `ProjectionReport`

| Field | Type | Description |
|---|---|---|
| `projected_keys` | `list[ReplayKey]` | One per resolved (scenario, node) target. |
| `unmatched_targets` | `list[UnmatchedTarget]` | Targets that did not resolve to a QA entry. |
| `orphan_qa_entries` | `list[OrphanEntry]` | QA entries no staged projection consumed. |
| `duplicate_targets` | `list[tuple[str, str]]` | `(scenario_id, node_id)` pairs hit by more than one projection. `build(strict=False)` applies last-projection-wins. |
| `matched` (property) | `int` | `len(projected_keys)`. |

### 6.3 `UnmatchedTarget`

| Field | Type | Description |
|---|---|---|
| `scenario_id` | `str` | Scenario name that was scanned. |
| `node_id` | `str` | Node id requested as a projection target. |
| `question_id` | `str \| None` | The node's question id when resolution reached that step. |
| `answering_model_id` | `str \| None` | Effective runtime answering model id at the target. |
| `reason` | `Literal["missing_scenario", "missing_node", "no_qa_entry"]` | Why resolution failed. |

### 6.4 `OrphanEntry`

| Field | Type | Description |
|---|---|---|
| `question_id` | `str` | Question id of the orphaned QA entry. |
| `answering_model_id` | `str \| None` | Captured `ReplayKey.answering_model_id`. |
| `reason` | `Literal["no_target_scenario", "model_id_never_requested"]` | `no_target_scenario`: question_id not attached to any node in any of the projection's declared scenarios. `model_id_never_requested`: question matched at least one targeted scenario / node, but the effective runtime model at every such target differed from the entry's answering_model_id. |

## 7. Exception Hierarchy

All replay-layer exceptions inherit from `karenina.replay.ReplayError`, which inherits from `karenina.exceptions.KareninaError`.

| Exception | Raised by | Carries |
|---|---|---|
| `ReplayError` | Base class | nothing |
| `ReplayMissError` | `ReplayStore.lookup` under `miss_policy="strict"` | `key: ReplayKey` |
| `ReplayHydrationError` | parse-bypass when `parsed_answer_fields` does not validate against the current `Answer` class under `replay_parse_on_hydration_mismatch="strict"` | `captured_fields: dict`, `inner: Exception` |
| `ReplayPersistenceError` | `load` for missing file, malformed JSON, version mismatch, schema error, or invalid `miss_policy` string | the offending detail in the message |
| `ProjectionError` | `ScenarioReplayBuilder.build(strict=True)` on any unmatched or duplicate target | the `ProjectionReport` |

## 8. Helpers

### 8.1 `hydrate_trace_messages`

```python
hydrate_trace_messages(raw: list[dict[str, Any]]) -> list[Message]
```

Rehydrate captured `trace_messages` dicts into port [`Message`](../../advanced-adapters/ports.md) objects. Used internally by the replay short-circuit; documented here for completeness because it is the seam where captured agent traces become indistinguishable from live ones from the perspective of every stage after `GenerateAnswer`.

## 9. Cross-References

- [Replay Store](../../advanced-pipeline/replay-store.md): conceptual walkthrough with worked examples.
- [Extending Runs](../../core_concepts/extending-runs.md): how `extend_template` / `extend_rubric` build a `ReplayStore` from prior results.
- [Sinks reference](sinks.md): how `ProgressiveFileSink` interacts with replay-driven extension runs.
- [Scenarios: Execution](../../core_concepts/scenarios/execution.md): `ScenarioExecutionResult.to_replay_store` for multi-turn captures.
