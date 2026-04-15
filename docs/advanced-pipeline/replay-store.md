---
jupyter:
  jupytext:
    formats: docs/advanced-pipeline//md,docs/notebooks/advanced/pipeline//ipynb
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

# Replay Store

The replay store lets you record the LLM responses produced by a verification run and re-run the same benchmark later without touching the model. The replay path short-circuits both the answering call and the parsing judge, so a fully captured run replays in milliseconds instead of seconds (or minutes for agent traces). Captures can be filtered, edited, persisted to disk, and mixed with live turns inside the same run, which makes the store useful for debugging, regression tests, deterministic CI, and hybrid scenario evaluation.

This page covers the user-facing API exposed by `karenina.replay`. For pipeline-internal details (where the short-circuits are wired into `GenerateAnswerStage` and `ParseTemplateStage`, and how the legacy `interface="manual"` shim auto-builds a store), see [Stages in Detail](stages.md).

```python tags=["hide-cell"]
# Notebook prerequisites
#
# This notebook never makes a real LLM call. The replay short-circuit fires
# before any adapter is constructed, so we can use any non-manual interface
# (we pick langchain/openai) with a placeholder API key. The placeholder is
# never read because the short-circuit returns first.
#
# We silence two unrelated warnings that fire during the in-notebook pipeline
# runs and would otherwise add noise to the rendered cells: the LangChain
# adapter cleanup timeout (a benign teardown race) and the openai-style API
# key complaint we suppress before any adapter is built.

import logging
import os

os.environ.setdefault("OPENAI_API_KEY", "placeholder-not-used-replay-only")

logging.getLogger("karenina.benchmark.verification.utils.resource_helpers").setLevel(logging.ERROR)
```

## Why replay

A verification run that hits a real model is expensive in three ways: every answering call burns provider tokens; every parsing call burns more; and the wall clock for a multi-question or multi-turn scenario can stretch to minutes. When you want to re-run the same benchmark to inspect a result, change a downstream rubric, or compare two answer templates against the same model output, paying that cost twice is wasteful.

The replay store solves this by recording the captured trace once and replaying it on subsequent runs. The pipeline behaves as if the model produced the captured response: the full result set, including parsed answer fields, verify result, and rubric evaluation, is reconstructed from the cached payload. The only stages that are skipped are the ones that would have called the model (`GenerateAnswerStage`) and the parsing judge (`ParseTemplateStage`), and only when the captured entry contains the necessary fields.

## The core types

`ReplayStore` is built around two value types: `ReplayKey` identifies *where* a captured turn lives in the evaluation graph, and `ReplayEntry` carries the captured payload itself. Both are Pydantic v2 models with `extra="forbid"`, so unknown fields raise at construction time.

```python
from karenina.replay import ReplayEntry, ReplayKey, ReplayStore

ReplayKey.model_fields.keys()
```

```python
ReplayEntry.model_fields.keys()
```

`ReplayKey` is frozen, which means two keys with the same field values are equal and hashable. This is what makes the lookup index O(1) and the deduplication inside `register()` deterministic.

```python
key_a = ReplayKey(question_id="urn:uuid:question-q1-12345678")
key_b = ReplayKey(question_id="urn:uuid:question-q1-12345678")
key_a == key_b, hash(key_a) == hash(key_b)
```

### Two identity modes in one store

A `ReplayKey` lives in one of two modes depending on whether `scenario_id` is set.

- **QA mode** (`scenario_id is None`): the key matches a single question. Lookup is keyed by `question_id` alone.
- **Scenario mode** (`scenario_id` is set): the key matches a turn at a specific node inside a scenario. Lookup is keyed by `(scenario_id, scenario_node)`. The `question_id` is still stored for diagnostics, but it is not part of the lookup key.

A single store can hold entries in both modes, and the right one is picked based on whether the executing pipeline turn is a QA turn or a scenario turn.

```python
qa_key = ReplayKey(question_id="urn:uuid:question-q1-12345678")
scenario_key = ReplayKey(
    question_id="urn:uuid:question-q1-12345678",
    scenario_id="syco-1",
    scenario_node="setup",
)
qa_key.scenario_id, scenario_key.scenario_id
```

### Optional refinements: model and visit

Every `ReplayKey` carries two optional refinements that make a registered entry more or less specific.

- `answering_model_id`: a string of the form `"interface:model_name"` (e.g. `"langchain:gpt-5"`) that ties the entry to one answering model. Setting it to `None` (the default) makes the entry a wildcard that matches any answering model.
- `visit_index`: an integer counting how many times the same node has been visited inside one scenario run. Setting it to `None` makes the entry match any visit.

When the pipeline asks the store for an entry, it walks a four-step specificity ladder: `(model, visit) → (model, None) → (None, visit) → (None, None)`. The first hit wins. So the most-specific entry overrides any wildcards, and a wildcard entry serves as a fallback for everything else.

```python
store = ReplayStore()
store.register(
    ReplayKey(question_id="q", answering_model_id="langchain:gpt-5"),
    ReplayEntry(raw_trace="gpt-5 specific answer"),
)
store.register(
    ReplayKey(question_id="q"),
    ReplayEntry(raw_trace="wildcard answer"),
)
hit_specific = store.lookup(question_id="q", answering_model_id="langchain:gpt-5")
hit_wildcard = store.lookup(question_id="q", answering_model_id="langchain:claude")
hit_specific.raw_trace, hit_wildcard.raw_trace
```

The wildcard ladder is what lets you capture one model's run, then replay the same store against a different model with `miss_policy="fall_through"`: the new model gets the wildcard entry instead of failing.

## ReplayEntry payload

A `ReplayEntry` holds everything the pipeline needs to reconstruct one turn. The fields fall into three groups.

The first group is the raw trace itself, which is always present.

- `raw_trace`: a string. The captured response text. May be empty when the model returned nothing (abstention, sufficiency skip, or a streaming-timeout truncation are legitimate empty traces).

The second group is the optional structured payload. Setting these makes the replay even cheaper because more downstream stages can be skipped.

- `parsed_answer_fields`: a `dict` suitable for `Answer.model_validate(...)`. When present, the parsing judge is bypassed entirely and the parsed answer is hydrated directly from the dict.
- `trace_messages`: a `list[dict]` of `Message.to_dict()` outputs, used to reconstruct an agent trace with tool calls.
- `agent_metrics`: a `dict` carrying the per-turn agent telemetry. The only key the replay path actually consumes is `limit_reached` (used to populate `recursion_limit_reached` on the result).

The third group is provenance metadata, useful for diagnostics but not consumed by the pipeline.

- `captured_model_id`: the canonical display string of the model that produced this entry.
- `captured_at`: an ISO timestamp of when the capture happened.

```python
entry = ReplayEntry(
    raw_trace="The capital of France is Paris.",
    parsed_answer_fields={"city": "Paris"},
    captured_model_id="langchain:gpt-5",
)
entry.model_dump(exclude_none=True)
```

## Building a store by hand

`ReplayStore.register(key, entry)` is the lowest-level mutation: it appends to the `entries` list (the source of truth) and rebuilds the internal indexes. Registering the same key twice is allowed; the second call overwrites the first and emits a `WARNING` log so you can spot accidental duplication.

```python
store = ReplayStore(miss_policy="fall_through")

store.register(
    ReplayKey(question_id="urn:uuid:question-paris-abcdef12"),
    ReplayEntry(
        raw_trace="The capital of France is Paris.",
        parsed_answer_fields={"city": "Paris"},
    ),
)
store.register(
    ReplayKey(question_id="urn:uuid:question-rome-87654321"),
    ReplayEntry(
        raw_trace="The capital of Italy is Rome.",
        parsed_answer_fields={"city": "Rome"},
    ),
)

len(store.entries), [k.question_id for (k, _) in store.entries]
```

### Lookup vs has_any_for

There are two retrieval methods. `lookup(...)` walks the specificity ladder and returns the matching entry (or `None` in fall-through mode, or raises `ReplayMissError` in strict mode). `has_any_for(...)` is a cheap outer-key probe that returns `True` if *any* entry exists for that question or scenario node, without walking the ladder. The scenario executor uses `has_any_for` to decide whether to reserve an `AnswerTraceCache` slot before the turn runs.

```python
hit = store.lookup(question_id="urn:uuid:question-paris-abcdef12")
miss = store.lookup(question_id="urn:uuid:question-not-here")
hit.raw_trace, miss
```

```python
store.has_any_for(question_id="urn:uuid:question-paris-abcdef12"), store.has_any_for(question_id="urn:uuid:question-not-here")
```

### Miss policies: strict vs fall_through

Each store has a `miss_policy` that controls what happens when `lookup()` cannot find a match. The default is `"fall_through"`, meaning misses return `None` and the live pipeline takes over for that turn. The alternative is `"strict"`, meaning misses raise `ReplayMissError` and the pipeline marks the turn as failed.

```python
from karenina.replay import ReplayMissError

strict_store = ReplayStore(miss_policy="strict")
strict_store.register(
    ReplayKey(question_id="urn:uuid:question-known-12345678"),
    ReplayEntry(raw_trace="canned"),
)

try:
    strict_store.lookup(question_id="urn:uuid:question-unknown-87654321")
except ReplayMissError as exc:
    print("strict miss:", exc)
```

Use `"strict"` when you have a frozen capture and want any drift (a new question, a renamed scenario node, an extra replicate) to surface as a clear error. Use `"fall_through"` when you want to capture one model's run and replay it against a slightly different benchmark, allowing live calls to fill the gaps.

## Running a verification with replay

The smallest end-to-end story: build a benchmark, build a store with one entry, attach the store to `VerificationConfig.replay_store`, and call `Benchmark.run_verification`. The `GenerateAnswerStage` short-circuit reads the store before the answering adapter is even constructed, so the LLM call never happens. When `parsed_answer_fields` is captured, the `ParseTemplateStage` bypass also skips the parsing judge.

```python
from karenina.benchmark import Benchmark
from karenina.schemas.config import ModelConfig
from karenina.schemas.verification.config import VerificationConfig

TEMPLATE_PARIS = """
from karenina.schemas.entities.answer import BaseAnswer
from karenina.schemas.entities.verified_field import VerifiedField
from karenina.schemas.primitives.comparisons import RegexMatch


class Answer(BaseAnswer):
    city: str = VerifiedField(
        description="Capital city",
        verify_with=RegexMatch(pattern="(?i)paris"),
        ground_truth="Paris",
        default="",
    )

    def verify(self) -> bool:
        return self.city.strip().lower() == "paris"
"""

bm = Benchmark.create(name="replay-demo", version="1.0.0")
qid = bm.add_question(
    question="What is the capital of France?",
    raw_answer="Paris",
    answer_template=TEMPLATE_PARIS,
)
qid
```

```python
store = ReplayStore(miss_policy="strict")
store.register(
    ReplayKey(question_id=qid),
    ReplayEntry(
        raw_trace="The capital of France is Paris.",
        parsed_answer_fields={"city": "Paris"},
    ),
)

config = VerificationConfig(
    answering_models=[
        ModelConfig(id="ans", model_name="gpt-fake", interface="langchain", model_provider="openai"),
    ],
    parsing_models=[
        ModelConfig(id="par", model_name="gpt-fake", interface="langchain", model_provider="openai"),
    ],
    replay_store=store,
)

result_set = bm.run_verification(config)
vr = result_set.results[0]
vr.template.raw_llm_response, vr.template.parsed_llm_response, vr.template.verify_result
```

The `gpt-fake` model name and the placeholder API key are never actually used: the replay short-circuit returns from `GenerateAnswerStage` before any adapter is built. This is intentional. You can use any non-manual interface in the `ModelConfig` and the run will succeed as long as every executed turn has a registered entry (in strict mode) or you accept fall-through to live calls for the misses.

### Capturing live runs

The dual to building a store by hand is capturing one from a previous run. `VerificationResultSet.to_replay_store(**kwargs)` is a convenience method that walks the result set and builds a store with one entry per successful turn. Under the hood it calls `karenina.replay.capture_from_result_set`, which exposes the same kwargs plus a few filters. Both routes return a `ReplayStore` with `miss_policy="fall_through"`.

```python
from karenina.replay import capture_from_result_set

# In production this comes from a real bm.run_verification(...) call
# against a real model. Here we use the result set we just built.
captured = result_set.to_replay_store(include_parsed=True)
hand = capture_from_result_set(result_set, include_parsed=True, only_successful=True)
len(captured.entries), len(hand.entries)
```

The `capture_from_result_set` function takes several filters: `include_parsed` controls whether the parsed Pydantic instance is captured, `include_agent_traces` controls whether the structured trace messages are captured, `only_successful` drops failed turns, and `answering_model_ids` / `scenario_ids` are allow-lists by canonical model display string and scenario id respectively. The scenario filter only applies to scenario turns, so QA turns pass through unaffected when the filter is set.

```python
from karenina.replay.capture import capture_from_scenario_result  # noqa: F401  # imported to show the alternative entry point for single scenarios
```

For a single scenario you can use `capture_from_scenario_result(scenario_result, answering_model_id=..., ...)`, which is preferable when you only have one `ScenarioExecutionResult` rather than a full `VerificationResultSet`. The `answering_model_id` argument is required because a `ScenarioExecutionResult` does not carry per-turn model identity.

## Persistence

Stores serialize to JSON via `save()` / `load()` (or the equivalent free functions `dump()` / `load()` in `karenina.replay.persistence`). The on-disk format is a versioned wrapper around a list of `(key, entry)` pairs and is sorted by key for diff-friendliness. Writes go through a tempfile + `os.replace` rename so a crash mid-write never leaves a half-written file in place.

```python
import json
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "replay.json"
    store.save(path)
    on_disk = json.loads(path.read_text())
    print("version:", on_disk["version"])
    print("miss_policy:", on_disk["miss_policy"])
    print("first key:", on_disk["entries"][0]["key"])
    print("first entry payload:", on_disk["entries"][0]["entry"])
    reloaded = ReplayStore.load(path)
    print("reloaded entries:", len(reloaded.entries))
```

The `version` field is what catches stale checkpoints when the schema evolves. Loading a file with a version this build does not understand raises `ReplayPersistenceError` with the offending version in the message; same goes for malformed JSON, missing top-level keys, or schema validation failures on individual entries. There is no silent fall-through: if the file is unreadable, the load call always raises.

The `miss_policy` recorded inside the file is informational; you can override it on load by passing `miss_policy=...`, which is the recommended way to swap in a strict policy for CI runs without rewriting the file.

```python
from karenina.replay import dump, load as load_store

with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "strict_replay.json"
    dump(store, path)
    strict = load_store(path, miss_policy="strict")
    print("loaded miss_policy:", strict.miss_policy)
```

## Hybrid scenarios: canned setup, live followup

Scenarios are where replay earns the most. A single scenario run touches one node per turn, and you frequently want to *replay one node and live-run the next* to test how a model reacts to a fixed setup. Because the scenario lookup key is `(scenario_id, scenario_node)`, the store can hold an entry for the setup node only; when the followup node runs, the store returns no match and the live pipeline takes over.

```python
from karenina.schemas.scenario.definition import ScenarioDefinition
from karenina.schemas.scenario.types import END, ScenarioEdge, ScenarioNode
from karenina.schemas.entities.question import Question

TEMPLATE_FREE = """
from karenina.schemas.entities.answer import BaseAnswer
from karenina.schemas.entities.verified_field import VerifiedField
from karenina.schemas.primitives.comparisons import RegexMatch


class Answer(BaseAnswer):
    answer: str = VerifiedField(
        description="Free form answer",
        verify_with=RegexMatch(pattern=".+"),
        ground_truth="non-empty",
        default="",
    )

    def verify(self) -> bool:
        return len(self.answer.strip()) > 0
"""

setup_q = Question(
    question="Name one gene linked to cystic fibrosis.",
    raw_answer="CFTR",
    answer_template=TEMPLATE_FREE,
)
followup_q = Question(
    question="Which chromosome is that gene on?",
    raw_answer="7",
    answer_template=TEMPLATE_FREE,
)
scenario = ScenarioDefinition(
    name="cf-2-turn",
    entry_node="setup",
    nodes={
        "setup": ScenarioNode(node_id="setup", question=setup_q),
        "followup": ScenarioNode(node_id="followup", question=followup_q),
    },
    edges=[
        ScenarioEdge(source="setup", target="followup"),
        ScenarioEdge(source="followup", target=END),
    ],
    outcome_criteria=[],
)

scenario_bm = Benchmark.create(name="hybrid-replay", version="1.0.0")
scenario_bm.add_scenario(scenario)
[s.name for s in scenario_bm.get_scenarios()]
```

Now register a replay entry for the setup node only. The `scenario_id` matches the scenario name, the `scenario_node` matches the node id, and the `question_id` is informational (the scenario lookup is keyed on the pair, not on the question id).

```python
from karenina.utils.checkpoint import generate_question_id

setup_qid = generate_question_id(setup_q.question)

hybrid_store = ReplayStore(miss_policy="fall_through")
hybrid_store.register(
    ReplayKey(
        question_id=setup_qid,
        scenario_id="cf-2-turn",
        scenario_node="setup",
    ),
    ReplayEntry(
        raw_trace="CFTR is a well-known gene linked to cystic fibrosis.",
        parsed_answer_fields={"answer": "CFTR"},
    ),
)

hybrid_store.has_any_for(scenario_id="cf-2-turn", scenario_node="setup", question_id=setup_qid), hybrid_store.has_any_for(
    scenario_id="cf-2-turn", scenario_node="followup", question_id=setup_qid
)
```

When the scenario runs with this store attached, the `setup` turn would short-circuit on the registered entry and the `followup` turn would fall through to a live model call. We do not actually execute the run here (it would require a real model for the followup), but the same `VerificationConfig.replay_store` machinery applies: pass `replay_store=hybrid_store` and the scenario manager handles the rest. The integration test `tests/replay/test_replay_scenario_live_e2e.py` exercises exactly this flow against a live vLLM endpoint, gated on `KARENINA_LIVE_TESTS=1`.

### Per-visit replay for revisited nodes

When a scenario revisits the same node (a routing loop, a corrective turn), the visit counter starts at 0 and increments after each turn. You can register entries with `visit_index` set to a specific integer to give different responses on different visits, or leave it as `None` to give the same response every time.

```python
revisit_store = ReplayStore()
revisit_store.register(
    ReplayKey(
        question_id="qid-ignored-for-scenario-lookup",
        scenario_id="loop-demo",
        scenario_node="probe",
        visit_index=0,
    ),
    ReplayEntry(raw_trace="first visit answer"),
)
revisit_store.register(
    ReplayKey(
        question_id="qid-ignored-for-scenario-lookup",
        scenario_id="loop-demo",
        scenario_node="probe",
        visit_index=1,
    ),
    ReplayEntry(raw_trace="second visit answer"),
)

[
    revisit_store.lookup(
        question_id="qid-ignored-for-scenario-lookup",
        scenario_id="loop-demo",
        scenario_node="probe",
        visit_index=v,
    ).raw_trace
    for v in (0, 1)
]
```

If you only register a single entry with `visit_index=None`, the same trace replays on every visit. The choice belongs to whoever builds the store: per-visit entries are useful for testing branching dialogue under fixed-but-distinct model responses, while a single wildcard entry is useful for stress-testing routing logic against a constant turn payload.

## Strict miss in the pipeline

When `miss_policy="strict"` is in effect and the pipeline asks for a turn that has no matching entry, `GenerateAnswerStage` catches the `ReplayMissError` and marks the turn as a permanent error. The result still flows through `FinalizeResultStage`, but `metadata.failure` is populated (non-`None`) with the originating stage and a reason identifying the missing key.

```python
strict_bm = Benchmark.create(name="strict-demo", version="1.0.0")
qid_known = strict_bm.add_question(
    question="What is the capital of France?",
    raw_answer="Paris",
    answer_template=TEMPLATE_PARIS,
)
qid_unknown = strict_bm.add_question(
    question="What is the capital of Italy?",
    raw_answer="Rome",
    answer_template=TEMPLATE_PARIS,
)

partial_store = ReplayStore(miss_policy="strict")
partial_store.register(
    ReplayKey(question_id=qid_known),
    ReplayEntry(raw_trace="Paris", parsed_answer_fields={"city": "Paris"}),
)

config_strict = VerificationConfig(
    answering_models=[
        ModelConfig(id="ans", model_name="gpt-fake", interface="langchain", model_provider="openai"),
    ],
    parsing_models=[
        ModelConfig(id="par", model_name="gpt-fake", interface="langchain", model_provider="openai"),
    ],
    replay_store=partial_store,
)
result_strict = strict_bm.run_verification(config_strict)
{
    "results": len(result_strict.results),
    "ok_count": sum(1 for r in result_strict.results if (r.metadata.failure is None)),
    "errors": [
        ((r.metadata.failure is None), (r.metadata.failure.reason if r.metadata.failure else "")[:60])
        for r in result_strict.results
    ],
}
```

In fall-through mode the same store would have replayed the Paris question and let the Italy question hit the live model.

## Projecting QA stores onto scenarios

Some benchmarks share a factual "ask" turn across many scenario variants: the same question is asked in several dialogue framings (casual vs authority, MCP vs no-MCP, plain vs agentic). Running the full LLM for that shared turn on every scenario is wasteful when the answer does not vary by framing. `ScenarioReplayBuilder` solves this by projecting a single QA capture onto the shared turn across many scenarios, turning one `(question, model)` QA run into N scenario-mode replay entries.

### Merge key

Projection matches on two axes: `question_id` (derived from the node's question text via `generate_question_id`) and `answering_model_id` (derived via `ModelIdentity.from_model_config(...).display_string`). The model comes from `ScenarioNode.model_override.answering_model` when set, otherwise from `config.answering_models[0]`.

### Three-phase flow

```python
from karenina.benchmark import Benchmark
from karenina.replay import ReplayStore, ScenarioReplayBuilder
from karenina.schemas.verification.config import VerificationConfig

builder = ScenarioReplayBuilder(bench, config=cfg)
builder.add_qa(qa_store, target_nodes=["ask"], scenarios=["s1", "s2"])
report = builder.validate()   # inspect before committing
store = builder.build(strict=True)
```

1. `add_qa` stages a projection (one QA store plus its target nodes and scenarios).
2. `validate` walks every staged projection without mutation and returns a `ProjectionReport` with `projected_keys`, `unmatched_targets`, `orphan_qa_entries`, and `duplicate_targets`.
3. `build` runs `validate` and registers the matched entries into a fresh `ReplayStore`. `strict=True` raises `ProjectionError` (carrying the report) on any unmatched or duplicate target. `strict=False` emits a warning and returns whatever it matched.

### Replicate canonicalization

`ScenarioReplayBuilder` emits `replicate=None` on every projected entry. The produced wildcard matches every replicate of the scenario turn under the R2 executor. The builder therefore requires QA stores captured with `replicate_selector="first"` or `"last"`, which emit `ReplayKey.replicate=None`. Passing a store captured with `replicate_selector="all"` (integer replicates preserved) is a hard error at `add_qa` time: the 3-axis specificity ladder does not fall through from a `replicate=None` probe to integer replicate entries. Recapture with `replicate_selector="first"` or `"last"` and retry.

### Worked example

```python
no_mcp_qa = ReplayStore.load("artifacts/qa_no_mcp.json")
mcp_qa = ReplayStore.load("artifacts/qa_mcp.json")

builder = ScenarioReplayBuilder(bench, config=no_mcp_cfg)
builder.add_qa(no_mcp_qa, target_nodes=["ask"],
               scenarios=["syco_hard_casual_no_mcp", "syco_hard_authority_no_mcp"])
builder.add_qa(mcp_qa, target_nodes=["ask"],
               scenarios=["syco_hard_casual_mcp", "syco_hard_authority_mcp"],
               config=mcp_cfg)

report = builder.validate()
assert not report.unmatched_targets, report
store = builder.build(strict=True)

# Attach to a verification config; Turn 1 is replayed from the canned QA answer,
# Turns 2 and 3 hit the live model.
replay_cfg = no_mcp_cfg.model_copy(update={"replay_store": store})
result_set = bench.run_verification(config=replay_cfg)
```

### Inspecting a projection before committing

`report.matched` is the number of targets that resolved to a QA entry. `report.unmatched_targets` classifies misses as `missing_scenario`, `missing_node`, or `no_qa_entry`. `report.orphan_qa_entries` classifies unused QA entries as `no_target_scenario` (no projection target referenced the question) or `model_id_never_requested` (question matched, but runtime model differed from capture). `report.duplicate_targets` lists `(scenario_id, node_id)` pairs hit by more than one projection; `build(strict=False)` applies last-projection-wins.

## Hydration mismatch policy

When a captured `parsed_answer_fields` dict cannot be validated against the current `Answer` class (the template changed, a field was renamed, a type tightened), the parse bypass has to decide whether to fail loudly or fall back to a live judge call. The setting is `VerificationConfig.replay_parse_on_hydration_mismatch`, which defaults to `"fall_through"`. The alternative is `"strict"`, which raises `ReplayHydrationError` and stops the pipeline with the captured fields embedded in the exception for debugging.

The default fall-through behaviour is the most user-friendly: it lets you iterate on the answer template without invalidating an existing capture, paying the live judge cost only on the entries that no longer match. Switch to strict mode in CI to detect template drift early.

## Legacy `interface="manual"` shim

Before the replay store existed, the only way to inject a hand-written response was to use `ModelConfig(interface="manual", manual_traces=...)`. That path is preserved bit-for-bit. When `Benchmark.run_verification` sees a `VerificationConfig` with no `replay_store` set and a manual answering model, it auto-builds a strict `ReplayStore` from the `ManualTraces` instance and threads it through the pipeline. From the user's perspective, nothing changes; from the pipeline's perspective, the manual interface becomes a thin shim over the unified replay path.

The auto-build is skipped if `config.replay_store is not None`, so passing your own store always wins. Replay honors per-replicate entries via the 3-axis specificity ladder (model, visit, replicate); canonicalize across replicates at capture time via `replicate_selector="first"` or `"last"` on `capture_from_result_set` when you want one answer to serve every replicate.

```python
from karenina.adapters.manual import ManualTraces

manual_bm = Benchmark.create(name="manual-shim-demo", version="1.0.0")
manual_qid = manual_bm.add_question(
    question="What is the capital of France?",
    raw_answer="Paris",
    answer_template=TEMPLATE_PARIS,
)

manual_traces = ManualTraces(manual_bm)
manual_traces.register_trace("What is the capital of France?", "Paris", map_to_id=True)

# This is the exact call Benchmark.run_verification makes internally when
# it detects an interface="manual" answering model with no replay_store set.
shimmed = ReplayStore.from_manual_traces(manual_traces, benchmark=manual_bm, miss_policy="strict")
[(k.question_id, e.raw_trace) for (k, e) in shimmed.entries]
```

The auto-built store carries one wildcard entry per registered trace (`answering_model_id=None`, `visit_index=None`), so any answering model in the config matches it. From `Benchmark.run_verification`'s point of view the resulting pipeline is indistinguishable from one configured with a hand-built `ReplayStore`. If you want to migrate off `interface="manual"` entirely, replace the `ManualTraces` setup with explicit `ReplayStore.register(...)` calls and drop the `manual` interface from the model config; the rest of the pipeline does not need to change.

## CLI integration

The `karenina verify` command accepts a `--replay /path/to/store.json` flag that loads a saved store and attaches it to the verification config before the run starts. The interaction with `interface="manual"` is the same: passing `--replay` always wins over auto-build, and any combination of fall-through replay plus a live model is valid.

```python
import os
import shlex
import subprocess

env = dict(os.environ)
env.update({"TERM": "dumb", "NO_COLOR": "1"})

help_text = subprocess.run(
    shlex.split("uv run karenina verify --help"),
    capture_output=True,
    text=True,
    env=env,
    check=True,
).stdout

print("`--replay` flag present:", "--replay" in help_text)
```

The flag is the recommended way to wire replay into a CI pipeline: capture once with a real model, commit the JSON file, and have CI re-run with `--replay captures/run.json` to deterministically reconstruct the result set without paying for a live call.

## Wire format reference

The JSON written by `ReplayStore.save()` looks like this. The schema is intentionally flat and stable; the `version` field is the only thing the loader uses to gate compatibility.

```python
import json
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "wire.json"
    sample = ReplayStore(miss_policy="strict")
    sample.register(
        ReplayKey(
            question_id="urn:uuid:question-paris-abcdef12",
            answering_model_id="langchain:gpt-5",
        ),
        ReplayEntry(
            raw_trace="Paris is the capital of France.",
            parsed_answer_fields={"city": "Paris"},
            captured_model_id="langchain:gpt-5",
            captured_at="2026-04-08T12:00:00+00:00",
        ),
    )
    sample.save(path)
    print(path.read_text())
```

Every key field that is `None` is still emitted by `model_dump()`, so the resulting JSON is dense but unambiguous. If you ever need to hand-edit a capture (to fix a typo in `raw_trace`, for example, or to relax a too-specific `answering_model_id`), the file is safe to edit as long as you preserve the wrapping object structure and run it through `ReplayStore.load(path)` to validate before committing.

## Exception hierarchy

The replay layer raises four exceptions, all rooted at `karenina.replay.ReplayError` (which itself inherits from `karenina.exceptions.KareninaError`).

```python
from karenina.replay import (
    ReplayError,
    ReplayHydrationError,
    ReplayMissError,
    ReplayPersistenceError,
)

[
    (cls.__name__, [base.__name__ for base in cls.__mro__[1:4]])
    for cls in (ReplayError, ReplayMissError, ReplayHydrationError, ReplayPersistenceError)
]
```

`ReplayMissError` carries the offending `key` as an attribute. `ReplayHydrationError` carries both the `captured_fields` dict and the `inner` exception that triggered the failure (typically a Pydantic `ValidationError`). `ReplayPersistenceError` is raised by `load()` for any file-level failure: missing file, malformed JSON, version mismatch, schema error, or invalid `miss_policy` string.

```python
fields = {"city": 42}  # wrong type for the Paris template

err = ReplayHydrationError(
    "fields failed validation",
    captured_fields=fields,
    inner=ValueError("city must be a string"),
)
type(err).__name__, err.captured_fields, type(err.inner).__name__
```

## Putting it together

The shortest, most idiomatic recipe for end-to-end replay is one capture call followed by one save call, then later one load call followed by one run call.

```python
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "captures.json"

    # 1. Capture from an existing live result set (here we reuse the
    # in-memory result_set we built above; in production this is a
    # real Benchmark.run_verification(live_config) result).
    captured = result_set.to_replay_store(include_parsed=True)
    captured.save(path)

    # 2. Later: load and re-run the same benchmark.
    reloaded = ReplayStore.load(path, miss_policy="strict")
    cfg = VerificationConfig(
        answering_models=[
            ModelConfig(id="ans", model_name="gpt-fake", interface="langchain", model_provider="openai"),
        ],
        parsing_models=[
            ModelConfig(id="par", model_name="gpt-fake", interface="langchain", model_provider="openai"),
        ],
        replay_store=reloaded,
    )
    replay_result = bm.run_verification(cfg)

    print("captured entries:", len(captured.entries))
    print("replay results:", len(replay_result.results))
    print("first replay result raw_llm_response:", replay_result.results[0].template.raw_llm_response)
    print("verify_result:", replay_result.results[0].template.verify_result)
```

The same pattern scales to a multi-question benchmark, a multi-turn scenario, or a mixed batch: the capture function walks the result set, the loader rebuilds the store, and the pipeline reads from the store before any LLM call happens.
