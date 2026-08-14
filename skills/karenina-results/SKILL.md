---
name: karenina-results
description: >
  Navigate and analyze karenina verification results. Use when extracting
  DataFrames, understanding scores, exporting results, or comparing across
  models and runs. Covers VerificationResult structure, template/rubric/scenario
  results, scoring, and export formats.
---

# Results and Analysis

This skill covers how to load, navigate, analyze, and export karenina
verification results. Every question that passes through the verification
pipeline produces a `VerificationResult`: a nested Pydantic model capturing
everything from the raw response to the final verdict.

**Source of truth**: `karenina/src/karenina/schemas/verification/` (result
models), `karenina/src/karenina/schemas/dataframes/` (DataFrame builders),
`karenina/src/karenina/schemas/results/` (result sets and views).

**Reference files in this skill**:
- [references/dataframe-columns.md](references/dataframe-columns.md): Column definitions for template and rubric DataFrames

---

## 1. Load Results

Results come from three sources:

### From a Benchmark Run

```python
from karenina.benchmark import Benchmark
from karenina.schemas.verification import VerificationConfig

benchmark = Benchmark.load("checkpoint.jsonld")
result_set = benchmark.run_verification(config)

# result_set is a VerificationResultSet
print(f"Total results: {len(result_set)}")
```

### From a JSON File

```python
from karenina.benchmark import ResultsIOManager

# Preferred: raises ValueError on malformed data; preserves metadata + scenario_results
result_set = ResultsIOManager.load_result_set_from_json("results.json")
```

`VerificationResultSet.model_validate(data)` remains an alternative when you
already have a parsed dict. For very large files, stream row-by-row with
`ResultsIOManager.iter_from_json(path)` (or `iter_results_from_json(path)` from
`karenina.benchmark.core.results_stream`); pass `raw=True` to yield raw dicts
instead of `VerificationResult` objects. Streaming supports only the v2.2
`{"results": [...]}` export and the legacy bare-array export — it yields **0
rows** on merged `{"runs": {...}}` exports (the shape
`ResultsStore.export_to_file` writes), because it scans a `results` key
that is not there — and a wrapped non-v2.2 file
(e.g. `{"format_version": "2.1", ..., "results": [...]}`) raises a raw
pydantic `extra_forbidden` error from the legacy metadata keys. For
merged-shape files, load with
`ResultsIOManager.load_result_set_from_json(path)`, which flattens the `runs`
object (recording run names in metadata); for v2.0/v2.1 files, use
`load_legacy_result_set` (below).

**Loading legacy (v2.0/v2.1) exports**: older exports (e.g. files under
`local_data/outputs/`, `format_version` 2.0/2.1) do not load with
`load_result_set_from_json` (it raises
`ValueError: Invalid verification result at index 0`, because legacy metadata
records models as flat strings and carries keys the current model forbids).
Load them with `ResultsIOManager.load_legacy_result_set(path)`, which
migrates each row — dropping the forbidden keys, building
`answering`/`parsing` `ModelIdentity` objects from the flat model strings,
folding the replicate fields into `replicate`, and synthesizing a
deterministic `result_id` — then validates; any failing row is named by index
in the error. Non-legacy exports are delegated to
`load_result_set_from_json` unchanged.

### From an Extended Run

Results produced by `Benchmark.extend_template` and `Benchmark.extend_rubric` (see `karenina-verification`, task 9, and the extending-runs guide in the `using-karenina` skill's `references/core_concepts/`) are standard `VerificationResultSet`s carrying a single effective `run_name`. `extend_template` appends new rows for new answerer / judge / replicate combinations (prior rows pass through verbatim). `extend_rubric` preserves row count: each prior row is deep-copied and enriched in place with the new trait scores, so `result.metadata.result_id` is stable across the extension. Load them like any other result set.

### From a Scenario Run

`VerificationResultSet` also has a scenario-specific field `scenario_results`
(`list[ScenarioExecutionResult] | None`) — populated only on scenario runs,
`None` on QA/template runs. It also carries `errors`, a list
of `(description, exception)` tuples collecting task-level failures. The engine
never raises for those failures: it completes the run and reports them here, so
`errors` is populated on QA runs too, not only scenarios.

```python
# Access ScenarioExecutionResult objects
if result_set.scenario_results:
    for scenario_result in result_set.scenario_results:
        for vr in scenario_result.turn_results:
            print(vr.metadata.question_id, vr.metadata.scenario_node, vr.template.verify_result)

# Check for task-level failures (populated on QA and scenario runs)
if result_set.errors:
    for desc, exc in result_set.errors:
        print(f"Failed: {desc}: {exc}")
```

---

## 2. Navigate VerificationResult

Each result uses nested composition with five optional sub-objects:

```
VerificationResult
  metadata              Always present: identification, timing, model info
  template              Present when template evaluation ran
  rubric                Present when rubric evaluation ran
  deep_judgment         Present when deep judgment ran (templates)
  deep_judgment_rubric  Present when deep judgment ran (rubrics)
```

Access fields through their sub-objects:

```python
result = result_set[0]

# Metadata (always present)
result.metadata.question_id       # MD5 hash of question text
result.metadata.template_id       # MD5 hash of template code
result.metadata.question_text     # Full question text
result.metadata.answering         # ModelIdentity (interface, model_name, tools)
result.metadata.parsing           # ModelIdentity
result.metadata.execution_time    # Seconds
result.metadata.timestamp         # 'YYYY-MM-DD HH:MM:SS' (space separator, no timezone)
result.metadata.result_id         # Deterministic 16-char SHA256
result.metadata.replicate         # Replicate number (1, 2, 3, ...)
result.metadata.failure           # Failure | None: structured failure metadata (None when clean)
result.metadata.caveats           # list[Caveat]: non-fatal quality flags
result.metadata.evaluation_mode   # Resolved mode, e.g. "template_only"
```

Failure is the single success/failure signal — there is no `metadata.error` or
`completed_without_errors`:

```python
if result.metadata.failure is None:
    pass  # success
else:
    f = result.metadata.failure
    f.category  # FailureCategory: content, timeout, abstention, parsing, ...
    f.stage     # str: pipeline stage where it failed
    f.reason    # str: human-readable explanation
    f.group     # FailureGroup: content | autofail | retry | abstained | system
```

```python
# Template results (if template evaluation ran)
result.template.verify_result              # bool or None: pass/fail
result.template.raw_llm_response           # Full LLM response text
result.template.parsed_llm_response        # dict: extracted fields
result.template.parsed_gt_response         # dict: ground truth fields
result.template.abstention_detected        # bool or None
result.template.embedding_similarity_score # float or None (0.0 to 1.0)
result.template.recursion_limit_reached    # bool

# Rubric results (if rubric evaluation ran)
result.rubric.llm_trait_scores             # dict[str, int | bool]
result.rubric.llm_trait_labels             # dict[str, str] (literal kind)
result.rubric.regex_trait_scores           # dict[str, bool]
result.rubric.callable_trait_scores        # dict[str, bool | int]
result.rubric.metric_trait_scores          # dict[str, dict[str, float]]
result.rubric.metric_trait_confusion_lists # dict[str, dict[str, list[str]]]
result.rubric.rubric_evaluation_performed  # bool: whether rubric evaluation ran
result.rubric.dynamic_rubric_promoted_traits # list[str] | None: promoted dynamic traits
result.rubric.agentic_trait_investigation_traces # dict[str, str] | None: raw agent traces

# Convenience methods on rubric
result.rubric.get_all_trait_scores()       # Flat dict across all types
result.rubric.get_trait_by_name("safety")  # (value, type) tuple
result.rubric.get_llm_trait_labels()       # Labels for literal traits
```

### ModelIdentity

Models are identified by composite objects, not plain strings:

```python
identity = result.metadata.answering
identity.interface      # "langchain", "claude_sdk", "manual", etc.
identity.model_name     # e.g. "claude-sonnet-4-6"
identity.tools          # Sorted list of MCP server names
identity.config_id      # ModelConfig.id when it differs from model_name (else None)
identity.display_string # Human-readable display
identity.canonical_key  # Unique key for grouping
```

### Evaluation Mode Affects Available Sub-objects

| Sub-object | template_only | template_and_rubric | rubric_only |
|------------|:------------:|:-------------------:|:-----------:|
| metadata | Always | Always | Always |
| template | Present | Present | None |
| rubric | None | Present | Present |
| deep_judgment | Optional | Optional | None |
| deep_judgment_rubric | None | Optional | Optional |

---

## 3. Extract DataFrames

### VerificationResultSet Views

The result set provides specialized views, each with its own DataFrame export:

| Accessor | Returns | Row Granularity |
|----------|---------|-----------------|
| `get_template_results()` | `TemplateResults` | One row per parsed field |
| `get_rubrics_results(include_deep_judgment=False)` | `RubricResults` | One row per trait |
| `get_judgment_results()` | `JudgmentResults` | Deep judgment data |
| `get_rubric_judgments_results()` | `RubricJudgmentResults` | Per-trait per-excerpt |
| `get_scenario_results()` | `ScenarioResults` | Scenario executions/turns/outcomes |

### Template DataFrames

```python
template_results = result_set.get_template_results()

# Field comparison: one row per parsed field per result
df = template_results.to_dataframe()

# Regex results: one row per regex pattern per result (0x0 frame with no
# columns when the run has no regex traits)
df_regex = template_results.to_regex_dataframe()

# Token usage: one row per usage stage per result
df_usage = template_results.to_usage_dataframe()
df_totals = template_results.to_usage_dataframe(totals_only=True)
```

### Rubric DataFrames

```python
rubric_results = result_set.get_rubrics_results()

# All traits (default)
df = rubric_results.to_dataframe()

# Include deep judgment columns (trait_reasoning / trait_excerpts / trait_hallucination_risk)
rubric_results_with_dj = result_set.get_rubrics_results(include_deep_judgment=True)

# Filter by trait type
df_llm = rubric_results.to_dataframe(trait_type="llm")
df_llm_score = rubric_results.to_dataframe(trait_type="llm_score")
df_llm_binary = rubric_results.to_dataframe(trait_type="llm_binary")
df_llm_literal = rubric_results.to_dataframe(trait_type="llm_literal")
df_regex = rubric_results.to_dataframe(trait_type="regex")
df_callable = rubric_results.to_dataframe(trait_type="callable")
df_metric = rubric_results.to_dataframe(trait_type="metric")
df_agentic = rubric_results.to_dataframe(trait_type="agentic")
```

### Scenario DataFrames

```python
scenario_results = result_set.get_scenario_results()

df_exec   = scenario_results.to_dataframe()          # 1 row per execution
df_turn   = scenario_results.to_turn_dataframe()     # 1 row per turn
df_outcome = scenario_results.to_outcome_dataframe() # 1 row per outcome criterion
```

See [references/dataframe-columns.md](references/dataframe-columns.md) for
complete column definitions.

---

## 4. Understand Scoring

### Template Scoring: field_match and verify_result

Template verification compares parsed fields between the ground truth and
the LLM response:

- **field_match**: Per-field boolean. `True` if ground truth value matches
  LLM value for that field. `False` if values differ or one is missing.
- **verify_result**: Per-question boolean from the template's `verify()`
  method. Typically `True` when all fields match, but custom `verify()` logic
  can implement partial matching, thresholds, or domain-specific rules.

Guards can override `verify_result`:
- Abstention detected: overrides to `False`
- Sufficiency check failed: overrides to `False`
- Embedding override: can flip `True` to `False` or `False` to `True`

Check `*_override_applied` fields to distinguish "failed on merits" from
"overridden by guard."

### Rubric Scoring: Trait Types

| Trait Type | Score Type | Range |
|-----------|-----------|-------|
| LLM (score) | int | 1 to 5 |
| LLM (binary) | bool | True/False |
| LLM (literal) | int + label | Index 0 to N-1, with class name label |
| Regex | bool | True/False |
| Callable | bool or int | Depends on callable |
| Metric | dict[str, float] | precision, recall, f1, etc. |
| Agentic | bool, int, float, or str | Depends on agent evaluation |

For literal-kind LLM traits, `trait_score` is the integer class index and
`trait_label` is the human-readable class name. Error state: score=-1, label
contains the invalid value.

### Scenario Scoring: Outcome Results

Scenario outcomes are evaluated after all turns complete:

```python
scenario_result.outcome_results
# {"final_correct": True, "knowledge_retained": False, "turn_count": 3}
```

Outcome values are `bool`, `int`, or `float` depending on the check type
(TurnCheck/ResultCheck = bool; CountTurns = int; FirstMatchIndex = int).

---

## 5. Export

There is no `benchmark.results` property — export through `ResultsStore` or
the canonical exporter functions at `karenina.benchmark`.

### JSON Export

```python
# From VerificationResultSet
result_set.model_dump()       # Returns dict
result_set.model_dump_json()  # Returns JSON string

# From ResultsStore
store = ResultsStore()
store.add(result_set, run_name="my-run")
store.export_to_file("results.json")  # writes JSON
```

For streaming/atomic v2.2 JSON, use
`export_verification_results_json_stream(job, results_iter, *,
is_complete=False, scenario_results=None, out_path=Path("results.json"))`.

### CSV Export

```python
# Via DataFrame
df = template_results.to_dataframe()

df.to_csv("template_results.csv", index=False)

# Canonical CSV exporter (consolidates rubric traits into columns).
# It needs a VerificationJob — job=None crashes on job_id, and the
# config must satisfy VerificationConfig validation (parsing model
# required; answering models required unless parsing_only=True).
from karenina.schemas.config import ModelConfig
from karenina.schemas.verification import VerificationConfig, VerificationJob

config = VerificationConfig(
    parsing_only=True,
    parsing_models=[
        ModelConfig(
            id="parse-haiku",
            model_provider="anthropic",
            model_name="claude-haiku-4-5",
            interface="langchain",
        )
    ],
)
job = VerificationJob(
    job_id="offline-export",
    run_name="my-run",
    status="completed",
    config=config,
    total_questions=len(result_set),
)
csv_str = export_verification_results_csv(job, result_set, global_rubric=None)
Path("template_results.csv").write_text(csv_str)
```
Markdown export: `df.to_markdown()` requires the `tabulate` package. It is
included in the `dev` extra (`uv sync --extra dev`); on a bare environment
use `uv run --with tabulate python ...` or export CSV/JSON instead.

## Post-Hoc Rubric Evaluation

Re-score archived results against a rubric without re-running the pipeline.
`karenina.benchmark` exposes:

```python
from karenina.benchmark import (
    evaluate_rubric_on_results,
    evaluate_rubric_on_texts,
    RowContext,
)

# results: VerificationResultSet (or list[VerificationResult]); rubric: Rubric
judgments = evaluate_rubric_on_results(
    results, rubric, parsing_model,
    text_selector=...,      # callable(result) -> str to judge
    row_context=lambda r: RowContext(question="...", ground_truth=None),
    row_filter=...,         # optional callable(result) -> bool
    max_workers=4,
)
for j in judgments:          # PostHocJudgment
    j.key, j.scores, j.labels, j.error

# Judge arbitrary texts directly: items are (ResultRowKey, text, RowContext)
evaluate_rubric_on_texts(items, rubric, parsing_model, max_workers=4)
```

`PostHocJudgment(key, sibling_keys, scores, labels, error,
representative_result_id, sibling_result_ids)` carries the verdict for one row.

---

## 6. Compare Across Models

### Group by Model

```python
# by: "answering" (default) | "parsing" | "both"
by_model = result_set.group_by_model(by="answering")
for model_key, model_results in by_model.items():
    template_results = model_results.get_template_results()
    pass_rate = template_results.aggregate_pass_rate(by="question_id")
    print(f"{model_key}: {pass_rate}")
```

### Group by Question

```python
by_question = result_set.group_by_question()
for qid, q_results in by_question.items():
    print(f"Question {qid}: {len(q_results)} results")
```

```python
# VerificationResultSet.filter signature
filtered = result_set.filter(
    question_ids=["q1", "q2"],
    answering_models=["langchain:claude-sonnet-4-6"],
    parsing_models=None,
    replicates=[1, 2],
    completed_only=True,
    has_template=False,
    has_rubric=False,
    has_judgment=False,
)

# Template-level filtering (question_ids / answering_models / parsing_models /
# replicates / passed_only / failed_only)
passed = template_results.filter(passed_only=True)
failed = template_results.filter(failed_only=True)
```

`answering_models` / `parsing_models` values match
`result.metadata.answering.display_string` — the `interface:model_name` key
(e.g. `"langchain:claude-sonnet-4-6"`, `"openai_endpoint:gpt-oss-120b"`), not
the bare model name. Matching is exact string equality with no fuzzy fallback,
so a bare name like `"claude-sonnet-4-6"` silently returns an empty set.
Display strings grow suffixes when identity carries them: `" (config_id)"` for
custom `ModelConfig.id`s and `" +[tool1, tool2]"` for MCP-enabled answerers —
inspect the exact keys in your data first:

```python
keys = sorted({r.metadata.answering_model for r in result_set})
```

### Aggregation

Built-in aggregation strategies:

| Strategy | Behavior | Best For |
|----------|----------|----------|
| `"mean"` | Arithmetic mean | Numeric scores |
| `"median"` | Median value | Scores with outliers |
| `"mode"` | Most common value | Categorical values |
| `"majority_vote"` | True if >50% True | Boolean traits, pass/fail |
| `"first"` | First non-null value | Metadata fields |
| `"count"` | Count occurrences | Distribution analysis |

```python
# Template aggregation
pass_rates = template_results.aggregate_pass_rate(by="question_id")

# Rubric aggregation
avg_scores = rubric_results.aggregate_llm_traits(strategy="mean", by="question_id")
regex_votes = rubric_results.aggregate_regex_traits(
    strategy="majority_vote", by="answering_model",
)

# Custom aggregator
class WeightedMean:
    def aggregate(self, series, **kwargs):
        return series.mean()

rubric_results.register_aggregator("weighted_mean", WeightedMean())
```

## Repairing Failed Rows

Re-run only the failed rows, then splice replacements back into an exported
result set. `karenina.benchmark` exposes:

```python
from karenina.benchmark import (
    RepairSelection, repair_results_export, select_repair_rows, splice_repaired_rows,
)

selection = RepairSelection(
    question_ids=[...], answerer_keys=[...], parser_keys=[...], replicates=[...],
    failure_groups=[...], failure_categories=[...], failure_stages=[...],
)  # .matches(result) -> bool

outcome = repair_results_export(
    benchmark, source_path, config, selection,
    mode="replay", output_path="repaired.json", dry_run=True,
)  # RepairOutcome: selected_count, replaced_count, mode, dry_run, ...
```

Or one-shot: `karenina repair SOURCE --benchmark B --preset P [--mode replay|live] [--dry-run]`.

---

## Key Imports

```python
# Result types
from karenina.schemas.verification import (
    VerificationResult,
    VerificationResultMetadata,
    VerificationResultTemplate,
    VerificationResultRubric,
    VerificationResultDeepJudgment,
    VerificationResultDeepJudgmentRubric,
    ModelIdentity,
)

# Result collections + row keys + failure metadata
from karenina.schemas.results import ResultRowKey, VerificationResultSet
from karenina.schemas.results.failure import Failure, FailureCategory

# DataFrame builders (used internally by views)
from karenina.schemas.dataframes.template import TemplateDataFrameBuilder
from karenina.schemas.dataframes.rubric import RubricDataFrameBuilder

# Scenario results
from karenina.schemas.scenario.state import (
    ScenarioExecutionResult,
    ScenarioState,
    TurnRecord,
)

# Results I/O, store, post-hoc & repair (paper workflows)
from karenina.benchmark import (
    ResultsIOManager,
    evaluate_rubric_on_results,
    evaluate_rubric_on_texts,
    RowContext,
    PostHocJudgment,
    RepairSelection,
    RepairOutcome,
    repair_results_export,
    select_repair_rows,
    splice_repaired_rows,
    export_verification_results_json_stream,
    export_verification_results_csv,
)
from karenina.benchmark.core import ResultsStore
```

<!-- AUTO:result-api -->
### VerificationResult Accessor Reference

> Auto-generated for karenina v0.1.0.

Always access through sub-objects (`result.metadata.*`,
`result.template.*`, `result.rubric.*`), never directly on result.

#### metadata (always present)

| Path | Type |
|------|------|
| `result.metadata.question_id` | `str` |
| `result.metadata.template_id` | `str` |
| `result.metadata.question_text` | `str` |
| `result.metadata.failure` | `Failure | None` |
| `result.metadata.caveats` | `list[Caveat]` |
| `result.metadata.warnings` | `list[str]` |
| `result.metadata.retry_counts` | `dict[str, dict[str, int]] | None` |
| `result.metadata.evaluation_mode` | `str | None` |
| `result.metadata.answering` | `ModelIdentity` |
| `result.metadata.parsing` | `ModelIdentity` |
| `result.metadata.execution_time` | `float` |
| `result.metadata.timestamp` | `str` |
| `result.metadata.result_id` | `str` |
| `result.metadata.run_name` | `str | None` |
| `result.metadata.replicate` | `int | None` |
| `result.metadata.scenario_id` | `str | None` |
| `result.metadata.scenario_node` | `str | None` |
| `result.metadata.scenario_turn` | `int | None` |
| `result.metadata.scenario_path` | `list[str] | None` |

#### template (when template evaluation ran)

| Path | Type |
|------|------|
| `result.template.verify_result` | `bool | None` |
| `result.template.raw_llm_response` | `str` |
| `result.template.parsed_llm_response` | `dict[str, Any] | None` |
| `result.template.parsed_gt_response` | `dict[str, Any] | None` |
| `result.template.abstention_detected` | `bool | None` |
| `result.template.abstention_reasoning` | `str | None` |
| `result.template.embedding_similarity_score` | `float | None` |
| `result.template.recursion_limit_reached` | `bool` |
| `result.template.field_results` | `dict[str, bool | None] | None` |
| `result.template.verify_granular_result` | `Any | None` |
| `result.template.response_timeout_partial` | `bool` |
| `result.template.usage_metadata` | `dict[str, dict[str, Any]] | None` |

#### rubric (when rubric evaluation ran)

| Path | Type |
|------|------|
| `result.rubric.rubric_evaluation_performed` | `bool` |
| `result.rubric.llm_trait_scores` | `dict[str, Any] | None` |
| `result.rubric.llm_trait_labels` | `dict[str, str] | None` |
| `result.rubric.regex_trait_scores` | `dict[str, bool] | None` |
| `result.rubric.callable_trait_scores` | `dict[str, bool | int | float] | None` |
| `result.rubric.metric_trait_scores` | `dict[str, dict[str, float]] | None` |
| `result.rubric.metric_trait_confusion_lists` | `dict[str, dict[str, list[str]]] | None` |
| `result.rubric.agentic_trait_scores` | `dict[str, int | bool | float | str | list[Any] | None] | None` |
| `result.rubric.dynamic_rubric_skipped_traits` | `dict[str, str] | None` |
| `result.rubric.dynamic_rubric_promoted_traits` | `list[str] | None` |
| `result.rubric.get_all_trait_scores()` | `dict[str, int | bool | float | str | list[Any] | dict[str, float] | None]` |
| `result.rubric.get_trait_by_name(name)` | `tuple[Any, str] | None` |
| `result.rubric.get_llm_trait_labels()` | `dict[str, str]` |

#### Paths That Do NOT Exist on VerificationResult

| Wrong | Correct |
|-------|---------|
| `result.template.parsed_output` | `result.template.parsed_llm_response` |
| `result.verify_result` | `result.template.verify_result` |
| `result.question_id` | `result.metadata.question_id` |
| `result.raw_response` | `result.template.raw_llm_response` |
| `result.error` | `result.metadata.failure (Failure.category / .stage / .reason)` |
| `result.metadata.error` | `result.metadata.failure (Failure.category / .stage / .reason)` |
| `result.metadata.completed_without_errors` | `result.metadata.failure is None` |

<!-- /AUTO:result-api -->

**Full karenina docs**: The `using-karenina` skill contains the complete karenina documentation in its `references/` directory. Consult it for API details not covered here.
