# Agentic Rubric Evaluation

This page covers the internal machinery of Stage 11b (AgenticRubricEvaluation): how the stage dispatches to individual or shared agent strategies, how the evaluator runs investigation and extraction, and how results flow into FinalizeResult. It is for contributors and power users who need to understand what happens under the hood.

For a conceptual overview of agentic traits and how they differ from other trait types, see the [agentic traits concept page](../core_concepts/rubrics/agentic-traits.md). For the general pipeline architecture, see [verification-pipeline.md](../core_concepts/verification-pipeline.md). For the parallel Stage 7b internals (agentic template parsing), see [agentic-evaluation.md](agentic-evaluation.md).

## 1. Pipeline Position

Stage 11b sits between [RubricEvaluation](stages.md) (Stage 11) and [DeepJudgmentRubricAutoFail](deep-judgment-rubrics.md) (Stage 12) in the pipeline. It evaluates only `AgenticRubricTrait` instances; standard trait types (LLM, regex, callable, metric) are handled by Stage 11.

The `StageOrchestrator.from_config()` method includes Stage 11b when the rubric contains agentic traits:

```python
# orchestrator.py
_dynamic_has_agentic = dynamic_rubric is not None and bool(dynamic_rubric.agentic_traits)

if evaluation_mode == "template_and_rubric" and (
    (rubric and rubric.agentic_traits) or _dynamic_has_agentic
):
    stages.append(AgenticRubricEvaluationStage())
```

In `rubric_only` mode, the same inclusion logic applies without the `evaluation_mode` guard. The `_dynamic_has_agentic` check ensures that agentic traits contributed by a dynamic rubric also trigger Stage 11b.

## 2. AgenticTraitEvaluator

**File**: `src/karenina/benchmark/verification/evaluators/rubric/agentic_trait.py`.

`AgenticTraitEvaluator` is the core evaluation unit. It takes a resolved `ModelConfig` at construction time and exposes two entry points: `evaluate_trait()` for the full two-step flow, and `run_extraction()` for extraction alone (used by the shared strategy).

### evaluate_trait()

Runs the complete two-step evaluation for a single trait:

1. **Investigation** (`_run_investigation()`): launches an agent via `AgentPort.run()` to investigate the response and/or workspace. Returns the raw investigation trace string.
2. **Extraction** (`run_extraction()`): sends the investigation trace to `ParserPort.parse_to_pydantic()` to extract a structured score.

### Error Handling

Errors at each step produce different return signatures:

| Failure point | Returned `(score, trace)` | Rationale |
|---------------|--------------------------|-----------|
| Agent investigation fails | `(None, None)` | No trace was produced; nothing to preserve |
| Score extraction fails | `(None, investigation_trace)` | The trace has diagnostic value even without a score |
| Both succeed | `(score, investigation_trace)` | Normal result |

Both failure modes log at `WARNING` level with `exc_info=True`, so the exception details appear in logs without halting the pipeline.

### _run_investigation()

Builds the agent invocation from the trait's configuration:

- **System prompt**: identifies the agent as an evaluation agent, embeds `trait.description`, and states the expected output kind (`trait.kind`)
- **User prompt**: assembled from context mode filtering (see Section 5)
- **Agent config**: `max_turns` from `trait.max_turns`, `timeout` from `trait.timeout_seconds`, `workspace_path` from the resolved workspace

Returns `result.raw_trace` from the agent run.

### run_extraction()

This method is public because the shared strategy needs to call it directly (one shared investigation, then per-trait extraction). It dispatches on `trait.kind`:

| Kind | Target schema | Extracted field |
|------|--------------|----------------|
| `boolean` | `SingleBooleanScore` | `.result` (bool) |
| `score` | `SingleNumericScore` | `.score` (int) |
| `literal` | `SingleLiteralClassification` | `.classification` (str, then resolved to int index) |
| `type[BaseModel]` (template kind) | The user's `BaseModel` subclass | `.model_dump()` (dict of all fields) |

For `literal` traits, `_resolve_literal_index()` maps the classification string to its position in the `trait.classes` dict. If the classification does not match any defined class, it returns `-1`.

For template kind traits, `run_extraction()` delegates to `_extract_template()` instead of the standard extraction flow. See Section 2.1 below.

The extraction prompt includes kind-specific context: score range for `score` traits, class descriptions for `literal` traits.

### _extract_template()

This method handles extraction for template kind traits. Instead of building a score-extraction prompt, it sends the investigation trace to `ParserPort.parse_to_pydantic()` with the user's `BaseModel` subclass as the target schema. The parser produces a populated instance of that class, and the method returns `model_dump()` of the result.

The system prompt instructs the parser to fill every field based on evidence from the investigation. Field descriptions on the `BaseModel` guide the parser toward correct extraction, making well-described fields important for accuracy.

The returned dict is stored in `agentic_trait_scores` with dot-notation keys: each field becomes `{trait_name}.{field_name}`. This flattening happens in Stage 11b's `_execute_individual` (or `_execute_shared`) after `evaluate_trait()` returns.

## 3. Strategy Dispatch

**File**: `src/karenina/benchmark/verification/stages/pipeline/agentic_rubric_evaluation.py`.

The stage's `execute()` method reads `context.agentic_rubric_strategy` (default: `"individual"`) and dispatches accordingly.

### Individual Strategy (`_execute_individual`)

Evaluates each trait with its own agent. For every `AgenticRubricTrait`:

1. Resolve the model via `_resolve_model()` (see Section 3.1)
2. Create an `AgenticTraitEvaluator` with the resolved model
3. Call `evaluator.evaluate_trait()` for the full investigation + extraction cycle
4. Collect `(score, trace)` into the result dicts

Traits whose resolved model lacks `agent_factory` support are skipped with `(None, None)`.

### Shared Strategy (`_execute_shared`)

Evaluates all traits with a single shared agent, then extracts per-trait scores:

1. Resolve models for all traits
2. Verify all valid models are identical (same `interface`, `model_provider`, `model_name`). If they differ, fall back to individual strategy automatically.
3. Build a combined investigation prompt listing all trait descriptions
4. Run one shared agent investigation
5. For each trait, call `evaluator.run_extraction()` against the shared trace

If the shared investigation fails (agent exception), the stage falls back to the individual strategy. Per-trait extraction failures within the shared strategy set `score=None` for that trait while preserving the shared trace.

### 3.1. Model Resolution (`_resolve_model`)

For each trait:

```
trait.model_override  (if set)  →  resolved model
        or
context.parsing_model (fallback) →  resolved model
```

After resolution, the method checks `AdapterRegistry.get_spec(model.interface)` to confirm that the interface has an `agent_factory` registered. If no agent support is available, `_resolve_model()` returns `None` and the trait is skipped.

## 4. Artifact Contract

### Requires

| Key | Source Stage |
|-----|-------------|
| `RAW_LLM_RESPONSE` | GenerateAnswer (Stage 2) |

### Produces

| Key | Type | Description |
|-----|------|-------------|
| `AGENTIC_RUBRIC_EVALUATION_PERFORMED` | `bool` | Always `True` when the stage runs |
| `AGENTIC_TRAIT_SCORES` | `dict[str, int \| bool \| float \| str \| list \| None]` | Trait name to score. Template-kind agentic traits contribute float, string, or list values under dot-notation keys. `None` indicates evaluation failure. |
| `AGENTIC_TRAIT_INVESTIGATION_TRACES` | `dict[str, str \| None]` | Trait name to investigation trace. `None` if agent failed before producing output. |
| `AGENTIC_TRAIT_EXTRACTION_METADATA` | `dict[str, dict[str, str \| None]]` | Per-trait extraction provenance, keyed by base trait name. Each entry records how the score was recovered. `method` is `local_json` (parsed from a JSON block in the trace), `parser_after_local_json_failed` (the `ParserPort` succeeded after the local JSON attempt failed), or `failed` (both paths failed). `local_json_error` and `parser_error` carry the corresponding error strings or `None`. |

All four are stored as both artifacts (for downstream stages) and result fields (for `FinalizeResult`), using `set_artifact_and_result()`.

### should_run() Conditions

The stage skips itself when any of the following are true:

- A prior stage set `context.error`
- `context.rubric` is `None`
- `rubric.agentic_traits` is empty

Note that the orchestrator already gates stage inclusion by `evaluation_mode` and the presence of `rubric.agentic_traits`, so `should_run()` only needs to check runtime conditions.

## 5. Context Mode Filtering

Each `AgenticRubricTrait` has a `context_mode` field that controls what the investigation agent receives. The evaluator builds the user prompt accordingly:

| `context_mode` | Agent sees trace? | Agent sees workspace? | Use case |
|----------------|:-----------------:|:---------------------:|----------|
| `workspace_only` | No | Yes | Strictest: agent must discover everything independently from workspace artifacts |
| `trace_and_workspace` | Yes | Yes | Balanced: agent reviews the answering trace and can verify workspace artifacts |
| `trace_only` | Yes | No | No workspace access; useful when evaluation depends only on response content |

The prompt construction logic in `_run_investigation()`:

```python
if trait.context_mode in ("trace_and_workspace", "trace_only") and raw_llm_response:
    user_parts.append(f"\n--- ANSWERING AGENT TRACE ---\n{raw_llm_response}\n--- END TRACE ---")

if workspace_path and trait.context_mode != "trace_only":
    user_parts.append(f"\nWorkspace directory: {workspace_path}")
```

The question text is always included, regardless of mode.

## 6. Shared Strategy Merging

When the shared strategy runs a single agent for multiple traits, the stage must reconcile potentially different per-trait configurations:

| Parameter | Merge rule | Rationale |
|-----------|-----------|-----------|
| `max_turns` | `max()` across all valid traits | The shared agent must have enough turns to investigate the most demanding trait |
| `timeout_seconds` | `max()` across all valid traits | Same reasoning for timeout |
| Include trace | `any()` trait has `trace_and_workspace` or `trace_only` | Union: if any trait needs the trace, include it |
| Include workspace | `any()` trait has `trace_and_workspace` or `workspace_only` | Union: if any trait needs workspace access, include it |

The combined investigation prompt lists all trait descriptions as bullet points:

```python
combined_desc_parts = [f"- {trait.name}: {trait.description}" for trait in valid_traits]
```

The agent is instructed to "report findings for each criterion clearly so scores can be extracted per trait."

## 7. Stage Interactions

### FinalizeResult Condition

`FinalizeResult` (Stage 13) creates the `VerificationResultRubric` sub-object when either standard rubric evaluation or agentic rubric evaluation was performed:

```python
agentic_evaluation_performed = context.get_result_field(
    ArtifactKeys.AGENTIC_RUBRIC_EVALUATION_PERFORMED, False
)

if rubric_evaluation_performed or agentic_evaluation_performed:
    # Build VerificationResultRubric ...
```

This means a rubric with only agentic traits (no LLM/regex/callable/metric traits) will still produce a rubric result section, because `agentic_evaluation_performed` is `True` even when `rubric_evaluation_performed` is `False`.

Within `VerificationResultRubric`, agentic results occupy three dedicated fields:

| Field | Type | Source |
|-------|------|--------|
| `agentic_trait_scores` | `dict[str, int \| bool \| float \| str \| list \| None] \| None` | `AGENTIC_TRAIT_SCORES` result field |
| `agentic_trait_investigation_traces` | `dict[str, str] \| None` | `AGENTIC_TRAIT_INVESTIGATION_TRACES` result field |
| `agentic_trait_extraction_metadata` | `dict[str, dict[str, str \| None]] \| None` | `AGENTIC_TRAIT_EXTRACTION_METADATA` result field (per-trait `method`, `local_json_error`, `parser_error`) |

### Runner Auto-Upgrade Check

In `run_single_model_verification()`, the runner automatically upgrades `evaluation_mode` from `"template_only"` to `"template_and_rubric"` when a rubric with any traits (including agentic traits) is provided:

```python
if (
    rubric
    and (
        rubric.llm_traits or rubric.regex_traits or rubric.callable_traits
        or rubric.metric_traits or rubric.agentic_traits
    )
    and evaluation_mode == "template_only"
):
    evaluation_mode = "template_and_rubric"
```

This ensures that passing a rubric with only agentic traits triggers Stage 11b without requiring the caller to explicitly set `evaluation_mode`.

### Orchestrator Registration Order

The orchestrator places Stage 11b after all standard rubric stages:

```
... → RubricEvaluationStage (11) → DeepJudgmentRubricAutoFailStage (12) → AgenticRubricEvaluationStage (11b) → FinalizeResultStage (13)
```

In `rubric_only` mode, Stage 11b appears after the standard rubric + deep judgment block and before FinalizeResult. Stage 11b does not depend on Stage 11's output; each handles disjoint trait types.

### Other Stage Interactions

| Stage | Interaction |
|-------|------------|
| **GenerateAnswer** (2) | Produces `RAW_LLM_RESPONSE`, the only artifact Stage 11b requires. Also resolves workspace (used by agentic traits that need workspace access). |
| **RubricEvaluation** (11) | Handles LLM, regex, callable, and metric traits. Does not touch agentic traits; the two stages operate on disjoint trait sets. |
| **DeepJudgmentRubricAutoFail** (12) | Operates on standard rubric traits. Does not interact with agentic trait results. |
| **FinalizeResult** (13) | Reads agentic result fields and wires them into `VerificationResultRubric` (see above). |

## 8. Trace Materialization

When any `AgenticRubricTrait` in the rubric has `materialize_trace=True`, Stage 11b writes the answering agent trace to a file instead of inlining it in the investigation prompt. This is useful for long traces that would consume excessive context.

### _write_trace_file staticmethod

**File**: `benchmark/verification/stages/pipeline/agentic_rubric_evaluation.py`

`_write_trace_file()` places the trace file under `<workspace>/.karenina/traces/` when a workspace path is available. If no workspace is set, it creates a temporary directory as a fallback. The filename encodes the question ID and, when present, the scenario turn number. The method returns the `Path` to the written file.

### Stage-Level Lifecycle

The stage handles materialization in a single pass, not per-trait:

1. Before evaluating any trait, the stage checks `any(t.materialize_trace for t in traits)`. If true, it calls `_write_trace_file()` once.
2. The resulting `trace_file_path` is passed to `evaluate_trait()` for every trait (regardless of whether that specific trait has `materialize_trace=True`). The evaluator checks `trait.materialize_trace` before using the path.
3. After all evaluations complete, the stage checks `any(t.persist_trace for t in traits)`. If no trait requests persistence, the trace file is deleted. If any trait sets `persist_trace=True`, the file remains.

### Prompt Behavior

When `materialize_trace=True` and a `trace_file_path` is available, `_run_investigation()` replaces the inline trace with a reference:

```
The full agent trace is saved to: /path/to/.karenina/traces/trace_q_xyz.txt
Use file tools (grep, search, read) to examine it.
```

This allows the investigation agent to selectively search the trace using file tools rather than processing the entire trace in its context window.

### Interaction with Context Modes

`materialize_trace=True` requires `context_mode` to include the trace (`"trace_only"` or `"trace_and_workspace"`). Setting it with `context_mode="workspace_only"` raises a `ValueError` at validation time, because there is no trace to materialize.

## 9. Key File Reference

| Domain | File (relative to `karenina/src/karenina/`) |
|--------|------|
| Stage implementation (Stage 11b) | `benchmark/verification/stages/pipeline/agentic_rubric_evaluation.py` |
| Evaluator (investigation + extraction) | `benchmark/verification/evaluators/rubric/agentic_trait.py` |
| AgenticRubricTrait schema | `schemas/entities/rubric.py` |
| ArtifactKeys (agentic rubric section) | `benchmark/verification/stages/core/base.py` |
| VerificationContext (agentic rubric fields) | `benchmark/verification/stages/core/base.py` |
| Stage orchestrator (Stage 11b registration) | `benchmark/verification/stages/core/orchestrator.py` |
| Pipeline runner (auto-upgrade, config threading) | `benchmark/verification/runner.py` |
| FinalizeResult (rubric result assembly) | `benchmark/verification/stages/pipeline/finalize_result.py` |
| VerificationResultRubric (agentic fields) | `schemas/verification/result_components.py` |
| Extraction output schemas | `schemas/outputs/rubric.py` |
| Adapter registry (agent_factory check) | `adapters/registry.py` |

## 10. Next Steps

- [Agentic Traits](../core_concepts/rubrics/agentic-traits.md): conceptual overview and usage guide
- [Agentic Evaluation](agentic-evaluation.md): Stage 7b internals (agentic template parsing)
- [Verification Pipeline](../core_concepts/verification-pipeline.md): the verification pipeline execution engine (13 stages with sub-stages 7a/7b and 11a/11b plus the always-on PlaceholderRetryAutoFail guard)
- [Rubrics](../core_concepts/rubrics/index.md): rubric architecture and trait types
- [Deep Judgment Rubrics](deep-judgment-rubrics.md): deep judgment for standard rubric traits
