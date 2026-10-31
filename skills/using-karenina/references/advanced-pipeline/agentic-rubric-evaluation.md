# Agentic Rubric Evaluation

This page covers the internal machinery of Stage 11b (AgenticRubricEvaluation): how the stage dispatches to individual or shared agent strategies, which judge interfaces it accepts, how the evaluator runs investigation and extraction, how judge usage and judge failures are recorded, and how results flow into FinalizeResult. It is for contributors and power users who need to understand what happens under the hood.

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

Runs the complete two-step evaluation for a single trait and returns an `AgenticTraitEvaluation`:

1. **Investigation** (`_run_investigation()`): runs the judge on the response and/or workspace through `AgentPort.run()` or a single `LLMPort` call (see below). Returns an `AgenticInvestigation`: the raw trace text, the structured trace messages, the usage, and whether the agent stopped early.
2. **Extraction** (`run_extraction()`): sends the investigation trace to `ParserPort.parse_to_pydantic()` to extract a structured score.

| `AgenticTraitEvaluation` field | Content |
|--------------------------------|---------|
| `score` | The extracted score (a dict for template kind), or `None` when the investigation or the extraction failed |
| `trace` | The investigation trace, or a failure record when the investigation raised |
| `messages` | The structured investigation messages, or `None` when the investigation raised |
| `investigation_usage` | Usage of the agent run or the single LLM call, or `None` |
| `extraction_usage` | Usage of the extraction parser call, or `None` when the score was recovered locally or extraction failed |
| `investigation_error` | The exception that ended the investigation, or `None` |
| `extraction_error` | The exception raised by score extraction, or `None` |
| `investigation_incomplete` | `"turn_limit"` or `"timeout"` when the agent stopped early and returned a partial trace, else `None` |

### Error Handling

The evaluator never raises for a judge problem. It records what happened, and the stage decides how to report it (see [Section 9](#9-judge-failures-and-partial-investigations)):

| Outcome | `score` | `trace` | `messages` | Recorded in |
|---------|---------|---------|------------|-------------|
| Investigation raises | `None` | `[investigation failed: <ExceptionType>: <message>]` | `None` | `investigation_error` |
| Extraction raises | `None` | The investigation trace | The investigation messages | `extraction_error` |
| Agent stops at its turn limit or timeout | Extracted from the partial trace | The partial trace | The partial messages | `investigation_incomplete` |
| Both steps succeed | The score | The investigation trace | The investigation messages | Nothing |

Both failure modes log at `WARNING` level with `exc_info=True`. A timeout or turn limit is not an exception here: the adapter returns the partial trace with `timeout_reached` or `limit_reached` set, and extraction still runs on it.

### _run_investigation()

Picks the port, then builds the judge invocation from the trait's configuration.

**Port selection.** The judge runs through `AgentPort` when any of these holds:

- `trait.context_mode` is not `"trace_only"`,
- a workspace path is set,
- the trace is materialized to a file (`materialize_trace=True`), or
- the judge's `ModelConfig` sets `mcp_urls_dict`.

Otherwise (a `trace_only` trait with the trace inlined and no MCP servers) it makes one `LLMPort` call. The structured messages then hold a single assistant message with the response text. Server-side tools configured on the judge still run inside that call, for example Google Search on a [`google_genai`](../advanced-adapters/available-adapters.md#google_genai-gemini-through-the-google-genai-sdk) judge with `builtin_tools`.

On the agent path the judge receives its MCP servers: every entry of `mcp_urls_dict` becomes a streamable HTTP server, built the same way GenerateAnswer builds the answering model's. MCP timeouts (`mcp_http_timeout`, `mcp_sse_read_timeout`) come from the same `ModelConfig`.

**Prompt.**

- **System prompt**: identifies the agent as an evaluation agent, embeds `trait.description`, states the expected output kind (`trait.kind`), and says what the judge may do, from its port capabilities: inspect files and run commands, inspect files only, use the response text and its tools, or use only the response text
- **User prompt**: assembled from context mode filtering (see Section 5)
- **Agent config**: `max_turns` from `trait.max_turns`, `timeout` from `trait.timeout_seconds`, `workspace_path` from the resolved workspace

Returns the agent's `raw_trace`, its `trace_messages`, its `usage`, and the early-stop reason (`"timeout"` from `timeout_reached`, `"turn_limit"` from `limit_reached`).

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

The parser call's usage is left in `last_extraction_usage` (`None` when no parser call succeeded), so the stage can record it.

### _extract_template()

This method handles extraction for template kind traits. Instead of building a score-extraction prompt, it sends the investigation trace to `ParserPort.parse_to_pydantic()` with the user's `BaseModel` subclass as the target schema. The parser produces a populated instance of that class, and the method returns `model_dump()` of the result.

The system prompt instructs the parser to fill every field based on evidence from the investigation. Field descriptions on the `BaseModel` guide the parser toward correct extraction, making well-described fields important for accuracy.

Template kind classes accept only `int`, `float`, `str`, `bool`, `list` of those, and `Optional` forms with a default (`_rubric_kind_validation.py`), so a `Literal` field is rejected at trait construction. For a closed set of values, use a `str` field declared with `Field(json_schema_extra={"enum": [...]})` (so the enum reaches the extraction schema) plus a `field_validator`. A checkpoint reload rebuilds the class from its JSON Schema as a plain `str` field, without the enum or the validator.

The returned dict is stored in `agentic_trait_scores` with dot-notation keys: each field becomes `{trait_name}.{field_name}`. This flattening happens in Stage 11b's `_execute_individual` (or `_execute_shared`) after `evaluate_trait()` returns.

## 3. Strategy Dispatch

**File**: `src/karenina/benchmark/verification/stages/pipeline/agentic_rubric_evaluation.py`.

The stage's `execute()` method reads `context.agentic_rubric_strategy` (default: `"individual"`) and dispatches accordingly.

### Individual Strategy (`_execute_individual`)

Evaluates each trait with its own agent. For every `AgenticRubricTrait`:

1. Resolve the model via `_resolve_model()` (see Section 3.1)
2. Create an `AgenticTraitEvaluator` with the resolved model
3. Call `evaluator.evaluate_trait()` for the full investigation + extraction cycle
4. Collect the score, trace, serialized messages, extraction metadata, judge failure (if any) and early-stop reason into the stage outputs, and record the investigation and extraction usage (Section 10)

A trait whose resolved model cannot judge it is recorded with no score and no trace. In practice `_validate_agent_support` rejects such configurations before any trait runs (Section 3.1).

### Shared Strategy (`_execute_shared`)

Evaluates all traits with a single shared agent, then extracts per-trait scores:

1. Resolve models for all traits
2. Verify all valid models are identical (same `interface`, `model_provider`, `model_name`). If they differ, fall back to individual strategy automatically.
3. Build a combined investigation prompt listing all trait descriptions
4. Run one shared agent investigation
5. For each trait, call `evaluator.run_extraction()` against the shared trace

The shared agent receives the MCP servers of the shared model, built from its `mcp_urls_dict`. If the shared investigation fails (agent exception), the stage falls back to the individual strategy. Per-trait extraction failures within the shared strategy set `score=None` for that trait while preserving the shared trace and messages, and a retryable extraction error is recorded as a judge failure (Section 9). When the shared agent stopped at its turn limit or timeout, every trait extracted from its trace is flagged as incomplete.

### 3.1. Model Resolution (`_resolve_model`)

For each trait:

```
trait.model_override  (if set)  →  resolved model
        or
context.parsing_model (fallback) →  resolved model
```

Whether the resolved model can judge the trait is decided by `AgenticRubricTrait.accepts_judge_interface(interface)`, from the interface's registered `AdapterSpec`:

| Judge interface | Can judge |
|-----------------|-----------|
| `agent_tier="deep_agent"` (`claude_agent_sdk`, `langchain_deep_agents`) | Any agentic trait |
| `agent_tier="tool_loop"` with `supports_mcp=True` (`langchain`, `openrouter`, `openai_endpoint`, `claude_tool`, `google_genai`) | Traits that need no file tools: `context_mode="trace_only"` with `materialize_trace=False` (`trait.needs_file_tools` is `False`). The trace is inlined in the prompt and the judge's only tools are its MCP servers and any server-side tools |
| Anything else (`manual`, `taskeval`, unregistered interfaces) | No trait |

The rule is checked in three places with the same requirement text:

1. The `model_override` validator rejects a non-qualifying override when the trait is built.
2. `_validate_agent_support()` runs at the start of `execute()` and raises `ValueError` listing every trait whose resolved model (override or parsing model) cannot judge it, with the interface, tier, `context_mode` and `materialize_trace`. Silent skipping would hide configuration errors.
3. `_resolve_model()` returns `None` (the trait is skipped with a warning) for a model that cannot judge the trait.

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
| `AGENTIC_TRAIT_INVESTIGATION_TRACES` | `dict[str, str \| None]` | Trait name to investigation trace. A failed investigation stores its failure record (`[investigation failed: ...]`). `None` for a skipped trait. |
| `AGENTIC_TRAIT_INVESTIGATION_MESSAGES` | `dict[str, list[dict] \| None]` | Trait name to the judge's serialized trace messages, in the same format as `template.trace_messages` (each message's `to_dict()` plus `block_index`). Tool calls, tool results and thinking are readable by field. The single-call `LLMPort` path stores one assistant message. `None` when the investigation failed or the trait was skipped. |
| `AGENTIC_TRAIT_EXTRACTION_METADATA` | `dict[str, dict[str, str \| None]]` | Per-trait extraction provenance, keyed by base trait name. For template kind traits, `method` is `local_json` (parsed from a JSON block in the trace), `parser_after_local_json_failed` (the `ParserPort` succeeded after the local JSON attempt failed), or `failed` (both paths failed), and `local_json_error` and `parser_error` carry the corresponding error strings or `None`. Any trait scored from a partial investigation also gets `investigation_incomplete` (`"turn_limit"` or `"timeout"`). |
| `AGENTIC_TRAIT_EVALUATION_ERRORS` | `dict[str, dict[str, str]]` | Trait name to the error that left it without a judgment (Section 9). Empty when every judge produced a judgment. |
| `AGENTIC_INVESTIGATION_FAILED` | `bool` | Set when at least one investigation raised. Raises the `agentic_investigation_failed` caveat |
| `AGENTIC_EXTRACTION_FAILED` | `bool` | Set when at least one extraction failed with a retryable error. Raises the `agentic_extraction_failed` caveat |
| `AGENTIC_INVESTIGATION_INCOMPLETE` | `bool` | Set when at least one score was extracted from a partial investigation. Raises the `agentic_investigation_incomplete` caveat |

All of them are stored as both artifacts (for downstream stages) and result fields (for `FinalizeResult`), using `set_artifact_and_result()`. The stage also stores the usage tracker under `USAGE_TRACKER` after adding the judge's usage (Section 10).

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

Within `VerificationResultRubric`, agentic results occupy five dedicated fields:

| Field | Type | Source |
|-------|------|--------|
| `agentic_trait_scores` | `dict[str, int \| bool \| float \| str \| list \| None] \| None` | `AGENTIC_TRAIT_SCORES` result field |
| `agentic_trait_investigation_traces` | `dict[str, str] \| None` | `AGENTIC_TRAIT_INVESTIGATION_TRACES` result field, with `None` entries (skipped traits) dropped |
| `agentic_trait_investigation_messages` | `dict[str, list[dict[str, Any]]] \| None` | `AGENTIC_TRAIT_INVESTIGATION_MESSAGES` result field, with `None` entries (failed investigations, skipped traits) dropped |
| `agentic_trait_evaluation_errors` | `dict[str, dict[str, str]] \| None` | `AGENTIC_TRAIT_EVALUATION_ERRORS` result field, `None` when empty |
| `agentic_trait_extraction_metadata` | `dict[str, dict[str, str \| None]] \| None` | `AGENTIC_TRAIT_EXTRACTION_METADATA` result field (per-trait `method`, `local_json_error`, `parser_error`, `investigation_incomplete`) |

The three `AGENTIC_*_FAILED` and `AGENTIC_INVESTIGATION_INCOMPLETE` flags become caveats on `metadata.caveats` through `collect_caveats()`. Older result files without the new fields stay readable (the fields default to `None`).

### Rubric Mode Check

`run_single_model_verification()` does NOT upgrade `evaluation_mode` automatically. When a rubric with any traits (including agentic traits) is provided while `evaluation_mode="template_only"`, the runner logs a warning and skips rubric evaluation:

```python
if (_has_rubric_traits or _has_dynamic_rubric_traits) and evaluation_mode == "template_only":
    logger.warning(
        "Rubric traits were provided but evaluation_mode='template_only'. "
        "Rubric evaluation will be skipped. Set evaluation_mode='template_and_rubric' "
        "to evaluate rubric traits."
    )
```

To trigger Stage 11b with agentic traits, set `evaluation_mode="template_and_rubric"` (or `"rubric_only"`) explicitly. The stage itself runs whenever the mode includes rubrics and agentic traits are attached.

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
| **FinalizeResult** (13) | Reads agentic result fields and wires them into `VerificationResultRubric` (see above). Turns a judge failure into `metadata.failure` and the agentic flags into caveats (Section 9). |

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

## 9. Judge Failures and Partial Investigations

A judge that cannot judge must not look like a judgment. The stage therefore separates three outcomes: a failed investigation, a failed extraction, and an investigation that stopped early.

### Failed Judges

A trait counts as a **judge failure** when:

- its investigation raised (always), or
- its extraction raised an error that the context's `ErrorRegistry` classifies as retryable (`connection`, `timeout`, `rate_limit`, `server_error`). Wrapped causes are classified too, as for parser errors.

A non-retryable extraction error (for example a schema validation failure) is not a judge failure: the score is `None`, the trace is kept, and nothing else is recorded, as before.

For each judge failure the stage stores an entry in `agentic_trait_evaluation_errors`:

```python
result.rubric.agentic_trait_evaluation_errors
# {
#     "citations": {
#         "phase": "extraction",          # or "investigation"
#         "error_type": "GeminiRateLimitError",
#         "message": "Gemini API error 429 RESOURCE_EXHAUSTED: ...",
#         "category": "rate_limit",       # ErrorCategory value, "permanent" is never worth retrying
#     }
# }
```

When any trait failed, the stage then:

1. Sets `AGENTIC_INVESTIGATION_FAILED` and/or `AGENTIC_EXTRACTION_FAILED`. `collect_caveats()` turns them into the `agentic_investigation_failed` and `agentic_extraction_failed` [caveats](../reference/api/failure-and-caveats.md#5-caveat-7-informational-flags). The caveats are set whatever failure category the row ends with.
2. Calls `context.mark_error()` with stage `AgenticRubricEvaluation` and the first failure's error category. The [failure classifier](../reference/api/failure-and-caveats.md#6-classifier-priority-8-rules) then reports a retry-exhausted failure (group `retry`) when that category used up its retry budget, otherwise an `UNEXPECTED_ERROR` failure (group `system`), both at stage `AgenticRubricEvaluation`. Earlier rules (auto-fail stages, a content failure, abstention, sufficiency) still win. In every case the row is a failure, even when its template verification passed.

Other traits on the same row are still evaluated and keep their scores.

### Partial Investigations

When the judge's agent hits its turn limit or wall-clock timeout, the adapter returns a partial trace instead of raising, and the score is still extracted from it. For a template kind trait the parser can then fill in a verdict for an investigation that never finished, so the stage flags it:

- `agentic_trait_extraction_metadata[trait]["investigation_incomplete"]` is `"turn_limit"` or `"timeout"`,
- the row gets the informational `agentic_investigation_incomplete` caveat.

The score is kept, no error is marked, and progressive-save resume does not re-run these rows. Filter on the caveat or the metadata entry when partial investigations should not count.

### Resume After a Judge Failure

With [progressive save](../reference/api/sinks.md), both `ProgressiveFileSink` and `AgenticProgressiveFileSink` re-run a QA row on resume when its answer is usable and either `agentic_trait_evaluation_errors` holds an error whose category is not `permanent`, or the row failed at `AgenticRubricEvaluation` with a transient failure category (`connection`, `timeout`, `rate_limit`, `server_error`). The executor reuses the stored answer trace, so the answering model is not called again. Permanent errors (for example a wrong model name) are never re-run, because they would fail again on every resume. Scenario rows are not re-run this way, because their unit of work is the whole scenario.

### Judge Failures in `extend_rubric`

[`extend_rubric`](../core_concepts/extending-runs.md) merges the rubric-only run's trait buckets onto a copy of each prior row, including `agentic_trait_investigation_messages`, `agentic_trait_evaluation_errors` and `agentic_trait_extraction_metadata`. A judge that failed while re-judging stays visible: the rubric-only row's agentic caveats (`agentic_investigation_failed`, `agentic_extraction_failed`, `agentic_investigation_incomplete`) are added to the merged row, and the rubric-only row's `Failure` is copied when it comes from `AgenticRubricEvaluation` and the prior row has no failure of its own. A prior failure (for example a content failure) is kept, and the caveats and `agentic_trait_evaluation_errors` still identify the judge failure.

## 10. Judge Usage

The judge's token usage is recorded in the context's usage tracker under the stage key `agentic_rubric_evaluation`, for both strategies:

- one entry per investigation (the agent run's summed usage, or the single `LLMPort` call's usage),
- one entry per extraction parser call.

Each entry is keyed by the judge's model string (`ModelIdentity.from_model_config(model, role="parsing").display_string`) and carries the same keys as the answering model's entries, including `cost_usd` and cache counts when reported, plus `reasoning_tokens` and `web_search_queries` when the adapter set them. Usage that reports no tokens and no web searches is skipped. The summary reaches the result as `template.usage_metadata["agentic_rubric_evaluation"]` and is included in `total`. See [Token Usage](../workflows/analyzing-results/verification-result.md#token-usage).

## 11. Key File Reference

| Domain | File (relative to `karenina/src/karenina/`) |
|--------|------|
| Stage implementation (Stage 11b) | `benchmark/verification/stages/pipeline/agentic_rubric_evaluation.py` |
| Evaluator (investigation + extraction) | `benchmark/verification/evaluators/rubric/agentic_trait.py` |
| AgenticRubricTrait schema | `schemas/entities/rubric.py` |
| ArtifactKeys (agentic rubric section) | `benchmark/verification/stages/core/base.py` |
| VerificationContext (agentic rubric fields) | `benchmark/verification/stages/core/base.py` |
| Stage orchestrator (Stage 11b registration) | `benchmark/verification/stages/core/orchestrator.py` |
| Pipeline runner (rubric-mode check, config threading) | `benchmark/verification/runner.py` |
| FinalizeResult (rubric result assembly) | `benchmark/verification/stages/pipeline/finalize_result.py` |
| VerificationResultRubric (agentic fields) | `schemas/verification/result_components.py` |
| Agentic caveats | `schemas/results/caveat.py`, `benchmark/verification/failure_classifier.py` |
| Resume of retryable judge failures | `benchmark/verification/sinks.py` (`_is_retryable_judge_failure`) |
| `extend_rubric` merge of judge failures | `benchmark/verification/extension.py` |
| Extraction output schemas | `schemas/outputs/rubric.py` |
| Judge interface rule (`accepts_judge_interface`) | `schemas/entities/rubric.py` |
| Adapter registry (agent tier and MCP support) | `adapters/registry.py` |

## 12. Next Steps

- [Agentic Traits](../core_concepts/rubrics/agentic-traits.md): conceptual overview and usage guide
- [Agentic Evaluation](agentic-evaluation.md): Stage 7b internals (agentic template parsing)
- [Verification Pipeline](../core_concepts/verification-pipeline.md): the verification pipeline execution engine (13 stages with sub-stages 7a/7b and 11a/11b plus the always-on PlaceholderRetryAutoFail guard)
- [Rubrics](../core_concepts/rubrics/index.md): rubric architecture and trait types
- [Deep Judgment Rubrics](deep-judgment-rubrics.md): deep judgment for standard rubric traits
