# Agentic Evaluation

This page covers the internal machinery of agentic evaluation: how the pipeline detects agentic adapters, resolves workspaces, runs the two-step investigation/extraction parse stage, and cleans up afterward. It is for contributors and power users who need to understand what happens under the hood.

For a conceptual overview and usage guide, see the [agentic evaluation concept page](../core_concepts/agentic-evaluation.md). For the general pipeline architecture, see [verification-pipeline.md](../core_concepts/verification-pipeline.md).

## 1. AdapterSpec.agent_tier

**File**: `src/karenina/adapters/registry.py`, `AdapterSpec` dataclass.

```python
agent_tier: str = "tool_loop"
```

This field distinguishes runtimes that are themselves agents with built-in tools (e.g., Claude Code, `agent_tier="deep_agent"`) from scaffolded adapters where the adapter explicitly orchestrates each tool call turn (LangChain, Claude Tool, `agent_tier="tool_loop"`).

The `GenerateAnswer` stage (stage 2) checks this field to decide whether to use `AgentPort` or `LLMPort` for the answering step:

```python
# generate_answer.py, Step 2
spec = AdapterRegistry.get_spec(answering_model.interface)
use_agent = bool(answering_model.mcp_urls_dict) or (spec is not None and spec.agent_tier == "deep_agent")
```

When `use_agent=True`, the stage calls `AgentPort.run()`, which captures the full conversation trace: tool calls, tool results, and intermediate reasoning. When `False`, it calls `LLMPort.invoke()`, which returns only the final text response.

| Interface | `agent_tier` | Reason |
|-----------|:------------------:|--------|
| `claude_agent_sdk` | `"deep_agent"` | The Claude CLI binary is itself an agent; the LLMPort path would lose tool call traces |
| `langchain` | `"tool_loop"` | The adapter orchestrates tool calls explicitly |
| `claude_tool` | `"tool_loop"` | Same: adapter-orchestrated |
| `manual` | `"tool_loop"` | Pre-recorded traces; no live agent |
| `openai_endpoint` | `"tool_loop"` | Routes to `langchain` |
| `openrouter` | `"tool_loop"` | Routes to `langchain` |

See [Adapters](../core_concepts/adapters.md) for the full adapter reference.

## 2. Workspace Resolution (GenerateAnswer Stage)

**File**: `src/karenina/benchmark/verification/stages/pipeline/generate_answer.py`, method `_resolve_workspace()`.

When `agentic_parsing=True`, the `GenerateAnswer` stage resolves a workspace directory before invoking the answering agent. This workspace is the working directory the agent operates in.

### Resolution Logic

```
workspace_root (from Benchmark)
  + question_workspace_path (from Question.workspace_path)
  = source directory

If workspace_copy=True:
  source is copied to: workspace_root / {workspace_path}_run_{timestamp}_pid{pid}
  context.workspace_path = the copy (safe to modify)
  context.workspace_is_copy = True

If workspace_copy=False:
  context.workspace_path = source directly (destructive)
  context.workspace_is_copy = False

If question_workspace_path is None:
  An empty directory is created: workspace_root / {question_id}_run_{timestamp}_pid{pid}
  context.workspace_path = new directory
  context.workspace_is_copy = True
```

The unique suffix (`run_{timestamp}_pid{pid}`) includes the replicate number when present, ensuring parallel verification runs do not collide:

```python
suffix = f"run_{timestamp}_pid{os.getpid()}"
if context.replicate is not None:
    suffix += f"_rep{context.replicate}"
```

The resolved `workspace_path` is passed to `AgentConfig.workspace_path`, which the Claude SDK adapter wires to `ClaudeAgentOptions.cwd`. This makes the agent's file system operations (Read, Bash, etc.) operate relative to the resolved workspace.

### Preconditions

- `context.workspace_root` must be set when `agentic_parsing=True`. If it is `None`, the stage raises `RuntimeError`.
- If `question_workspace_path` points to a directory that does not exist under `workspace_root`, the stage raises `RuntimeError`.

## 3. AgenticParseTemplateStage (Stage 7b)

**File**: `src/karenina/benchmark/verification/stages/pipeline/agentic_parse_template.py`.

This stage replaces `ParseTemplateStage` (stage 7a) when `agentic_parsing=True` in `VerificationConfig`. The selection happens in `StageOrchestrator.from_config()`:

```python
# orchestrator.py
if agentic_parsing and agentic_parsing_trigger == "dynamic":
    stages.append(DynamicParseTemplateStage())
elif agentic_parsing:
    stages.append(AgenticParseTemplateStage())
else:
    stages.append(ParseTemplateStage())
```

Only one template parsing stage is ever present in a pipeline run; the classical, always-agentic, and dynamic-agentic stages are mutually exclusive.

When `agentic_parsing=True` and `agentic_parsing_trigger="dynamic"`, the orchestrator uses `DynamicParseTemplateStage` in the same template parsing slot. Dynamic parsing first attempts a direct final-message parse, then escalates to the same investigation/extraction helpers used by `AgenticParseTemplateStage` when needed.

### AgenticParseTemplateStage Artifact Contract

| Direction | Artifact Keys |
|-----------|--------------|
| **Requires** | `RAW_ANSWER`, `ANSWER`, `RAW_LLM_RESPONSE` |
| **Produces** | `PARSED_ANSWER`, `PARSING_MODEL_STR`, `INVESTIGATION_TRACE`, `AGENTIC_PARSING_PERFORMED` |

`DynamicParseTemplateStage` shares the same required inputs and final `PARSED_ANSWER` output. It also records `DYNAMIC_PARSING_PERFORMED`, `DYNAMIC_PARSE_DECISION`, and `DYNAMIC_DECISION_REASONING`; when it escalates, it records the same `INVESTIGATION_TRACE` and `AGENTIC_PARSING_PERFORMED` fields as `AgenticParseTemplateStage`.

### should_run() Conditions

The stage skips itself when any of the following are true:

- A prior stage set `context.error`
- `agentic_parsing` is `False`
- `recursion_limit_reached` is `True`
- Trace validation failed
- Abstention was detected
- Sufficiency was detected as insufficient

### Step 1: Investigation

Calls `AgentPort.run()` (via `get_agent(context.parsing_model)`) with:

- **System prompt**: instructs the agent to independently verify the answering agent's work, with the JSON schema of the answer template (from `build_parsing_schema()`) embedded as the target structure
- **User prompt**: built from the question text plus context controlled by `agentic_judge_context`
- **Workspace path**: for tool access (Read, Bash, etc.)
- **Agent config**: `max_turns` and `timeout` from `agentic_parsing_max_turns` and `agentic_parsing_timeout`

The `agentic_judge_context` field controls what context the investigation agent receives:

| Mode | Agent sees | Use case |
|------|-----------|----------|
| `workspace_only` | Only workspace path (agent must discover everything independently) | Strictest evaluation; agent cannot be influenced by the answering trace |
| `trace_and_workspace` | Answering trace + workspace path | Balanced; agent can review the answering agent's reasoning and verify artifacts |
| `trace_only` | Only answering trace (equivalent to classical stage 7a) | No workspace access; useful when workspace is not relevant |

The return value is the raw text of the investigation agent's conversation trace.

### Step 2: Extraction

Calls `ParserPort.parse_to_pydantic()` (via `get_parser(context.parsing_model)`) with:

- **System prompt**: instructs the parser to extract structured data from the investigation report
- **Input**: the investigation trace from Step 1
- **Target schema**: the answer template class (same Pydantic `BaseAnswer` subclass that classical stage 7a would use)

The return value is a parsed answer instance, identical in type to what `ParseTemplateStage` would produce. All downstream stages (VerifyTemplate, EmbeddingCheck, etc.) work identically regardless of which parse stage ran.

### Dynamic Agentic Parsing

Classic agentic template parsing runs the investigation step for every item. Dynamic mode adds a cheaper screening step first: it asks the parsing model to inspect only the final AI message and return exactly one JSON object. If the final message contains enough information to populate the relaxed template schema, the decision response is used directly.

Sufficient final-message responses must use this shape:

```json
{"reasoning": "why the final message supports the fields", "sufficient": true, "answer": {"field": "value"}}
```

Insufficient final-message responses must use this shape:

```json
{"reasoning": "what is missing and where it likely lives", "sufficient": false}
```

Enable dynamic parsing with both fields:

```json
{
  "agentic_parsing": true,
  "agentic_parsing_trigger": "dynamic"
}
```

`agentic_parsing_trigger` accepts `"always"` and `"dynamic"`. `"always"` is the default and preserves existing agentic parsing behavior: the investigation/extraction flow runs for every item. `"dynamic"` runs the decision call first and escalates to the same investigation/extraction flow when the final message is insufficient, malformed, or cannot validate against the relaxed template schema.

Dynamic parsing still requires a deep-agent parsing interface such as `claude_agent_sdk` or `langchain_deep_agents`. The decision call itself uses the plain LLM port. On `claude_agent_sdk`, that plain LLM port may still spawn a Claude CLI subprocess per item, so the cost win comes from skipping the investigation/extraction session for final messages that are already sufficient, not from making the first call free.

Temperature handling is best effort. Interfaces that honor `ModelConfig.temperature` should apply it to the decision call; the `claude_agent_sdk` plain LLM port currently ignores temperature, so dynamic parsing does not promise cross-interface determinism.

Dynamic recovery works best with `agentic_judge_context="workspace_only"` or `"trace_and_workspace"`. With `agentic_judge_context="trace_only"`, escalation re-reads the trace context after the final message was already judged insufficient, so it cannot recover by inspecting workspace artifacts.

Dynamic parsing adds result fields on `VerificationResultTemplate` and flattened storage columns:

| Result field | Storage column |
|--------------|----------------|
| `dynamic_parsing_performed` | `template_dynamic_parsing_performed` |
| `dynamic_parse_decision` | `template_dynamic_parse_decision` |
| `dynamic_decision_reasoning` | `template_dynamic_decision_reasoning` |

Karenina creates result tables with `CREATE TABLE IF NOT EXISTS` and does not alter existing result tables. To write dynamic results to persistent storage, use a fresh database or manually add the three columns before writing dynamic results to an existing database.

Template replay and extension runs have two limitations:

- On `extend_template` and other replayed template-parsing runs, answer generation is replayed before workspace resolution, so dynamic escalation may run without a workspace.
- The decision call is judge-side and is not captured in replay entries, so replayed template-parsing runs regenerate the direct-versus-escalated decision. `extend_rubric` uses `rubric_only` and skips template parsing, so dynamic parsing does not run there.

### Result Storage

The stage stores four artifacts and two result builder fields:

```python
context.set_artifact(ArtifactKeys.PARSED_ANSWER, parsed_answer)
context.set_artifact(ArtifactKeys.PARSING_MODEL_STR, model_str)
context.set_artifact(ArtifactKeys.INVESTIGATION_TRACE, investigation_trace)
context.set_artifact(ArtifactKeys.AGENTIC_PARSING_PERFORMED, True)

context.set_result_field(ArtifactKeys.INVESTIGATION_TRACE, investigation_trace)
context.set_result_field(ArtifactKeys.AGENTIC_PARSING_PERFORMED, True)
```

The stage also sets `TEMPLATE_EVALUATOR` to `None` and `DEEP_JUDGMENT_PERFORMED` to `False`, since agentic parsing does not use the classical template evaluator or deep judgment extraction.

## 4. Ground Truth Stripping in BaseAnswer.model_json_schema()

**File**: `src/karenina/schemas/entities/answer.py`.

`VerifiedField` stores ground truth and verification metadata in `json_schema_extra["__verification__"]`. This metadata must never reach the judge LLM, as it would leak correct answers.

`BaseAnswer` overrides `model_json_schema()` to recursively strip all `__verification__` keys from the generated JSON schema:

```python
@classmethod
def model_json_schema(cls, *args, **kwargs):
    schema = super().model_json_schema(*args, **kwargs)

    def _strip_verification(obj):
        if isinstance(obj, dict):
            obj.pop("__verification__", None)
            for value in obj.values():
                _strip_verification(value)
        elif isinstance(obj, list):
            for item in obj:
                _strip_verification(item)

    _strip_verification(schema)
    return schema
```

This protects all code paths that generate JSON schemas: the Claude SDK parser adapter, the agentic investigation stage, and `build_parsing_schema()`. The stripping happens at the source rather than per-adapter, so any new code path that calls `model_json_schema()` is automatically protected.

Extraction hints (field descriptions used in [prompt assembly](prompt-assembly.md)) flow through Pydantic `FieldInfo` objects, not through `model_json_schema()`, so they are unaffected by this stripping.

## 5. Workspace Cleanup (FinalizeResult Stage)

**File**: `src/karenina/benchmark/verification/stages/pipeline/finalize_result.py`.

`FinalizeResult` (stage 13) handles workspace cleanup after all other stages have completed. It applies triple-guard protection before deleting any directory:

1. `context.workspace_path` is set (a workspace was resolved by `GenerateAnswer`)
2. `context.workspace_cleanup` is `True` (the `workspace_cleanup` setting from `VerificationConfig`)
3. `context.workspace_is_copy` is `True` (the directory is a working copy, not an original)

```python
if context.workspace_path and context.workspace_cleanup and context.workspace_is_copy:
    try:
        shutil.rmtree(context.workspace_path)
    except Exception:
        logger.warning("Failed to clean up workspace: %s", context.workspace_path, exc_info=True)
```

Only working copies created by `workspace_copy=True` (or freshly created empty directories) are eligible for cleanup. Original workspace directories are never deleted. Cleanup failures are logged as warnings but do not affect the pipeline result.

## 6. VerificationResultTemplate Extensions

**File**: `src/karenina/schemas/verification/result_components.py`.

Two fields on `VerificationResultTemplate` carry agentic parsing data into the result:

| Field | Type | Set by |
|-------|------|--------|
| `investigation_trace` | `str \| None` | `AgenticParseTemplateStage` via `context.set_result_field()` |
| `agentic_parsing_performed` | `bool` | `AgenticParseTemplateStage` via `context.set_result_field()` |

These are wired into the `VerificationResultTemplate` constructor by `FinalizeResult`:

```python
template = VerificationResultTemplate(
    ...
    investigation_trace=context.get_result_field(ArtifactKeys.INVESTIGATION_TRACE),
    agentic_parsing_performed=context.get_result_field(ArtifactKeys.AGENTIC_PARSING_PERFORMED, False),
    ...
)
```

When agentic parsing was not used, `investigation_trace` is `None` and `agentic_parsing_performed` is `False`.

## 7. ArtifactKeys for Agentic Parsing

**File**: `src/karenina/benchmark/verification/stages/core/base.py`, class `ArtifactKeys`.

Three constants in the "Agentic Parsing" section:

```python
INVESTIGATION_TRACE = "investigation_trace"
WORKSPACE_PATH = "workspace_path"
AGENTIC_PARSING_PERFORMED = "agentic_parsing_performed"
```

`INVESTIGATION_TRACE` and `AGENTIC_PARSING_PERFORMED` are used as both artifact keys and result field keys. `WORKSPACE_PATH` is used only as an artifact key (the workspace path is not persisted in the result).

## 8. Pipeline Threading

The agentic configuration flows from the `Benchmark` facade through the batch runner and into each individual pipeline context. The chain:

```
Benchmark.workspace_root
  -> Benchmark.run_verification(config, ...)
    -> run_verification_batch(workspace_root=self._workspace_root, ...)
      -> generate_task_queue(workspace_root=..., ...)
        -> task dict["workspace_root"]  (overrides config value)
        -> task dict includes extract_feature_flags(config):
             agentic_parsing, agentic_parsing_trigger, agentic_judge_context,
             agentic_parsing_max_turns, agentic_parsing_timeout,
             workspace_copy, workspace_cleanup
      -> _run_single_task(task)
        -> run_single_model_verification(workspace_root=..., ...)
          -> VerificationContext(workspace_root=..., agentic_parsing=..., ...)
            -> GenerateAnswer._resolve_workspace()
```

`VerificationConfig` fields (`agentic_parsing`, `agentic_parsing_trigger`, `workspace_copy`, `workspace_cleanup`, `agentic_judge_context`, `agentic_parsing_max_turns`, `agentic_parsing_timeout`) flow via `extract_feature_flags(config)` into each task dict. The `workspace_root` is provided separately by the `Benchmark` facade and overrides any value in the config at the task queue generation step.

### VerificationContext Fields

The following `VerificationContext` fields (in `stages/core/base.py`) control agentic evaluation at runtime:

| Field | Type | Default | Source |
|-------|------|---------|--------|
| `agentic_parsing` | `bool` | `False` | `VerificationConfig` via `extract_feature_flags` |
| `agentic_parsing_trigger` | `str` | `"always"` | `VerificationConfig` via `extract_feature_flags` |
| `agentic_judge_context` | `str` | `"workspace_only"` | `VerificationConfig` via `extract_feature_flags` |
| `agentic_parsing_max_turns` | `int` | `15` | `VerificationConfig` via `extract_feature_flags` |
| `agentic_parsing_timeout` | `float` | `120.0` | `VerificationConfig` via `extract_feature_flags` |
| `question_workspace_path` | `str \| None` | `None` | `Question.workspace_path` via task dict |
| `workspace_path` | `Path \| None` | `None` | Set by `GenerateAnswer._resolve_workspace()` |
| `workspace_is_copy` | `bool` | `False` | Set by `GenerateAnswer._resolve_workspace()` |
| `workspace_root` | `Path \| None` | `None` | `Benchmark.workspace_root` via task dict |
| `workspace_copy` | `bool` | `True` | `VerificationConfig` via `extract_feature_flags` |
| `workspace_cleanup` | `bool` | `True` | `VerificationConfig` via `extract_feature_flags` |

## 9. Interaction with Other Pipeline Stages

Agentic parsing affects several other stages:

| Stage | Interaction |
|-------|------------|
| **ValidateTemplate** (1) | Unaffected. Runs identically; produces the `ANSWER` and `RAW_ANSWER` artifacts that Stage 7b requires. |
| **GenerateAnswer** (2) | Resolves workspace when `agentic_parsing=True`. Also uses `AgentPort` when `agent_tier=="deep_agent"`. |
| **RecursionLimitAutoFail** (3) | If the answering agent hit the recursion limit, agentic template parsing skips itself. |
| **AbstentionCheck** (5) | If abstention is detected, agentic template parsing skips itself. |
| **SufficiencyCheck** (6) | If the response is insufficient, agentic template parsing skips itself. |
| **VerifyTemplate** (8) | Unaffected. Receives the same `PARSED_ANSWER` artifact regardless of whether Stage 7a, Stage 7b, or dynamic parsing produced it. |
| **EmbeddingCheck** (9) | Unaffected. Checks `field_verification_result` produced by Stage 8. |
| **DeepJudgmentAutoFail** (10) | Skips itself when agentic parsing was used, because agentic template parsing sets `DEEP_JUDGMENT_PERFORMED` to `False`. |
| **FinalizeResult** (13) | Reads agentic parsing result fields such as `INVESTIGATION_TRACE`, `AGENTIC_PARSING_PERFORMED`, and dynamic parsing decision metadata from the result builder. Handles workspace cleanup. |

## 10. Key File Reference

| Domain | File (relative to `karenina/src/karenina/`) |
|--------|------|
| AdapterSpec with `agent_tier` | `adapters/registry.py` |
| Claude SDK registration (sets `agent_tier="deep_agent"`) | `adapters/claude_agent_sdk/registration.py` |
| Workspace resolution and answer generation | `benchmark/verification/stages/pipeline/generate_answer.py` |
| Agentic parse stage (Stage 7b) | `benchmark/verification/stages/pipeline/agentic_parse_template.py` |
| Classical parse stage (Stage 7a) | `benchmark/verification/stages/pipeline/parse_template.py` |
| Stage orchestrator (selects classical, always-agentic, or dynamic-agentic parsing) | `benchmark/verification/stages/core/orchestrator.py` |
| ArtifactKeys and VerificationContext | `benchmark/verification/stages/core/base.py` |
| Ground truth stripping | `schemas/entities/answer.py` |
| Result components (`investigation_trace` and dynamic parsing fields) | `schemas/verification/result_components.py` |
| Workspace cleanup | `benchmark/verification/stages/pipeline/finalize_result.py` |
| Feature flag extraction | `benchmark/verification/utils/task_helpers.py` |
| JSON schema builder | `benchmark/verification/utils/schema_builder.py` |
| Pipeline runner | `benchmark/verification/runner.py` |
| Batch runner | `benchmark/verification/batch_runner.py` |
| Benchmark facade | `benchmark/benchmark.py` |

## 11. Next Steps

- [Verification Pipeline](../core_concepts/verification-pipeline.md): the 13-stage execution engine
- [Adapters](../core_concepts/adapters.md): port/adapter architecture and available interfaces
- [Answer Templates](../core_concepts/answer-templates.md): writing templates and VerifiedField
- [Prompt Assembly](prompt-assembly.md): how prompts are constructed for each LLM call
