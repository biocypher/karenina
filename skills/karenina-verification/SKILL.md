---
name: karenina-verification
description: >
  Configure and run karenina's verification pipeline: model selection, adapter
  choice, guard configuration, replicates, presets, prompt assembly, deep
  judgment, MCP integration, pipeline execution, and initial result inspection.
  Use when setting up VerificationConfig, ModelConfig, choosing adapters,
  debugging auto-fails from guards, creating presets, running run_verification(),
  or understanding pipeline output.
---

# Karenina Verification Pipeline Configuration

This skill covers how to configure and operate karenina's verification pipeline. It is organized by user task: what you want to accomplish, with the config fields and code you need.

**Source of truth for all fields**: `karenina/src/karenina/schemas/verification/config.py` (VerificationConfig) and `karenina/src/karenina/schemas/config/models.py` (ModelConfig).

**Reference files in this skill**:
- [references/config-fields.md](references/config-fields.md): Complete field-by-field tables for VerificationConfig and ModelConfig
- [references/adapter-selection.md](references/adapter-selection.md): Decision table for choosing an adapter
- [references/guards.md](references/guards.md): Guard stages, triggers, auto-fail behavior, and configuration

**Full documentation**: The `using-karenina` skill contains the complete karenina docs in its `references/` directory. Consult it for API details not covered here (evaluation modes, result structure, prompt assembly, etc.).

---

## 1. Configure Models

VerificationConfig uses two model lists:

- `answering_models`: list of ModelConfig for generating answers (the "answering LLM")
- `parsing_models`: list of ModelConfig for parsing and judging responses (the "judge LLM")

Each ModelConfig requires at minimum `id`, `model_name`, and (for most interfaces) `model_provider`.

```python
from karenina.schemas.config import ModelConfig
from karenina.schemas.verification import VerificationConfig

config = VerificationConfig(
    answering_models=[
        ModelConfig(
            id="answerer",
            model_provider="anthropic",
            model_name="claude-sonnet-4-6",
            temperature=0.1,
            max_tokens=16384,
        )
    ],
    parsing_models=[
        ModelConfig(
            id="judge",
            model_provider="anthropic",
            model_name="claude-haiku-4-5",
            temperature=0.0,
        )
    ],
)
```

**Key ModelConfig fields** (see [references/config-fields.md](references/config-fields.md) for all fields):

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `id` | `str \| None` | None | Identifier for this model (required for non-manual interfaces) |
| `model_provider` | `str \| None` | None | Provider: `"anthropic"`, `"openai"`, `"google"`, `"litellm"` |
| `model_name` | `str \| None` | None | Model name (e.g., `"claude-sonnet-4-6"`, `"gpt-4.1"`) |
| `temperature` | `float` | 0.1 | Sampling temperature |
| `max_tokens` | `int` | 16384 | Maximum response tokens |
| `interface` | `str` | `"langchain"` | Adapter interface (see Task 2) |
| `system_prompt` | `str \| None` | None | Custom system prompt (answering models get a default when omitted) |

If you omit `system_prompt`, VerificationConfig applies a default to **answering models only**: `"You are an expert assistant. Answer the question accurately and concisely."`. Parsing models get no default prompt — `system_prompt` stays `None` — so set it explicitly on the parsing ModelConfig if you want the judge to run with one. The exception is interactive CLI mode (`karenina init` / `--interactive`), which pre-fills the parsing prompt from `DEFAULT_PARSING_SYSTEM_PROMPT` (`"You are a validation assistant..."`) on your behalf.

**Complete ModelConfig fields** (all have `extra="forbid"`, so passing unknown fields raises `ValidationError`):

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `id` | `str \| None` | `None` | Required for non-manual interfaces |
| `model_provider` | `str \| None` | `None` | `"anthropic"`, `"openai"`, `"google"`, `"litellm"` |
| `model_name` | `str \| None` | `None` | Required for non-manual interfaces |
| `temperature` | `float` | `0.1` | |
| `max_tokens` | `int` | `16384` | |
| `interface` | `str` | `"langchain"` | See adapter table below |
| `system_prompt` | `str \| None` | `None` | |
| `max_retries` | `int` | `2` | |
| `mcp_urls_dict` | `dict[str, str] \| None` | `None` | MCP server URLs |
| `mcp_tool_filter` | `list[str] \| None` | `None` | |
| `mcp_tool_description_overrides` | `dict[str, str] \| None` | `None` | |
| `mcp_http_timeout` | `float \| None` | `None` | MCP streamable-HTTP timeout (None = SDK default 30s) |
| `mcp_sse_read_timeout` | `float \| None` | `None` | MCP SSE read timeout (None = SDK default 300s) |
| `endpoint_base_url` | `str \| None` | `None` | For `openai_endpoint` |
| `endpoint_api_key` | `SecretStr \| None` | `None` | For `openai_endpoint` |
| `anthropic_base_url` | `str \| None` | `None` | For claude_tool/claude_agent_sdk |
| `anthropic_api_key` | `SecretStr \| None` | `None` | Override env var |
| `agent_runtime` | `AgentRuntimeConfig \| None` | `None` | Typed runtime settings (backend/sandbox/container) |
| `extra_kwargs` | `dict[str, Any] \| None` | `None` | Vendor-specific params |
| `manual_traces` | `ManualTraces \| None` | `None` | Required when `interface="manual"` |
| `agent_middleware` | `AgentMiddlewareConfig \| None` | `None` | For MCP-enabled agents |
| `max_context_tokens` | `int \| None` | `None` | Summarization threshold |
| `agent_timeout` | `int \| None` | `None` | Agent execution timeout (default 180s) |
| `request_timeout` | `float \| None` | `None` | HTTP request timeout for individual LLM calls (None = provider SDK default) |
| `retry_policy` | `RetryPolicy \| None` | `None` | Per-category retry policy; None uses pipeline-level `RetryPolicy` |

**Fields that do NOT exist on ModelConfig**: `top_p`, `top_k`, `frequency_penalty`, `presence_penalty`, `stop`, `name`, `provider`. Passing any of these raises `ValidationError`.

### AgentRuntimeConfig (nested on `ModelConfig.agent_runtime`)

Typed execution-runtime settings for agent adapters (Claude Agent SDK, LangChain Deep Agents). Backend-specific validation is left to the adapter.

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `backend` | `Literal["native","filesystem","container","docker","local_shell"] \| None` | `None` | Execution backend; None lets the adapter choose |
| `access_mode` | `Literal["read_write","read_only"]` | `"read_write"` | Filesystem access mode |
| `sandbox_enabled` | `bool` | `True` | Enable sandboxing |
| `container_runtime` | `Literal["docker","singularity","apptainer"]` | `"docker"` | Container runtime |
| `container_image` | `str \| None` | `None` | Container image name |
| `container_network` | `Literal["bridge","none"]` | `"bridge"` | Container network mode |
| `container_add_hosts` | `tuple[str, ...]` | `()` | Extra container host mappings |
| `read_max_bytes` | `int \| None` | `None` | Max bytes readable from filesystem (None = unlimited) |

### ModelIdentity (read-only, on results)

`ModelIdentity` appears on `result.metadata.answering` and `result.metadata.parsing`. It is NOT constructed directly; it is created internally by the pipeline via `ModelIdentity.from_model_config()`. Its fields:

| Field | Type | Notes |
|-------|------|-------|
| `interface` | `str` | Required |
| `model_name` | `str` | Required |
| `tools` | `list[str]` | MCP tool names (empty for parsing models) |

Import: `from karenina.schemas.verification.model_identity import ModelIdentity`

---

## 2. Choose an Adapter

The `interface` field on ModelConfig selects which adapter handles LLM communication. The default (`"langchain"`) works for most cases.

| Need | `interface=` | Why |
|------|-------------|-----|
| General-purpose (Anthropic, OpenAI, Google) | `"langchain"` (default) | Broadest provider compatibility, MCP support |
| OpenRouter models | `"openrouter"` | Routes through LangChain with OpenRouter config |
| OpenAI-compatible endpoint | `"openai_endpoint"` | Custom endpoints; requires `endpoint_base_url` and `endpoint_api_key` |
| Native Anthropic SDK | `"claude_agent_sdk"` | Direct Anthropic API; falls back to LangChain if unavailable |
| Anthropic tool-use evaluation | `"claude_tool"` | Native tool schemas via Anthropic SDK; falls back to LangChain |
| Deep agent evaluation | `"langchain_deep_agents"` | Autonomous task execution agents; no fallback |
| Offline/pre-recorded traces | `"manual"` | No live LLM calls; requires `manual_traces` |

**Decision rule**: Use `"langchain"` unless you have a specific reason not to. Switch to `"claude_tool"` for tool-use benchmarks, `"claude_agent_sdk"` for native Anthropic features, or `"manual"` for evaluating pre-recorded outputs.

See [references/adapter-selection.md](references/adapter-selection.md) for the full decision table with MCP support and provider requirements.

---

## 3. Understand and Configure Guards

Guards are pipeline stages that auto-fail a question BEFORE parsing. When a guard triggers, no template parsing or rubric evaluation runs for that question. This saves LLM calls but means you get no structured output for that question.

The four guards, in pipeline order:

| Guard | Stage Name | Default | Configurable? |
|-------|-----------|---------|---------------|
| Recursion limit | `RecursionLimitAutoFail` | Always on | Threshold via agent config |
| Trace validation | `TraceValidationAutoFail` | Always on (MCP only) | Not disablable |
| Abstention check | `AbstentionCheck` | Off | `abstention_enabled=True` to enable |
| Sufficiency check | `SufficiencyCheck` | Off | `sufficiency_enabled=True` to enable |

**Debugging guard auto-fails**: If a question unexpectedly shows `verify_result=False` with no parsed template, a guard likely fired. Check `result.template.recursion_limit_reached`, `result.template.abstention_detected`, and `result.metadata.failure` (`.category` / `.stage` / `.reason`) to identify which guard triggered.

Note the third `verify_result` state: `False` means a guard or the verification stage explicitly recorded a failure, but a **stage error** (e.g. an auth or connection failure at `GenerateAnswer`) leaves `result.template.verify_result` as `None` — no verdict was ever produced — with `result.metadata.failure` carrying the failing stage name in `.stage` and a `category` that reflects the classified error type (`connection`, `rate_limit`, `server_error`, `timeout`; `unexpected_error` is only the fallback when nothing else matches).

```python
# Enable both optional guards
config = VerificationConfig(
    answering_models=[...],
    parsing_models=[...],
    abstention_enabled=True,
    sufficiency_enabled=True,
)

# Disable guards when you want parsing to proceed regardless
config = VerificationConfig(
    answering_models=[...],
    parsing_models=[...],
    abstention_enabled=False,   # default
    sufficiency_enabled=False,  # default
)
```

See [references/guards.md](references/guards.md) for per-guard details: what triggers each one, what the auto-fail result looks like, and how to configure or disable it.

---

## 4. Set Up Replicates

`replicate_count` controls how many times each question/model combination runs. Each replicate is an independent LLM call, useful for measuring response variance.

```python
config = VerificationConfig(
    answering_models=[...],
    parsing_models=[...],
    replicate_count=3,  # Run each question 3 times
)
```

Default is 1. The total number of pipeline executions is: `len(answering_models) x len(parsing_models) x len(questions) x replicate_count`.

---

## 5. Use Presets

Presets are JSON files that bundle a VerificationConfig for reuse. A preset file wraps the config under a `"config"` key: `{"config": { ...VerificationConfig fields... }}` — `from_preset()` reads `preset["config"]`. Comments and metadata can live at the top level next to `"config"`. There are no bundled presets: preset names are user-created JSON files in the presets dir (`KARENINA_PRESETS_DIR` or `./presets`). Use `karenina preset list` to see what exists.

**Load a preset**:

```python
from pathlib import Path
from karenina.schemas.verification import VerificationConfig

config = VerificationConfig.from_preset(Path("presets/my-preset.json"))
```

**CLI**:

```bash
karenina verify --preset presets/my-preset.json
```

**Save current config as a preset**:

```python
config.save_preset(
    name="my-experiment",
    description="Sonnet answerer, Haiku judge, 3 replicates",
)
```

Preset round-trips preserve ModelConfig fields: `save_preset` keeps `max_tokens`, `max_retries`, `request_timeout`, `agent_timeout`, `mcp_http_timeout`, `mcp_sse_read_timeout`, `retry_policy`, `mcp_tool_description_overrides`, MCP settings, `extra_kwargs`, `agent_runtime`, and `agent_middleware` (only `manual_traces` is excluded). Legacy preset files that lack these fields simply fall back to the ModelConfig defaults on load.

**Apply overrides on top of a preset**:

```python
base = VerificationConfig.from_preset(Path("presets/base.json"))
config = VerificationConfig.from_overrides(
    base,
    replicate_count=5,
    abstention=True,
    deep_judgment_mode="full",
)
```

See [assets/preset-minimal.json](assets/preset-minimal.json) for a minimal example and [assets/preset-full.json](assets/preset-full.json) for a fully commented example showing all fields.

---

## 6. Configure Prompt Assembly

Prompt assembly controls what text reaches each LLM in the pipeline. You do not write prompts directly; the pipeline assembles them from your configuration.

**What each LLM sees**:

| LLM Role | Receives | User Controls |
|----------|----------|---------------|
| Answering model | Question text + `system_prompt` from ModelConfig + few-shot examples (if configured) | `system_prompt`, `few_shot_config` |
| Judge (parsing) model | Raw response + template schema + extraction instructions + `system_prompt` from ModelConfig | `system_prompt`, `include_extraction_hints` |
| Rubric evaluator | Raw response + trait description + question text | Trait descriptions (defined in rubric) |
| Abstention/sufficiency detector | Raw response + question text | `prompt_config.abstention_detection`, `prompt_config.sufficiency_detection` |
| Agentic investigation agent | Question + workspace/trace + template schema | `prompt_config.agentic_parsing` |
| Agentic extraction parser | Investigation trace + template schema | `prompt_config.agentic_parsing` |

**Inject custom instructions per pipeline stage** using `PromptConfig`:

```python
from karenina.schemas.verification.prompt_config import PromptConfig

config = VerificationConfig(
    answering_models=[...],
    parsing_models=[...],
    prompt_config=PromptConfig(
        generation="Focus on clinical evidence only.",
        parsing="Be strict about extracting exact drug names.",
        rubric_evaluation="Evaluate hedging language carefully.",
    ),
)
```

PromptConfig fields: `generation`, `parsing`, `abstention_detection`, `sufficiency_detection`, `rubric_evaluation` (fallback for all rubric tasks), `agentic_parsing` (fallback for agentic parsing tasks), `deep_judgment` (fallback for all deep judgment tasks).

**Few-shot configuration**: Control example selection per question.

```python
from karenina.schemas.config import FewShotConfig

config = VerificationConfig(
    answering_models=[...],
    parsing_models=[...],
    few_shot_config=FewShotConfig(
        pool_mode="k-shot",
        pool_k=3,
    ),
)
```

---

## 7. Enable Deep Judgment

Deep judgment adds an extra verification pass that extracts supporting excerpts from the response and validates them against the source text. It runs after standard template verification.

**Template deep judgment** (verifies template field values):

```python
config = VerificationConfig(
    answering_models=[...],
    parsing_models=[...],
    deep_judgment_mode="full",
    deep_judgment_max_excerpts_per_attribute=3,
    deep_judgment_fuzzy_match_threshold=0.80,
    deep_judgment_excerpt_retry_attempts=2,
)
```

**Rubric deep judgment** (verifies rubric trait scores):

```python
config = VerificationConfig(
    answering_models=[...],
    parsing_models=[...],
    evaluation_mode="template_and_rubric",
    deep_judgment_rubric_mode="enable_all",  # Apply to all LLM traits
    deep_judgment_rubric_global_excerpts=True,
)
```

Deep judgment rubric modes:
- `"disabled"` (default): No rubric deep judgment
- `"enable_all"`: Apply deep judgment to all LLM traits
- `"use_checkpoint"`: Use settings saved in a previous checkpoint
- `"custom"`: Per-trait configuration via `deep_judgment_rubric_config`

**Search-enhanced deep judgment** (validates excerpts against external evidence):

```python
config = VerificationConfig(
    answering_models=[...],
    parsing_models=[...],
    deep_judgment_mode="full",
    deep_judgment_search_enabled=True,
    deep_judgment_search_tool="tavily",  # or pass a callable
)
```

---

## 8. MCP Integration

For evaluating tool-using agents (models that call external tools via MCP), configure MCP URLs on the answering model:

```python
config = VerificationConfig(
    answering_models=[
        ModelConfig(
            id="agent",
            model_provider="anthropic",
            model_name="claude-sonnet-4-6",
            interface="langchain",  # or "claude_tool"
            mcp_urls_dict={
                "filesystem": "http://localhost:8080/mcp",
                "database": "http://localhost:8081/mcp",
            },
            mcp_tool_filter=["read_file", "list_dir"],  # optional: limit available tools
            agent_middleware=AgentMiddlewareConfig(
                limits=AgentLimitConfig(model_call_limit=25, tool_call_limit=50),
            ),
        )
    ],
    parsing_models=[...],
    use_full_trace_for_template=False,  # Pass only final AI message to template parsing
    use_full_trace_for_rubric=True,     # Pass full agent trace to rubric evaluation
)
```

**Trace control**: `use_full_trace_for_template` and `use_full_trace_for_rubric` control what portion of the MCP agent trace is passed to evaluation. The full trace is always captured regardless of these settings.

**Per-request MCP timeouts**: `ModelConfig.mcp_http_timeout` (streamable-HTTP connection timeout; None = MCP SDK default 30s) and `ModelConfig.mcp_sse_read_timeout` (in-session SSE read timeout; None = MCP SDK default 300s) are now configurable per answering model.

For detailed MCP integration patterns, see `skills/using-karenina/references/advanced-adapters/mcp-integration.md`.

---

## 9. Extend a Prior Run

Two facades extend a completed verification run without re-generating answers: `Benchmark.extend_template` and `Benchmark.extend_rubric`. Both replay prior traces via an internal `ReplayStore`, so live LLM calls happen only where coverage is genuinely new. Full treatment: `skills/using-karenina/references/core_concepts/extending-runs.md`.

### When to use which

| Goal | Facade |
|------|--------|
| Add a new judge/parsing model, or more answerers, or more replicates to an existing run | `extend_template` |
| Score the same traces against a new rubric | `extend_rubric` |

Both return a standard `VerificationResultSet`. `store=True` (default) writes to the results manager under the effective `run_name`.

### extend_template

Signature: `extend_template(prior_results, config, *, run_name=None, question_ids=None, async_enabled=None, progress_callback=None, sink=None, store=True) -> VerificationResultSet`.

The caller expresses the FINAL shape as a full union. Output is a symmetric `(answerers x judges x replicates)` matrix. Mechanics: prior `(question, answerer, replicate)` traces are replayed at generation time; only new answerers / replicates miss the replay store and run live. Parsing always runs live, so new judges (parsing_models) see every trace.

Shape rules (validated up front):

- `config.answering_models` must be a SUPERSET of the answerers observed in `prior_results`.
- `config.replicate_count` must be `>=` the observed replicate count.
- `config.parsing_models` carries old + new judges (typically: prior judges plus the ones you want to add).
- `config.replay_store` must be `None` (the helper installs its own).

```python
from karenina.benchmark import Benchmark

benchmark = Benchmark.load("checkpoint.jsonld")
prior = benchmark.results.get_result_set(run_name="baseline")

config = VerificationConfig(
    answering_models=[sonnet, haiku],          # superset of prior answerers
    parsing_models=[old_judge, new_judge],     # add a new judge
    replicate_count=3,                         # >= prior replicate_count
    evaluation_mode="template_only",
)

result_set = benchmark.extend_template(prior, config, run_name="rejudged")
```

Implementation lives in `karenina/src/karenina/benchmark/verification/extension.py::extend_template_run`.

### extend_rubric

Signature: `extend_rubric(prior_results, config, *, run_name=None, question_ids=None, async_enabled=None, progress_callback=None, sink=None, store=True) -> VerificationResultSet`.

Attaches a new rubric to a prior run and scores every prior trace against it. There is no `rubric=` kwarg: attach the rubric to the benchmark first with `set_global_rubric` and/or `set_question_rubric`. The helper clones config with `replay_store=<store>` and `evaluation_mode="rubric_only"`, runs verification, then deep-copies prior rows and enriches them in place with the new trait scores.

Row count is preserved: output has exactly `len(prior_results.results)` rows with the same `result_id` values. Trait names are unioned with any rubric fields already on prior rows; a same-name collision raises `ValueError` naming the bucket and trait.

Shape rules (strict equality, not superset):

- `config.answering_models` must equal prior EXACTLY.
- `config.parsing_models` must equal prior EXACTLY.
- `config.replicate_count` must equal prior EXACTLY.

v1 scope: LLM, regex, callable, agentic traits and `DynamicRubric` are supported. Metric traits are rejected because they depend on parsed template fields, which `rubric_only` skips.

```python
benchmark.set_global_rubric(new_rubric)

config = VerificationConfig(
    answering_models=prior_config.answering_models,   # exact match
    parsing_models=prior_config.parsing_models,       # exact match
    replicate_count=prior_config.replicate_count,     # exact match
    evaluation_mode="rubric_only",
)

result_set = benchmark.extend_rubric(prior, config, run_name="with_safety_rubric")
```

Implementation lives in `extension.py::extend_rubric_run`.

### Common validation errors

- `ValueError: config.replay_store must be None`: the facade installs its own; do not pre-populate.
- `ValueError: config.answering_models must be a superset of prior answerers` (`extend_template`) or `must equal prior exactly` (`extend_rubric`).
- `ValueError: trait name 'X' collides in bucket 'llm_traits'`: a prior rubric already scored a trait by that name; rename or drop it.
- `ValueError: metric traits are not supported by extend_rubric`: parsed template fields are unavailable under `rubric_only`.

---

## 10. Progressive Save and Resume (Sinks)

Long runs (many questions, multiple models/replicates, expensive endpoints) should use a `ResultSink` so results are persisted incrementally and interrupted runs resume without re-doing completed work. Sinks are available from both Python and the CLI; they share the same on-disk layout.

**Module**: `karenina.benchmark.verification.sinks`

**Concrete sinks**:

| Sink | Purpose |
|------|---------|
| `ProgressiveFileSink` | Append-only `.results.jsonl` sidecar + `.state` manifest. Assembles final export on clean completion; retains sidecars on partial failure. |
| `DBSink` | Incremental SQLite writer (one row per completed result). Resume via DB is NOT implemented: `completed_triples()` returns empty. |
| `CompositeSink([...])` | Fans `on_start` / `on_result` / `on_finalize` across children; unions their `completed_triples()`. Use for file + DB together. |
| `InMemorySink` | Test helper; no I/O. |

**Fresh run**:

```python
from karenina.benchmark import Benchmark
from karenina.benchmark.verification.sinks import ProgressiveFileSink

benchmark = Benchmark.load("checkpoint.jsonld")
sink = ProgressiveFileSink(
    output_path=Path("results.json"),
    config=config,
    benchmark_path="checkpoint.jsonld",
)
result_set = benchmark.run_verification(config=config, sink=sink)
```

**Resume**:

```python
result_set = benchmark.resume_verification("results.json.state")
# Or override config (e.g. bump request_timeout):
tweaked = sink.config.model_copy(update={"request_timeout": 300.0})
benchmark.resume_verification("results.json.state", config=tweaked)
```

**Resume via classmethods**: `ProgressiveFileSink.open_or_resume(output_path, config, benchmark_path, *, global_rubric=None, config_updater=None)` resumes from the existing `.state` sidecar or creates a fresh sink; `ProgressiveFileSink.load_for_resume(state_path, *, global_rubric=None, config_updater=None)` rebuilds a sink from a `.state` file (this is what `resume_verification` uses under the hood).

**Compose file + database**:

```python
from karenina.benchmark.verification.sinks import CompositeSink, DBSink

sink = CompositeSink([
    ProgressiveFileSink(output_path=Path("results.json"), config=config,
                        benchmark_path="checkpoint.jsonld"),
    DBSink(storage_url="sqlite:///runs.sqlite", benchmark_name="my-benchmark", run_name="my-run"),
])
benchmark.run_verification(config=config, sink=sink)
```

**Resume semantics**:

- Resume is **triple-level**: the unit of work is `(question_id, answering_canonical_key, parsing_canonical_key, replicate)`. Multi-model / multi-replicate fan-outs skip only completed triples, not whole questions.
- For **scenario** benchmarks, slot 0 of the triple holds `scenario_id` instead of `question_id`. Scenarios are **combo-atomic**: all turns of one combo persist together or not at all. Interrupted combos re-run from turn 1. `Benchmark.resume_verification()` auto-dispatches between QA and scenario based on the benchmark.
- Internally, the sink's `completed_triples()` is merged into `config.skip_triples` before the task queue is built, so the engine never re-runs already-done work.
- The engine never raises for task-level failures. On partial failure the driver still flushes results through the sink and calls `on_finalize(all_complete=False)`, so sidecars stay on disk for the next resume. For QA runs the per-question failures live on each result's `result.metadata.failure`; `VerificationResultSet.errors` (the facade) and `EngineRunResult.errors` (`run_verification_batch`) are populated by scenario executions, not QA runs.

**CLI equivalents**: `--progressive-save` creates a `ProgressiveFileSink`; `--resume <state>` calls `Benchmark.resume_verification` under the hood. CLI flags (`--preset`, model overrides, feature flags) are ignored when `--resume` is passed; the config is restored from the state file.

See [Progressive Save tutorial](../using-karenina/references/workflows/running-verification/progressive-save.md) for a walkthrough.

---

## Gotchas

1. **`evaluation_mode` defaults to `"template_only"` and has NO `"auto"` mode**. Attaching a rubric does NOT auto-upgrade the mode — to score attached rubrics you must explicitly set `evaluation_mode="template_and_rubric"` (or `"rubric_only"`). `rubric_enabled` is now a computed property (`evaluation_mode in ("template_and_rubric", "rubric_only")`); passing it as a config field is silently stripped.

2. **`parsing_only=True` disables answer generation**. Required for TaskEval where you supply pre-recorded outputs. When `parsing_only=True`, `answering_models` can be empty.

3. **`async_enabled=True` by default**. The pipeline runs questions in parallel with `async_max_workers=2`. Set `async_max_workers` to control parallelism, or `async_enabled=False` for sequential execution.

4. **Guards run BEFORE parsing**. A guard auto-fail means no template or rubric evaluation happens for that question. If you see unexpected failures with no parsed output, check `result.template.recursion_limit_reached`, `result.template.abstention_detected`, and `result.metadata.failure` (`.category` / `.stage` / `.reason`).

5. **`model_provider` must match the actual provider**. Setting `model_provider="openai"` with an Anthropic model name causes cryptic import errors at adapter creation time, not at config validation.

6. **`include_extraction_hints=True` by default**. Extraction hints are appended to the parsing prompt to help the judge populate template fields. Disable with `include_extraction_hints=False` if they interfere with your template schema.

7. **Agentic features require `"langchain_deep_agents"` or `"claude_agent_sdk"` interface**. When using `agentic_parsing=True` or `AgenticRubricTrait`, set the parsing model's `interface` to `"langchain_deep_agents"` or `"claude_agent_sdk"`. The default `"langchain"` interface technically passes validation (it has a basic agent), but the agent is too minimal to properly investigate artifacts or workspaces. No warning is emitted; the pipeline just produces subpar results.

8. **Workspace settings**. `workspace_copy=True` (default) copies question workspaces before execution, protecting originals. Set `workspace_cleanup=True` (default) to delete working copies after the run.

9. **Failures never abort the batch**. If some questions fail during verification (in both sequential and parallel modes), the run completes with every successful result; each failed question records the reason on its own `result.metadata.failure` (QA runs never populate `VerificationResultSet.errors` — that field is scenario-execution only; `EngineRunResult.errors` from `run_verification_batch` likewise covers scenario/batch-level errors). There is no `VerificationBatchError`: inspect results and `.errors` rather than catching an exception.
