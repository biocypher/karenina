# VerificationConfig Reference

This is the exhaustive reference for all `VerificationConfig` fields. For a tutorial introduction with examples, see [Basic Verification](../../notebooks/running-verification/basic-verification.ipynb).

`VerificationConfig` is a Pydantic model with **~38 user-facing fields** organized into the categories below. Field counts can drift slightly with new releases; the source of truth is `karenina/schemas/verification/config.py`.

---

## Models

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `answering_models` | `list[ModelConfig]` | `[]` | List of answering model configurations. Each defines a model that generates responses to benchmark questions. Default system prompt applied automatically if not set. |
| `parsing_models` | `list[ModelConfig]` | *(required)* | List of parsing (judge) model configurations. At least one is required. Each defines a model that parses LLM responses into structured templates. Default system prompt applied automatically if not set. |

**Default system prompts** (applied when model has no explicit `system_prompt`):

- Answering: *"You are an expert assistant. Answer the question accurately and concisely."*
- Parsing: *"You are a validation assistant. Parse and validate responses against the given Pydantic template."*

See [ModelConfig Reference](model-config.md) for all `ModelConfig` fields.

---

## Execution

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `replicate_count` | `int` | `1` | Number of times to run each question/model combination. Higher values allow measuring variance across runs. Must be >= 1. |
| `parsing_only` | `bool` | `False` | When `True`, only parsing models are required (no answering models needed). Used for TaskEval and similar use cases where answers are pre-generated. |

---

## Evaluation Mode

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `evaluation_mode` | `Literal["template_only", "template_and_rubric", "rubric_only"]` | `"template_only"` | Determines which pipeline stages run. `template_only`: template verification only. `template_and_rubric`: both template and rubric evaluation. `rubric_only`: skip template verification, evaluate rubrics on raw response. When set to `template_and_rubric` or `rubric_only`, rubric evaluation is automatically enabled. |
| `rubric_trait_names` | `list[str] \| None` | `None` | Optional filter to evaluate only specific rubric traits by name. When `None`, all traits are evaluated. |
| `rubric_evaluation_strategy` | `Literal["batch", "sequential"] \| None` | `"batch"` | How LLM rubric traits are evaluated. `batch`: all LLM traits in a single call (efficient, requires JSON output). `sequential`: traits evaluated one-by-one (more reliable, higher cost). |
| `agentic_rubric_strategy` | `Literal["individual", "shared"]` | `"individual"` | How agentic rubric traits are evaluated. `individual`: one agent per trait (default, most reliable). `shared`: one agent evaluates all traits that share a model (efficient, but falls back to individual when models differ). |
| `agentic_rubric_parallel` | `bool` | `False` | Run individual agentic rubric trait sessions concurrently. Only applies when `agentic_rubric_strategy="individual"`. Wired into the orchestrator: each trait gets a concurrent agent session bounded by the global LLM semaphore. |

---

## Trace Filtering

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `use_full_trace_for_template` | `bool` | `False` | If `True`, the judge sees the full agent trace when parsing the template and when running the abstention/sufficiency checks. If `False`, the judge sees only the final AI message. The full trace is always captured in `raw_llm_response` regardless. |
| `use_full_trace_for_rubric` | `bool` | `True` | If `True`, pass full agent trace to rubric evaluation. If `False`, extract only the final AI message. The full trace is always captured in `raw_llm_response` regardless. |

!!! note
    If `use_full_trace_for_template=False` and the trace doesn't end with an AI message, the trace validation stage will fail with an error.

---

## Pre-Parsing Checks

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `abstention_enabled` | `bool` | `False` | Enable abstention/refusal detection. When the model refuses to answer, parsing is skipped and the result is auto-failed. |
| `sufficiency_enabled` | `bool` | `False` | Enable response sufficiency detection. When the response lacks enough information to fill the template, parsing is skipped and the result is auto-failed. |
| `include_extraction_hints` | `bool` | `True` | Controls whether extraction hints are included in parsing prompts. Extraction hints provide the judge LLM with guidance on how to extract specific template fields from the response. Enabled by default. |

See [Full Evaluation](../../notebooks/running-verification/full-evaluation.ipynb) for usage examples.

---

## Embedding Check

| Field | Type | Default | Env Var | Description |
|-------|------|---------|---------|-------------|
| `embedding_check_enabled` | `bool` | `False` | `EMBEDDING_CHECK` | Enable semantic similarity verification as a fallback after template verify(). |
| `embedding_check_model` | `str` | `"all-MiniLM-L6-v2"` | `EMBEDDING_CHECK_MODEL` | SentenceTransformer model name for computing embeddings. |
| `embedding_check_threshold` | `float` | `0.85` | `EMBEDDING_CHECK_THRESHOLD` | Cosine similarity threshold. Constrained to [0.0, 1.0]. Values above this threshold are considered semantically matching. |

**Environment variable precedence:** Env vars are applied only when the field is not explicitly set. Explicit arguments always take priority over env vars.

---

## Async Execution

| Field | Type | Default | Env Var | Description |
|-------|------|---------|---------|-------------|
| `async_enabled` | `bool` | `True` | `KARENINA_ASYNC_ENABLED` | Enable parallel execution of verification across questions. |
| `async_max_workers` | `int` | `2` | `KARENINA_ASYNC_MAX_WORKERS` | Maximum number of concurrent verification workers when async is enabled. Must be >= 1. |
| `max_concurrent_requests` | `int \| None` | `None` | `KARENINA_MAX_CONCURRENT_LLM_REQUESTS` | Global cap on concurrent LLM requests across all workers. `None` means no global limit (concurrency bounded by `async_max_workers` only). Set to 16-64 for self-hosted inference servers (vLLM, SGLang). |
| `task_ordering` | `Literal["auto", "prefix_cache", "distribute_answerers", "generation_order", "random"]` | `"auto"` | — | Task queue ordering strategy. `auto` picks `distribute_answerers` when answerers span more than one identity, else `prefix_cache`. `prefix_cache` groups by answering model for KV cache hits. `distribute_answerers` round-robins across answerer identities. `generation_order` preserves template-first loop order. `random` shuffles tasks. |
| `answerer_concurrency_limits` | `int \| dict[str, int] \| None` | `None` | — | Per-answerer concurrency cap, enforced at task start. Pass an `int` to apply the same cap to every entry in `answering_models` (keyed by `ModelConfig.id`). Pass a `dict` keyed by `ModelConfig.id` for per-model caps; answerers not in the dict run uncapped. `None` disables caps. |
| `request_timeout` | `float \| None` | `120.0` | — | HTTP request timeout (seconds) for all LLM calls in the pipeline (answer generation, parsing, rubric evaluation, guardrail calls). Set to `None` to use provider SDK defaults. |

Both sequential and parallel execution modes collect per-question errors without aborting. If any questions fail (or the parallel batch exceeds its timeout), `VerificationBatchError` is raised with `partial_results` and `errors` attributes so callers can recover partial progress. See [Basic Verification: Error Handling](../../notebooks/running-verification/basic-verification.ipynb) for usage examples.

---

## Deep Judgment — Templates

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `deep_judgment_mode` | `Literal["disabled", "reasoning_only", "full"]` | `"disabled"` | Template deep-judgment mode. `"disabled"`: off. `"reasoning_only"`: reasoning traces without excerpts (2 LLM calls). `"full"`: excerpts + reasoning (3+ LLM calls). |
| `deep_judgment_max_excerpts_per_attribute` | `int` | `3` | Maximum number of excerpts to extract per template attribute during deep judgment. |
| `deep_judgment_fuzzy_match_threshold` | `float` | `0.80` | Fuzzy match similarity threshold for validating excerpts against the original trace. |
| `deep_judgment_excerpt_retry_attempts` | `int` | `2` | Number of retry attempts for excerpt extraction when fuzzy matching fails. |
| `deep_judgment_search_enabled` | `bool` | `False` | Enable search-enhanced excerpt validation. When enabled, excerpts are verified against external evidence to detect hallucination. |
| `deep_judgment_search_tool` | `str \| Callable` | `"tavily"` | Search tool for excerpt validation. Built-in: `"tavily"`. Can also be any callable with signature `(str \| list[str]) -> (str \| list[str])`. Requires `TAVILY_API_KEY` for built-in tool. |

---

## Deep Judgment — Rubrics

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `deep_judgment_rubric_mode` | `Literal["disabled", "enable_all", "use_checkpoint", "custom"]` | `"disabled"` | Controls how deep judgment is applied to rubric traits. `disabled`: off. `enable_all`: apply to all LLM traits. `use_checkpoint`: use settings saved in checkpoint. `custom`: use per-trait configuration from `deep_judgment_rubric_config`. |
| `deep_judgment_rubric_global_excerpts` | `bool` | `True` | For `enable_all` mode: globally enable or disable excerpt extraction for all traits. |
| `deep_judgment_rubric_config` | `DeepJudgmentRubricCustomConfig \| None` | `None` | Per-trait configuration for `custom` mode. See structure below. |
| `deep_judgment_rubric_max_excerpts_default` | `int` | `7` | Default maximum excerpts per rubric trait (used as fallback when per-trait config omits this setting). |
| `deep_judgment_rubric_fuzzy_match_threshold_default` | `float` | `0.80` | Default fuzzy match threshold for rubric excerpt validation. |
| `deep_judgment_rubric_excerpt_retry_attempts_default` | `int` | `2` | Default retry attempts for rubric excerpt extraction. |
| `deep_judgment_rubric_search_tool` | `str \| Callable` | `"tavily"` | Search tool for rubric hallucination detection. Same options as `deep_judgment_search_tool`. |

### Custom Mode Config Structure

The `deep_judgment_rubric_config` dict (for `custom` mode) expects:

```json
{
  "global": {
    "TraitName": {
      "enabled": true,
      "excerpt_enabled": true,
      "max_excerpts": 5,
      "fuzzy_match_threshold": 0.80,
      "excerpt_retry_attempts": 2,
      "search_enabled": false
    }
  },
  "question_specific": {
    "question-id": {
      "TraitName": {
        "enabled": true,
        "excerpt_enabled": false
      }
    }
  }
}
```

Each trait entry is validated as a `DeepJudgmentTraitConfig` with these fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `True` | Whether deep judgment is enabled for this trait. |
| `excerpt_enabled` | `bool` | `True` | Whether to extract excerpts for this trait. |
| `max_excerpts` | `int \| None` | `None` | Max excerpts (falls back to `deep_judgment_rubric_max_excerpts_default`). |
| `fuzzy_match_threshold` | `float \| None` | `None` | Fuzzy threshold (falls back to global default). |
| `excerpt_retry_attempts` | `int \| None` | `None` | Retry attempts (falls back to global default). |
| `search_enabled` | `bool` | `False` | Enable search validation for this trait's excerpts. |

---

## Agentic Parsing

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `agentic_parsing` | `bool` | `False` | Enable agentic parsing (Stage 7b). The judge uses tools to independently verify artifacts before extracting structured data. Requires a parsing model with `agent_tier='deep_agent'`. |
| `agentic_judge_context` | `Literal["workspace_only", "trace_and_workspace", "trace_only"]` | `"workspace_only"` | What context the investigation agent receives. `workspace_only`: question + workspace path. `trace_and_workspace`: answering agent trace + workspace path. `trace_only`: equivalent to classical Stage 7a parsing. |
| `agentic_parsing_max_turns` | `int` | `15` | Max turns for the investigation agent. Must be >= 1. |
| `agentic_parsing_timeout` | `float` | `120.0` | Timeout in seconds for the investigation agent. Must be >= 0.0. |
| `agentic_parsing_materialize_trace` | `bool` | `False` | Write the answering agent trace to a file for Stage 7b's investigation agent. Used by the coding-task agentic parsing path. For scenario handover-level trace materialization, use the `transcript_materialize` handover strategy on `ScenarioEdge` instead. |
| `agentic_parsing_persist_trace` | `bool` | `False` | When `True`, the materialized trace file is kept after extraction. When `False` (default), it is cleaned up in a finally block after the stage runs. Ignored when `agentic_parsing_materialize_trace=False`. |

---

## Workspace

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `workspace_copy` | `bool` | `True` | When `True`, pre-existing question workspaces are copied to a sibling working directory before execution, protecting the original for re-runs. When `False`, the pipeline works directly in the original directory (destructive). |
| `workspace_cleanup` | `bool` | `True` | Whether to delete working copies after the run. Only applies to copied or auto-created workspaces, never to original source directories. |

The `workspace_root` directory is configured on `Benchmark`, not on `VerificationConfig`.

---

## Replay Store

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `replay_store` | `ReplayStore \| None` | `None` | Replay layer (see `karenina.replay`). When provided, the pipeline short-circuits to canned traces on matching keys and runs live otherwise. Excluded from serialization (can hold large captured traces). Loaded by the CLI from `--replay <path>`. |
| `replay_parse_on_hydration_mismatch` | `Literal["fall_through", "strict"]` | `"fall_through"` | What to do when a replay key matches the answering call but downstream parse inputs differ from the captured run. `fall_through` falls back to live execution, `strict` raises an error. |
| `skip_triples` | `frozenset[tuple[str, str, str, int \| None]] \| None` | `None` | Set of completed `(question_id, answering_canonical_key, parsing_canonical_key, replicate)` tuples that the executor must skip. Populated by the resume path (loaded from a `.state` file) and by [`extend_template`](../../notebooks/core_concepts/extending-runs.ipynb) so already-completed work is not re-run. Excluded from serialization and `repr`. |

See the [Replay Store](../../advanced-pipeline/replay-store.md) advanced page for the keying scheme, and [Extending Runs](../../notebooks/core_concepts/extending-runs.ipynb) for how `skip_triples` drives `extend_template` / `extend_rubric`.

---

## Retry and Error Handling

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `retry_policy` | `RetryPolicy` | `RetryPolicy()` | Per-category retry budgets for transient LLM errors. See [Error Handling](../../advanced-pipeline/error-handling.md) for the policy categories and how to compose custom policies. |
| `custom_error_patterns` | `list[ErrorPatternConfig]` | `[]` | User-defined error patterns appended to the `ErrorRegistry`. Lets callers classify provider-specific or self-hosted error strings into the registry's transient/fatal/timeout categories. |
| `max_requeue_count` | `int` | `5` | Maximum times a task can be requeued in the parallel executor's IN_PROGRESS cache loop before generating the answer fresh. Must be >= 1. |

---

## Scenario Execution

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `scenario_turn_limit` | `int` | `20` | Maximum turns before forced termination in scenario execution. Must be >= 1. |

---

## Additional Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `few_shot_config` | `FewShotConfig \| None` | `None` | Few-shot prompting configuration. Controls example injection into prompts. See [Few-Shot Configuration](../../notebooks/core_concepts/few-shot.ipynb). |
| `prompt_config` | `PromptConfig \| None` | `None` | Per-task prompt instruction overrides. Injects custom instructions into specific pipeline stages. See [Full Evaluation](../../notebooks/running-verification/full-evaluation.ipynb) for usage and [PromptConfig Reference](prompt-config.md) for all fields. |
| `db_config` | `DBConfig \| None` | `None` | `DBConfig` instance for automatic result persistence to a database. When set, results are saved after each verification run. See DBConfig fields below. |

### DBConfig Fields

`DBConfig` controls the database connection for auto-saving verification results. Import from `karenina.storage`:

```python
from karenina.storage import DBConfig

db_config = DBConfig(storage_url="sqlite:///results.db")
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `storage_url` | `str` | *(required)* | SQLAlchemy database URL (e.g. `sqlite:///results.db`, `postgresql://user:pass@host/db`) |
| `auto_create` | `bool` | `True` | Automatically create tables and views if missing |
| `auto_commit` | `bool` | `True` | Commit transactions automatically after operations |
| `echo` | `bool` | `False` | Log all SQL statements (useful for debugging) |
| `pool_size` | `int` | `5` | Connection pool size (non-SQLite only) |
| `max_overflow` | `int` | `10` | Max connections beyond pool_size (non-SQLite only) |
| `pool_recycle` | `int` | `3600` | Recycle connections after N seconds (-1 to disable) |
| `pool_pre_ping` | `bool` | `True` | Test connections before use |

SQLite databases automatically set `pool_size=1` and `max_overflow=0`.

Auto-save is controlled by the `AUTOSAVE_DATABASE` environment variable (`true`/`false`, default `true`). Auto-save only runs when `db_config` is set — without it, no database writes occur. Auto-save is non-blocking: failures are logged but do not raise exceptions.

---

## Convenience Methods

### `from_overrides()`

Create a `VerificationConfig` by applying selective overrides to an optional base config. This is the canonical way to construct configs programmatically.

```python
config = VerificationConfig.from_overrides(
    answering_model="claude-haiku-4-5",
    answering_provider="anthropic",
    answering_id="my-answering",
    parsing_model="claude-haiku-4-5",
    parsing_provider="anthropic",
    parsing_id="my-parsing",
    evaluation_mode="template_and_rubric",
    abstention=True,
)
```

| Parameter | Maps To | Description |
|-----------|---------|-------------|
| `answering_model` | `answering_models[0].model_name` | Answering model name |
| `answering_provider` | `answering_models[0].model_provider` | Answering model provider |
| `answering_id` | `answering_models[0].id` | Answering model identifier |
| `answering_interface` | `answering_models[0].interface` | Answering adapter interface |
| `parsing_model` | `parsing_models[0].model_name` | Parsing model name |
| `parsing_provider` | `parsing_models[0].model_provider` | Parsing model provider |
| `parsing_id` | `parsing_models[0].id` | Parsing model identifier |
| `parsing_interface` | `parsing_models[0].interface` | Parsing adapter interface |
| `temperature` | Both models' `temperature` | Shared temperature override |
| `manual_traces` | `answering_models[0].manual_traces` | Pre-recorded traces (sets interface to `manual`) |
| `replicate_count` | `replicate_count` | Number of replicates |
| `abstention` | `abstention_enabled` | Enable abstention detection |
| `sufficiency` | `sufficiency_enabled` | Enable sufficiency detection |
| `embedding_check` | `embedding_check_enabled` | Enable embedding check |
| `deep_judgment_mode` | `deep_judgment_mode` | Template deep-judgment mode (`"disabled"`, `"reasoning_only"`, `"full"`) |
| `evaluation_mode` | `evaluation_mode` | Sets the evaluation mode |
| `embedding_threshold` | `embedding_check_threshold` | Embedding similarity threshold |
| `embedding_model` | `embedding_check_model` | Embedding model name |
| `async_execution` | `async_enabled` | Enable async execution |
| `async_workers` | `async_max_workers` | Number of async workers |
| `use_full_trace_for_template` | `use_full_trace_for_template` | Trace filtering for templates |
| `use_full_trace_for_rubric` | `use_full_trace_for_rubric` | Trace filtering for rubrics |
| `deep_judgment_rubric_mode` | `deep_judgment_rubric_mode` | Rubric deep judgment mode |
| `deep_judgment_rubric_excerpts` | `deep_judgment_rubric_global_excerpts` | Global excerpt toggle |
| `deep_judgment_rubric_max_excerpts` | `deep_judgment_rubric_max_excerpts_default` | Max excerpts per trait |
| `deep_judgment_rubric_fuzzy_threshold` | `deep_judgment_rubric_fuzzy_match_threshold_default` | Fuzzy match threshold |
| `deep_judgment_rubric_retry_attempts` | `deep_judgment_rubric_excerpt_retry_attempts_default` | Retry attempts |
| `deep_judgment_rubric_search_tool` | `deep_judgment_rubric_search_tool` | Rubric search tool |
| `deep_judgment_rubric_config` | `deep_judgment_rubric_config` | Custom per-trait config |

### Preset Methods

| Method | Description |
|--------|-------------|
| `save_preset(name, description, presets_dir)` | Save config as a preset JSON file |
| `from_preset(filepath)` | Load a `VerificationConfig` from a preset file |
| `sanitize_preset_name(name)` | Convert preset name to safe filename |
| `validate_preset_metadata(name, description)` | Validate preset name and description |

See [Presets](../../workflows/configuration/presets.md) for usage details.

### Inspection Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_few_shot_config()` | `FewShotConfig \| None` | Get the active few-shot configuration |
| `is_few_shot_enabled()` | `bool` | Check if few-shot prompting is enabled |

---

## Configuration Precedence

Fields are resolved in this order (highest priority first):

1. **Explicit arguments** passed to the constructor or `from_overrides()`
2. **Environment variables** (only for fields that support them — embedding and async settings)
3. **Field defaults** defined on the class

See [Configuration Hierarchy](../../workflows/configuration/index.md) for the full precedence model including presets and CLI arguments.
