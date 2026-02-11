# VerificationConfig Reference

This is the exhaustive reference for all `VerificationConfig` fields. For a tutorial introduction with examples, see [VerificationConfig Tutorial](../../06-running-verification/verification-config.md).

`VerificationConfig` is a Pydantic model with **33 fields** organized into 10 categories below.

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
| `replicate_count` | `int` | `1` | Number of times to run each question/model combination. Higher values allow measuring variance across runs. |
| `parsing_only` | `bool` | `False` | When `True`, only parsing models are required (no answering models needed). Used for TaskEval and similar use cases where answers are pre-generated. |

---

## Evaluation Mode

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `evaluation_mode` | `Literal["template_only", "template_and_rubric", "rubric_only"]` | `"template_only"` | Determines which pipeline stages run. `template_only`: template verification only. `template_and_rubric`: both template and rubric evaluation. `rubric_only`: skip template verification, evaluate rubrics on raw response. |
| `rubric_enabled` | `bool` | `False` | Master switch for rubric evaluation. Must be `True` when `evaluation_mode` is `template_and_rubric` or `rubric_only`. Must be `False` when `evaluation_mode` is `template_only`. |
| `rubric_trait_names` | `list[str] \| None` | `None` | Optional filter to evaluate only specific rubric traits by name. When `None`, all traits are evaluated. |
| `rubric_evaluation_strategy` | `Literal["batch", "sequential"] \| None` | `"batch"` | How LLM rubric traits are evaluated. `batch`: all LLM traits in a single call (efficient, requires JSON output). `sequential`: traits evaluated one-by-one (more reliable, higher cost). |

**Validation rules:**

- `evaluation_mode="rubric_only"` requires `rubric_enabled=True`
- `evaluation_mode="template_and_rubric"` requires `rubric_enabled=True`
- `evaluation_mode="template_only"` requires `rubric_enabled=False`

---

## Trace Filtering

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `use_full_trace_for_template` | `bool` | `False` | If `True`, pass full agent trace to template parsing. If `False`, extract only the final AI message. The full trace is always captured in `raw_llm_response` regardless. |
| `use_full_trace_for_rubric` | `bool` | `True` | If `True`, pass full agent trace to rubric evaluation. If `False`, extract only the final AI message. The full trace is always captured in `raw_llm_response` regardless. |

!!! note
    If `use_full_trace_for_template=False` and the trace doesn't end with an AI message, the trace validation stage will fail with an error.

---

## Pre-Parsing Checks

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `abstention_enabled` | `bool` | `False` | Enable abstention/refusal detection. When the model refuses to answer, parsing is skipped and the result is auto-failed. |
| `sufficiency_enabled` | `bool` | `False` | Enable response sufficiency detection. When the response lacks enough information to fill the template, parsing is skipped and the result is auto-failed. |

See [Response Quality Checks](../../06-running-verification/response-quality-checks.md) for details.

---

## Embedding Check

| Field | Type | Default | Env Var | Description |
|-------|------|---------|---------|-------------|
| `embedding_check_enabled` | `bool` | `False` | `EMBEDDING_CHECK` | Enable semantic similarity verification as a fallback after template verify(). |
| `embedding_check_model` | `str` | `"all-MiniLM-L6-v2"` | `EMBEDDING_CHECK_MODEL` | SentenceTransformer model name for computing embeddings. |
| `embedding_check_threshold` | `float` | `0.85` | `EMBEDDING_CHECK_THRESHOLD` | Cosine similarity threshold (0.0–1.0). Values above this threshold are considered semantically matching. |

**Environment variable precedence:** Env vars are applied only when the field is not explicitly set. Explicit arguments always take priority over env vars.

---

## Async Execution

| Field | Type | Default | Env Var | Description |
|-------|------|---------|---------|-------------|
| `async_enabled` | `bool` | `True` | `KARENINA_ASYNC_ENABLED` | Enable parallel execution of verification across questions. |
| `async_max_workers` | `int` | `2` | `KARENINA_ASYNC_MAX_WORKERS` | Maximum number of concurrent verification workers when async is enabled. |

---

## Deep Judgment — Templates

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `deep_judgment_enabled` | `bool` | `False` | Enable multi-stage deep judgment analysis for template verification. Adds excerpt extraction, fuzzy matching, and reasoning to parsed results. |
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
| `deep_judgment_rubric_config` | `dict[str, Any] \| None` | `None` | Per-trait configuration for `custom` mode. See structure below. |
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

## Additional Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `few_shot_config` | `FewShotConfig \| None` | `None` | Few-shot prompting configuration. Controls example injection into prompts. See [Few-Shot Configuration](../../core_concepts/few-shot.md). |
| `prompt_config` | `PromptConfig \| None` | `None` | Per-task prompt instruction overrides. Injects custom instructions into specific pipeline stages. See [PromptConfig Tutorial](../../06-running-verification/prompt-config.md) and [PromptConfig Reference](prompt-config.md). |
| `db_config` | `Any \| None` | `None` | `DBConfig` instance for automatic result persistence to a database. When set, results are saved after each verification run. See [Database Persistence](../../06-running-verification/database-persistence.md). |

---

## Convenience Methods

### `from_overrides()`

Create a `VerificationConfig` by applying selective overrides to an optional base config. This is the canonical way to construct configs programmatically.

```python
config = VerificationConfig.from_overrides(
    answering_model="gpt-4o",
    answering_provider="openai",
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
| `deep_judgment` | `deep_judgment_enabled` | Enable template deep judgment |
| `evaluation_mode` | `evaluation_mode` + `rubric_enabled` | Sets both evaluation mode and rubric flag |
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

See [Presets](../../03-configuration/presets.md) for usage details.

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

See [Configuration Hierarchy](../../03-configuration/index.md) for the full precedence model including presets and CLI arguments.
