# Preset Schema

A preset file is a JSON document that wraps a complete `VerificationConfig` with metadata. This page documents the full schema specification. For creating and using presets, see [Presets](../../workflows/configuration/presets.md).

---

## Top-Level Structure

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Quick Test",
  "description": "Fast configuration for smoke tests",
  "config": { ... },
  "created_at": "2026-01-15T10:30:00+00:00",
  "updated_at": "2026-01-15T10:30:00+00:00"
}
```

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `id` | string | Yes | UUID format (36 chars) | Auto-generated unique identifier |
| `name` | string | Yes | Non-empty, max 100 chars | Human-readable preset name |
| `description` | string \| null | No | Max 500 chars if provided | Optional description |
| `config` | object | Yes | Valid `VerificationConfig` dict | Complete verification configuration |
| `created_at` | string | Yes | ISO 8601 format | Creation timestamp |
| `updated_at` | string | Yes | ISO 8601 format | Last modification timestamp |

!!! note "Auto-generated fields"
    The `id`, `created_at`, and `updated_at` fields are set automatically by `save_preset()`. When loading a preset, only the `config` key is used — the metadata fields are informational.

---

## The `config` Object

The `config` key contains a `VerificationConfig` dictionary. All fields are documented in [VerificationConfig Reference](verification-config.md). The key differences when stored in a preset:

**Included in presets:**

- Model configurations (`answering_models`, `parsing_models`) — sanitized per interface
- Evaluation settings (`evaluation_mode`, `replicate_count`, etc.)
- Feature flags (`abstention_enabled`, `sufficiency_enabled`, `embedding_check_enabled`, etc.)
- Deep judgment settings (template and rubric)
- Async execution settings (`async_enabled`, `async_max_workers`)
- Few-shot configuration (`few_shot_config`)
- Prompt configuration (`prompt_config`)
- Retry settings (`retry_policy`, `custom_error_patterns`)

**Excluded from presets:**

| Field | Reason |
|-------|--------|
| `manual_traces` | Runtime-specific; must be provided via `--manual-traces` or Python API |
| `db_config` | Environment-specific database configuration |
| `replay_store` | Runtime-only object (can hold large captured traces); excluded from serialization |
| `skip_triples` | Internal plumbing for `Benchmark.extend_template`; never user-facing |

---

## Model Configuration Sanitization

When a preset is saved, each model in `answering_models` and `parsing_models` is sanitized to remove interface-irrelevant fields.

### Always included

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Model identifier |
| `model_provider` | string \| null | Provider (required for `langchain` interface) |
| `model_name` | string | Model name |
| `temperature` | float | Sampling temperature |
| `interface` | string | Adapter interface |
| `system_prompt` | string \| null | Custom system prompt |

### Conditionally included

| Field | Included When | Description |
|-------|--------------|-------------|
| `max_retries` | Present in config | Retry attempts |
| `endpoint_base_url` | `interface == "openai_endpoint"` and value non-empty | Custom endpoint URL |
| `endpoint_api_key` | `interface == "openai_endpoint"` and value non-empty | Custom API key |
| `anthropic_base_url` | `interface` in `("claude_tool", "claude_agent_sdk")` and value non-empty | Custom Anthropic endpoint |
| `anthropic_api_key` | `interface` in `("claude_tool", "claude_agent_sdk")` and value non-empty | Custom Anthropic API key |
| `mcp_urls_dict` | Non-empty dict | MCP server URLs |
| `mcp_tool_filter` | Non-empty list | Tool inclusion filter |
| `agent_middleware` | Non-empty dict | Agent middleware config |
| `extra_kwargs` | Non-empty dict | Vendor-specific parameters |
| `max_context_tokens` | Not None | Token limit for summarization |

### Always excluded from models

| Field | Reason |
|-------|--------|
| `manual_traces` | Runtime-specific, never serialized in presets |

---

## Retry Configuration

Two top-level fields on `VerificationConfig` control the central retry layer (`RetryExecutor`). Both are optional. When omitted, the pipeline falls back to `RetryPolicy()` defaults and the built-in `ErrorRegistry` rules. For the system's behavior at runtime, see [Error Handling and Retries](../../advanced-pipeline/error-handling.md).

### `retry_policy`

A nested object grouping one `CategoryRetryConfig` per retryable error category, plus an optional `timeout_escalation` block. Each category controls how many retries fire and how the per-attempt backoff grows.

```yaml
retry_policy:
  connection:
    max_attempts: 3
    backoff_min: 1.0
    backoff_max: 10.0
    backoff_multiplier: 2.0
  timeout:
    max_attempts: 3
    backoff_min: 5.0
    backoff_max: 30.0
    backoff_multiplier: 2.0
  rate_limit:
    max_attempts: 5
    backoff_min: 5.0
    backoff_max: 30.0
    backoff_multiplier: 2.0
  server_error:
    max_attempts: 2
    backoff_min: 2.0
    backoff_max: 15.0
    backoff_multiplier: 2.0
  timeout_escalation:
    strategy: additive       # or "multiplicative", "linear"
    increment: 15.0
    multiplier: 1.0
    max_timeout: 180.0
```

`CategoryRetryConfig` fields (each category):

| Field | Type | Default | Constraints | Description |
|-------|------|---------|-------------|-------------|
| `max_attempts` | int | varies | `>= 0` | Number of retries (not total calls); `0` disables retry for this category. |
| `backoff_min` | float | `1.0` | `>= 0` | Lower bound for the per-attempt backoff in seconds. |
| `backoff_max` | float | `10.0` | `>= 0` | Upper bound for the per-attempt backoff in seconds. |
| `backoff_multiplier` | float | `2.0` | `>= 1.0` | Exponential growth factor between attempts. |

Per-category `max_attempts` defaults: `connection=3`, `timeout=3`, `rate_limit=5`, `server_error=2`. Permanent errors are never retried.

`timeout_escalation` (optional `TimeoutEscalationConfig`):

| Field | Type | Default | Constraints | Description |
|-------|------|---------|-------------|-------------|
| `strategy` | string | (required) | `"additive"`, `"multiplicative"`, or `"linear"` | Growth function for the per-attempt timeout on `TIMEOUT` retries. |
| `increment` | float | `0.0` | `>= 0`; required `> 0` for `additive` | Seconds added per retry; used by `additive` only. |
| `multiplier` | float | `1.0` | `>= 1.0`; required `> 1.0` for `multiplicative` | Factor applied per retry; used by `multiplicative` only. |
| `max_timeout` | float \| null | `null` | `>= 0`; required for `linear` | Cap (`additive`/`multiplicative`) or endpoint (`linear`). |

### `custom_error_patterns`

A list of declarative `ErrorPatternConfig` entries. The pipeline registers each entry on the shared `ErrorRegistry` at the start of every verification run, so the rules apply uniformly across all adapters. Use this to teach karenina about provider-specific exceptions that the built-in classifier does not recognize.

```yaml
custom_error_patterns:
  - pattern: VllmQueueTimeout
    category: rate_limit
    match_type: type_name
  - pattern: context length exceeded
    category: permanent
    match_type: message_substring
```

`ErrorPatternConfig` fields:

| Field | Type | Default | Constraints | Description |
|-------|------|---------|-------------|-------------|
| `pattern` | string | (required) |   | Either an exception class name (`type_name`) or a substring of `str(exc)` (`message_substring`). |
| `category` | string | (required) | `"connection"`, `"timeout"`, `"rate_limit"`, `"server_error"`, `"permanent"` | Target `ErrorCategory`. |
| `match_type` | string | `"message_substring"` | `"type_name"` or `"message_substring"` | How `pattern` is matched against incoming exceptions. Substring matches are lowercased. |

User rules run before built-in rules, so a custom entry can override any default classification. For the full match order and the available categories, see [Error Handling and Retries](../../advanced-pipeline/error-handling.md).

### Worked Example: Both Blocks Together

A preset that widens the rate-limit budget for a flaky vLLM host and reclassifies its queue-timeout exception:

```yaml
config:
  answering_models: [...]
  parsing_models: [...]
  retry_policy:
    connection:
      max_attempts: 3
      backoff_min: 1.0
      backoff_max: 10.0
      backoff_multiplier: 2.0
    timeout:
      max_attempts: 4
      backoff_min: 10.0
      backoff_max: 60.0
      backoff_multiplier: 2.0
    rate_limit:
      max_attempts: 8
      backoff_min: 10.0
      backoff_max: 60.0
      backoff_multiplier: 2.0
    server_error:
      max_attempts: 1
      backoff_min: 2.0
      backoff_max: 15.0
      backoff_multiplier: 2.0
    timeout_escalation:
      strategy: additive
      increment: 15.0
      max_timeout: 180.0
  custom_error_patterns:
    - pattern: VllmQueueTimeout
      category: rate_limit
      match_type: type_name
    - pattern: context length exceeded
      category: permanent
      match_type: message_substring
```

JSON form (the persisted preset format) is the same structure with `null` instead of omitted optional fields.

---

## Complete Example

A preset with two answering models, template + rubric evaluation, and deep judgment enabled:

```json
{
  "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "name": "Production Full Evaluation",
  "description": "Multi-model template+rubric with deep judgment for production benchmarks",
  "config": {
    "answering_models": [
      {
        "id": "claude-haiku-4-5",
        "model_provider": "anthropic",
        "model_name": "claude-haiku-4-5",
        "temperature": 0.0,
        "interface": "langchain",
        "system_prompt": null
      },
      {
        "id": "claude-sonnet",
        "model_provider": null,
        "model_name": "claude-sonnet-4-5",
        "temperature": 0.0,
        "interface": "claude_agent_sdk",
        "system_prompt": null
      }
    ],
    "parsing_models": [
      {
        "id": "claude-haiku-4-5-parser",
        "model_provider": "anthropic",
        "model_name": "claude-haiku-4-5",
        "temperature": 0.0,
        "interface": "langchain",
        "system_prompt": null
      }
    ],
    "replicate_count": 1,
    "parsing_only": false,
    "evaluation_mode": "template_and_rubric",
    "rubric_trait_names": null,
    "rubric_evaluation_strategy": "batch",
    "use_full_trace_for_template": false,
    "use_full_trace_for_rubric": true,
    "abstention_enabled": true,
    "sufficiency_enabled": true,
    "embedding_check_enabled": false,
    "embedding_check_model": "all-MiniLM-L6-v2",
    "embedding_check_threshold": 0.85,
    "async_enabled": true,
    "async_max_workers": 4,
    "deep_judgment_mode": "full",
    "deep_judgment_max_excerpts_per_attribute": 3,
    "deep_judgment_fuzzy_match_threshold": 0.80,
    "deep_judgment_excerpt_retry_attempts": 2,
    "deep_judgment_search_enabled": false,
    "deep_judgment_search_tool": "tavily",
    "deep_judgment_rubric_mode": "enable_all",
    "deep_judgment_rubric_global_excerpts": true,
    "deep_judgment_rubric_max_excerpts_default": 7,
    "deep_judgment_rubric_fuzzy_match_threshold_default": 0.80,
    "deep_judgment_rubric_excerpt_retry_attempts_default": 2,
    "deep_judgment_rubric_search_tool": "tavily",
    "deep_judgment_rubric_config": null,
    "few_shot_config": null,
    "prompt_config": null
  },
  "created_at": "2026-02-06T14:30:00+00:00",
  "updated_at": "2026-02-06T14:30:00+00:00"
}
```

---

## Minimal Example

A preset with only the required fields:

```json
{
  "id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
  "name": "Quick Smoke Test",
  "description": null,
  "config": {
    "answering_models": [
      {
        "id": "answering-1",
        "model_provider": "anthropic",
        "model_name": "claude-haiku-4-5",
        "temperature": 0.1,
        "interface": "langchain",
        "system_prompt": null
      }
    ],
    "parsing_models": [
      {
        "id": "parsing-1",
        "model_provider": "anthropic",
        "model_name": "claude-haiku-4-5",
        "temperature": 0.1,
        "interface": "langchain",
        "system_prompt": null
      }
    ]
  },
  "created_at": "2026-02-06T14:30:00+00:00",
  "updated_at": "2026-02-06T14:30:00+00:00"
}
```

Fields not specified in the `config` object use `VerificationConfig` defaults (e.g., `replicate_count` defaults to `1`, `evaluation_mode` defaults to `"template_only"`).

---

## Metadata Validation

When saving a preset, metadata is validated via `validate_preset_metadata()`:

| Field | Rule | Error |
|-------|------|-------|
| `name` | Non-empty string | `ValueError: Preset name cannot be empty` |
| `name` | Max 100 characters | `ValueError: Preset name cannot exceed 100 characters` |
| `description` | Max 500 characters (if provided) | `ValueError: Description cannot exceed 500 characters` |

---

## Filename Sanitization

The `name` field is converted to a safe filename via `sanitize_preset_name()`:

1. Convert to lowercase
2. Replace spaces with hyphens
3. Remove non-alphanumeric characters (except hyphens)
4. Collapse consecutive hyphens to a single hyphen
5. Strip leading and trailing hyphens
6. Limit to 96 characters
7. Append `.json` extension

| Name | Filename |
|------|----------|
| `"Quick Test"` | `quick-test.json` |
| `"Haiku vs Sonnet Comparison"` | `haiku-vs-sonnet-comparison.json` |
| `"My Config!"` | `my-config.json` |
| `""` (empty after sanitization) | `preset.json` |

---

## Error Handling

### Saving

| Error | Condition |
|-------|-----------|
| `ValueError` | Name fails validation (empty or exceeds 100 chars) |
| `ValueError` | Description exceeds 500 characters |
| `ValueError` | File with same sanitized name already exists |

### Loading

| Error | Condition |
|-------|-----------|
| `FileNotFoundError` | Preset file does not exist |
| `json.JSONDecodeError` | JSON is malformed |
| `ValueError` | Missing `config` key or invalid configuration |

---

## Preset Directory Resolution

When saving or loading presets, the directory is resolved in this order:

1. **Explicit `presets_dir` parameter** — if passed to `save_preset()` or `resolve_preset_path()`
2. **`KARENINA_PRESETS_DIR` environment variable** — if set
3. **`./presets/`** — relative to the current working directory

When loading by name (not path), `resolve_preset_path()` searches the presets directory for `{name}.json`.

---

## Related

- [Presets Tutorial](../../workflows/configuration/presets.md) — creating, loading, and managing presets
- [Using Presets in Verification](../../notebooks/running-verification/full-evaluation.ipynb) — preset workflows
- [VerificationConfig Reference](verification-config.md) — complete `config` object fields
- [ModelConfig Reference](model-config.md) — model configuration fields within presets
