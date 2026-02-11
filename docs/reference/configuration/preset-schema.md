# Preset Schema

A preset file is a JSON document that wraps a complete `VerificationConfig` with metadata. This page documents the full schema specification. For creating and using presets, see [Presets](../../03-configuration/presets.md).

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
- Evaluation settings (`evaluation_mode`, `rubric_enabled`, `replicate_count`, etc.)
- Feature flags (`abstention_enabled`, `sufficiency_enabled`, `embedding_check_enabled`, etc.)
- Deep judgment settings (template and rubric)
- Async execution settings (`async_enabled`, `async_max_workers`)
- Few-shot configuration (`few_shot_config`)
- Prompt configuration (`prompt_config`)

**Excluded from presets:**

| Field | Reason |
|-------|--------|
| `manual_traces` | Runtime-specific; must be provided via `--manual-traces` or Python API |
| `db_config` | Environment-specific database configuration |
| `run_name` | Per-job metadata, not part of reusable configuration |

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
        "model_name": "claude-sonnet-4-5-20250514",
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
    "rubric_enabled": true,
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
    "deep_judgment_enabled": true,
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
| `"GPT-4 vs Claude Comparison"` | `gpt-4-vs-claude-comparison.json` |
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

- [Presets Tutorial](../../03-configuration/presets.md) — creating, loading, and managing presets
- [Using Presets in Verification](../../06-running-verification/using-presets.md) — preset workflows
- [VerificationConfig Reference](verification-config.md) — complete `config` object fields
- [ModelConfig Reference](model-config.md) — model configuration fields within presets
