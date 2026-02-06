# Presets

Presets are reusable JSON files that capture a complete `VerificationConfig` — model choices, feature toggles, and execution settings — so you can reload the same configuration across multiple benchmark runs without reconfiguring manually each time.

---

## Why Use Presets?

- **Consistency** — The same preset guarantees identical configuration across runs, eliminating configuration drift
- **Reusability** — Switch between test, production, and experimental setups instantly
- **Shareability** — Exchange preset files with teammates or commit them to version control
- **Simplicity** — Replace 15+ configuration parameters with a single file reference

---

## Preset File Format

A preset file is a JSON wrapper around a `VerificationConfig` dictionary:

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Quick Test",
  "description": "Fast configuration for smoke tests",
  "config": {
    "answering_models": [
      {
        "id": "gpt-4.1-mini",
        "model_provider": "openai",
        "model_name": "gpt-4.1-mini",
        "temperature": 0.0,
        "interface": "langchain",
        "system_prompt": "You are an expert assistant..."
      }
    ],
    "parsing_models": [...],
    "replicate_count": 1,
    "rubric_enabled": false,
    "evaluation_mode": "template_only",
    "deep_judgment_enabled": false,
    "abstention_enabled": false,
    "async_enabled": true,
    "async_max_workers": 2
  },
  "created_at": "2026-01-15T10:30:00+00:00",
  "updated_at": "2026-01-15T10:30:00+00:00"
}
```

The top-level fields:

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Auto-generated UUID |
| `name` | str | Descriptive name (max 100 characters) |
| `description` | str \| null | Optional description (max 500 characters) |
| `config` | dict | Full `VerificationConfig` as a dictionary |
| `created_at` | str | ISO 8601 timestamp |
| `updated_at` | str | ISO 8601 timestamp |

The `config` object contains all `VerificationConfig` fields. Model configurations are sanitized when saved — interface-specific fields that don't apply are removed, and `manual_traces` are excluded (they must be provided at runtime).

See [Preset Schema Reference](../10-configuration-reference/preset-schema.md) for the complete schema specification.

---

## Creating Presets

### Python API

Build a `VerificationConfig` and call `save_preset()`:

```python
from karenina.schemas import VerificationConfig, ModelConfig

model = ModelConfig(
    id="gpt-4.1-mini",
    model_provider="openai",
    model_name="gpt-4.1-mini",
    temperature=0.0,
    interface="langchain",
)

config = VerificationConfig(
    answering_models=[model],
    parsing_models=[model],
    replicate_count=3,
    rubric_enabled=True,
    deep_judgment_enabled=True,
)

metadata = config.save_preset(
    name="Production Config",
    description="Standard setup with deep judgment and rubric evaluation",
)
print(f"Saved to: {metadata['filepath']}")
# Saved to: /path/to/presets/production-config.json
```

`save_preset()` parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Preset name (converted to filename: `"Quick Test"` → `quick-test.json`) |
| `description` | str \| None | Optional description |
| `presets_dir` | Path \| None | Custom directory (default: `KARENINA_PRESETS_DIR` env var or `./presets/`) |

!!! note "Duplicate names"
    `save_preset()` raises `ValueError` if a file with the same sanitized name already exists. Delete the existing preset first, or use a different name.

### What Gets Saved

**Included**: model configurations (answering and parsing), evaluation settings (replicate count, evaluation mode), rubric settings, feature flags (abstention, deep judgment, embedding check, sufficiency), few-shot configuration, async execution settings, prompt config.

**Excluded**: `run_name` (job-specific), `db_config` (environment-specific), `manual_traces` (must be uploaded at runtime).

---

## Loading Presets

### Python API

Load a preset and use it for verification:

```python
from pathlib import Path
from karenina import Benchmark
from karenina.schemas import VerificationConfig

benchmark = Benchmark.load("my_benchmark.jsonld")

# Load preset by file path
config = VerificationConfig.from_preset(Path("presets/production-config.json"))

results = benchmark.run_verification(config)
```

### CLI

Use the `--preset` flag with `karenina verify`:

```bash
# Load preset by path
karenina verify checkpoint.jsonld --preset presets/production-config.json

# Load preset by name (searches in presets directory)
karenina verify checkpoint.jsonld --preset production-config
```

### Overriding Preset Values

Presets provide a base configuration that you can override at runtime:

**CLI**: Any flag passed alongside `--preset` overrides the preset value:

```bash
# Preset has async_max_workers=2, CLI overrides to 8
karenina verify checkpoint.jsonld --preset production-config.json --async-workers 8
```

**Python API**: Load the preset, then use `from_overrides()` to apply changes:

```python
# Load preset as base, override specific fields
base_config = VerificationConfig.from_preset(Path("presets/production-config.json"))
config = VerificationConfig.from_overrides(
    base=base_config,
    answering_model="claude-sonnet-4-5-20250514",
    answering_provider="anthropic",
    answering_id="claude-sonnet",
)
```

---

## Managing Presets via CLI

The `karenina preset` command provides three subcommands:

### List presets

```bash
karenina preset list
```

Shows all `.json` files in the presets directory with their names and modification dates.

### Show preset details

```bash
karenina preset show production-config
```

Displays the full configuration as formatted JSON plus a summary of key settings (model count, replicates, feature flags).

Accepts either a preset name or a file path:

```bash
karenina preset show presets/production-config.json
```

### Delete a preset

```bash
karenina preset delete production-config
```

Prompts for confirmation before deleting the preset file.

---

## Preset Directory Resolution

Karenina looks for presets in this order:

1. **`KARENINA_PRESETS_DIR` environment variable** — If set, this directory is used
2. **`./presets/`** — Relative to the current working directory

Set a custom preset directory:

```bash
# In .env file
KARENINA_PRESETS_DIR=/shared/team/presets

# Or via shell export
export KARENINA_PRESETS_DIR=/shared/team/presets
```

The `karenina init` command creates a `presets/` directory in your workspace automatically.

---

## Resolution in the Configuration Hierarchy

Presets sit between CLI arguments and environment variables in the [configuration hierarchy](index.md):

```
CLI Arguments  >  Preset Values  >  Environment Variables  >  Built-in Defaults
```

When you combine a preset with CLI overrides:

1. **Load the preset** — All preset values become the base configuration
2. **Apply CLI overrides** — Explicitly-passed flags replace preset values
3. **Fill gaps from environment** — Any fields not set by preset or CLI fall back to environment variables, then defaults

This means a preset's value for `async_max_workers=2` overrides an environment variable `KARENINA_ASYNC_MAX_WORKERS=8`, but a CLI flag `--async-workers 4` overrides both.

---

## Filename Sanitization

Preset names are converted to safe filenames automatically:

- Converted to lowercase
- Spaces replaced with hyphens
- Non-alphanumeric characters removed (except hyphens)
- Consecutive hyphens collapsed
- Length limited to 96 characters
- `.json` extension appended

Examples:

| Preset Name | Filename |
|------------|----------|
| `"Quick Test"` | `quick-test.json` |
| `"GPT-4 vs Claude Comparison"` | `gpt-4-vs-claude-comparison.json` |
| `"My Config!"` | `my-config.json` |

---

## Next Steps

- [Configuration Hierarchy](index.md) — How presets interact with CLI args, env vars, and defaults
- [Environment Variables](environment-variables.md) — Setting project-wide defaults
- [Workspace Initialization](../02-installation/workspace-init.md) — Creating a project with a `presets/` directory
- [Running Verification](../06-running-verification/index.md) — Using presets in verification workflows
- [Preset Schema Reference](../10-configuration-reference/preset-schema.md) — Complete preset JSON schema
