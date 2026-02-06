# Configuration

Karenina provides a layered configuration system that lets you control LLM providers, verification features, execution behavior, and data persistence. Settings can come from multiple sources, with a clear precedence order that ensures predictable behavior.

---

## Configuration Hierarchy

When Karenina resolves a configuration value, it checks sources in this order — the first source that provides a value wins:

```
┌──────────────────────────────────┐
│     CLI Arguments (highest)      │  karenina verify --async-workers 4
├──────────────────────────────────┤
│     Preset File Values           │  --preset my-config.json
├──────────────────────────────────┤
│     Environment Variables        │  KARENINA_ASYNC_MAX_WORKERS=8
├──────────────────────────────────┤
│     Built-in Defaults (lowest)   │  async_max_workers = 2
└──────────────────────────────────┘
```

**Key principle**: Explicit always beats implicit. If you pass a value directly (via CLI or code), it overrides everything else. Environment variables only take effect for fields that aren't set by CLI arguments or presets.

### How It Works in Practice

When you run a command like:

```bash
karenina verify checkpoint.jsonld --preset gpt-4o.json --answering-model claude-sonnet-4-5-20250514
```

Karenina resolves configuration in three steps:

1. **Load the preset** — All values from `gpt-4o.json` become the base configuration
2. **Apply CLI overrides** — `--answering-model claude-sonnet-4-5-20250514` replaces the preset's answering model
3. **Fill remaining gaps** — Any fields not set by the preset or CLI are resolved from environment variables, then from built-in defaults

The same precedence applies when using the Python API:

```python
from karenina.schemas import VerificationConfig

# Explicit arguments override env vars and defaults
config = VerificationConfig(
    answering_models=[model],
    parsing_models=[model],
    embedding_check_enabled=False,  # Overrides EMBEDDING_CHECK env var
    async_max_workers=4,            # Overrides KARENINA_ASYNC_MAX_WORKERS env var
)
```

---

## Configuration Layers

### 1. CLI Arguments

The highest-priority configuration source. Any flag passed on the command line overrides all other sources.

```bash
karenina verify checkpoint.jsonld \
    --interface langchain \
    --answering-model gpt-4.1-mini \
    --parsing-model gpt-4.1-mini \
    --provider openai \
    --async-workers 4 \
    --deep-judgment
```

CLI arguments are converted to a `VerificationConfig` internally using `from_overrides()`, which selectively applies only the flags you provide — unset flags don't override preset or default values.

See [CLI Reference](../09-cli-reference/index.md) for all available flags.

### 2. Presets

Presets are reusable JSON files containing a full `VerificationConfig`. They capture model choices, feature toggles, and execution settings in a shareable format.

```bash
# Use a preset as the base configuration
karenina verify checkpoint.jsonld --preset my-config.json

# Override specific preset values with CLI flags
karenina verify checkpoint.jsonld --preset my-config.json --async-workers 8
```

Presets are loaded via `VerificationConfig.from_preset()` in the Python API:

```python
from pathlib import Path
from karenina.schemas import VerificationConfig

config = VerificationConfig.from_preset(Path("presets/my-config.json"))
```

See [Presets](presets.md) for the preset file format, creation, and management.

### 3. Environment Variables

Environment variables provide project-wide defaults. They're read during `VerificationConfig` initialization — but only for fields that weren't already set by CLI arguments or preset values.

The most commonly used environment variables:

| Variable | Purpose | Default |
|----------|---------|---------|
| `OPENAI_API_KEY` | OpenAI API key | — |
| `ANTHROPIC_API_KEY` | Anthropic API key | — |
| `GOOGLE_API_KEY` | Google API key | — |
| `KARENINA_ASYNC_ENABLED` | Enable parallel execution | `true` |
| `KARENINA_ASYNC_MAX_WORKERS` | Number of parallel workers | `2` |
| `KARENINA_PRESETS_DIR` | Directory for preset files | `./presets/` |
| `EMBEDDING_CHECK` | Enable embedding similarity fallback | `false` |
| `DB_PATH` | SQLite database path | `dbs/karenina.db` |

Set environment variables via a `.env` file (recommended) or shell export:

```bash
# .env file (recommended — add to .gitignore)
OPENAI_API_KEY="sk-..."
KARENINA_ASYNC_MAX_WORKERS=4
```

See [Environment Variables](environment-variables.md) for the full list and usage examples.

### 4. Built-in Defaults

Every configuration field has a default value defined in `VerificationConfig`. These are used when no other source provides a value:

| Setting | Default |
|---------|---------|
| Async execution | Enabled |
| Parallel workers | 2 |
| Embedding check | Disabled |
| Abstention detection | Disabled |
| Deep judgment | Disabled |
| Rubric evaluation | Disabled |
| Evaluation mode | `template_only` |
| Database auto-save | Enabled |

---

## Workspace Initialization

The `karenina init` command creates a standard project layout with sensible defaults:

```bash
karenina init
```

This creates:

```
your-project/
├── dbs/               # Database storage
├── presets/            # Preset JSON files
├── mcp_presets/        # MCP configuration presets
├── checkpoints/        # Benchmark checkpoint files
├── defaults.json       # Default model/provider settings
└── .env                # Environment variable template
```

Use `--advanced` for an interactive setup that lets you customize directories, default provider, and model settings.

See [Workspace Initialization](workspace-init.md) for details on all created files and options.

---

## Precedence Examples

### Environment variable with no explicit override

```bash
export KARENINA_ASYNC_MAX_WORKERS="8"
```

```python
config = VerificationConfig(
    answering_models=[model],
    parsing_models=[model],
)
print(config.async_max_workers)  # 8 (from env var)
```

### Explicit argument overrides environment variable

```bash
export KARENINA_ASYNC_MAX_WORKERS="8"
```

```python
config = VerificationConfig(
    answering_models=[model],
    parsing_models=[model],
    async_max_workers=4,  # Explicit arg wins
)
print(config.async_max_workers)  # 4 (explicit overrides env)
```

### Preset overrides environment variable

```bash
export EMBEDDING_CHECK="true"
```

```python
# Preset file contains: embedding_check_enabled=false
config = VerificationConfig.from_preset(Path("my-preset.json"))
print(config.embedding_check_enabled)  # False (preset overrides env)
```

### CLI overrides preset

```bash
# Preset has async_max_workers=2, CLI overrides it
karenina verify checkpoint.jsonld --preset my-config.json --async-workers 8
# Result: async_max_workers=8 (CLI wins)
```

---

## Next Steps

- [Environment Variables](environment-variables.md) — Full list of environment variables and how to set them
- [Presets](presets.md) — Creating, managing, and sharing configuration presets
- [Workspace Initialization](workspace-init.md) — Setting up a new karenina project with `karenina init`
- [Core Concepts](../04-core-concepts/index.md) — Understanding templates, rubrics, and evaluation modes
- [Running Verification](../06-running-verification/index.md) — Putting configuration into practice
