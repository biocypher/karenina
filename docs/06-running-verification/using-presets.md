# Using Presets

Presets let you save and reload a complete `VerificationConfig` from a JSON file — model choices, feature flags, execution settings — so verification runs are consistent without reconfiguring every time.

This page covers using presets in the verification workflow. For creating presets, managing preset files, and the preset directory system, see [Presets](../03-configuration/presets.md).

---

## Loading a Preset

### Python API

Use `VerificationConfig.from_preset()` to load a preset file into a config object, then pass it to `run_verification()`:

```python
from pathlib import Path
from karenina import Benchmark
from karenina.schemas import VerificationConfig

benchmark = Benchmark.load("my_benchmark.jsonld")

# Load preset by file path
config = VerificationConfig.from_preset(Path("presets/production-config.json"))

results = benchmark.run_verification(config)
```

`from_preset()` reads the `"config"` key from the preset JSON and constructs a `VerificationConfig` from it. All fields not present in the preset fall back to environment variables, then built-in defaults — following the standard [configuration hierarchy](../03-configuration/index.md).

### CLI

Use the `--preset` flag with `karenina verify`:

```bash
# Load preset by path
karenina verify checkpoint.jsonld --preset presets/production-config.json

# Load preset by name (searches in presets directory)
karenina verify checkpoint.jsonld --preset production-config
```

When using `--preset` by name (without a path), the CLI searches for matching files in the presets directory (`KARENINA_PRESETS_DIR` env var, or `./presets/` by default).

---

## Overriding Preset Values

Presets provide a base configuration that you can override at runtime. This is the most common pattern — load a shared preset, then adjust specific settings for the current run.

### CLI Overrides

Any flag passed alongside `--preset` overrides the corresponding preset value:

```bash
# Preset has async_max_workers=2, CLI overrides to 8
karenina verify checkpoint.jsonld --preset production-config.json --async-workers 8

# Preset has deep_judgment_enabled=false, CLI enables it
karenina verify checkpoint.jsonld --preset production-config.json --deep-judgment

# Override the answering model while keeping everything else from the preset
karenina verify checkpoint.jsonld --preset production-config.json --answering-model gpt-4o

# Override evaluation mode and enable rubric features
karenina verify checkpoint.jsonld --preset production-config.json \
    --evaluation-mode template_and_rubric --abstention
```

### Python API Overrides

Load the preset, then use `from_overrides()` to apply changes:

```python
from pathlib import Path
from karenina.schemas import VerificationConfig

# Load preset as base
base_config = VerificationConfig.from_preset(Path("presets/production-config.json"))

# Override specific fields
config = VerificationConfig.from_overrides(
    base=base_config,
    answering_model="claude-sonnet-4-5-20250514",
    answering_provider="anthropic",
    answering_id="claude-sonnet",
    answering_interface="langchain",
)
```

`from_overrides()` creates a new `VerificationConfig` by dumping the base config to a dictionary, applying your non-`None` overrides, and reconstructing. The original preset config is not modified.

Common override parameters:

| Parameter | Overrides |
|-----------|-----------|
| `answering_model` | Answering model name |
| `answering_provider` | Answering model provider |
| `answering_id` | Answering model ID |
| `answering_interface` | Answering model interface |
| `parsing_model` | Parsing model name |
| `parsing_provider` | Parsing model provider |
| `replicate_count` | Number of replicates |
| `abstention` | Abstention detection (`abstention_enabled`) |
| `sufficiency` | Sufficiency detection (`sufficiency_enabled`) |
| `embedding_check` | Embedding check (`embedding_check_enabled`) |
| `deep_judgment` | Deep judgment (`deep_judgment_enabled`) |
| `evaluation_mode` | Evaluation mode string |
| `rubric_enabled` | Rubric evaluation flag |

See [VerificationConfig](verification-config.md) for the full list of configurable fields.

---

## Override Precedence

When combining presets with overrides, the [configuration hierarchy](../03-configuration/index.md) applies:

```
CLI Arguments / from_overrides()  (highest priority)
        ↓
Preset Values
        ↓
Environment Variables
        ↓
Built-in Defaults  (lowest priority)
```

Concretely:

1. **Load the preset** — All preset values become the base configuration
2. **Apply overrides** — CLI flags or `from_overrides()` parameters replace preset values
3. **Fill gaps from environment** — Fields not set by preset or overrides fall back to environment variables, then defaults

This means a preset value of `async_max_workers=2` overrides an environment variable `KARENINA_ASYNC_MAX_WORKERS=8`, but a CLI flag `--async-workers 4` overrides both.

---

## Combining Presets with Other Configuration

### Preset + Feature Flags

Enable features not in the preset:

```bash
# Preset has basic config; add abstention and deep judgment for this run
karenina verify checkpoint.jsonld --preset quick-test.json \
    --abstention --deep-judgment --evaluation-mode template_and_rubric
```

```python
config = VerificationConfig.from_overrides(
    base=VerificationConfig.from_preset(Path("presets/quick-test.json")),
    abstention=True,
    deep_judgment=True,
    evaluation_mode="template_and_rubric",
    rubric_enabled=True,
)
```

### Preset + Manual Traces

Use a preset's parsing configuration with pre-recorded traces:

```bash
karenina verify checkpoint.jsonld --preset production-config.json \
    --interface manual --manual-traces traces/my_traces.json
```

!!! note "Manual traces are never saved in presets"
    The `manual_traces` field is excluded when saving presets because traces are runtime-specific. You must always provide them at runtime via `--manual-traces` or the `manual_traces` parameter.

### Preset + Output

Combine preset configuration with output options:

```bash
# Run with preset and save results to CSV
karenina verify checkpoint.jsonld --preset production-config.json \
    --output results.csv --verbose

# Run with preset and progressive save for long benchmarks
karenina verify checkpoint.jsonld --preset production-config.json \
    --progressive-save --output progress.json
```

---

## Common Patterns

### Development vs Production

Maintain separate presets for different stages:

```bash
# Fast smoke test during development
karenina verify checkpoint.jsonld --preset dev-quick.json

# Full evaluation for production
karenina verify checkpoint.jsonld --preset production-full.json
```

### Model Comparison

Use the same preset as a base, overriding only the model:

```python
from pathlib import Path
from karenina import Benchmark
from karenina.schemas import VerificationConfig

benchmark = Benchmark.load("my_benchmark.jsonld")
base = VerificationConfig.from_preset(Path("presets/eval-config.json"))

models = [
    ("gpt-4o", "openai"),
    ("claude-sonnet-4-5-20250514", "anthropic"),
    ("gemini-2.0-flash", "google_genai"),
]

for model_name, provider in models:
    config = VerificationConfig.from_overrides(
        base=base,
        answering_model=model_name,
        answering_provider=provider,
        answering_id=model_name,
    )
    results = benchmark.run_verification(config, run_name=model_name)
```

This ensures all models are evaluated with identical settings (same parsing model, same feature flags, same replicates) — only the answering model changes.

### Shared Team Configuration

Commit presets to version control so the team uses consistent configurations:

```
project/
├── presets/
│   ├── default.json          # Standard evaluation
│   ├── quick-test.json       # Fast iteration
│   └── full-evaluation.json  # Comprehensive with deep judgment
├── checkpoints/
│   └── my_benchmark.jsonld
└── .env                      # API keys (not committed)
```

---

## Next Steps

- [Presets](../03-configuration/presets.md) — Creating, managing, and understanding preset files
- [VerificationConfig](verification-config.md) — All configuration fields and their defaults
- [Python API](python-api.md) — Complete verification workflow in code
- [CLI Verification](cli.md) — Running verification from the command line
- [Configuration Hierarchy](../03-configuration/index.md) — How presets interact with CLI args, env vars, and defaults
