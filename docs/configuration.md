# Configuration

This guide covers all configuration options available in Karenina, including environment variables, programmatic configuration, and the precedence mechanism.

---

## Overview

Karenina provides flexible configuration to control:
- **LLM providers and models**: OpenAI, Google, Anthropic, OpenRouter
- **Features**: Embedding check, abstention detection, rubric evaluation
- **Execution**: Parallel vs sequential processing
- **Database**: SQLite database location

---

## Configuration Methods

Karenina supports two configuration approaches:

1. **Environment Variables**: Global settings affecting all operations
2. **Programmatic Configuration**: Per-operation settings via Python code

### Configuration Precedence

Understanding precedence is crucial for predictable behavior:

1. **Explicit arguments** (including preset values) - Always used if provided
2. **Environment variables** - Used only if explicitly set AND no programmatic argument provided
3. **Field defaults** - Used if no env var set and no explicit argument

**Key principle**: Environment variables are only read when they are explicitly set. If an environment variable is not set, Karenina uses the field default. If you explicitly pass an argument (or load from a preset), it always takes precedence over environment variables.

---

## Environment Variables

### API Keys

API keys are required to use LLM providers. These are always configured via environment variables.

| Variable | Description | Required For |
|----------|-------------|--------------|
| `OPENAI_API_KEY` | OpenAI API key | OpenAI models (GPT-4, etc.) |
| `GOOGLE_API_KEY` | Google API key | Google models (Gemini) |
| `ANTHROPIC_API_KEY` | Anthropic API key | Anthropic models (Claude) |
| `OPENROUTER_API_KEY` | OpenRouter API key | OpenRouter unified access |

#### Setting API Keys

**Option 1: Using .env file (Recommended)**

Create a `.env` file in your project root:

```bash
# .env
OPENAI_API_KEY="sk-..."
GOOGLE_API_KEY="AIza..."
ANTHROPIC_API_KEY="sk-ant-..."
OPENROUTER_API_KEY="sk-or-..."
```

**Important**: Add `.env` to `.gitignore` to prevent committing secrets:

```bash
echo ".env" >> .gitignore
```

If using Python's `dotenv` package, load the file:

```python
from dotenv import load_dotenv
load_dotenv()

from karenina import Benchmark
```

**Option 2: Shell export (temporary)**

Set for current session only:

```bash
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="AIza..."
```

**Option 3: Shell config (permanent)**

Add to `~/.bashrc` or `~/.zshrc`:

```bash
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.bashrc
source ~/.bashrc
```

**Option 4: In Python code**

Set before importing Karenina:

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

from karenina import Benchmark
```

#### Best Practices

- ✅ Use `.env` files for local development
- ✅ Use different keys for dev and production
- ✅ Rotate keys regularly
- ✅ Add `.env` to `.gitignore`
- ❌ **Never** commit API keys to version control
- ❌ **Never** share keys between team members

---

### Feature Toggles

These settings enable or configure advanced features. They can be set via environment variables OR programmatically via `VerificationConfig`.

| Variable | Description | Default | Values |
|----------|-------------|---------|--------|
| `EMBEDDING_CHECK` | Enable semantic similarity fallback | `false` | `true`, `false` |
| `EMBEDDING_CHECK_MODEL` | SentenceTransformer model for embeddings | `all-MiniLM-L6-v2` | Model name |
| `EMBEDDING_CHECK_THRESHOLD` | Similarity threshold (0.0-1.0) | `0.85` | Float |
| `ABSTENTION_CHECK_ENABLED` | Enable refusal/abstention detection | `false` | `true`, `false` |

**What these features do**:

- **`EMBEDDING_CHECK`**: Enables semantic similarity fallback for answer verification. When enabled, if template parsing fails, Karenina computes embedding similarity between the LLM response and expected answer. If similarity exceeds the threshold, the answer is marked as correct. See [Embedding Check](advanced/embedding-check.md) for details.

- **`EMBEDDING_CHECK_MODEL`**: The SentenceTransformer model used for computing embeddings. Common options: `all-MiniLM-L6-v2` (default, fast), `all-mpnet-base-v2` (better quality), `multi-qa-mpnet-base-dot-v1` (optimized for semantic search).

- **`EMBEDDING_CHECK_THRESHOLD`**: Cosine similarity threshold (0.0-1.0) for accepting answers. Higher values = stricter matching. `0.85` is a balanced default. Increase for stricter validation, decrease for more lenient matching.

- **`ABSTENTION_CHECK_ENABLED`**: Enables detection of refusals or abstentions (e.g., "I cannot answer this question"). When enabled, Karenina uses an LLM to determine if the model refused to answer. See [Abstention Detection](advanced/abstention-detection.md) for details.

**Example**:
```bash
export EMBEDDING_CHECK="true"
export EMBEDDING_CHECK_MODEL="all-mpnet-base-v2"
export EMBEDDING_CHECK_THRESHOLD="0.90"
export ABSTENTION_CHECK_ENABLED="true"
```

**Precedence**: These can also be configured via `VerificationConfig` (see below). Programmatic arguments take precedence over environment variables.

---

### Execution Control

These settings control how Karenina executes verification tasks. They can be set via environment variables OR programmatically via `VerificationConfig`.

| Variable | Description | Default | Values |
|----------|-------------|---------|--------|
| `KARENINA_ASYNC_ENABLED` | Enable parallel execution | `true` | `true`, `false` |
| `KARENINA_ASYNC_MAX_WORKERS` | Number of parallel workers | `2` | Integer |

**What these settings do**:

- **`KARENINA_ASYNC_ENABLED`**: Controls whether verification runs in parallel (multiple questions at once) or sequentially (one at a time). Parallel execution is faster but uses more API quota. Set to `false` for sequential execution (useful for debugging or rate-limit-sensitive scenarios).

- **`KARENINA_ASYNC_MAX_WORKERS`**: Number of parallel workers (questions processed simultaneously). Higher values = faster execution but more API quota usage and potential rate limits. Recommended: `2-4` for typical use, `1` for debugging, `8-16` for bulk processing (if your API quota allows).

**Example**:
```bash
# Enable parallel processing with 4 workers
export KARENINA_ASYNC_ENABLED="true"
export KARENINA_ASYNC_MAX_WORKERS="4"

# Disable for sequential execution (debugging)
export KARENINA_ASYNC_ENABLED="false"
```

**Precedence**: These can also be configured via `VerificationConfig` (see below). Programmatic arguments take precedence over environment variables.

---

### Database Location

| Variable | Description | Default |
|----------|-------------|---------|
| `DB_PATH` | SQLite database file path | `dbs/karenina.db` |

**What this setting does**:

- **`DB_PATH`**: Path to the SQLite database file used for storing benchmark data. Relative paths are resolved from the current working directory. Absolute paths are recommended for production use.

**Example**:
```bash
# Relative path
export DB_PATH="dbs/my_project.db"

# Absolute path (recommended for production)
export DB_PATH="/path/to/project/dbs/benchmark.db"
```

**Best practices**:
- Use descriptive database names (`genomics_benchmark.db`, not `test.db`)
- Organize databases by project or domain
- Back up databases regularly
- Use absolute paths for production

---

### Presets Directory

| Variable | Description | Default |
|----------|-------------|---------|
| `KARENINA_PRESETS_DIR` | Directory for storing configuration presets | `benchmark_presets/` |

**What this setting does**:

- **`KARENINA_PRESETS_DIR`**: Directory where `VerificationConfig` presets are saved and loaded. Used by `config.save_preset()` and `VerificationConfig.from_preset()`. See [Configuration Presets](advanced/presets.md) for details.

**Example**:
```bash
export KARENINA_PRESETS_DIR="/path/to/my/presets"
```

---

## Programmatic Configuration

### Model Configuration

Use `ModelConfig` to specify LLM models for answering and parsing:

```python
from karenina.schemas import ModelConfig

model_config = ModelConfig(
    id="my-model",                 # Unique identifier
    model_name="gpt-4.1-mini",     # Model name
    model_provider="openai",       # Provider: openai, google, anthropic, openrouter
    temperature=0.0,               # Temperature (0.0 = deterministic)
    system_prompt="Custom prompt"  # Optional system prompt
)
```

**Supported providers**:
- `"openai"` - OpenAI (GPT-4, GPT-4o, GPT-4.1-mini, etc.)
- `"google"` - Google (Gemini models)
- `"anthropic"` - Anthropic (Claude models)
- `"openrouter"` - OpenRouter (unified access to multiple providers)

**Common models**:
- `"gpt-4.1-mini"` (default) - Fast, cost-effective, recommended for most use cases
- `"gpt-4o"` - Higher quality, more expensive
- `"claude-3-5-sonnet-20241022"` - Anthropic's flagship model
- `"gemini-2.0-flash-exp"` - Google's latest fast model

**Temperature**: Set to `0.0` for deterministic benchmarking (recommended). Higher values introduce randomness.

---

### Verification Configuration

Use `VerificationConfig` to control verification behavior. This is the primary configuration object for running verification.

```python
from karenina.schemas import VerificationConfig, ModelConfig

# Define models
model_config = ModelConfig(
    id="answering-model",
    model_name="gpt-4.1-mini",
    model_provider="openai",
    temperature=0.0
)

# Create verification config
config = VerificationConfig(
    # Models
    answering_models=[model_config],        # Models to benchmark (can test multiple)
    parsing_models=[model_config],          # Models for parsing/grading responses

    # Execution
    replicate_count=3,                      # Number of times to run each question

    # Feature toggles
    rubric_enabled=True,                    # Enable rubric-based evaluation
    deep_judgment_enabled=False,            # Enable multi-stage parsing with excerpts
    abstention_enabled=False,               # Enable abstention/refusal detection

    # Embedding check settings (can override env vars)
    embedding_check_enabled=True,           # Enable semantic similarity fallback
    embedding_check_model="all-MiniLM-L6-v2",  # Embedding model
    embedding_check_threshold=0.85,         # Similarity threshold (0.0-1.0)

    # Async execution settings (can override env vars)
    async_enabled=True,                     # Enable parallel execution
    async_max_workers=4                     # Number of parallel workers
)
```

**Configuration options**:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `answering_models` | `list[ModelConfig]` | Models to benchmark (can test multiple) | Required |
| `parsing_models` | `list[ModelConfig]` | Models for parsing/grading responses | Required |
| `replicate_count` | `int` | Number of times to run each question | `1` |
| `rubric_enabled` | `bool` | Enable rubric-based evaluation | `False` |
| `deep_judgment_enabled` | `bool` | Enable multi-stage parsing with excerpts | `False` |
| `abstention_enabled` | `bool` | Enable abstention/refusal detection | `False` |
| `embedding_check_enabled` | `bool` | Enable semantic similarity fallback | `False` |
| `embedding_check_model` | `str` | SentenceTransformer model for embeddings | `"all-MiniLM-L6-v2"` |
| `embedding_check_threshold` | `float` | Similarity threshold (0.0-1.0) | `0.85` |
| `async_enabled` | `bool` | Enable parallel execution | `True` |
| `async_max_workers` | `int` | Number of parallel workers | `2` |

**See also**:
- [Running Verification](using-karenina/verification.md) - Complete guide to verification
- [Deep-Judgment](advanced/deep-judgment.md) - Multi-stage parsing feature
- [Abstention Detection](advanced/abstention-detection.md) - Refusal detection
- [Embedding Check](advanced/embedding-check.md) - Semantic similarity fallback
- [Configuration Presets](advanced/presets.md) - Saving and loading configurations

---

## Default Values

Karenina uses these defaults when no configuration is provided:

```python
# Model defaults
model_name = "gpt-4.1-mini"
model_provider = "openai"
temperature = 0.0

# Feature defaults
embedding_check_enabled = False
embedding_check_model = "all-MiniLM-L6-v2"
embedding_check_threshold = 0.85
abstention_enabled = False
deep_judgment_enabled = False
rubric_enabled = False

# Execution defaults
async_enabled = True
async_max_workers = 2

# Database default
db_path = "dbs/karenina.db"
```

---

## Using .env Files

**Recommended approach** for managing environment variables:

1. Create a `.env` file in your project root:

```bash
# .env
# API Keys
OPENAI_API_KEY="sk-..."
GOOGLE_API_KEY="AIza..."
ANTHROPIC_API_KEY="sk-ant-..."

# Database
DB_PATH="dbs/karenina.db"

# Feature toggles
EMBEDDING_CHECK="true"
EMBEDDING_CHECK_MODEL="all-mpnet-base-v2"
EMBEDDING_CHECK_THRESHOLD="0.90"
ABSTENTION_CHECK_ENABLED="true"

# Execution control
KARENINA_ASYNC_ENABLED="true"
KARENINA_ASYNC_MAX_WORKERS="4"

# Presets
KARENINA_PRESETS_DIR="benchmark_presets"
```

2. Add `.env` to `.gitignore`:

```bash
echo ".env" >> .gitignore
```

3. Load in Python (if needed):

```python
from dotenv import load_dotenv
load_dotenv()

from karenina import Benchmark
```

---

## Configuration Precedence Examples

### Example 1: No env vars, no explicit args → defaults

```python
# No environment variables set
config = VerificationConfig(
    answering_models=[model],
    parsing_models=[model]
)

print(config.embedding_check_enabled)  # False (field default)
print(config.async_max_workers)  # 2 (field default)
```

### Example 2: Env vars set, no explicit args → env values

```bash
export EMBEDDING_CHECK="true"
export KARENINA_ASYNC_MAX_WORKERS="8"
```

```python
config = VerificationConfig(
    answering_models=[model],
    parsing_models=[model]
)

print(config.embedding_check_enabled)  # True (from env)
print(config.async_max_workers)  # 8 (from env)
```

### Example 3: Env vars set, explicit args → explicit args override

```bash
export EMBEDDING_CHECK="true"
export KARENINA_ASYNC_MAX_WORKERS="8"
```

```python
config = VerificationConfig(
    answering_models=[model],
    parsing_models=[model],
    embedding_check_enabled=False,  # Overrides env (env says True)
    async_max_workers=4  # Overrides env (env says 8)
)

print(config.embedding_check_enabled)  # False (explicit arg)
print(config.async_max_workers)  # 4 (explicit arg)
```

### Example 4: Loading preset with env vars → preset values override

```bash
export EMBEDDING_CHECK="true"
export KARENINA_ASYNC_MAX_WORKERS="8"
```

```python
# Preset file contains: embedding_check_enabled=False, async_max_workers=5
config = VerificationConfig.from_preset(Path("my-preset.json"))

print(config.embedding_check_enabled)  # False (from preset)
print(config.async_max_workers)  # 5 (from preset)
```

---

## Configuration Verification

Verify your configuration is set correctly:

```python
import os
from karenina.schemas import ModelConfig, VerificationConfig

# Check API keys (masked for security)
print("API Keys:")
print(f"  OpenAI: {'✓' if os.getenv('OPENAI_API_KEY') else '✗'}")
print(f"  Google: {'✓' if os.getenv('GOOGLE_API_KEY') else '✗'}")
print(f"  Anthropic: {'✓' if os.getenv('ANTHROPIC_API_KEY') else '✗'}")

# Check environment variables
print("\nEnvironment Variables:")
print(f"  EMBEDDING_CHECK: {os.getenv('EMBEDDING_CHECK', 'not set')}")
print(f"  KARENINA_ASYNC_ENABLED: {os.getenv('KARENINA_ASYNC_ENABLED', 'not set')}")
print(f"  DB_PATH: {os.getenv('DB_PATH', 'not set (using default)')}")

# Create a config and check effective values
model = ModelConfig(
    id="test",
    model_name="gpt-4.1-mini",
    model_provider="openai",
    temperature=0.0
)

config = VerificationConfig(
    answering_models=[model],
    parsing_models=[model]
)

print("\nEffective VerificationConfig Values:")
print(f"  embedding_check_enabled: {config.embedding_check_enabled}")
print(f"  embedding_check_model: {config.embedding_check_model}")
print(f"  embedding_check_threshold: {config.embedding_check_threshold}")
print(f"  async_enabled: {config.async_enabled}")
print(f"  async_max_workers: {config.async_max_workers}")
```

---

## Troubleshooting Configuration

### API key not found

```bash
# Check if key is set
echo $OPENAI_API_KEY

# Set temporarily
export OPENAI_API_KEY="sk-..."

# Set in Python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
```

### Wrong model selected

```python
# Verify model config
model_config = ModelConfig(
    id="test",
    model_name="gpt-4.1-mini",  # Check spelling
    model_provider="openai",     # Check provider
    temperature=0.0
)

# Print for debugging
print(model_config.model_dump())
```

### Features not activating

```bash
# Check environment variables
env | grep -E "EMBEDDING|ABSTENTION|ASYNC"

# Set explicitly
export EMBEDDING_CHECK="true"
export ABSTENTION_CHECK_ENABLED="true"
export KARENINA_ASYNC_ENABLED="true"

# Restart Python session
```

### Environment variables not being read

**Issue**: You set an env var but it's not being used.

**Solution**: Environment variables are only read if not explicitly provided in code. If you pass an explicit argument, it takes precedence. This is by design.

```python
# This will NOT use EMBEDDING_CHECK env var
config = VerificationConfig(
    ...,
    embedding_check_enabled=False  # Explicit arg takes precedence
)

# This WILL use EMBEDDING_CHECK env var (if set)
config = VerificationConfig(...)  # No explicit arg
```

### Preset not overriding environment variables

**Issue**: You load a preset but env vars seem to be used instead.

**Solution**: Presets should always override env vars. If this isn't working, check:
1. The preset file actually contains the fields
2. The preset JSON is valid
3. You're not passing explicit arguments after loading the preset

```python
# Correct: Preset values will override env vars
config = VerificationConfig.from_preset(Path("preset.json"))

# Incorrect: Explicit args override preset
config = VerificationConfig.from_preset(Path("preset.json"))
config.embedding_check_enabled = True  # This overrides preset
```

---

## Best Practices

### API Keys
- Store in environment variables or `.env` files
- **Never** commit API keys to version control
- Use different keys for dev and production
- Rotate keys regularly

### Model Selection
- Use `gpt-4.1-mini` for development (fast, cost-effective)
- Use `gpt-4o` for production (higher quality)
- Set `temperature=0.0` for deterministic benchmarking
- Test with multiple models for comparison

### Features
- Enable `EMBEDDING_CHECK` for better recall on fuzzy matches
- Enable `ABSTENTION_CHECK_ENABLED` for safety/refusal testing
- Enable `rubric_enabled` for detailed evaluation
- Document which features are enabled in your project

### Execution
- Use `async_enabled=True` with `async_max_workers=2-4` for typical workloads
- Increase `async_max_workers` for bulk processing (watch API rate limits)
- Set `async_enabled=False` for debugging or when order matters

### Database
- Use descriptive database names (`genomics_benchmark.db`, not `test.db`)
- Organize databases by project or domain
- Back up databases regularly
- Use absolute paths for production deployments

### Configuration Management
- Use `.env` files for global defaults
- Use `VerificationConfig` for operation-specific settings
- Save common configurations as presets (see [Configuration Presets](advanced/presets.md))
- Document your configuration choices in project README

---

## Next Steps

- **[Quick Start](quickstart.md)** - Create your first benchmark
- **[Running Verification](using-karenina/verification.md)** - Complete verification guide
- **[Configuration Presets](advanced/presets.md)** - Save and load configurations
- **[Advanced Features](advanced/)** - Deep-dive into specific features
- **[API Reference](api-reference.md)** - Complete API documentation
