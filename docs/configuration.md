# Configuration

This guide explains how to configure Karenina for your benchmarking workflows.

## What is Configuration?

**Configuration** in Karenina controls:
- LLM provider selection (OpenAI, Google, Anthropic, OpenRouter)
- Model selection for answering and parsing
- Feature toggles (embedding check, abstention detection)
- Execution behavior (parallel vs sequential)
- Database location

Configuration uses a combination of environment variables and programmatic settings to provide flexibility across different use cases.

## Configuration Methods

Karenina supports two configuration approaches:

1. **Environment Variables**: Global settings that affect all operations
2. **Programmatic Configuration**: Per-operation settings via Python code

Both methods work together with clear precedence rules.

## Environment Variables

### LLM API Keys

Required for accessing LLM providers:

```bash
# OpenAI (for GPT models)
export OPENAI_API_KEY="sk-..."

# Google (for Gemini models)
export GOOGLE_API_KEY="AIza..."

# Anthropic (for Claude models)
export ANTHROPIC_API_KEY="sk-ant-..."

# OpenRouter (for unified access)
export OPENROUTER_API_KEY="sk-or-..."
```

**Security Best Practices**:
- Never commit API keys to version control
- Use `.env` files (added to `.gitignore`)
- Rotate keys regularly
- Use separate keys for development and production

### Database Configuration

Optional database location:

```bash
# Default: dbs/karenina.db
export DB_PATH="dbs/genomics.db"
```

### Feature Toggles

Enable optional features:

```bash
# Embedding Check (semantic similarity fallback)
export EMBEDDING_CHECK="true"
export EMBEDDING_CHECK_MODEL="all-MiniLM-L6-v2"
export EMBEDDING_CHECK_THRESHOLD="0.85"

# Abstention Detection (refusal detection)
export ABSTENTION_CHECK_ENABLED="true"
```

### Execution Control

Control parallel execution:

```bash
# Enable parallel verification (default: true)
export KARENINA_ASYNC_ENABLED="true"
export KARENINA_ASYNC_MAX_WORKERS="2"

# Disable for sequential execution
export KARENINA_ASYNC_ENABLED="false"
```

## Programmatic Configuration

### Model Configuration

Configure LLM models using `ModelConfig`:

```python
from karenina.schemas import ModelConfig

# Configure answering model
answering_model = ModelConfig(
    model_name="gpt-4.1-mini",
    model_provider="openai",
    temperature=0.0
)

# Configure parsing model (can be different)
parsing_model = ModelConfig(
    model_name="gpt-4.1-mini",
    model_provider="openai",
    temperature=0.0
)
```

**Supported Providers**:
- `"openai"` - OpenAI GPT models
- `"google"` - Google Gemini models
- `"anthropic"` - Anthropic Claude models
- `"openrouter"` - OpenRouter unified access

**Common Models**:
- `"gpt-4.1-mini"` (default) - Fast, cost-effective
- `"gpt-4o"` - More capable, higher cost
- `"claude-3-5-sonnet-20241022"` - Anthropic's flagship
- `"gemini-2.0-flash-exp"` - Google's latest

### Verification Configuration

Configure verification behavior:

```python
from karenina.schemas import VerificationConfig, ModelConfig

model_config = ModelConfig(
    model_name="gpt-4.1-mini",
    model_provider="openai",
    temperature=0.0
)

config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    replicate_count=3,
    rubric_enabled=True,
    deep_judgment_enabled=False,
    abstention_enabled=False
)
```

## Default Configuration

Karenina uses sensible defaults:

```python
# Default LLM Model
model_name = "gpt-4.1-mini"
model_provider = "openai"
temperature = 0.0

# Default Features
embedding_check = False
abstention_check = False
deep_judgment = False

# Default Execution
async_enabled = True
max_workers = 2

# Default Database
db_path = "dbs/karenina.db"
```

## Configuration Precedence

Settings are applied in this order (highest to lowest):

1. **Programmatic Configuration** (in code) - Highest priority
2. **Environment Variables** - Medium priority
3. **Default Values** (hardcoded) - Lowest priority

**Example**:
```python
import os

# 1. Set environment variable
os.environ["EMBEDDING_CHECK"] = "true"

# 2. Environment variable takes effect globally
# All verifications will have embedding check enabled

# 3. Override for specific operation
config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    # This specific config ignores environment variable
)
```

## Common Configuration Scenarios

### Scenario 1: Change Default Model

Switch from gpt-4.1-mini to gpt-4o:

```python
from karenina import Benchmark
from karenina.schemas import ModelConfig, VerificationConfig

# Create benchmark
benchmark = Benchmark.create(
    name="Drug Target Benchmark",
    description="Testing drug target knowledge",
    version="1.0.0"
)

benchmark.add_question(
    question="What is the approved drug target of Venetoclax?",
    raw_answer="BCL2"
)

# Use GPT-4o instead of default
model_config = ModelConfig(
    model_name="gpt-4o",  # Changed from gpt-4.1-mini
    model_provider="openai",
    temperature=0.0
)

# Generate templates with new model
benchmark.generate_all_templates(model_config=model_config)

# Verify with same model
config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    replicate_count=3
)

results = benchmark.run_verification(config)
```

### Scenario 2: Enable Embedding Check

Add semantic similarity fallback:

```python
import os
from karenina import Benchmark
from karenina.schemas import ModelConfig, VerificationConfig

# Enable embedding check globally
os.environ["EMBEDDING_CHECK"] = "true"
os.environ["EMBEDDING_CHECK_MODEL"] = "all-MiniLM-L6-v2"
os.environ["EMBEDDING_CHECK_THRESHOLD"] = "0.85"

# Create benchmark
benchmark = Benchmark.create(
    name="Genomics Knowledge Benchmark",
    description="Testing LLM knowledge of genomics",
    version="1.0.0"
)

benchmark.add_question(
    question="Which chromosome contains the HBB gene?",
    raw_answer="Chromosome 11"
)

# Generate and verify (embedding check active)
model_config = ModelConfig(
    model_name="gpt-4.1-mini",
    model_provider="openai",
    temperature=0.0
)

benchmark.generate_all_templates(model_config=model_config)

config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    replicate_count=3
)

results = benchmark.run_verification(config)

# Check if embedding check rescued any failures
override_count = sum(
    1 for r in results.values() if r.embedding_override_applied
)
print(f"Embedding check rescued {override_count} failures")
```

### Scenario 3: Multi-Model Comparison

Compare multiple models:

```python
from karenina import Benchmark
from karenina.schemas import ModelConfig, VerificationConfig

# Create benchmark
benchmark = Benchmark.create(
    name="Model Comparison Benchmark",
    description="Comparing GPT-4.1-mini vs GPT-4o",
    version="1.0.0"
)

benchmark.add_question(
    question="What is the approved drug target of Venetoclax?",
    raw_answer="BCL2"
)

# Configure multiple models
gpt_mini = ModelConfig(
    model_name="gpt-4.1-mini",
    model_provider="openai",
    temperature=0.0
)

gpt_4o = ModelConfig(
    model_name="gpt-4o",
    model_provider="openai",
    temperature=0.0
)

# Generate templates once
benchmark.generate_all_templates(model_config=gpt_mini)

# Verify with multiple models
config = VerificationConfig(
    answering_models=[gpt_mini, gpt_4o],  # Compare both
    parsing_models=[gpt_mini],
    replicate_count=3
)

results = benchmark.run_verification(config)

# Analyze per-model performance
for question_id, result in results.items():
    print(f"Question: {result.question}")
    print(f"GPT-4.1-mini: {result.verify_result}")
    print(f"Models tested: {result.models_used}")
```

### Scenario 4: Development vs Production

Different configurations for different environments:

```python
import os
from karenina import Benchmark
from karenina.schemas import ModelConfig, VerificationConfig

# Development configuration
if os.getenv("ENV") == "development":
    model_config = ModelConfig(
        model_name="gpt-4.1-mini",  # Cheaper model
        model_provider="openai",
        temperature=0.0
    )
    replicate_count = 1  # Faster
    rubric_enabled = False  # Skip for speed

# Production configuration
else:
    model_config = ModelConfig(
        model_name="gpt-4o",  # Better model
        model_provider="openai",
        temperature=0.0
    )
    replicate_count = 5  # More replicates
    rubric_enabled = True  # Full evaluation

# Create benchmark
benchmark = Benchmark.create(
    name="Drug Target Benchmark",
    description="Testing drug target knowledge",
    version="1.0.0"
)

benchmark.add_question(
    question="What is the approved drug target of Venetoclax?",
    raw_answer="BCL2"
)

# Generate and verify with environment-specific config
benchmark.generate_all_templates(model_config=model_config)

config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    replicate_count=replicate_count,
    rubric_enabled=rubric_enabled
)

results = benchmark.run_verification(config)
```

### Scenario 5: Custom Database Location

Use a specific database file:

```python
import os
from pathlib import Path
from karenina import Benchmark

# Set custom database path
os.environ["DB_PATH"] = "dbs/genomics_production.db"

# Or use programmatic path
db_path = Path("dbs/genomics_production.db")

# Create benchmark
benchmark = Benchmark.create(
    name="Genomics Benchmark",
    description="Production genomics evaluation",
    version="1.0.0"
)

benchmark.add_question(
    question="What is the approved drug target of Venetoclax?",
    raw_answer="BCL2"
)

# Save to custom database
benchmark.save_to_db(db_path)

# Load from custom database
loaded = Benchmark.load_from_db(
    db_path,
    "Genomics Benchmark"
)
```

## Environment Variable Reference

### LLM API Keys

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key | For OpenAI models |
| `GOOGLE_API_KEY` | Google API key | For Google models |
| `ANTHROPIC_API_KEY` | Anthropic API key | For Anthropic models |
| `OPENROUTER_API_KEY` | OpenRouter API key | For OpenRouter models |

### Feature Toggles

| Variable | Description | Default | Values |
|----------|-------------|---------|--------|
| `EMBEDDING_CHECK` | Enable semantic fallback | `false` | `true`, `false` |
| `EMBEDDING_CHECK_MODEL` | SentenceTransformer model | `all-MiniLM-L6-v2` | Model name |
| `EMBEDDING_CHECK_THRESHOLD` | Similarity threshold | `0.85` | `0.0` - `1.0` |
| `ABSTENTION_CHECK_ENABLED` | Enable refusal detection | `false` | `true`, `false` |

### Execution Control

| Variable | Description | Default | Values |
|----------|-------------|---------|--------|
| `KARENINA_ASYNC_ENABLED` | Enable parallel execution | `true` | `true`, `false` |
| `KARENINA_ASYNC_MAX_WORKERS` | Number of parallel workers | `2` | Integer |

### Database

| Variable | Description | Default |
|----------|-------------|---------|
| `DB_PATH` | SQLite database path | `dbs/karenina.db` |

## Setting Environment Variables

### Using .env File (Recommended)

Create a `.env` file in your project root:

```bash
# .env
OPENAI_API_KEY="sk-..."
GOOGLE_API_KEY="AIza..."
DB_PATH="dbs/karenina.db"
EMBEDDING_CHECK="true"
EMBEDDING_CHECK_THRESHOLD="0.85"
```

**Important**: Add `.env` to `.gitignore` to avoid committing secrets!

### Using Shell Export

```bash
# Temporary (current session only)
export OPENAI_API_KEY="sk-..."
export EMBEDDING_CHECK="true"

# Permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.bashrc
source ~/.bashrc
```

### Using Python

```python
import os

# Set before importing karenina
os.environ["OPENAI_API_KEY"] = "sk-..."
os.environ["EMBEDDING_CHECK"] = "true"

from karenina import Benchmark
```

## Configuration Best Practices

### API Keys

**Do**:
- Store in environment variables or `.env` files
- Use different keys for dev and production
- Rotate keys regularly
- Add `.env` to `.gitignore`

**Don't**:
- Commit API keys to version control
- Share keys between team members
- Use production keys in development

### Model Selection

**Do**:
- Use `gpt-4.1-mini` for development (fast, cheap)
- Use `gpt-4o` for production (better quality)
- Test with multiple models for comparison
- Set `temperature=0.0` for deterministic results

**Don't**:
- Use expensive models unnecessarily
- Set high temperatures for benchmarking
- Mix different models without documentation

### Feature Toggles

**Do**:
- Enable `EMBEDDING_CHECK` for better recall
- Enable `ABSTENTION_CHECK_ENABLED` for safety testing
- Document which features are enabled
- Test with features on and off

**Don't**:
- Enable all features by default (performance impact)
- Forget to document enabled features
- Use different settings across environments without reason

### Database Configuration

**Do**:
- Use descriptive database names
- Organize databases by project or domain
- Back up databases regularly
- Use absolute paths for clarity

**Don't**:
- Share databases across unrelated projects
- Store databases in temporary directories
- Use generic names like `test.db`

## Checking Configuration

### View Current Settings

```python
import os

# Check API keys (masked)
openai_key = os.getenv("OPENAI_API_KEY")
print(f"OpenAI key: {'Set' if openai_key else 'Not set'}")

# Check feature toggles
embedding_check = os.getenv("EMBEDDING_CHECK", "false")
print(f"Embedding check: {embedding_check}")

# Check database path
db_path = os.getenv("DB_PATH", "dbs/karenina.db")
print(f"Database: {db_path}")
```

### Verify Configuration

```python
from karenina.schemas import ModelConfig

# Create model config
model_config = ModelConfig(
    model_name="gpt-4.1-mini",
    model_provider="openai",
    temperature=0.0
)

# Verify settings
print(f"Model: {model_config.model_name}")
print(f"Provider: {model_config.model_provider}")
print(f"Temperature: {model_config.temperature}")
```

## Troubleshooting

### Issue: API Key Not Found

**Error**: `ValueError: OPENAI_API_KEY not set`

**Solutions**:
```bash
# Check if key is set
echo $OPENAI_API_KEY

# Set key
export OPENAI_API_KEY="sk-..."

# Or in Python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
```

### Issue: Wrong Model Selected

**Error**: Model not behaving as expected

**Solutions**:
```python
# Verify model configuration
model_config = ModelConfig(
    model_name="gpt-4.1-mini",  # Check spelling
    model_provider="openai",     # Check provider
    temperature=0.0              # Check temperature
)

# Print config for debugging
print(model_config.model_dump())
```

### Issue: Features Not Working

**Error**: Embedding check or abstention not activating

**Solutions**:
```bash
# Verify environment variables are set
env | grep -E "EMBEDDING|ABSTENTION"

# Set explicitly
export EMBEDDING_CHECK="true"
export ABSTENTION_CHECK_ENABLED="true"

# Restart Python session to pick up changes
```

### Issue: Database Conflicts

**Error**: `sqlite3.OperationalError: database is locked`

**Solutions**:
```python
# Use different database paths for parallel operations
import os
os.environ["DB_PATH"] = "dbs/worker_1.db"

# Or use absolute paths
from pathlib import Path
db_path = Path("/absolute/path/to/dbs/karenina.db")
```

## Complete Configuration Example

This example shows all configuration options:

```python
import os
from pathlib import Path
from karenina import Benchmark
from karenina.schemas import ModelConfig, VerificationConfig

# 1. Set environment variables
os.environ["OPENAI_API_KEY"] = "sk-..."
os.environ["DB_PATH"] = "dbs/genomics.db"
os.environ["EMBEDDING_CHECK"] = "true"
os.environ["EMBEDDING_CHECK_THRESHOLD"] = "0.85"
os.environ["ABSTENTION_CHECK_ENABLED"] = "true"
os.environ["KARENINA_ASYNC_ENABLED"] = "true"
os.environ["KARENINA_ASYNC_MAX_WORKERS"] = "4"

# 2. Create benchmark
benchmark = Benchmark.create(
    name="Genomics Knowledge Benchmark",
    description="Comprehensive genomics evaluation",
    version="1.0.0"
)

# 3. Add questions
benchmark.add_question(
    question="What is the approved drug target of Venetoclax?",
    raw_answer="BCL2",
    author={"name": "Pharma Curator"}
)

benchmark.add_question(
    question="Which chromosome contains the HBB gene?",
    raw_answer="Chromosome 11",
    author={"name": "Genetics Curator"}
)

# 4. Configure models
model_config = ModelConfig(
    model_name="gpt-4.1-mini",
    model_provider="openai",
    temperature=0.0
)

# 5. Generate templates
benchmark.generate_all_templates(model_config=model_config)

# 6. Configure verification
config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    replicate_count=3,
    rubric_enabled=True,
    deep_judgment_enabled=False
)

# 7. Run verification
results = benchmark.run_verification(config)

# 8. Analyze results
success_count = sum(1 for r in results.values() if r.verify_result)
total_count = len(results)
print(f"Success rate: {success_count}/{total_count}")

# 9. Save to database
db_path = Path(os.getenv("DB_PATH", "dbs/karenina.db"))
benchmark.save_to_db(db_path)
```

## Related Documentation

- **Quick Start**: Basic usage without configuration details
- **Advanced Features**: Features that require configuration
  - [Embedding Check](advanced/embedding-check.md) - Semantic fallback configuration
  - [Abstention Detection](advanced/abstention-detection.md) - Refusal detection setup
  - [Deep-Judgment](advanced/deep-judgment.md) - Multi-stage parsing configuration
- **API Reference**: Complete ModelConfig and VerificationConfig documentation
- **Troubleshooting**: Solutions for common configuration issues

## Summary

Karenina configuration uses:

1. **Environment Variables** for global settings (API keys, feature toggles)
2. **Programmatic Configuration** for per-operation settings (models, parameters)
3. **Clear Precedence** (code > environment > defaults)

**Default model**: `gpt-4.1-mini` with `temperature=0.0` for deterministic benchmarking.

Configure based on your needs:
- Development → Fast, cheap models with minimal features
- Production → Better models with full feature set
- Research → Multiple models with all features enabled
