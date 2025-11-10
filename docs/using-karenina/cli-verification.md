# CLI Verification

Command-line interface for running benchmark verifications without the GUI.

## Overview

The Karenina CLI provides a streamlined way to run verification workflows from the command line, ideal for:

- **Automated testing**: Integrate verifications into CI/CD pipelines
- **Batch processing**: Run verifications on multiple benchmarks
- **Remote execution**: Run verifications on servers without GUI access
- **Quick testing**: Rapidly test configurations during development

## Quick Start

```bash
# Install karenina with CLI support
pip install karenina

# Run verification with a preset
karenina verify checkpoint.jsonld --preset default.json --questions 0,1 --verbose

# Run with CLI arguments only (no preset)
karenina verify checkpoint.jsonld --answering-model gpt-4.1-mini --parsing-model gpt-4.1-mini

# Interactive mode
karenina verify checkpoint.jsonld --interactive --mode basic
```

## Command Structure

### Main Commands

```bash
karenina verify BENCHMARK [OPTIONS]    # Run verification
karenina preset list                    # List available presets
karenina preset show NAME               # Show preset configuration
karenina preset delete NAME             # Delete a preset
```

## Verify Command

The `verify` command runs verification on a benchmark with flexible configuration options.

### Basic Usage

```bash
karenina verify BENCHMARK_PATH [OPTIONS]
```

**Required Arguments:**
- `BENCHMARK_PATH`: Path to benchmark JSON-LD file

### Configuration Modes

The CLI supports three configuration modes with a clear priority hierarchy:

**Priority: Interactive > Preset + CLI Args > CLI Args Only**

#### 1. Preset-Based Configuration

Use a saved preset configuration file:

```bash
karenina verify checkpoint.jsonld --preset default.json --questions 0,1
```

#### 2. CLI Arguments with Preset Override

Override specific preset values with CLI arguments:

```bash
# Use preset but override the answering model
karenina verify checkpoint.jsonld \
  --preset default.json \
  --answering-model gpt-4o \
  --temperature 0.2
```

**Hierarchy**: CLI flags > preset > env variables > defaults

#### 3. CLI Arguments Only

Build configuration entirely from CLI arguments:

```bash
karenina verify checkpoint.jsonld \
  --answering-model gpt-4.1-mini \
  --parsing-model gpt-4.1-mini \
  --rubric \
  --evaluation-mode template_and_rubric
```

#### 4. Interactive Mode

Build configuration through guided prompts:

```bash
# Basic mode (essential parameters only)
karenina verify checkpoint.jsonld --interactive --mode basic

# Advanced mode (all parameters)
karenina verify checkpoint.jsonld --interactive --mode advanced
```

### CLI Options Reference

#### Configuration Sources

| Option | Type | Description |
|--------|------|-------------|
| `--preset PATH` | Path | Path to preset configuration file |
| `--interactive` | Flag | Enable interactive configuration builder |
| `--mode MODE` | Choice | Interactive mode: `basic` or `advanced` (default: `basic`) |

#### Output and Filtering

| Option | Type | Description |
|--------|------|-------------|
| `--output PATH` | Path | Output file path (`.json` or `.csv`) |
| `--questions SPEC` | String | Question indices: `0,1,2` or `0-5` or `0-2,5` |
| `--question-ids IDS` | String | Comma-separated question IDs |
| `--verbose` | Flag | Show progress bar with real-time updates |

#### Model Configuration

| Option | Type | Description |
|--------|------|-------------|
| `--answering-model NAME` | String | Answering model name (e.g., `gpt-4o-mini`) |
| `--answering-provider PROVIDER` | String | Provider for langchain interface (e.g., `openai`) |
| `--answering-id ID` | String | Model ID for tracking |
| `--parsing-model NAME` | String | Parsing model name |
| `--parsing-provider PROVIDER` | String | Provider for langchain interface |
| `--parsing-id ID` | String | Model ID for tracking |
| `--temperature FLOAT` | Float | Model temperature (0.0-2.0, default: 0.1) |
| `--interface TYPE` | Choice | Interface type: `langchain`, `openrouter`, `openai_endpoint` |

#### General Settings

| Option | Type | Description |
|--------|------|-------------|
| `--replicate-count N` | Integer | Number of replicates per verification (default: 1) |

#### Feature Flags

| Option | Type | Description |
|--------|------|-------------|
| `--rubric / --no-rubric` | Boolean | Enable/disable rubric evaluation |
| `--abstention / --no-abstention` | Boolean | Enable/disable abstention detection |
| `--embedding-check / --no-embedding-check` | Boolean | Enable/disable embedding similarity check |
| `--deep-judgment / --no-deep-judgment` | Boolean | Enable/disable deep judgment analysis |

#### Advanced Settings

| Option | Type | Description |
|--------|------|-------------|
| `--evaluation-mode MODE` | Choice | `template_only`, `template_and_rubric`, `rubric_only` |
| `--embedding-threshold FLOAT` | Float | Embedding similarity threshold (0.0-1.0) |
| `--embedding-model NAME` | String | Embedding model name |
| `--async / --no-async` | Boolean | Enable/disable async execution |
| `--async-workers N` | Integer | Number of parallel workers |

## Usage Examples

### Common Workflows

#### 1. Quick Test with Preset

```bash
karenina verify checkpoint.jsonld \
  --preset default.json \
  --questions 0,1 \
  --output results.json \
  --verbose
```

**Output:**
```
Loading benchmark...
✓ Loaded benchmark: Karenina LLM Benchmark Checkpoint
  Total questions: 12
Loading preset...
✓ Loaded preset from: default.json
Filtered to 2 question(s) by indices

Starting verification...
  Questions: 2
  Answering models: 1
  Parsing models: 1
  Replicates: 1

  Verifying questions... ✓ ━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:05

Exporting results to results.json...
✓ Results exported to results.json

Verification Summary:
  Total: 2
  Passed: 2
  Failed: 0
  Duration: 5.64s

✓ Verification complete!
```

#### 2. Override Preset Model

```bash
# Use preset configuration but test with a different model
karenina verify checkpoint.jsonld \
  --preset default.json \
  --answering-model gpt-4o \
  --questions 0-2
```

#### 3. CLI Arguments Only (No Preset)

!!! warning "Required Parameters Without Preset"
    When running without a preset, you **must** specify:

    - **Interface**: `--interface` (langchain, openrouter, or openai_endpoint)
    - **Model names**: `--answering-model` and `--parsing-model`
    - **Provider** (for langchain): `--answering-provider` and `--parsing-provider`

    **Example for langchain interface:**
    ```bash
    karenina verify checkpoint.jsonld \
      --interface langchain \
      --answering-model gpt-4.1-mini \
      --answering-provider openai \
      --parsing-model gpt-4.1-mini \
      --parsing-provider openai \
      --output results.csv
    ```

    **Example for openrouter interface:**
    ```bash
    karenina verify checkpoint.jsonld \
      --interface openrouter \
      --answering-model openai/gpt-4.1-mini \
      --parsing-model openai/gpt-4.1-mini \
      --output results.csv
    ```

**Optional parameters** (use defaults if not specified):
- Temperature: 0.1 (deterministic evaluation)
- Replicate count: 1
- Advanced features: All disabled
- Evaluation mode: template_only

#### 4. Enable Advanced Features

```bash
karenina verify checkpoint.jsonld \
  --preset default.json \
  --rubric \
  --evaluation-mode template_and_rubric \
  --deep-judgment \
  --embedding-check \
  --questions 0-5
```

#### 5. Range and Mixed Question Selection

```bash
# Questions 0-2 and question 5
karenina verify checkpoint.jsonld \
  --preset default.json \
  --questions "0-2,5" \
  --output results.json

# Specific question IDs
karenina verify checkpoint.jsonld \
  --preset default.json \
  --question-ids "question_1,question_3,question_7" \
  --output results.csv
```

#### 6. Interactive Configuration

```bash
# Basic mode - essential parameters only
karenina verify checkpoint.jsonld --interactive --mode basic

# Advanced mode - full configuration options
karenina verify checkpoint.jsonld --interactive --mode advanced
```

**Interactive Flow:**

1. **Question Selection**: Display table of available questions, select subset
2. **Replicate Count**: Number of verification replicates
3. **Feature Flags**: Enable rubric, abstention, embedding check, deep judgment
4. **Models**: Configure answering and parsing models
5. **Advanced Settings** (advanced mode only): Evaluation mode, deep judgment settings, few-shot config
6. **Save Preset**: Optionally save configuration as preset

## Preset Management

### List Presets

```bash
karenina preset list
```

**Output:**
```
      Available Presets
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Name          ┃ Modified   ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ default       │ 2025-11-03 │
│ gpt-oss       │ 2025-11-08 │
│ gpt-oss-tools │ 2025-11-08 │
└───────────────┴────────────┘

Total: 3 preset(s)
```

### Show Preset Configuration

```bash
karenina preset show gpt-oss
```

**Output:**
```
Preset: gpt-oss
Path: /path/to/presets/gpt-oss.json

Configuration:
{
  "answering_models": [
    {
      "id": "answering-1",
      "model_name": "gpt-oss",
      "interface": "openai_endpoint",
      "temperature": 0.1,
      ...
    }
  ],
  ...
}

Summary:
  Answering models: 1
  Parsing models: 1
  Replicates: 1
  Rubric enabled: False
  Abstention enabled: False
  Embedding check enabled: False
  Deep judgment enabled: False
```

### Delete Preset

```bash
karenina preset delete old-config
```

**Output:**
```
About to delete preset: /path/to/presets/old-config.json
Are you sure? [y/N]: y
✓ Preset deleted successfully!
```

## Output Formats

### JSON Output

Comprehensive export with all verification metadata (43+ fields):

```bash
karenina verify checkpoint.jsonld \
  --preset default.json \
  --questions 0,1 \
  --output results.json
```

**JSON Structure:**
```json
{
  "metadata": {
    "job_id": "uuid",
    "run_name": "cli-verification",
    "config": { ... },
    "total_questions": 2,
    "successful_count": 2,
    "failed_count": 0,
    "start_time": 1234567890.0,
    "end_time": 1234567895.0
  },
  "results": {
    "verification-id-1": {
      "verification_id": "...",
      "question_id": "q1",
      "question_text": "...",
      "verify_result": true,
      "answering_model": "openai/gpt-4.1-mini",
      "parsing_model": "openai/gpt-4.1-mini",
      "execution_time": 2.5,
      "deep_judgment_result": { ... },
      "usage_tracking": { ... },
      ...
    }
  }
}
```

### CSV Output

Tabular format with 43 columns including deep judgment, usage tracking, and agent metrics:

```bash
karenina verify checkpoint.jsonld \
  --preset default.json \
  --questions 0-5 \
  --output results.csv
```

**CSV Columns Include:**

- Core: `verification_id`, `question_id`, `question_text`, `verify_result`
- Models: `answering_model`, `parsing_model`, `replicate_number`
- Timing: `execution_time`, `timestamp`
- Deep Judgment: `deep_judgment_enabled`, `deep_judgment_pass`, `excerpts_found`, `search_used`
- Usage: `answering_prompt_tokens`, `answering_completion_tokens`, `parsing_prompt_tokens`, `parsing_completion_tokens`
- Abstention: `abstention_detected`, `abstention_confidence`
- Embedding: `embedding_check_pass`, `embedding_similarity`
- Errors: `error_message`, `completed_without_errors`

## Environment Variables

### `KARENINA_PRESETS_DIR`

Override the default preset directory:

```bash
export KARENINA_PRESETS_DIR=/path/to/my/presets
karenina preset list
```

**Default Locations:**

1. `KARENINA_PRESETS_DIR` environment variable (if set)
2. `benchmark_presets/` in current directory (default)

## Configuration Hierarchy

When both preset and CLI arguments are provided, the hierarchy is:

**CLI flags > Preset values > Environment variables > Built-in defaults**

### Example

```bash
# Preset has: answering_model = gpt-4.1-mini, temperature = 0.1
# CLI overrides: answering_model = gpt-4o
# Result: answering_model = gpt-4o, temperature = 0.1 (from preset)

karenina verify checkpoint.jsonld \
  --preset default.json \
  --answering-model gpt-4o
```

## Error Handling

### Fail-Fast Validation

The CLI validates inputs before running verification to avoid wasted API calls:

```bash
karenina verify checkpoint.jsonld \
  --preset default.json \
  --output results.txt
```

**Output:**
```
Error: Invalid output format: .txt. Output file must have .json or .csv extension.
```

### Common Errors

#### Invalid Output Extension
```
Error: Invalid output format: .txt. Output file must have .json or .csv extension.
```
**Fix**: Use `.json` or `.csv` extension

#### Preset Not Found
```
Error: Preset 'nonexistent' not found in /path/to/presets.
Use 'karenina preset list' to see available presets.
```
**Fix**: Check preset name with `karenina preset list`

#### Missing Parent Directory
```
Error: Parent directory does not exist: /nonexistent/path
```
**Fix**: Create parent directory or use existing path

#### Invalid Question Indices
```
Error: Index out of range: 15 (total questions: 12)
```
**Fix**: Use valid indices within range 0-(n-1)

#### No Finished Templates
```
Warning: No finished templates found in benchmark
```
**Fix**: Ensure benchmark has finished template generation

## Progress Monitoring

### Verbose Mode

Enable real-time progress updates with `--verbose`:

```bash
karenina verify checkpoint.jsonld \
  --preset default.json \
  --questions 0-5 \
  --verbose
```

**Progress Bar Features:**

- Spinner animation
- Progress bar with percentage
- Pass/fail indicators (✓/✗)
- Time remaining estimate
- Real-time status updates

**Example Output:**
```
Verifying questions... ✓ ━━━━━━━━━━━━━━━━━━━━━━━━━ 75% 0:00:05
```

## Best Practices

### 1. Use Presets for Consistency

Save common configurations as presets to ensure consistency:

```bash
# Create preset interactively
karenina verify checkpoint.jsonld --interactive --mode basic
# (follow prompts and save as preset)

# Reuse preset
karenina verify other-benchmark.jsonld --preset my-config
```

### 2. Test with Subset First

Test configurations on a small subset before running full verification:

```bash
# Test with 2 questions first
karenina verify checkpoint.jsonld --preset default.json --questions 0,1

# If successful, run full verification
karenina verify checkpoint.jsonld --preset default.json
```

### 3. Use Verbose for Long Runs

Enable progress monitoring for verifications that take more than a few seconds:

```bash
karenina verify large-benchmark.jsonld \
  --preset default.json \
  --verbose
```

### 4. Override for Quick Tests

Use preset with overrides for quick model comparisons:

```bash
# Test different models with same configuration
karenina verify checkpoint.jsonld --preset default.json --answering-model gpt-4o-mini --questions 0-2
karenina verify checkpoint.jsonld --preset default.json --answering-model gpt-4o --questions 0-2
```

### 5. Export Results for Analysis

Always export results for later analysis:

```bash
# JSON for programmatic access
karenina verify checkpoint.jsonld \
  --preset default.json \
  --output results-$(date +%Y%m%d).json

# CSV for spreadsheet analysis
karenina verify checkpoint.jsonld \
  --preset default.json \
  --output results-$(date +%Y%m%d).csv
```

## Integration with CI/CD

### Example GitHub Actions Workflow

```yaml
name: Benchmark Verification
on: [push]

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install karenina
        run: pip install karenina

      - name: Run verification
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          karenina verify benchmark.jsonld \
            --preset ci-config.json \
            --output results.csv \
            --verbose

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: verification-results
          path: results.csv
```

## Limitations

### CLI vs GUI

The CLI is optimized for automation but has some limitations compared to the GUI:

| Feature | CLI | GUI |
|---------|-----|-----|
| Multiple answering models | Single model only | Multiple models |
| Multiple parsing models | Single model only | Multiple models |
| Few-shot configuration | Interactive mode only | Full UI editor |
| MCP tools | Interactive mode only | Full UI configuration |
| Result visualization | Export only | Live charts and tables |
| Preset creation | Interactive mode | Full form editor |

**Workaround**: Use interactive mode for complex configurations, then reuse saved presets.

## Related Documentation

- [Verification](./verification.md) - Core verification concepts and workflow
- [Preset Management](../advanced/presets.md) - Saving and managing configurations
- [Export Formats](./saving-loading.md#exporting-verification-results) - Understanding output formats
- [Configuration](../configuration.md) - VerificationConfig details
- [API Reference](../api-reference.md) - Programmatic API usage

## Implementation Notes

**Location**: `karenina/src/karenina/cli/`

**Key Modules:**
- [verify.py](../../karenina/src/karenina/cli/verify.py) - Main verify command with config builder
- [preset.py](../../karenina/src/karenina/cli/preset.py) - Preset management commands
- [interactive.py](../../karenina/src/karenina/cli/interactive.py) - Interactive configuration builder
- [utils.py](../../karenina/src/karenina/cli/utils.py) - Helper functions and utilities

**Testing**: [test_cli_utils.py](../../karenina/tests/test_cli_utils.py) - 34 unit tests covering all utilities
