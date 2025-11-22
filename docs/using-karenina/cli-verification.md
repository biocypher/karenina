# CLI Verification

Command-line interface for running benchmark verifications without the GUI.

## Overview

The Karenina CLI provides a streamlined way to run verification workflows from the command line, ideal for:

- **Automated testing**: Integrate verifications into CI/CD pipelines
- **Batch processing**: Run verifications on multiple benchmarks
- **Remote execution**: Run verifications on servers without GUI access
- **Quick testing**: Rapidly test configurations during development

**Quick Navigation:**

- [Quick Start](#quick-start) - Installation and basic examples
- [Command Structure](#command-structure) - Main commands overview
- [Verify Command](#verify-command) - Core verification command details
- [CLI Options Reference](#cli-options-reference) - Complete option documentation
- [Usage Examples](#usage-examples) - Common workflows and patterns
- [Preset Management](#preset-management) - List, show, and delete presets
- [Output Formats](#output-formats) - JSON and CSV export options
- [Configuration Hierarchy](#configuration-hierarchy) - Understanding precedence rules
- [Error Handling](#error-handling) - Common errors and solutions
- [Progress Monitoring](#progress-monitoring) - Real-time verification tracking
- [Best Practices](#best-practices) - Tips for effective CLI usage
- [CI/CD Integration](#integration-with-cicd) - Automation and pipeline setup
- [Limitations](#limitations) - Known constraints and workarounds

---

## Quick Start

```bash
# Install karenina with CLI support
pip install karenina

# Run verification with a preset
karenina verify checkpoint.jsonld --preset default.json --questions 0,1 --verbose

# Run with interactive mode (guided configuration)
karenina verify checkpoint.jsonld --interactive --mode basic

# Run with explicit configuration
karenina verify checkpoint.jsonld \
  --interface langchain \
  --answering-model gpt-4.1-mini --answering-provider openai \
  --parsing-model gpt-4.1-mini --parsing-provider openai

# Evaluate pre-generated traces
karenina verify checkpoint.jsonld \
  --interface manual \
  --manual-traces traces/my_traces.json \
  --parsing-model gpt-4.1-mini --parsing-provider openai
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

**Configuration Hierarchy:**
The CLI supports flexible configuration with clear precedence: **CLI flags > Preset values > Environment variables > Defaults**

See [Usage Examples](#usage-examples) below for detailed workflows.

### CLI Options Reference

#### Configuration Sources

| Option | Type | Description | Default | Required? |
|--------|------|-------------|---------|-----------|
| `--preset PATH` | Path | Path to preset configuration file | - | No |
| `--interactive` | Flag | Enable interactive configuration builder | false | No |
| `--mode MODE` | Choice | Interactive mode: `basic` or `advanced` | basic | No |

#### Output and Filtering

| Option | Type | Description | Default | Required? |
|--------|------|-------------|---------|-----------|
| `--output PATH` | Path | Output file path (`.json` or `.csv`) | - | No |
| `--questions SPEC` | String | Question indices: `0,1,2` or `0-5` or `0-2,5` | - | No |
| `--question-ids IDS` | String | Comma-separated question IDs | - | No |
| `--verbose` | Flag | Show progress bar with real-time updates | false | No |

#### Model Configuration

| Option | Type | Description | Default | Required? |
|--------|------|-------------|---------|-----------|
| `--interface TYPE` | Choice | Interface type: `langchain`, `openrouter`, `openai_endpoint`, `manual` | - | Yes (without preset) |
| `--answering-model NAME` | String | Answering model name (e.g., `gpt-4o-mini`) | - | Yes (without preset, not needed for manual) |
| `--answering-provider PROVIDER` | String | Provider for langchain interface (e.g., `openai`) | - | Yes (without preset, langchain only) |
| `--answering-id ID` | String | Model ID for tracking | answering-1 | No |
| `--parsing-model NAME` | String | Parsing model name | - | Yes (without preset) |
| `--parsing-provider PROVIDER` | String | Provider for langchain interface | - | Yes (without preset, langchain only) |
| `--parsing-id ID` | String | Model ID for tracking | parsing-1 | No |
| `--temperature FLOAT` | Float | Model temperature (0.0-2.0) | 0.1 | No |
| `--manual-traces PATH` | Path | JSON file with pre-generated traces | - | Yes (when `--interface manual`) |

#### General Settings

| Option | Type | Description | Default | Required? |
|--------|------|-------------|---------|-----------|
| `--evaluation-mode MODE` | Choice | `template_only`, `template_and_rubric`, `rubric_only` | template_only | No |
| `--replicate-count N` | Integer | Number of replicates per verification | 1 | No |
| `--no-async` | Flag | Disable async execution (enabled by default) | false | No |
| `--async-workers N` | Integer | Number of parallel workers | 2 | No |

#### Feature Flags

| Option | Type | Description | Default | Required? |
|--------|------|-------------|---------|-----------|
| `--abstention` | Flag | Enable abstention detection | false | No |
| `--deep-judgment` | Flag | Enable deep judgment analysis for templates | false | No |
| `--deep-judgment-rubric-mode MODE` | Choice | Deep judgment mode for rubric traits: `disable_all`, `enable_all`, `use_trait_config` | use_trait_config | No |
| `--embedding-check` | Flag | Enable embedding similarity check | false | No |
| `--embedding-threshold FLOAT` | Float | Embedding similarity threshold (0.0-1.0) | 0.85 | No |
| `--embedding-model NAME` | String | Embedding model name | all-MiniLM-L6-v2 | No |

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
  --evaluation-mode template_and_rubric \
  --deep-judgment \
  --embedding-check \
  --abstention \
  --questions 0-5
```

#### 5. Enable Deep Judgment for Rubrics

Deep judgment for rubrics provides evidence-based evaluation by extracting supporting excerpts for trait judgments.

**Use Case 1: Enable for All Rubric Traits**

Override individual trait settings and enable deep judgment globally:

```bash
karenina verify checkpoint.jsonld \
  --preset default.json \
  --evaluation-mode template_and_rubric \
  --deep-judgment-rubric-mode enable_all \
  --questions 0-5
```

**Use Case 2: Disable for All Rubric Traits**

Disable deep judgment even for traits configured with `deep_judgment_enabled=True`:

```bash
karenina verify checkpoint.jsonld \
  --preset default.json \
  --evaluation-mode template_and_rubric \
  --deep-judgment-rubric-mode disable_all \
  --questions 0-5
```

**Use Case 3: Respect Individual Trait Configuration (Default)**

Only use deep judgment for traits that have `deep_judgment_enabled=True`:

```bash
karenina verify checkpoint.jsonld \
  --preset default.json \
  --evaluation-mode template_and_rubric \
  --deep-judgment-rubric-mode use_trait_config \
  --questions 0-5
```

Or omit the flag entirely (defaults to `use_trait_config`):

```bash
karenina verify checkpoint.jsonld \
  --preset default.json \
  --evaluation-mode template_and_rubric \
  --questions 0-5
```

**Deep Judgment Modes Explained:**

| Mode | Behavior | When to Use |
|------|----------|-------------|
| `enable_all` | Forces deep judgment ON for ALL rubric traits | Testing, comprehensive evidence gathering, audit requirements |
| `disable_all` | Forces deep judgment OFF for ALL rubric traits | Quick testing, cost optimization, speed priority |
| `use_trait_config` | Respects individual trait `deep_judgment_enabled` settings | Normal operation, selective deep judgment on key traits |

**Performance Impact:**

- **enable_all**: Highest cost and latency, maximum transparency
- **disable_all**: Lowest cost and latency, fastest evaluation
- **use_trait_config**: Balanced approach, optimize per trait

**Example with Multiple Advanced Features:**

```bash
# Full-featured verification with deep judgment for both templates and rubrics
karenina verify checkpoint.jsonld \
  --preset default.json \
  --evaluation-mode template_and_rubric \
  --deep-judgment \
  --deep-judgment-rubric-mode enable_all \
  --embedding-check \
  --abstention \
  --output results.json \
  --verbose
```

For more details on deep judgment for rubrics, see the [Rubrics documentation](./rubrics.md#deep-judgment-for-llm-rubric-traits).

#### 6. Range and Mixed Question Selection

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

#### 7. Interactive Configuration

```bash
# Basic mode - essential parameters only
karenina verify checkpoint.jsonld --interactive --mode basic

# Advanced mode - full configuration options
karenina verify checkpoint.jsonld --interactive --mode advanced
```

**Interactive Flow:**

1. **Question Selection**: Display table of available questions, select subset
2. **Replicate Count**: Number of verification replicates
3. **Feature Configuration**:
   - **Evaluation Mode**: Choose how to evaluate answers
     - `template_only`: Verify structured output only (fastest)
     - `template_and_rubric`: Verify structure + evaluate quality criteria
     - `rubric_only`: Evaluate quality criteria only (no structure required)
   - Enable abstention detection, embedding check, deep judgment
4. **Models**: Configure answering and parsing models
5. **Advanced Settings** (advanced mode only): Rubric trait filtering, deep judgment settings, few-shot config
6. **Save Preset**: Optionally save configuration as preset
7. **Output Configuration**: Prompt to configure result export (file format and filename)
8. **Run Verification**: Execute verification with progress display and save results

#### 8. Manual Trace Evaluation

Evaluate pre-generated LLM responses without making live API calls:

```bash
# Evaluate pre-recorded traces
karenina verify checkpoint.jsonld \
  --interface manual \
  --manual-traces traces/my_traces.json \
  --parsing-model gpt-4.1-mini \
  --parsing-provider openai \
  --questions 0-5 \
  --output results.csv
```

**Trace File Format** (`traces/my_traces.json`):

```json
{
  "936dbc8755f623c951d96ea2b03e13bc": "Pre-generated answer for question 1",
  "8f2e2b1e4d5c6a7b8c9d0e1f2a3b4c5d": "Pre-generated answer for question 2"
}
```

**Use Cases:**

- Evaluate answers from external systems
- Compare different answer generation approaches
- Test verification/rubric systems with controlled answers
- Integrate pre-recorded experiment results

**Important Notes:**

- Question hashes are MD5 hashes of question text
- Export CSV mapper from benchmark to get question hashes
- Only answering model uses manual interface; parsing model still needs LLM
- Traces bypass answer generation but verification still runs normally

**Example Output:**

```
Loading benchmark...
✓ Loaded benchmark: Karenina LLM Benchmark Checkpoint
  Total questions: 12
Loading manual traces from traces/my_traces.json...
✓ Loaded 6 manual trace(s)
Building configuration from CLI arguments
Filtered to 6 question(s) by indices

Starting verification...
  Questions: 6
  Answering models: 1 (manual)
  Parsing models: 1
  Replicates: 1

Verification Summary:
  Total: 6
  Passed: 5
  Failed: 1
  Duration: 3.42s

✓ Verification complete!
```

For complete documentation on the manual trace system, including trace formats and programmatic usage, see the [Manual Traces documentation](../advanced/manual-traces.md).

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

#### Manual Traces File Not Found
```
Error: Manual traces file not found: traces/my_traces.json
```
**Fix**: Verify trace file path is correct and file exists

#### Missing --manual-traces with Manual Interface
```
Configuration errors:
  • --manual-traces is required when --interface manual.
    Provide a JSON file mapping question hashes to answer traces.
```
**Fix**: Add `--manual-traces PATH` flag when using `--interface manual`

#### Invalid Trace File Format
```
Error: Invalid trace file format: expected JSON object (dict), got list
```
**Fix**: Ensure trace file is a JSON object with `{question_hash: trace_string}` format

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

## Related Documentation

- [Verification](./verification.md) - Core verification concepts and workflow
- [Preset Management](../advanced/presets.md) - Saving and managing configurations
- [Export Formats](./saving-loading.md#exporting-verification-results) - Understanding output formats
- [Configuration](../configuration.md) - VerificationConfig details
- [API Reference](../api-reference.md) - Programmatic API usage
