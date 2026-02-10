# Running Verification from the CLI

The `karenina verify` command runs verification directly from the terminal. It supports all the same configuration options available in the Python API — model selection, feature flags, evaluation modes, progressive save, and resume.

This page covers practical CLI usage for running verification. For the complete option reference, see [karenina verify Reference](../09-cli-reference/verify.md).

---

## Basic Usage

The simplest way to run verification uses a preset file that contains your model and configuration choices:

```bash
karenina verify checkpoint.jsonld --preset my-config.json
```

This loads the benchmark from `checkpoint.jsonld`, applies the configuration from the preset, and runs verification on all questions.

Without a preset, you must specify model configuration explicitly:

```bash
karenina verify checkpoint.jsonld \
  --answering-model claude-haiku-4-5 \
  --parsing-model claude-haiku-4-5 \
  --interface langchain \
  --answering-provider anthropic \
  --parsing-provider anthropic
```

---

## Common Options

### Model Configuration

| Option | Description | Default |
|--------|-------------|---------|
| `--answering-model` | Model name for generating answers | Required without preset |
| `--parsing-model` | Model name for parsing responses | Required without preset |
| `--interface` | Adapter interface (`langchain`, `openrouter`, `openai_endpoint`, `claude_agent_sdk`, `claude_tool`, `manual`) | Required without preset |
| `--answering-provider` | Provider for langchain interface | Required for langchain |
| `--parsing-provider` | Provider for langchain interface | Required for langchain |
| `--temperature` | Model temperature (0.0–2.0) | From preset or default |

### Question Selection

Run verification on a subset of questions instead of the full benchmark:

```bash
# By index (0-based)
karenina verify checkpoint.jsonld --preset config.json --questions 0,1,2

# By index range
karenina verify checkpoint.jsonld --preset config.json --questions 5-10

# By question ID
karenina verify checkpoint.jsonld --preset config.json \
  --question-ids "urn:uuid:question-abc123,urn:uuid:question-def456"
```

### Feature Flags

Enable optional verification features:

```bash
# Abstention detection — detect when models refuse to answer
karenina verify checkpoint.jsonld --preset config.json --abstention

# Sufficiency detection — check if responses contain enough information
karenina verify checkpoint.jsonld --preset config.json --sufficiency

# Embedding similarity check
karenina verify checkpoint.jsonld --preset config.json --embedding-check

# Deep judgment for templates
karenina verify checkpoint.jsonld --preset config.json --deep-judgment

# Deep judgment for rubric traits
karenina verify checkpoint.jsonld --preset config.json \
  --deep-judgment-rubric-mode enable_all
```

### Evaluation Mode

Control whether verification uses templates, rubrics, or both:

```bash
# Template only (default)
karenina verify checkpoint.jsonld --preset config.json \
  --evaluation-mode template_only

# Template and rubric evaluation
karenina verify checkpoint.jsonld --preset config.json \
  --evaluation-mode template_and_rubric

# Rubric only (no template parsing)
karenina verify checkpoint.jsonld --preset config.json \
  --evaluation-mode rubric_only
```

### Output

Save results to a file:

```bash
# JSON output
karenina verify checkpoint.jsonld --preset config.json --output results.json

# CSV output
karenina verify checkpoint.jsonld --preset config.json --output results.csv
```

### Async Execution

By default, verification runs asynchronously. To disable or tune:

```bash
# Disable async (run sequentially)
karenina verify checkpoint.jsonld --preset config.json --no-async

# Set number of parallel workers
karenina verify checkpoint.jsonld --preset config.json --async-workers 4
```

---

## Overriding Preset Values

When using a preset, you can override individual settings with CLI flags. CLI arguments take precedence over preset values:

```bash
# Load preset but override the answering model
karenina verify checkpoint.jsonld --preset production.json \
  --answering-model gpt-4o

# Load preset but enable additional features
karenina verify checkpoint.jsonld --preset production.json \
  --abstention --deep-judgment
```

This follows the standard [configuration hierarchy](../03-configuration/index.md): CLI arguments > Preset values > Environment variables > Defaults.

---

## Progressive Save and Resume

For long-running verification jobs, progressive save writes intermediate results after each question so you can recover from crashes:

```bash
# Enable progressive save
karenina verify checkpoint.jsonld --preset config.json --progressive-save
```

This creates two files alongside the benchmark:
- `checkpoint.jsonld.tmp` — Partial results saved after each question
- `checkpoint.jsonld.state` — Job state (configuration, completed questions, progress)

To resume an interrupted job:

```bash
# Resume from state file (config loaded from state, other flags ignored)
karenina verify --resume checkpoint.jsonld.state
```

!!! note
    When resuming, the configuration is loaded from the state file. Other CLI configuration options are ignored to ensure consistency with the original run.

Check job status without resuming:

```bash
karenina verify-status checkpoint.jsonld.state
karenina verify-status checkpoint.jsonld.state --show-tasks
karenina verify-status checkpoint.jsonld.state --show-questions
```

---

## Manual Traces

Run verification on pre-recorded model responses instead of calling live LLMs:

```bash
karenina verify checkpoint.jsonld \
  --interface manual \
  --manual-traces traces/my_traces.json \
  --parsing-model claude-haiku-4-5 \
  --parsing-provider anthropic
```

The manual traces JSON file maps question hashes to response strings. The parsing model is still called to extract structured answers from the traces.

See [Manual Interface](manual-interface.md) for the trace file format and workflow details.

---

## Interactive Mode

Interactive mode guides you through configuration with prompts instead of requiring all options upfront:

```bash
# Basic interactive mode
karenina verify checkpoint.jsonld --interactive

# Advanced interactive mode (more options)
karenina verify checkpoint.jsonld --interactive --mode advanced
```

---

## Examples

### Quick evaluation with a preset

```bash
karenina verify benchmark.jsonld --preset default.json --verbose
```

### Compare two models

```bash
# Run with model A
karenina verify benchmark.jsonld --preset config.json \
  --answering-model gpt-4o --output results-gpt4o.json

# Run with model B
karenina verify benchmark.jsonld --preset config.json \
  --answering-model claude-sonnet-4-5-20250929 --interface claude_agent_sdk \
  --output results-claude.json
```

### Full-featured verification

```bash
karenina verify benchmark.jsonld --preset production.json \
  --evaluation-mode template_and_rubric \
  --abstention \
  --sufficiency \
  --embedding-check \
  --deep-judgment \
  --progressive-save \
  --output results.json \
  --verbose
```

### Run on specific questions with replicates

```bash
karenina verify benchmark.jsonld --preset config.json \
  --questions 0,1,2 \
  --replicate-count 3 \
  --output results.csv
```

---

## Next Steps

- [Python API](python-api.md) — Run verification programmatically with more control
- [VerificationConfig](verification-config.md) — Configure all verification settings
- [Using Presets](using-presets.md) — Save and reuse configurations
- [Response Quality Checks](response-quality-checks.md) — Abstention and sufficiency detection
- [karenina verify Reference](../09-cli-reference/verify.md) — Complete option reference
