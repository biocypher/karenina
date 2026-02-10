# karenina verify

Run verification on a benchmark checkpoint file.

```
karenina verify [OPTIONS] [BENCHMARK_PATH]
```

The `verify` command is the primary way to evaluate LLM outputs against benchmark criteria. It loads a checkpoint, configures the verification pipeline, runs evaluation, and exports results.

---

## Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `BENCHMARK_PATH` | `PATH` | Yes (unless `--resume`) | Path to the benchmark JSON-LD checkpoint file |

---

## Options

### Configuration Sources

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--preset` | `PATH` | — | Path to a preset configuration JSON file |
| `--interactive` | flag | `False` | Launch interactive configuration wizard |
| `--mode` | `TEXT` | `basic` | Interactive mode: `basic` or `advanced` |

Configuration priority: interactive selection > CLI arguments + preset > preset only > defaults.

### Output and Filtering

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output` | `PATH` | — | Output file path (`.json` or `.csv`). If omitted, you will be prompted. |
| `--questions` | `TEXT` | — | Question indices: comma-separated (`0,1,2`) or range (`5-10`) |
| `--question-ids` | `TEXT` | — | Comma-separated question IDs (URN format) |
| `--verbose` | flag | `False` | Show a progress bar during verification |

### Model Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--answering-model` | `TEXT` | — | Answering model name (required without `--preset`) |
| `--answering-provider` | `TEXT` | — | Answering model provider for LangChain (required without `--preset`) |
| `--answering-id` | `TEXT` | `answering-1` | Answering model ID |
| `--parsing-model` | `TEXT` | — | Parsing model name (required without `--preset`) |
| `--parsing-provider` | `TEXT` | — | Parsing model provider for LangChain (required without `--preset`) |
| `--parsing-id` | `TEXT` | `parsing-1` | Parsing model ID |
| `--temperature` | `FLOAT` | — | Model temperature (`0.0`–`2.0`) |
| `--interface` | `TEXT` | — | Model interface (required without `--preset`). Values: `langchain`, `openrouter`, `openai_endpoint`, `claude_agent_sdk`, `claude_tool`, `manual` |

### General Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--replicate-count` | `INTEGER` | — | Number of replicates per verification task |
| `--evaluation-mode` | `TEXT` | `template_only` | Evaluation mode: `template_only`, `template_and_rubric`, or `rubric_only` |

### Feature Flags

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--abstention` | flag | `False` | Enable abstention detection (stage 5) |
| `--sufficiency` | flag | `False` | Enable trace sufficiency detection (stage 6) |
| `--embedding-check` | flag | `False` | Enable embedding similarity check (stage 9) |
| `--deep-judgment` | flag | `False` | Enable deep judgment for templates (stage 10) |

### Deep Judgment — Rubric Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--deep-judgment-rubric-mode` | `TEXT` | `disabled` | Mode: `disabled`, `enable_all`, `use_checkpoint`, or `custom` |
| `--deep-judgment-rubric-excerpts` | flag | `True` | Enable excerpt extraction for rubric traits (`enable_all` mode) |
| `--deep-judgment-rubric-max-excerpts` | `INTEGER` | `7` | Maximum excerpts per rubric trait (`enable_all` mode) |
| `--deep-judgment-rubric-fuzzy-threshold` | `FLOAT` | `0.8` | Fuzzy match threshold for excerpt validation (`0.0`–`1.0`) |
| `--deep-judgment-rubric-retry-attempts` | `INTEGER` | `2` | Retry attempts for excerpt extraction |
| `--deep-judgment-rubric-search` | flag | `False` | Enable search-based hallucination detection for rubric excerpts |
| `--deep-judgment-rubric-search-tool` | `TEXT` | `tavily` | Search tool for rubric hallucination detection |
| `--deep-judgment-rubric-config` | `PATH` | — | Path to custom deep judgment config JSON (`custom` mode) |

### Embedding Check Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--embedding-threshold` | `FLOAT` | `0.85` | Embedding similarity threshold (`0.0`–`1.0`) |
| `--embedding-model` | `TEXT` | `all-MiniLM-L6-v2` | Embedding model name |

### Trace Filtering (MCP)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--use-full-trace-for-template` / `--use-final-message-for-template` | flag | `False` | Use full MCP agent trace (`True`) or only the final AI message (`False`) for template parsing |
| `--use-full-trace-for-rubric` / `--use-final-message-for-rubric` | flag | `True` | Use full MCP agent trace (`True`) or only the final AI message (`False`) for rubric evaluation |

### Async Execution

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--no-async` | flag | `False` | Disable async execution (run sequentially) |
| `--async-workers` | `INTEGER` | `2` or `KARENINA_ASYNC_MAX_WORKERS` | Number of async workers |

### Manual Traces

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--manual-traces` | `PATH` | — | JSON file with manual traces (`{question_hash: trace_string}`). Required when `--interface manual`. |

### Progressive Save and Resume

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--progressive-save` | flag | `False` | Enable incremental saving to `.tmp` and `.state` files for crash recovery. Requires `--output`. |
| `--resume` | `PATH` | — | Resume from a `.state` file. Loads config from state; other config options are ignored. |

---

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Verification completed successfully |
| `1` | Error — no templates found, invalid arguments, benchmark load failure, or unexpected error |
| `130` | Interrupted by user (Ctrl+C) |

When interrupted with Ctrl+C during a progressive save run, the command prints a resume hint:

```
Interrupted by user.
To resume: karenina verify --resume results.json
```

---

## Progressive Save and Resume

Progressive save enables crash recovery for long-running verification jobs. When enabled, results are saved incrementally as each question completes.

### How it works

1. **Start with progressive save**: The command creates two sidecar files next to the output path:
    - `<output>.tmp` — Accumulated results so far
    - `<output>.state` — Job state including config, benchmark path, and task manifest

2. **On crash or interrupt**: The `.state` file preserves which tasks have completed. The `.tmp` file holds partial results.

3. **Resume**: Pass the `.state` file path to `--resume`. The command reloads config from state, identifies pending tasks, and runs only those. All other CLI options are ignored when resuming.

4. **Finalize**: When all tasks complete, results are written to the output file and the `.tmp`/`.state` sidecar files are cleaned up.

### Check progress

Use `karenina verify-status` to inspect a state file:

```bash
karenina verify-status results.json.state
```

See [verify-status](verify-status.md) for full details.

---

## Examples

### Basic verification with a preset

```bash
karenina verify checkpoint.jsonld --preset default.json
```

### Override preset model

```bash
karenina verify checkpoint.jsonld --preset default.json \
  --answering-model gpt-4o
```

### CLI arguments only (no preset)

```bash
karenina verify checkpoint.jsonld \
  --answering-model claude-haiku-4-5 \
  --parsing-model claude-haiku-4-5 \
  --interface langchain \
  --answering-provider anthropic \
  --parsing-provider anthropic
```

### Specific questions by index

```bash
karenina verify checkpoint.jsonld --preset default.json --questions 0,1,2
```

### Specific questions by range

```bash
karenina verify checkpoint.jsonld --preset default.json --questions 5-10
```

### Enable feature flags

```bash
karenina verify checkpoint.jsonld --preset default.json \
  --abstention --sufficiency --deep-judgment
```

### Template and rubric evaluation

```bash
karenina verify checkpoint.jsonld --preset default.json \
  --evaluation-mode template_and_rubric
```

### Export to CSV with progress bar

```bash
karenina verify checkpoint.jsonld --preset default.json \
  --output results.csv --verbose
```

### Manual traces

```bash
karenina verify checkpoint.jsonld \
  --interface manual \
  --manual-traces traces/my_traces.json \
  --parsing-model claude-haiku-4-5 \
  --parsing-provider anthropic
```

### Progressive save and resume

```bash
# Start with progressive save
karenina verify checkpoint.jsonld --preset default.json \
  --output results.json --progressive-save

# Resume after interruption
karenina verify --resume results.json.state
```

### Model comparison (multiple runs)

```bash
karenina verify checkpoint.jsonld --preset default.json \
  --answering-model gpt-4o --output results_gpt4o.json

karenina verify checkpoint.jsonld --preset default.json \
  --answering-model claude-sonnet-4-5-20250514 --output results_claude.json
```

### Full-featured run

```bash
karenina verify checkpoint.jsonld --preset default.json \
  --evaluation-mode template_and_rubric \
  --abstention --sufficiency --embedding-check --deep-judgment \
  --replicate-count 3 --output results.json --verbose
```

### Interactive mode

```bash
karenina verify checkpoint.jsonld --interactive
karenina verify checkpoint.jsonld --interactive --mode advanced
```

---

## Related

- [CLI Verification Tutorial](../06-running-verification/cli.md) — Step-by-step guide for common CLI workflows
- [Python API](../06-running-verification/python-api.md) — Programmatic verification alternative
- [VerificationConfig Reference](../10-configuration-reference/verification-config.md) — All configuration fields
- [Configuration Hierarchy](../03-configuration/index.md) — How CLI, presets, and env vars interact
- [Using Presets](../06-running-verification/using-presets.md) — Preset loading and overrides
