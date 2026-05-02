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
| `--interface` | `TEXT` | — | Model interface (required without `--preset`). Values: `langchain`, `openrouter`, `openai_endpoint`, `claude_agent_sdk`, `claude_tool`, `langchain_deep_agents`, `manual`, `taskeval` |

### General Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--replicate-count` | `INTEGER` | — | Number of replicates per verification task |
| `--evaluation-mode` | `TEXT` | `template_only` | Evaluation mode: `template_only`, `template_and_rubric`, or `rubric_only` |

### Feature Flags

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--abstention / --no-abstention` | flag pair | None (use preset/default) | Enable or disable abstention detection (stage 5) |
| `--sufficiency / --no-sufficiency` | flag pair | None (use preset/default) | Enable or disable trace sufficiency detection (stage 6) |
| `--embedding-check / --no-embedding-check` | flag pair | None (use preset/default) | Enable or disable embedding similarity check (stage 9) |
| `--deep-judgment / --no-deep-judgment` | flag pair | None (use preset/default) | Enable or disable deep judgment for templates (stage 10) |

Feature flags are tri-state: passing `--flag` enables the feature, `--no-flag` disables it, and passing neither preserves the preset default (or the built-in default of `False` when no preset is used).

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
| `--embedding-threshold` | `FLOAT` | None (defers to env var or `0.85`) | Embedding similarity threshold (`0.0`--`1.0`). When omitted, respects `EMBEDDING_CHECK_THRESHOLD` env var; falls back to `0.85`. |
| `--embedding-model` | `TEXT` | None (defers to env var or `all-MiniLM-L6-v2`) | Embedding model name. When omitted, respects `EMBEDDING_CHECK_MODEL` env var; falls back to `all-MiniLM-L6-v2`. |

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

### Replay Store

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--replay` | `PATH` | — | Path to a replay store JSON (built by `VerificationResultSet.to_replay_store()`). The store is loaded at startup and assigned to `VerificationConfig.replay_store`; lookups during the run hit the store first and the pipeline short-circuits to the canned traces on matching keys. Misses follow the `replay_parse_on_hydration_mismatch` policy (`fall_through` by default, which runs live), so the run can mix replayed and fresh execution. |

See the [Replay Store](../../advanced-pipeline/replay-store.md) advanced page for the keying scheme and capture/hydration semantics.

### Progressive Save and Resume

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--progressive-save` | flag | `False` | Enable incremental saving to `.results.jsonl` and `.state` sidecar files for crash recovery. Requires `--output`. |
| `--resume` | `PATH` | — | Resume from a `.state` file. Loads config from state; other config options are ignored. Resume is triple-level: completed (question, answering-model, parsing-model, replicate) tuples are skipped. |

---

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Verification completed successfully |
| `1` | Error: no templates found, invalid arguments, benchmark load failure, partial batch failure (`VerificationBatchError`), or unexpected error |
| `130` | Interrupted by user (Ctrl+C) |

If some questions fail while others succeed, the executor raises `VerificationBatchError` with both `partial_results` and `errors`. The CLI catches this, writes any partial results to the output file, and exits with code `1`. In parallel mode, a timeout also triggers `VerificationBatchError`.

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
    - `<output>.results.jsonl` — Append-only, one completed result per line (O(1) write per task)
    - `<output>.state` — Job state including config snapshot, benchmark path, task manifest, and the set of completed triples

2. **On crash or interrupt**: Both sidecars are retained. The `.state` file records which triples have completed; the `.results.jsonl` file holds the results themselves.

3. **Resume**: Pass the `.state` file path to `--resume`. The command reloads config from state, merges the completed triple set into `skip_triples`, and runs only the pending triples. All other CLI config options are ignored when resuming.

4. **Finalize**: On full completion, the final export is assembled from the JSONL sidecar, written to the output path, and both sidecars are deleted. On partial failure (`VerificationBatchError`), the sidecars are retained so `--resume` picks up only the failed triples.

Resume is **triple-level**: the unit of work is `(question_id, answering_canonical_key, parsing_canonical_key, replicate)`. Multi-model / multi-replicate fan-outs skip only the completed tuples, not whole questions.

Progressive save is also available from the Python API via `Benchmark.run_verification(sink=ProgressiveFileSink(...))` and `Benchmark.resume_verification(state_path)`. See the [Progressive Save tutorial](../../workflows/running-verification/progressive-save.md).

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
  --answering-model claude-haiku-4-5
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

### Disable a preset feature

```bash
# Preset has abstention_enabled=true; override it off
karenina verify checkpoint.jsonld --preset default.json --no-abstention
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
  --answering-model claude-haiku-4-5 --output results_haiku.json

karenina verify checkpoint.jsonld --preset default.json \
  --answering-model claude-sonnet-4-5 --output results_sonnet.json
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

## Interactive Modes: Basic vs Advanced

`--interactive` launches a step-by-step wizard that builds a [VerificationConfig](../configuration/verification-config.md) through prompts. The `--mode` flag selects how many configuration knobs the wizard exposes (default: `basic`).

**Basic mode** (`--mode basic`) walks through:

1. Question selection (subset of finished templates)
2. Replicate count
3. Feature configuration: evaluation mode, abstention, sufficiency, embedding check, deep-judgment template mode
4. Answering models (one or more)
5. Parsing models (one or more)
6. Optional preset save
7. Progress bar / progressive save toggles

**Advanced mode** (`--mode advanced`) runs the same flow plus an "Advanced Configuration" block before model collection. The advanced block adds prompts for:

- Filtering specific rubric trait names (when rubrics are enabled)
- Tuning deep-judgment template parameters (max excerpts, fuzzy threshold, retry attempts, search-based validation, search tool) when the template deep-judgment mode is not `disabled`
- Configuring rubric deep-judgment (mode, global excerpt toggle, per-trait defaults, search settings, custom config)
- [Few-shot](../../notebooks/core_concepts/few-shot.ipynb) prompting configuration
- Async execution settings (enabled flag, max workers)

Per-model prompts are also richer in advanced mode: MCP tool configuration is offered for the answering model only when `--mode advanced` is set and the chosen interface is not `manual`.

Source: `karenina/src/karenina/cli/interactive.py:78` (`build_config_interactively`).

---

## Related

- [Running Verification](../../notebooks/running-verification/basic-verification.ipynb) — Step-by-step guide for common verification workflows
- [Basic Verification (Python API)](../../notebooks/running-verification/basic-verification.ipynb) — Programmatic verification alternative
- [VerificationConfig Reference](../configuration/verification-config.md) — All configuration fields
- [Configuration Hierarchy](../../workflows/configuration/index.md) — How CLI, presets, and env vars interact
- [Full Evaluation](../../notebooks/running-verification/full-evaluation.ipynb) — Preset loading and overrides
