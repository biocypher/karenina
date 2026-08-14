---
name: karenina-cli
description: >
  Run karenina evaluations from the command line. Use when the user wants to
  run verification via terminal commands instead of Python scripts. Covers
  environment setup, preset management, running karenina verify with flags,
  progressive save/resume, model comparison, and result export. Fastest path
  from checkpoint to results.
---

# CLI Evaluation Workflow

Run karenina evaluations entirely from the terminal. This skill gets you from
a benchmark checkpoint to verification results using CLI commands.

**Full CLI reference**: The `using-karenina` skill contains complete command
documentation in `references/reference/cli/`. Consult it for exhaustive flag
tables and edge cases.

---

## Prerequisites

1. karenina installed (`uv pip install -e karenina` from the monorepo, or `pip install karenina`)
2. A benchmark checkpoint file (`.jsonld`) with questions and templates attached
3. API key for your LLM provider set in the environment:

```bash
# Pick the provider(s) you need
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."
```

Verify the CLI is available:

```bash
karenina --help
```

---

## Quick Start (3 Commands)

The fastest path from checkpoint to results:

```bash
# 1. Create a preset (one-time setup)
cat > my-preset.json << 'EOF'
{
  "config": {
    "answering_models": [
      {"id": "answerer", "model_provider": "anthropic", "model_name": "claude-sonnet-4-5"}
    ],
    "parsing_models": [
      {"id": "judge", "model_provider": "anthropic", "model_name": "claude-haiku-4-5"}
    ],
    "replicate_count": 1
  }
}
EOF

# 2. Run verification
karenina verify checkpoint.jsonld --preset my-preset.json --output results.json

# 3. Done. Results are in results.json
```

---

## Step-by-Step Procedure

### Step 1: Check Environment

Confirm karenina is installed and your API key is set:

```bash
karenina --help
echo $ANTHROPIC_API_KEY  # or whichever provider you use
```

If using a workspace initialized with `karenina init`, the presets directory
and database paths are already configured. Otherwise, set them manually:

```bash
export KARENINA_PRESETS_DIR="./presets"
mkdir -p "$KARENINA_PRESETS_DIR"
```

### Step 2: Create or Select a Preset

Presets bundle all model and pipeline configuration into a reusable JSON file.
This is the recommended approach: configure once, run many times.

**Minimal preset** (answering + parsing model):

```json
{
  "config": {
    "answering_models": [
      {
        "id": "answerer",
        "model_provider": "anthropic",
        "model_name": "claude-sonnet-4-5"
      }
    ],
    "parsing_models": [
      {
        "id": "judge",
        "model_provider": "anthropic",
        "model_name": "claude-haiku-4-5"
      }
    ],
    "replicate_count": 1
  }
}
```

**Preset with features enabled**:

```json
{
  "config": {
    "answering_models": [
      {"id": "answerer", "model_provider": "anthropic", "model_name": "claude-sonnet-4-5"}
    ],
    "parsing_models": [
      {"id": "judge", "model_provider": "anthropic", "model_name": "claude-haiku-4-5"}
    ],
    "replicate_count": 3,
    "abstention_enabled": true,
    "evaluation_mode": "template_and_rubric"
  }
}
```

**List existing presets**:

```bash
karenina preset list
```

**Inspect a preset**:

```bash
karenina preset show my-preset
```

### Step 3: Run Verification

```bash
karenina verify checkpoint.jsonld --preset my-preset.json --output results.json
```

**Without a preset** (all config via CLI flags):

```bash
karenina verify checkpoint.jsonld \
  --answering-model claude-sonnet-4-5 \
  --answering-provider anthropic \
  --parsing-model claude-haiku-4-5 \
  --parsing-provider anthropic \
  --interface langchain \
  --output results.json
```

**Filter to specific questions**:

```bash
# By index (0-based)
karenina verify checkpoint.jsonld --preset my-preset.json --questions 0,1,2

# By range
karenina verify checkpoint.jsonld --preset my-preset.json --questions 5-10

# By question ID
karenina verify checkpoint.jsonld --preset my-preset.json \
  --question-ids "936dbc87,8f2e2b1e"
```

**Replay cached answers**: `--replay replay.json` short-circuits generation
with a `ReplayStore` JSON on matching keys.

### Step 4: Use Feature Flags

Feature flags enable optional pipeline stages. Each flag supports
`--flag` (enable) and `--no-flag` (disable), so you can override preset
defaults in either direction.

```bash
# Enable guards and deep judgment
karenina verify checkpoint.jsonld --preset my-preset.json \
  --abstention --sufficiency --deep-judgment

# Disable a feature that the preset enables
karenina verify checkpoint.jsonld --preset my-preset.json \
  --no-abstention

# Enable rubric evaluation (requires rubrics attached to the benchmark)
karenina verify checkpoint.jsonld --preset my-preset.json \
  --evaluation-mode template_and_rubric

# Enable embedding similarity check with custom threshold
karenina verify checkpoint.jsonld --preset my-preset.json \
  --embedding-check --embedding-threshold 0.90
```

**Flag summary**:

| Flag pair | Default | Purpose |
|-----------|---------|---------|
| `--abstention / --no-abstention` | Off | Detect model abstention |
| `--sufficiency / --no-sufficiency` | Off | Check trace sufficiency |
| `--embedding-check / --no-embedding-check` | Off | Embedding similarity check |
| `--deep-judgment / --no-deep-judgment` | Off | Deep judgment |

When neither `--flag` nor `--no-flag` is passed, the preset's value (or the
default) is preserved.

### Step 5: Export and Inspect Results

Results are written to the `--output` path in JSON or CSV format (determined
by file extension):

```bash
# JSON (full structured data)
karenina verify checkpoint.jsonld --preset my-preset.json --output results.json

# CSV (flat tabular format)
karenina verify checkpoint.jsonld --preset my-preset.json --output results.csv
```

Show a progress bar during verification:

```bash
karenina verify checkpoint.jsonld --preset my-preset.json \
  --output results.json --verbose
```

### Step 6: Compare Models

Run the same benchmark with different models and compare results:

```bash
# Run with Sonnet
karenina verify checkpoint.jsonld --preset my-preset.json \
  --answering-model claude-sonnet-4-5 --output results_sonnet.json

# Run with Haiku
karenina verify checkpoint.jsonld --preset my-preset.json \
  --answering-model claude-haiku-4-5 --output results_haiku.json

# Run with GPT
karenina verify checkpoint.jsonld --preset my-preset.json \
  --answering-model gpt-4.1 --answering-provider openai \
  --output results_gpt.json
```

CLI flags override the preset's model, so a single preset works across
providers. Only the flags you pass are overridden; everything else stays.

### Step 7: Progressive Save and Resume

For long-running jobs, enable progressive save to checkpoint after each
completed task. If the process is interrupted (crash, Ctrl+C, partial batch
failure), resume without re-running already-completed triples.

```bash
# Start with progressive save
karenina verify checkpoint.jsonld --preset my-preset.json \
  --output results.json --progressive-save

# Check progress of a running or interrupted job
karenina verify-status results.json.state

# Resume an interrupted job (config loaded from state file)
karenina verify --resume results.json.state
```

Sidecar layout next to `results.json`:

- `results.json.results.jsonl` — append-only, one completed result per line
- `results.json.state` — manifest, config snapshot, completed triple set

On full completion, both sidecars are removed and the final export is written.
Sidecars survive only when the run is *interrupted* — a crash or Ctrl+C exits
with code 130 without finalizing, so `--resume` picks up where it left off.
If the run finishes — even when some tasks failed and their failures were
captured as execution-error rows in the export — the export is finalized and
both sidecars are deleted; there is nothing to resume. For a completed export
that contains error rows, re-run just those rows with `karenina repair`
(see below), not `--resume`.

**Resume is triple-level.** The unit of work is
`(question_id, answering_canonical_key, parsing_canonical_key, replicate)`,
not the question. Multi-model / multi-replicate fan-outs skip only the
completed tuples, not whole questions. CLI flags (`--answering-model`,
`--preset`, feature flags) are ignored when `--resume` is passed; the entire
config is restored from the state file. The one exception is
`--manual-traces`: manual-interface runs must re-supply it on resume, because
the state file cannot store the traces (see Gotcha 8).

The same machinery is available from the Python API via
`Benchmark.run_verification(sink=ProgressiveFileSink(...))` and
`Benchmark.resume_verification(state_path)`. See the
[Progressive Save tutorial](../using-karenina/references/workflows/running-verification/progressive-save.md)
for code examples and `CompositeSink` composition (file + database).

## Repairing Failed Rows

Re-run only the failed rows of a completed run instead of the whole
benchmark. `repair` selects rows by failure group/category/stage (or by a
specific question/model/replicate), re-executes them, and splices the
replacements back into the export:

```bash
karenina repair results.json --benchmark checkpoint.jsonld --preset my-preset.json \
  --failure-group autofail --failure-stage verify_template

# One specific triple
karenina repair results.json --benchmark checkpoint.jsonld --preset my-preset.json \
  --question-id 936dbc87 --answerer-key answerer --parser-key judge --replicate 1

# Preview the selection without writing
karenina repair results.json --benchmark checkpoint.jsonld --preset my-preset.json \
  --all --dry-run
```

`--mode replay` (default) reuses recorded traces to avoid live API calls;
`--mode live` regenerates answers. Selection filters: `--question-id`,
`--answerer-key`, `--parser-key`, `--replicate`, `--failure-group`,
`--failure-category`, `--failure-stage`; `--all` selects every row;
`--dry-run` prints the selection without writing.

## Reproducible Run Directories

Write each run into a timestamped directory with a masked manifest instead of
a bare output file:

```bash
karenina verify checkpoint.jsonld --preset my-preset.json \
  --run-root runs --run-label otp-sonnet
```

`--run-root` creates a timestamped run directory, forces JSON output plus
progressive save, and writes a `RunManifest` with credentials masked
(API keys/secrets masked; numeric limits like `max_tokens` and timeouts are
preserved). `--run-label` names the run.

Python equivalent: `from karenina.benchmark import managed_run_directory` — a
context manager that records run status (running/interrupted/failed/completed)
automatically.

---

## Common Workflows

### Manual Traces (Offline Evaluation)

Evaluate pre-recorded LLM responses without making live API calls for the
answering model. A live parsing model is still required.

```bash
karenina verify checkpoint.jsonld \
  --interface manual \
  --manual-traces traces.json \
  --parsing-model claude-haiku-4-5 \
  --parsing-provider anthropic \
  --output results.json
```

The trace file maps question hashes to response strings:

```json
{
  "385a58a826db8f732c488dafa37433f8": "Aspirin inhibits COX enzymes.",
  "2a32bfdb18e3e9df6e86e9225bb22615": "Venetoclax targets BCL-2."
}
```

### Interactive Mode

Let the CLI walk you through configuration interactively:

```bash
karenina verify checkpoint.jsonld --interactive
karenina verify checkpoint.jsonld --interactive --mode advanced
```

### Full-Featured Run

```bash
karenina verify checkpoint.jsonld --preset my-preset.json \
  --evaluation-mode template_and_rubric \
  --abstention --sufficiency --embedding-check --deep-judgment \
  --replicate-count 3 \
  --output results.json --verbose --progressive-save
```

---

## Gotchas

1. **Config errors are reported before benchmark loading.** If you pass an
   invalid config (e.g., missing `--interface` without a preset), the error
   appears immediately, not after the benchmark loads.

2. **Feature flags are tri-state.** Passing `--abstention` enables it,
   `--no-abstention` disables it, and passing neither preserves the preset
   default. This means a preset with `abstention_enabled: true` keeps that
   setting unless you explicitly pass `--no-abstention`.

3. **Embedding defaults defer to environment variables.** When you do not pass
   `--embedding-threshold` or `--embedding-model`, the CLI does not override
   env vars (`EMBEDDING_CHECK_THRESHOLD`, `EMBEDDING_CHECK_MODEL`). Pass them
   explicitly to override.

4. **`--evaluation-mode` defaults to `template_only`.** Rubrics attached to
   your benchmark are NOT evaluated unless you pass
   `--evaluation-mode template_and_rubric` (or `rubric_only`) explicitly.
   There is no `auto` value and no auto-upgrade: attaching a rubric does not
   change the mode.

5. **Output format is determined by file extension.** `.json` produces
   structured JSON; `.csv` produces flat tabular output. Other extensions are
   rejected.

6. **`--resume` ignores all other config flags.** The entire config is
   restored from the state file. Model, preset, and feature flags on the
   command line are silently ignored. Exception: `--manual-traces` must be
   passed again for manual-interface runs (see Gotcha 8).

7. **Async is on by default.** Verification runs with 2 parallel workers.
   Use `--no-async` for sequential execution, or `--async-workers N` to tune
   parallelism. The `KARENINA_ASYNC_MAX_WORKERS` env var also works.

8. **`--resume` supports the manual interface via `--manual-traces`.**
   `ManualTraces` objects cannot be serialized into the state file's config
   snapshot, so the state is written without them. Pass the same
   `--manual-traces` file on resume: the CLI loads it against the benchmark
   recorded in the state and attaches it to the manual answering model
   before the config is revalidated, then resumes normally. Resuming a
   manual-interface state *without* `--manual-traces` still fails with
   `Value error, manual_traces is required when interface=manual` (on
   `answering_models.0`) before any work starts; the `.results.jsonl`
   sidecar still shows which triples completed.

**Also available** (run with `--help` for details): `karenina optimize`,
`optimize-history`, and `optimize-compare` (GEPA prompt optimization);
`karenina preset delete`; `karenina verify-status --json/-j`, `-q`, `-t`;
and on `karenina verify`: `--temperature`, `--answering-id`, `--parsing-id`,
`--deep-judgment-rubric-excerpts`, `--deep-judgment-rubric-search`,
`--use-full-trace-for-template`, `--use-full-trace-for-rubric`, and the
`--workspace-output-mode` / `--workspace-output-dir` /
`--workspace-output-exclude` sidecar-capture family.

---

## Key Commands

| Command | Purpose | Details |
|---------|---------|---------|
| `karenina verify` | Run verification | [Full reference](../using-karenina/references/reference/cli/verify.md) |
| `karenina verify-status` | Check progressive save progress | [Full reference](../using-karenina/references/reference/cli/verify-status.md) |
| `karenina preset list` | List presets | [Full reference](../using-karenina/references/reference/cli/preset.md) |
| `karenina preset show` | Inspect a preset | [Full reference](../using-karenina/references/reference/cli/preset.md) |
| `karenina init` | Initialize workspace | [Full reference](../using-karenina/references/reference/cli/init.md) |
| `karenina serve` | Start webapp server | [Full reference](../using-karenina/references/reference/cli/serve.md) |
| `karenina repair` | Repair failed rows of a run | `karenina repair SOURCE --benchmark B --preset P` |
| `karenina analyze-errors` | Materialize a run for an error-analyst agent | `karenina analyze-errors --results R --checkpoint C --out-dir D` |

**Full karenina docs**: The `using-karenina` skill contains the complete
karenina documentation in its `references/` directory. Consult it for API
details not covered here.
