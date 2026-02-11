# karenina verify-status

Inspect a progressive save state file and show job status.

```
karenina verify-status [OPTIONS] STATE_FILE
```

The `verify-status` command shows the progress of a verification job that was started with `--progressive-save`. It reads the `.state` sidecar file and displays completed and pending tasks, elapsed time, model configuration, and file status.

---

## Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `STATE_FILE` | `PATH` | Yes | Path to the `.state` file to inspect |

---

## Options

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--show-tasks` | `-t` | flag | `False` | Show detailed list of completed and pending task IDs |
| `--show-questions` | `-q` | flag | `False` | Show list of completed and pending question IDs |
| `--json` | `-j` | flag | `False` | Output status as JSON instead of formatted text |

---

## Default Output

Without any flags, the command displays a Rich-formatted summary with five sections:

1. **Header** — "Job Complete" (green) or "Job In Progress" (yellow)
2. **Progress bar** — Visual completion indicator with percentage and counts
3. **Summary table** — State file path, output path, benchmark path, completed/pending/total counts, timing
4. **Configuration table** — Answering models, parsing models, replicate count
5. **Files table** — Whether `.tmp` and `.state` sidecar files exist, with file sizes

If the job is incomplete, a resume hint is shown:

```
To resume: karenina verify --resume results.json.state
```

---

## JSON Output

With `--json`, the command outputs a single JSON object suitable for scripting:

```json
{
  "state_file": "results.json.state",
  "output_path": "results.json",
  "benchmark_path": "checkpoint.jsonld",
  "progress": {
    "total_tasks": 200,
    "completed_count": 90,
    "pending_count": 110,
    "progress_percent": 45.0
  },
  "timing": {
    "created_at": "2026-01-15T10:30:00",
    "last_updated_at": "2026-01-15T11:15:42",
    "elapsed_seconds": 2742.5
  },
  "config": {
    "answering_models": ["answering-1 (gpt-4o)"],
    "parsing_models": ["parsing-1 (gpt-4o-mini)"],
    "replicate_count": 1
  },
  "files": {
    "tmp_file_exists": true,
    "tmp_file_size": 524288
  },
  "completed_question_ids": ["urn:uuid:abc123", "urn:uuid:def456"],
  "pending_question_ids": ["urn:uuid:ghi789"],
  "completed_task_ids": ["urn:uuid:abc123\tanswering-1\tparsing-1"],
  "pending_task_ids": ["urn:uuid:ghi789\tanswering-1\tparsing-1"]
}
```

---

## Question IDs vs Task IDs

A **question ID** identifies a single benchmark question (URN format). A **task ID** identifies a specific verification task — the combination of a question, answering model, parsing model, and optional replicate number:

```
{question_id}\t{answering_canonical_key}\t{parsing_canonical_key}[\trep{N}]
```

For a benchmark with 10 questions, 2 answering models, and 3 replicates, there are 10 unique question IDs but 60 task IDs (10 x 2 x 3).

Use `--show-questions` for a high-level view by question. Use `--show-tasks` for the full task-level breakdown.

---

## State File Format

The `.state` file is a JSON file created by `karenina verify --progressive-save`. It contains:

| Field | Type | Description |
|-------|------|-------------|
| `format_version` | `string` | State file format version (`"1.0"`) |
| `created_at` | `string` | ISO 8601 timestamp when the job started |
| `last_updated_at` | `string` | ISO 8601 timestamp of the last completed task |
| `benchmark_path` | `string` | Path to the benchmark checkpoint |
| `output_path` | `string` | Path to the output file |
| `config_hash` | `string` | MD5 hash of the verification config |
| `config` | `object` | Full VerificationConfig as JSON |
| `task_manifest` | `array` | All task IDs for the job |
| `completed_task_ids` | `array` | Task IDs that have finished |
| `total_tasks` | `integer` | Total number of tasks |
| `completed_count` | `integer` | Number of completed tasks |
| `start_time` | `float` | Unix timestamp when the job started |

The `.state` file is created alongside the output file (e.g., `results.json.state` for `results.json`).

---

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | State file inspected successfully |
| `1` | Error — state file not found or invalid format |

---

## Examples

### Basic status check

```bash
karenina verify-status results.json.state
```

### Show question-level progress

```bash
karenina verify-status results.json.state --show-questions
```

### Show all task IDs (verbose)

```bash
karenina verify-status results.json.state --show-tasks
```

### JSON output for scripting

```bash
karenina verify-status results.json.state --json
```

### Pipe JSON to jq for filtering

```bash
karenina verify-status results.json.state --json | jq '.progress'
```

### Full progressive save workflow

```bash
# 1. Start verification with progressive save
karenina verify checkpoint.jsonld --preset default.json \
  --output results.json --progressive-save

# 2. If interrupted, check status
karenina verify-status results.json.state

# 3. Resume from where it left off
karenina verify --resume results.json.state
```

---

## Related

- [verify](verify.md) — The `--progressive-save` and `--resume` options that create and use state files
- [Running Verification](../../06-running-verification/index.md) — Verification workflow guides
- [CLI Reference](index.md) — Overview of all CLI commands
