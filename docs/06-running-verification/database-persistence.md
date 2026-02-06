# Database Persistence

Karenina can automatically persist verification results to a database as part of
the verification workflow. This is useful for long-running jobs, tracking results
across multiple runs, and querying results programmatically.

## How Auto-Save Works

When you include a `DBConfig` in your `VerificationConfig`, results are
automatically saved to the specified database after verification completes:

```
Verification runs
       │
       ▼
Results generated in memory
       │
       ▼
Auto-save to database (if db_config set + AUTOSAVE_DATABASE != false)
       │
       ▼
Results returned to caller
```

Auto-save is **non-blocking** — if the database write fails, verification still
succeeds and results are returned normally. Failures are logged but do not raise
exceptions.

## Configuring DBConfig

`DBConfig` controls the database connection. It is a Pydantic model imported from
`karenina.storage`:

```python
from karenina.storage import DBConfig

db_config = DBConfig(
    storage_url="sqlite:///my_results.db",
    auto_create=True,   # Create tables if they don't exist (default: True)
)
```

### DBConfig Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `storage_url` | `str` | *(required)* | SQLAlchemy database URL (e.g. `sqlite:///results.db`, `postgresql://user:pass@host/db`) |
| `auto_create` | `bool` | `True` | Automatically create tables and views if missing |
| `auto_commit` | `bool` | `True` | Commit transactions automatically after operations |
| `echo` | `bool` | `False` | Log all SQL statements (useful for debugging) |
| `pool_size` | `int` | `5` | Connection pool size (non-SQLite only) |
| `max_overflow` | `int` | `10` | Max connections beyond pool_size (non-SQLite only) |
| `pool_recycle` | `int` | `3600` | Recycle connections after N seconds (-1 to disable) |
| `pool_pre_ping` | `bool` | `True` | Test connections before use |

SQLite databases automatically set `pool_size=1` and `max_overflow=0` since
SQLite does not support concurrent writes.

### Database URL Examples

```python
# SQLite (file-based, no server needed)
db_config = DBConfig(storage_url="sqlite:///results.db")

# SQLite (absolute path)
db_config = DBConfig(storage_url="sqlite:////data/karenina/results.db")

# PostgreSQL
db_config = DBConfig(
    storage_url="postgresql://user:password@localhost:5432/karenina",
    pool_size=10,
    max_overflow=20,
)
```

## Enabling Auto-Save During Verification

Pass `db_config` in your `VerificationConfig`:

```python
from karenina import Benchmark
from karenina.schemas import VerificationConfig, ModelConfig
from karenina.storage import DBConfig

benchmark = Benchmark.load("my_benchmark.jsonld")

config = VerificationConfig(
    answering_models=[
        ModelConfig(id="gpt-4o", model_provider="openai",
                    model_name="gpt-4o", interface="langchain")
    ],
    parsing_models=[
        ModelConfig(id="gpt-judge", model_provider="openai",
                    model_name="gpt-4o", temperature=0.0,
                    interface="langchain")
    ],
    evaluation_mode="template_and_rubric",
    db_config=DBConfig(storage_url="sqlite:///results.db"),
)

# Results are auto-saved to the database after verification completes
results = benchmark.run_verification(config)
```

### What Gets Saved

Auto-save creates these records:

1. **Benchmark record** — Created automatically if the benchmark doesn't already exist in the database
2. **Verification run** — A run record with status, config, timing, and progress counts
3. **Verification results** — One result row per question, containing the full `VerificationResult` data (template results, rubric scores, metadata)

### Controlling Auto-Save with Environment Variables

The `AUTOSAVE_DATABASE` environment variable controls whether auto-save runs:

| Value | Behavior |
|-------|----------|
| `true`, `1`, `yes` (default) | Auto-save enabled when `db_config` is set |
| `false`, `0`, `no` | Auto-save disabled even if `db_config` is set |

```bash
# Disable auto-save temporarily
export AUTOSAVE_DATABASE=false
```

Even with `AUTOSAVE_DATABASE=true`, auto-save only runs if `db_config` is
provided in the `VerificationConfig`. Without `db_config`, no database writes
occur.

## Saving Benchmarks to Database

You can also save benchmarks directly (independent of verification):

```python
from karenina import Benchmark
from pathlib import Path

benchmark = Benchmark.load("my_benchmark.jsonld")

# Save to database
benchmark.save_to_db(storage="sqlite:///benchmarks.db")

# Save to database with a checkpoint backup
benchmark.save_to_db(
    storage="sqlite:///benchmarks.db",
    checkpoint_path=Path("backup.jsonld"),
)
```

This persists the benchmark structure (questions, templates, rubrics) to the
database, making it loadable without the original checkpoint file.

## Loading from Database

Load a previously saved benchmark by name:

```python
from karenina import Benchmark

benchmark = Benchmark.load_from_db(
    benchmark_name="My Benchmark",
    storage="sqlite:///benchmarks.db",
)

print(f"Loaded: {benchmark.name}, {len(benchmark.questions)} questions")
```

## Querying Verification Results

### Load Results

Use `load_verification_results` to retrieve results with flexible filtering:

```python
from karenina.storage import DBConfig, load_verification_results

db_config = DBConfig(storage_url="sqlite:///results.db")

# Load all results for a benchmark
results = load_verification_results(db_config, benchmark_name="My Benchmark")

# Filter by run name
results = load_verification_results(db_config, run_name="run-2026-02-06")

# Filter by specific questions
results = load_verification_results(
    db_config,
    benchmark_name="My Benchmark",
    question_ids=["q1", "q2"],
)

# Filter by answering model
results = load_verification_results(
    db_config,
    answering_model="gpt-4o",
)

# Limit number of results
results = load_verification_results(db_config, limit=100)
```

The `as_dict` parameter controls the return format:

- `as_dict=True` (default): Returns `dict[str, VerificationResult]` keyed by `{question_id}_{index}`
- `as_dict=False`: Returns `list[dict[str, Any]]` with metadata fields (id, run_id)

### Query Helpers

The `karenina.storage` module provides several query helpers for aggregated
statistics. All accept a `DBConfig` and return dictionaries:

| Function | Purpose |
|----------|---------|
| `get_benchmark_summary(db_config, benchmark_name)` | Benchmark question counts and status |
| `get_verification_run_summary(db_config, run_name)` | Run progress and statistics |
| `get_latest_verification_results(db_config, benchmark_name, question_id)` | Most recent results |
| `get_model_performance(db_config, model_name, model_role)` | Model accuracy statistics |
| `get_rubric_scores_aggregate(db_config, question_id, min_evaluations)` | Rubric trait score distributions |
| `get_verification_history_timeline(db_config, benchmark_name, limit)` | Timeline of verification runs |
| `get_database_statistics(db_config)` | Overall counts (benchmarks, questions, runs, results) |

```python
from karenina.storage import DBConfig, get_database_statistics, get_model_performance

db_config = DBConfig(storage_url="sqlite:///results.db")

# Overall database summary
stats = get_database_statistics(db_config)
print(f"Benchmarks: {stats['total_benchmarks']}")
print(f"Results: {stats['total_verification_results']}")

# Model performance
perf = get_model_performance(db_config, model_name="gpt-4o")
for entry in perf:
    print(f"  {entry['model_role']}: {entry['total_runs']} runs")
```

### Importing Results from JSON

If you have exported results as JSON (e.g., from `benchmark.export_results()`),
you can import them into the database:

```python
import json
from karenina.storage import DBConfig, import_verification_results

db_config = DBConfig(storage_url="sqlite:///results.db")

with open("exported_results.json") as f:
    json_data = json.load(f)

run_id, imported, skipped = import_verification_results(
    json_data=json_data,
    db_config=db_config,
    benchmark_name="My Benchmark",
    run_name="imported-run",
    source_filename="exported_results.json",
)

print(f"Imported {imported} results, skipped {skipped}")
```

## Database Schema Overview

The database uses the following tables:

```
benchmarks
    │
    ├── benchmark_questions ──── questions
    │
    └── verification_runs
            │
            ├── verification_results
            │
            └── import_metadata
```

| Table | Purpose |
|-------|---------|
| `benchmarks` | Benchmark definitions (name, version, metadata) |
| `questions` | Question text and answers (deduplicated by content hash) |
| `benchmark_questions` | Links questions to benchmarks with templates and rubrics |
| `verification_runs` | Run metadata (config, timing, status, progress) |
| `verification_results` | Individual results per question (auto-generated from `VerificationResult` schema) |
| `import_metadata` | Audit trail for imported results |

In addition to tables, the database creates several **views** for aggregated
queries (benchmark summaries, model performance, rubric score aggregates, etc.).
Views are created automatically when `auto_create=True`.

## Database Management

### Initialize a Database

Tables and views are created automatically when `auto_create=True` on `DBConfig`.
For explicit control:

```python
from karenina.storage import DBConfig, init_database

db_config = DBConfig(storage_url="sqlite:///results.db", auto_create=False)
init_database(db_config)  # Create all tables and views
```

### Close Connections

Engine connections are cached per URL for efficiency. To close them:

```python
from karenina.storage import close_engine

# Close a specific engine's connection pool
close_engine(db_config)
```

---

## Next Steps

- [Analyzing Results](../07-analyzing-results/index.md) — Work with `VerificationResult` and DataFrames
- [Exporting Results](../07-analyzing-results/exporting.md) — Export to JSON, CSV, or checkpoint formats
- [VerificationConfig Reference](../10-configuration-reference/verification-config.md) — Full configuration options including `db_config`
- [Python API](python-api.md) — Running verification programmatically
