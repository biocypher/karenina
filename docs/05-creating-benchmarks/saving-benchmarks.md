---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Saving Benchmarks

Once you have created a benchmark, added questions, and defined templates and rubrics, you need to persist it. Karenina offers two persistence options: **JSON-LD checkpoint files** and **database storage**.

For background on the JSON-LD checkpoint format, see [Checkpoints](../04-core-concepts/checkpoints.md).

```python tags=["hide-cell"]
# Setup cell (hidden in rendered docs).
# No mocking needed — all examples use local save operations.
import os
os.chdir(os.path.dirname(os.path.abspath("__file__")))
```

---

## Saving to a JSON-LD File

The most common way to persist a benchmark is as a JSON-LD checkpoint file using `benchmark.save()`:

```python
from pathlib import Path
import tempfile
from karenina import Benchmark

# Create a benchmark with some content
benchmark = Benchmark(name="Drug Target Benchmark", version="1.0.0")
benchmark.add_question(
    question="What is the putative target of imatinib?",
    raw_answer="BCR-ABL",
)
benchmark.add_question(
    question="What is the capital of France?",
    raw_answer="Paris",
)

# Save to a JSON-LD file
tmpdir = tempfile.mkdtemp()
path = Path(tmpdir) / "my_benchmark.jsonld"
benchmark.save(path)

print(f"Saved to: {path.name}")
print(f"File exists: {path.exists()}")
print(f"Questions saved: {benchmark.question_count}")
```

### What Happens on Save

When you call `save()`:

1. The benchmark's `dateModified` timestamp is updated to the current time
2. A deep copy of the checkpoint is made (in-memory state is not modified)
3. Deep judgment configuration is stripped from rubric traits (for backward compatibility)
4. The checkpoint is written as JSON with 2-space indentation and UTF-8 encoding

### File Extension Handling

The `save()` method accepts `.jsonld` and `.json` extensions. If you provide a different extension, it is automatically converted to `.jsonld`:

```python
# All of these produce valid checkpoint files:
# benchmark.save(Path("benchmark.jsonld"))   # kept as-is
# benchmark.save(Path("benchmark.json"))     # kept as-is
# benchmark.save(Path("benchmark.txt"))      # saved as benchmark.jsonld
```

!!! note
    `save()` silently overwrites existing files. If you need to preserve earlier versions, use timestamped filenames or a backup strategy.

### Verifying the Saved File

You can verify a saved checkpoint by loading it back:

```python
loaded = Benchmark.load(path)

print(f"Name: {loaded.name}")
print(f"Questions: {loaded.question_count}")
print(f"Round-trip OK: {loaded.question_count == benchmark.question_count}")
```

---

## Saving to a Database

For workflows where you need queryable storage, version tracking, or integration with other tools, save to a database using `benchmark.save_to_db()`:

```python
import os

db_path = Path(tmpdir) / "benchmarks.db"
storage_url = f"sqlite:///{db_path}"

saved = benchmark.save_to_db(storage=storage_url)

print(f"Saved to database: {db_path.name}")
print(f"Database exists: {db_path.exists()}")
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `storage` | `str` | *(required)* | Database URL (e.g., `"sqlite:///benchmarks.db"`) |
| `checkpoint_path` | `Path \| None` | `None` | Optional reference to a checkpoint file for traceability |

### Key Behaviors

- **Auto-creates** the database and tables if they don't exist
- **Updates in place** if a benchmark with the same name already exists (metadata is updated, questions are upserted)
- **Deduplicates** questions by their content hash (MD5 of question text)
- **Stores global rubric** as serialized JSON in the benchmark metadata
- **Returns** the `Benchmark` instance for method chaining

### Loading from a Database

Load a previously saved benchmark by name:

```python
loaded_from_db = Benchmark.load_from_db(
    benchmark_name="Drug Target Benchmark",
    storage=storage_url,
)

print(f"Name: {loaded_from_db.name}")
print(f"Questions: {loaded_from_db.question_count}")
```

---

## When to Use Each Approach

| Criterion | JSON-LD File | Database |
|-----------|-------------|----------|
| **Portability** | Easy to share, version control, email | Tied to database location |
| **Queryability** | Must load entire file | SQL queries on questions, results |
| **Collaboration** | Git-friendly, merge-compatible | Needs shared database access |
| **Large benchmarks** | Single large file | Indexed, queryable storage |
| **Verification results** | Stored in checkpoint | Stored with relational links |

**Best practice**: Use JSON-LD files as the primary format for sharing and version control, and database storage for workflows that need querying or automatic result persistence.

---

## Using Both Together

A common pattern is to save to both formats — the file for portability and the database for querying:

```python
# Save to file for sharing
file_path = Path(tmpdir) / "drug_targets_v1.jsonld"
benchmark.save(file_path)

# Save to database with file reference
benchmark.save_to_db(
    storage=storage_url,
    checkpoint_path=file_path,
)

print(f"File: {file_path.name}")
print(f"Database: {db_path.name}")
```

---

## Cleanup

```python
# Clean up temporary files
import shutil
shutil.rmtree(tmpdir, ignore_errors=True)
```

---

## Next Steps

- [Adding Questions](adding-questions.md) — populate your benchmark before saving
- [Defining Rubrics](defining-rubrics.md) — add quality assessment traits
- [Running Verification](../06-running-verification/index.md) — run evaluation on your saved benchmark
- [Analyzing Results](../07-analyzing-results/index.md) — inspect and export results
- [Checkpoints](../04-core-concepts/checkpoints.md) — understand the JSON-LD format
