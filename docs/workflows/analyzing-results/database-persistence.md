---
jupyter:
  jupytext:
    formats: docs/workflows/analyzing-results//md,docs/notebooks/analyzing-results//ipynb
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Database Persistence

This tutorial shows how to save benchmarks and verification results to a database for long-term storage. While JSON-LD checkpoints are great for portability and sharing, database storage enables querying, filtering, and comparing results across runs. Use database persistence when you need to track results over time, compare across verification runs, or build analysis pipelines.

**What you'll learn:**

- Configure database access with `DBConfig`
- Save a benchmark to the database with `save_benchmark()`
- Load a benchmark from the database with `load_benchmark()`
- Save verification results with `save_verification_results()`
- Load and filter results with `load_verification_results()`
- Import results from JSON exports with `import_verification_results()`
- Detect duplicates before saving

```python tags=["hide-cell"]
# Setup cell: hidden in rendered documentation.
# Creates a temporary SQLite database, a small benchmark, and mock verification results.
import datetime, tempfile
from pathlib import Path
from uuid import uuid4

from karenina import Benchmark
from karenina.schemas.verification import VerificationResult
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.schemas.verification.result_components import (
    VerificationResultMetadata, VerificationResultTemplate,
)
from karenina.storage import DBConfig
from karenina.storage.operations import (
    import_verification_results, load_benchmark, load_verification_results,
    save_benchmark, save_verification_results,
)

_tmpdir = tempfile.mkdtemp(prefix="karenina_db_tutorial_")
_db_path = Path(_tmpdir) / "tutorial.db"
_db_url = f"sqlite:///{_db_path}"

_benchmark = Benchmark.create(
    name="Drug Target QA",
    description="Evaluate LLM accuracy on drug target identification",
    version="1.0.0", creator="Pharmacology Team",
)
_q1_id = _benchmark.add_question(
    question="What is the primary target of venetoclax?",
    raw_answer="BCL2 (B-cell lymphoma 2)",
)
_q2_id = _benchmark.add_question(
    question="What is the mechanism of action of imatinib?",
    raw_answer="BCR-ABL tyrosine kinase inhibitor",
)
_q3_id = _benchmark.add_question(
    question="What receptor does trastuzumab target?",
    raw_answer="HER2 (human epidermal growth factor receptor 2)",
)

_answering = ModelIdentity(model_name="claude-haiku-4-5", interface="langchain")
_parsing = ModelIdentity(model_name="claude-haiku-4-5", interface="langchain")
_ts = datetime.datetime.now(tz=datetime.UTC).isoformat()
_run_id = str(uuid4())

def _make_result(qid, verified, response):
    rid = VerificationResultMetadata.compute_result_id(qid, _answering, _parsing, _ts)
    return VerificationResult(
        metadata=VerificationResultMetadata(
            question_id=qid, template_id="tmpl_mock",
            completed_without_errors=True, question_text="mock",
            answering=_answering, parsing=_parsing,
            execution_time=1.2, timestamp=_ts, result_id=rid,
        ),
        template=VerificationResultTemplate(
            raw_llm_response=response, verify_result=verified,
            template_verification_performed=True,
        ),
    )

_mock_results = {
    f"{_q1_id}_0": _make_result(_q1_id, True, "BCL2 is the primary target."),
    f"{_q2_id}_0": _make_result(_q2_id, True, "Imatinib inhibits BCR-ABL."),
    f"{_q3_id}_0": _make_result(_q3_id, False, "Trastuzumab targets EGFR."),
}
```

---

## Configure Database Access

All database operations require a `DBConfig` object that specifies how to connect. The most important field is `storage_url`, which uses SQLAlchemy URL format. For local development, SQLite is the simplest option.

```python
db_config = DBConfig(
    storage_url=_db_url,
    auto_create=True,   # Create tables automatically on first use
    auto_commit=True,    # Commit after each operation
    echo=False,          # Set True to see SQL statements (debugging)
)

print(f"Storage URL:  {db_config.storage_url}")
print(f"Dialect:      {db_config.dialect}")
print(f"Is SQLite:    {db_config.is_sqlite}")
print(f"Auto-create:  {db_config.auto_create}")
print(f"Pool size:    {db_config.pool_size}")
```

With `auto_create=True`, tables are created automatically on first use. For production databases (PostgreSQL, MySQL), configure connection pooling with `pool_size`, `max_overflow`, and `pool_recycle`.

---

## Save a Benchmark

`save_benchmark()` writes the benchmark's questions, templates, rubrics, and metadata to the database. If a benchmark with the same name already exists, it updates the existing record (upsert behavior).

```python
benchmark = _benchmark
save_benchmark(benchmark, db_config)

print(f"Saved benchmark: {benchmark.name}")
print(f"Questions saved: {benchmark.question_count}")
print(f"Database file:   {_db_path.name}")
```

You can also pass a plain URL string instead of a `DBConfig` object; the function creates a `DBConfig` internally with default settings.

---

## Load a Benchmark

`load_benchmark()` reconstructs a full `Benchmark` object from the database, including all questions, templates, rubrics, and metadata.

```python
loaded = load_benchmark("Drug Target QA", db_config)

print(f"Loaded:    {loaded.name}")
print(f"Version:   {loaded.version}")
print(f"Questions: {loaded.question_count}")
print(f"Round-trip match: {loaded.question_count == benchmark.question_count}")
```

The loaded benchmark is fully functional: you can run verification, add questions, or modify templates on it.

---

## Save Verification Results

After running verification, save the results to the database for long-term tracking. Results are grouped by a `run_id` that identifies the verification run.

```python
run_id = _run_id
results = _mock_results

save_verification_results(
    results=results,
    db_config=db_config,
    run_id=run_id,
    benchmark_name="Drug Target QA",
    run_name="initial-run",
)

print(f"Saved {len(results)} results")
print(f"Run ID:   {run_id[:16]}...")
print(f"Run name: initial-run")
```

The benchmark must already exist in the database (call `save_benchmark()` first). Each result is associated with the benchmark through the run record.

---

## Load Verification Results

`load_verification_results()` retrieves results from the database with flexible filtering. The default return format is a dictionary mapping result keys to `VerificationResult` objects.

### Load by benchmark name

```python
loaded_results = load_verification_results(
    db_config=db_config,
    benchmark_name="Drug Target QA",
)

print(f"Loaded {len(loaded_results)} results")
for key, result in loaded_results.items():
    status = "PASS" if result.template and result.template.verify_result else "FAIL"
    print(f"  {key}: {status}")
```

### Load by run name

```python
run_results = load_verification_results(
    db_config=db_config,
    run_name="initial-run",
)

print(f"Results from 'initial-run': {len(run_results)}")
```

### Filter by question IDs

```python
subset = load_verification_results(
    db_config=db_config,
    question_ids=[_q1_id, _q3_id],
)

print(f"Filtered to 2 questions: {len(subset)} results")
```

### Filter by answering model

```python
model_results = load_verification_results(
    db_config=db_config,
    answering_model="claude-haiku-4-5",
)

print(f"Results from claude-haiku-4-5: {len(model_results)}")
```

---

## Import from JSON Export

If you have results exported as JSON (from `export_verification_results_to_file()`), you can import them into the database with `import_verification_results()`. This is useful for migrating results from file-based workflows to database storage.

```python
# Create a mock JSON export in v2.0 format
mock_json_export = {
    "format_version": "2.0",
    "metadata": {
        "benchmark_name": "Drug Target QA",
        "karenina_version": "0.1.0",
        "timestamp": _ts,
    },
    "shared_data": {},
    "results": [
        {
            "metadata": {
                "question_id": _q1_id,
                "template_id": "tmpl_import",
                "completed_without_errors": True,
                "question_text": "What is the primary target of venetoclax?",
                "answering": {"model_name": "claude-sonnet-4-20250514", "interface": "langchain"},
                "parsing": {"model_name": "claude-haiku-4-5", "interface": "langchain"},
                "execution_time": 2.1,
                "timestamp": _ts,
                "result_id": "import_result_001",
            },
            "template": {
                "raw_llm_response": "BCL2 is the direct target.",
                "verify_result": True,
                "template_verification_performed": True,
            },
        },
    ],
}

import_run_id, imported, skipped = import_verification_results(
    json_data=mock_json_export,
    db_config=db_config,
    benchmark_name="Drug Target QA",
    run_name="imported-from-file",
    source_filename="results_export.json",
)

print(f"Import run ID: {import_run_id[:16]}...")
print(f"Imported:      {imported}")
print(f"Skipped:       {skipped}")
```

The function returns `(run_id, imported_count, skipped_count)`. Results are skipped when their `question_id` cannot be matched to a question in the database. Matching is attempted by direct ID, URN-wrapped ID, and text hash.

---

## Duplicate Detection

Before saving a benchmark that may overlap with existing data, use `detect_duplicates_only=True` to preview conflicts without modifying the database. This returns a tuple of the benchmark and a list of duplicate entries.

```python
# Modify a question locally to create a difference
benchmark_copy = Benchmark.create(
    name="Drug Target QA",
    description="Updated description",
    version="2.0.0",
    creator="Pharmacology Team",
)
benchmark_copy.add_question(
    question="What is the primary target of venetoclax?",
    raw_answer="BCL2 (updated answer with more detail)",
)

result = save_benchmark(
    benchmark_copy,
    db_config,
    detect_duplicates_only=True,
)

# When detect_duplicates_only=True, result is a tuple
returned_benchmark, duplicates = result
print(f"Duplicates found: {len(duplicates)}")
for dup in duplicates:
    print(f"  Question: {dup['question_text'][:50]}...")
    print(f"  Old answer: {dup['old_version']['raw_answer'][:40]}...")
    print(f"  New answer: {dup['new_version']['raw_answer'][:40]}...")
```

This lets you inspect what would change before committing. When you are satisfied, call `save_benchmark()` again without the flag to apply the changes.

---

## JSON-LD vs Database

Both storage approaches are complementary. Use the one that fits your workflow, or use both together.

| Feature | JSON-LD Checkpoint | Database |
|---------|-------------------|----------|
| Portability | File-based, easy to share | Requires DB access |
| Querying | Load entire file | Filter by any field |
| History | One snapshot per file | Multiple runs stored |
| Best for | Sharing, version control | Long-term tracking, analysis |

A common pattern: save checkpoints to JSON-LD for version control and sharing, then import them into a database for cross-run analysis and trend tracking.

---

## Cleanup

```python
import shutil
shutil.rmtree(_tmpdir, ignore_errors=True)
print(f"Cleaned up: {_tmpdir}")
```

---

## Next Steps

- [Verification Result Structure](verification-result.md): Understand the fields available in each result
- [Exporting Results](exporting.md): Save results as JSON or CSV for sharing
- [DataFrame Analysis](../../notebooks/analyzing-results/dataframe-analysis.ipynb): Convert results to pandas DataFrames for deeper analysis
- [Checkpoints](../../core_concepts/questions-and-benchmarks/checkpoints.md): File-based persistence with JSON-LD
