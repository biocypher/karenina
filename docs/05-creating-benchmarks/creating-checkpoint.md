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

# Creating a Checkpoint

A **checkpoint** is a Karenina benchmark file in JSON-LD format. Creating one is the first step in building any benchmark. This page covers the `Benchmark` constructor, metadata fields, and inspecting your new benchmark.

For background on what checkpoints are and why they use JSON-LD, see [Checkpoints](../core_concepts/checkpoints.md).

```python tags=["hide-cell"]
# Setup cell (hidden in rendered docs).
# No mocking needed — all examples create Benchmark objects locally.
```

---

## Creating a Benchmark

Use the `Benchmark` constructor or the `Benchmark.create()` class method (they are equivalent):

```python
from karenina import Benchmark

# Using the constructor
benchmark = Benchmark(name="Genomics Knowledge Benchmark")

# Or equivalently, using the class method
benchmark = Benchmark.create(name="Genomics Knowledge Benchmark")

print(f"Created: {benchmark.name}")
```

## Constructor Parameters

The constructor accepts four parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *(required)* | Name of the benchmark |
| `description` | `str` | `""` | Human-readable description of the benchmark's purpose |
| `version` | `str` | `"0.1.0"` | Version string for tracking benchmark evolution |
| `creator` | `str` | `"Karenina Benchmarking System"` | Creator name or organization |

Here is a benchmark with all metadata fields:

```python
benchmark = Benchmark.create(
    name="Drug Target Identification",
    description="Evaluates LLM knowledge of drug mechanisms and molecular targets",
    version="1.0.0",
    creator="Pharmacology Research Lab",
)

print(f"Name:        {benchmark.name}")
print(f"Description: {benchmark.description}")
print(f"Version:     {benchmark.version}")
print(f"Creator:     {benchmark.creator}")
```

## Inspecting a Benchmark

A freshly created benchmark starts empty. You can check its state using these properties and methods:

```python
print(f"Question count:  {benchmark.question_count}")
print(f"Finished count:  {benchmark.finished_count}")
print(f"Is empty:        {benchmark.is_empty}")
print(f"Is complete:     {benchmark.is_complete}")
print(f"Progress:        {benchmark.get_progress()}%")
```

### Timestamps

Every benchmark tracks when it was created and last modified:

```python
print(f"Created at:  {benchmark.created_at}")
print(f"Modified at: {benchmark.modified_at}")
```

### Mutable Metadata

All metadata fields are writable after creation:

```python
benchmark.description = "Updated description with expanded scope"
benchmark.version = "1.1.0"

print(f"New version:     {benchmark.version}")
print(f"New description: {benchmark.description}")
```

## Saving Your Checkpoint

Once you've created a benchmark (and added questions — see the next page), save it as a JSON-LD checkpoint file:

```python
from pathlib import Path
import tempfile

# Save to a file
with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "my_benchmark.jsonld"
    benchmark.save(path)
    print(f"Saved to: {path.name}")
    print(f"File exists: {path.exists()}")
```

For details on all persistence options (JSON-LD and database), see [Saving Benchmarks](saving-benchmarks.md).

---

## Next Steps

- [Adding Questions](adding-questions.md) — populate your benchmark with evaluation content
- [Writing Templates](writing-templates.md) — define evaluation criteria for your questions
- [Defining Rubrics](defining-rubrics.md) — add quality assessment traits
- [Saving Benchmarks](saving-benchmarks.md) — persist to JSON-LD or database
