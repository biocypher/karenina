---
jupyter:
  jupytext:
    formats: docs/core_concepts/questions-and-benchmarks//md,docs/notebooks/core_concepts/questions-and-benchmarks//ipynb
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Benchmarks

A benchmark is Karenina's self-contained evaluation package. It bundles everything needed to reproduce an evaluation into a single portable unit: questions, answer templates, rubric traits, and metadata. You can save it, share it, load it in another environment, and get the same evaluation.

```python tags=["hide-cell"]
# Mock cell: ensures examples execute without live API keys.
# This cell is hidden in rendered documentation.
from unittest.mock import MagicMock, patch

mock_modules = {}
for mod in ["sqlalchemy", "sqlalchemy.orm", "sqlalchemy.ext", "sqlalchemy.ext.declarative"]:
    mock_modules[mod] = MagicMock()

with patch.dict("sys.modules", mock_modules):
    from karenina.benchmark import Benchmark
```

## The Benchmark as a Package

Think of a benchmark as a sealed envelope containing a complete evaluation. Anyone who receives it has everything they need to run the evaluation without additional context. The benchmark carries:

- **Questions**: what to ask the LLM, plus the expected answers (`raw_answer`)
- **Answer templates**: the structured schemas and `verify()` logic that determine correctness
- **Rubric traits**: quality assessments (conciseness, safety, citation style, etc.)
- **Metadata**: name, version, description, creator, timestamps, and arbitrary custom properties

These four components are independently attachable. A question can exist without a template, without rubric traits, or without both. The benchmark holds them together as a unit.

```
┌─────────────────────────────────────────────┐
│                 Benchmark                   │
│                                             │
│  name: "Drug Target Identification"         │
│  version: "1.0.0"                           │
│  creator: "Pharmacology Team"               │
│                                             │
│  ┌────────────────────────────────────────┐ │
│  │ Global Rubric Traits                   │ │
│  │  conciseness, citation_style, safety   │ │
│  └────────────────────────────────────────┘ │
│                                             │
│  ┌─────────────┐ ┌─────────────┐           │
│  │ Question 1  │ │ Question 2  │  ...       │
│  │             │ │             │           │
│  │ text        │ │ text        │           │
│  │ raw_answer  │ │ raw_answer  │           │
│  │ template    │ │ template    │           │
│  │ q-traits    │ │ (no traits) │           │
│  └─────────────┘ └─────────────┘           │
└─────────────────────────────────────────────┘
```

What is *not* inside the benchmark: verification results. Results are generated when the benchmark is run and are stored separately (see [Persistence](#persistence) below).

## Benchmark-Level Metadata

Every benchmark carries identity metadata that tracks its evolution and purpose:

| Field | Purpose |
|-------|---------|
| `name` | Human-readable identifier |
| `version` | Semantic version string (e.g., `"1.0.0"`) |
| `description` | What this benchmark evaluates |
| `creator` | Who authored it |
| `date_created` | When the benchmark was first created |
| `date_modified` | When it was last changed |

Beyond these built-in fields, benchmarks support **custom properties**: arbitrary key-value pairs for domain-specific metadata. Use them for attributes that characterize the benchmark as a whole (domain, regulatory context, target audience). Per-question attributes belong on the question itself; see [Questions](questions.md).

```python
benchmark = Benchmark.create(
    name="Drug Target Identification",
    description="Evaluate LLM accuracy on identifying drug targets from literature",
    version="1.0.0",
    creator="Pharmacology Team",
)

benchmark.set_custom_property("domain", "pharmacology")
benchmark.set_custom_property("target_audience", "drug discovery teams")

print(f"Name:    {benchmark.name}")
print(f"Version: {benchmark.version}")
print(f"Domain:  {benchmark.get_custom_property('domain')}")
```

## Questions Inside the Benchmark

Questions are the basic building blocks within a benchmark. Each question carries the text sent to the LLM, a reference answer (`raw_answer`), and optional attachments (a template, rubric traits, few-shot examples, intrinsic metadata).

The benchmark assigns each question a **deterministic ID** derived from its text. The same question text always produces the same ID, which enables reliable cross-referencing across checkpoint files and database records.

Questions also have a **registry entry** that tracks benchmark membership state: whether the question is marked `finished` (ready for the verification pipeline) and when it was added.

For the full question data model, see [Questions](questions.md).

## Persistence

Benchmarks exist in memory while you build and run them. To share or reuse a benchmark, you persist it. Karenina provides two persistence paths:

### Checkpoint Files

A checkpoint is a **JSON-LD file** (`.jsonld`) that captures the benchmark's definition: questions, templates, rubric traits, and metadata. Checkpoints are portable, human-readable, and version-control friendly. They are the primary way to share benchmarks across teams and environments.

Checkpoints capture the benchmark's *definition*, not its results. They answer: "what should be evaluated and how?"

```python
from pathlib import Path

# Save
# benchmark.save(Path("drug_targets_v1.jsonld"))

# Load in another environment
# benchmark = Benchmark.load(Path("drug_targets_v1.jsonld"))
```

For the checkpoint format, Schema.org mapping, and serialization details, see [Checkpoints](../checkpoints.md).

### Database Storage

The database stores both benchmark definitions and verification results. When you run verification, results are written to the database and linked to the benchmark and its questions. The database is where results live; the benchmark itself does not contain them.

```
┌──────────────────────────────────────┐
│              Database                │
│                                      │
│  ┌────────────┐  ┌────────────────┐  │
│  │ Benchmarks │  │ Verification   │  │
│  │            │  │ Results        │  │
│  │ questions  │◄─┤                │  │
│  │ templates  │  │ per-question   │  │
│  │ rubrics    │  │ per-model      │  │
│  │ metadata   │  │ per-run        │  │
│  └────────────┘  └────────────────┘  │
└──────────────────────────────────────┘
```

**When to use which:**

| Path | Best for |
|------|----------|
| Checkpoint files | Sharing, version control, portability, archival |
| Database | Running verification, storing results, querying across benchmarks |

In practice, you often use both: author a benchmark and save it as a checkpoint for sharing, then load it into a database-backed session to run verification and collect results.

## Key Concepts

**A benchmark is a definition, not an execution.** The benchmark describes *what* to evaluate and *how* to check it. Running the evaluation is a separate step that requires a [`VerificationConfig`](../evaluation-modes.md) specifying which models to use, how many replicates, and other runtime settings.

**Results live outside the benchmark.** After verification runs, results are stored in the database, not inside the benchmark object. This separation means you can run the same benchmark multiple times with different models or configurations and compare results without modifying the benchmark itself.

**Templates are executable code.** Answer templates are stored as Python source code strings. This makes benchmarks fully reproducible (the exact verification logic travels with the benchmark) but also means template code must be syntactically valid Python.

**Questions without templates still exist.** A question can live in a benchmark without a template. This is useful during iterative authoring or for rubric-only evaluation. See [Evaluation Modes](../evaluation-modes.md) for how the pipeline handles different combinations of templates and rubrics.

## Next Steps

- [Questions](questions.md): the Question schema, deterministic IDs, `raw_answer`, metadata layers, the `finished` flag
- [Answer Templates](../answer-templates.md): how to write templates, field types, verification patterns
- [Rubrics](../rubrics/index.md): trait types, global vs question-specific, quality assessment
- [Templates vs Rubrics](../template-vs-rubric.md): when to use which
- [Checkpoints](../checkpoints.md): how benchmarks persist as JSON-LD files
- [Evaluation Modes](../evaluation-modes.md): template-only, rubric-only, and combined evaluation
- [Benchmark Operations](../../workflows/creating-benchmarks/benchmark-operations.md): full operational API (creating, populating, readiness, filtering)
- [Few-Shot](../few-shot.md): configuring example injection for parsing accuracy
- [Creating Benchmarks](../../workflows/creating-benchmarks/index.md): step-by-step authoring scenarios
