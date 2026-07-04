---
jupyter:
  jupytext:
    formats: docs/core_concepts/questions-and-benchmarks//md,docs/notebooks/core_concepts/questions-and-benchmarks//ipynb
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

# Benchmarks: The Reproducible Package

A **Benchmark** is Karenina's self-contained evaluation package. It is the "sealed envelope" that bundles everything needed to reproduce an evaluation into a single portable unit.

While a [Question object](../questions/) is the minimal unit of evaluation, the Benchmark is the container that organizes these units, attaches quality standards, and preserves the identity of the entire evaluation for sharing and version control.

```python tags=["hide-cell"]
# Mock cell: ensures examples execute without live API keys.
# This cell is hidden in rendered documentation.
from unittest.mock import MagicMock, patch

mock_modules = {}
for mod in [
    "sqlalchemy",
    "sqlalchemy.orm",
    "sqlalchemy.ext",
    "sqlalchemy.ext.declarative",
    "sqlalchemy.engine",
    "sqlalchemy.sql",
    "sqlalchemy.event",
    "karenina.storage",
    "karenina.storage.base",
    "karenina.storage.engine",
    "karenina.storage.db_config",
    "karenina.storage.models",
    "karenina.storage.generated_models",
    "karenina.storage.auto_mapper",
    "karenina.storage.operations",
]:
    mock_modules[mod] = MagicMock()

with patch.dict("sys.modules", mock_modules):
    from karenina.benchmark import Benchmark
```

## 1. Core Components of a Benchmark

Think of a benchmark as a complete kit. When you share a benchmark, you are handing someone a package composed of:

1.  **Questions**: The primary building blocks (see [Questions](../questions/)).
    *   **Answer Templates**: The executable Python logic contained **within each question** that determines correctness (see [Answer Templates](../../answer-templates/)).
    *   **Question-Specific Rubrics**: Quality standards defined for a single question.
    *   **Workspace Paths**: For [agentic tasks](../../agentic-evaluation/), optional per-question directories containing starter code, tests, or data files.
2.  **Global Rubric Traits**: Quality standards (safety, conciseness) defined at the **benchmark level** that apply to every question in the set (see [Rubrics](../../../../core_concepts/rubrics/)).
3.  **Workspace Root**: For [agentic evaluation](../../agentic-evaluation/), the root directory on the local filesystem where task workspaces live. Each question can reference a subdirectory within this root via its `workspace_path`.
4.  **Identity Metadata**: The name, version, and authorship information that defines the benchmark's purpose.

```
┌─────────────────────────────────────────────┐
│                 Benchmark                   │
│                                             │
│  name: "Drug Target Identification"         │
│  version: "1.0.0"                           │
│  creator: "Pharmacology Team"               │
│  workspace_root: /data/tasks (optional)     │
│                                             │
│  ┌────────────────────────────────────────┐ │
│  │ Global Rubric Traits                   │ │
│  │  conciseness, citation_style, safety   │ │
│  └────────────────────────────────────────┘ │
│                                             │
│  ┌─────────────┐ ┌─────────────┐            │
│  │ Question 1  │ │ Question 2  │  ...       │
│  │             │ │             │            │
│  │ text        │ │ text        │            │
│  │ raw_answer  │ │ raw_answer  │            │
│  │ notes       │ │ (no notes)  │            │
│  │ template    │ │ template    │            │
│  │ q-traits    │ │ (no traits) │            │
│  │ workspace:  │ │ (no wksp)   │            │
│  │  task_01/   │ │             │            │
│  └─────────────┘ └─────────────┘            │
└─────────────────────────────────────────────┘
```

```python
# Creating a benchmark and populating its identity
benchmark = Benchmark.create(
    name="Drug Target Identification",
    description="Evaluate LLM accuracy on identifying drug targets from literature",
    version="1.0.0",
    creator="Pharmacology Team",
)

# You can also add custom properties for domain-specific tracking
benchmark.set_custom_property("domain", "pharmacology")

print(f"Benchmark: {benchmark.name} v{benchmark.version}")
print(f"Domain:    {benchmark.get_custom_property('domain')}")
```

## 2. Portable by Design

Karenina is designed so that evaluation logic travels with the data. The [Question object](../questions/) is the atomic unit of portability: each question carries its own prompt, reference answer, verification logic, and qualitative evaluation criteria as a single self-contained package.

*   **Templates as Code**: [Answer templates](../../answer-templates/) are stored as Python source code strings inside each question. The exact parsing schema and `verify()` logic travel with the question, not in an external file or registry.
*   **Rubrics Included**: Question-level [rubric traits](../../../../core_concepts/rubrics/) are also embedded in the question definition. The instructions for qualitative evaluation (safety, citation style, conciseness) are part of the portable unit, not a separate configuration.
*   **Self-Contained Definitions**: A benchmark's [checkpoint file](../../../../core_concepts/questions-and-benchmarks/checkpoints/) contains the complete evaluation definition: all questions, their templates, their rubrics, and all metadata. No external data files or source repositories are needed to understand or re-run the evaluation (though you still need the Karenina runtime and LLM API access).

## 3. Benchmark vs. Run: "The What" vs. "The How"

A common point of confusion is what a benchmark *doesn't* control.

*   **The Benchmark** defines **WHAT** to evaluate (questions, logic, rubrics). It is a static definition.
*   **The VerificationConfig** defines **HOW** to run it (which models to use, how many replicates, timeouts).

This separation is powerful: it means you can run the exact same benchmark against Claude, GPT-4, and Gemini, or run it multiple times with different temperatures, without ever modifying the benchmark itself. For more on execution settings, see [Evaluation Modes](../../evaluation-modes/).

## 4. A Benchmark's Journey

### 4.1. Authoring & Populating
You start by creating a benchmark and adding questions. At this stage, questions are often marked `finished=False` while you refine their [answer templates](../../answer-templates/) and [rubrics](../../../../core_concepts/rubrics/).

### 4.2. Persisting (Saving)
Benchmarks exist in memory while you work, but they must be persisted to be useful. Karenina offers two paths:

| Path | Primary Use | What it Stores |
| :--- | :--- | :--- |
| **[Checkpoint File](../../../../core_concepts/questions-and-benchmarks/checkpoints/) (`.jsonld`)** | **Sharing & Versioning** | The Benchmark *definition* (Questions, Logic, Metadata). |
| **[Database](../../../../workflows/analyzing-results/)** | **Execution & Analysis** | The Benchmark *plus* all the [Verification Results](../../../../workflows/analyzing-results/verification-result/) from every run. |

For details on the portable JSON-LD format, see [Checkpoints](../../../../core_concepts/questions-and-benchmarks/checkpoints/). For working with stored results, see [Analyzing Results](../../../../workflows/analyzing-results/).

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

### 4.3. Execution
When you "run" a benchmark, the framework pulls the questions and logic from the benchmark, uses a [VerificationConfig](../../../../reference/configuration/verification-config/) to call the LLMs, and then writes the results to the database.

## 5. Workspaces for Agentic Benchmarks

For [agentic evaluation](../../agentic-evaluation/) (coding tasks, data analysis), a benchmark can define a `workspace_root`: the root directory on the local filesystem where task workspaces live. Each question can then set a `workspace_path` (a relative path within the root) pointing to its starter code, test files, or data.

```python
from pathlib import Path

# Set workspace root when loading or after loading
benchmark = Benchmark.load("coding-tasks.jsonld", workspace_root=Path("/data/tasks"))
benchmark.set_workspace_root(Path("/data/tasks"))
```

The workspace root belongs to the benchmark because it describes where the task data lives. Operational settings for how the pipeline handles workspaces (copying, cleanup) belong on [VerificationConfig](../../../../reference/configuration/verification-config/). This separation means the same benchmark can be verified with different operational strategies without modification.

`workspace_root` is not persisted in [checkpoints](../../../../core_concepts/questions-and-benchmarks/checkpoints/) because it is a local filesystem path that varies between machines. When loading a checkpoint on a different machine, supply the new root:

```python
benchmark = Benchmark.load("checkpoint.jsonld", workspace_root=Path("/new/machine/tasks"))
```

For full details on workspace copying, cleanup, and agentic judging, see [Agentic Evaluation](../../agentic-evaluation/).

## 6. Detailed Reference: Metadata Fields

Every benchmark carries built-in identity fields to track its evolution.

| Field | Description |
| :--- | :--- |
| `name` | Human-readable identifier for the benchmark. |
| `version` | Semantic version string (e.g., `"1.2.0"`) to track iterations. |
| `description` | A detailed explanation of what this benchmark evaluates. |
| `creator` | The person or team responsible for the benchmark. |
| `date_created` | ISO timestamp of when the benchmark was initialized. |
| `date_modified` | ISO timestamp of the last change to the benchmark definition. |
| `workspace_root` | Root directory for [agentic task](../../agentic-evaluation/) workspaces (not persisted in checkpoints). |

## 7. Next Steps

*   [Questions](../questions/): Understanding the minimal unit of evaluation.
*   [Answer Templates](../../answer-templates/): Writing the verification logic.
*   [Checkpoints](../../../../core_concepts/questions-and-benchmarks/checkpoints/): How benchmarks persist as portable JSON-LD files.
*   [Evaluation Modes](../../evaluation-modes/): How to configure and execute a benchmark run.
*   [Agentic Evaluation](../../agentic-evaluation/): Workspace setup and agentic judging for coding tasks.
*   [Benchmark Operations](../../../../workflows/creating-benchmarks/benchmark-operations/): The full API for managing benchmarks programmatically.
