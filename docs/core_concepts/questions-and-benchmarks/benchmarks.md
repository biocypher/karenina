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

While a [Question object](questions.md) is the minimal unit of evaluation, the Benchmark is the container that organizes these units, attaches quality standards, and preserves the identity of the entire evaluation for sharing and version control.

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

## 1. Core Components of a Benchmark

Think of a benchmark as a complete kit. When you share a benchmark, you are handing someone a package composed of:

1.  **Questions**: The primary building blocks (see [Questions](questions.md)).
    *   **Answer Templates**: The executable Python logic contained **within each question** that determines correctness (see [Answer Templates](../answer-templates.md)).
    *   **Question-Specific Rubrics**: Quality standards defined for a single question.
2.  **Global Rubric Traits**: Quality standards (safety, conciseness) defined at the **benchmark level** that apply to every question in the set (see [Rubrics](../rubrics/index.md)).
3.  **Identity Metadata**: The name, version, and authorship information that defines the benchmark's purpose.

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
│  ┌─────────────┐ ┌─────────────┐            │
│  │ Question 1  │ │ Question 2  │  ...       │
│  │             │ │             │            │
│  │ text        │ │ text        │            │
│  │ raw_answer  │ │ raw_answer  │            │
│  │ notes       │ │ (no notes)  │            │
│  │ template    │ │ template    │            │
│  │ q-traits    │ │ (no traits) │            │
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

Karenina is designed so that evaluation logic travels with the data. The [Question object](questions.md) is the atomic unit of portability: each question carries its own prompt, reference answer, verification logic, and qualitative evaluation criteria as a single self-contained package.

*   **Templates as Code**: [Answer templates](../answer-templates.md) are stored as Python source code strings inside each question. The exact parsing schema and `verify()` logic travel with the question, not in an external file or registry.
*   **Rubrics Included**: Question-level [rubric traits](../rubrics/index.md) are also embedded in the question definition. The instructions for qualitative evaluation (safety, citation style, conciseness) are part of the portable unit, not a separate configuration.
*   **Self-Contained Definitions**: A benchmark's [checkpoint file](checkpoints.md) contains the complete evaluation definition: all questions, their templates, their rubrics, and all metadata. No external data files or source repositories are needed to understand or re-run the evaluation (though you still need the Karenina runtime and LLM API access).

## 3. Benchmark vs. Run: "The What" vs. "The How"

A common point of confusion is what a benchmark *doesn't* control.

*   **The Benchmark** defines **WHAT** to evaluate (questions, logic, rubrics). It is a static definition.
*   **The VerificationConfig** defines **HOW** to run it (which models to use, how many replicates, timeouts).

This separation is powerful: it means you can run the exact same benchmark against Claude, GPT-4, and Gemini, or run it multiple times with different temperatures, without ever modifying the benchmark itself. For more on execution settings, see [Evaluation Modes](../evaluation-modes.md).

## 4. A Benchmark's Journey

### 4.1. Authoring & Populating
You start by creating a benchmark and adding questions. At this stage, questions are often marked `finished=False` while you refine their [answer templates](../answer-templates.md) and [rubrics](../rubrics/index.md).

### 4.2. Persisting (Saving)
Benchmarks exist in memory while you work, but they must be persisted to be useful. Karenina offers two paths:

| Path | Primary Use | What it Stores |
| :--- | :--- | :--- |
| **[Checkpoint File](checkpoints.md) (`.jsonld`)** | **Sharing & Versioning** | The Benchmark *definition* (Questions, Logic, Metadata). |
| **[Database](../../workflows/analyzing-results/index.md)** | **Execution & Analysis** | The Benchmark *plus* all the [Verification Results](../../workflows/analyzing-results/verification-result.md) from every run. |

For details on the portable JSON-LD format, see [Checkpoints](checkpoints.md). For working with stored results, see [Analyzing Results](../../workflows/analyzing-results/index.md).

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
When you "run" a benchmark, the framework pulls the questions and logic from the benchmark, uses a [VerificationConfig](../../reference/configuration/verification-config.md) to call the LLMs, and then writes the results to the database.

## 5. Detailed Reference: Metadata Fields

Every benchmark carries built-in identity fields to track its evolution.

| Field | Description |
| :--- | :--- |
| `name` | Human-readable identifier for the benchmark. |
| `version` | Semantic version string (e.g., `"1.2.0"`) to track iterations. |
| `description` | A detailed explanation of what this benchmark evaluates. |
| `creator` | The person or team responsible for the benchmark. |
| `date_created` | ISO timestamp of when the benchmark was initialized. |
| `date_modified` | ISO timestamp of the last change to the benchmark definition. |

## 6. Next Steps

*   [Questions](questions.md): Understanding the minimal unit of evaluation.
*   [Answer Templates](../answer-templates.md): Writing the verification logic.
*   [Checkpoints](checkpoints.md): How benchmarks persist as portable JSON-LD files.
*   [Evaluation Modes](../evaluation-modes.md): How to configure and execute a benchmark run.
*   [Benchmark Operations](../../workflows/creating-benchmarks/benchmark-operations.md): The full API for managing benchmarks programmatically.
