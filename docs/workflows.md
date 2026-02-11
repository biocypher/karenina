# Workflows

This section is about *doing* — step-by-step guides that take you from a starting point to a concrete outcome. Each page assumes you understand the relevant concepts (or links to them) and focuses on how to accomplish a specific task. For the underlying mental models, see [Core Concepts](core_concepts/index.md). For exhaustive option tables, see [Reference](reference.md).

---

## In This Section

| Workflow | What You'll Do |
|----------|---------------|
| [Configuration](03-configuration/index.md) | Set up the configuration hierarchy: CLI args, presets, environment variables, defaults |
| [Creating Benchmarks](05-creating-benchmarks/index.md) | Author questions, write templates, define rubrics, and save checkpoints |
| [Running Verification](06-running-verification/index.md) | Configure and execute evaluation via Python API or CLI |
| [Analyzing Results](07-analyzing-results/index.md) | Inspect results, build DataFrames, export data, and iterate |

---

## End-to-End Flow

```
Configure            Create Benchmark          Run Verification          Analyze Results
──────────────  →    ─────────────────    →    ─────────────────    →    ─────────────────
Set up env vars      Create checkpoint         Load benchmark            Explore result structure
Configure presets    Add questions             Configure models          Filter and group
                     Write templates           Choose eval mode          Build DataFrames
                     Define rubrics            Execute pipeline          Export and iterate
                     Save (.jsonld)            Collect results
```

Each workflow section has an overview page with a visual diagram, followed by dedicated pages for each step. Pages include executable notebook examples where applicable.

---

## Prerequisites

Before starting these workflows, make sure you've completed the [Getting Started](getting-started/index.md) section — particularly [Installation](getting-started/installation.md). Configuration is covered in the first subsection below.
