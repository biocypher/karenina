# Workflows

This section provides step-by-step guides for the three main Karenina workflows: creating benchmarks, running verification, and analyzing results. Each workflow builds on the previous one.

---

## In This Section

| Workflow | What You'll Do |
|----------|---------------|
| [Creating Benchmarks](05-creating-benchmarks/index.md) | Author questions, write templates, define rubrics, and save checkpoints |
| [Running Verification](06-running-verification/index.md) | Configure and execute evaluation via Python API or CLI |
| [Analyzing Results](07-analyzing-results/index.md) | Inspect results, build DataFrames, export data, and iterate |

---

## End-to-End Flow

```
Create Benchmark          Run Verification          Analyze Results
─────────────────    →    ─────────────────    →    ─────────────────
Create checkpoint         Load benchmark            Explore result structure
Add questions             Configure models          Filter and group
Write templates           Choose eval mode          Build DataFrames
Define rubrics            Execute pipeline          Export and iterate
Save (.jsonld)            Collect results
```

Each workflow section has an overview page with a visual diagram, followed by dedicated pages for each step. Pages include executable notebook examples where applicable.

---

## Prerequisites

Before starting these workflows, make sure you've completed the [Getting Started](getting-started.md) section — particularly [Installation](02-installation/index.md) and [Configuration](03-configuration/index.md).
