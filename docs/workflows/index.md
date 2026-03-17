# Workflows

This section is about *doing* — step-by-step guides that take you from a starting point to a concrete outcome. Each page assumes you understand the relevant concepts (or links to them) and focuses on how to accomplish a specific task. For the underlying mental models, see [Core Concepts](../core_concepts/index.md). For exhaustive option tables, see [Reference](../reference/index.md).

---

## In This Section

| Workflow | What You'll Do |
|----------|---------------|
| [Configuration](configuration/index.md) | Set up the configuration hierarchy: CLI args, presets, environment variables, defaults |
| [Evaluating with TaskEval](../notebooks/task-eval/index.ipynb) | Evaluate pre-recorded agent traces against templates and rubrics |
| [Scenarios](../notebooks/scenarios/sycophancy-tutorial.ipynb) | Build and run a multi-turn scenario benchmark with branching paths and outcome criteria |
| [Creating Benchmarks](creating-benchmarks/index.md) | Author questions, write templates, define rubrics, and save checkpoints |
| [Running Verification](running-verification/index.md) | Configure and execute evaluation via Python API or CLI |
| [Analyzing Results](analyzing-results/index.md) | Inspect results, build DataFrames, export data, and iterate |

---

## End-to-End Flows

### Benchmark Flow (closed-loop)

```
Configure            Create Benchmark          Run Verification          Analyze Results
──────────────  →    ─────────────────    →    ─────────────────    →    ─────────────────
Set up env vars      Create checkpoint         Load benchmark            Explore result structure
Configure presets    Add questions             Configure models          Filter and group
                     Write templates           Choose eval mode          Build DataFrames
                     Define rubrics            Execute pipeline          Export and iterate
                     Save (.jsonld)            Collect results
```

### Scenario Flow (multi-turn, closed-loop)

```
Configure            Build Scenario Graph      Run Verification          Inspect Outcomes
──────────────  →    ─────────────────    →    ─────────────────    →    ─────────────────
Set up env vars      Define nodes (questions)  Load benchmark            Evaluate outcome criteria
Configure models     Define edges (conditions) Configure models          Inspect per-turn results
                     Add outcome criteria      Execute pipeline          Export results
                     add_scenario()            Collect results
```

### TaskEval Flow (open-loop)

```
Configure            Log Outputs               Attach Criteria           Evaluate & Inspect
──────────────  →    ─────────────────    →    ─────────────────    →    ─────────────────
Set up env vars      log() plain text          add_template(Answer)      evaluate(config)
Configure models     log_trace() messages      add_rubric(Rubric)        Inspect results
                     Scope by step_id          Scope by step_id          Export JSON/Markdown
```

Each workflow section has an overview page with a visual diagram, followed by dedicated pages for each step. Pages include executable notebook examples where applicable.

---

## Prerequisites

Before starting these workflows, make sure you've completed the [Getting Started](../getting-started/index.md) section — particularly [Installation](../getting-started/installation.md). Configuration is covered in the first subsection below.
