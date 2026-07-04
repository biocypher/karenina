# Reading Paths

Choose the path that matches your goal:

**New User** — Learn Karenina from the ground up:

> [Installation](../getting-started/installation.md) → [Quick Start](../notebooks/quickstart.ipynb) → [Core Concepts](../core_concepts/index.md) → [Creating Benchmarks](../workflows/creating-benchmarks/index.md) → [Running Verification](../workflows/running-verification/index.md) → [Analyzing Results](../workflows/analyzing-results/index.md)

**TaskEval User**: Evaluate existing outputs (agent traces, external text):

> [Installation](../getting-started/installation.md) → [TaskEval](../notebooks/core_concepts/task-eval.ipynb) → [TaskEval Workflow](../notebooks/task-eval/index.ipynb) → [Answer Templates](../notebooks/core_concepts/answer-templates.ipynb) → [Rubrics](../core_concepts/rubrics/index.md) → [Analyzing Results](../workflows/analyzing-results/index.md)

**Power User** — Dive into advanced features:

> [Core Concepts](../core_concepts/index.md) → [Pipeline Internals](../advanced-pipeline/index.md) → [Adapter Architecture](../advanced-adapters/index.md)

**CLI User** — Use Karenina from the command line:

> [Installation](../getting-started/installation.md) → [Configuration](../workflows/configuration/index.md) → [CLI Reference](../reference/cli/index.md)

**Contributor** — Extend Karenina with custom adapters or pipeline stages:

> [Adapter Architecture](../advanced-adapters/index.md) → [Contributing](../contributing.md)

---

## Getting Started

| Section | What You'll Learn |
|---------|-------------------|
| [Installation](../getting-started/installation.md) | Requirements, install commands, optional dependencies, troubleshooting |
| [Quick Start: Benchmark](../notebooks/quickstart.ipynb) | Hands-on walkthrough from zero to a working benchmark |
| [Quick Start: TaskEval](../notebooks/quickstart-taskeval.ipynb) | Evaluate pre-recorded outputs (agent traces, external text) |
| [Workspace Init](../getting-started/workspace-init.md) | Set up a project directory with `karenina init` |

## Core Concepts

| Section | What You'll Learn |
|---------|-------------------|
| [Overview](../core_concepts/index.md) | How all concepts fit together, ordered by pipeline flow |
| [Questions & Benchmarks](../core_concepts/questions-and-benchmarks/index.md) | The central objects: questions bundled with templates, rubrics, and metadata |
| [Checkpoints](../core_concepts/questions-and-benchmarks/checkpoints.md) | The JSON-LD benchmark format: questions, templates, rubrics, and metadata |
| [Answer Templates](../notebooks/core_concepts/answer-templates.ipynb) | Pydantic models that define how a Judge LLM evaluates correctness |
| [Rubrics](../core_concepts/rubrics/index.md) | Quality assessment with five trait types: LLM, Regex, Callable, Metric, Agentic |
| [Templates vs Rubrics](../notebooks/core_concepts/template-vs-rubric.ipynb) | When to use which evaluation unit, and when to use both together |
| [Evaluation Modes](../notebooks/core_concepts/evaluation-modes.ipynb) | Template-only, template-and-rubric, and rubric-only evaluation |
| [Verification Pipeline](../notebooks/core_concepts/verification-pipeline.ipynb) | The 13-stage engine that executes evaluation end to end |
| [Prompt Assembly](../notebooks/core_concepts/prompt-assembly.ipynb) | How prompts are constructed for pipeline LLM calls |
| [Results & Scoring](../core_concepts/results-and-scoring.md) | What verification produces: pass/fail, scores, traits, and metrics |
| [Adapters](../core_concepts/adapters.md) | LLM backend interfaces: LangChain, Claude SDK, OpenRouter, and more |

## Workflows

| Section | What You'll Learn |
|---------|-------------------|
| [Configuration](../workflows/configuration/index.md) | Configuration hierarchy: CLI args, presets, environment variables, defaults |
| [Evaluating with TaskEval](../notebooks/task-eval/index.ipynb) | Evaluate pre-recorded agent traces against templates and rubrics |
| [Creating Benchmarks](../workflows/creating-benchmarks/index.md) | Author questions, write templates, define rubrics, and save checkpoints |
| [Running Verification](../workflows/running-verification/index.md) | Configure and execute evaluation via Python API or CLI |
| [Analyzing Results](../workflows/analyzing-results/index.md) | Inspect results, build DataFrames, export data, and iterate |

## Reference

| Section | What You'll Learn |
|---------|-------------------|
| [CLI Reference](../reference/cli/index.md) | Complete documentation for all CLI commands |
| [Configuration Reference](../reference/configuration/index.md) | Exhaustive tables for all configuration options |

## Advanced

| Section | What You'll Learn |
|---------|-------------------|
| [Pipeline Internals](../advanced-pipeline/index.md) | The 13-stage verification pipeline, deep judgment, and prompt assembly |
| [Adapter Architecture](../advanced-adapters/index.md) | Ports and adapters pattern, custom adapter creation, MCP deep dive |

## Contributing

| Section | What You'll Learn |
|---------|-------------------|
| [Contributing Guide](../contributing.md) | How to create adapters, extend the pipeline, and contribute |
