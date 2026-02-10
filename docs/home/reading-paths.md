# Reading Paths

Choose the path that matches your goal:

**New User** — Learn Karenina from the ground up:

> [Installation](../getting-started/installation.md) → [Quick Start](../notebooks/quickstart.ipynb) → [Core Concepts](../core-concepts.md) → [Creating Benchmarks](../05-creating-benchmarks/index.md) → [Running Verification](../06-running-verification/index.md) → [Analyzing Results](../07-analyzing-results/index.md)

**Power User** — Dive into advanced features:

> [Core Concepts](../core-concepts.md) → [Pipeline Internals](../11-advanced-pipeline/index.md) → [Adapter Architecture](../12-advanced-adapters/index.md)

**CLI User** — Use Karenina from the command line:

> [Installation](../getting-started/installation.md) → [Configuration](../03-configuration/index.md) → [CLI Reference](../09-cli-reference/index.md)

**Contributor** — Extend Karenina with custom adapters or pipeline stages:

> [Adapter Architecture](../12-advanced-adapters/index.md) → [Contributing](../13-contributing/index.md)

---

## Getting Started

| Section | What You'll Learn |
|---------|-------------------|
| [Installation](../getting-started/installation.md) | Requirements, install commands, optional dependencies, troubleshooting |
| [Quick Start](../notebooks/quickstart.ipynb) | Hands-on walkthrough from zero to a working benchmark |
| [Workspace Init](../getting-started/workspace-init.md) | Set up a project directory with `karenina init` |

## Core Concepts

| Section | What You'll Learn |
|---------|-------------------|
| [Overview](../core-concepts.md) | How all concepts fit together, ordered by pipeline flow |
| [Questions & Benchmarks](../core_concepts/questions-and-benchmarks.md) | The central objects: questions bundled with templates, rubrics, and metadata |
| [Checkpoints](../core_concepts/checkpoints.md) | The JSON-LD benchmark format: questions, templates, rubrics, and metadata |
| [Templates vs Rubrics](../core_concepts/template-vs-rubric.md) | The two evaluation units: correctness (templates) vs quality (rubrics) |
| [Answer Templates](../core_concepts/answer-templates.md) | Pydantic models that define how a Judge LLM evaluates correctness |
| [Rubrics](../core_concepts/rubrics/index.md) | Quality assessment with four trait types: LLM, regex, callable, metric |
| [Evaluation Modes](../core_concepts/evaluation-modes.md) | Template-only, template-and-rubric, and rubric-only evaluation |
| [Verification Pipeline](../core_concepts/verification-pipeline.md) | The 13-stage engine that executes evaluation end to end |
| [Prompt Assembly](../core_concepts/prompt-assembly.md) | How prompts are constructed for pipeline LLM calls |
| [Results & Scoring](../core_concepts/results-and-scoring.md) | What verification produces: pass/fail, scores, traits, and metrics |
| [Adapters](../core_concepts/adapters.md) | LLM backend interfaces: LangChain, Claude SDK, OpenRouter, and more |

## Workflows

| Section | What You'll Learn |
|---------|-------------------|
| [Configuration](../03-configuration/index.md) | Configuration hierarchy: CLI args, presets, environment variables, defaults |
| [Creating Benchmarks](../05-creating-benchmarks/index.md) | Author questions, write templates, define rubrics, and save checkpoints |
| [Running Verification](../06-running-verification/index.md) | Configure and execute evaluation via Python API or CLI |
| [Analyzing Results](../07-analyzing-results/index.md) | Inspect results, build DataFrames, export data, and iterate |

## Reference

| Section | What You'll Learn |
|---------|-------------------|
| [CLI Reference](../09-cli-reference/index.md) | Complete documentation for all CLI commands |
| [Configuration Reference](../10-configuration-reference/index.md) | Exhaustive tables for all configuration options |

## Advanced

| Section | What You'll Learn |
|---------|-------------------|
| [Pipeline Internals](../11-advanced-pipeline/index.md) | The 13-stage verification pipeline, deep judgment, and prompt assembly |
| [Adapter Architecture](../12-advanced-adapters/index.md) | Ports and adapters pattern, custom adapter creation, MCP deep dive |

## Contributing

| Section | What You'll Learn |
|---------|-------------------|
| [Contributing Guide](../13-contributing/index.md) | How to create adapters, extend the pipeline, and contribute |
