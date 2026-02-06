# Reading Paths

Choose the path that matches your goal:

**New User** — Learn Karenina from the ground up:

> [Installation](../02-installation/index.md) → [Quick Start](../notebooks/quickstart.ipynb) → [Core Concepts](../core-concepts.md) → [Creating Benchmarks](../05-creating-benchmarks/index.md) → [Running Verification](../06-running-verification/index.md) → [Analyzing Results](../07-analyzing-results/index.md)

**Power User** — Dive into advanced features:

> [Core Concepts](../core-concepts.md) → [Pipeline Internals](../11-advanced-pipeline/index.md) → [Adapter Architecture](../12-advanced-adapters/index.md)

**CLI User** — Use Karenina from the command line:

> [Installation](../02-installation/index.md) → [Configuration](../03-configuration/index.md) → [CLI Reference](../09-cli-reference/index.md)

**Contributor** — Extend Karenina with custom adapters or pipeline stages:

> [Adapter Architecture](../12-advanced-adapters/index.md) → [Contributing](../13-contributing/index.md)

---

## Getting Started

| Section | What You'll Learn |
|---------|-------------------|
| [Installation](../02-installation/index.md) | Requirements, install commands, optional dependencies, troubleshooting |
| [Quick Start](../notebooks/quickstart.ipynb) | Hands-on walkthrough from zero to a working benchmark |
| [Workspace Init](../02-installation/workspace-init.md) | Set up a project directory with `karenina init` |

## Core Concepts

| Section | What You'll Learn |
|---------|-------------------|
| [Overview](../core-concepts.md) | How all concepts fit together, with reading paths by experience level |
| [Templates vs Rubrics](../04-core-concepts/template-vs-rubric.md) | The two evaluation units: correctness (templates) vs quality (rubrics) |
| [Checkpoints](../04-core-concepts/checkpoints.md) | The JSON-LD benchmark format: questions, templates, rubrics, and metadata |
| [Answer Templates](../04-core-concepts/answer-templates.md) | Pydantic models that define how a Judge LLM evaluates correctness |
| [Rubrics](../04-core-concepts/rubrics/index.md) | Quality assessment with four trait types: LLM, regex, callable, metric |
| [Evaluation Modes](../04-core-concepts/evaluation-modes.md) | Template-only, template-and-rubric, and rubric-only evaluation |
| [Adapters](../04-core-concepts/adapters.md) | LLM backend interfaces: LangChain, Claude SDK, OpenRouter, and more |

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
