# Karenina Documentation

Welcome to the Karenina documentation. Karenina is a framework for defining, running, and analyzing LLM benchmarks in a rigorous and reproducible way.

## Getting Started

- [Introduction](01-introduction/index.md) — What Karenina is and a quickstart example
- [Installation](02-installation/index.md) — Setup for all scenarios
- [Configuration](03-configuration/index.md) — Environment variables, presets, and workspace init

## Core Concepts

- [Checkpoints](04-core-concepts/checkpoints.md) — JSON-LD benchmark format
- [Answer Templates](04-core-concepts/answer-templates.md) — Structured evaluation via Pydantic models
- [Rubrics](04-core-concepts/rubrics/index.md) — Quality assessment with 4 trait types
- [Evaluation Modes](04-core-concepts/evaluation-modes.md) — Template-only, rubric-only, or both
- [Adapters](04-core-concepts/adapters.md) — LLM backend interfaces

## Workflows

- [Creating Benchmarks](05-creating-benchmarks/index.md) — Author questions, templates, and rubrics
- [Running Verification](06-running-verification/index.md) — Execute evaluation pipelines
- [Analyzing Results](07-analyzing-results/index.md) — Inspect, export, and iterate on results

## Reference

- [CLI Reference](09-cli-reference/index.md) — Complete command documentation
- [Configuration Reference](10-configuration-reference/index.md) — All configuration options

## Advanced

- [Pipeline Internals](11-advanced-pipeline/index.md) — 13-stage verification pipeline
- [Adapter Architecture](12-advanced-adapters/index.md) — Ports, adapters, and MCP integration

## Contributing

- [Contributing Guide](13-contributing/index.md) — How to extend Karenina
