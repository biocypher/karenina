# Karenina Documentation

Karenina is a comprehensive benchmarking system for Large Language Models (LLMs) that provides structured evaluation workflows through question extraction, answer template generation, and benchmark verification.

## Quickstart

See Quickstart for a minimal Benchmark-only example, or jump to the Benchmark Guide for deeper usage.

## Core Architecture

Karenina follows a three-stage pipeline:

1. **Question Extraction**: Process files (Excel, CSV, TSV) to extract structured questions
2. **Answer Template Generation**: Use LLMs to create Pydantic validation schemas
3. **Benchmark Verification**: Evaluate model responses against structured templates

## Key Features

- **Unified Benchmark Management**: High-level Benchmark class for complete workflow orchestration
- **JSON-LD Format**: Standardized linked data format with schema.org vocabulary
- **GUI Interoperability**: Full bidirectional compatibility between Python library and GUI
- **Multi-format Support**: Excel, CSV, TSV file processing
- **LLM Provider Abstraction**: Unified interface for OpenAI, Google, Anthropic, OpenRouter
- **Structured Validation**: Pydantic-based answer templates with type checking
- **Session Management**: Stateful conversation handling for interactive workflows
- **Code Generation**: Automatic Python file generation for questions and templates
- **Rubric Evaluation**: Qualitative assessment beyond correctness checking
- **Multi-Model Testing**: Compare multiple LLM configurations with replicates
- **Manual Verification**: Human-in-the-loop validation workflows
- **Comprehensive Metadata**: Track everything from creation to evaluation
- **Health Checks**: Built-in validation and readiness assessment
- **Multiple Export Formats**: CSV, Markdown, JSON, and Python file exports

## Documentation

### Getting Started
- [Installation](installation.md)
- [Quickstart](quickstart.md)
- [Examples](examples.md)

### Technical Documentation
- [Architecture](architecture.md)
- [API Reference](api-reference.md)
- [Development Guide](development.md)

### Module API Reference

- Benchmark, Core, Verification, Models: see API Reference anchors
- `questions`, `answers`, `schemas`, `llm`, `utils`: see API Reference

### User Guides

- [Benchmark Management](guides/benchmark-management.md): Complete Benchmark class usage guide
- [JSON-LD Format](guides/jsonld-format.md): Understanding the standardized data format
- [Question Extraction](guides/question-extraction.md): File processing workflows
- [Answer Generation](guides/answer-generation.md): Template creation process
- [Benchmark Execution](guides/benchmark-execution.md): Evaluation procedures
- [LLM Providers](guides/llm-providers.md): Provider configuration and usage

## Quick Navigation

| What do you want to do? | Where to look |
|-------------------------|---------------|
| Get started quickly | [Quick Start](quickstart.md) |
| See code examples | [Examples](examples.md) |
| **Create and manage benchmarks** | **[Benchmark Management Guide](guides/benchmark-management.md)** |
| **Work with JSON-LD format** | **[JSON-LD Format Guide](guides/jsonld-format.md)** |
| **Integrate Python with GUI** | **[Integration Examples](examples.md#working-with-benchmarks)** |
| Understand the architecture | [Architecture](architecture.md) |
| Look up a function | [API Reference](api-reference.md) |
| Contribute code | [Development Guide](development.md) |
| Extract questions from files | [Question Extraction Guide](guides/question-extraction.md) |
| Generate answer templates | [Answer Generation Guide](guides/answer-generation.md) |
| Run benchmarks | [Benchmark Execution Guide](guides/benchmark-execution.md) |
| Configure LLM providers | [LLM Providers Guide](guides/llm-providers.md) |
