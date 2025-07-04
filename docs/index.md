# Karenina Documentation

Karenina is a comprehensive benchmarking system for Large Language Models (LLMs) that provides structured evaluation workflows through question extraction, answer template generation, and benchmark verification.

## Quick Start

```python
from karenina.questions.extractor import extract_and_generate_questions
from karenina.answers.generator import generate_answer_templates_from_questions_file
from karenina.benchmark.runner import run_benchmark

# Extract questions from file
questions = extract_and_generate_questions("data/questions.xlsx", "questions.py")

# Generate answer templates
templates = generate_answer_templates_from_questions_file("questions.py")

# Run benchmark evaluation
results = run_benchmark(questions, responses, templates)
```

## Core Architecture

Karenina follows a three-stage pipeline:

1. **Question Extraction**: Process files (Excel, CSV, TSV) to extract structured questions
2. **Answer Template Generation**: Use LLMs to create Pydantic validation schemas
3. **Benchmark Verification**: Evaluate model responses against structured templates

## Key Features

- **Multi-format Support**: Excel, CSV, TSV file processing
- **LLM Provider Abstraction**: Unified interface for OpenAI, Google, Anthropic, OpenRouter
- **Structured Validation**: Pydantic-based answer templates with type checking
- **Session Management**: Stateful conversation handling for interactive workflows
- **Code Generation**: Automatic Python file generation for questions and templates

## Module Overview

- [`llm`](api/llm.md): LLM interface and session management
- [`questions`](api/questions.md): File processing and question extraction
- [`answers`](api/answers.md): Template generation and validation
- [`benchmark`](api/benchmark.md): Evaluation and verification workflows
- [`schemas`](api/schemas.md): Data models and validation classes
- [`utils`](api/utils.md): Utility functions and helpers

## Guides

- [Question Extraction](guides/question-extraction.md): File processing workflows
- [Answer Generation](guides/answer-generation.md): Template creation process
- [Benchmark Execution](guides/benchmark-execution.md): Evaluation procedures
- [LLM Providers](guides/llm-providers.md): Provider configuration and usage
