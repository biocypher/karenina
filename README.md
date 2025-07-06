# Karenina

Core benchmarking library for Large Language Models (LLMs).

## Overview

Karenina is a Python library that provides the foundational components for benchmarking and evaluating Large Language Models. It includes functionality for:

- **Question Extraction**: Extract and process questions from Excel, CSV, and TSV files
- **Answer Template Generation**: Generate structured answer templates using LLMs
- **LLM Interface**: Unified interface for multiple LLM providers (OpenAI, Google, Anthropic, OpenRouter)
- **Benchmarking**: Run verification workflows to test LLM responses
- **Evaluation**: Evaluate and score LLM-generated answers

## Installation

```bash
pip install karenina
```

For development:
```bash
pip install karenina[dev]
```

## Quick Start

### Extract Questions from a File

```python
from karenina.questions.extractor import extract_and_generate_questions

# Extract questions from Excel/CSV/TSV
extract_and_generate_questions(
    file_path="questions.xlsx",
    output_path="questions.py",
    question_column="Question",
    answer_column="Answer"
)
```

### Generate Answer Templates

```python
from karenina.answers.generator import generate_answer_template
from karenina.llm.interface import create_llm
from karenina.schemas.question_class import Question

# Create an LLM instance
llm = create_llm(
    provider="openai",
    model_name="gpt-4",
    api_key="your-api-key"
)

# Generate answer template for a question
question = Question(
    id="q1",
    question="What is the capital of France?",
    raw_answer="Paris"
)

template = await generate_answer_template(llm, question)
```

### Run Benchmarks

```python
from karenina.benchmark.runner import run_question_verification
from karenina.benchmark.models import VerificationConfig

# Configure verification
config = VerificationConfig(
    answering_model_provider="openai",
    answering_model_name="gpt-4",
    parsing_model_provider="openai",
    parsing_model_name="gpt-3.5-turbo"
)

# Run verification
result = await run_question_verification(
    question=question,
    answer_template=template,
    config=config
)
```

## Core Components

### Questions Module
- `extractor.py`: Extract questions from various file formats
- `reader.py`: Read and load question files

### Answers Module
- `generator.py`: Generate answer templates using LLMs
- `reader.py`: Read and parse answer template files

### LLM Module
- `interface.py`: Unified LLM interface and session management
- `providers.py`: Provider-specific implementations

### Benchmark Module
- `runner.py`: Execute benchmark verification workflows
- `verifier.py`: Validate LLM responses against templates
- `models.py`: Data models for benchmarking
- `verification/`: Modular verification components

### Schemas Module
- `question_class.py`: Question data model
- `answer_class.py`: Base answer template class

### Utils Module
- `code_parser.py`: Parse code blocks from LLM responses
- `text_utils.py`: Text processing utilities

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/karenina.git
cd karenina

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=karenina

# Run specific test file
pytest tests/test_specific.py
```

### Code Quality

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Type checking
mypy src/karenina
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
