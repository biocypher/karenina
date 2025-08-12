# Quick Start Guide

Get up and running with Karenina in minutes.

## Installation

```bash
# Using pip
pip install karenina

# Using uv (recommended)
uv add karenina
```

## Environment Setup

Set up your LLM provider API keys:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export GOOGLE_API_KEY="your-google-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

## Basic Usage

### 1. Extract Questions

```python
from karenina.questions.extractor import extract_and_generate_questions

# Extract questions from Excel/CSV/TSV
questions = extract_and_generate_questions(
    file_path="data/questions.xlsx",
    output_path="questions.py",
    question_column="Question",
    answer_column="Answer"
)
```

### 2. Generate Answer Templates

```python
from karenina.answers.generator import generate_answer_template

# Generate validation template
template_code = generate_answer_template(
    question="What is the capital of France?",
    raw_answer="Paris",
    model="gpt-4",
    model_provider="openai"
)
```

### 3. Run Verification

```python
from karenina.benchmark.models import VerificationConfig, ModelConfiguration
from karenina.benchmark.verification.orchestrator import run_question_verification

# Configure models
config = VerificationConfig(
    answering_models=[
        ModelConfiguration(
            id="gpt4",
            model_provider="openai",
            model_name="gpt-4",
            temperature=0.1,
            interface="langchain",
            system_prompt="You are an expert assistant."
        )
    ],
    parsing_models=[
        ModelConfiguration(
            id="gpt35",
            model_provider="openai",
            model_name="gpt-3.5-turbo",
            temperature=0.0,
            interface="langchain",
            system_prompt="Parse responses accurately."
        )
    ]
)

# Run verification
results = run_question_verification(
    question_id="q1",
    question_text="What is the capital of France?",
    template_code=template_code,
    config=config
)
```

## Next Steps

- See [Examples](examples.md) for comprehensive usage examples
- Check [Architecture](architecture.md) to understand system design
- Review [API Reference](api-reference.md) for detailed function documentation
- Read [Development Guide](development.md) if you want to contribute
