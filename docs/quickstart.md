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

### Option 1: Simplified Workflow with Benchmark Class (Recommended)

The Benchmark class provides a streamlined interface for the complete workflow:

```python
from karenina.benchmark import Benchmark

# Create a new benchmark
benchmark = Benchmark.create(
    name="Quick Start Assessment",
    description="Getting started with Karenina",
    version="1.0.0"
)

# Add questions directly
benchmark.add_question(
    question="What is the capital of France?",
    raw_answer="Paris is the capital of France"
)

benchmark.add_question(
    question="What is 2 + 2?",
    raw_answer="2 + 2 equals 4"
)

# Generate templates for all questions
# benchmark.generate_all_templates(model="gpt-4", model_provider="openai")

# Check progress
print(f"Benchmark has {len(benchmark)} questions")
print(f"Progress: {benchmark.get_progress():.1f}%")

# Save for GUI compatibility or later use
benchmark.save("quick_start_benchmark.jsonld")

# Run verification (when templates are ready)
# config = VerificationConfig(...)
# results = benchmark.run_verification(config)
```

### Option 2: Traditional Step-by-Step Workflow

For more control over each step:

#### 1. Extract Questions

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

#### 2. Generate Answer Templates

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

#### 3. Run Verification

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

## Working with GUI Exports

Load benchmarks created in the Karenina GUI:

```python
# Load a benchmark exported from the GUI
gui_benchmark = Benchmark.load("gui_exported_benchmark.jsonld")

# Inspect the loaded benchmark
print(f"Loaded: {gui_benchmark.name}")
print(f"Questions: {len(gui_benchmark)}")
print(f"Finished: {gui_benchmark.finished_count}")

# Add more questions programmatically
gui_benchmark.add_question(
    "What is machine learning?",
    "Machine learning is a subset of AI that learns from data"
)

# Save back for GUI import
gui_benchmark.save("enhanced_benchmark.jsonld")
```

## Complete Example

Here's a complete workflow from file to verification results:

```python
from karenina.benchmark import Benchmark
from karenina.questions.extractor import extract_questions_from_file

# Create benchmark from Excel file
questions = extract_questions_from_file(
    "data/sample_questions.xlsx",
    "Question",
    "Answer"
)

benchmark = Benchmark.create("Complete Example")

# Add extracted questions
for question in questions[:5]:  # First 5 questions
    benchmark.add_question(
        question=question.question,
        raw_answer=question.raw_answer,
        custom_metadata={"source": "sample_questions.xlsx"}
    )

# Add metadata
benchmark.set_custom_property("domain", "general_knowledge")
benchmark.creator = "Quick Start Tutorial"

# Check readiness
print(f"Benchmark readiness: {benchmark.get_progress():.1f}%")
health = benchmark.get_health_report()
print(f"Health score: {health['health_score']}/100")

# Save the benchmark
benchmark.save("complete_example.jsonld")
print("Saved benchmark ready for template generation and verification")
```

## Next Steps

- See [Examples](examples.md) for comprehensive usage examples
- Learn [Benchmark Management](guides/benchmark-management.md) for advanced features
- Understand [JSON-LD Format](guides/jsonld-format.md) for GUI interoperability
- Check [Architecture](architecture.md) to understand system design
- Review [API Reference](api-reference.md) for detailed function documentation
- Read [Development Guide](development.md) if you want to contribute
