# Karenina

<div align="center">

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue)](http://mypy-lang.org/)

**A comprehensive benchmarking system for Large Language Models (LLMs)**

[Installation](#-installation) • [Quick Start](#-quick-start) • [Features](#-features) • [Documentation](#-documentation) • [Contributing](#-contributing)

</div>

---

## 📑 Table of Contents

- [About Karenina](#-about-karenina)
- [Architecture](#%EF%B8%8F-architecture)
- [Understanding the Problem](#-the-problem)
  - [Approach 1: Constrained Output](#1-constrain-the-answering-models-output)
  - [Approach 2: LLM as Judge](#2-use-an-llm-as-a-judge-free-text-evaluation)
  - [The Karenina Strategy](#-the-karenina-strategy)
- [Quick Start](#-quick-start)
- [Command-Line Interface](#-command-line-interface)
- [Why Templates](#-why-templates)
- [Templates vs Rubrics](#-templates-vs-rubrics)
- [Features](#-features)
- [Installation](#-installation)
- [Documentation](#-documentation)
- [Contributing](#-contributing)

---


## 🎯 About Karenina

Karenina is a framework designed to standardize domain expertise and concepts into runnable benchmarks. The core challenge Karenina addresses is: *making the formulation of domain-specific benchmarks accessible to non-LLM-technical experts, allowing them to focus their time and expertise on knowledge rather than infrastructure.*

**Key Concepts:**

- **Benchmarks** are expressed as **parametrizable code templates**, which can be evaluated with an **LLM-as-a-judge** model to evaluate performance
- **Standardized schema** (building on existing standards such as *schema.org*) enables rich, consistent, and extensible benchmark definitions
- **Tools to generate benchmarks at scale** while maintaining quality and consistency
- **JSON-LD format** enables seamless integration between Python library and GUI interface
- **Utilities to run and manage benchmarks**, although its primary focus remains on standardization and accessibility rather than execution infrastructure

At the heart of Karenina are two key concepts: **templates** and **rubrics**. Templates verify factual correctness through structured answer parsing, while rubrics assess qualitative traits, format compliance, and quantitative metrics.

## 🏗️ Architecture

Karenina is a **standalone Python library** that can be used independently for all benchmarking workflows through Python code.

### Graphical User Interface

To guarantee **additional accessibility** to the framework, a **web-based graphical interface** is available for users who prefer not to work with code. This no-code interface covers most features provided by the backend, including:

- **Visual question and metadata extraction** from files (Excel, CSV, TSV)
- **Template generation** with interactive preview and editing
- **No-code rubric curation** (LLM-based, regex, and metric traits)
- **Checkpointing and verification execution** with real-time progress monitoring
- **Results visualization** and export management

The GUI makes the Karenina framework accessible to domain experts, curators, and non-technical users who want to create and run benchmarks without writing Python code.

**Implementation**: The graphical interface is built using two companion packages:
- [karenina-server](https://github.com/biocypher/karenina-server) - Exposes the karenina backend as a FastAPI-based REST API
- [karenina-gui](https://github.com/biocypher/karenina-gui) - TypeScript/React web application providing the user interface

**Note**: Coordination and deployment instructions for the full web-based stack are still a work in progress and will be released soon.

## 🤔 The Problem

Let us introduce how Karenina approaches the problem of LLM benchmarking by considering a simple example: we want to task an LLM with a simple **multiple-choice question**:

```python
question = "Which protein regulates programmed cell death (apoptosis)?"
possible_answers = ["BCL2", "p53", "Insulin", "Hemoglobin"]
```

When we query a standard LLM, it usually responds in **free text** (e.g., `"BCL2 is the protein that regulates apoptosis by preventing cell death."`). To evaluate such an answer programmatically we could use the following approaches:

### 1. Constrain the answering model's output

We directly instruct the answering model to return a response in a machine-friendly format.

**Example prompt:**

```text
You are answering a multiple-choice question.
Return only the letter of your choice.

Question: Which protein regulates programmed cell death (apoptosis)?
Options:
A) BCL2
B) p53
C) Insulin
D) Hemoglobin

Answer:
```

**Model output:** `A`

The main advantage of this approach is its simplicity and reliability: once the model respects the instruction, evaluation can be fully automated with minimal overhead. However, its weakness lies in the fragility of prompt adherence. Many general-purpose LLMs do not always comply with rigid output constraints, especially across diverse domains or when questions become complex. In practice, this means users must design very careful prompts and may still face occasional formatting failures. Moreover, every time we have a different answer/question format we may need to come up with different dedicated prompting and parsing strategies.

### 2. Use an LLM as a judge (free-text evaluation)

Instead of constraining the answering model, we can keep its output free-form and rely on a **judge LLM** to interpret it.

**Example:**

* **Answering model output:** `"BCL2 is an anti-apoptotic protein that prevents cell death."`
* **Judge model prompt:**
  ```text
  The following is a student's answer to a multiple-choice question.
  Question: Which protein regulates programmed cell death (apoptosis)?
  Options: BCL2, p53, Insulin, Hemoglobin.
  Student's answer: "BCL2 is an anti-apoptotic protein that prevents cell death."
  Which option does this correspond to? Provide a justification.
  ```
* **Judge model output:** `"The student clearly selected BCL2, which is correct as it regulates apoptosis."`

The advantage here is flexibility: the answering model is free to behave naturally, without tight formatting constraints, which is particularly useful in open-ended or exploratory settings. However, this shifts the ambiguity to the judge's response, which is also often free text. While the judge usually interprets correctly, the result again requires parsing, and subtle differences in wording may cause errors or inconsistencies. Thus, while this strategy increases robustness to different kinds of answers, it does so at the cost of reintroducing unstructured evaluation one step later.

## 🧠 The Karenina Strategy

To reduce ambiguity, Karenina adopts a **third approach** that combines the advantages of both approaches:

- The **answering model** remains unconstrained, generating natural free text
- The **judge model** is required to return results in a **structured format** (JSON), validated through a Pydantic class

This setup allows the judge to flexibly interpret free text while ensuring that its own output remains standardized and machine-readable.

### Example Workflow

**1. Define a Pydantic template:**

```python
from karenina.domain.answers import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    answer: str = Field(description="The name of the protein mentioned in the response")

    def model_post_init(self, __context):
        self.correct = {"answer": "BCL2"}

    def verify(self) -> bool:
        return self.answer.strip().upper() == self.correct["answer"].strip().upper()
```

**Key aspects:**
- The `answer` attribute uses `Field` description to guide the judge
- The `verify` method implements custom validation logic

**2. Answering model generates free text:**

```
"BCL2 is the protein that regulates apoptosis by preventing cell death."
```

**3. Judge model parses into structured format:**

```python
from langchain_core.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=Answer)
prompt = parser.get_format_instructions()
prompt += "\n LLM Answer: BCL2 is the protein that regulates apoptosis by preventing cell death."

judge_answer = llm.invoke(prompt)
```

**Judge output (structured JSON):**

```json
{"answer": "BCL2"}
```

**4. Verification step:**

```python
populated_answer = Answer(**judge_answer)
result = populated_answer.verify()  # True
```

## 🚀 Quick Start

Get started with Karenina in just a few minutes! This example demonstrates the core workflow: create a benchmark, add questions, generate templates, and run verification.

### 1. Create a Benchmark

```python
from karenina import Benchmark

# Create a new benchmark
benchmark = Benchmark.create(
    name="Genomics Knowledge Benchmark",
    description="Testing LLM knowledge of genomics and molecular biology",
    version="1.0.0",
    creator="Your Name"
)
```

### 2. Add Questions

```python
# Add questions with answers
questions = [
    ("How many chromosomes are in a human somatic cell?", "46"),
    ("What is the approved drug target of Venetoclax?", "BCL2"),
    ("How many protein subunits does hemoglobin A have?", "4")
]

question_ids = []
for q, a in questions:
    qid = benchmark.add_question(
        question=q,
        raw_answer=a,
        author={"name": "Bio Curator"}
    )
    question_ids.append(qid)
```

**Note:** You can also extract questions from Excel, CSV, or TSV files. See [Adding Questions](docs/using-karenina/adding-questions.md) for file extraction examples.

### 3. Generate Templates Automatically

```python
from karenina.schemas import ModelConfig

# Configure the LLM for template generation
model_config = ModelConfig(
    id="gpt-4.1-mini",
    model_provider="openai",
    model_name="gpt-4.1-mini",
    temperature=0.1,
    interface="langchain"
)

# Generate templates for all questions
benchmark.generate_all_templates(model_config=model_config)
```

**Note:** Templates can also be written manually for complex custom logic. See [Templates Guide](docs/using-karenina/templates.md) for details.

### 4. Add Rubrics (Optional)

```python
from karenina.schemas import RubricTrait

# Create a global rubric to assess answer quality
benchmark.create_global_rubric(
    name="Answer Quality",
    traits=[
        RubricTrait(
            name="Conciseness",
            description="Rate how concise the answer is (1-5)",
            kind="score"
        )
    ]
)
```

### 5. Run Verification

```python
from karenina.schemas import VerificationConfig

# Configure verification
config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    rubric_enabled=True
)

# Run verification
results = benchmark.run_verification(config)

# Analyze results
passed = sum(1 for r in results.values() if r.verify_result)
print(f"Pass Rate: {(passed/len(results)*100):.1f}%")
```

### 6. Save and Export

```python
# Save benchmark checkpoint
benchmark.save("genomics_benchmark.jsonld")

# Export results to CSV
from pathlib import Path
benchmark.export_verification_results_to_file(
    file_path=Path("results.csv"),
    format="csv"
)
```

Congratulations! You've created your first Karenina benchmark with automatic template generation and rubric-based evaluation.

**Next steps**: Explore the [complete tutorial](docs/quickstart.md) for:
- Question-specific rubrics (regex and metric-based)
- File extraction from Excel/CSV
- Multiple model comparison
- Few-shot prompting
- Result analysis and visualization

## 💻 Command-Line Interface

For users who prefer working from the terminal, Karenina provides a comprehensive CLI for running verifications without writing Python code. The CLI is ideal for automation, CI/CD pipelines, and quick testing.

### Basic Usage

```bash
# Run verification with a preset configuration
karenina verify checkpoint.jsonld --preset default.json --verbose

# Run with CLI arguments only (no preset required)
karenina verify checkpoint.jsonld \
  --answering-model gpt-4.1-mini \
  --parsing-model gpt-4.1-mini \
  --output results.csv

# Override preset values with CLI flags
karenina verify checkpoint.jsonld \
  --preset default.json \
  --answering-model gpt-4o \
  --questions 0-5

# Interactive configuration builder
karenina verify checkpoint.jsonld --interactive --mode basic
```

### Preset Management

```bash
# List available presets
karenina preset list

# Show preset configuration
karenina preset show gpt-oss

# Delete a preset
karenina preset delete old-config
```

### Key Features

- **Flexible Configuration**: Use presets, CLI arguments, or interactive mode
- **Question Filtering**: Select specific questions by index or ID (e.g., `0-5`, `0,2,4`)
- **Multiple Output Formats**: Export results to JSON or CSV with comprehensive metadata
- **Progress Monitoring**: Real-time progress bars with pass/fail indicators
- **Fail-Fast Validation**: Validates inputs before running to avoid wasted API calls
- **CI/CD Ready**: Easy integration with GitHub Actions and other automation tools

### Configuration Hierarchy

The CLI supports flexible configuration with clear precedence:

**CLI flags > Preset values > Environment variables > Defaults**

This means you can use presets for base configuration and override specific values with CLI arguments as needed.

For complete CLI documentation, including all options, examples, and CI/CD integration guides, see [CLI Verification](docs/using-karenina/cli-verification.md).

## 🎯 Why Templates

Templates play a central role in Karenina by standardizing how answers are parsed, verified, and evaluated. Their use provides several key benefits:

### 1. Unified Parsing and Evaluation

Templates allow parsing to happen **directly through the judge LLM**. The free-text answer from the answering model is mapped into a structured format (e.g., a Pydantic class), ensuring that:

* Evaluation logic is **bundled with the question-answer pair** itself
* The same benchmark can seamlessly accommodate **different answer formats** without custom code

### 2. Streamlined Benchmark Creation

Since LLMs are proficient at code generation, they can often **auto-generate Pydantic classes** from raw question-answer pairs. This means that large portions of benchmark creation can be partially automated, reducing manual effort while improving consistency.

### 3. Cognitive Offloading for the Judge

By embedding the evaluation schema in templates, the **judge LLM's task is simplified**. Instead of reasoning about both the content and the evaluation logic, the judge focuses only on interpreting the free-text answer and filling in the template.

### 4. Extensibility and Reusability

Templates make it straightforward to extend benchmarks:

* New tasks can be added by defining new templates without re-engineering downstream code
* The same evaluation logic can be reused across multiple benchmarks with minimal adaptation

### 5. Transparency and Debuggability

By encoding evaluation criteria into explicit, inspectable templates, benchmarks become more transparent. This allows developers to:

* **Audit** the evaluation rules directly
* **Debug** failures more easily by inspecting the structured outputs rather than opaque free text

---

## 🎨 Templates vs Rubrics

While templates excel at verifying **factual correctness**, many evaluation scenarios require assessing **qualitative traits**, format compliance, or quantitative metrics. This is where **rubrics** complement templates.

### Feature Comparison

| Aspect | Answer Templates | Rubrics |
|--------|-----------------|---------|
| **Purpose** | Verify factual correctness | Assess qualitative traits, format, and metrics |
| **Evaluation Method** | Programmatic field comparison | Four approaches:<br>• LLM judgment<br>• Regex patterns<br>• Custom Python functions<br>• Term extraction + metrics |
| **Best for** | Precise, unambiguous answers | Subjective qualities, format validation, custom logic, quantitative analysis |
| **Trait Types** | Single verification method | **Four types:**<br>• LLM-based (qualitative)<br>• Regex-based (format)<br>• Callable (custom Python)<br>• Metric-based (term extraction) |
| **Output** | Pass/fail per field | • Boolean (binary traits)<br>• Scores 1-5 (score traits)<br>• Precision/Recall/F1 (metric traits) |
| **Examples** | `"BCL2"`, `"46 chromosomes"` | • "Is the answer concise?" (LLM)<br>• Match email pattern (regex)<br>• Extract diseases for F1 score (metric) |
| **Scope** | Per question | Global or per question |

### Rubric Types

Karenina supports **four types of rubric traits**, each suited for different evaluation needs:

**1. LLM-Based Traits**

AI-evaluated qualitative assessments where a judge LLM evaluates subjective qualities:

- **Score-based (1-5):** "Rate the scientific accuracy of the answer"
- **Binary (pass/fail):** "Does the answer mention safety concerns?"

**2. Regex Pattern Traits**

Deterministic validation using regular expressions for format compliance:

- "Answer must contain a DNA sequence (pattern: `[ATCG]+`)"
- "Response must include enzyme names (pattern: `\w+ase\b`)"

**3. Callable Traits**

Custom Python functions for domain-specific evaluation logic:

- Word count validation: "Is the response between 50-500 words?"
- Custom scoring: "Count technical terms from a predefined list"
- Complex business rules that can't be expressed as regex

**4. Metric-Based Traits**

Quantitative evaluation using confusion matrix metrics:

- Define terms that SHOULD appear (True Positives)
- Define terms that SHOULD NOT appear (False Positives)
- System computes precision, recall, F1, and optionally specificity/accuracy

**When to use what:**

- Use **templates** when you need to verify specific factual content or structured data
- Use **LLM-based rubrics** for subjective quality assessment (clarity, conciseness, tone)
- Use **regex rubrics** for format compliance and deterministic keyword checks
- Use **callable rubrics** for custom logic that requires programmatic evaluation
- Use **metric rubrics** when evaluating classification accuracy by extracting and measuring term coverage
- Use **both together** for comprehensive evaluation covering correctness AND quality

[Learn more about Templates →](docs/using-karenina/templates.md) | [Learn more about Rubrics →](docs/using-karenina/rubrics.md)

## ✨ Features

Karenina provides comprehensive tools for every stage of the benchmarking workflow:

### Core Capabilities

- **Question Management**: Extract questions from files (Excel, CSV, TSV) with rich metadata support
- **Answer Templates**: Pydantic-based templates for structured evaluation and programmatic verification
- **Rubric Evaluation**: Assess qualitative traits using four types:
  - LLM-based traits (binary pass/fail or 1-5 scale)
  - Regex-based traits (pattern matching for format validation)
  - Callable traits (custom Python functions)
  - Metric-based traits (precision, recall, F1, accuracy)
- **Benchmark Verification**: Run evaluations with four supported interfaces:
  - `langchain` (OpenAI, Google Gemini, Anthropic Claude)
  - `openrouter` (OpenRouter platform)
  - `openai_endpoint` (OpenAI-compatible endpoints for local models)
  - `manual` (Manual trace replay for testing/debugging)

### Advanced Features

- **Deep-Judgment Parsing**: Extract verbatim excerpts, reasoning traces, and confidence scores with configurable modes (disabled, enable_all, per-trait custom)
- **Abstention Detection**: Identify when models refuse to answer questions
- **Embedding Check**: Semantic similarity fallback using SentenceTransformers to reduce false negatives
- **Few-Shot Prompting**: Configure examples globally or per question with flexible selection modes
- **Task-Centric Evaluation (TaskEval)**: Attach verification criteria to existing agent traces for evaluation without re-running
- **Multi-Model Comparison**: Run evaluations across multiple answering models in a single batch
- **Async Execution**: Parallel processing with configurable worker pools for faster batch runs
- **GEPA Integration**: Prompt optimization framework with train/test splitting, feedback generation, and improvement tracking
- **MCP Integration**: Support for Model Context Protocol servers and tool use tracking
- **Search-Enhanced Validation**: Tavily search integration for hallucination detection and evidence cross-referencing
- **Database Persistence**: SQLite storage with versioning and 10+ analytical views
- **Export & Reporting**: CSV and JSON formats for analysis with selective column export
- **Preset Management**: Save and reuse verification configurations with full hierarchy support

[View complete feature catalog →](docs/features.md)

## 📦 Installation

### Prerequisites

- Python 3.11 or higher
- Git
- `uv` (Python's fast package manager - recommended)

### Install uv

If you don't have `uv` installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For other installation methods, see [uv's documentation](https://docs.astral.sh/uv/getting-started/installation/).

### Install Karenina

**Note:** Karenina is not yet published to PyPI. Install from the GitHub repository:

```bash
# Clone the repository
git clone https://github.com/biocypher/karenina.git
cd karenina

# Install with uv (recommended)
uv pip install -e .

# Or use pip
pip install -e .
```

The `-e` flag installs in editable mode, allowing you to pull updates with `git pull` without reinstalling.

### Environment Setup

Configure API keys for LLM providers:

| Provider | Variable | Models |
|----------|----------|--------|
| OpenAI | `OPENAI_API_KEY` | GPT-4, GPT-4 mini |
| Google | `GOOGLE_API_KEY` | Gemini |
| Anthropic | `ANTHROPIC_API_KEY` | Claude |
| OpenRouter | `OPENROUTER_API_KEY` | Unified access |

**Recommended: Create a `.env` file in your project root**

```bash
OPENAI_API_KEY="sk-..."
GOOGLE_API_KEY="AIza..."
ANTHROPIC_API_KEY="sk-ant-..."
```

Then add `.env` to `.gitignore` to prevent committing secrets:

```bash
echo ".env" >> .gitignore
```

**Alternative: Export to your shell**

```bash
export OPENAI_API_KEY="sk-..."
```

**Note**: API keys can also be passed programmatically via `extra_kwargs` in `ModelConfig`. See the [Configuration Guide](docs/configuration.md) for all options including feature toggles, execution control, and database settings.

### Verify Installation

Test that Karenina is installed correctly:

```python
from karenina import Benchmark

# Create a simple benchmark
benchmark = Benchmark.create(
    name="test-benchmark",
    description="Installation verification",
    version="1.0.0"
)

print(f"✓ Karenina installed successfully!")
print(f"✓ Benchmark created: {benchmark.name}")
```

For detailed setup instructions, troubleshooting, and development installation, see the [Installation Guide](docs/install.md).

## 📚 Documentation

Ready to explore more of Karenina's capabilities? Check out our comprehensive documentation:

### Viewing Documentation Locally

You can view the full documentation with a live preview using MkDocs:

```bash
# From the karenina directory
uv run mkdocs serve
```

Then open your browser to `http://127.0.0.1:8000` to browse the documentation with full navigation and search.

### Getting Started
- [**Documentation Index**](docs/index.md) - Complete documentation overview with navigation
- [**Installation Guide**](docs/install.md) - Detailed setup instructions and requirements
- [**Quick Start Tutorial**](docs/quickstart.md) - Step-by-step guide to your first benchmark
- [**Features Overview**](docs/features.md) - Complete feature catalog

### User Guides
- [**Defining Benchmarks**](docs/using-karenina/defining-benchmark.md) - Benchmark creation and metadata
- [**Adding Questions**](docs/using-karenina/adding-questions.md) - File extraction and management
- [**Templates**](docs/using-karenina/templates.md) - Creating and customizing answer templates
- [**Rubrics**](docs/using-karenina/rubrics.md) - Evaluation criteria and trait types
- [**Verification**](docs/using-karenina/verification.md) - Running evaluations and analyzing results
- [**CLI Verification**](docs/using-karenina/cli-verification.md) - Command-line interface for automation
- [**Saving & Loading**](docs/using-karenina/saving-loading.md) - Checkpoints, database, and export

### Advanced Features
- [**Deep-Judgment**](docs/advanced/deep-judgment.md) - Extract detailed feedback with excerpts
- [**Few-Shot Prompting**](docs/advanced/few-shot.md) - Guide responses with examples
- [**Abstention Detection**](docs/advanced/abstention-detection.md) - Handle model refusals
- [**Embedding Check**](docs/advanced/embedding-check.md) - Semantic similarity fallback
- [**Presets**](docs/advanced/presets.md) - Save and reuse verification configurations

### Reference
- [**API Reference**](docs/api-reference.md) - Complete API documentation
- [**Configuration**](docs/configuration.md) - Environment variables and defaults
- [**Troubleshooting**](docs/troubleshooting.md) - Common issues and solutions

## 🤝 Contributing

We welcome contributions to Karenina! Please see our contributing guidelines for more information on how to get involved.
