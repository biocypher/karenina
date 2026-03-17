# Karenina

<div align="center">

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue)](http://mypy-lang.org/)

**Structured LLM evaluation: from factual Q&A to agentic coding tasks**

[About](#-about-karenina) • [The Problem](#-the-problem) • [Getting Started](#-getting-started) • [CLI](#-command-line-interface) • [Features](#-key-features) • [Architecture](#%EF%B8%8F-architecture) • [Installation](#-installation) • [Docs](#-documentation) • [Contributing](#-contributing)

</div>

> **⚠️ Experimental Project:** Karenina is still experimental and under active, fast-paced development. APIs and features may change without notice. Best effort has been applied in creating a correct set of documentation, however some errors and imprecisions may be present. If you encounter any, please [open an issue](https://github.com/biocypher/karenina/issues) on the GitHub repository and we will try to get them fixed as soon as possible.

---

## 📑 Table of Contents

- [About Karenina](#-about-karenina)
  - [Why This Approach](#why-this-approach)
- [The Problem](#-the-problem)
  - [Approach 1: Constrained Output](#1-constrain-the-answering-models-output)
  - [Approach 2: LLM as Judge](#2-use-an-llm-as-a-judge-free-text-evaluation)
- [The Karenina Strategy](#-the-karenina-strategy)
  - [Example Workflow](#example-workflow)
  - [Beyond Correctness: Rubric Evaluation](#beyond-correctness-rubric-evaluation)
- [Getting Started](#-getting-started)
- [Command-Line Interface](#-command-line-interface)
- [Key Features](#-key-features)
- [Architecture](#%EF%B8%8F-architecture)
- [Installation](#-installation)
- [Documentation](#-documentation)
- [Contributing](#-contributing)

---


## 🎯 About Karenina

Karenina is an open-source Python framework for defining, running, and sharing LLM evaluations. It covers the full evaluation spectrum: simple factual Q&A, tool-augmented interactions via [MCP](docs/core_concepts/mcp-overview.md), and fully [agentic coding and data analysis tasks](docs/core_concepts/agentic-evaluation.md) where both the answering model and the judge operate in a real workspace with file and code access.

It formalizes ground truth as structured **[answer templates](docs/core_concepts/answer-templates.md)**: Pydantic models that encode what a correct response looks like, letting a Judge LLM parse free-form responses into those schemas for programmatic verification. Combined with **[rubrics](docs/core_concepts/rubrics/index.md)** for quality assessment (LLM judgment, regex, callable, and metric traits), Karenina provides a flexible [evaluation pipeline](docs/core_concepts/verification-pipeline.md) from quick correctness checks to complex multi-trait scoring. It supports two modes: **[Benchmark](docs/getting-started/quickstart.md)** (closed-loop: define questions, generate responses, evaluate) and **[TaskEval](docs/core_concepts/task-eval.md)** (open-loop: supply pre-recorded outputs, evaluate with the same pipeline).

The core challenge Karenina addresses is making the formulation of domain-specific benchmarks accessible to non-LLM-technical experts, allowing them to focus their time and expertise on knowledge rather than infrastructure. LLM-assisted template generation automates most code writing, and a JSON-LD format (building on schema.org vocabularies) provides seamless portability between the Python library, REST API, and web GUI.

### Why This Approach

1. **Naturalistic evaluation.** Traditional benchmarks force models into artificial formats (multiple-choice letters, regex-compliant strings) that differ from real-world usage and signal to the model that it is being evaluated. In Karenina, the answering model is never constrained: it produces the same kind of response a real user would receive. A separate Judge LLM evaluates the natural response after the fact.

2. **Portable, self-contained benchmarks.** Each question carries its own verification logic and quality checks. A benchmark bundles questions, evaluation criteria, and metadata into a single [portable checkpoint](docs/core_concepts/questions-and-benchmarks/checkpoints.md) that anyone can reload, re-run against different models, or extend with new questions. Evaluation criteria travel with the data.

3. **Bootstrapped authoring.** LLMs can [auto-generate evaluation code](docs/workflows/creating-benchmarks/scaled-authoring.md) from a simple spreadsheet of questions and answers, bootstrapping benchmark creation in minutes. Quality checks are defined declaratively, so adding them requires no custom infrastructure.

4. **Expressivity.** Templates combine natural-language field descriptions with programmatic verification logic, allowing flexible definitions of what it means to "pass": multiple attributes of different types, combined with arbitrary rules (exact match, normalization, numeric tolerance, partial credit, or any custom Python logic).

5. **Agentic evaluation, not just Q&A.** Modern LLM deployments increasingly involve agents that write code, run tests, and produce file artifacts. Karenina evaluates these workflows natively: the answering model operates in a [workspace](docs/core_concepts/agentic-evaluation.md) with tool access, and an independent judge agent inspects the resulting artifacts (files created, tests passed, code compiled) rather than relying on the conversation trace alone. The same template and rubric primitives apply whether the task is a factual question or a multi-step coding challenge.

6. **Benchmarks that measure what you care about.** Public benchmarks create incentives for model providers to optimize for the test rather than for real-world usefulness. By lowering the cost of creating domain-specific evaluations, Karenina lets teams build internal suites that measure the capabilities that actually matter for their deployment. When anyone can spin up a benchmark on their own terms, evaluation becomes harder to game, creating a race to the top where genuine model improvement is the only winning strategy.

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
from karenina.schemas.entities import BaseAnswer, VerifiedField, BooleanMatch

class Answer(BaseAnswer):
    identifies_bcl2_as_target: bool = VerifiedField(
        description=(
            "True if the response identifies BCL2 (including Bcl-2, BCL-2, or "
            "B-cell lymphoma 2) as the direct pharmacological target of venetoclax."
        ),
        ground_truth=True,
        verify_with=BooleanMatch(),
    )
```

**Key aspects:**
- The `description` guides the Judge LLM on what to extract (here, a boolean judgment about BCL2 identification)
- `ground_truth` declares the expected value; `verify_with` selects the verification primitive for programmatic comparison
- A single template can mix multiple field types with different verification primitives

`VerifiedField` supports any type that Pydantic supports:

| Type | Use Case | Example |
|------|----------|---------|
| `str` | Names, terms, identifiers | Drug target, gene symbol |
| `int` | Counts, quantities | Number of chromosomes |
| `float` | Measurements, scores | Temperature, percentage |
| `bool` | Yes/no judgments | "Does the response mention X?" |
| `list[str]` | Multiple items | List of proteins, symptoms |
| `Literal[...]` | Fixed categories | Classification labels, mutation types |

See [Answer Templates](docs/core_concepts/answer-templates.md) for the full guide on field types, verification patterns, and writing good field descriptions.

**2. Answering model generates free text:**

```
"Venetoclax is a selective BCL2 inhibitor that acts as a BH3 mimetic,
binding directly to the BCL2 protein to restore apoptosis in cancer cells."
```

**3. Judge model parses into structured format:**

The framework sends the free-text response to the Judge LLM along with the template's JSON schema (derived automatically from the Pydantic class). The judge extracts the relevant information and returns structured JSON:

```json
{"identifies_bcl2_as_target": true}
```

**4. Verification step:**

```python
populated_answer = Answer(**judge_output)
result = populated_answer.verify()  # True
```

### Beyond Correctness: Rubric Evaluation

Templates and rubrics are not alternative ways of doing the same thing. They evaluate **orthogonal dimensions**. A response can pass its template (correct protein extracted) while failing a rubric trait (unclear reasoning). Conversely, a response can score well on rubric traits (concise, well-cited) while failing its template (wrong answer).

The key distinction is the **ground-truth boundary**. Templates live on the ground-truth side: `verify()` compares parsed fields against `self.correct`. Without ground truth, template verification has nothing to compare against. Rubrics live on the observable side: the evaluator judges properties visible in the response text itself, without access to the correct answer.

**Litmus test**: if the evaluator cannot make the judgment without knowing the correct answer, it belongs in the template. If the evaluator can judge by reading the response alone, it belongs in a rubric.

| Needs ground truth (use a template) | Observable in the response (use a rubric) |
|---|---|
| "Did the response identify BCL2 as the target?" | "Does the response cite specific trials or data?" |
| "Is the mechanism of action accurate?" | "Is the reasoning presented as a chain of steps?" |

Rubrics support five trait types (LLM traits have three sub-kinds: boolean, score, literal):

| Trait Type | Returns | LLM Required | Use Case |
|---|---|---|---|
| **[LLMRubricTrait](docs/core_concepts/rubrics/llm-traits.md)** (boolean) | `bool` | Yes | Binary quality judgment (safety, conciseness) |
| **[LLMRubricTrait](docs/core_concepts/rubrics/llm-traits.md)** (score) | `int` | Yes | Numeric rating within a configurable range |
| **[LLMRubricTrait](docs/core_concepts/rubrics/llm-traits.md)** (literal) | `int` | Yes | Classification into ordered categories (e.g., tone: formal/casual/technical) |
| **[RegexTrait](docs/core_concepts/rubrics/regex-traits.md)** | `bool` | No | Deterministic pattern matching (citations, format compliance) |
| **[CallableTrait](docs/core_concepts/rubrics/callable-traits.md)** | `bool` or `int` | No | Custom Python logic (word count, readability, structure checks) |
| **[MetricRubricTrait](docs/core_concepts/rubrics/metric-traits.md)** | metrics dict | Yes | Precision/recall/F1 over expected content items |
| **[AgenticRubricTrait](docs/core_concepts/agentic-evaluation.md#9-agentic-rubric-evaluation)** | `bool` or `int` | Yes | Agent investigates workspace artifacts before scoring (code quality, test coverage) |

Here is how a complete evaluation looks for our venetoclax question, combining the template above with two rubric traits:

```python
from karenina.schemas.entities.rubric import LLMRubricTrait, RegexTrait

# Template (from above): verifies BCL2 is identified as the target → PASS/FAIL

# LLM trait: does the response explain *how* the drug works?
mechanism_trait = LLMRubricTrait(
    name="explains_mechanism",
    description=(
        "True if the response explains how venetoclax interacts with its target "
        "(e.g., BH3 mimetic, inhibition of anti-apoptotic activity). "
        "False if the target is stated without mechanistic context."
    ),
    kind="boolean",
    higher_is_better=True,
)

# Regex trait: does the response include citations?
citation_trait = RegexTrait(
    name="has_citations",
    description="The response includes at least one numbered citation.",
    pattern=r"\[\d+\]",
    higher_is_better=True,
)
```

A response could correctly identify BCL2 (template passes) but fail to explain the mechanism (LLM trait returns `False`) and include no citations (regex trait returns `False`). The template verdict and rubric scores are independent.

Together, templates and rubrics give you both a correctness verdict and a quality profile for every response:

| Dimension | Templates | Rubrics |
|-----------|-----------|---------|
| Question answered | *Did the model get it right?* | *How well did the model answer?* |
| Evaluates | Correctness against ground truth | Observable qualities of the response |
| Operates on | Parsed, structured data (Pydantic schema) | Raw response trace (full text) |
| Requires ground truth | Yes (`self.correct`) | No (judges by reading the response alone) |
| Method | Judge LLM parses into schema, then `verify()` checks | Trait evaluators assess the raw text (LLM, regex, callable, or metric) |
| Output | Pass/fail | Boolean, integer score, or metrics dict |

[Answer Templates](docs/core_concepts/answer-templates.md) | [Rubrics](docs/core_concepts/rubrics/index.md) | [Templates vs Rubrics](docs/core_concepts/template-vs-rubric.md)

## 🚀 Getting Started

**Prerequisites:** Python 3.11+, an API key for at least one LLM provider ([see Installation](#-installation)), and karenina installed.

```python
from karenina import Benchmark
from karenina.schemas import ModelConfig, VerificationConfig

# Create a benchmark and add questions
benchmark = Benchmark.create(name="My Benchmark", version="1.0.0")
qid = benchmark.add_question(
    question="What is the approved drug target of Venetoclax?",
    raw_answer="BCL2",
    author={"name": "Curator", "email": "curator@example.com"},
)

# Generate answer templates automatically
benchmark.generate_all_templates(
    model="claude-haiku-4-5", model_provider="anthropic", temperature=0.0
)

# Configure and run verification
config = VerificationConfig(
    answering_models=[ModelConfig(
        id="haiku", model_name="claude-haiku-4-5",
        model_provider="anthropic", interface="langchain",
    )],
    parsing_models=[ModelConfig(
        id="haiku", model_name="claude-haiku-4-5",
        model_provider="anthropic", interface="langchain", temperature=0.0,
    )],
    evaluation_mode="template_and_rubric",
    rubric_enabled=True,
)
results = benchmark.run_verification(config)

# Inspect results
template_results = results.get_template_results()
template_results.aggregate_pass_rate(by="question_id")
```

**Full tutorials:**

- **[Quick Start: Benchmark](docs/getting-started/quickstart.md)**: End-to-end walkthrough for closed-loop evaluation
- **[Quick Start: TaskEval](docs/getting-started/quickstart-taskeval.md)**: Evaluate pre-recorded outputs without defining questions

## 💻 Command-Line Interface

The CLI provides automation and CI/CD integration without writing Python code.

```bash
# Run verification with a preset configuration
karenina verify checkpoint.jsonld --preset default.json --verbose

# Run with CLI arguments (no preset required)
karenina verify checkpoint.jsonld \
  --answering-model claude-haiku-4-5 \
  --parsing-model claude-haiku-4-5 \
  --output results.csv

# Start the web server (serves GUI + API)
karenina serve --port 8080
```

- **Flexible configuration**: presets, CLI arguments, interactive mode, or environment variables
- **Question filtering**: select specific questions by index or ID (e.g., `0-5`, `0,2,4`)
- **Progressive save**: automatic checkpointing with `--resume` for long runs
- **CI/CD ready**: deterministic exit codes (0 success, 1 error, 130 interrupted) and JSON/CSV output

[CLI Reference](docs/reference/cli/index.md)

## ✨ Key Features

- **[Structured evaluation via answer templates](docs/core_concepts/answer-templates.md)**: Pydantic models parsed by a Judge LLM, verified programmatically
- **[Agentic evaluation](docs/core_concepts/agentic-evaluation.md)**: evaluate coding and data analysis tasks where models and judges operate in workspaces with tool access
- **[5 rubric trait types](docs/core_concepts/rubrics/index.md)**: LLM (boolean, score, literal), regex, callable, metric, agentic
- **[2 evaluation modes](docs/core_concepts/evaluation-modes.md)**: Benchmark (closed-loop) and TaskEval (open-loop, pre-recorded outputs)
- **[6 LLM interfaces](docs/core_concepts/adapters.md)**: `langchain`, `claude_agent_sdk`, `claude_tool`, `openrouter`, `openai_endpoint`, `manual`
- **[13-stage configurable verification pipeline](docs/core_concepts/verification-pipeline.md)**: each stage can be enabled or disabled independently
- **[MCP integration](docs/core_concepts/mcp-overview.md)**: Model Context Protocol servers with tool use tracking
- **[Few-shot prompting](docs/core_concepts/few-shot.md)**: global or per-question examples with flexible selection modes
- **[Deep judgment](docs/workflows/running-verification/deep-judgment.md)**: extract verbatim excerpts, reasoning traces, and confidence scores
- **Async parallel execution**: configurable worker pools for batch runs
- **[CLI for automation and CI/CD](docs/reference/cli/index.md)**: presets, question filtering, progressive save, and deterministic exit codes

[View full documentation](docs/home/index.md)

## 🏗️ Architecture

Karenina uses a **hexagonal architecture** ([Ports & Adapters](docs/core_concepts/adapters.md)) for LLM interactions. Three protocol interfaces define what the application needs:

- **LLMPort**: basic LLM text generation
- **AgentPort**: agentic LLM with tool use, MCP support, and [workspace access](docs/core_concepts/agentic-evaluation.md)
- **ParserPort**: structured output parsing into Pydantic models

Each supported interface provides adapter implementations for these ports. An adapter factory handles instantiation, and an [AdapterInstructionRegistry](docs/core_concepts/prompt-assembly.md) manages interface-specific prompt transformations.

For [agentic evaluation](docs/core_concepts/agentic-evaluation.md), `AgentPort` powers both the answering model (operating in a workspace with tool access) and the judge (independently inspecting workspace artifacts). These two roles are independently configurable: an agentic answering model can be paired with a classical parser, or a classical answering model's response can be verified by an agentic judge.

### Ecosystem

| Package | Type | Purpose |
|---------|------|---------|
| **karenina** | Python library | Core evaluation framework (standalone) |
| **[karenina-server](https://github.com/biocypher/karenina-server)** | FastAPI backend | REST API exposing karenina functionality |
| **[karenina-gui](https://github.com/biocypher/karenina-gui)** | React/TypeScript | No-code web interface for benchmark management |

A web-based graphical interface covers most features provided by the backend, making benchmark creation and verification accessible to domain experts who prefer not to write Python code.

[Architecture documentation](docs/core_concepts/adapters.md)

## 📦 Installation

### Prerequisites

- Python 3.11 or higher
- Git
- `uv` (Python's fast package manager, recommended)

### Install uv

If you don't have `uv` installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For other installation methods, see [uv's documentation](https://docs.astral.sh/uv/getting-started/installation/).

### Install Karenina

**Note:** Karenina is not yet published to PyPI. Install from the GitHub repository:

```bash
# Install with uv (recommended)
uv pip install "karenina @ git+https://github.com/biocypher/karenina.git"

# Pin to a specific version for reproducibility
# uv pip install "karenina @ git+https://github.com/biocypher/karenina.git@v0.1.0"

# Or use pip
pip install "karenina @ git+https://github.com/biocypher/karenina.git"
```

This installs the core package with all required dependencies, including MCP client support, LangChain integrations, Anthropic SDK, and the CLI.

### Optional Dependencies

| Extra | Purpose |
|-------|---------|
| `dev` | Development and testing tools (pytest, ruff, mypy, mkdocs) |
| `search` | Web search integration for agentic verification (Tavily) |
| `examples` | Running example notebooks (Jupyter) |
| `embeddings` | Embedding similarity checks (SentenceTransformers) |
| `gepa` | GEPA prompt optimization integration |

```bash
# Example: install with embedding support
uv pip install "karenina[embeddings] @ git+https://github.com/biocypher/karenina.git"
```

### Environment Setup

Configure API keys for the LLM providers you plan to use:

```bash
# In your shell or a .env file in your working directory
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AI..."
export OPENROUTER_API_KEY="sk-or-..."  # Required for openrouter interface
```

Alternatively, use `karenina init` to generate a `.env` template with all supported variables. See the [Configuration Guide](docs/reference/configuration/index.md) for the full configuration hierarchy.

### Verify Installation

```bash
# Check CLI is available
karenina --version

# Check Python import
python -c "import karenina; print(karenina.__version__)"
```

For detailed setup instructions, troubleshooting, and development installation, see the [Installation Guide](docs/getting-started/installation.md).

## 📚 Documentation

View the full documentation locally with MkDocs:

```bash
uv run mkdocs serve
```

Then open your browser to `http://127.0.0.1:8000`.

### Getting Started
- [**Documentation Index**](docs/home/index.md): Complete documentation overview
- [**Installation Guide**](docs/getting-started/installation.md): Setup and requirements
- [**Quick Start: Benchmark**](docs/getting-started/quickstart.md): Your first evaluation end-to-end
- [**Quick Start: TaskEval**](docs/getting-started/quickstart-taskeval.md): Evaluate pre-recorded outputs

### Core Concepts
- [**Answer Templates**](docs/core_concepts/answer-templates.md): Structured correctness verification
- [**Rubrics**](docs/core_concepts/rubrics/index.md): Quality assessment with four trait types
- [**Templates vs Rubrics**](docs/core_concepts/template-vs-rubric.md): When to use which
- [**Verification Pipeline**](docs/core_concepts/verification-pipeline.md): The 13-stage evaluation engine
- [**Agentic Evaluation**](docs/core_concepts/agentic-evaluation.md): Workspace-based evaluation for coding and data analysis tasks

### Running Verification
- [**Verification Config**](docs/reference/configuration/verification-config.md): Configure and run evaluations
- [**CLI Reference**](docs/reference/cli/index.md): Command-line interface
- [**Analyzing Results**](docs/workflows/analyzing-results/index.md): DataFrames, export, and iteration

### Advanced
- [**Few-Shot Prompting**](docs/core_concepts/few-shot.md): Guide responses with examples
- [**Deep Judgment**](docs/workflows/running-verification/deep-judgment.md): Extract excerpts and reasoning traces
- [**Presets**](docs/workflows/configuration/presets.md): Save and reuse verification configurations

## 🤝 Contributing

We welcome contributions to Karenina! Please see our contributing guidelines for more information on how to get involved.
