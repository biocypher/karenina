# Karenina Documentation

Welcome to the Karenina documentation. Karenina is a Python library for systematic LLM benchmarking with structured, reproducible evaluation.

---

## What is Karenina?

Karenina is a framework for defining benchmarks in a rigorous and reproducible way. With Karenina, benchmarks can be created, shared, reviewed, and executed across a wide range of large language models (LLMs).

**Core Philosophy**: Enable domain experts to create benchmarks without deep LLM-technical expertise, allowing them to focus on knowledge rather than infrastructure.

**Key Capabilities**:

- Create benchmarks from scratch or existing question sets
- Define precise evaluation criteria using code-based templates
- Evaluate answers using both rule-based and LLM-as-judge strategies
- Support natural free-form responses and constrained formats
- Track performance across multiple models and configurations
- Share and reproduce benchmark results

**Package Architecture**: Karenina is a standalone Python library that can optionally integrate with:

- **karenina-server**: FastAPI-based REST API wrapper
- **karenina-gui**: React-based web interface

---

## Understanding Karenina's Approach

### The Problem

Consider a simple multiple-choice question:

```python
question = "What is the capital of Italy?"
possible_answers = ["Rome", "Milan", "Paris", "New York"]
```

When we query a standard LLM, it usually responds in free text (e.g., `"I think the answer is Rome, because it is the capital of Italy."`). To evaluate such an answer programmatically, we have three approaches:

#### 1. Constrain the Answering Model's Output

We directly instruct the answering model to return a response in a machine-friendly format.

**Example prompt:**

```text
You are answering a multiple-choice question.
Return only the letter of your choice.

Question: What is the capital of Italy?
Options:
A) Rome
B) Milan
C) Paris
D) New York

Answer:
```

**Model output:** `A`

**Pros**: Simple and reliable when models comply

**Cons**: Fragile prompt adherence; requires different strategies for different formats

---

#### 2. Use an LLM as a Judge (Free-Text Evaluation)

Instead of constraining the answering model, we keep its output free-form and rely on a **judge LLM** to interpret it.

**Example:**

* **Answering model output:** `"The capital of Italy is Rome, of course."`
* **Judge model prompt:**
  ```text
  The following is a student's answer to a multiple-choice question.
  Question: What is the capital of Italy?
  Options: Rome, Milan, Paris, New York.
  Student's answer: "The capital of Italy is Rome, of course."
  Which option does this correspond to? Provide a justification.
  ```
* **Judge model output:** `"The student clearly selected Rome, which is correct."`

**Pros**: Flexible, allows natural answering
**Cons**: Judge response is also free text, requiring parsing; potential inconsistencies

---

### The Karenina Strategy

Karenina adopts a **third approach** that combines the advantages of both:

* The **answering model** remains unconstrained, generating natural free text
* The **judge model** returns results in a **structured format** (validated through Pydantic classes)

This setup allows the judge to flexibly interpret free text while ensuring that its own output remains standardized and machine-readable.

#### Example Workflow

**1. Define a Pydantic template:**

```python
from karenina.domain.answers import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    answer: str = Field(description="The name of the city in the response")

    def model_post_init(self, __context):
        self.correct = {"answer": "Rome"}

    def verify(self) -> bool:
        return self.answer.strip().lower() == self.correct["answer"].strip().lower()
```

**Key aspects:**
- The `answer` attribute uses `Field` description to guide the judge
- The `verify` method implements custom validation logic

**2. Answering model generates free text:**

```
"The capital of Italy is Rome."
```

**3. Judge model parses into structured format:**

```python
from langchain_core.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=Answer)
prompt = parser.get_format_instructions()
prompt += "\n LLM Answer: The capital of Italy is Rome."

judge_answer = llm.invoke(prompt)
```

**Judge output (structured JSON):**

```json
{"answer": "Rome"}
```

**4. Verification step:**

```python
populated_answer = Answer(**judge_answer)
result = populated_answer.verify()  # True
```

---

## Why Templates?

Templates play a central role in Karenina by standardizing how answers are parsed, verified, and evaluated. Their use provides several key benefits:

### 1. Unified Parsing and Evaluation

Templates allow parsing to happen **directly through the judge LLM**. The free-text answer is mapped into a structured format, ensuring that:

- Evaluation logic is **bundled with the question-answer pair**
- The same benchmark can accommodate **different answer formats** without custom code

### 2. Streamlined Benchmark Creation

Since LLMs are proficient at code generation, they can **auto-generate Pydantic classes** from raw question-answer pairs. This means large portions of benchmark creation can be automated, reducing manual effort while improving consistency.

### 3. Cognitive Offloading for the Judge

By embedding the evaluation schema in templates, the **judge LLM's task is simplified**. Instead of reasoning about both content and evaluation logic, the judge focuses only on interpreting the free-text answer and filling in the template.

### 4. Extensibility and Reusability

Templates make it straightforward to extend benchmarks:

- New tasks can be added by defining new templates
- The same evaluation logic can be reused across multiple benchmarks

### 5. Transparency and Debuggability

By encoding evaluation criteria into explicit, inspectable templates, benchmarks become more transparent. This allows developers to:

- **Audit** evaluation rules directly
- **Debug** failures by inspecting structured outputs

---

## Core Features

Karenina provides comprehensive tools for every stage of the benchmarking workflow:

### Question Management
Extract and manage benchmark questions from files (Excel, CSV, TSV) with rich metadata support.

### Answer Templates
Pydantic-based templates that define how to evaluate model outputs. Templates specify what information should be extracted and how to verify correctness programmatically.

### Rubric Evaluation
Assess qualitative aspects of answers using three trait types:

- **LLM-based traits**: AI evaluates qualities (binary pass/fail or 1-5 scale)
- **Regex-based traits**: Pattern matching for format validation
- **Metric-based traits**: Confusion matrix metrics (precision, recall, F1, accuracy)

### Benchmark Verification
Run evaluations with four supported interfaces:

- **`langchain`**: OpenAI, Google Gemini, Anthropic Claude (via LangChain)
- **`openrouter`**: OpenRouter platform
- **`openai_endpoint`**: OpenAI-compatible endpoints (local models)
- **`manual`**: Manual trace replay for testing/debugging

### Advanced Evaluation
- **Deep-Judgment Parsing**: Extract excerpts, reasoning traces, and confidence scores
- **Abstention Detection**: Identify when models refuse to answer
- **Embedding Check**: Semantic similarity fallback for false negatives
- **Few-Shot Prompting**: Configure examples globally or per question

### Data Management
- **Database Persistence**: SQLite storage with versioning
- **Checkpoint Format**: JSON-LD for benchmark state
- **Export & Reporting**: CSV and JSON formats

[View complete feature catalog →](features.md)

## Feature Comparison: Templates vs Rubrics

Karenina uses two evaluation units:

| Aspect | Answer Templates | Rubrics |
|--------|-----------------|---------|
| **Purpose** | Verify factual correctness | Assess qualitative traits |
| **Evaluation** | Programmatic comparison | LLM judgment, regex, or metrics |
| **Best for** | Precise, unambiguous answers | Subjective qualities, format validation |
| **Trait Types** | N/A | LLM-based, regex-based, metric-based |
| **Output** | Pass/fail per field | Boolean, score (1-5), or metrics |
| **Examples** | "BCL2", "46 chromosomes" | "Is the answer concise?", "Must match pattern" |
| **Scope** | Per question | Global or per question |

[Learn more about Templates →](using-karenina/templates.md) | [Learn more about Rubrics →](using-karenina/rubrics.md)

---

## Next Steps

**New to Karenina?**

1. [Install Karenina](install.md) on your system
2. Follow the [Quick Start](quickstart.md) to create your first benchmark
3. Explore the [Features Overview](features.md) to see what's possible

**Ready to build benchmarks?**

- Read the [User Guides](#user-guides) for comprehensive tutorials
- Check the [API Reference](api-reference.md) for complete method documentation
- Browse [Advanced Features](#advanced-features) for specialized capabilities

**Need help?**

- Review [Troubleshooting](troubleshooting.md) for common issues
- Check [Configuration](configuration.md) for environment variables and defaults

---

## Quick Navigation

### 🚀 Getting Started
- **[Installation](install.md)** - Set up Karenina on your system
- **[Quick Start](quickstart.md)** - Create your first benchmark in 10 minutes
- **[Features Overview](features.md)** - Complete feature catalog

### 📖 User Guides
- **[Defining Benchmarks](using-karenina/defining-benchmark.md)** - Benchmark creation and metadata
- **[Adding Questions](using-karenina/adding-questions.md)** - File extraction and management
- **[Templates](using-karenina/templates.md)** - Creating and customizing answer templates
- **[Rubrics](using-karenina/rubrics.md)** - Evaluation criteria and trait types
- **[Verification](using-karenina/verification.md)** - Running evaluations and analyzing results
- **[Saving & Loading](using-karenina/saving-loading.md)** - Checkpoints, database, and export

### 🔬 Advanced Features
- **[Deep-Judgment](advanced/deep-judgment.md)** - Extract detailed feedback with excerpts
- **[Few-Shot Prompting](advanced/few-shot.md)** - Guide responses with examples
- **[Abstention Detection](advanced/abstention-detection.md)** - Handle model refusals
- **[Embedding Check](advanced/embedding-check.md)** - Semantic similarity fallback
- **[Presets](advanced/presets.md)** - Save and reuse verification configurations
- **[System Integration](advanced/integration.md)** - Server and GUI integration

### 📚 Reference
- **[API Reference](api-reference.md)** - Complete API documentation
- **[Configuration](configuration.md)** - Environment variables and defaults
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions

---

[Get Started →](quickstart.md) | [View All Features →](features.md)
