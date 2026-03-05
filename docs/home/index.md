# What is Karenina?

!!! warning "Experimental Project"

    Karenina is an experimental project still making its baby steps towards maturity. Best effort has been applied in creating a correct set of documentation, however some errors and imprecisions may be present. If you encounter any, please [open an issue](https://github.com/biocypher/karenina/issues) on the GitHub repository and we will try to get them fixed as soon as possible.

**Karenina** is a Python framework for structured LLM evaluation. It supports two modes:

- **Benchmark** (closed-loop): Define questions, generate responses, and evaluate them through a 13-stage verification pipeline.
- **TaskEval** (open-loop): Supply pre-recorded outputs from any source and evaluate them using the same pipeline.

Both modes share the same evaluation engine: answer templates for correctness, rubrics for quality, and a Judge LLM that parses free-text responses into structured schemas.

**New here?** Start with the **[Quick Start: Benchmark](../notebooks/quickstart.ipynb)** to run your first evaluation end-to-end, or the **[Quick Start: TaskEval](../notebooks/quickstart-taskeval.ipynb)** if you already have outputs to evaluate.

## Documentation Structure

This documentation is organized into four sections, each serving a different reader mindset:

| Section | Reader mindset | What you'll find |
|---------|---------------|-----------------|
| **[Core Concepts](../core_concepts/index.md)** | *"Help me understand"* | Mental models and explanations — what templates, rubrics, pipelines, and adapters *are* and why they exist. Read this when you need to build intuition before doing anything. |
| **[Workflows](../workflows/index.md)** | *"Help me do it"* | Step-by-step task guides — creating benchmarks, running verification, analyzing results. Read this when you have a specific goal and want to accomplish it. |
| **[Reference](../reference/index.md)** | *"Help me look it up"* | Exhaustive tables of CLI flags, config fields, environment variables, and schemas. Read this when you already know what you want to do and just need the exact syntax. |
| **[Advanced](../advanced.md)** | *"Help me extend it"* | Pipeline internals, adapter architecture, and custom stages. Read this when you need to debug, customize, or contribute to karenina itself. |

---

## Key Capabilities

**Shared evaluation engine** (both modes):

- **Define precise evaluation criteria** using code-based answer templates (Pydantic models)
- **Evaluate answers** using both rule-based verification and LLM-as-judge strategies
- **Support natural, unconstrained outputs**, no rigid response formats required
- **Assess response quality** with rubrics (LLM judgment, regex, callable, and metric traits)

**Benchmark mode**:

- **Create benchmarks** from scratch or from existing question sets
- **Track performance** across multiple models and configurations
- **Share and reproduce** benchmark results via JSON-LD checkpoint files

**TaskEval mode**:

- **Evaluate any free text** from agent workflows or external systems
- **Log structured traces** preserving tool calls and conversation history
- **Score per-step** with step-scoped templates and rubrics for multi-phase agent workflows

## When to Use Karenina

| Scenario | Mode |
|----------|------|
| Compare LLM performance across consistent criteria | Benchmark |
| Evaluate free-form outputs with structured logic (not string matching) | Both |
| Verify factual accuracy *and* assess quality (clarity, safety, format) | Both |
| Run hundreds of questions across multiple models automatically | Benchmark |
| Share portable evaluation suites that anyone can re-run | Benchmark |
| Score agent workflow outputs after execution | TaskEval |
| Evaluate multi-step agent traces per phase | TaskEval |

## Ecosystem Overview

Karenina has three packages that work together:

| Package | Type | Purpose |
|---------|------|---------|
| **karenina** | Python library | Core evaluation framework (this documentation) |
| **karenina-server** | FastAPI backend | REST API exposing karenina functionality |
| **karenina-gui** | React/TypeScript | No-code web interface for benchmark management |

This documentation covers the **karenina** Python library. The server and GUI have their own documentation.

## How It Works

Karenina uses a **two-unit evaluation approach** shared by both modes:

1. **Answer Templates** verify *correctness*: did the model give the right answer? A Judge LLM parses the response into a structured Pydantic schema, then a programmatic `verify()` method checks it against ground truth.

2. **Rubrics** assess *quality*: how well did the model answer? Trait evaluators examine the raw response for qualities like safety, conciseness, format compliance, or extraction completeness.

The two modes differ in where the response comes from:

| Dimension | Benchmark | TaskEval |
|-----------|-----------|----------|
| Response source | Pipeline generates via answering model | You supply pre-recorded outputs |
| Starting point | Questions (define what to ask) | Traces (record what happened) |
| Pipeline stages | All 13 stages | Skips stage 2 (answer generation) |
| Persistence | JSON-LD checkpoints | In-memory TaskEvalResult |

A common pattern: use a template to verify the model extracted the correct answer, then use rubrics to check that the response was concise, cited sources, and avoided hallucination. This works identically in both modes.

For a deeper discussion, see [Templates vs Rubrics](../core_concepts/template-vs-rubric.md) and [Philosophy](philosophy.md).

---

## Next Steps

- [Philosophy](philosophy.md) — Why LLM-as-judge evaluation works
- [Answer Templates](../notebooks/core_concepts/answer-templates.ipynb) — How a Judge LLM parses and verifies responses
- [Rubrics](../core_concepts/rubrics/index.md) — Trait-based quality assessment
- [Templates vs Rubrics](../core_concepts/template-vs-rubric.md) — When to use which, and when to use both
- [TaskEval](../core_concepts/task-eval.md): Evaluate pre-recorded outputs without defining questions
- [Installation](../getting-started/installation.md) — Install karenina and set up API keys
- [Core Concepts](../core_concepts/index.md) — Deep dive into checkpoints, pipelines, adapters, and more
