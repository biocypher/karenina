# What is Karenina?

**Karenina** is a Python framework for defining, running, and sharing LLM benchmarks in a rigorous and reproducible way. It enables systematic evaluation of large language model performance through structured, verifiable testing.

**New here?** Start with the **[Quick Start](../notebooks/quickstart.ipynb)** — a hands-on walkthrough that takes you from zero to a working benchmark in minutes.

## Documentation Structure

This documentation is organized into four sections, each serving a different reader mindset:

| Section | Reader mindset | What you'll find |
|---------|---------------|-----------------|
| **[Core Concepts](../core_concepts/index.md)** | *"Help me understand"* | Mental models and explanations — what templates, rubrics, pipelines, and adapters *are* and why they exist. Read this when you need to build intuition before doing anything. |
| **[Workflows](../workflows.md)** | *"Help me do it"* | Step-by-step task guides — creating benchmarks, running verification, analyzing results. Read this when you have a specific goal and want to accomplish it. |
| **[Reference](../reference.md)** | *"Help me look it up"* | Exhaustive tables of CLI flags, config fields, environment variables, and schemas. Read this when you already know what you want to do and just need the exact syntax. |
| **[Advanced](../advanced.md)** | *"Help me extend it"* | Pipeline internals, adapter architecture, and custom stages. Read this when you need to debug, customize, or contribute to karenina itself. |

---

## Key Capabilities

- **Create benchmarks** from scratch or from existing question sets
- **Define precise evaluation criteria** using code-based answer templates (Pydantic models)
- **Evaluate answers** using both rule-based verification and LLM-as-judge strategies
- **Support natural, unconstrained outputs** — no rigid response formats required
- **Assess response quality** with rubrics (LLM judgment, regex, callable, and metric traits)
- **Track performance** across multiple models and configurations
- **Share and reproduce** benchmark results via JSON-LD checkpoint files

## When to Use Karenina

Karenina is designed for data scientists and ML engineers who need to:

- **Compare models systematically** across consistent criteria, not ad-hoc prompting
- **Go beyond simple string matching** — evaluate free-form LLM outputs with structured logic
- **Combine correctness and quality checks** — verify factual accuracy *and* assess response qualities like clarity, safety, or format compliance
- **Automate evaluation at scale** — run hundreds of questions across multiple models with a single configuration
- **Reproduce results** — share benchmarks as portable JSON-LD files that anyone can re-run

## Ecosystem Overview

Karenina has three packages that work together:

| Package | Type | Purpose |
|---------|------|---------|
| **karenina** | Python library | Core benchmarking framework (this documentation) |
| **karenina-server** | FastAPI backend | REST API exposing karenina functionality |
| **karenina-gui** | React/TypeScript | No-code web interface for benchmark management |

This documentation covers the **karenina** Python library. The server and GUI have their own documentation.

## How It Works

Karenina uses a **two-unit evaluation approach**:

1. **Answer Templates** verify *correctness* — did the model give the right answer? A Judge LLM parses the model's free-text response into a structured Pydantic schema, then a programmatic `verify()` method checks it against ground truth.

2. **Rubrics** assess *quality* — how well did the model answer? Trait evaluators examine the raw response for qualities like safety, conciseness, format compliance, or extraction completeness.

These two units are complementary. A common pattern: use a template to verify the model extracted the correct answer, then use rubrics to check that the response was concise, cited sources, and avoided hallucination.

For a deeper discussion, see [Templates vs Rubrics](../core_concepts/template-vs-rubric.md) and [Philosophy](philosophy.md).

---

## Next Steps

- [Philosophy](philosophy.md) — Why LLM-as-judge evaluation works
- [Templates vs Rubrics](../core_concepts/template-vs-rubric.md) — Understanding the two evaluation units
- [Installation](../getting-started/installation.md) — Install karenina and set up API keys
- [Core Concepts](../core-concepts.md) — Deep dive into checkpoints, templates, rubrics, and more
