# What is Karenina?

!!! warning "Experimental Project"

    Karenina is an experimental project still making its baby steps towards maturity. Best effort has been applied in creating a correct set of documentation, however some errors and imprecisions may be present. If you encounter any, please [open an issue](https://github.com/biocypher/karenina/issues) on the GitHub repository and we will try to get them fixed as soon as possible.

**Karenina** is an open-source Python framework that simplifies how you define, run, and share LLM evaluations. It covers the full evaluation spectrum: simple factual Q&A, tool-augmented interactions where models use external tools via [MCP](../notebooks/core_concepts/mcp-overview.ipynb), and fully [agentic coding and data analysis tasks](../notebooks/core_concepts/agentic-evaluation.ipynb) where both the answering model and the judge operate in a real workspace with file and code access.

Its core idea is formalizing ground truth as structured [answer templates](../notebooks/core_concepts/answer-templates.ipynb): Pydantic models that encode what a correct response looks like, letting a [Judge LLM](philosophy.md#the-llm-as-judge-approach) parse free-form responses into those schemas for programmatic verification. Combined with [rubrics](../core_concepts/rubrics/index.md) for quality assessment and support for classical methods like regex, Karenina provides a flexible [evaluation pipeline](../notebooks/core_concepts/verification-pipeline.ipynb) from quick checks to complex multi-trait scoring. It supports three evaluation modes:

- **[Q/A Benchmark](../core_concepts/questions-and-benchmarks/index.md)**: Define questions and answer pairs, generate responses, and evaluate them through a staged verification pipeline.
- **[Scenarios](../core_concepts/scenarios/index.md)**: Define conversation graphs with branching paths and outcome criteria to evaluate behavior across multiple turns.
- **[TaskEval](../notebooks/core_concepts/task-eval.ipynb)**: Supply pre-recorded outputs from any source and evaluate them using the same pipeline.

**New here?** Start with the **[Quick Start: Q/A Benchmark](../notebooks/quickstart.ipynb)** to run your first evaluation end-to-end, the **[Quick Start: Scenarios](../notebooks/quickstart-scenarios.ipynb)** to build a multi-turn evaluation with branching, or the **[Quick Start: TaskEval](../notebooks/quickstart-taskeval.ipynb)** if you already have outputs to evaluate.

## Why This Approach

1. **Naturalistic evaluation.** Traditional benchmarks force models into artificial formats (multiple-choice letters, regex-compliant strings) that differ from real-world usage and signal to the model that it is being evaluated. In Karenina, the answering model is never constrained: it produces the same kind of response a real user would receive. A separate [Judge LLM](philosophy.md#the-llm-as-judge-approach) evaluates the natural response after the fact.

2. **Portable, self-contained benchmarks.** Each [question](../core_concepts/questions-and-benchmarks/index.md) carries its own verification logic and quality checks. A benchmark bundles questions, evaluation criteria, and metadata into a single [portable checkpoint](../core_concepts/questions-and-benchmarks/checkpoints.md) that anyone can reload, re-run against different models, or extend with new questions. Evaluation criteria travel with the data.

3. **Bootstrapped authoring.** LLMs can [auto-generate evaluation code](../notebooks/creating-benchmarks/scaled-authoring.ipynb) from a simple spreadsheet of questions and answers, bootstrapping benchmark creation in minutes. Quality checks are defined declaratively, so adding them requires no custom infrastructure.

4. **Expressivity.** [Templates](../notebooks/core_concepts/answer-templates.ipynb) combine natural-language field descriptions with programmatic verification logic, allowing flexible definitions of what it means to "pass": multiple attributes of different types, combined with arbitrary rules (exact match, normalization, numeric tolerance, partial credit, or any custom Python logic). [Scenarios](../core_concepts/scenarios/index.md) extend this expressivity to multi-turn conversations: define branching evaluation graphs where each turn's result determines the next question, then assert compound outcome criteria over the full conversation (e.g., "the model answered correctly on turn 1 and resisted a sycophantic challenge on turn 2").

5. **Agentic evaluation, not just Q&A.** Modern LLM deployments increasingly involve agents that write code, run tests, and produce file artifacts. Karenina evaluates these workflows natively: the answering model operates in a [workspace](../notebooks/core_concepts/agentic-evaluation.ipynb#2-workspaces) with tool access, and an independent [judge agent](../notebooks/core_concepts/agentic-evaluation.ipynb#4-two-step-agentic-judging-stage-7b) inspects the resulting artifacts (files created, tests passed, code compiled) rather than relying on the conversation trace alone. The same template and rubric primitives apply whether the task is a factual question or a multi-step coding challenge.

6. **Benchmarks that measure what you care about.** Public benchmarks create incentives for model providers to optimize for the test rather than for real-world usefulness. By lowering the cost of creating domain-specific evaluations, Karenina lets teams build internal suites that measure the capabilities that actually matter for their deployment. When anyone can spin up a benchmark on their own terms, evaluation becomes harder to game, creating a race to the top where genuine model improvement is the only winning strategy.

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

**Shared evaluation engine** (all modes):

- **Define precise evaluation criteria** using code-based answer templates (Pydantic models)
- **Evaluate answers** using both rule-based verification and LLM-as-judge strategies
- **Support natural, unconstrained outputs**, no rigid response formats required
- **Assess response quality** with rubrics (LLM judgment, regex, callable, and metric traits)
- **Evaluate agentic tasks** where models operate in workspaces with tool access (coding, data analysis)
- **Judge workspace artifacts** with an independent agent that inspects files, re-runs tests, and verifies outputs

**[Q/A Benchmark](../core_concepts/questions-and-benchmarks/index.md) mode**:

- **Create benchmarks** from scratch or from existing question sets
- **Track performance** across multiple models and configurations
- **Share and reproduce** benchmark results via JSON-LD checkpoint files

**[Scenario mode](../core_concepts/scenarios/index.md)**:

- **Define conversation graphs** with nodes (questions) and edges (routing conditions) for multi-turn evaluation
- **Test sycophancy, error correction, and progressive disclosure** across branching conversation paths
- **Assert outcome criteria** over the full conversation result after execution

**[TaskEval](../notebooks/core_concepts/task-eval.ipynb) mode**:

- **Evaluate any free text** from agent workflows or external systems
- **Log structured traces** preserving tool calls and conversation history
- **Score per-step** with step-scoped templates and rubrics for multi-phase agent workflows

## When to Use Karenina

| Need | Mode |
|------|------|
| Compare LLM performance across consistent criteria | Q/A Benchmark |
| Evaluate free-form outputs with structured logic (not string matching) | All modes |
| Verify factual accuracy *and* assess quality (clarity, safety, format) | All modes |
| Run hundreds of questions across multiple models automatically | Q/A Benchmark |
| Share portable evaluation suites that anyone can re-run | Q/A Benchmark |
| Evaluate coding or data analysis tasks with workspace artifacts | Q/A Benchmark |
| Score agent workflow outputs after execution | TaskEval |
| Evaluate multi-step agent traces per phase | TaskEval |
| Test sycophancy resistance across multi-turn conversation paths | [Scenarios](../core_concepts/scenarios/index.md) |
| Evaluate multi-turn reasoning with branching conditions | [Scenarios](../core_concepts/scenarios/index.md) |
| Assess error correction behavior across conversation turns | [Scenarios](../core_concepts/scenarios/index.md) |

## Ecosystem Overview

Karenina has three packages that work together:

| Package | Type | Purpose |
|---------|------|---------|
| **karenina** | Python library | Core evaluation framework (this documentation) |
| **karenina-server** | FastAPI backend | REST API exposing karenina functionality |
| **karenina-gui** | React/TypeScript | No-code web interface for benchmark management |

This documentation covers the **karenina** Python library. The server and GUI have their own documentation.

## How It Works

Karenina uses a **two-unit evaluation approach** shared by all modes:

1. **Answer Templates** verify *correctness*: did the model give the right answer? A Judge LLM parses the response into a structured Pydantic schema, then a programmatic `verify()` method checks it against ground truth.

2. **Rubrics** assess *quality*: how well did the model answer? Trait evaluators examine the raw response for qualities like safety, conciseness, format compliance, or extraction completeness.

The three modes differ in where the response comes from and how the conversation is structured:

| Dimension | Q/A Benchmark | Scenarios | TaskEval |
|-----------|---------------|-----------|----------|
| Response source | Pipeline generates via answering model | Pipeline generates across multiple turns | You supply pre-recorded outputs |
| Starting point | Questions (define what to ask) | Scenario graph (nodes, edges, outcome criteria) | Traces (record what happened) |
| Pipeline stages | All 13 stages | All 13 stages (per turn) | Skips stage 2 (answer generation) |
| Persistence | JSON-LD checkpoints | JSON-LD checkpoints | In-memory TaskEvalResult |

A common pattern: use a template to verify the model extracted the correct answer, then use rubrics to check that the response was concise, cited sources, and avoided hallucination. This works identically across all modes.

For [agentic tasks](../notebooks/core_concepts/agentic-evaluation.ipynb), both steps extend to workspace inspection: the answering model writes code and artifacts into a workspace directory, and the judge agent independently examines those artifacts before filling in the template. The evaluation engine is the same; only the source of evidence changes (workspace files instead of conversation text).

For a deeper discussion, see [Templates vs Rubrics](../notebooks/core_concepts/template-vs-rubric.ipynb) and [Philosophy](philosophy.md).

---

## Next Steps

- [Philosophy](philosophy.md) — Why LLM-as-judge evaluation works
- [Answer Templates](../notebooks/core_concepts/answer-templates.ipynb) — How a Judge LLM parses and verifies responses
- [Rubrics](../core_concepts/rubrics/index.md) — Trait-based quality assessment
- [Templates vs Rubrics](../notebooks/core_concepts/template-vs-rubric.ipynb) — When to use which, and when to use both
- [Agentic Evaluation](../notebooks/core_concepts/agentic-evaluation.ipynb) — Workspace-based evaluation for coding and data analysis tasks
- [Scenarios](../core_concepts/scenarios/index.md): Multi-turn conversation graph evaluation with branching paths and outcome criteria
- [TaskEval](../core_concepts/task-eval.md): Evaluate pre-recorded outputs without defining questions
- [Installation](../getting-started/installation.md) — Install karenina and set up API keys
- [Core Concepts](../core_concepts/index.md) — Deep dive into checkpoints, pipelines, adapters, and more
