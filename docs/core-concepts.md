# Core Concepts

This section covers the fundamental building blocks of karenina. Each concept has a dedicated page with detailed explanations and, where applicable, executable examples.

## Concepts at a Glance

| Concept | What It Is | Page |
|---------|-----------|------|
| **Templates vs Rubrics** | The two evaluation units: correctness (templates) vs quality (rubrics) | [Templates vs Rubrics](04-core-concepts/template-vs-rubric.md) |
| **Checkpoints** | JSON-LD files that store benchmarks (questions, templates, rubrics, results) | [Checkpoints](04-core-concepts/checkpoints.md) |
| **Answer Templates** | Pydantic models that define how a Judge LLM parses and verifies responses | [Answer Templates](04-core-concepts/answer-templates.md) |
| **Rubrics** | Trait-based evaluation of response quality (LLM, regex, callable, metric) | [Rubrics](04-core-concepts/rubrics/index.md) |
| **Evaluation Modes** | Three modes controlling which evaluation units run (`template_only`, `template_and_rubric`, `rubric_only`) | [Evaluation Modes](04-core-concepts/evaluation-modes.md) |
| **Adapters** | LLM backend interfaces (LangChain, Claude SDK, Claude Tool, Manual, and more) | [Adapters](04-core-concepts/adapters.md) |
| **MCP** | Tool-augmented evaluation via Model Context Protocol servers | [MCP Overview](04-core-concepts/mcp-overview.md) |
| **Manual Interface** | Evaluation using pre-recorded LLM traces instead of live API calls | [Manual Interface](04-core-concepts/manual-interface.md) |
| **Adele** | 18-dimension question classification system | [Adele](04-core-concepts/adele.md) |
| **Few-Shot** | Example injection for improved LLM parsing accuracy | [Few-Shot](04-core-concepts/few-shot.md) |

---

## How Concepts Fit Together

Karenina's evaluation workflow connects these concepts in a clear pipeline:

```
Checkpoint (.jsonld)
 ├── Questions          ← what to evaluate
 ├── Answer Templates   ← how to judge correctness
 └── Rubric Traits      ← how to judge quality
         │
         ▼
 Evaluation Mode        ← which evaluation units to run
         │
         ▼
 Adapter                ← which LLM backend to use
  ├── LangChain, Claude SDK, Claude Tool, ...
  └── optionally with MCP tools
         │
         ▼
 Verification Results   ← pass/fail, scores, traits
```

1. A **checkpoint** bundles questions with their templates and rubric traits
2. **Answer templates** define the structured schema a Judge LLM fills in, then `verify()` checks correctness
3. **Rubric traits** evaluate quality dimensions of the raw response (safety, clarity, format compliance, etc.)
4. The **evaluation mode** determines whether templates, rubrics, or both are used
5. An **adapter** connects to the LLM backend that generates and parses responses
6. **MCP** servers can provide tools to the answering model for tool-augmented evaluation
7. The **manual interface** allows replaying pre-recorded traces without live API calls

---

## Reading Paths

Choose the path that matches your goal:

### New User

Start with the distinction between the two evaluation units, then read the concepts in order:

**[Templates vs Rubrics](04-core-concepts/template-vs-rubric.md)** → **[Checkpoints](04-core-concepts/checkpoints.md)** → **[Answer Templates](04-core-concepts/answer-templates.md)** → **[Rubrics](04-core-concepts/rubrics/index.md)** → **[Evaluation Modes](04-core-concepts/evaluation-modes.md)** → **[Adapters](04-core-concepts/adapters.md)**

Then proceed to [Creating Benchmarks](05-creating-benchmarks/index.md) and [Running Verification](06-running-verification/index.md) for hands-on workflows.

### Power User

Jump directly to the concept you need:

- Designing evaluation criteria? Start with [Answer Templates](04-core-concepts/answer-templates.md) and [Rubrics](04-core-concepts/rubrics/index.md)
- Choosing an LLM backend? See [Adapters](04-core-concepts/adapters.md)
- Adding tool use to evaluation? See [MCP Overview](04-core-concepts/mcp-overview.md)
- Classifying questions by difficulty? See [Adele](04-core-concepts/adele.md)
- Improving parsing accuracy? See [Few-Shot](04-core-concepts/few-shot.md)

### Contributor

For architecture internals, see:

- [Advanced Pipeline](11-advanced-pipeline/index.md) — 13 verification stages, deep judgment, prompt assembly
- [Advanced Adapters](12-advanced-adapters/index.md) — ports/adapters pattern, writing custom adapters

---

## Concept Details

### Templates vs Rubrics

Karenina's evaluation rests on two complementary building blocks: **answer templates** verify factual correctness by having a Judge LLM parse responses into structured schemas, while **rubrics** assess response quality through trait evaluators that examine the raw text. Understanding when to use each — and when to use both together — is the foundation for effective benchmark design.

[Read more about templates vs rubrics →](04-core-concepts/template-vs-rubric.md)

### Checkpoints

A **checkpoint** is a JSON-LD file that stores everything needed to define and reproduce a benchmark: questions, answer templates, rubric traits, metadata, and optionally verification results. Checkpoints use [Schema.org](https://schema.org) types for interoperability.

[Read more about checkpoints →](04-core-concepts/checkpoints.md)

### Answer Templates

**Answer templates** are Pydantic models that tell a Judge LLM how to parse a candidate response into structured fields. Each template implements a `verify()` method that compares parsed values against ground truth. This is the core mechanism for evaluating factual correctness.

[Read more about answer templates →](04-core-concepts/answer-templates.md)

### Rubrics

**Rubrics** evaluate qualitative traits of the raw response — independent of whether the answer is factually correct. Karenina provides four trait types:

- **LLM traits** — Subjective assessment via a Judge LLM (boolean, score, or literal kinds)
- **Regex traits** — Pattern matching for format compliance
- **Callable traits** — Custom Python functions for deterministic checks
- **Metric traits** — Precision/recall/F1 for extraction completeness

Rubrics can be applied globally (all questions) or per-question.

[Read more about rubrics →](04-core-concepts/rubrics/index.md)

### Evaluation Modes

Karenina supports three evaluation modes that control which units run during verification:

| Mode | Templates | Rubrics |
|------|-----------|---------|
| `template_only` (default) | Yes | No |
| `template_and_rubric` | Yes | Yes |
| `rubric_only` | No | Yes |

[Read more about evaluation modes →](04-core-concepts/evaluation-modes.md)

### Adapters

**Adapters** are LLM backend interfaces that handle the actual communication with language models. Karenina uses a hexagonal architecture where adapters implement port protocols:

| Interface | Description |
|-----------|-------------|
| `langchain` | Default adapter — supports all LLMs via LangChain |
| `openrouter` | OpenRouter API (routes through LangChain) |
| `openai_endpoint` | OpenAI-compatible endpoints (routes through LangChain) |
| `claude_agent_sdk` | Native Anthropic Agent SDK |
| `claude_tool` | Direct Anthropic SDK with tool use |
| `manual` | Pre-recorded traces (no live API calls) |

[Read more about adapters →](04-core-concepts/adapters.md)

### MCP Overview

The **Model Context Protocol** (MCP) enables tool-augmented evaluation, where the answering model can use external tools (databases, APIs, code execution) during verification. This is essential for evaluating agentic capabilities.

[Read more about MCP →](04-core-concepts/mcp-overview.md)

### Manual Interface

The **manual interface** allows you to evaluate pre-recorded LLM traces instead of making live API calls. This is useful for reproducibility, cost reduction, and evaluating responses from models not directly supported by karenina adapters.

[Read more about the manual interface →](04-core-concepts/manual-interface.md)

### Adele

**Adele** is an 18-dimension question classification system that characterizes questions along axes like reasoning depth, domain specificity, and answer format. Classifications are stored in checkpoint metadata and can guide template design and evaluation strategy.

[Read more about Adele →](04-core-concepts/adele.md)

### Few-Shot

**Few-shot configuration** controls how example responses are injected into the Judge LLM's parsing prompt. Providing examples can improve parsing accuracy, especially for complex or ambiguous response formats. Modes include `all`, `k-shot`, `custom`, and `none`.

[Read more about few-shot configuration →](04-core-concepts/few-shot.md)
