# Core Concepts

This section is about *understanding* — what Karenina's building blocks are, why they exist, and how they relate to each other. If you're looking for step-by-step task guides, see [Workflows](../workflows.md). If you need exact field names or CLI flags, see [Reference](../reference.md).

## Concepts at a Glance

Concepts are ordered to follow the evaluation pipeline — from what you're evaluating, through how evaluation works, to what comes out the other end.

| Concept | What It Is | Page |
|---------|-----------|------|
| **Questions & Benchmarks** | The central objects: questions bundled with templates, rubrics, and metadata | [Questions & Benchmarks](questions-and-benchmarks.md) |
| **Checkpoints** | JSON-LD files that store benchmarks (questions, templates, rubrics, results) | [Checkpoints](checkpoints.md) |
| **Templates vs Rubrics** | The two evaluation units: correctness (templates) vs quality (rubrics) | [Templates vs Rubrics](template-vs-rubric.md) |
| **Answer Templates** | Pydantic models that define how a Judge LLM parses and verifies responses | [Answer Templates](answer-templates.md) |
| **Rubrics** | Trait-based evaluation of response quality (LLM, regex, callable, metric) | [Rubrics](rubrics/index.md) |
| **Evaluation Modes** | Three modes controlling which evaluation units run (`template_only`, `template_and_rubric`, `rubric_only`) | [Evaluation Modes](evaluation-modes.md) |
| **Verification Pipeline** | The 13-stage engine that executes evaluation end to end | [Verification Pipeline](verification-pipeline.md) |
| **Prompt Assembly** | How prompts are constructed for pipeline LLM calls (tri-section pattern) | [Prompt Assembly](prompt-assembly.md) |
| **Results & Scoring** | What verification produces: pass/fail, scores, traits, and metrics | [Results & Scoring](results-and-scoring.md) |
| **Adapters** | LLM backend interfaces (LangChain, Claude SDK, Claude Tool, Manual, and more) | [Adapters](adapters.md) |
| **MCP** | Tool-augmented evaluation via Model Context Protocol servers | [MCP Overview](mcp-overview.md) |
| **Manual Interface** | Evaluation using pre-recorded LLM traces instead of live API calls | [Manual Interface](manual-interface.md) |
| **ADeLe** | 18-dimension question classification system ([Zhou et al., 2025](https://arxiv.org/abs/2503.06378)) | [ADeLe](adele.md) |
| **Few-Shot** | Example injection for improved LLM parsing accuracy | [Few-Shot](few-shot.md) |

---

## How Concepts Fit Together

Karenina's evaluation workflow connects these concepts in a clear pipeline:

```
Questions & Benchmarks
 ├── Questions           ← what to evaluate
 ├── Answer Templates    ← how to judge correctness
 └── Rubric Traits       ← how to judge quality
         │
         ▼
 Checkpoint (.jsonld)    ← portable persistence
         │
         ▼
 Evaluation Mode         ← which evaluation units to run
         │
         ▼
 Adapter                 ← which LLM backend to use
  ├── LangChain, Claude SDK, Claude Tool, ...
  └── optionally with MCP tools
         │
         ▼
 Verification Pipeline   ← 13-stage execution engine
  ├── Prompt Assembly     ← constructs all LLM prompts
  └── Stage by stage      ← generate, parse, verify, evaluate
         │
         ▼
 Results & Scoring       ← pass/fail, scores, traits, metrics
```

1. A **benchmark** bundles questions with their templates and rubric traits
2. **Checkpoints** persist benchmarks as portable JSON-LD files
3. **Answer templates** define the structured schema a Judge LLM fills in, then `verify()` checks correctness
4. **Rubric traits** evaluate quality dimensions of the raw response (safety, clarity, format compliance, etc.)
5. The **evaluation mode** determines whether templates, rubrics, or both are used
6. An **adapter** connects to the LLM backend that generates and parses responses
7. **MCP** servers can provide tools to the answering model for tool-augmented evaluation
8. The **verification pipeline** orchestrates 13 stages from generation through scoring
9. **Prompt assembly** constructs all LLM prompts using a tri-section pattern
10. **Results** capture everything that happened — pass/fail, scores, excerpts, and metadata

---

## Concept Details

### Questions & Benchmarks

A **benchmark** is the central object in Karenina: a self-contained evaluation unit bundling questions, templates, rubrics, and metadata. **Questions** are the atomic unit — each has text, an expected answer, and optionally an attached template and question-specific rubric traits.

[Read more about questions and benchmarks →](questions-and-benchmarks.md)

### Checkpoints

A **checkpoint** is a JSON-LD file that stores everything needed to define and reproduce a benchmark: questions, answer templates, rubric traits, metadata, and optionally verification results. Checkpoints use [Schema.org](https://schema.org) types for interoperability.

[Read more about checkpoints →](checkpoints.md)

### Templates vs Rubrics

Karenina's evaluation rests on two complementary building blocks: **answer templates** verify factual correctness by having a Judge LLM parse responses into structured schemas, while **rubrics** assess response quality through trait evaluators that examine the raw text. Understanding when to use each — and when to use both together — is the foundation for effective benchmark design.

[Read more about templates vs rubrics →](template-vs-rubric.md)

### Answer Templates

**Answer templates** are Pydantic models that tell a Judge LLM how to parse a candidate response into structured fields. Each template implements a `verify()` method that compares parsed values against ground truth. This is the core mechanism for evaluating factual correctness.

[Read more about answer templates →](answer-templates.md)

### Rubrics

**Rubrics** evaluate qualitative traits of the raw response — independent of whether the answer is factually correct. Karenina provides four trait types: LLM traits, regex traits, callable traits, and metric traits. Rubrics can be applied globally (all questions) or per-question.

[Read more about rubrics →](rubrics/index.md)

### Evaluation Modes

Karenina supports three evaluation modes that control which units run during verification:

| Mode | Templates | Rubrics |
|------|-----------|---------|
| `template_only` (default) | Yes | No |
| `template_and_rubric` | Yes | Yes |
| `rubric_only` | No | Yes |

[Read more about evaluation modes →](evaluation-modes.md)

### Verification Pipeline

The **verification pipeline** is a 13-stage execution engine. Stages are grouped into setup, generation, guards, template processing, rubric evaluation, and finalization. The evaluation mode controls which stages are active.

[Read more about the verification pipeline →](verification-pipeline.md)

### Prompt Assembly

The **PromptAssembler** constructs all LLM prompts using a tri-section pattern: task instructions (from the pipeline stage), adapter instructions (backend-specific adjustments), and user instructions (your custom overrides via `PromptConfig`).

[Read more about prompt assembly →](prompt-assembly.md)

### Results & Scoring

The pipeline produces a **`VerificationResult`** per question containing template results (pass/fail, parsed fields), rubric results (per-trait scores), and metadata (timing, model info, errors). Result collections support aggregation and DataFrame export.

[Read more about results and scoring →](results-and-scoring.md)

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

[Read more about adapters →](adapters.md)

### MCP Overview

The **Model Context Protocol** (MCP) enables tool-augmented evaluation, where the answering model can use external tools (databases, APIs, code execution) during verification. This is essential for evaluating agentic capabilities.

[Read more about MCP →](mcp-overview.md)

### Manual Interface

The **manual interface** allows you to evaluate pre-recorded LLM traces instead of making live API calls. This is useful for reproducibility, cost reduction, and evaluating responses from models not directly supported by karenina adapters.

[Read more about the manual interface →](manual-interface.md)

### ADeLe

**ADeLe** (Annotated Demand Levels; [Zhou et al., 2025](https://arxiv.org/abs/2503.06378)) is an 18-dimension question classification system that characterizes questions along axes like reasoning depth, domain specificity, and answer format. Classifications are stored in checkpoint metadata and can guide template design and evaluation strategy.

[Read more about ADeLe →](adele.md)

### Few-Shot

**Few-shot configuration** controls how example responses are injected into the Judge LLM's parsing prompt. Providing examples can improve parsing accuracy, especially for complex or ambiguous response formats. Modes include `all`, `k-shot`, `custom`, and `none`.

[Read more about few-shot configuration →](few-shot.md)
