# Creating Benchmarks

This section walks through building benchmarks end-to-end — from creating an empty checkpoint to saving a fully populated benchmark with questions, templates, rubrics, and few-shot examples.

Each tutorial is a complete, self-contained scenario. Choose the one that matches your evaluation needs, or work through them in order to learn all the tools.

## Choose Your Scenario

| Scenario | Evaluation Strategy | What You'll Learn |
|----------|-------------------|-------------------|
| [Benchmark Operations](../../notebooks/creating-benchmarks/benchmark-operations.ipynb) | All | Full Benchmark API: creating, populating, templates, rubrics, readiness, filtering, collection protocols |
| [Factual QA Benchmark](../../notebooks/creating-benchmarks/factual-qa-benchmark.ipynb) | Template-only | Hand-written templates: boolean check, string normalization, numeric tolerance, regex in `verify()`, partial credit |
| [Full Evaluation Benchmark](../../notebooks/creating-benchmarks/full-evaluation-benchmark.ipynb) | Template + rubric | Custom templates combined with all 6 rubric trait types (LLM boolean, LLM score, LLM literal, regex, callable, metric) |
| [Quality Assessment](../../notebooks/creating-benchmarks/quality-assessment-benchmark.ipynb) | Rubric-only | No templates: quality evaluation for tasks with no single correct answer (safety, empathy, clarity) |
| [Choosing Rubric Traits](../../notebooks/creating-benchmarks/choosing-rubric-traits.ipynb) | Template + rubric | Need-driven trait selection: 7 evaluation needs mapped to the right trait type, decision flowchart |
| [Scaled Authoring](../../notebooks/creating-benchmarks/scaled-authoring.ipynb) | Power user | Bulk ingestion, `generate_all_templates()`, `AnswerBuilder`, ADeLe classification, few-shot examples |

---

## Evaluation Strategies

Karenina supports three evaluation strategies. Every benchmark uses one of these:

```
┌─────────────────────────────────────────────────────────┐
│                    Template-only                         │
│  Questions + Templates → Correctness (pass/fail)        │
│  "Is the extracted information correct?"                 │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│               Template + Rubric                          │
│  Questions + Templates + Rubrics → Correctness + Quality │
│  "Is it correct AND well-written?"                       │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    Rubric-only                            │
│  Questions + Rubrics → Quality assessment                │
│  "Is it safe, clear, and appropriate?"                   │
└─────────────────────────────────────────────────────────┘
```

| Strategy | Required | Optional | Best For |
|----------|----------|----------|----------|
| **Template-only** | Questions, templates | ADeLe, few-shot | Factual questions with definitive answers |
| **Template + rubric** | Questions, templates, rubrics | ADeLe, few-shot | Comprehensive evaluation (correctness + quality) |
| **Rubric-only** | Questions, rubrics | ADeLe, few-shot | Subjective tasks, communication, safety |

See [Evaluation Modes](../../notebooks/core_concepts/evaluation-modes.ipynb) for details on how these strategies map to pipeline behavior.

---

## Common Workflow

All four scenarios follow this general pattern:

```
Create benchmark
    │
    ▼
Add questions (with or without templates)
    │
    ▼
Define evaluation criteria (templates, rubrics, or both)
    │
    ▼
Save checkpoint
    │
    ▼
Reload and verify round-trip
```

### Key APIs

| Operation | Method | Covered In |
|-----------|--------|------------|
| Create benchmark | `Benchmark.create(name, description, version)` | All scenarios |
| Add question | `benchmark.add_question(question, raw_answer, answer_template=...)` | All scenarios |
| Add template to existing question | `benchmark.add_answer_template(question_id, code_string)` | [Factual QA](../../notebooks/creating-benchmarks/factual-qa-benchmark.ipynb) |
| Generate templates automatically | `benchmark.generate_all_templates(model, model_provider)` | [Scaled Authoring](../../notebooks/creating-benchmarks/scaled-authoring.ipynb) |
| Build templates programmatically | `AnswerBuilder().add_attribute(...).compile()` | [Scaled Authoring](../../notebooks/creating-benchmarks/scaled-authoring.ipynb) |
| Add global rubric trait | `benchmark.add_global_rubric_trait(trait)` | [Full Evaluation](../../notebooks/creating-benchmarks/full-evaluation-benchmark.ipynb), [Quality Assessment](../../notebooks/creating-benchmarks/quality-assessment-benchmark.ipynb) |
| Add per-question rubric trait | `benchmark.add_question_rubric_trait(question_id, trait)` | [Full Evaluation](../../notebooks/creating-benchmarks/full-evaluation-benchmark.ipynb), [Quality Assessment](../../notebooks/creating-benchmarks/quality-assessment-benchmark.ipynb) |
| Check readiness | `benchmark.check_readiness()` | [Benchmark Operations](../../notebooks/creating-benchmarks/benchmark-operations.ipynb) |
| Filter questions | `benchmark.filter_questions(finished=True, has_template=True)` | [Benchmark Operations](../../notebooks/creating-benchmarks/benchmark-operations.ipynb) |
| Save checkpoint | `benchmark.save("path.jsonld")` | All scenarios |
| Load checkpoint | `Benchmark.load("path.jsonld")` | All scenarios |

---

## Core Concepts

These concept pages provide the foundational knowledge that the scenarios build on:

- [Answer Templates](../../notebooks/core_concepts/answer-templates.ipynb) — What templates are, field types, `verify()` semantics
- [Rubrics](../../core_concepts/rubrics/index.md) — Trait types (LLM, regex, callable, metric), global vs per-question
- [Checkpoints](../../core_concepts/questions-and-benchmarks/checkpoints.md) — JSON-LD format, save/load behavior
- [Evaluation Modes](../../notebooks/core_concepts/evaluation-modes.ipynb) — How template-only, template+rubric, and rubric-only map to pipeline stages
- [ADeLe Classification](../../core_concepts/adele.md) — Question complexity dimensions
- [Few-Shot Examples](../../core_concepts/few-shot.md) — Configuration modes and example selection

---

## Next Steps

Once your benchmark is built and saved, proceed to:

- [Running Verification](../running-verification/index.md) — Execute the benchmark against LLMs
- [Analyzing Results](../analyzing-results/index.md) — Inspect and compare verification outcomes
