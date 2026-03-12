---
jupyter:
  jupytext:
    formats: docs/workflows/task-eval//md,docs/notebooks/task-eval//ipynb
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Evaluating with TaskEval

TaskEval evaluates pre-recorded text or agent traces without defining questions or generating answers. You log outputs, attach evaluation criteria ([templates](../../core_concepts/answer-templates.md) for correctness, [rubrics](../../core_concepts/rubrics/index.md) for quality, or both), and run the judge LLM. For the underlying concepts, see [TaskEval](../../core_concepts/task-eval.md).

## Overview

```
Log outputs → Attach criteria → Evaluate → Inspect results
```

## Choose Your Scenario

| Scenario | Focus Area | What You'll Learn |
|----------|-----------|-------------------|
| [Basic Evaluation](../../notebooks/task-eval/basic-evaluation.ipynb) | Template + rubric | Create TaskEval, log text/traces, attach templates and rubrics, configure `VerificationConfig`, inspect results |
| [Quality Assessment](../../notebooks/task-eval/quality-assessment.ipynb) | Rubric-only | LLM, regex, and callable traits, rubric-only evaluation, compare scores across outputs |
| [Multi-Step Evaluation](../../notebooks/task-eval/multi-step-evaluation.ipynb) | Step-scoped | Named steps, `target` routing, step-scoped criteria, per-step vs global evaluation |

---

## Common Workflow

All three scenarios follow this general pattern:

```
Create TaskEval
    │
    ▼
Log outputs (text, traces, or both)
    │
    ▼
Attach evaluation criteria (templates, rubrics, or both)
    │
    ▼
Configure VerificationConfig (parsing_only=True)
    │
    ▼
Evaluate and inspect results
```

### Key APIs

| Operation | Method | Covered In |
|-----------|--------|------------|
| Create instance | `TaskEval(task_id=..., metadata=...)` | All scenarios |
| Log text | `task.log(text)` | [Basic Evaluation](../../notebooks/task-eval/basic-evaluation.ipynb) |
| Log traces | `task.log_trace(messages)` | [Basic Evaluation](../../notebooks/task-eval/basic-evaluation.ipynb), [Multi-Step](../../notebooks/task-eval/multi-step-evaluation.ipynb) |
| Add template | `task.add_template(AnswerClass)` | [Basic Evaluation](../../notebooks/task-eval/basic-evaluation.ipynb), [Multi-Step](../../notebooks/task-eval/multi-step-evaluation.ipynb) |
| Add rubric | `task.add_rubric(rubric)` | All scenarios |
| Evaluate globally | `task.evaluate(config)` | All scenarios |
| Evaluate one step | `task.evaluate(config, step_id="...")` | [Multi-Step](../../notebooks/task-eval/multi-step-evaluation.ipynb) |
| Inspect results | `result.summary()`, `result.display()` | All scenarios |
| Export results | `result.export_json()`, `result.export_markdown()` | [Basic Evaluation](../../notebooks/task-eval/basic-evaluation.ipynb) |

---

## Core Concepts

These concept pages provide the foundational knowledge that the scenarios build on:

- [TaskEval](../../core_concepts/task-eval.md): Object structure, pipeline integration, merge strategies
- [Answer Templates](../../notebooks/core_concepts/answer-templates.ipynb): Template structure, field types, `verify()` semantics
- [Rubrics](../../core_concepts/rubrics/index.md): Trait types (LLM, regex, callable, metric), global vs per-question
- [Evaluation Modes](../../notebooks/core_concepts/evaluation-modes.ipynb): How template-only, template+rubric, and rubric-only map to pipeline stages
- [Verification Pipeline](../../core_concepts/verification-pipeline.md): The 13-stage engine that TaskEval feeds into

---

## Next Steps

- [Analyzing Results](../analyzing-results/index.md): DataFrame analysis, export, and iteration
- [Running Verification](../running-verification/index.md): Benchmark-mode verification workflows
- [Creating Benchmarks](../creating-benchmarks/index.md): Build benchmarks with questions and templates
