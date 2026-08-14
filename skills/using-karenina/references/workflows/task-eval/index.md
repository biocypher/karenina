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
| [Basic Evaluation](basic-evaluation.md) | Template + rubric | Create TaskEval, log text/traces, attach templates and rubrics, configure `VerificationConfig`, inspect results |
| [Quality Assessment](quality-assessment.md) | Rubric-only | LLM, regex, and callable traits, rubric-only evaluation, compare scores across outputs |
| [Multi-Step Evaluation](multi-step-evaluation.md) | Step-scoped | Named steps, `target` routing, step-scoped criteria, per-step vs global evaluation |

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
| Log text | `task.log(text)` | [Basic Evaluation](basic-evaluation.md) |
| Log traces | `task.log_trace(messages)` | [Basic Evaluation](basic-evaluation.md), [Multi-Step](multi-step-evaluation.md) |
| Add template | `task.add_template(AnswerClass)` | [Basic Evaluation](basic-evaluation.md), [Multi-Step](multi-step-evaluation.md) |
| Add rubric | `task.add_rubric(rubric)` | All scenarios |
| Evaluate globally | `task.evaluate(config)` | All scenarios |
| Evaluate one step | `task.evaluate(config, step_id="...")` | [Multi-Step](multi-step-evaluation.md) |
| Inspect results | `result.summary()`, `result.display()` | All scenarios |
| Export results | `result.export_json()`, `result.export_markdown()` | [Basic Evaluation](basic-evaluation.md) |

---

## Core Concepts

These concept pages provide the foundational knowledge that the scenarios build on:

- [TaskEval](../../core_concepts/task-eval.md): Object structure, pipeline integration, merge strategies
- [Answer Templates](../../core_concepts/answer-templates.md): Template structure, field types, `verify()` semantics
- [Rubrics](../../core_concepts/rubrics/index.md): Trait types (LLM, regex, callable, metric), global vs per-question
- [Evaluation Modes](../../core_concepts/evaluation-modes.md): How template-only, template+rubric, and rubric-only map to pipeline stages
- [Verification Pipeline](../../core_concepts/verification-pipeline.md): The 13-stage engine (with sub-stages 7a/7b and 11a/11b) that TaskEval feeds into

---

## Next Steps

- [Analyzing Results](../analyzing-results/index.md): DataFrame analysis, export, and iteration
- [Running Verification](../running-verification/index.md): Benchmark-mode verification workflows
- [Creating Benchmarks](../creating-benchmarks/index.md): Build benchmarks with questions and templates
