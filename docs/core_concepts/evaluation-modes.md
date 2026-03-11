---
jupyter:
  jupytext:
    formats: docs/core_concepts//md,docs/notebooks/core_concepts//ipynb
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

# Evaluation Modes

An **evaluation mode** is the setting that tells Karenina which evaluation path to run for a verification: **template verification**, **rubric evaluation**, or both.

The most important idea is that evaluation mode answers a simple operational question:

> When this response is evaluated, should Karenina check correctness, quality, or both?

That choice determines which pipeline stages run, which inputs are examined, and which result fields become meaningful.

```python tags=["hide-cell"]
from karenina.schemas import ModelConfig, VerificationConfig
```

## 1. What It Is

Karenina supports three pipeline modes:

| Mode | Template verification | Rubric evaluation | Default | Best for |
|---|---|---|---|---|
| `template_only` | Yes | No | Yes | Closed-answer questions where correctness is the main signal |
| `template_and_rubric` | Yes | Yes | No | Runs where you need correctness and quality side by side |
| `rubric_only` | No | Yes | No | Open-ended outputs, trace audits, or quality-only evaluation |

Use [answer templates](../notebooks/core_concepts/answer-templates.ipynb) when you can say what the right answer is. Use [rubrics](rubrics/index.md) when you want to score observable qualities of the response. Use evaluation mode to decide whether a run activates one path or both.

## 2. Philosophy / Core Idea

Karenina separates two questions:

1. **Did the model get the answer right?**
2. **Was the response good in the ways I care about?**

Those are often related, but they are not the same.

- Templates answer the first question by parsing into a schema and then running deterministic `verify()` code.
- Rubrics answer the second question by evaluating the raw response or trace with LLM, regex, callable, and metric traits.

Evaluation mode exists because different tasks need different combinations of those two ideas.

| If this describes your task... | Prefer this mode | Why |
|---|---|---|
| "There is a real ground-truth answer, and pass/fail correctness is the main outcome." | `template_only` | Keeps the run focused on correctness |
| "I care whether the answer is right, but I also care about tone, safety, format, or completeness." | `template_and_rubric` | Produces correctness and quality signals from the same run |
| "There is no single correct answer, or I only want to score response quality." | `rubric_only` | Avoids pretending there is a meaningful template-based correctness check |

Two useful litmus tests:

- If you would struggle to write a defensible `verify()` method, that is usually a sign to prefer `rubric_only`.
- If a rubric trait would just restate a known factual answer, that is usually a sign to keep the core decision in a template instead.

## 3. Overview / Anatomy

### What each side sees

Evaluation mode matters because Karenina's two evaluation systems operate on different inputs and hide different things.

| System | What it sees | What stays hidden or handled elsewhere |
|---|---|---|
| Template flow | Question text, the evaluation input passed to parsing, and the `Answer` schema | Ground truth stored on `self.correct`; rubric traits |
| Rubric flow | Question text, the rubric evaluation input, and the attached rubric traits | Template `verify()` logic; structured parsed fields unless your trait recreates them itself |

For trace-based runs, the defaults are intentionally asymmetric:

- Template parsing uses only the **final AI message** by default: `use_full_trace_for_template=False`
- Rubric evaluation uses the **full trace** by default: `use_full_trace_for_rubric=True`

That matches the usual abstraction boundary:

- templates judge the final answer
- rubrics often need to inspect the broader behavior that produced it

### What each mode turns on

| Mode | Primary outputs | Determinism profile | What to watch for |
|---|---|---|---|
| `template_only` | `result.template.verify_result`, parsed fields, optional embedding/deep-judgment metadata | `verify()` is deterministic after parsing; parsing may still use an LLM | Best when correctness is the decision you care about |
| `template_and_rubric` | Template outputs plus `result.rubric.*_trait_scores` | Mixed: deterministic template verification plus whatever trait types you use | Best when you want correctness and quality without separate runs |
| `rubric_only` | Rubric trait scores | Depends on trait types: regex/callable can be deterministic; LLM and metric traits are not | Best when quality is the product, not an add-on |

## 4. How It Works / Lifecycle / Pipeline

At the orchestration layer, evaluation mode controls which stages `StageOrchestrator.from_config(...)` adds to the pipeline.

| Stage | `template_only` | `template_and_rubric` | `rubric_only` |
|---|:---:|:---:|:---:|
| `ValidateTemplate` | Yes | Yes | No |
| `GenerateAnswer` | Yes | Yes | Yes |
| `RecursionLimitAutoFail` | Yes | Yes | Yes |
| `TraceValidationAutoFail` | Yes | Yes | Yes |
| `AbstentionCheck` | Optional | Optional | Optional |
| `SufficiencyCheck` | Optional | Optional | No |
| `ParseTemplate` | Yes | Yes | No |
| `VerifyTemplate` | Yes | Yes | No |
| `EmbeddingCheck` | Present; only runs when enabled and needed | Present; only runs when enabled and needed | No |
| `DeepJudgmentAutoFail` | Optional | Optional | No |
| `RubricEvaluation` | No | Yes | Yes |
| `DeepJudgmentRubricAutoFail` | No | Optional | Optional |
| `FinalizeResult` | Yes | Yes | Yes |

Three execution details matter in practice:

1. `rubric_only` truly skips template stages at the orchestrator level.
2. `template_and_rubric` evaluates rubrics on the raw response, not on parsed template fields.
3. Rubric stages are only added when a non-empty `Rubric` is actually attached.

That means `template_and_rubric` is not "template mode with nicer reporting." It is a combined run with two distinct evaluation paths:

```
raw response
  ├─ template path: parse -> verify() -> optional embedding/deep judgment
  └─ rubric path:   score raw text/trace with attached traits
```

## 5. Patterns / Worked Examples

### Canonical pattern: one factual question, one optional quality layer

Suppose your benchmark asks:

> Which protein does venetoclax target?

You have:

- a template that verifies the extracted target is `BCL2`
- a rubric with traits like `conciseness` and `has_citation`

The same benchmark can be run in three ways:

| Mode | What Karenina checks | What you read from the result |
|---|---|---|
| `template_only` | Did the parsed target match `BCL2`? | `result.template.verify_result` |
| `template_and_rubric` | Did the parsed target match `BCL2`? And was the response concise / cited? | `result.template.verify_result` and `result.rubric.*_trait_scores` |
| `rubric_only` | Only the quality traits on the raw output | `result.rubric.*_trait_scores` |

### Configuration example

```python
answering = [
    ModelConfig(
        id="answerer",
        model_name="claude-haiku-4-5",
        model_provider="anthropic",
    )
]

parsing = [
    ModelConfig(
        id="judge",
        model_name="claude-haiku-4-5",
        model_provider="anthropic",
    )
]

template_only_config = VerificationConfig(
    answering_models=answering,
    parsing_models=parsing,
)

template_and_rubric_config = VerificationConfig(
    answering_models=answering,
    parsing_models=parsing,
    evaluation_mode="template_and_rubric",
    rubric_enabled=True,
)

rubric_only_config = VerificationConfig(
    answering_models=answering,
    parsing_models=parsing,
    evaluation_mode="rubric_only",
    rubric_enabled=True,
)
```

```python tags=["hide-cell"]
assert template_only_config.evaluation_mode == "template_only"
assert template_only_config.rubric_enabled is False
assert template_and_rubric_config.evaluation_mode == "template_and_rubric"
assert template_and_rubric_config.rubric_enabled is True
assert rubric_only_config.evaluation_mode == "rubric_only"
assert rubric_only_config.rubric_enabled is True
```

### Decision guidance

| Goal | Prefer | Prefer this over |
|---|---|---|
| Strict factual benchmarking | `template_only` | `rubric_only`, which cannot tell you whether the answer is factually correct |
| Factual benchmarking plus style/safety/compliance review | `template_and_rubric` | Running one template-only pass and a separate rubric-only pass on the same benchmark |
| Trace review for agent behavior, safety, or formatting | `rubric_only` | A weak template that encodes vague judgments as pseudo-correctness |

## 6. Detailed Reference

### Configuration rules

`VerificationConfig` enforces these combinations:

| `evaluation_mode` | Required `rubric_enabled` |
|---|---|
| `template_only` | `False` |
| `template_and_rubric` | `True` |
| `rubric_only` | `True` |

If you use `VerificationConfig.from_overrides(...)`, setting `evaluation_mode` automatically sets `rubric_enabled` to the matching value.

### Result-shape guidance

When rubric evaluation runs, rubric outputs are stored under `result.rubric`.

When template stages do not run, Karenina still returns a `template` section with the raw response and related metadata, but template-specific correctness fields behave differently:

- in ordinary `rubric_only` runs, `result.template.verify_result` is typically unset (`None`)
- guard stages such as recursion-limit, trace-validation, or abstention checks can still force `result.template.verify_result` to `False`

So in `rubric_only`, treat rubric scores as the primary output and `verify_result` as a guard/failure signal rather than a correctness verdict.

### Low-level runner behavior

At the low-level `run_single_model_verification(...)` API, passing a non-empty rubric while leaving `evaluation_mode="template_only"` causes the runner to upgrade internally to `template_and_rubric`. `VerificationConfig` does not do that silently; it validates consistency up front.

### TaskEval behavior

[TaskEval](task-eval.md) infers the mode from what you attach:

- template only -> `template_only`
- rubric only -> `rubric_only`
- both -> `template_and_rubric`

In `rubric_only`, `TaskEval` can evaluate logs even when no explicit template is attached; internally it creates a minimal synthetic template so the shared pipeline can still run.

## 7. Next Steps

- [Templates vs Rubrics](../notebooks/core_concepts/template-vs-rubric.ipynb): the conceptual distinction between correctness and quality
- [Answer Templates](../notebooks/core_concepts/answer-templates.ipynb): how parsing and `verify()` work
- [Rubrics](rubrics/index.md): trait types and selection guidance
- [Verification Pipeline](verification-pipeline.md): the full stage-by-stage execution model
- [Results and Scoring](results-and-scoring.md): how to read `VerificationResult`
- [VerificationConfig Reference](../reference/configuration/verification-config.md): complete configuration fields
