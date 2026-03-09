---
jupyter:
  jupytext:
    formats: docs/core_concepts/rubrics//md,docs/notebooks/core_concepts/rubrics//ipynb
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Metric Traits

Metric traits use the **parsing model's judgment** to turn a response into a confusion matrix, then compute deterministic metrics from that matrix. They are useful when you already know the set of items a good answer should cover and you want coverage-style signals such as precision, recall, F1, specificity, or accuracy. For an overview of all trait types and how rubrics fit into the evaluation framework, see the [rubrics index](index.md).

```python tags=["hide-cell"]
# Mock cell: ensures examples execute without live API keys.
# This cell is hidden in rendered documentation.
```

## Overview

A `MetricRubricTrait` evaluates a response during [stage 11 (RubricEvaluation)](../verification-pipeline.md) of the [verification pipeline](../verification-pipeline.md). Unlike other trait types that return a single boolean or score, metric traits return a **dictionary of metrics** computed from confusion-matrix buckets.

The evaluation has two parts:

1. The parsing model reads the question, the response trace, and your instructions, then assigns content to confusion-matrix buckets.
2. Karenina computes the requested metrics programmatically from the bucket counts.

This split matters: the judgment step is LLM-based, but the math is deterministic. If the same confusion lists are produced again, the metric values will be identical.

Two modes are available:

| Mode | You define | Karenina can compute | Best For |
|------|------------|----------------------|----------|
| `tp_only` | What **should** be present | `precision`, `recall`, `f1` | Extraction coverage and expected-item recall |
| `full_matrix` | What **should** be present and what **should not** be present | `precision`, `recall`, `specificity`, `accuracy`, `f1` | Coverage plus factual exclusion checks |

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | *(required)* | Human-readable identifier |
| `description` | `str \| None` | `None` | Optional scope or evaluator context |
| `evaluation_mode` | `Literal["tp_only", "full_matrix"]` | `"tp_only"` | Whether you define only TP instructions or both TP and TN |
| `metrics` | `list[str]` | *(required)* | Metrics to compute for this trait |
| `tp_instructions` | `list[str]` | *(required)* | Items that should be present in the answer |
| `tn_instructions` | `list[str]` | `[]` | Items that should not be present; required in `full_matrix` mode |
| `repeated_extraction` | `bool` | `True` | Deduplicate repeated excerpts using case-insensitive exact matching |

## Why the Instruction Lists Matter

For metric traits, the instruction lists are the scoring surface. The parsing model does not infer your rubric from the metric names alone; it relies on `tp_instructions`, `tn_instructions`, and optionally `description` to understand what counts.

```
Question + Response Trace + TP/TN Instructions
                    ↓
         ┌──────────────────┐
         │   Parsing Model  │
         │                  │
         └────────┬─────────┘
                  ↓
      Confusion lists per trait (TP/FN/FP/TN)
                  ↓
      Deterministic metric computation
                  ↓
    VerificationResult.rubric.metric_trait_scores
```

**Good instruction lists are:**

- **Atomic**: each instruction represents one countable thing.
- **Observable**: the evaluator can decide from the response text alone.
- **Non-overlapping**: one answer excerpt should not satisfy many nearly identical instructions.
- **Consistent in granularity**: avoid mixing tiny facts with very broad requirements in the same trait.

**Strong**:

    "Mentions BCL2"
    "States that BCL2 inhibits apoptosis"
    "Mentions chromosome 18"

**Weak**:

    "Covers the biology well"
    "Is scientifically thorough"
    "Talks about the important details"

The `description` field is still useful, but it is supporting context, not the main scoring mechanism. Use it to narrow scope or explain domain context; use the instruction lists to define what is actually counted.

## TP-Only Mode

In `tp_only` mode, you define what **should** appear in the answer. The parsing model then separates the response into three usable buckets:

- **TP (True Positive)**: answer content that correctly matches a TP instruction
- **FN (False Negative)**: expected content from the TP instructions that is missing
- **FP (False Positive)**: answer content that looks like a candidate match in the same domain but should not count

Available metrics: `precision`, `recall`, `f1`

This mode is best when you care about coverage of expected items and do not need an explicit list of forbidden claims.

### A Subtle Point About FP in `tp_only` Mode

`tp_only` mode does **not** mean "every extra sentence becomes a false positive." In practice, FP is for answer content that appears to be trying to satisfy the TP instruction set but is not actually correct for that set.

For example, if your TP instructions describe expected lung diseases, then unrelated filler text is not especially meaningful as FP. But incorrect diseases in the same category are.

### Writing Good TP Instructions

A good TP list defines the target set clearly enough that recall and precision are interpretable.

- Use one instruction per target fact, entity, reference, or concept.
- Write instructions at the same level of specificity.
- Prefer concrete language over umbrella phrases.
- Avoid pairs like `"Mentions BCL2"` and `"Discusses BCL2"` in the same trait unless they really mean different things.

```python
from karenina.schemas import MetricRubricTrait

reference_trait = MetricRubricTrait(
    name="Reference Coverage",
    description="Check whether the answer covers the canonical papers for this topic.",
    evaluation_mode="tp_only",
    metrics=["precision", "recall", "f1"],
    tp_instructions=[
        "Mentions Tsujimoto et al., Science, 1985",
        "Mentions Hockenbery et al., Nature, 1990",
        "Mentions Adams & Cory, Science, 1998",
    ],
)

print(reference_trait.get_required_buckets())
```

### Metric Formulas in `tp_only` Mode

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Precision | TP / (TP + FP) | Of the candidate matches the evaluator accepted or rejected, how many were correct? |
| Recall | TP / (TP + FN) | Of the expected items, how many were found? |
| F1 | 2 * P * R / (P + R) | Balanced summary of precision and recall |

If you define 3 TP instructions and the answer contains 2 of them plus 1 incorrect candidate in the same domain:

- TP = 2
- FN = 1
- FP = 1
- Precision = 2/3
- Recall = 2/3
- F1 = 2/3

## Full Matrix Mode

In `full_matrix` mode, you define both what **should** appear and what **should not** appear. This gives the evaluator an explicit negative set and unlocks `specificity` and `accuracy`.

- **TP (True Positive)**: answer content that matches a TP instruction
- **FN (False Negative)**: TP instruction content that is missing
- **TN (True Negative)**: TN instructions that are correctly absent
- **FP (False Positive)**: answer content that matches a TN instruction and therefore should not be present

Available metrics: `precision`, `recall`, `specificity`, `accuracy`, `f1`

Use this mode when absence matters, for example factual accuracy, prohibited claims, or mutually exclusive alternatives.

### Writing Good TN Instructions

TN instructions should capture concrete wrong claims, not vague badness.

**Strong**:

    "Claims BCL2 is pro-apoptotic"
    "States BCL2 is on chromosome 1"

**Weak**:

    "Contains inaccurate biology"
    "Says something wrong"

Good TN instructions are explicit enough that the evaluator can recognize both the presence and the absence of the claim.

```python
accuracy_trait = MetricRubricTrait(
    name="Content Accuracy",
    description="Check for correct BCL2 claims and flag known false statements.",
    evaluation_mode="full_matrix",
    metrics=["precision", "recall", "specificity", "accuracy", "f1"],
    tp_instructions=[
        "Mentions BCL2 gene",
        "States that BCL2 inhibits apoptosis",
        "References cancer relevance",
    ],
    tn_instructions=[
        "Claims BCL2 is pro-apoptotic",
        "States BCL2 is on chromosome 1",
    ],
)

print(accuracy_trait.get_required_buckets())
```

### Additional Metrics in `full_matrix` Mode

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Specificity | TN / (TN + FP) | Of the things that should be absent, how many actually stayed absent? |
| Accuracy | (TP + TN) / (TP + TN + FP + FN) | Overall correctness across both expected and forbidden items |

## Choosing a Mode

| Scenario | Mode | Why |
|----------|------|-----|
| Check whether key entities or references were mentioned | `tp_only` | You care about expected coverage, not explicit forbidden claims |
| Measure extraction quality against a known answer set | `tp_only` | Precision, recall, and F1 usually give the right signal |
| Detect both correct facts and known false claims | `full_matrix` | You need an explicit negative set |
| Evaluate factual accuracy where incorrect assertions matter | `full_matrix` | Specificity and accuracy depend on TN instructions |

If you are unsure, start with `tp_only`. Move to `full_matrix` when you can clearly enumerate the important wrong claims you want to guard against.

## The `repeated_extraction` Field

By default, `repeated_extraction=True` deduplicates each confusion-list bucket using case-insensitive exact matching. This prevents repeated excerpts from inflating metric counts.

```python
dedup_trait = MetricRubricTrait(
    name="Entity Check",
    evaluation_mode="tp_only",
    metrics=["precision", "recall"],
    tp_instructions=["Mentions mitochondria", "Mentions apoptosis"],
    repeated_extraction=True,
)

no_dedup_trait = MetricRubricTrait(
    name="Entity Frequency",
    evaluation_mode="tp_only",
    metrics=["precision", "recall"],
    tp_instructions=["Mentions mitochondria", "Mentions apoptosis"],
    repeated_extraction=False,
)
```

Leave deduplication enabled unless repeated occurrences are genuinely part of what you want to count.

## Results and Confusion Lists

Metric trait outputs are stored separately from other rubric trait types:

- `VerificationResult.rubric.metric_trait_scores`: metric values per trait
- `VerificationResult.rubric.metric_trait_confusion_lists`: raw TP/TN/FP/FN lists per trait

For a TP-only trait, the confusion lists typically look like this:

```python
confusion_lists = {
    "Reference Coverage": {
        "tp": [
            "Tsujimoto et al., Science, 1985",
            "Adams & Cory, Science, 1998",
        ],
        "fn": ["Hockenbery et al., Nature, 1990"],
        "fp": ["BAX, Cell, 1993"],
        "tn": [],
    }
}
```

The exact strings may be answer excerpts (`tp`, `fp`) or instruction-derived items (`fn`, `tn`), depending on the bucket. This makes it possible to inspect not just the metric values, but *why* those values were produced.

## Validation

Metric traits enforce validation when you construct them:

- `tp_instructions` must always be non-empty
- `tn_instructions` must be non-empty in `full_matrix` mode
- Metric names must be valid
- `specificity` and `accuracy` are not allowed in `tp_only` mode

```python
from pydantic import ValidationError

try:
    MetricRubricTrait(
        name="invalid",
        evaluation_mode="tp_only",
        metrics=["specificity"],
        tp_instructions=["Mentions BCL2"],
    )
except ValidationError as e:
    print("specificity requires full_matrix mode")

try:
    MetricRubricTrait(
        name="invalid",
        evaluation_mode="full_matrix",
        metrics=["accuracy"],
        tp_instructions=["Mentions BCL2"],
    )
except ValidationError as e:
    print("full_matrix mode requires tn_instructions")
```

## Using Metric Traits in a Rubric

Metric traits are added to a `Rubric` through the `metric_traits` field:

```python
from karenina.schemas import Rubric

coverage_trait = MetricRubricTrait(
    name="Topic Coverage",
    evaluation_mode="tp_only",
    metrics=["recall", "f1"],
    tp_instructions=[
        "Discusses the mechanism of action",
        "Mentions clinical applications",
        "References recent studies",
    ],
)

accuracy_trait = MetricRubricTrait(
    name="Factual Accuracy",
    evaluation_mode="full_matrix",
    metrics=["precision", "recall", "accuracy"],
    tp_instructions=[
        "States the drug targets HER2",
        "Mentions breast cancer indication",
    ],
    tn_instructions=[
        "Claims the drug is a small molecule",
    ],
)

rubric = Rubric(metric_traits=[coverage_trait, accuracy_trait])
print(rubric.get_metric_trait_names())
```

## Next Steps

- [LLM Rubric Traits](llm-traits.md): Open-ended boolean, score, and literal judgments
- [Regex Traits](regex-traits.md): Deterministic pattern matching on response text
- [Callable Traits](callable-traits.md): Custom local Python evaluation logic
- [Rubrics Overview](index.md): When to use each trait type
- [Full Evaluation Benchmark](../../workflows/creating-benchmarks/full-evaluation-benchmark.md): Adding traits to end-to-end benchmarks
