---
jupyter:
  jupytext:
    formats: docs/core_concepts/rubrics//md,docs/notebooks/core_concepts/rubrics//ipynb
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

# Metric Traits

Metric traits use the **parsing model as a bucket assigner** to turn a response into confusion-list buckets, then compute deterministic metrics from those buckets. They are the rubric trait type for **countable coverage and absence checks**: use them when you already know the concrete items a good answer should include or avoid, and you want signals such as precision, recall, F1, specificity, or accuracy. For an overview of all rubric trait types, see the [rubrics index](../../../../core_concepts/rubrics/).

```python tags=["hide-cell"]
# Mock cell: ensures examples execute without live API keys.
# This cell is hidden in rendered documentation.
```

## 1. What Metric Traits Are

A `MetricRubricTrait` evaluates a response during [RubricEvaluation](../../verification-pipeline/) of the [verification pipeline](../../verification-pipeline/). Unlike other trait types that return a single boolean or score, metric traits return a **dictionary of metrics** computed from confusion-list buckets.

Metric traits are meant for evaluations you can decompose into a **small checklist of countable items**. Typical examples include whether a biomedical answer mentions the expected entities, covers the canonical papers, states the key mechanism claims, or avoids a known set of false statements.

Use `MetricRubricTrait` when the evaluation is best expressed as: "Here are the items that should be present, and maybe here are the items that should not be present." If the check requires a holistic judgment like clarity or tone, prefer [LLM traits](../llm-traits/). If the check can be expressed as an exact pattern or local Python function, prefer [Regex traits](../regex-traits/) or [Callable traits](../callable-traits/).

### 1.1 Philosophy

The most important idea is that the instruction lists are the evaluation spec. The metric names do not define your rubric. `precision`, `recall`, and `f1` only tell Karenina **how to summarize the buckets after assignment**. The real scoring surface is:

- `tp_instructions`: the items that should be present
- `tn_instructions`: the items that should be absent in `full_matrix` mode
- `description`: optional scope or domain context

That means good metric traits define **countable, observable units**:

- one fact, entity, reference, or claim per instruction
- the same level of granularity across the list
- boundaries clear enough that one answer snippet matches at most one intended instruction

**The abstraction boundary.** Metric traits are strongest when the evaluation question is "how much of this checklist was covered?" or "did any of these known-wrong claims appear?" They are weaker fits for open-ended correctness or overall answer quality.

| Better fit for Metric Traits | Better fit for other tools |
|------------------------------|----------------------------|
| "How many expected references did the answer cover?" | "Is the answer clear and well organized?" → [LLM trait](../llm-traits/) |
| "Did the answer mention the expected entities and avoid these known false claims?" | "Does the answer match a citation format exactly?" → [Regex trait](../regex-traits/) |
| "What fraction of the expected checklist items were present?" | "Did the parsed answer match the gold structured fields?" → template verification |

A useful litmus test: if you can write the evaluation as a short checklist of atomic items and you care about coverage-style metrics over that checklist, a metric trait is probably the right abstraction.

## 2. Overview

Metric-trait evaluation has two parts:

1. The parsing model reads the question, the raw response trace, and your instructions, then assigns content to confusion-list buckets.
2. Karenina computes the requested metrics programmatically from the bucket counts.

This split matters: the bucket assignment is LLM-based, but the math is deterministic. If the same confusion lists are produced again, the metric values will be identical.

Two modes are available:

| Mode | You define | Karenina can compute | Best For |
|------|------------|----------------------|----------|
| `tp_only` | What **should** be present | `precision`, `recall`, `f1` | Extraction coverage and expected-item recall |
| `full_matrix` | What **should** be present and what **should not** be present | `precision`, `recall`, `specificity`, `accuracy`, `f1` | Coverage plus factual exclusion checks |

In both modes, Karenina stores the derived metrics in `VerificationResult.rubric.metric_trait_scores` and the underlying buckets in `VerificationResult.rubric.metric_trait_confusion_lists`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | *(required)* | Human-readable identifier |
| `description` | `str \| None` | `None` | Optional scope or evaluator context |
| `evaluation_mode` | `Literal["tp_only", "full_matrix"]` | `"tp_only"` | Whether you define only TP instructions or both TP and TN |
| `metrics` | `list[str]` | *(required)* | Metrics to compute for this trait |
| `tp_instructions` | `list[str]` | *(required)* | Items that should be present in the answer |
| `tn_instructions` | `list[str]` | `[]` | Items that should not be present; required in `full_matrix` mode |
| `repeated_extraction` | `bool` | `True` | Deduplicate repeated excerpts using case-insensitive exact matching |

MetricRubricTrait includes `higher_is_better: bool | None`, defaulting to `None`. A value of `None` means directionality does not apply (the returned metrics carry their own interpretation). Set it to `True` or `False` only when downstream analysis tools need an explicit direction signal.

## 3. Why the Instruction Lists Matter

For metric traits, the instruction lists are the scoring surface. The parsing model does not infer your rubric from the metric names alone; it relies on `tp_instructions`, `tn_instructions`, and optionally `description` to understand what counts.

```
Previous Stages
                  │
                  ▼
┌─── RubricEvaluation ─────────────────────────────────┐
│                                                      │
│  Question + Response Trace + TP/TN Instructions      │
│                     │                                │
│                     ▼                                │
│          ┌──────────────────┐                        │
│          │   Parsing Model  │  ← System: evaluator   │
│          │    (LLM call)    │     role               │
│          │                  │  ← User: question +    │
│          │                  │    trace + TP/TN instr.│
│          └────────┬─────────┘                        │
│                   ▼                                  │
│  Confusion lists: TP / FN / FP / TN                  │
│                   │                                  │
│                   ▼                                  │
│  Deterministic metric computation                    │
│  (precision, recall, F1, specificity, accuracy)      │
└───────────────────┬──────────────────────────────────┘
                    │
                    ▼
FinalizeResult
  → VerificationResult.rubric.metric_trait_scores
  → VerificationResult.rubric.metric_trait_confusion_lists
```

Metric traits skip stage 12 (DeepJudgmentRubric), which applies only to [LLM traits](../llm-traits/) with [deep judgment](../../../../advanced-pipeline/deep-judgment-rubrics/) enabled.

The `description` field is useful, but it is supporting context, not the main scoring mechanism. Use it to narrow scope or explain domain context; use the instruction lists to define what is actually counted.

**Good instruction lists are:**

- **Atomic**: each instruction represents one countable thing.
- **Observable**: the evaluator can decide from the response text alone.
- **Non-overlapping**: one answer excerpt should not satisfy many nearly identical instructions.
- **Consistent in granularity**: avoid mixing tiny facts with very broad requirements in the same trait.

**Strong**:

    "Mentions BCL2"
    "States that BCL2 inhibits apoptosis"
    "States BCL2 is on chromosome 18"

**Weak**:

    "Covers the biology well"
    "Is scientifically thorough"
    "Talks about the important details"

## 4. Canonical Worked Example

The easiest way to understand metric traits is to follow one trait all the way from instruction lists to buckets to metrics.

Suppose the question is:

> "Briefly describe BCL2 and why it matters in cancer."

And the answer says:

> "BCL2 is an anti-apoptotic gene that helps cells survive and is important in cancer. It is located on chromosome 1."

### 4.1 TP-Only Version

```python
from karenina.schemas import MetricRubricTrait

coverage_trait = MetricRubricTrait(
    name="BCL2 Coverage",
    description="Check whether the answer covers the core canonical BCL2 facts.",
    evaluation_mode="tp_only",
    metrics=["precision", "recall", "f1"],
    tp_instructions=[
        "Mentions BCL2 gene",
        "States that BCL2 inhibits apoptosis",
        "References cancer relevance",
        "States BCL2 is on chromosome 18",
    ],
)

print(coverage_trait.get_required_buckets())
```

A reasonable bucket assignment for the answer above is:

```python
tp_only_confusion_lists = {
    "BCL2 Coverage": {
        "tp": [
            "BCL2 is an anti-apoptotic gene",
            "helps cells survive",
            "is important in cancer",
        ],
        "fn": ["States BCL2 is on chromosome 18"],
        "fp": ["It is located on chromosome 1"],
        "tn": [],
    }
}
```

This yields:

- TP = 3
- FN = 1
- FP = 1
- Precision = 3 / (3 + 1) = 0.75
- Recall = 3 / (3 + 1) = 0.75
- F1 = 0.75

This example shows an important point: in `tp_only` mode, FP does **not** mean every extra sentence. It means answer content that appears to be trying to satisfy the TP instruction set, but is wrong for that set. The incorrect chromosome claim is a meaningful FP because it is a candidate answer to one of the same factual slots the TP list is measuring.

### 4.2 Full-Matrix Version

Now extend the same trait with explicit wrong claims:

```python
accuracy_trait = MetricRubricTrait(
    name="BCL2 Accuracy",
    description="Check for the core BCL2 facts and known false statements.",
    evaluation_mode="full_matrix",
    metrics=["precision", "recall", "specificity", "accuracy", "f1"],
    tp_instructions=[
        "Mentions BCL2 gene",
        "States that BCL2 inhibits apoptosis",
        "References cancer relevance",
        "States BCL2 is on chromosome 18",
    ],
    tn_instructions=[
        "States BCL2 is on chromosome 1",
        "Claims BCL2 is pro-apoptotic",
    ],
)

print(accuracy_trait.get_required_buckets())
```

For the same answer, a reasonable bucket assignment is:

```python
full_matrix_confusion_lists = {
    "BCL2 Accuracy": {
        "tp": [
            "BCL2 is an anti-apoptotic gene",
            "helps cells survive",
            "is important in cancer",
        ],
        "fn": ["States BCL2 is on chromosome 18"],
        "fp": ["It is located on chromosome 1"],
        "tn": ["Claims BCL2 is pro-apoptotic"],
    }
}
```

This yields:

- TP = 3
- FN = 1
- FP = 1
- TN = 1
- Precision = 3 / (3 + 1) = 0.75
- Recall = 3 / (3 + 1) = 0.75
- Specificity = 1 / (1 + 1) = 0.50
- Accuracy = (3 + 1) / (3 + 1 + 1 + 1) = 4 / 6
- F1 = 0.75

The extra signal in `full_matrix` mode is that the evaluator now has an explicit negative set. Instead of merely treating the wrong chromosome claim as a bad candidate match, it can recognize it as one of the concrete claims that should not appear.

## 5. TP-Only Mode

In `tp_only` mode, you define what **should** appear in the answer. The parsing model then separates the response into three usable buckets:

- **TP (True Positive)**: answer content that correctly matches a TP instruction
- **FN (False Negative)**: expected content from the TP instructions that is missing
- **FP (False Positive)**: answer content that looks like a candidate match in the same domain but should not count

Available metrics: `precision`, `recall`, `f1`

This mode is best when you care about coverage of expected items and do not need an explicit list of forbidden claims.

### 5.1 Writing Good TP Instructions

A good TP list defines the target set clearly enough that recall and precision are interpretable.

- Use one instruction per target fact, entity, reference, or concept.
- Write instructions at the same level of specificity.
- Prefer concrete language over umbrella phrases.
- Avoid pairs like `"Mentions BCL2"` and `"Discusses BCL2"` in the same trait unless they really mean different things.

```python
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

### 5.2 Metric Formulas in `tp_only` Mode

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Precision | TP / (TP + FP) | Of the candidate matches the evaluator accepted or rejected, how many were correct? |
| Recall | TP / (TP + FN) | Of the expected items, how many were found? |
| F1 | 2 * P * R / (P + R) | Balanced summary of precision and recall |

## 6. Full Matrix Mode

In `full_matrix` mode, you define both what **should** appear and what **should not** appear. This gives the evaluator an explicit negative set and unlocks `specificity` and `accuracy`.

- **TP (True Positive)**: answer content that matches a TP instruction
- **FN (False Negative)**: TP instruction content that is missing
- **TN (True Negative)**: TN instructions that are correctly absent
- **FP (False Positive)**: answer content that matches a TN instruction and therefore should not be present

Available metrics: `precision`, `recall`, `specificity`, `accuracy`, `f1`

Use this mode when absence matters, for example factual accuracy, prohibited claims, or mutually exclusive alternatives.

### 6.1 Writing Good TN Instructions

TN instructions should capture concrete wrong claims, not vague badness.

**Strong**:

    "Claims BCL2 is pro-apoptotic"
    "States BCL2 is on chromosome 1"

**Weak**:

    "Contains inaccurate biology"
    "Says something wrong"

Good TN instructions are explicit enough that the evaluator can recognize both the presence and the absence of the claim.

### 6.2 Additional Metrics in `full_matrix` Mode

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Specificity | TN / (TN + FP) | Of the things that should be absent, how many actually stayed absent? |
| Accuracy | (TP + TN) / (TP + TN + FP + FN) | Overall correctness across both expected and forbidden items |

## 7. Choosing a Mode

| Scenario | Mode | Why |
|----------|------|-----|
| Check whether key entities or references were mentioned | `tp_only` | You care about expected coverage, not explicit forbidden claims |
| Measure extraction quality against a known answer set | `tp_only` | Precision, recall, and F1 usually give the right signal |
| Detect both correct facts and known false claims | `full_matrix` | You need an explicit negative set |
| Evaluate factual accuracy where incorrect assertions matter | `full_matrix` | Specificity and accuracy depend on TN instructions |

If you are unsure, start with `tp_only`. Move to `full_matrix` when you can clearly enumerate the important wrong claims you want to guard against.

## 8. The `repeated_extraction` Field

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

## 9. Results and Validation

Metric trait outputs are stored separately from other rubric trait types:

- `VerificationResult.rubric.metric_trait_scores`: metric values per trait
- `VerificationResult.rubric.metric_trait_confusion_lists`: raw TP/TN/FP/FN lists per trait

The exact strings in those lists may be answer excerpts (`tp`, `fp`) or instruction-derived items (`fn`, `tn`), depending on the bucket. This makes it possible to inspect not just the metric values, but *why* those values were produced.

Metric traits enforce validation when you construct them:

- `tp_instructions` must always be non-empty
- `tn_instructions` must be non-empty in `full_matrix` mode
- metric names must be valid
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

Metric traits are added to a `Rubric` through the `metric_traits` field:

```python
from karenina.schemas import Rubric

rubric = Rubric(metric_traits=[coverage_trait, accuracy_trait])
print(rubric.get_metric_trait_names())
```

## 10. Next Steps

- [LLM Rubric Traits](../llm-traits/): open-ended boolean, score, and literal judgments
- [Regex Traits](../regex-traits/): deterministic pattern matching on response text
- [Callable Traits](../callable-traits/): custom local Python evaluation logic
- [Templates vs rubrics](../../template-vs-rubric/): choosing between correctness checks and rubric-style evaluation
- [Rubrics Overview](../../../../core_concepts/rubrics/): when to use each trait type
