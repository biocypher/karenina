---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Metric Traits

Metric traits measure **extraction completeness** using a confusion-matrix approach. You define a set of instructions describing what the response should (or should not) contain, and the parsing model checks each one. The result is a set of precision, recall, and F1 metrics computed from the confusion matrix.

```python tags=["hide-cell"]
# Mock cell: ensures examples execute without live API keys.
# This cell is hidden in rendered documentation.
```

## Overview

A `MetricRubricTrait` evaluates how well a response covers a set of expected items. Unlike other trait types that return a single boolean or score, metric traits return **multiple metrics** (precision, recall, F1, and optionally specificity and accuracy) computed from a confusion matrix.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | *(required)* | Human-readable identifier |
| `description` | `str \| None` | `None` | What this trait evaluates |
| `evaluation_mode` | `Literal["tp_only", "full_matrix"]` | `"tp_only"` | Evaluation approach |
| `metrics` | `list[str]` | *(required)* | Metrics to compute (mode-dependent) |
| `tp_instructions` | `list[str]` | *(required)* | What SHOULD be present in the answer |
| `tn_instructions` | `list[str]` | `[]` | What should NOT be present (required in `full_matrix` mode) |
| `repeated_extraction` | `bool` | `True` | Deduplicate repeated excerpts |

**Key characteristics:**

- Returns **multiple metrics** (not a single value) as a dictionary
- Requires an **LLM call** -- the parsing model categorizes instructions into confusion matrix buckets
- Two evaluation modes: `tp_only` (precision/recall/F1) and `full_matrix` (adds specificity/accuracy)
- Instructions are natural-language descriptions, not regex patterns

## TP-Only Mode

In `tp_only` mode, you define instructions for what **should be present** in the response. The parsing model then categorizes each instruction:

- **TP (True Positive)**: Instruction found in the answer
- **FN (False Negative)**: Instruction missing from the answer
- **FP (False Positive)**: Extra content not matching any instruction

Available metrics: `precision`, `recall`, `f1`

```python
from karenina.schemas import MetricRubricTrait

# Check if a response covers key references
reference_trait = MetricRubricTrait(
    name="Reference Coverage",
    description="Check if response covers key references",
    evaluation_mode="tp_only",
    metrics=["precision", "recall", "f1"],
    tp_instructions=[
        "Mentions Tsujimoto et al., Science, 1985",
        "Mentions Hockenbery et al., Nature, 1990",
        "Mentions Adams & Cory, Science, 1998",
    ],
)

print(f"Trait: {reference_trait.name}")
print(f"Mode: {reference_trait.evaluation_mode}")
print(f"Metrics: {reference_trait.metrics}")
print(f"TP instructions: {len(reference_trait.tp_instructions)}")
print(f"Required buckets: {reference_trait.get_required_buckets()}")
```

### Metric Formulas

The metrics are computed from the confusion matrix counts:

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| Precision | TP / (TP + FP) | Of items found, how many were expected? |
| Recall | TP / (TP + FN) | Of expected items, how many were found? |
| F1 | 2 * P * R / (P + R) | Harmonic mean of precision and recall |

For example, if you define 3 TP instructions and the response contains 2 of them plus 1 extra item:

- TP = 2, FN = 1, FP = 1
- Precision = 2/3 = 0.67, Recall = 2/3 = 0.67, F1 = 0.67

## Full Matrix Mode

In `full_matrix` mode, you define both what **should** and what **should not** be present. This adds specificity and accuracy to the available metrics:

- **TP (True Positive)**: TP instruction found in the answer
- **FN (False Negative)**: TP instruction missing from the answer
- **TN (True Negative)**: TN instruction correctly absent
- **FP (False Positive)**: TN instruction incorrectly present

Available metrics: `precision`, `recall`, `specificity`, `accuracy`, `f1`

```python
# Check content accuracy with both positive and negative assertions
accuracy_trait = MetricRubricTrait(
    name="Content Accuracy",
    description="Check content accuracy for BCL2 gene information",
    evaluation_mode="full_matrix",
    metrics=["precision", "recall", "specificity", "accuracy", "f1"],
    tp_instructions=[
        "Mentions BCL2 gene",
        "Discusses apoptosis regulation",
        "References cancer research",
    ],
    tn_instructions=[
        "Claims BCL2 is pro-apoptotic",       # Should NOT be present (it's anti-apoptotic)
        "States BCL2 is on chromosome 1",      # Should NOT be present (it's on chromosome 18)
    ],
)

print(f"Trait: {accuracy_trait.name}")
print(f"Mode: {accuracy_trait.evaluation_mode}")
print(f"Metrics: {accuracy_trait.metrics}")
print(f"TP instructions: {len(accuracy_trait.tp_instructions)}")
print(f"TN instructions: {len(accuracy_trait.tn_instructions)}")
print(f"Required buckets: {accuracy_trait.get_required_buckets()}")
```

### Additional Metrics

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| Specificity | TN / (TN + FP) | Of things that should be absent, how many actually are? |
| Accuracy | (TP + TN) / (TP + TN + FP + FN) | Overall correctness across all instructions |

## Choosing a Mode

| Scenario | Mode | Why |
|----------|------|-----|
| Check if key facts are mentioned | `tp_only` | Only care about presence of expected items |
| Verify entity extraction coverage | `tp_only` | Measure what was found vs. missed |
| Check both correct and incorrect claims | `full_matrix` | Need to verify absence of wrong information |
| Evaluate factual accuracy | `full_matrix` | Important to catch both missing truths and present falsehoods |

## The `repeated_extraction` Option

By default, `repeated_extraction=True` deduplicates repeated excerpts using case-insensitive exact matching. This prevents the same instruction from being counted multiple times in the confusion matrix.

```python
# Default: deduplicate repeated extractions
dedup_trait = MetricRubricTrait(
    name="Entity Check",
    evaluation_mode="tp_only",
    metrics=["precision", "recall"],
    tp_instructions=["Mentions mitochondria", "Mentions apoptosis"],
    repeated_extraction=True,  # default
)
print(f"Repeated extraction: {dedup_trait.repeated_extraction}")

# Disable deduplication if the same instruction can legitimately appear multiple times
no_dedup_trait = MetricRubricTrait(
    name="Keyword Frequency",
    evaluation_mode="tp_only",
    metrics=["precision", "recall"],
    tp_instructions=["Mentions mitochondria", "Mentions apoptosis"],
    repeated_extraction=False,
)
print(f"Repeated extraction: {no_dedup_trait.repeated_extraction}")
```

## Confusion Matrix Output

When a metric trait is evaluated during verification, the results include both the computed metrics and the raw confusion matrix lists. This lets you inspect exactly which instructions were found, missed, or incorrectly present.

The result structure in `VerificationResult.rubric_results` includes:

    metric_trait_confusion_lists: dict[str, dict[str, list[str]]]

For example, a TP-only trait might produce:

```python
# Example output structure (from VerificationResult)
confusion_lists = {
    "Reference Coverage": {
        "tp": ["Mentions Tsujimoto et al., Science, 1985", "Mentions Adams & Cory, Science, 1998"],
        "fn": ["Mentions Hockenbery et al., Nature, 1990"],
        "fp": ["Discusses BCL2 protein structure"],
    }
}
```

And a full-matrix trait would additionally include a `"tn"` key with instructions that were correctly absent.

## Validation

Metric traits enforce strict validation at construction time:

- `tp_instructions` must be non-empty (always required)
- `tn_instructions` must be non-empty in `full_matrix` mode
- Metrics must be valid for the chosen mode
- You cannot request `specificity` or `accuracy` in `tp_only` mode (they require TN)

```python
from pydantic import ValidationError

# Trying to use specificity in tp_only mode raises an error
try:
    MetricRubricTrait(
        name="invalid",
        evaluation_mode="tp_only",
        metrics=["specificity"],  # Not available in tp_only mode
        tp_instructions=["test"],
    )
except ValidationError as e:
    print(f"Validation error: specificity requires full_matrix mode")

# Missing tn_instructions in full_matrix mode
try:
    MetricRubricTrait(
        name="invalid",
        evaluation_mode="full_matrix",
        metrics=["precision"],
        tp_instructions=["test"],
        # tn_instructions missing!
    )
except ValidationError as e:
    print(f"Validation error: full_matrix requires tn_instructions")
```

## Using Metric Traits in a Rubric

Metric traits are added to a `Rubric` via the `metric_traits` field:

```python
from karenina.schemas import Rubric

# Create traits for different aspects
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
        "Claims the drug is a small molecule",  # It's a monoclonal antibody
    ],
)

rubric = Rubric(metric_traits=[coverage_trait, accuracy_trait])
print(f"Metric traits: {rubric.get_metric_trait_names()}")
print(f"Total traits: {len(rubric.metric_traits)}")
```

## Next Steps

- [LLM Rubric Traits](llm-traits.md) -- Boolean and score evaluation via LLM judgment
- [Regex Traits](regex-traits.md) -- Pattern matching on raw response text
- [Callable Traits](callable-traits.md) -- Custom Python function evaluation
- [Rubrics Overview](index.md) -- When to use each trait type
- [Defining Rubrics](../../05-creating-benchmarks/defining-rubrics.md) -- Adding traits to benchmarks
