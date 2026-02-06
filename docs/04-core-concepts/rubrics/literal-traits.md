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

# Literal Traits

Literal traits are a kind of `LLMRubricTrait` that perform **ordered categorical classification**. Instead of a binary yes/no (boolean) or a numeric scale (score), the parsing model classifies the response into one of several predefined categories. The result is the **class index** — an integer indicating which category was selected.

```python tags=["hide-cell"]
# Mock cell: ensures examples execute without live API keys.
# This cell is hidden in rendered documentation.
```

## Overview

A literal trait is created by setting `kind="literal"` on `LLMRubricTrait` and providing a `classes` dictionary. The classes define the available categories and their descriptions, which are shown to the parsing model.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Human-readable identifier for the trait |
| `description` | `str \| None` | Detailed description shown to the parsing model |
| `kind` | `"literal"` | Must be `"literal"` for categorical classification |
| `classes` | `dict[str, str]` | Class name → description mapping (2-20 classes, order matters) |
| `higher_is_better` | `bool` | Whether higher class indices indicate better performance |

Key characteristics:

- **Ordered**: Dictionary order determines indices (0, 1, 2, ...)
- **Auto-ranged**: `min_score` is set to 0, `max_score` to `len(classes) - 1` automatically
- **Descriptive**: Each class has a name and a description to guide the parsing model
- **2-20 classes**: Must have at least 2 and at most 20 categories

## Tone Classification

A common use case is classifying the tone or style of a response:

```python
from karenina.schemas import LLMRubricTrait

tone_trait = LLMRubricTrait(
    name="Response Tone",
    description="Classify the overall tone of this response.",
    kind="literal",
    classes={
        "overly_simple": "Uses childish language, oversimplifies to the point of inaccuracy",
        "accessible": "Clear and approachable while remaining accurate",
        "technical": "Uses domain-specific jargon, assumes background knowledge",
    },
    higher_is_better=False,  # Context-dependent — no inherent "better" direction
)

print(f"Kind: {tone_trait.kind}")
print(f"Classes: {list(tone_trait.classes.keys())}")
print(f"Score range: {tone_trait.min_score} to {tone_trait.max_score}")
```

The parsing model receives the class names and descriptions, then selects the one that best fits the response. The result is the class index:

- `0` → "overly_simple"
- `1` → "accessible"
- `2` → "technical"

## Quality Tiers

Another common pattern is defining quality levels where order is meaningful:

```python
quality_trait = LLMRubricTrait(
    name="Answer Quality",
    description="Rate the overall quality of this answer.",
    kind="literal",
    classes={
        "poor": "Incorrect, misleading, or largely irrelevant",
        "acceptable": "Broadly correct but missing important details",
        "good": "Correct and well-structured with adequate detail",
        "excellent": "Comprehensive, precise, and well-organized",
    },
    higher_is_better=True,  # Higher index = better quality
)

print(f"Classes: {list(quality_trait.classes.keys())}")
print(f"Score range: {quality_trait.min_score} to {quality_trait.max_score}")
print(f"higher_is_better: {quality_trait.higher_is_better}")
```

Here `higher_is_better=True` because later classes (higher indices) represent better quality:

- `0` → "poor" (worst)
- `1` → "acceptable"
- `2` → "good"
- `3` → "excellent" (best)

## How `higher_is_better` Works

The `higher_is_better` field tells Karenina which direction is "better" when interpreting scores:

| `higher_is_better` | Interpretation | Example |
|---------------------|---------------|---------|
| `True` | Higher class indices are better | Quality: poor(0) → excellent(3) |
| `False` | Lower class indices are better — or no inherent direction | Tone: classification without preference |

This field does **not** change how the parsing model classifies — it only affects how results are interpreted in summaries and comparisons.

## Working with Class Names

`LLMRubricTrait` provides helper methods for converting between class names and indices:

```python
# Get the ordered list of class names
print(f"Class names: {quality_trait.get_class_names()}")

# Get the index for a specific class
print(f"Index of 'good': {quality_trait.get_class_index('good')}")
print(f"Index of 'poor': {quality_trait.get_class_index('poor')}")

# Invalid class names return -1
print(f"Index of 'unknown': {quality_trait.get_class_index('unknown')}")
```

The `get_class_index()` method returns `-1` for unrecognized class names. This value is also accepted by `validate_score()` as a valid error state for literal traits.

## Score Validation

Literal traits validate scores the same way as score traits — the value must be an integer within the auto-derived range:

```python
# Valid scores
print(f"Is 0 valid? {quality_trait.validate_score(0)}")   # First class
print(f"Is 3 valid? {quality_trait.validate_score(3)}")   # Last class
print(f"Is -1 valid? {quality_trait.validate_score(-1)}")  # Error state

# Invalid scores
print(f"Is 4 valid? {quality_trait.validate_score(4)}")    # Out of range
print(f"Is True valid? {quality_trait.validate_score(True)}")  # Boolean rejected
```

## Writing Good Class Descriptions

The quality of class descriptions directly affects how well the parsing model classifies responses. Good descriptions are:

- **Mutually exclusive**: Each class should be clearly distinct from the others
- **Observable**: Describe what the model should look for in the response
- **Ordered consistently**: If using `higher_is_better`, ensure the natural ordering matches

**Good** — clear criteria the model can evaluate:

    "poor": "Incorrect, misleading, or largely irrelevant to the question"
    "acceptable": "Broadly correct but missing important details or nuance"
    "good": "Correct and well-structured with adequate supporting detail"
    "excellent": "Comprehensive, precise, well-organized, and addresses edge cases"

**Weak** — vague or overlapping:

    "bad": "A bad answer"
    "ok": "An okay answer"
    "good": "A good answer"

## Deep Judgment

Like boolean and score traits, literal traits support [deep judgment](llm-traits.md#deep-judgment-optional) for evidence-based evaluation. When enabled, the parsing model extracts verbatim excerpts from the response and provides reasoning for its classification.

```python
deep_quality_trait = LLMRubricTrait(
    name="Answer Quality (Deep)",
    description="Rate the overall quality of this answer.",
    kind="literal",
    classes={
        "poor": "Incorrect, misleading, or largely irrelevant",
        "acceptable": "Broadly correct but missing important details",
        "good": "Correct and well-structured with adequate detail",
        "excellent": "Comprehensive, precise, and well-organized",
    },
    higher_is_better=True,
    deep_judgment_enabled=True,
    deep_judgment_excerpt_enabled=True,
)

print(f"Deep judgment enabled: {deep_quality_trait.deep_judgment_enabled}")
print(f"Excerpt extraction: {deep_quality_trait.deep_judgment_excerpt_enabled}")
```

See [LLM Traits — Deep Judgment](llm-traits.md#deep-judgment-optional) for configuration details including retry attempts, fuzzy match thresholds, and search-enhanced detection.

## Using Literal Traits in a Rubric

Literal traits are added to rubrics just like other trait types — as global or question-specific traits:

```python
from karenina.schemas import Rubric

rubric = Rubric(
    llm_traits=[
        LLMRubricTrait(
            name="Response Tone",
            description="Classify the overall tone.",
            kind="literal",
            classes={
                "overly_simple": "Oversimplifies, childish language",
                "accessible": "Clear and accurate without jargon",
                "technical": "Domain-specific, assumes expertise",
            },
            higher_is_better=False,
        ),
        LLMRubricTrait(
            name="Answer Quality",
            description="Rate the overall quality.",
            kind="literal",
            classes={
                "poor": "Incorrect or irrelevant",
                "acceptable": "Broadly correct, missing details",
                "good": "Correct and well-structured",
                "excellent": "Comprehensive and precise",
            },
            higher_is_better=True,
        ),
    ],
)

print(f"Rubric has {len(rubric.llm_traits)} literal traits")
for trait in rubric.llm_traits:
    print(f"  {trait.name}: {len(trait.classes)} classes, range 0-{trait.max_score}")
```

## Next Steps

- [LLM Traits](llm-traits.md) — boolean and score kinds of `LLMRubricTrait`
- [Regex Traits](regex-traits.md) — pattern matching on raw response text
- [Callable Traits](callable-traits.md) — custom Python evaluation functions
- [Rubrics Overview](index.md) — when to use each trait type
- [Defining Rubrics](../../05-creating-benchmarks/defining-rubrics.md) — adding traits to a benchmark
