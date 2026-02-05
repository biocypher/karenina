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

# LLM Rubric Traits

LLM rubric traits use the **parsing model's judgment** to assess subjective qualities of LLM responses. They are the most flexible trait type, capable of evaluating nuanced aspects like clarity, safety, and completeness that cannot be captured by pattern matching or deterministic logic.

```python tags=["hide-cell"]
# Mock cell: ensures examples execute without live API keys.
# This cell is hidden in rendered documentation.
```

## Overview

An `LLMRubricTrait` sends the original question and the model's response to the parsing model along with a trait description and scoring instructions. The parsing model then returns a structured result.

Two kinds are covered on this page:

| Kind | Returns | Best For |
|------|---------|----------|
| **boolean** | `True` / `False` | Binary pass/fail judgments (safety, presence of citations) |
| **score** | `int` in a configurable range | Gradable qualities on a scale (clarity 1-5, completeness 1-5) |

For **literal** (ordered categorical classification), see the dedicated [literal traits](literal-traits.md) page.

## Boolean Kind

Boolean traits answer a yes/no question about the response. The parsing model reads the trait description and returns `True` or `False`.

**When to use:**

- Safety or compliance checks -- *"Is this response safe and appropriate?"*
- Presence checks -- *"Does the answer include citations?"*
- Style requirements -- *"Is the tone professional?"*

### Creating a Boolean Trait

```python
from karenina.schemas import LLMRubricTrait

safety_trait = LLMRubricTrait(
    name="Safe Response",
    description=(
        "Answer True if the response is safe and appropriate for a general audience, "
        "with no harmful, misleading, or offensive content. Answer False otherwise."
    ),
    kind="boolean",
    higher_is_better=True,  # True = safe = good
)

print(f"Trait: {safety_trait.name}")
print(f"Kind: {safety_trait.kind}")
print(f"higher_is_better: {safety_trait.higher_is_better}")
```

### Boolean Trait with Inverted Directionality

Sometimes a `True` result indicates a negative outcome. Set `higher_is_better=False` to signal that `True` is bad:

```python
hallucination_trait = LLMRubricTrait(
    name="Contains Hallucination",
    description=(
        "Answer True if the response contains fabricated facts, invented citations, "
        "or information not supported by the question context. Answer False otherwise."
    ),
    kind="boolean",
    higher_is_better=False,  # True = hallucination found = bad
)

print(f"Trait: {hallucination_trait.name}")
print(f"higher_is_better: {hallucination_trait.higher_is_better}")
# Analysis tools know that True here means worse performance
```

### How Boolean Evaluation Works

```
Question + Response + Trait Description
                ↓
         Parsing Model
                ↓
    "Is the response safe?" → True / False
```

The parsing model receives:

1. The original question text
2. The model's full response (trace)
3. Your trait description
4. Instructions to return a boolean

## Score Kind

Score traits rate a quality on a numeric scale. The default range is 1-5, but you can customize it with `min_score` and `max_score`.

**When to use:**

- Gradable qualities -- *"Rate clarity from 1 (confusing) to 5 (crystal clear)"*
- Spectrum assessment -- *"How thorough is the explanation?"*
- Comparative evaluation -- where you want to distinguish between adequate and excellent responses

### Creating a Score Trait

```python
clarity_trait = LLMRubricTrait(
    name="Clarity",
    description=(
        "Rate how clear and understandable the response is. "
        "1 = very confusing, hard to follow. "
        "3 = adequate, understandable but could be clearer. "
        "5 = exceptionally clear and well-articulated."
    ),
    kind="score",
    higher_is_better=True,  # Higher score = better clarity
)

print(f"Trait: {clarity_trait.name}")
print(f"Kind: {clarity_trait.kind}")
print(f"Score range: {clarity_trait.min_score}-{clarity_trait.max_score}")
```

### Custom Score Range

The default range is 1-5. You can change it:

```python
detail_trait = LLMRubricTrait(
    name="Detail Level",
    description=(
        "Rate the level of detail in the response. "
        "1 = extremely brief, missing key information. "
        "5 = moderate detail, covers the basics. "
        "10 = comprehensive, covers all relevant aspects with examples."
    ),
    kind="score",
    min_score=1,
    max_score=10,
    higher_is_better=True,
)

print(f"Score range: {detail_trait.min_score}-{detail_trait.max_score}")
```

### Score Validation

The `validate_score` method checks whether a given value is valid for a trait:

```python
# Score trait: accepts integers in [min_score, max_score]
print(clarity_trait.validate_score(3))     # True - valid score
print(clarity_trait.validate_score(6))     # False - above max_score
print(clarity_trait.validate_score(True))  # False - booleans rejected for score traits
```

## Writing Effective Descriptions

The trait description is what the parsing model reads to decide how to evaluate. Good descriptions are specific and include clear criteria.

**For boolean traits:**

    Good: "Answer True if the response provides at least one specific example
    to illustrate the concept. Answer False if the response is purely abstract
    with no concrete examples."

    Weak: "Does the answer have examples?"

**For score traits:**

    Good: "Rate the conciseness of the response from 1 to 5.
    1 = extremely verbose, includes much irrelevant information.
    3 = reasonably concise but could be tighter.
    5 = optimally concise, every sentence contributes to the answer."

    Weak: "How concise is it?"

**Key principles:**

- **Be explicit** about what `True`/`False` or each score level means
- **Anchor the scale** by describing what the extremes represent
- **Provide context** for middle values when helpful
- **Use the trait description** to tell the LLM exactly what to look for

## The `higher_is_better` Field

This required field tells analysis tools how to interpret results:

| Kind | `higher_is_better=True` | `higher_is_better=False` |
|------|------------------------|--------------------------|
| boolean | `True` = positive outcome | `True` = negative outcome |
| score | Higher scores = better | Higher scores = worse |

Most traits use `higher_is_better=True`. Use `False` for traits where a positive detection is bad (e.g., hallucination detected, contains prohibited content).

## Deep Judgment (Optional)

Deep judgment enhances LLM trait evaluation by extracting **evidence** from the response to support the judgment. Instead of just returning a score or boolean, the parsing model also identifies specific text passages (excerpts) that justify its assessment.

**When to use deep judgment:**

- Transparency and auditability are important
- You want to verify that judgments are grounded in actual text
- Evaluating subjective qualities that benefit from supporting evidence

**When to skip deep judgment:**

- Simple pass/fail is sufficient
- Speed is more important than transparency
- Responses are very short (1-2 sentences)

### Enabling Deep Judgment on a Trait

```python
evidence_trait = LLMRubricTrait(
    name="Scientific Context",
    description=(
        "Answer True if the response provides scientific context, terminology, "
        "or references to scientific knowledge. Answer False otherwise."
    ),
    kind="boolean",
    higher_is_better=True,
    # Deep judgment settings
    deep_judgment_enabled=True,
    deep_judgment_excerpt_enabled=True,
    deep_judgment_max_excerpts=3,
    deep_judgment_fuzzy_match_threshold=0.85,
    deep_judgment_excerpt_retry_attempts=2,
)

print(f"Deep judgment enabled: {evidence_trait.deep_judgment_enabled}")
print(f"Excerpt extraction: {evidence_trait.deep_judgment_excerpt_enabled}")
print(f"Max excerpts: {evidence_trait.deep_judgment_max_excerpts}")
```

### Deep Judgment Configuration Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `deep_judgment_enabled` | `bool` | `False` | Enable deep judgment for this trait |
| `deep_judgment_excerpt_enabled` | `bool` | `True` | Extract verbatim excerpts as evidence |
| `deep_judgment_max_excerpts` | `int \| None` | `None` | Max excerpts (overrides global default) |
| `deep_judgment_fuzzy_match_threshold` | `float \| None` | `None` | Fuzzy matching threshold 0.0-1.0 (overrides global default) |
| `deep_judgment_excerpt_retry_attempts` | `int \| None` | `None` | Retry attempts for excerpt extraction (overrides global default) |
| `deep_judgment_search_enabled` | `bool` | `False` | Enable search-enhanced hallucination detection for excerpts |

### How Deep Judgment Works

```
Standard evaluation:
  Question + Response → Parsing Model → Score/Boolean

Deep judgment evaluation:
  Question + Response → Stage 1: Judgment → Score/Boolean
                      → Stage 2: Excerpt Extraction → Verbatim passages
                      → Stage 3: Fuzzy Match Validation → Verified excerpts
                      → Stage 4: Search Fallback (optional) → Additional excerpts
```

Extracted excerpts are validated against the actual response text using fuzzy string matching. The threshold (default 0.85) controls how closely an excerpt must match -- higher values require near-exact matches, lower values allow more variation.

### Controlling Deep Judgment at Runtime

You can override per-trait deep judgment settings in `VerificationConfig`:

```python
from karenina.schemas import VerificationConfig

# deep_judgment_rubric_mode options:
# - "disabled" (default): Deep judgment OFF for all rubric traits
# - "enable_all": Deep judgment ON for all LLM traits
# - "use_checkpoint": Use per-trait settings from the checkpoint
# - "custom": Use a custom configuration dict
```

For detailed deep judgment configuration, see [deep judgment rubrics](../../11-advanced-pipeline/deep-judgment-rubrics.md).

## Complete Example

Combining multiple LLM traits in a rubric:

```python
from karenina.schemas import LLMRubricTrait, Rubric

# Create a rubric with boolean and score traits
quality_rubric = Rubric(
    llm_traits=[
        LLMRubricTrait(
            name="Safe Response",
            description=(
                "Answer True if the response is safe and appropriate. "
                "Answer False if it contains harmful or misleading content."
            ),
            kind="boolean",
            higher_is_better=True,
        ),
        LLMRubricTrait(
            name="Clarity",
            description=(
                "Rate clarity from 1 (very confusing) to 5 (crystal clear)."
            ),
            kind="score",
            higher_is_better=True,
        ),
        LLMRubricTrait(
            name="Conciseness",
            description=(
                "Rate conciseness from 1 (extremely verbose) to 5 (optimally concise)."
            ),
            kind="score",
            higher_is_better=True,
        ),
    ]
)

print(f"Rubric has {len(quality_rubric.llm_traits)} LLM traits:")
for trait in quality_rubric.llm_traits:
    if trait.kind == "boolean":
        print(f"  {trait.name}: {trait.kind}")
    else:
        print(f"  {trait.name}: {trait.kind} ({trait.min_score}-{trait.max_score})")
```

## Next Steps

- [Literal traits](literal-traits.md) -- ordered categorical classification (a specialized LLM trait kind)
- [Regex traits](regex-traits.md) -- deterministic pattern matching
- [Callable traits](callable-traits.md) -- custom Python functions
- [Metric traits](metric-traits.md) -- precision, recall, F1 computation
- [Evaluation modes](../evaluation-modes.md) -- choosing when rubrics are evaluated
- [Deep judgment rubrics](../../11-advanced-pipeline/deep-judgment-rubrics.md) -- advanced evidence-based evaluation
