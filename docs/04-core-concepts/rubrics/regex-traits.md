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

# Regex Traits

Regex traits use **regular expression pattern matching** to perform deterministic, repeatable checks on LLM responses. They require no LLM call for evaluation -- making them fast, free, and perfectly reproducible.

```python tags=["hide-cell"]
# Mock cell: ensures examples execute without live API keys.
# This cell is hidden in rendered documentation.
```

## Overview

A `RegexTrait` searches the model's response text for a regex pattern. If the pattern is found, the result is `True`; if not, `False`. You can invert this logic with `invert_result` and control case sensitivity with `case_sensitive`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | *(required)* | Human-readable identifier |
| `description` | `str \| None` | `None` | What this trait evaluates |
| `pattern` | `str` | *(required)* | Regex pattern to search for |
| `case_sensitive` | `bool` | `True` | Whether matching is case-sensitive |
| `invert_result` | `bool` | `False` | Invert the boolean result |
| `higher_is_better` | `bool` | *(required)* | Whether a match indicates a positive outcome |

**Key characteristics:**

- Always returns a **boolean** (`True` / `False`)
- No LLM call required -- evaluated locally with Python's `re` module
- Pattern is validated at construction time (invalid regex raises `ValueError`)
- Operates on the **raw response trace** (the model's full output text)

## Basic Usage

The simplest regex trait checks whether a pattern appears in the response:

```python
from karenina.schemas import RegexTrait

# Check that the response includes a citation like [1], [2], etc.
citation_trait = RegexTrait(
    name="Has Citations",
    description="Check that the response includes numbered citations",
    pattern=r"\[\d+\]",
    higher_is_better=True,  # Finding citations is good
)

# Evaluate against sample responses
print(citation_trait.evaluate("The drug targets BCL2 [1] and KRAS [2]."))  # True
print(citation_trait.evaluate("The drug targets BCL2 and KRAS."))          # False
```

## Case-Insensitive Matching

By default, matching is case-sensitive. Set `case_sensitive=False` to ignore case:

```python
keyword_trait = RegexTrait(
    name="Mentions Machine Learning",
    description="Check that the response mentions machine learning",
    pattern=r"\bmachine learning\b",
    case_sensitive=False,
    higher_is_better=True,
)

# Both match because case_sensitive=False
print(keyword_trait.evaluate("Machine Learning is a broad field."))   # True
print(keyword_trait.evaluate("We used machine learning techniques.")) # True
print(keyword_trait.evaluate("We used deep learning techniques."))    # False
```

## Inverted Matching (Negative Checks)

Use `invert_result=True` when you want to check that a pattern is **absent**. The match result is flipped: a match becomes `False`, no match becomes `True`.

```python
# Check that the response does NOT contain hedging language
no_hedging_trait = RegexTrait(
    name="No Hedging",
    description="Ensure the response avoids hedging phrases",
    pattern=r"\b(maybe|perhaps|possibly|might be|could be)\b",
    case_sensitive=False,
    invert_result=True,       # No match = True (good)
    higher_is_better=True,    # True = no hedging = good
)

print(no_hedging_trait.evaluate("The answer is 42."))              # True (no hedging)
print(no_hedging_trait.evaluate("The answer is perhaps 42."))      # False (hedging found)
```

## Enforcing Answer Formats

A common use case is verifying that the model followed a required answering format:

```python
# Verify the answer follows the [ANSWER] format
format_trait = RegexTrait(
    name="Answer Format",
    description="Check that the answer is enclosed in [ANSWER] tags",
    pattern=r"\[ANSWER\].*?\[/ANSWER\]",
    higher_is_better=True,
)

print(format_trait.evaluate("The gene is [ANSWER]BCL2[/ANSWER]."))  # True
print(format_trait.evaluate("The gene is BCL2."))                    # False
```

## The `higher_is_better` Field

This required field tells analysis tools how to interpret the result:

| `higher_is_better` | Meaning |
|--------------------|---------|
| `True` | A match (or inverted non-match) is a **positive** outcome |
| `False` | A match (or inverted non-match) is a **negative** outcome |

Most regex traits use `higher_is_better=True`. Use `False` for traits where finding the pattern indicates a problem, and you are **not** using `invert_result`:

```python
# Detecting prohibited content (match = bad)
prohibited_trait = RegexTrait(
    name="Contains PII",
    description="Detect personally identifiable information (email addresses)",
    pattern=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    higher_is_better=False,  # Finding PII is bad
)

print(prohibited_trait.evaluate("Contact me at user@example.com"))  # True (PII found = bad)
print(prohibited_trait.evaluate("No personal info here."))          # False (no PII = good)
```

!!! tip "invert_result vs higher_is_better"

    These two fields serve different purposes:

    - **`invert_result`** changes the **evaluation output** -- it flips `True` to `False` and vice versa
    - **`higher_is_better`** changes the **interpretation** -- it tells analysis tools whether `True` is good or bad

    For "absence" checks, you can use either approach:

    - `invert_result=True, higher_is_better=True` -- output `True` when pattern is absent (good)
    - `invert_result=False, higher_is_better=False` -- output `True` when pattern is present (bad)

    Both encode the same intent. The `invert_result` approach is usually clearer because `True` means "passed the check."

## Using Regex Traits in a Rubric

Regex traits can be combined with other trait types in a `Rubric`:

```python
from karenina.schemas import RegexTrait, Rubric

quality_rubric = Rubric(
    regex_traits=[
        RegexTrait(
            name="Has Citations",
            pattern=r"\[\d+\]",
            higher_is_better=True,
        ),
        RegexTrait(
            name="No Hedging",
            pattern=r"\b(maybe|perhaps|possibly|might be|could be)\b",
            case_sensitive=False,
            invert_result=True,
            higher_is_better=True,
        ),
    ]
)

print(f"Rubric has {len(quality_rubric.regex_traits)} regex traits:")
for trait in quality_rubric.regex_traits:
    print(f"  {trait.name}: pattern={trait.pattern!r}")
```

## Pattern Validation

The regex pattern is validated at construction time. Invalid patterns raise a `ValueError`:

```python
try:
    bad_trait = RegexTrait(
        name="Invalid",
        pattern=r"[unclosed",  # Invalid regex
        higher_is_better=True,
    )
except ValueError as e:
    print(f"Error: {e}")
```

## Next Steps

- [LLM rubric traits](llm-traits.md) -- subjective assessment via LLM judgment
- [Literal traits](literal-traits.md) -- ordered categorical classification
- [Callable traits](callable-traits.md) -- custom Python functions
- [Metric traits](metric-traits.md) -- precision, recall, F1 computation
- [Defining rubrics](../../05-creating-benchmarks/defining-rubrics.md) -- adding traits to benchmarks
