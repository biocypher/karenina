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

# Regex Traits

Regex traits use **regular expression pattern matching** to perform deterministic, repeatable checks on LLM responses. They are the rubric trait type for **exact textual predicates**: use them when success or failure can be defined as "this pattern appears in the raw response trace" or "this pattern does not appear." They require no LLM call for evaluation, making them fast, free, and perfectly reproducible.

```python tags=["hide-cell"]
# Mock cell: ensures examples execute without live API keys.
# This cell is hidden in rendered documentation.
```

## 1. What Regex Traits Are

A `RegexTrait` searches the model's raw response trace for a regex pattern during [RubricEvaluation](../verification-pipeline.md) of the [verification pipeline](../verification-pipeline.md). If the pattern is found, the result is `True`; if not, `False`. You can invert this logic with `invert_result` and control case sensitivity with `case_sensitive`.

Regex traits are meant for checks that can be reduced to an **exact textual rule**. Typical examples include whether a response contains bracket citations, follows a required answer tag format, includes a disclaimer string, or avoids a small set of prohibited phrases.

Use `RegexTrait` when the evaluation is genuinely about the presence or absence of a literal text pattern. If the check requires semantic interpretation, prefer [LLM traits](llm-traits.md). If the check is deterministic but more complex than a regex match, prefer [Callable traits](callable-traits.md).

### 1.1 Philosophy

The most important idea is that the `pattern` field is the evaluation spec. The regex pattern, not the `description`, determines what passes and what fails.

That means good regex traits define **textual contracts**:

- what exact string form should count as a match
- what spelling, punctuation, or whitespace variation should still count
- what nearby text should *not* accidentally count

**The abstraction boundary.** Regex traits are strongest when the evaluation question is "does the text literally contain this form?" They are a poor fit for questions like "does the response discuss safety?" or "does the answer mention evidence in a meaningful way?" unless those ideas are tied to a specific textual marker.

| Better fit for Regex Traits | Better fit for other tools |
|-----------------------------|----------------------------|
| "Does the answer contain bracket citations like `[3]`?" | "Does the answer use evidence well?" → [LLM trait](llm-traits.md) |
| "Is the answer wrapped in `[ANSWER]...[/ANSWER]` tags?" | "Does the answer satisfy structured gold fields?" → template verification |
| "Does the response avoid these exact hedge words?" | "Does the answer follow a deterministic rule that needs parsing logic?" → [Callable trait](callable-traits.md) |

A useful litmus test: if you can point to the literal text span that should make the trait pass or fail, a regex trait is probably the right abstraction.

<div class="admonition tip">
<p class="admonition-title">Regex evaluation also exists inside templates</p>
<p>If you need regex checks as part of template verification rather than as rubric traits, see <a href="../answer-templates.md#regex-checks">Answer Templates, section 4.4: Regex Checks</a>. Template-level <code>self.regex</code> checks run against the same raw response trace, but they live inside the template verification flow instead of the rubric system.</p>
</div>

## 2. Overview

Unlike [LLM traits](llm-traits.md), which send the response to the parsing model for judgment, regex traits are evaluated entirely locally using Python's `re` module. This means zero cost, zero latency beyond the regex match itself, and perfect reproducibility: the same response always produces the same result.

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
- No LLM call required; evaluated locally with Python's `re` module
- Pattern is validated at construction time
- Operates on the **raw response trace** (the model's full output text)

## 3. How Regex Evaluation Works

During rubric evaluation, regex traits are evaluated locally with no LLM call:

```
Previous Stages
                  │
                  ▼
┌─── RubricEvaluation ─────────────────────────────┐
│                                                   │
│  Raw Response Trace                  No LLM call  │
│           │                                       │
│           ▼                                       │
│   re.search(pattern, text)                        │
│           │                                       │
│           ▼                                       │
│   Match found? → True / False                     │
│           │                                       │
│   Apply invert_result (if set)                    │
│           │                                       │
│   Final boolean result                            │
└───────────┬───────────────────────────────────────┘
            │
            ▼
FinalizeResult → VerificationResult.rubric
```

Regex traits skip stage 12 (DeepJudgmentRubric), which applies only to [LLM traits](llm-traits.md) with [deep judgment](../../advanced-pipeline/deep-judgment-rubrics.md) enabled.

Python's `re.search()` scans the entire raw response text. This makes regex traits ideal for format compliance checks, keyword presence, and any requirement that can be written as a literal text pattern.

## 4. Writing Good Patterns

The main challenge with regex traits is not the API. It is choosing the right textual contract.

Good patterns are usually:

- **Specific**: they match the intended textual form, not a broad category of nearby text
- **Bounded**: they use anchors or word boundaries when partial matches would be misleading
- **Tolerance-aware**: they allow the whitespace or punctuation variation you actually expect
- **Readable**: they are short enough that future readers can tell what they enforce

**Weak**:

    r"citation"
    r"ANSWER"
    r".*@.*"

**Stronger**:

    r"\[\d+\]"
    r"\[ANSWER\].*?\[/ANSWER\]"
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"

Two common tools make patterns safer:

- `\b` word boundaries when you want whole words, not substrings
- explicit escaping like `\[` and `\]` when matching literal punctuation

### 4.1 Basic Presence Check

```python
from karenina.schemas import RegexTrait

citation_trait = RegexTrait(
    name="Has Citations",
    description="Check that the response includes numbered citations",
    pattern=r"\[\d+\]",
    higher_is_better=True,
)

print(citation_trait.evaluate("The drug targets BCL2 [1] and KRAS [2]."))  # True
print(citation_trait.evaluate("The drug targets BCL2 and KRAS."))          # False
```

### 4.2 Enforcing Answer Formats

A common use case is verifying that the model followed a required answering format:

```python
format_trait = RegexTrait(
    name="Answer Format",
    description="Check that the answer is enclosed in [ANSWER] tags",
    pattern=r"\[ANSWER\].*?\[/ANSWER\]",
    higher_is_better=True,
)

print(format_trait.evaluate("The gene is [ANSWER]BCL2[/ANSWER]."))  # True
print(format_trait.evaluate("The gene is BCL2."))                   # False
```

## 5. Case Sensitivity

By default, matching is case-sensitive. Set `case_sensitive=False` when the textual contract should ignore case.

```python
keyword_trait = RegexTrait(
    name="Mentions Machine Learning",
    description="Check that the response mentions machine learning",
    pattern=r"\bmachine learning\b",
    case_sensitive=False,
    higher_is_better=True,
)

print(keyword_trait.evaluate("Machine Learning is a broad field."))   # True
print(keyword_trait.evaluate("We used machine learning techniques.")) # True
print(keyword_trait.evaluate("We used deep learning techniques."))    # False
```

Use case-insensitive matching deliberately. If capitalization itself matters to the required format, keep `case_sensitive=True`.

## 6. Inverted Matching

Use `invert_result=True` when you want to check that a pattern is **absent**. The raw regex match is still the same; only the returned boolean is flipped.

```python
no_hedging_trait = RegexTrait(
    name="No Hedging",
    description="Ensure the response avoids hedging phrases",
    pattern=r"\b(maybe|perhaps|possibly|might be|could be)\b",
    case_sensitive=False,
    invert_result=True,
    higher_is_better=True,
)

print(no_hedging_trait.evaluate("The answer is 42."))         # True
print(no_hedging_trait.evaluate("The answer is perhaps 42.")) # False
```

This is usually the clearest way to express an absence check because `True` means the response passed the rule.

## 7. The `higher_is_better` Field

This required field tells analysis tools how to interpret the result:

| `higher_is_better` | Meaning |
|--------------------|---------|
| `True` | A match (or inverted non-match) is a positive outcome |
| `False` | A match (or inverted non-match) is a negative outcome |

`invert_result` and `higher_is_better` solve different problems:

- `invert_result` changes the **evaluation output**
- `higher_is_better` changes the **downstream interpretation**

Most regex traits use `higher_is_better=True`. Use `False` when finding the pattern indicates a problem and you want the raw `True` result to mean "problem detected."

```python
prohibited_trait = RegexTrait(
    name="Contains PII",
    description="Detect personally identifiable information (email addresses)",
    pattern=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    higher_is_better=False,
)

print(prohibited_trait.evaluate("Contact me at user@example.com"))  # True
print(prohibited_trait.evaluate("No personal info here."))          # False
```

<div class="admonition tip">
<p class="admonition-title">invert_result vs higher_is_better</p>
<p>These two fields serve different purposes:</p>
<ul>
<li><strong><code>invert_result</code></strong> changes the <strong>evaluation output</strong>: it flips <code>True</code> to <code>False</code> and vice versa</li>
<li><strong><code>higher_is_better</code></strong> changes the <strong>interpretation</strong>: it tells analysis tools whether <code>True</code> is good or bad</li>
</ul>
<p>For "absence" checks, you can use either approach:</p>
<ul>
<li><code>invert_result=True, higher_is_better=True</code>: output <code>True</code> when pattern is absent (good)</li>
<li><code>invert_result=False, higher_is_better=False</code>: output <code>True</code> when pattern is present (bad)</li>
</ul>
<p>Both encode the same intent. The <code>invert_result</code> approach is usually clearer because <code>True</code> means "passed the check."</p>
</div>

## 8. Using Regex Traits in a Rubric

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

## 9. Pattern Validation

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

## 10. Next Steps

- [LLM rubric traits](llm-traits.md): boolean, score, and literal kinds
- [Callable traits](callable-traits.md): custom Python functions
- [Metric traits](metric-traits.md): precision, recall, F1 computation
- [Templates vs rubrics](../../notebooks/core_concepts/template-vs-rubric.ipynb): choosing between correctness checks and rubric-style evaluation
- [Full Evaluation Benchmark](../../workflows/creating-benchmarks/full-evaluation-benchmark.md): adding traits to benchmarks
