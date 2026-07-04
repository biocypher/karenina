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

# Verification Primitives

Verification primitives are deterministic comparison functions that decide whether a judge-extracted value is correct. Each primitive is assigned to a field via `VerifiedField(verify_with=...)` and runs after the Judge LLM has parsed the response into a structured [answer template](answer-templates.md). Primitives never call an LLM; they execute pure Python comparison logic against a known ground truth or against parameters embedded in the primitive itself.

```python tags=["hide-cell"]
# Mock cell: ensures examples execute without live API keys.
# This cell is hidden in rendered documentation.
from unittest.mock import MagicMock, patch

mock_modules = {}
for mod in [
    "sqlalchemy", "sqlalchemy.engine", "sqlalchemy.orm", "sqlalchemy.ext",
    "sqlalchemy.ext.declarative", "sqlalchemy.engine",
    "sqlalchemy.sql", "sqlalchemy.event",
    "karenina.storage", "karenina.storage.base",
    "karenina.storage.engine", "karenina.storage.db_config",
    "karenina.storage.models", "karenina.storage.generated_models",
    "karenina.storage.auto_mapper", "karenina.storage.operations",
]:
    mock_modules[mod] = MagicMock()

with patch.dict("sys.modules", mock_modules):
    from karenina.schemas.entities import BaseAnswer, VerifiedField
    from karenina.schemas.primitives import (
        BooleanMatch,
        ContainsAll,
        ContainsAny,
        DateMatch,
        DateRange,
        DateTolerance,
        ExactMatch,
        LiteralMatch,
        NumericExact,
        NumericRange,
        NumericTolerance,
        OrderedMatch,
        RegexMatch,
        SemanticMatch,
        SetContainment,
        TraceContains,
        TraceLength,
        TraceRegex,
    )
    from karenina.schemas.primitives import SynonymMap
```

```python
from typing import Literal

from karenina.schemas.entities import BaseAnswer, VerifiedField
from karenina.schemas.primitives import (
    BooleanMatch,
    ContainsAll,
    ContainsAny,
    DateMatch,
    DateRange,
    DateTolerance,
    ExactMatch,
    LiteralMatch,
    NumericExact,
    NumericRange,
    NumericTolerance,
    OrderedMatch,
    RegexMatch,
    SemanticMatch,
    SetContainment,
    TraceContains,
    TraceLength,
    TraceRegex,
)
from karenina.schemas.primitives import SynonymMap
```

## 1. What Verification Primitives Are

A verification primitive is a small, stateless function object that answers one question: **does the extracted value match what was expected?** It receives two inputs (the judge-extracted value and the ground truth), applies a comparison rule, and returns `True` or `False`. That is its entire job.

Primitives are the last link in the [answer template](answer-templates.md) evaluation chain:

```
Response (free text)  →  Judge LLM  →  Extracted values  →  Primitives  →  Pass/Fail
```

Primitives do not see the original question, the full response text, or the judge's reasoning. They see only the extracted value and the ground truth (or, for trace primitives, the raw response text). This deliberate narrowness makes them deterministic: the same extracted value always produces the same verdict, regardless of which judge model was used.

### What Primitives Are Not

Primitives are not [rubric traits](rubrics/index.md). Rubrics assess observable qualities of the response (safety, conciseness, citation style) without requiring ground truth. Primitives verify correctness against a known expected answer. Rubrics answer "how well?"; primitives answer "right or wrong?"

Primitives are also not the verification *pipeline*. The [verification pipeline](verification-pipeline.md) orchestrates 13 stages including answer generation, judge parsing, and result storage. Primitives participate in a single stage: `verify_template` (stage 8).

## 2. Core Idea: Deterministic Checks After LLM Parsing

The most important idea behind verification primitives is the **separation between LLM judgment and programmatic verification**.

In Karenina's LLM-as-judge approach, only the Judge LLM reads the response and extracts structured data. Once the judge has filled in the template, everything that follows is deterministic code. Primitives are that deterministic code. They never consult an LLM. Swap out the judge model, and the same primitives still produce the same verdict from the same extracted values.

This separation provides three properties:

1. **Reproducibility**: given the same extracted values, verification always produces the same result
2. **Transparency**: you can inspect exactly why a field passed or failed by examining the primitive's parameters and the extracted value
3. **Speed**: primitives execute in microseconds with no API calls, no latency, no cost

The alternative, asking an LLM "is this answer correct?", is what [rubric traits](rubrics/index.md) do for subjective qualities. For factual correctness with known ground truth, primitives are faster, cheaper, and more reliable.

### Walkthrough: From Response to Verdict

Suppose the benchmark asks: *"How many pairs of chromosomes does a normal human somatic cell have?"*

**You define the template:**

```python
class Answer(BaseAnswer):
    pair_count: int = VerifiedField(
        description="The number of chromosome pairs in a normal human somatic cell",
        ground_truth=23,
        verify_with=NumericExact(),
    )
```

**The answering model responds:**

> "Human somatic cells are diploid, containing 46 chromosomes organized into 23 pairs..."

**The Judge LLM extracts:** `{"pair_count": 23}` (guided by the field description and type)

**The primitive runs the check:** `NumericExact().check(23, 23)` returns `True`

The judge does not see `ground_truth` or `verify_with`. The primitive does not see the original response. Each party does one job.

```python
# Direct instantiation to demonstrate the primitive in isolation.
# In practice, the pipeline handles parsing and verification automatically.
parsed = Answer(pair_count=23)
print(f"Correct:  pair_count=23  -> verify(): {parsed.verify()}")

parsed_wrong = Answer(pair_count=46)
print(f"Wrong:    pair_count=46  -> verify(): {parsed_wrong.verify()}")
```

## 3. Two Categories: Parsed vs Trace

Karenina provides 23 primitives in two categories:

| Category | Count | Operates on | Included in judge schema? | Method |
|----------|-------|-------------|--------------------------|--------|
| **Parsed** | 20 | Judge-extracted field value + ground truth | Yes | `check(extracted, expected)` |
| **Trace** | 3 | Raw LLM response text | No (field excluded from schema) | `check_trace(raw_trace)` |

**Parsed primitives** are the default. The Judge LLM sees the field in the JSON schema, extracts a value, and the primitive compares that value against `ground_truth`. Most evaluation uses parsed primitives.

**Trace primitives** bypass the judge entirely for that field. The field is removed from the JSON schema sent to the judge, so the judge never attempts to extract it. Instead, the pipeline runs the primitive directly against the raw response text. Trace fields must be typed as `bool` because the primitive returns a boolean (pattern found or not found), and that result is compared against `ground_truth` to determine pass or fail (see [Section 6](#6-trace-primitives) for details).

### When to Use Each Category

| Situation | Use |
|-----------|-----|
| Value must be interpreted from natural language | Parsed primitive |
| Value must be extracted from context (synonyms, paraphrases) | Parsed primitive |
| Check is a simple pattern match (regex, substring) | Trace primitive |
| Check is a mechanical constraint (response length) | Trace primitive |
| You need the extracted value in results for analysis | Parsed primitive |

Use trace primitives when the check is a pure pattern match or length constraint that does not benefit from LLM interpretation. For example, checking whether the response contains a clinical trial identifier (`NCT\d{8}`) is more reliable as a regex than as a judge extraction.

## 4. Choosing the Right Primitive

### By Data Type

| Field Type | Natural Primitive | When to Use an Alternative |
|-----------|-------------------|---------------------------|
| `bool` | `BooleanMatch` | Always use `BooleanMatch` for parsed booleans |
| `str` | `ExactMatch` | `ContainsAny`/`ContainsAll` for multiple acceptable answers; `RegexMatch` for format validation; `SemanticMatch` for meaning-based comparison |
| `int`, `float` | `NumericExact` | `NumericTolerance` for measurements with acceptable variance; `NumericGraded` for distance-graded partial credit; `NumericRange` when no single correct value exists; `NumericMinimum`/`NumericMaximum` for one-sided bounds; `NumericRangeGraded` and `NumericThresholdGraded` for the graded versions of a band and a one-sided bound |
| `list[str]` | `SetContainment` | `OrderedMatch` when element order matters |
| `Literal[...]` | `LiteralMatch` | Always use `LiteralMatch` for Literal fields |
| `str` (date) | `DateMatch` | `DateTolerance` for approximate dates; `DateRange` when any date in a window is acceptable |

### By Verification Need

| Need | Primitive | Key Parameter |
|------|-----------|--------------|
| Exact string after normalization | `ExactMatch` | `normalize` |
| Any of several acceptable substrings | `ContainsAny` | `substrings` |
| All required terms present | `ContainsAll` | `substrings` |
| Format matches a pattern | `RegexMatch` | `pattern` |
| Meaning is similar (requires embeddings) | `SemanticMatch` | `threshold` |
| Exact number | `NumericExact` | (none) |
| Number within tolerance | `NumericTolerance` | `tolerance`, `mode` |
| Number graded by distance (partial credit) | `NumericGraded` | `cutoff`, `full_credit` |
| Number in a range | `NumericRange` | `min`, `max`, `exclusive_min`, `exclusive_max` |
| Number in a range, graded outside it | `NumericRangeGraded` | `min`, `max`, `margin` |
| Number at least N (`ground_truth`) | `NumericMinimum` | `exclusive` |
| Number at most N (`ground_truth`) | `NumericMaximum` | `exclusive` |
| Number past a one-sided bound, graded near it | `NumericThresholdGraded` | `direction`, `margin` |
| Set membership | `SetContainment` | `mode` |
| Ordered list equality | `OrderedMatch` | `normalize` |
| Fixed category match | `LiteralMatch` | (none) |
| Date equality | `DateMatch` | `format` |
| Date within tolerance | `DateTolerance` | `tolerance`, `unit` |
| Date in a range | `DateRange` | `min`, `max` |
| Regex in raw response | `TraceRegex` | `pattern`, `count_min` |
| Substring in raw response | `TraceContains` | `substring` |
| Response length constraint | `TraceLength` | `min`, `max`, `unit` |

### Decision Heuristics

- **Start simple.** `BooleanMatch`, `ExactMatch`, and `NumericExact` cover most cases. Reach for more complex primitives only when these are insufficient.
- **Use normalization before adding alternatives.** If `ExactMatch` fails because of case or whitespace differences, add normalizers rather than switching to `ContainsAny`.
- **Prefer parsed primitives over trace primitives** unless the check is a pure pattern match. Parsed primitives benefit from the judge's ability to interpret context and synonyms.
- **Use `ground_truth` when a single correct answer exists.** Use parameter-based primitives (`ContainsAny`, `NumericRange`, `NumericMinimum`, `NumericMaximum`, `DateRange`) when the answer space is a set or range rather than a single point.

## 5. Parsed Primitives Reference

All parsed primitives subclass `VerificationPrimitive` and implement `check(extracted, expected) -> bool`. The `extracted` argument is the value the Judge LLM produced; `expected` is the `ground_truth` from `VerifiedField`.

### 5.1. Boolean

#### BooleanMatch

Compare the extracted boolean to the ground truth boolean. Both values are coerced to `bool` before comparison.

No parameters.

**Applies to:** `bool`

```python
class Answer(BaseAnswer):
    is_approved: bool = VerifiedField(
        description="Whether the drug is FDA-approved",
        ground_truth=True,
        verify_with=BooleanMatch(),
    )


parsed = Answer(is_approved=True)
print(f"True  -> verify(): {parsed.verify()}")

parsed_wrong = Answer(is_approved=False)
print(f"False -> verify(): {parsed_wrong.verify()}")
```

---

### 5.2. String

#### ExactMatch

Normalize both values, then compare for string equality.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `normalize` | `list[Normalizer]` | `["lowercase", "strip"]` | Normalizers applied to both values before comparison |

**Applies to:** `str`, `int`, `float`

```python
class Answer(BaseAnswer):
    target: str = VerifiedField(
        description="Protein target name",
        ground_truth="BCL2",
        verify_with=ExactMatch(normalize=["lowercase", "strip"]),
    )


for value in ["BCL2", "bcl2", " Bcl2 ", "KRAS"]:
    parsed = Answer(target=value)
    print(f"{value!r:>10} -> verify(): {parsed.verify()}")
```

#### ContainsAny

Pass if the extracted text contains at least one of the specified substrings. This primitive ignores `ground_truth`; the expected values are supplied via the `substrings` parameter.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `substrings` | `list[str]` | required | At least one must appear in the extracted value |
| `normalize` | `list[Normalizer]` | `[]` | Normalizers applied before comparison |

**Applies to:** `str`

```python
class Answer(BaseAnswer):
    mechanism: str = VerifiedField(
        description="Mechanism of action described in the response",
        ground_truth="N/A",  # ignored by ContainsAny; required by VerifiedField
        verify_with=ContainsAny(substrings=["apoptosis", "autophagy"]),
    )


parsed = Answer(mechanism="Induces apoptosis by inhibiting BCL2")
print(f"Contains 'apoptosis': {parsed.verify()}")

parsed_miss = Answer(mechanism="Inhibits cell proliferation")
print(f"Contains neither:     {parsed_miss.verify()}")
```

<div class="admonition note">
<p class="admonition-title">Primitives that ignore ground_truth</p>
<p><code>ContainsAny</code>, <code>ContainsAll</code>, <code>RegexMatch</code>, <code>NumericRange</code>, and <code>DateRange</code> carry their expected values in constructor parameters, not in <code>ground_truth</code>. You must still provide a <code>ground_truth</code> value because it is a required parameter of <code>VerifiedField</code>, but the primitive does not use it. By convention, use a placeholder such as <code>"N/A"</code> for strings or <code>0</code> for numbers.</p>
</div>

#### ContainsAll

Pass if the extracted text contains all of the specified substrings. This primitive ignores `ground_truth`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `substrings` | `list[str]` | required | All must appear in the extracted value |
| `normalize` | `list[Normalizer]` | `[]` | Normalizers applied before comparison |

**Applies to:** `str`

```python
class Answer(BaseAnswer):
    summary: str = VerifiedField(
        description="Trial design summary",
        ground_truth="N/A",
        verify_with=ContainsAll(substrings=["phase III", "randomized", "double-blind"]),
    )


parsed = Answer(summary="A phase III, randomized, double-blind study")
print(f"Contains all:  {parsed.verify()}")

parsed_miss = Answer(summary="A phase III, open-label study")
print(f"Missing terms: {parsed_miss.verify()}")
```

#### RegexMatch

Pass if the extracted text matches the specified regex pattern. This primitive ignores `ground_truth`. Uses `re.search()`, so the pattern can match anywhere in the string.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pattern` | `str` | required | Regular expression to match against the extracted value |
| `flags` | `list[str]` | `[]` | Regex flag names (e.g., `["IGNORECASE"]`) |

**Applies to:** `str`

```python
class Answer(BaseAnswer):
    identifier: str = VerifiedField(
        description="ClinicalTrials.gov identifier",
        ground_truth="N/A",
        verify_with=RegexMatch(pattern=r"NCT\d{8}"),
    )


parsed = Answer(identifier="NCT02141282")
print(f"Valid ID:   {parsed.verify()}")

parsed_bad = Answer(identifier="CT-2014-001")
print(f"Invalid ID: {parsed_bad.verify()}")
```

#### SemanticMatch

Pass if the embedding similarity between the extracted value and `ground_truth` meets the threshold.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | `float` | `0.85` | Minimum cosine similarity score to pass |

**Applies to:** `str`

```python
class Answer(BaseAnswer):
    rationale: str = VerifiedField(
        description="Clinical rationale for the treatment",
        ground_truth="Targets the BCL2 anti-apoptotic protein",
        verify_with=SemanticMatch(threshold=0.80),
    )
```

<div class="admonition warning">
<p class="admonition-title">SemanticMatch requires the embedding_check pipeline stage</p>
<p><code>SemanticMatch</code> cannot be called directly; its <code>check()</code> method raises <code>NotImplementedError</code>. It serves as a marker that tells the pipeline's <code>embedding_check</code> stage (stage 9) to compute embedding similarity for this field. You must enable <code>embedding_check</code> in your <code>VerificationConfig</code> and configure an embedding model. Calling <code>verify()</code> directly on a template with <code>SemanticMatch</code> fields will raise an error.</p>
</div>

---

### 5.3. Numeric

#### NumericExact

Pass if the extracted value equals the ground truth after float coercion.

No parameters.

**Applies to:** `int`, `float`

```python
class Answer(BaseAnswer):
    patient_count: int = VerifiedField(
        description="Number of patients enrolled",
        ground_truth=342,
        verify_with=NumericExact(),
    )


parsed = Answer(patient_count=342)
print(f"342 -> {parsed.verify()}")

parsed_wrong = Answer(patient_count=340)
print(f"340 -> {parsed_wrong.verify()}")
```

#### NumericTolerance

Pass if the extracted value is within a specified tolerance of the ground truth.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tolerance` | `float` | required | Allowed deviation from ground truth |
| `mode` | `Literal["relative", "absolute"]` | `"relative"` | `"relative"`: fraction of ground truth; `"absolute"`: raw difference |

**Applies to:** `int`, `float`

In `"relative"` mode, the check is `|extracted - expected| / |expected| <= tolerance`. When the expected value is zero, only an exact match passes. In `"absolute"` mode, the check is `|extracted - expected| <= tolerance`.

```python
class Answer(BaseAnswer):
    hazard_ratio: float = VerifiedField(
        description="Hazard ratio from primary analysis",
        ground_truth=0.72,
        verify_with=NumericTolerance(tolerance=0.05, mode="absolute"),
    )


for value in [0.72, 0.70, 0.77, 0.80]:
    parsed = Answer(hazard_ratio=value)
    print(f"{value} -> verify(): {parsed.verify()}")
```

<div class="admonition tip">
<p class="admonition-title">Choosing tolerance values</p>
<p>Use <code>NumericExact()</code> for exact counts (chromosomes, enrolled patients). Use <code>NumericTolerance(tolerance=..., mode="absolute")</code> for physical measurements with known precision (body temperature, boiling points). Use <code>NumericTolerance(tolerance=..., mode="relative")</code> for values that span wide ranges where a percentage margin makes more sense (e.g., <code>tolerance=0.1</code> to accept within 10%).</p>
</div>

#### NumericRange

Pass if the extracted value falls within the specified bounds. Bounds are inclusive by default; set `exclusive_min` or `exclusive_max` to use strict inequality on the corresponding side. This primitive ignores `ground_truth`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min` | `float \| None` | `None` | Lower bound; no lower bound if `None` |
| `max` | `float \| None` | `None` | Upper bound; no upper bound if `None` |
| `exclusive_min` | `bool` | `False` | If `True`, use strict `>` for the lower bound instead of `>=` |
| `exclusive_max` | `bool` | `False` | If `True`, use strict `<` for the upper bound instead of `<=` |

**Applies to:** `int`, `float`

```python
class Answer(BaseAnswer):
    p_value: float = VerifiedField(
        description="Primary endpoint p-value",
        ground_truth=0,
        verify_with=NumericRange(min=0.0, max=0.05),
    )


for value in [0.001, 0.05, 0.10]:
    parsed = Answer(p_value=value)
    print(f"{value} -> verify(): {parsed.verify()}")
```

---

#### NumericMinimum

Pass if the extracted value is at least the `ground_truth` value (inclusive by default).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `exclusive` | `bool` | `False` | If `True`, use strict `>` instead of `>=` |

**Applies to:** `int`, `float`

---

#### NumericMaximum

Pass if the extracted value does not exceed the `ground_truth` value (inclusive by default).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `exclusive` | `bool` | `False` | If `True`, use strict `<` instead of `<=` |

**Applies to:** `int`, `float`

---

#### NumericGraded

Score the extracted value by its distance from the `ground_truth`, giving partial credit that decays to zero at a cutoff. This is the one numeric primitive whose contribution to `verify_granular()` is **continuous** rather than 0/1: a near-miss earns a fractional score. `verify()` stays binary, driven by `check()` as for every primitive. The reference value is carried in `ground_truth` (as with `NumericTolerance`).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cutoff` | `float` | required | Distance at which graded credit reaches 0. Must be `> 0` |
| `full_credit` | `float \| None` | `None` | Optional inner band that earns full credit. When set, must satisfy `0 <= full_credit < cutoff` |
| `mode` | `Literal["relative", "absolute"]` | `"relative"` | `"relative"`: distance is `\|extracted - expected\| / \|expected\|` (a fraction, e.g. `0.10` is 10%); `"absolute"`: raw difference (percentage-points when the reference is itself a percentage) |
| `decay` | `Literal["linear", "quadratic"]` | `"linear"` | Shape of the decay between the inner band and the cutoff |

**Applies to:** `int`, `float`

There are two band shapes:

- **Single-band** (`full_credit` unset): `cutoff` is both the binary gate and the zero-credit distance. `check()` passes anywhere within the cutoff, and the score decays from 1.0 at the reference to 0.0 at the cutoff.
- **Double-band** (`full_credit` set): the score is 1.0 within the inner `full_credit` band, decays to 0.0 at the cutoff, and is 0.0 beyond. `check()` gates at the **inner** band, so the binary pass stays tight (for example at a known reporting precision) while a near-miss between `full_credit` and `cutoff` is `verify()` False yet still earns partial credit in `verify_granular()`.

When the reference is zero in `"relative"` mode, only an exact match scores (mirroring `NumericTolerance`); use `"absolute"` mode whenever the reference can be zero. The per-field graded scores are surfaced alongside the binary results: see the `field_scores` field on the result and the `field_score` column in the [results DataFrame](../workflows/analyzing-results/dataframe-analysis.md).

---

#### NumericRangeGraded

Score the extracted value against an acceptance band `[min, max]` with soft shoulders, giving partial credit that decays to zero outside the band. This is the graded companion to `NumericRange`: `check()` passes only inside the band (the same hard gate), and `score()` additionally awards decaying credit to values that fall just outside, out to a margin. The band is carried in the `min` and `max` parameters, so `ground_truth` is not used for scoring.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min` | `float` | required | Lower edge of the full-credit band |
| `max` | `float` | required | Upper edge of the full-credit band. Must satisfy `min < max` |
| `margin` | `float` | required | Shoulder width. Must be `> 0` |
| `mode` | `Literal["relative", "absolute"]` | `"absolute"` | `"absolute"`: the shoulder is `margin` raw units wide on each side; `"relative"`: the shoulder is `margin` times the band width `(max - min)` |
| `decay` | `Literal["linear", "quadratic"]` | `"linear"` | Shape of the decay across a shoulder |
| `exclusive_min` | `bool` | `False` | If `True`, the binary gate treats the lower edge as exclusive: a value exactly at `min` fails `check()`. `score()` is unaffected (a value on the edge still earns full credit) |
| `exclusive_max` | `bool` | `False` | If `True`, the binary gate treats the upper edge as exclusive: a value exactly at `max` fails `check()`. `score()` is unaffected |

**Applies to:** `int`, `float`

A value inside `[min, max]` scores 1.0. Outside, credit decays from 1.0 at the nearer edge to 0.0 at that edge plus or minus the shoulder, and is 0.0 beyond. `check()` stays binary at the band edges, so a near-miss just outside the band is `verify()` False yet still earns partial credit in `verify_granular()`. Use this when the intended answer is genuinely an interval where any value inside is equally correct. When a single reference point exists, prefer `NumericGraded` centered on that point.

---

#### NumericThresholdGraded

Score the extracted value against a one-sided bound with a soft shoulder, giving full credit anywhere on the correct side and decaying partial credit just past the bound. This is the graded companion to `NumericMinimum` and `NumericMaximum`: the threshold is carried in `ground_truth` (the same convention), and `direction` selects the side.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `direction` | `Literal["max", "min"]` | required | `"max"`: pass iff `extracted <= threshold`; `"min"`: pass iff `extracted >= threshold` |
| `margin` | `float` | required | Shoulder width past the threshold. Must be `> 0` |
| `mode` | `Literal["relative", "absolute"]` | `"relative"` | `"relative"`: the shoulder is `margin` times `\|threshold\|`; `"absolute"`: raw units |
| `decay` | `Literal["linear", "quadratic"]` | `"linear"` | Shape of the decay across the shoulder |
| `exclusive` | `bool` | `False` | If `True`, the binary gate uses a strict inequality at the threshold |

**Applies to:** `int`, `float`

Unlike `NumericGraded`, which is symmetric around a point, a value far on the correct side of the threshold still scores 1.0. A value on the wrong side earns decaying credit out to the margin, then 0.0. `check()` matches `NumericMaximum` or `NumericMinimum` exactly, so converting a one-sided bound to this primitive preserves the binary pass while adding partial credit for near-misses.

---

### 5.4. List

#### SetContainment

Compare lists as sets with configurable containment modes.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | `str` | `"exact"` | `"exact"`, `"subset"`, `"superset"`, or `"overlap"` |
| `min_overlap` | `int \| None` | `None` | Minimum shared elements; only used in `"overlap"` mode (defaults to 1 if not set) |

**Applies to:** `list[str]`

| Mode | Passes when | Use case |
|------|-------------|----------|
| `"exact"` | Extracted set equals expected set | Must name all and only the expected items |
| `"subset"` | Extracted is a subset of expected | Extracted items must all be valid; not all expected items are required |
| `"superset"` | Extracted is a superset of expected | All expected items must appear; extra items are acceptable |
| `"overlap"` | At least `min_overlap` elements in common | Partial credit or flexible matching |

```python
class Answer(BaseAnswer):
    indications: list[str] = VerifiedField(
        description="Approved indications listed in the response",
        ground_truth=["CLL", "SLL", "AML"],
        verify_with=SetContainment(mode="superset"),
    )


parsed = Answer(indications=["CLL", "SLL", "AML", "NHL"])
print(f"Superset (extra OK): {parsed.verify()}")

parsed_missing = Answer(indications=["CLL", "SLL"])
print(f"Missing AML:         {parsed_missing.verify()}")
```

#### OrderedMatch

Pass if each element of the extracted list matches the corresponding element of the ground truth list after normalization. Lists must have the same length.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `normalize` | `list[Normalizer]` | `["lowercase", "strip"]` | Normalizers applied to each element before comparison |

**Applies to:** `list[str]`

```python
class Answer(BaseAnswer):
    authors: list[str] = VerifiedField(
        description="Author names in order of appearance",
        ground_truth=["Smith J", "Jones A", "Patel R"],
        verify_with=OrderedMatch(normalize=["lowercase", "strip"]),
    )


parsed = Answer(authors=["smith j", "jones a", "patel r"])
print(f"Correct order: {parsed.verify()}")

parsed_wrong = Answer(authors=["Jones A", "Smith J", "Patel R"])
print(f"Wrong order:   {parsed_wrong.verify()}")
```

---

### 5.5. Categorical

#### LiteralMatch

Pass if the extracted value exactly matches the ground truth. Designed for fields typed as `Literal[...]`, where Pydantic generates an `enum` in the JSON schema that constrains the judge to a fixed set of values.

No parameters.

**Applies to:** `Literal[...]`

```python
class Answer(BaseAnswer):
    trial_phase: Literal["I", "II", "III", "IV"] = VerifiedField(
        description="Clinical trial phase",
        ground_truth="III",
        verify_with=LiteralMatch(),
    )


parsed = Answer(trial_phase="III")
print(f"Correct phase: {parsed.verify()}")

parsed_wrong = Answer(trial_phase="II")
print(f"Wrong phase:   {parsed_wrong.verify()}")
```

---

### 5.6. Date and Time

#### DateMatch

Parse both values as dates and compare for equality. Only the date portion is compared; time is ignored. When `format` is `None`, python-dateutil is used for flexible date parsing.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `format` | `str \| None` | `None` | strptime format string; uses python-dateutil if `None` |

**Applies to:** `str`

```python
class Answer(BaseAnswer):
    approval_date: str = VerifiedField(
        description="FDA approval date",
        ground_truth="2016-04-11",
        verify_with=DateMatch(),
    )


parsed = Answer(approval_date="April 11, 2016")
print(f"Flexible parsing: {parsed.verify()}")

parsed_wrong = Answer(approval_date="2016-04-12")
print(f"Wrong date:       {parsed_wrong.verify()}")
```

#### DateTolerance

Pass if the extracted date is within the specified tolerance of the ground truth date.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tolerance` | `int` | required | Maximum allowed difference |
| `unit` | `Literal["days", "hours", "minutes"]` | `"days"` | Time unit: `"days"`, `"hours"`, or `"minutes"` |

**Applies to:** `str`

```python
class Answer(BaseAnswer):
    approval_date: str = VerifiedField(
        description="FDA approval date",
        ground_truth="2016-04-11",
        verify_with=DateTolerance(tolerance=30, unit="days"),
    )


parsed = Answer(approval_date="2016-04-25")
print(f"Within 30 days: {parsed.verify()}")

parsed_far = Answer(approval_date="2016-06-15")
print(f"Beyond 30 days: {parsed_far.verify()}")
```

#### DateRange

Pass if the extracted date falls within the specified bounds (inclusive). This primitive ignores `ground_truth`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min` | `str \| None` | `None` | Earliest acceptable date; no lower bound if `None` |
| `max` | `str \| None` | `None` | Latest acceptable date; no upper bound if `None` |

**Applies to:** `str`

```python
class Answer(BaseAnswer):
    submission_date: str = VerifiedField(
        description="NDA submission date",
        ground_truth="N/A",
        verify_with=DateRange(min="2015-01-01", max="2015-12-31"),
    )


parsed = Answer(submission_date="2015-06-15")
print(f"In range:     {parsed.verify()}")

parsed_out = Answer(submission_date="2016-02-01")
print(f"Out of range: {parsed_out.verify()}")
```

---

## 6. Trace Primitives

Trace primitives operate on the raw LLM response text, bypassing the judge entirely. Fields that use a trace primitive are **removed from the JSON schema** sent to the judge, so the judge never sees or attempts to extract them. The pipeline evaluates them directly after parsing completes.

### How Trace Primitives Use `ground_truth`

Trace fields must be typed as `bool`. The primitive's `check_trace()` method returns a boolean (pattern found, substring present, length within bounds), and the pipeline compares that result against `bool(ground_truth)`:

```
pass = primitive.check_trace(raw_response) == bool(ground_truth)
```

Setting `ground_truth=True` means "the check should succeed" (the pattern should be found, the substring should be present). Setting `ground_truth=False` inverts the logic: the field passes when the check does *not* succeed. This lets you test for both presence and absence using the same primitive.

### TraceRegex

Pass if the specified regex pattern is found in the raw response. When `count_min` is set, pass only if the pattern matches at least that many times.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pattern` | `str` | required | Regular expression to search for in the raw response |
| `count_min` | `int \| None` | `None` | Minimum number of matches required; any match passes if `None` |

```python
class Answer(BaseAnswer):
    cites_trial: bool = VerifiedField(
        description="Whether the response cites a clinical trial",
        ground_truth=True,
        verify_with=TraceRegex(pattern=r"NCT\d{8}"),
    )


# Trace primitives normally receive _raw_trace from the pipeline.
# Here we set it manually to demonstrate the primitive in isolation.
parsed = Answer(cites_trial=True)
parsed._raw_trace = "The MURANO trial (NCT02005471) demonstrated superior PFS."
print(f"Contains NCT ID: {parsed.verify()}")

parsed_no = Answer(cites_trial=True)
parsed_no._raw_trace = "The trial demonstrated superior PFS."
print(f"No NCT ID:       {parsed_no.verify()}")
```

### TraceContains

Pass if the specified substring appears in the raw response.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `substring` | `str` | required | Text to search for in the raw response |

```python
class Answer(BaseAnswer):
    mentions_limitations: bool = VerifiedField(
        description="Whether the response mentions study limitations",
        ground_truth=True,
        verify_with=TraceContains(substring="limitation"),
    )
```

**Testing for absence:** set `ground_truth=False` to verify that a pattern or substring does *not* appear. For example, to check that a response avoids a specific brand name:

```python
class Answer(BaseAnswer):
    avoids_brand_name: bool = VerifiedField(
        description="Whether the response avoids the brand name",
        ground_truth=False,
        verify_with=TraceContains(substring="Venclexta"),
    )


# TraceContains returns True if "Venclexta" is found.
# Since ground_truth=False, the pipeline compares True == False -> fail.
# If the substring is absent, the pipeline compares False == False -> pass.
parsed = Answer(avoids_brand_name=True)
parsed._raw_trace = "The selective BCL2 inhibitor was approved for CLL."
print(f"Brand name absent: {parsed.verify()}")

parsed_found = Answer(avoids_brand_name=True)
parsed_found._raw_trace = "Venclexta (venetoclax) was approved for CLL."
print(f"Brand name present: {parsed_found.verify()}")
```

### TraceLength

Pass if the raw response length falls within the specified bounds.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min` | `int \| None` | `None` | Minimum length; no lower bound if `None` |
| `max` | `int \| None` | `None` | Maximum length; no upper bound if `None` |
| `unit` | `str` | `"chars"` | Unit of measurement: `"chars"` or `"words"` |

```python
class Answer(BaseAnswer):
    is_substantive: bool = VerifiedField(
        description="Whether the response is substantive in length",
        ground_truth=True,
        verify_with=TraceLength(min=100, unit="words"),
    )
```

---

## 7. Normalizers

Normalizers preprocess string values before comparison. They are used by `ExactMatch`, `ContainsAny`, `ContainsAll`, and `OrderedMatch`. Normalizers are applied sequentially to both the extracted value and the expected value (or to each substring for `ContainsAny`/`ContainsAll`).

### Built-in Normalizers

| Normalizer | Description |
|-----------|-------------|
| `"lowercase"` | Convert to lowercase |
| `"strip"` | Remove leading and trailing whitespace |
| `"remove_punctuation"` | Remove all punctuation characters (`string.punctuation`) |
| `"collapse_whitespace"` | Replace runs of whitespace with a single space, then strip |
| `SynonymMap(mapping={...})` | Map known synonyms to canonical forms via exact key lookup |

The `Normalizer` type alias is `str | SynonymMap`. Normalizers are applied in the order they are listed.

### SynonymMap

`SynonymMap` performs exact key lookup: if the entire input string matches a key in the mapping, it is replaced with the corresponding value. If no key matches, the string passes through unchanged.

```python
class Answer(BaseAnswer):
    gene: str = VerifiedField(
        description="Gene name",
        ground_truth="BCL2",
        verify_with=ExactMatch(normalize=[
            "lowercase",
            "strip",
            SynonymMap(mapping={"bcl-2": "bcl2", "b-cell lymphoma 2": "bcl2"}),
        ]),
    )


for value in ["BCL2", "Bcl-2", "B-cell lymphoma 2", "KRAS"]:
    parsed = Answer(gene=value)
    print(f"{value!r:>24} -> verify(): {parsed.verify()}")
```

<div class="admonition tip">
<p class="admonition-title">Normalizer ordering matters</p>
<p><code>SynonymMap</code> uses exact key lookup against the entire input string. If your mapping keys are lowercase (e.g., <code>"bcl-2"</code>), place <code>"lowercase"</code> before the <code>SynonymMap</code> so that <code>"Bcl-2"</code> is first lowercased to <code>"bcl-2"</code> before the synonym lookup runs. The order of normalizers in the list determines the order of application.</p>
</div>

---

## 8. Writing Custom Primitives

When none of the 23 built-in primitives fit your verification need, you can write a custom one. Both base classes are in `karenina.schemas.primitives`.

### Custom Parsed Primitive

Subclass `VerificationPrimitive` and implement `check(extracted, expected) -> bool`:

```python
from typing import Any
from karenina.schemas.primitives import VerificationPrimitive


class CaseInsensitiveContains(VerificationPrimitive):
    """Pass if ground truth appears as a substring of the extracted value."""

    def check(self, extracted: Any, expected: Any) -> bool:
        return str(expected).lower() in str(extracted).lower()
```

### Custom Trace Primitive

Subclass `TracePrimitive` and implement `check_trace(raw_trace) -> bool`. Expected values are stored as constructor parameters on the primitive, not passed as arguments:

```python
from karenina.schemas.primitives import TracePrimitive


class TraceWordCount(TracePrimitive):
    """Pass if the response word count falls within bounds."""

    min_words: int = 0
    max_words: int = 10000

    def check_trace(self, raw_trace: str) -> bool:
        count = len(raw_trace.split())
        return self.min_words <= count <= self.max_words
```

### Registration

Custom primitives can be registered using the `@_register_primitive` decorator, which enables serialization and deserialization through the primitive registry. This is currently a private API intended as an internal extension point; its interface may change between releases. For most use cases, passing the primitive instance directly to `VerifiedField(verify_with=...)` works without registration.

---

## 9. Next Steps

- [Answer Templates](answer-templates.md): how primitives fit into the template lifecycle, field patterns, and `VerifiedField` parameters
- [Rubrics](rubrics/index.md): quality evaluation without ground truth (the complement to primitives)
- [Verification Pipeline](verification-pipeline.md): how the `verify_template` and `embedding_check` stages execute primitives
- [Templates vs Rubrics](template-vs-rubric.md): when to use primitives (correctness) vs rubric traits (quality)
- [Evaluation Modes](evaluation-modes.md): combining template verification and rubric evaluation
