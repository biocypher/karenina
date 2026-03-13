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

Verification primitives are pluggable verification strategies. They are passed to the `verify_with=` parameter of `VerifiedField` and determine how the field's extracted value is compared to its expected value (or how the raw LLM response is evaluated directly).

There are 18 primitives in two categories:

- **Parsed primitives** (15): Operate on values extracted by the judge LLM. The judge parses the response into a structured schema, and the primitive compares the extracted value to `ground_truth`.
- **Trace primitives** (3): Operate on the raw LLM response text before any parsing. Fields using trace primitives are excluded from the judge's schema.

Basic usage:

```python
from karenina.schemas.entities.answer import BaseAnswer, VerifiedField
from karenina.schemas.entities.primitives import ExactMatch

class MyTemplate(BaseAnswer):
    target: str = VerifiedField(
        description="Drug target name",
        ground_truth="BCL2",
        verify_with=ExactMatch(normalize=["lowercase", "strip"]),
    )
```

---

## 1. Parsed Primitives

Parsed primitives receive two values: the judge-extracted field value and the `ground_truth` set on the field. They return a boolean indicating whether verification passed.

### 1.1. Boolean

#### BooleanMatch

Compare the extracted boolean to the ground truth boolean.

No parameters.

**Applies to:** `bool`

```python
is_approved: bool = VerifiedField(
    description="Whether the drug is FDA-approved",
    ground_truth=True,
    verify_with=BooleanMatch(),
)
```

---

### 1.2. String

#### ExactMatch

Normalize both values, then compare for string equality.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `normalize` | `list[Normalizer]` | `["lowercase", "strip"]` | Normalizers applied before comparison |

**Applies to:** `str`, `int`, `float`

```python
target: str = VerifiedField(
    description="Protein target name",
    ground_truth="BCL2",
    verify_with=ExactMatch(normalize=["lowercase", "strip"]),
)
```

#### ContainsAny

Pass if the extracted text contains at least one of the specified substrings.

This primitive ignores `ground_truth`. The expected values are supplied via the `substrings` parameter.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `substrings` | `list[str]` | required | At least one must appear in the extracted value |
| `normalize` | `list[Normalizer]` | `[]` | Normalizers applied before comparison |

**Applies to:** `str`

```python
mechanism: str = VerifiedField(
    description="Mechanism of action",
    verify_with=ContainsAny(substrings=["apoptosis", "autophagy"]),
)
```

#### ContainsAll

Pass if the extracted text contains all of the specified substrings.

This primitive ignores `ground_truth`. The expected values are supplied via the `substrings` parameter.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `substrings` | `list[str]` | required | All must appear in the extracted value |
| `normalize` | `list[Normalizer]` | `[]` | Normalizers applied before comparison |

**Applies to:** `str`

```python
summary: str = VerifiedField(
    description="Trial summary",
    verify_with=ContainsAll(substrings=["phase III", "randomized", "double-blind"]),
)
```

#### RegexMatch

Pass if the extracted text matches the specified regex pattern.

This primitive ignores `ground_truth`. The expected pattern is supplied via the `pattern` parameter.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pattern` | `str` | required | Regular expression to match against the extracted value |
| `flags` | `list[str]` | `[]` | Regex flags (e.g., `["IGNORECASE"]`) |

**Applies to:** `str`

```python
identifier: str = VerifiedField(
    description="ClinicalTrials.gov identifier",
    verify_with=RegexMatch(pattern=r"NCT\d{8}"),
)
```

#### SemanticMatch

Pass if the embedding similarity between the extracted value and `ground_truth` meets the threshold.

This primitive requires embedding infrastructure. Use it together with the `embedding_check` pipeline stage rather than standalone verification.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | `float` | `0.85` | Minimum cosine similarity score to pass |

**Applies to:** `str`

```python
rationale: str = VerifiedField(
    description="Clinical rationale",
    ground_truth="Targets the BCL2 anti-apoptotic protein",
    verify_with=SemanticMatch(threshold=0.80),
)
```

---

### 1.3. Numeric

#### NumericExact

Pass if the extracted value equals the ground truth after float coercion.

No parameters.

**Applies to:** `int`, `float`

```python
patient_count: int = VerifiedField(
    description="Number of patients enrolled",
    ground_truth=342,
    verify_with=NumericExact(),
)
```

#### NumericTolerance

Pass if the extracted value is within a specified tolerance of the ground truth.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tolerance` | `float` | required | Allowed deviation from ground truth |
| `mode` | `str` | `"relative"` | Tolerance mode: `"relative"` (fraction of ground truth) or `"absolute"` (raw difference) |

**Applies to:** `int`, `float`

```python
hazard_ratio: float = VerifiedField(
    description="Hazard ratio from primary analysis",
    ground_truth=0.72,
    verify_with=NumericTolerance(tolerance=0.05, mode="absolute"),
)
```

#### NumericRange

Pass if the extracted value falls within the specified bounds.

This primitive ignores `ground_truth`. The expected bounds are supplied via `min` and `max`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min` | `float \| None` | `None` | Lower bound (inclusive); no lower bound if `None` |
| `max` | `float \| None` | `None` | Upper bound (inclusive); no upper bound if `None` |

**Applies to:** `int`, `float`

```python
p_value: float = VerifiedField(
    description="Primary endpoint p-value",
    verify_with=NumericRange(min=0.0, max=0.05),
)
```

---

### 1.4. List

#### SetContainment

Pass based on set-relationship comparison between the extracted list and the ground truth list.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | `str` | `"exact"` | Comparison mode: `"exact"`, `"subset"`, `"superset"`, or `"overlap"` |
| `min_overlap` | `int \| None` | `None` | Minimum number of shared elements; only applicable in `"overlap"` mode |

**Applies to:** `list[str]`

```python
indications: list[str] = VerifiedField(
    description="Approved indications",
    ground_truth=["CLL", "SLL", "NHL"],
    verify_with=SetContainment(mode="subset"),
)
```

#### OrderedMatch

Pass if each element of the extracted list matches the corresponding element of the ground truth list after normalization.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `normalize` | `list[Normalizer]` | `["lowercase", "strip"]` | Normalizers applied to each element before comparison |

**Applies to:** `list[str]`

```python
authors: list[str] = VerifiedField(
    description="Author names in order",
    ground_truth=["Smith J", "Jones A", "Patel R"],
    verify_with=OrderedMatch(normalize=["lowercase", "strip"]),
)
```

---

### 1.5. Categorical

#### LiteralMatch

Pass if the extracted value exactly matches the ground truth. Designed for fields typed as `Literal[...]`.

No parameters.

**Applies to:** `Literal[...]`

```python
from typing import Literal

trial_phase: Literal["I", "II", "III", "IV"] = VerifiedField(
    description="Clinical trial phase",
    ground_truth="III",
    verify_with=LiteralMatch(),
)
```

---

### 1.6. Date and Time

#### DateMatch

Parse both values as dates and compare for equality.

When `format` is `None`, python-dateutil is used for flexible date parsing.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `format` | `str \| None` | `None` | strptime format string; uses python-dateutil if `None` |

**Applies to:** `date`, `str`

```python
approval_date: str = VerifiedField(
    description="FDA approval date",
    ground_truth="2016-04-11",
    verify_with=DateMatch(),
)
```

#### DateTolerance

Pass if the extracted date is within the specified tolerance of the ground truth date.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tolerance` | `int` | required | Maximum allowed difference |
| `unit` | `str` | `"days"` | Time unit: `"days"`, `"hours"`, or `"minutes"` |

**Applies to:** `date`, `str`

```python
approval_date: str = VerifiedField(
    description="FDA approval date",
    ground_truth="2016-04-11",
    verify_with=DateTolerance(tolerance=30, unit="days"),
)
```

#### DateRange

Pass if the extracted date falls within the specified bounds.

This primitive ignores `ground_truth`. The expected bounds are supplied via `min` and `max`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min` | `str \| None` | `None` | Earliest acceptable date (inclusive); no lower bound if `None` |
| `max` | `str \| None` | `None` | Latest acceptable date (inclusive); no upper bound if `None` |

**Applies to:** `date`, `str`

```python
submission_date: str = VerifiedField(
    description="NDA submission date",
    verify_with=DateRange(min="2015-01-01", max="2015-12-31"),
)
```

---

## 2. Trace Primitives

Fields using trace primitives are excluded from the judge's parsing schema. The field type must be `bool` because the primitive evaluates the raw response directly rather than a judge-extracted value.

Trace primitives receive the full raw LLM response text and return a boolean. The expected values are embedded in constructor parameters, not in `ground_truth`.

### TraceRegex

Pass if the specified regex pattern is found in the raw response. When `count_min` is set, pass if the pattern matches at least that many times.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pattern` | `str` | required | Regular expression to search for in the raw response |
| `count_min` | `int \| None` | `None` | Minimum number of matches required; any match passes if `None` |

```python
cites_trial: bool = VerifiedField(
    description="Whether the response cites a clinical trial",
    verify_with=TraceRegex(pattern=r"NCT\d{8}"),
)
```

### TraceContains

Pass if the specified substring appears in the raw response.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `substring` | `str` | required | Text to search for in the raw response |

```python
mentions_limitations: bool = VerifiedField(
    description="Whether the response mentions study limitations",
    verify_with=TraceContains(substring="limitation"),
)
```

### TraceLength

Pass if the raw response length falls within the specified bounds.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min` | `int \| None` | `None` | Minimum length; no lower bound if `None` |
| `max` | `int \| None` | `None` | Maximum length; no upper bound if `None` |
| `unit` | `str` | `"chars"` | Unit of measurement: `"chars"` or `"words"` |

```python
is_substantive: bool = VerifiedField(
    description="Whether the response is substantive in length",
    verify_with=TraceLength(min=100, unit="words"),
)
```

---

## 3. Normalizers

Normalizers preprocess string values before comparison. They are applied sequentially to both the extracted value and the expected value.

| Normalizer | Type | Description |
|-----------|------|-------------|
| `"lowercase"` | `str` | Convert to lowercase |
| `"strip"` | `str` | Strip leading and trailing whitespace |
| `"remove_punctuation"` | `str` | Remove all punctuation characters |
| `"collapse_whitespace"` | `str` | Replace runs of whitespace with a single space, then strip |
| `SynonymMap(mapping={...})` | model | Map known synonyms to canonical forms before comparison |

The `Normalizer` type alias is `str | SynonymMap`. Normalizers are applied in the order they are listed.

```python
from karenina.schemas.entities.normalizers import SynonymMap
from karenina.schemas.entities.primitives import ExactMatch
from karenina.schemas.entities.answer import BaseAnswer, VerifiedField

class GeneTemplate(BaseAnswer):
    gene: str = VerifiedField(
        description="Gene name",
        ground_truth="BCL2",
        verify_with=ExactMatch(normalize=[
            SynonymMap(mapping={"bcl2": "BCL2", "bcl-2": "BCL2"}),
            "lowercase",
            "strip",
        ]),
    )
```

---

## 4. Writing Custom Primitives

Both primitive base classes are in `karenina.schemas.entities.primitives`.

For a **parsed primitive**, subclass `VerificationPrimitive` and implement `check`:

```python
from typing import Any
from karenina.schemas.entities.primitives import VerificationPrimitive

class MyParsedPrimitive(VerificationPrimitive):
    my_param: str

    def check(self, extracted: Any, expected: Any) -> bool:
        # extracted: the judge-extracted field value
        # expected: the ground_truth set on the field
        return self.my_param in str(extracted)
```

For a **trace primitive**, subclass `TracePrimitive` and implement `check_trace`. The expected values are embedded in the primitive's own constructor parameters, not passed as arguments to `check_trace`.

```python
from karenina.schemas.entities.primitives import TracePrimitive

class MyTracePrimitive(TracePrimitive):
    keyword: str

    def check_trace(self, raw_trace: str) -> bool:
        return self.keyword.lower() in raw_trace.lower()
```

Custom primitives can be registered using the `@_register_primitive` decorator. This is currently a private API and is intended as an advanced, internal extension point. Its interface may change between releases.
