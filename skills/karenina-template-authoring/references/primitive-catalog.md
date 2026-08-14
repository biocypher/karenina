# Verification Primitive Catalog

All primitives are importable from `karenina.schemas.primitives`:

```python
from karenina.schemas.primitives import BooleanMatch, ExactMatch, ...
```

They are also re-exported from `karenina.schemas.entities`:

```python
from karenina.schemas.entities import BooleanMatch, ExactMatch, ...
```

---

## Parsed Primitives

Parsed primitives operate on values extracted by the judge LLM. The field is
included in the judge's parsing schema.

### BooleanMatch

Compares extracted bool to ground truth bool. Both values are coerced to `bool`
before comparison.

- **Field type**: `bool`
- **Constructor**: `BooleanMatch()` (no parameters)
- **ground_truth**: `True` or `False`

```python
identifies_target: bool = VerifiedField(
    description="True if the response identifies the correct drug target.",
    ground_truth=True,
    verify_with=BooleanMatch(),
)
```

**Common mistakes**: Using BooleanMatch with a `str` field. The field annotation
must be `bool`.

---

### ExactMatch

Normalizes both strings then compares for equality. Default normalization:
lowercase + strip whitespace.

- **Field type**: `str`
- **Constructor**: `ExactMatch(normalize=["lowercase", "strip"])`
- **Parameters**:
  - `normalize`: list of normalizer names. Options: `"lowercase"`, `"strip"`,
    `"remove_punctuation"`, `"collapse_whitespace"`, or a `SynonymMap` instance.
- **ground_truth**: `str`

```python
drug_name: str = VerifiedField(
    description="The exact name of the drug mentioned in the response.",
    ground_truth="aspirin",
    verify_with=ExactMatch(),
)
```

**Common mistakes**: Using ExactMatch for free-text or multi-word answers. The
judge LLM rarely produces the exact same phrasing. Use BooleanMatch instead for
anything beyond single-word or controlled-vocabulary extraction.

---

### ContainsAny

Checks that extracted text contains at least one of the given substrings.
Ground truth is ignored; the substrings are provided in the constructor.

- **Field type**: `str`
- **Constructor**: `ContainsAny(substrings=["...", "..."])`
- **Parameters**:
  - `substrings`: list of substrings to search for.
  - `normalize`: optional list of normalizer names applied to both the extracted
    text and each substring before checking.
- **ground_truth**: any value (not used by the check, but required by VerifiedField)

```python
mentions_pathway: str = VerifiedField(
    description="The signaling pathway discussed in the response.",
    ground_truth="MAPK",
    verify_with=ContainsAny(substrings=["MAPK", "MAP kinase", "Ras-Raf-MEK-ERK"]),
)
```

**Common mistakes**: Using ContainsAny for multi-item extraction. If you need to
check that the response mentions multiple items, use `list[str]` with
`SetContainment` instead.

---

### ContainsAll

Checks that extracted text contains all of the given substrings.

- **Field type**: `str`
- **Constructor**: `ContainsAll(substrings=["...", "..."])`
- **Parameters**:
  - `substrings`: list of substrings that must all be present.
  - `normalize`: optional list of normalizer names.
- **ground_truth**: any value (not used by the check)

```python
complete_description: str = VerifiedField(
    description="A description of the drug mechanism.",
    ground_truth="inhibits COX-2",
    verify_with=ContainsAll(substrings=["inhibits", "COX-2"]),
)
```

---

### RegexMatch

Checks that extracted text matches a regex pattern.

- **Field type**: `str`
- **Constructor**: `RegexMatch(pattern="...", flags=[])`
- **Parameters**:
  - `pattern`: regex pattern string.
  - `flags`: list of `re` flag names as strings (e.g., `["IGNORECASE"]`).
- **ground_truth**: any value (not used by the check)

```python
gene_symbol: str = VerifiedField(
    description="The gene symbol extracted from the response.",
    ground_truth="BRCA1",
    verify_with=RegexMatch(pattern=r"^BRCA[12]$"),
)
```

---

### SemanticMatch

Checks embedding similarity between extracted and expected text. Requires an
embedding model configured at runtime. The check itself is delegated to the
`embedding_check` pipeline stage.

- **Field type**: `str`
- **Constructor**: `SemanticMatch(threshold=0.85)`
- **Parameters**:
  - `threshold`: minimum cosine similarity (default 0.85).
- **ground_truth**: `str` (the reference text to compare against)

```python
summary: str = VerifiedField(
    description="A summary of the drug's mechanism of action.",
    ground_truth="Aspirin irreversibly inhibits cyclooxygenase enzymes.",
    verify_with=SemanticMatch(threshold=0.80),
)
```

**Common mistakes**: Using SemanticMatch without configuring an embedding model
in the pipeline. The `check()` method raises `NotImplementedError` if called
directly; it must be evaluated via the embedding_check pipeline stage.

---

### LiteralMatch

Exact equality for `Literal`-typed fields. No normalization.

- **Field type**: `str` (typically `Literal["a", "b", "c"]`)
- **Constructor**: `LiteralMatch()` (no parameters)
- **ground_truth**: one of the literal values

```python
from typing import Literal

severity: Literal["low", "medium", "high"] = VerifiedField(
    description="The severity level: must be one of 'low', 'medium', or 'high'.",
    ground_truth="high",
    verify_with=LiteralMatch(),
)
```

---

### NumericExact

Exact numeric equality after float coercion.

- **Field type**: `int` or `float`
- **Constructor**: `NumericExact()` (no parameters)
- **ground_truth**: `int` or `float`

```python
patient_count: int = VerifiedField(
    description="The number of patients in the study.",
    ground_truth=150,
    verify_with=NumericExact(),
)
```

**Common mistakes**: Using NumericExact for values where minor rounding
differences are expected. Use NumericTolerance instead.

---

### NumericTolerance

Checks that extracted number is within tolerance of expected.

- **Field type**: `int` or `float`
- **Constructor**: `NumericTolerance(tolerance=..., mode="relative")`
- **Parameters**:
  - `tolerance`: the tolerance value.
  - `mode`: `"relative"` (default) or `"absolute"`.
    - Relative: `|extracted - expected| / |expected| <= tolerance`
    - Absolute: `|extracted - expected| <= tolerance`
- **ground_truth**: `int` or `float`

```python
efficacy_rate: float = VerifiedField(
    description="The reported efficacy rate as a percentage (e.g. 85.5).",
    ground_truth=85.5,
    verify_with=NumericTolerance(tolerance=0.05, mode="relative"),
)
```

---

### NumericGraded

Scores an extracted number by its distance from the expected value, giving
partial credit that decays to zero at a cutoff. This is the one numeric
primitive whose contribution to `verify_granular()` is continuous (partial
credit) rather than 0/1. `verify()` stays binary, driven by `check()`. Reach for
it when a near-miss is meaningfully better than a far-off answer (continuous
measurements: means, ratios, p-values, percentages); keep `NumericExact` for
counts where only one value is correct.

- **Field type**: `int` or `float`
- **Constructor**: `NumericGraded(cutoff=..., mode="relative")` (single-band) or
  `NumericGraded(cutoff=..., full_credit=..., mode="absolute")` (double-band)
- **Parameters**:
  - `cutoff`: distance at which credit reaches 0 (must be `> 0`).
  - `full_credit`: optional inner band earning full credit; when set,
    `0 <= full_credit < cutoff`.
  - `mode`: `"relative"` (default, fractional distance) or `"absolute"` (raw
    units, i.e. percentage-points when the reference is a percentage).
  - `decay`: `"linear"` (default) or `"quadratic"`.
- **ground_truth**: `int` or `float`
- **Bands**:
  - Single-band (no `full_credit`): `check()` passes within `cutoff`; the score
    decays from 1.0 at the reference to 0.0 at the cutoff.
  - Double-band (`full_credit` set): `check()` gates at the inner band (so the
    binary pass stays tight), the score is 1.0 within `full_credit` and decays to
    0.0 at the cutoff. A near-miss between the two bands is `verify()` False but
    still earns partial credit.
- **Per-field detail**: graded scores are surfaced in `field_scores` on the
  result and the `field_score` column of the template DataFrame.

```python
mean_distance: float = VerifiedField(
    description="The reported mean patristic distance (e.g. 1.67).",
    ground_truth=1.67,
    verify_with=NumericGraded(cutoff=0.25, mode="relative"),
)
```

---

### NumericRange

Checks that extracted number falls within a range. Either bound can be `None`
for open-ended ranges. Boundaries are inclusive by default.

- **Field type**: `int` or `float`
- **Constructor**: `NumericRange(min=..., max=...)`
- **Parameters**:
  - `min`: lower bound (inclusive by default, or `None`).
  - `max`: upper bound (inclusive by default, or `None`).
  - `exclusive_min`: if `True`, use strict `>` for the lower bound.
  - `exclusive_max`: if `True`, use strict `<` for the upper bound.
- **ground_truth**: any value (not used by the check)

```python
sample_size: int = VerifiedField(
    description="The sample size reported in the study.",
    ground_truth=100,
    verify_with=NumericRange(min=50, max=500),
)
```

### NumericRangeGraded

The graded companion to `NumericRange`: an acceptance band `[min, max]` with soft
shoulders. `check()` passes only inside the band (the same hard gate as
`NumericRange`), while `score()` gives decaying partial credit to values just
outside, out to a margin. Reach for it when any value inside an interval is
equally correct but a near-miss outside still deserves credit. When a single
reference point exists, prefer `NumericGraded` centered on that point instead.

- **Field type**: `int` or `float`
- **Constructor**: `NumericRangeGraded(min=..., max=..., margin=...)`
- **Parameters**:
  - `min`, `max`: the full-credit band (must satisfy `min < max`).
  - `margin`: shoulder width (must be `> 0`).
  - `mode`: `"absolute"` (default, raw units) or `"relative"` (a fraction of the
    band width `max - min`).
  - `decay`: `"linear"` (default) or `"quadratic"`.
  - `exclusive_min`, `exclusive_max`: strict inequalities at the band edges.
- **ground_truth**: any value (not used by the check or score)
- **Per-field detail**: graded scores are surfaced in `field_scores` and the
  `field_score` DataFrame column.

```python
spearman_corr: float = VerifiedField(
    description="The replicate Spearman correlation, expected to be weak.",
    ground_truth=0,
    verify_with=NumericRangeGraded(min=0.001, max=0.09, margin=0.02, mode="absolute"),
)
```

### NumericMinimum

Checks that extracted number is at least the ground truth value. Inclusive
by default.

- **Field type**: `int` or `float`
- **Constructor**: `NumericMinimum()`
- **Parameters**:
  - `exclusive`: if `True`, use strict `>` instead of `>=`. Default: `False`.
- **ground_truth**: the minimum threshold

```python
citation_count: int = VerifiedField(
    description="Number of citations in the response.",
    ground_truth=5,
    verify_with=NumericMinimum(),
)
```

### NumericMaximum

Checks that extracted number does not exceed the ground truth value. Inclusive
by default.

- **Field type**: `int` or `float`
- **Constructor**: `NumericMaximum()`
- **Parameters**:
  - `exclusive`: if `True`, use strict `<` instead of `<=`. Default: `False`.
- **ground_truth**: the maximum threshold

```python
error_count: int = VerifiedField(
    description="Number of factual errors identified.",
    ground_truth=3,
    verify_with=NumericMaximum(),
)
```

### NumericThresholdGraded

The graded companion to `NumericMinimum` and `NumericMaximum`: a one-sided bound
with a soft shoulder. The threshold is the `ground_truth` (same convention as the
plain bounds). Any value on the correct side scores full credit, however far,
while values just past the bound earn decaying partial credit. Reach for it when
the question asks a value to be below or above a threshold and being deep on the
correct side is fully correct.

- **Field type**: `int` or `float`
- **Constructor**: `NumericThresholdGraded(direction="max", margin=...)`
- **Parameters**:
  - `direction`: `"max"` (pass iff `<= threshold`) or `"min"` (pass iff `>= threshold`).
  - `margin`: shoulder width past the threshold (must be `> 0`).
  - `mode`: `"relative"` (default, a fraction of `|threshold|`) or `"absolute"`.
  - `decay`: `"linear"` (default) or `"quadratic"`.
  - `exclusive`: if `True`, the binary gate uses a strict inequality.
- **ground_truth**: the threshold value
- **Per-field detail**: graded scores are surfaced in `field_scores` and the
  `field_score` DataFrame column.

```python
adj_p_value: float = VerifiedField(
    description="Adjusted p-value, expected to clear the 1e-30 significance bound.",
    ground_truth=1e-30,
    verify_with=NumericThresholdGraded(direction="max", margin=1.0, mode="relative"),
)
```

---

### SetContainment

Compares lists as sets with configurable containment mode.

- **Field type**: `list[str]`, `list[int]`, or similar
- **Constructor**: `SetContainment(mode="exact")`
- **Parameters**:
  - `mode`: `"exact"` (default), `"subset"`, `"superset"`, or `"overlap"`.
    - `"exact"`: extracted and expected contain the same elements.
    - `"subset"`: extracted is a subset of expected.
    - `"superset"`: extracted is a superset of expected.
    - `"overlap"`: at least `min_overlap` elements in common.
  - `min_overlap`: required for `"overlap"` mode.
- **ground_truth**: `list` of the appropriate type

```python
adverse_events: list[str] = VerifiedField(
    description="List of adverse events mentioned in the response.",
    ground_truth=["nausea", "headache", "fatigue"],
    verify_with=SetContainment(mode="exact"),
)
```

**Common mistakes**: Using `mode="exact"` when the response might include
additional valid items. Use `mode="superset"` if the extracted list should
contain at least the ground truth items but may include extras.

---

### OrderedMatch

Compares lists element-by-element after normalization. Both lists must be the
same length and each corresponding pair must match.

- **Field type**: `list[str]`
- **Constructor**: `OrderedMatch(normalize=["lowercase", "strip"])`
- **Parameters**:
  - `normalize`: list of normalizer names.
- **ground_truth**: `list[str]`

```python
treatment_steps: list[str] = VerifiedField(
    description="The treatment steps in the correct order.",
    ground_truth=["diagnosis", "medication", "follow-up"],
    verify_with=OrderedMatch(),
)
```

---

### DateMatch

Parses and compares dates. Uses python-dateutil for flexible parsing when no
format is specified.

- **Field type**: `str`
- **Constructor**: `DateMatch(format=None)`
- **Parameters**:
  - `format`: optional strftime format string for strict parsing.
- **ground_truth**: `str` (a date string)

```python
approval_date: str = VerifiedField(
    description="The FDA approval date of the drug (any common date format).",
    ground_truth="2023-03-15",
    verify_with=DateMatch(),
)
```

---

### DateTolerance

Checks that extracted date is within tolerance of expected date.

- **Field type**: `str`
- **Constructor**: `DateTolerance(tolerance=..., unit="days")`
- **Parameters**:
  - `tolerance`: integer tolerance value.
  - `unit`: `"days"` (default), `"hours"`, or `"minutes"`.
- **ground_truth**: `str` (a date string)

```python
publication_date: str = VerifiedField(
    description="The publication date of the study.",
    ground_truth="2024-01-15",
    verify_with=DateTolerance(tolerance=30, unit="days"),
)
```

---

### DateRange

Checks that extracted date falls within a range. Either bound can be `None`.

- **Field type**: `str`
- **Constructor**: `DateRange(min=None, max=None)`
- **Parameters**:
  - `min`: earliest acceptable date string (or `None`).
  - `max`: latest acceptable date string (or `None`).
- **ground_truth**: any value (not used by the check)

```python
trial_date: str = VerifiedField(
    description="The date the clinical trial began.",
    ground_truth="2023-06-01",
    verify_with=DateRange(min="2023-01-01", max="2024-12-31"),
)
```

---

## Trace Primitives

Trace primitives operate on the raw LLM response text rather than
judge-extracted values. Fields using a TracePrimitive are excluded from the
judge's parsing schema. The pipeline evaluates them directly.

Trace fields are still required Pydantic fields: instantiation needs a
placeholder value, and `verify()` raises `ValueError` unless the raw response
text is attached via `answer.__dict__["_raw_trace"]`. See
`field-type-guide.md` ("Testing Templates Offline") for a working recipe.

### TraceRegex

Checks for a regex pattern in the raw LLM response.

- **Field type**: `bool`
- **Constructor**: `TraceRegex(pattern="...", count_min=None)`
- **Parameters**:
  - `pattern`: regex pattern string.
  - `count_min`: if set, requires at least this many matches.
- **ground_truth**: `True` or `False`

```python
cites_references: bool = VerifiedField(
    description="True if the response contains citation-like patterns.",
    ground_truth=True,
    verify_with=TraceRegex(pattern=r"\[\d+\]"),
)
```

---

### TraceContains

Checks for a substring in the raw LLM response.

- **Field type**: `bool`
- **Constructor**: `TraceContains(substring="...")`
- **Parameters**:
  - `substring`: the exact substring to search for.
- **ground_truth**: `True` or `False`

```python
mentions_disclaimer: bool = VerifiedField(
    description="True if the response contains a medical disclaimer.",
    ground_truth=True,
    verify_with=TraceContains(substring="consult your doctor"),
)
```

---

### TraceLength

Checks the length of the raw LLM response.

- **Field type**: `bool`
- **Constructor**: `TraceLength(min=None, max=None, unit="chars")`
- **Parameters**:
  - `min`: minimum length (or `None`).
  - `max`: maximum length (or `None`).
  - `unit`: `"chars"` (default) or `"words"`.
- **ground_truth**: `True` or `False`

```python
sufficient_length: bool = VerifiedField(
    description="True if the response is at least 100 words.",
    ground_truth=True,
    verify_with=TraceLength(min=100, unit="words"),
)
```

---

## Composition Nodes

Composition nodes combine multiple field results into a single verify() result.
Define them in a `VerificationStrategy` inner class on your template.

### AllOf

All referenced fields must pass.

```python
from karenina.schemas.entities import AllOf, FieldCheck

class Answer(BaseAnswer):
    # ... fields ...

    class VerificationStrategy:
        verify_strategy = AllOf(conditions=[
            FieldCheck(field="field_a"),
            FieldCheck(field="field_b"),
        ])
```

### AnyOf

At least one referenced field must pass.

```python
from karenina.schemas.entities import AnyOf, FieldCheck

class VerificationStrategy:
    verify_strategy = AnyOf(conditions=[
        FieldCheck(field="field_a"),
        FieldCheck(field="field_b"),
    ])
```

### AtLeastN

N or more referenced fields must pass.

```python
from karenina.schemas.entities import AtLeastN, FieldCheck

class VerificationStrategy:
    verify_strategy = AtLeastN(n=2, conditions=[
        FieldCheck(field="field_a"),
        FieldCheck(field="field_b"),
        FieldCheck(field="field_c"),
    ])
```

Nodes are recursive: you can nest `AllOf` inside `AnyOf`, etc.

---

## Normalizers

Several parsed primitives accept a `normalize` parameter. Available normalizer
names:

| Name | Effect |
|------|--------|
| `"lowercase"` | Converts to lowercase |
| `"strip"` | Strips leading/trailing whitespace |
| `"remove_punctuation"` | Removes all punctuation characters |
| `"collapse_whitespace"` | Collapses runs of whitespace to a single space |

You can also use a `SynonymMap` instance to map known synonyms to canonical
forms:

```python
from karenina.schemas.primitives import SynonymMap

drug_name: str = VerifiedField(
    description="The drug name mentioned.",
    ground_truth="aspirin",
    verify_with=ExactMatch(normalize=[
        "lowercase",
        "strip",
        SynonymMap(mapping={"asa": "aspirin", "acetylsalicylic acid": "aspirin"}),
    ]),
)
```
