# Field Type Guide

## Python Field Types and Compatible Primitives

Each `VerifiedField` has a Python type annotation that determines which
verification primitives can be used with it.

### bool

Use for yes/no judgments. The judge LLM extracts `True` or `False`.

| Primitive | Notes |
|-----------|-------|
| `BooleanMatch()` | The standard choice. Coerces both values to bool. |

**Trace primitives** also use `bool` fields, but the field is excluded from the
judge's parsing schema:

| Primitive | Notes |
|-----------|-------|
| `TraceRegex(pattern="...")` | Regex on raw response |
| `TraceContains(substring="...")` | Substring check on raw response |
| `TraceLength(min=..., max=...)` | Length check on raw response |

### str

Use for extracting specific text values.

| Primitive | When to use |
|-----------|-------------|
| `ExactMatch()` | Single-word or controlled-vocabulary extraction |
| `ContainsAny(substrings=[...])` | Extracted text should mention at least one of several terms |
| `ContainsAll(substrings=[...])` | Extracted text must mention all of several terms |
| `RegexMatch(pattern="...")` | Extracted text must match a pattern |
| `SemanticMatch(threshold=0.85)` | Meaning-based comparison (requires embedding model) |
| `LiteralMatch()` | Exact equality, no normalization (for Literal types) |
| `DateMatch()` | Date comparison with flexible parsing |
| `DateTolerance(tolerance=..., unit="days")` | Date within tolerance |
| `DateRange(min="...", max="...")` | Date within range |

### int / float

Use for numeric extraction.

| Primitive | When to use |
|-----------|-------------|
| `NumericExact()` | Exact equality after float coercion |
| `NumericTolerance(tolerance=..., mode="relative")` | Within tolerance (relative or absolute) |
| `NumericRange(min=..., max=...)` | Within a range |

### list[str] / list[int]

Use for extracting multiple items.

| Primitive | When to use |
|-----------|-------------|
| `SetContainment(mode="exact")` | Same elements regardless of order |
| `SetContainment(mode="subset")` | Extracted is subset of expected |
| `SetContainment(mode="superset")` | Extracted contains all expected (and maybe more) |
| `SetContainment(mode="overlap", min_overlap=N)` | At least N items in common |
| `OrderedMatch()` | Same elements in the same order |

### Optional[str] / str | None

Nullable fields. The judge may extract `None` if the information is absent.
Use the same primitives as `str`, but be aware that `None` will fail most
string comparisons.

---

## Writing VerifiedField Descriptions

The `description` parameter is the most important part of a `VerifiedField`.
It goes directly into the JSON schema that the judge LLM reads when parsing
the response. The judge does not see the ground truth or the verification
primitive; the description is its sole instruction.

### The Four Elements

Every description must address four elements. The weight of each varies by
field type, but omitting any invites unpredictable extractions.

| Element | What It Does | Most Critical For |
|---------|-------------|-------------------|
| **What to extract** | Names the specific value, not just the topic. "The gene identified as most frequent," not "the gene mentioned." | All field types |
| **Format** | Tells the judge how to write the value: casing, notation, symbol conventions, one item per entry for lists. | String and list fields |
| **Scope** | Draws a boundary around what counts. Which mentions are in, which are out. For booleans, this defines the True/False threshold. | Boolean and list fields |
| **Disambiguation** | Provides a fallback rule for edge cases: ambiguous mentions, multiple candidates, negated references. | All field types |

### Rules

1. **Write as an instruction to the judge.** The judge reads the description
   to decide what to extract and (for bool fields) what True/False mean.

2. **Be explicit about True/False.** For bool fields, always state what True
   means AND what False means. Define the threshold; do not let the judge
   invent one:

   ```python
   # Bad: the judge must invent a threshold for "mentions"
   description="True if the response mentions a drug-drug interaction"

   # Good: threshold is explicit
   description=(
       "True if the response describes a drug interaction confirmed in "
       "human clinical trials or recognized in prescribing guidelines. "
       "False if the interaction is described as theoretical, observed only "
       "in vitro, or explicitly called clinically insignificant."
   )
   ```

3. **List all acceptable variations.** If the expected value has synonyms,
   abbreviations, or alternate spellings, enumerate them:

   ```python
   description=(
       "True if the response identifies BCL2 as the target. "
       "Acceptable forms: BCL2, Bcl-2, BCL-2, B-cell lymphoma 2."
   )
   ```

4. **Specify disambiguation rules.** Tell the judge how to handle ambiguous
   situations, multiple candidates, or missing information:

   ```python
   description=(
       "Extract the gene the response singles out as the most frequently "
       "mutated in human cancers. If the response names multiple genes "
       "without ranking frequency, extract the first gene mentioned."
   )
   ```

5. **Keep descriptions self-contained.** The judge sees only the JSON schema
   (field name, type, description). It does not see the ground truth or the
   primitive. Every piece of context the judge needs must be in the description.

6. **Avoid vague language.** Words like "appropriate", "relevant", or "correct"
   are ambiguous without context. Be concrete about what qualifies.

### Bad vs. Good Examples

**String field: specify the target, not just the topic.**

```python
# Bad: which gene? The response may mention several
description="The gene mentioned in the response"

# Good: names the specific value, format, scope, and disambiguation
description=(
    "The gene that the response identifies as the single most frequently "
    "mutated gene in human cancers. Extract the official HGNC gene symbol "
    "in uppercase (e.g., 'TP53', not 'p53' or 'tumor protein p53'). "
    "If the response names multiple genes without singling one out as the "
    "most frequent, extract the first gene mentioned."
)
```

**Boolean field: define the threshold, not just the question.**

```python
# Bad: vague, no True/False explanation
description="Is the drug target correct?"

# Good: specific, covers both outcomes
description=(
    "True if the response identifies EGFR (epidermal growth factor receptor) "
    "as the primary target of erlotinib. False if a different target is named "
    "or no target is specified."
)
```

**Numeric field: specify format and edge cases.**

```python
# Bad: documentation-style, no edge case handling
description="The number of patients enrolled in the study."

# Good: tells the judge what to extract, format, and fallback
description=(
    "The total number of patients enrolled in the study, as an integer. "
    "If a range is given (e.g., '100-150'), extract the lower bound. "
    "If no enrollment number is mentioned, return 0."
)
```

**List field: control what gets included.**

```python
# Bad: no inclusion/exclusion criteria
description="The symptoms mentioned in the response"

# Good: scope and disambiguation defined
description=(
    "List the symptoms the response attributes directly to the condition. "
    "Include only symptoms explicitly linked to the condition; exclude "
    "symptoms mentioned in differential diagnoses or as treatment side "
    "effects. If no symptoms are mentioned, return an empty list."
)
```

---

## Ground Truth Type Matching

The `ground_truth` value must be compatible with the field's Python type
annotation:

| Field annotation | ground_truth type | Example |
|-----------------|------------------|---------|
| `bool` | `bool` | `True`, `False` |
| `str` | `str` | `"aspirin"` |
| `int` | `int` | `42` |
| `float` | `float` | `3.14` |
| `list[str]` | `list[str]` | `["a", "b", "c"]` |
| `list[int]` | `list[int]` | `[1, 2, 3]` |
| `str \| None` | `str` or `None` | `"aspirin"` or `None` |
| `Literal["a", "b"]` | `str` (one of the literals) | `"a"` |

**What a mismatch actually does depends on the primitive:**

- Numeric primitives (`NumericExact`, `NumericTolerance`, `NumericMinimum`,
  `NumericMaximum`, ...) call `float()` on the ground truth. A non-coercible
  value such as `ground_truth="one point six"` makes `verify()` raise
  `ValueError` (`could not convert string to float: 'one point six'`). A
  coercible one such as `"1.6"` is quietly accepted.
- `BooleanMatch` coerces both sides with `bool()`, so a non-bool ground truth
  passes **silently and wrongly**: `ground_truth="dimensional"` becomes
  `True`, and the field passes whenever the judge extracts `True` — no error
  is ever raised.
- `ExactMatch` coerces both sides with `str()`, so a non-str ground truth is
  compared **silently** after stringification: `ground_truth=42` on a `str`
  field compares against `"42"`.

**Construction-time warning**: for obvious numeric/bool mismatches,
`VerifiedField` logs a `WARNING` through the logger
`karenina.schemas.entities.verified_field` when the template class is
defined, e.g. `ground_truth 'dimensional' may not match BooleanMatch:
expected a boolean or bool-like value (True/False, 0/1, 'yes'/'no')`. This is
a log line, not an exception — the template still constructs and runs. Watch
pipeline logs for it.

Because `BooleanMatch` and `ExactMatch` coerce silently, always smoke-verify
a new template offline: instantiate it with a known-good extraction and
assert `verify()` returns `True` before running the pipeline (recipe below).

**Note**: For primitives that ignore ground_truth (e.g., `ContainsAny`,
`ContainsAll`, `RegexMatch`, `NumericRange`, `DateRange`), you still must
provide a ground_truth value. Use a representative value of the correct type.

---

## Testing Templates Offline (Including Trace Fields)

Templates can be exercised without any LLM: instantiate the class with
known-good extracted values and call `verify()`. Trace fields
(`TraceRegex`, `TraceContains`, `TraceLength`) need two extra steps:
they are still required Pydantic fields, so instantiation needs a
placeholder value, and `verify()` raises
`ValueError: Field '...' uses a TracePrimitive but requires _raw_trace to be set`
until the raw response text is attached via `_raw_trace`:

```python
from karenina.schemas.entities import BaseAnswer, VerifiedField, TraceContains


class Answer(BaseAnswer):
    mentions_aspirin: bool = VerifiedField(
        # Placeholder description: trace fields are excluded from the judge's
        # parsing schema, so the judge never sees it.
        description="True if the response mentions aspirin.",
        ground_truth=True,
        verify_with=TraceContains(substring="aspirin"),
    )


# The bool field is required: instantiation needs a placeholder value
# (it is ignored during verification).
answer = Answer(mentions_aspirin=True)

# Attach the raw response text; without this line verify() raises ValueError.
answer.__dict__["_raw_trace"] = "The patient takes aspirin daily."

assert answer.verify() is True
```

---

## Weight Parameter

The `weight` parameter (default `1.0`) controls how much a field contributes
to the `verify_granular()` score.

```python
# This field counts 3x as much as a default field
critical_finding: bool = VerifiedField(
    description="True if the critical safety finding is reported.",
    ground_truth=True,
    verify_with=BooleanMatch(),
    weight=3.0,
)
```

The granular score is: `sum(weight * pass) / sum(weight)` across all fields.

`weight` does not affect `verify()` (binary pass/fail). To change which fields
are required for `verify()`, use a `VerificationStrategy` inner class with
composition nodes (`AllOf`, `AnyOf`, `AtLeastN`).

---

## extraction_hint Parameter

The `extraction_hint` parameter provides optional formatting guidance to the
judge LLM. It appears alongside the field in the schema.

```python
trial_date: str = VerifiedField(
    description="The date the clinical trial started.",
    ground_truth="2023-06-15",
    verify_with=DateMatch(),
    extraction_hint="Return the date in YYYY-MM-DD format.",
)
```

Use `extraction_hint` when:
- The field needs a specific format (dates, numbers with units).
- The judge might be confused about which value to extract among several
  candidates.
- You want to provide an example of the expected output format.

Do not duplicate information already in the `description`.

**Note on auto-generated templates**: The auto-generation pipeline no longer
produces `extraction_hint`; all guidance is embedded directly in the
`description` using the four-element framework (what to extract, format, scope,
disambiguation). `extraction_hint` remains available for manually authored
templates when you want supplementary formatting guidance separate from the main
description.
