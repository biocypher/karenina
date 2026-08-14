---
name: karenina-template-authoring
description: >
  Create answer templates (BaseAnswer subclasses with VerifiedField) for
  karenina LLM evaluation. Use when writing, fixing, or improving templates
  that define what to extract from LLM responses and how to verify correctness.
  Covers field types, verification primitives, description writing, and
  ground truth configuration. Invoke for any template-related work.
---

# Answer Template Authoring

## What This Skill Does

Templates are Pydantic schemas that tell the judge LLM what to extract from a
response and how to verify it. Each template is a `BaseAnswer` subclass where
every field you want evaluated uses `VerifiedField` with a verification
primitive and ground truth value.

## Interactive Procedure

Follow these six steps when creating a new template.

### Step 1: Understand the Domain

Ask the user:

> What are you evaluating? Describe the domain and what a correct answer
> looks like.

Use the response to determine the number of fields, their types, and which
verification primitives fit.

### Step 2: Identify Extraction Targets

Ask the user:

> What specific facts or values should be extracted from the response?

For each fact, determine: (a) is it a yes/no judgment, a specific string, a
number, a date, or a list of items? (b) what does "correct" look like?

### Step 3: Propose Template Structure

Present a table:

| Field name | Python type | Primitive | Ground truth |
|------------|------------|-----------|-------------|
| ... | ... | ... | ... |

Wait for the user to confirm or revise before writing code.

### Step 4: Generate the Template

Write the `BaseAnswer` subclass. Use the skeleton from `assets/template-skeleton.py`
as a starting point. Ensure every `VerifiedField.description` is written as an
instruction to the judge LLM (see Gotchas below).

### Step 5: Validate

Run the validation script on the generated template:

```bash
uv run skills/karenina-template-authoring/scripts/validate_template.py path/to/template.py
```

If validation fails: show the error, ask the user for clarification, regenerate
the affected fields, and re-validate.

### Step 6: User Confirmation

Present the final template and ask if any fields need adjustment. Confirm the
template is ready for use.

## Quick-Start Template

```python
"""Minimal karenina answer template skeleton."""
from karenina.schemas.entities import BaseAnswer, VerifiedField, BooleanMatch


class Answer(BaseAnswer):
    """Template for evaluating [YOUR DOMAIN]."""

    identifies_key_fact: bool = VerifiedField(
        description=(
            "True if the response identifies [SPECIFIC CONCEPT] "
            "(also acceptable: [SYNONYM1], [SYNONYM2]) as [SPECIFIC ROLE]. "
            "False if a different [ENTITY TYPE] is named or [CONCEPT] is "
            "not mentioned. If multiple [ENTITIES] are listed without "
            "ranking, return True only if [CONCEPT] is among them."
        ),
        ground_truth=True,
        verify_with=BooleanMatch(),
    )
```

## Gotchas

These are the most common mistakes when authoring templates. Read them all
before writing or reviewing any template.

### 1. Always Inherit from BaseAnswer

Templates MUST inherit from `BaseAnswer`, never from `BaseModel` or plain
`pydantic.BaseModel`. The pipeline discovers templates by type and the
auto-generated `verify()` / `verify_granular()` methods live on `BaseAnswer`.

```python
# Correct
from karenina.schemas.entities import BaseAnswer, VerifiedField

class Answer(BaseAnswer): ...

# Wrong: will not work in the pipeline
from pydantic import BaseModel

class Answer(BaseModel): ...
```

**`VerifiedField` import path**: `VerifiedField` lives in
`karenina.schemas.entities.verified_field`, NOT in `answer.py`. Import it from
the package:

```python
# Correct
from karenina.schemas.entities import BaseAnswer, VerifiedField

# Wrong: VerifiedField is not defined in answer.py
from karenina.schemas.entities.answer import VerifiedField  # ImportError
```

### 2. Write Descriptions for the Judge, Not for Documentation

`VerifiedField.description` is the judge LLM's sole instruction for what to
extract and how. The judge sees each field's name, type, and description in
a JSON schema. For Literal types, it also sees an `enum` of allowed values.
It does NOT see ground truth or the verification primitive.

Every description must address **four elements**:

1. **What to extract**: the specific value or judgment, not just the topic
2. **Format**: casing, notation, conventions (for str/list; booleans skip this)
3. **Scope**: what counts and what does not; for booleans, the True/False threshold
4. **Disambiguation**: fallback rule for edge cases, multiple candidates, or unclear responses

**Write descriptions aware of how each type is verified**:

- `bool` + BooleanMatch: define a clear True/False threshold. The judge answers
  true/false; semantic equivalence is fine.
- `str` + ExactMatch (lowercase + strip): the extracted string is compared
  literally after lowercasing. Include exact format guidance in the description
  (e.g., "Extract the HGNC gene symbol in uppercase").
- `list[str]` + SetContainment (exact mode): the judge must extract ALL
  qualifying items and ONLY qualifying items. Specify inclusion/exclusion criteria.
- `Literal[...]` + LiteralMatch: the judge already sees the enum options. Focus
  the description on which option to select and how to disambiguate.
- `int` + NumericExact / `float` + NumericTolerance: guide toward a specific
  number; for floats, minor rounding is tolerated. For graded partial credit in
  `verify_granular()` while `verify()` stays binary, use `NumericGraded` (distance
  from a point), `NumericRangeGraded` (distance outside an acceptance band), or
  `NumericThresholdGraded` (distance past a one-sided bound).

**Field naming matters**: names appear in the JSON schema alongside the
description. Choose names that communicate scope and intent:
`identifies_kras_as_most_frequent` (clear) vs `kras_mutation` (ambiguous).

Always mention acceptable variations of the expected value (abbreviations,
alternate spellings, synonyms).

**Good examples** (real generated templates showing the four elements in action):

Q: "Is fulvestrant withdrawn?" A: "No" (simple bool, clear threshold)

```python
class Answer(BaseAnswer):
    is_withdrawn: bool = VerifiedField(
        description="True if the candidate response states or clearly implies that fulvestrant has been withdrawn from the market (i.e., is no longer available or approved for use). False if the response states or implies that fulvestrant has NOT been withdrawn and remains available or approved. If the response hedges (e.g., 'withdrawn in some countries but not others'), treat the global/general market status as the deciding factor. If the response is entirely silent on withdrawal status, return False.",
        ground_truth=False,
        verify_with=BooleanMatch(),
    )
```

Q: "What's casimersen drug modality?" A: "Oligonucleotide" (bool-first for concept identification)

```python
class Answer(BaseAnswer):
    identifies_casimersen_as_oligonucleotide: bool = VerifiedField(
        description="True if the candidate response identifies casimersen's drug modality as an oligonucleotide or any recognized subtype or synonym thereof, including but not limited to antisense oligonucleotide (ASO), phosphorodiamidate morpholino oligomer (PMO), morpholino oligonucleotide, or exon-skipping oligonucleotide. False if the response assigns casimersen a different drug modality (e.g., small molecule, antibody, gene therapy) or does not mention its modality at all. If the response provides a more specific oligonucleotide subtype without using the word 'oligonucleotide' explicitly, still return True, as subtypes fall under the oligonucleotide umbrella.",
        ground_truth=True,
        verify_with=BooleanMatch(),
    )
```

Q: "What is the maximum trial phase? What is the status?" A: "Phase II, Completed" (multi-field decomposition with Literal)

```python
class Answer(BaseAnswer):
    max_trial_phase: Literal['Phase I', 'Phase II', 'Phase III', 'Phase IV'] = VerifiedField(
        description="Select the highest clinical trial phase the response identifies for ozanezumab in ALS trials. Choose the option that matches the maximum phase mentioned (e.g., if both Phase I and Phase II are mentioned, select 'Phase II'). If a combined phase is mentioned (e.g., Phase I/II), select the higher number. If the response does not specify a phase or is ambiguous, select the highest phase that can be reasonably inferred from the description.",
        ground_truth='Phase II',
        verify_with=LiteralMatch(),
    )
    trial_status_is_completed: bool = VerifiedField(
        description="True if the response states or clearly implies that the highest-phase ozanezumab ALS trial has a status of 'Completed'. False if the trial is described as ongoing, active, recruiting, terminated early, withdrawn, suspended, or if no status is mentioned. Do not conflate 'terminated' or 'withdrawn' with 'completed'; only trials described as having successfully finished (e.g., 'completed', 'finished', 'concluded') qualify as True.",
        ground_truth=True,
        verify_with=BooleanMatch(),
    )
```

Q: "What are the child terms of anus disease?" A: "proctitis, anal neoplasm, anal polyp, imperforate anus" (list decomposed into individual bool checks)

```python
class Answer(BaseAnswer):
    has_proctitis: bool = VerifiedField(
        description="True if the candidate response lists 'proctitis' as a child term of anus disease (EFO_0009660). False if proctitis is not mentioned at all, or is mentioned only in a context other than being a child/subtype of anus disease. Accept minor spelling or capitalization variants (e.g., 'Proctitis') as True. If the response is ambiguous about whether proctitis is a child term versus a related or parent term, default to False.",
        ground_truth=True,
        verify_with=BooleanMatch(),
    )
    has_anal_neoplasm: bool = VerifiedField(
        description="True if the candidate response lists 'anal neoplasm' (or a semantically equivalent term such as 'anal tumor' or 'neoplasm of the anus') as a child term of anus disease (EFO_0009660). False if the term is absent or mentioned only outside the context of child terms. Accept capitalization variants (e.g., 'Anal Neoplasm') as True.",
        ground_truth=True,
        verify_with=BooleanMatch(),
    )
    # ... has_anal_polyp and has_imperforate_anus follow the same pattern
```

See `references/field-type-guide.md` for the full four-element framework with
before/after examples for each field type.

### 3. Ground Truth Type Must Match Field Annotation

The `ground_truth` value must match the Python type of the field:

| Field type | ground_truth type | Example |
|-----------|------------------|---------|
| `bool` | `bool` | `ground_truth=True` |
| `str` | `str` | `ground_truth="aspirin"` |
| `int` | `int` | `ground_truth=42` |
| `float` | `float` | `ground_truth=3.14` |
| `list[str]` | `list[str]` | `ground_truth=["a", "b"]` |

### 4. ExactMatch Normalizers: Only Four Are Valid

`ExactMatch(normalize=[...])` accepts only these normalizer names: `"lowercase"`, `"strip"`, `"remove_punctuation"`, `"collapse_whitespace"`. There is no `"uppercase"` normalizer. Using an invalid name raises a `ValueError` (`Unknown normalizer: ...`) from `verify()`, not at construction time. The default is `["lowercase", "strip"]`, which is correct for most use cases.

### 5. Do Not Use ExactMatch for Free-Text Fields

`ExactMatch` compares normalized strings for equality. The judge LLM will
almost never produce the exact same phrasing. Casing-only variants DO pass —
the default `lowercase` + `strip` normalization equalizes them — so the
failure mode is wording variation, which is the realistic case. For free-text
extraction, use `BooleanMatch` and phrase the description as a yes/no
question.

```python
# Wrong: the judge's extracted text will rarely match exactly
explanation: str = VerifiedField(
    description="The explanation of the mechanism.",
    ground_truth="Drug X inhibits enzyme Y.",
    verify_with=ExactMatch(),
)

# Correct: convert to a boolean judgment
explains_mechanism: bool = VerifiedField(
    description=(
        "True if the response explains that Drug X inhibits enzyme Y. "
        "False if the mechanism is missing, incorrect, or describes a "
        "different pathway."
    ),
    ground_truth=True,
    verify_with=BooleanMatch(),
)
```

### 6. BooleanMatch Is the Safest Default

When in doubt, use `BooleanMatch`. It works for any fact that can be rephrased
as "does the response contain / state / demonstrate X?" The judge extracts a
`bool`, and BooleanMatch compares it to the ground truth bool.

### 7. Use SetContainment for Multi-Value Extraction

For fields where multiple items must be identified (e.g., "list all side
effects"), use `list[str]` with `SetContainment`, not `str` with
`ContainsAny`.

```python
# Correct
side_effects: list[str] = VerifiedField(
    description="List of side effects mentioned in the response.",
    ground_truth=["nausea", "headache", "dizziness"],
    verify_with=SetContainment(mode="exact"),
)

# Wrong: mixes concerns; ContainsAny checks substrings, not list items
side_effects: str = VerifiedField(
    description="The side effects mentioned.",
    ground_truth="nausea",
    verify_with=ContainsAny(substrings=["nausea", "headache"]),
)
```

### 8. verify() and verify_granular() Are Auto-Generated

Templates with at least one `VerifiedField` get `verify()` and
`verify_granular()` automatically. Do not override them unless you need
custom verification logic (e.g., cross-field constraints).

- `verify()` returns `True` if all fields pass. If the template defines a
  `VerificationStrategy` inner class with a `verify_strategy` attribute
  (an `AllOf` / `AnyOf` / `AtLeastN` composition tree), that tree is evaluated
  instead of the all-fields-pass default.
- `verify_granular()` returns a score in 0.0 to 1.0: a flat weighted average
  by default, or composition-aware scoring (e.g. `AnyOf` takes the best single
  field, `AtLeastN` the top-N) when a `verify_strategy` is present.

### 9. Weight Controls Granular Scoring

The `weight` parameter (default 1.0) affects `verify_granular()` scoring.
Increase it for critical fields that should count more heavily.

```python
primary_target: bool = VerifiedField(
    description="True if the correct drug target is identified.",
    ground_truth=True,
    verify_with=BooleanMatch(),
    weight=3.0,  # 3x more important than default
)
```

### 10. extraction_hint Is Optional Formatting Guidance

`extraction_hint` provides additional formatting guidance to the judge LLM
alongside the JSON schema. Use it when the field needs specific formatting
(e.g., "Return the date in YYYY-MM-DD format").

### 11. Class Name Convention

The template class name does not affect pipeline discovery (the pipeline finds
classes by type, not by name). Convention is to name the class `Answer`, but
any name works.

### 12. Field Descriptions Should List All Acceptable Variations

When the expected value has known synonyms, abbreviations, or alternate
spellings, list them all in the description:

```python
description=(
    "True if the response identifies BCL2 as the target. "
    "Acceptable forms: BCL2, Bcl-2, BCL-2, B-cell lymphoma 2."
)
```

### 13. Conditional Ground Truth (Scenario-Dependent Thresholds)

When the correct value of a field depends on an earlier scenario node's parsed
result, set `ground_truth` to a `ConditionalGroundTruth`. It resolves a
`source` dot-path against scenario `node_results`, picks the matching case, and
falls back to `default` (also used outside scenario contexts):

```python
from karenina.schemas.entities.conditional import ConditionalGroundTruth, GroundTruthCase

pivot_topic: str = VerifiedField(
    description="The topic the assistant pivots to after the prior turn.",
    ground_truth=ConditionalGroundTruth(
        source="node_results.probe.parsed.topic",
        cases={"off_topic": GroundTruthCase(value="refusal")},
        default=GroundTruthCase(value="answer"),
    ),
    verify_with=ExactMatch(),
)
```

Each `GroundTruthCase` may also pass `verify_with=<primitive>` to override the
field's primitive for that case.

## Reference Pointers

For detailed information on specific topics:

- **All verification primitives**: `references/primitive-catalog.md`
- **Field types and description writing**: `references/field-type-guide.md`
- **Template skeleton**: `assets/template-skeleton.py`
- **Validation script**: `scripts/validate_template.py`
- **Full karenina docs**: The `using-karenina` skill contains the complete karenina docs in its `references/` directory (consult for how templates interact with the pipeline, evaluation modes, or result structure)
