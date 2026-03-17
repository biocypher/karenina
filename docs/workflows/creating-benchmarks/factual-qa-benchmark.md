---
jupyter:
  jupytext:
    formats: docs/workflows/creating-benchmarks//md,docs/notebooks/creating-benchmarks//ipynb
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Factual QA Benchmark

When your questions have definitive correct answers (a specific name, a numeric value, a known date), you can verify correctness entirely through template logic. This notebook introduces **answer templates**, the core mechanism in karenina for structuring and verifying LLM responses. You declare fields with `VerifiedField`, attach a verification primitive to each, and the pipeline handles the rest.

This notebook walks through five template patterns for factual verification, progressing from simple boolean checks to multi-field partial credit. If your evaluation also needs subjective quality assessment (clarity, completeness, reasoning quality), see [Full Evaluation Benchmark](full-evaluation-benchmark.ipynb) which combines templates with rubric traits.

**What you'll learn:**

- Boolean check: check concept presence without string matching
- String extraction: extract and verify values with built-in normalization
- Numeric tolerance: accept answers within a specified range
- Regex checks: match patterns against the raw response trace without LLM parsing
- Multi-field with weighted scoring: evaluate multiple dimensions with `weight` parameters

```python tags=["hide-cell"]
# Setup cell: hidden in rendered documentation.
# No mocking needed. All examples create Benchmark objects locally.
import re
import tempfile
from pathlib import Path

from pydantic import Field

from karenina.schemas.entities import BaseAnswer, VerifiedField
from karenina.schemas.entities.primitives import (
    BooleanMatch,
    ExactMatch,
    NumericTolerance,
    TraceRegex,
)
```

---

## Create the Benchmark

```python
from karenina import Benchmark

benchmark = Benchmark.create(
    name="Biomedical Factual QA",
    description="Evaluates LLM knowledge of biomedical facts using template-only verification",
    version="1.0.0",
)
print(f"Created: {benchmark.name}")
print(f"Questions: {benchmark.question_count}")
```

---

## Add Questions with Templates

Each question below demonstrates a different template pattern, ordered from simplest to most expressive. We start with a boolean check (the minimal approach), then progress through string extraction, numeric comparison, and regex handling, culminating in multi-field weighted scoring that combines several techniques. Each pattern builds on the previous. If you find a simpler pattern sufficient for your use case, you likely don't need the more complex ones.

For notebook compatibility, we use a two-step approach: first add the question, then define and attach the template class using `update_template()`.

### Question 1: Boolean Check (Cancer Genetics)

A boolean field checks whether a concept is present in the response, delegating synonym handling to the judge through the field description. No string matching needed.

```python
# First, add the question
q1_id = benchmark.add_question(
    question="What gene is most commonly mutated in human cancers?",
    raw_answer="TP53 is the most commonly mutated gene in human cancers",
)

# Then define and attach the template
class Answer(BaseAnswer):
    identifies_tp53: bool = VerifiedField(
        description=(
            "True if the response identifies TP53 (including p53, TP53, or "
            "'tumor protein p53') as the single most commonly mutated gene "
            "in human cancers. False if TP53 is mentioned as one of several "
            "commonly mutated genes without being singled out as the most "
            "frequent (e.g., 'TP53, KRAS, and PIK3CA are commonly mutated')."
        ),
        ground_truth=True,
        verify_with=BooleanMatch(),
    )

benchmark.update_template(q1_id, Answer)
print(f"Q1 added with template: {q1_id[:50]}...")
```

These templates use `VerifiedField` with built-in verification primitives. For cases requiring custom logic, see [Advanced: Custom Verification](../../core_concepts/answer-templates.md#8-advanced-custom-verification).

A boolean check avoids string matching entirely. Instead of extracting "TP53" as text and comparing it, we ask the judge "Did the response identify TP53 as the most common?" This is more reliable because the judge handles synonyms (TP53, p53, tumor protein p53) through its description. The `BooleanMatch()` primitive compares the extracted boolean against `ground_truth=True` automatically; no `verify()` method needed.

The key design decision is in the field description: it must be specific enough to disambiguate edge cases. The `identifies_tp53` description explicitly states that merely listing TP53 alongside other cancer genes should return False, because the response must single it out as the *most commonly mutated*. Without this, the judge would likely return True for "TP53, KRAS, and PIK3CA are frequently mutated in cancer," which mentions TP53 but doesn't actually answer the question. Invest time in writing precise descriptions; they are the instructions your judge LLM follows.

<div class="admonition tip">
<p class="admonition-title">When to use boolean checks</p>
<p>Prefer boolean fields when you need to check for the presence of a concept and the concept has multiple valid surface forms. The judge handles normalization, and <code>BooleanMatch()</code> handles verification automatically. For checking multiple concepts independently with weighted scoring, see <a href="#question-5-multi-field-with-weighted-scoring-vaccine-mechanism">Multi-Field with Weighted Scoring</a>.</p>
</div>

### Question 2: String Extraction (Blood Type Identification)

Using `str` fields tells the judge to extract a specific piece of text from the response and populate the template with it. In this role the judge acts more like a *parser* than an evaluator: the field description guides what to look for and helps the model produce clean output, but the judge chooses the value based on what the response actually says.

Crucially, `str` fields let you design verification tests without exposing the ground truth to the judge. This matters because a judge that sees the expected answer may anchor on it rather than faithfully extracting what the response contains. With boolean fields, it is common practice (though not required) to embed the ground truth in the description itself ("True if the response identifies BCL2..."), which can introduce this anchoring effect. String fields avoid this: the description specifies *what kind of value* to extract, not what the correct value is. The `ground_truth` value on the `VerifiedField` is used only during verification, which the judge never sees.

The tradeoff: since free text is far more variable than a boolean, you need an appropriate primitive with normalization to handle formatting differences in the extracted string.

```python
q2_id = benchmark.add_question(
    question="What is the most common blood type worldwide?",
    raw_answer="O+",
)

class Answer(BaseAnswer):
    blood_type: str = VerifiedField(
        description=(
            "The blood type stated in the response as the most common worldwide. "
            "Use standard notation: uppercase letter followed by '+' or '-' "
            "(e.g., 'O+' not 'O positive' or 'type O'). If the response uses "
            "the full name ('O positive'), normalize to shorthand ('O+'). "
            "If the response names multiple blood types, extract the one "
            "identified as the most common overall. If the response distinguishes "
            "by region, extract the type stated as most common globally."
        ),
        ground_truth="O+",
        verify_with=ExactMatch(normalize=["lowercase", "strip"]),
    )

benchmark.update_template(q2_id, Answer)
print(f"Q2 added with template: {q2_id[:50]}...")
```

In the example above, the field description asks for "the blood type stated in the response" without ever mentioning that "O+" is correct. The judge extracts whatever it finds; `ExactMatch` then compares the extracted value against `ground_truth="O+"`. The `normalize=["lowercase", "strip"]` parameter applies lowercasing and whitespace stripping before comparison, handling minor formatting differences from the judge. Because the description already asks for standard notation (e.g., "O+" not "O positive"), the normalization here handles casing and whitespace, while the description handles format normalization at the extraction step.

<div class="admonition tip">
<p class="admonition-title">Boolean vs. string: ground truth exposure</p>
<p>With <strong>boolean</strong> fields, it is common practice to include the ground truth in the description ("True if BCL2 is identified"), though this is not strictly required. With <strong>string</strong> fields, the description specifies what to extract without revealing the expected value, keeping the judge unbiased. Use string when you want to prevent anchoring, need programmatic control over matching, or need the extracted value for downstream analysis.</p>
</div>

### Question 3: Numeric Tolerance (Normal Body Temperature)

A body temperature of 37.2 degrees Celsius is clinically normal even though the textbook value is 37.0. The `NumericTolerance` primitive handles this: the judge extracts a raw number, and the primitive decides what counts as "close enough" based on a tolerance you specify.

```python
q3_id = benchmark.add_question(
    question="What is the normal human body temperature in degrees Celsius?",
    raw_answer="37.0",
)

class Answer(BaseAnswer):
    temperature_celsius: float = VerifiedField(
        description=(
            "The normal body temperature stated in the response, in degrees "
            "Celsius. Extract the value stated as the standard or average "
            "normal temperature, not extreme or atypical values. If the "
            "response gives the value in Fahrenheit (e.g., 98.6 F), extract "
            "the Celsius equivalent. If a range is given (e.g., 36.5-37.5), "
            "extract the midpoint. If the response gives temperatures for "
            "different measurement sites (oral, rectal, axillary), extract "
            "the oral temperature or the one presented as the primary value."
        ),
        ground_truth=37.0,
        verify_with=NumericTolerance(tolerance=0.5, mode="absolute"),
    )

benchmark.update_template(q3_id, Answer)
print(f"Q3 added with template: {q3_id[:50]}...")
```

The `mode="absolute"` setting means the extracted value must be within 0.5 of 37.0 (i.e., 36.5 to 37.5). For exact counts where only one value is correct, use `NumericTolerance(tolerance=0, mode="absolute")`. Here is the same pattern applied to a chromosome count with an `int` field:

```python
from karenina.schemas.entities.primitives import NumericExact

# Example: exact numeric match
class Answer(BaseAnswer):
    pair_count: int = VerifiedField(
        description=(
            "The number of chromosome pairs in a normal human somatic cell, "
            "as a whole number. If the response gives the total count (e.g., 46), "
            "extract the number of pairs (23)."
        ),
        ground_truth=23,
        verify_with=NumericExact(),
    )

print("Exact-match example defined (not added to benchmark)")
```

Note: this chromosome example is a standalone illustration, not added to the benchmark. `NumericExact()` is a convenience primitive equivalent to `NumericTolerance(tolerance=0, mode="absolute")`.

<div class="admonition tip">
<p class="admonition-title">Choosing tolerance values</p>
<p>Use <code>NumericExact()</code> or <code>NumericTolerance(tolerance=0)</code> for exact counts (chromosomes, electrons). Use <code>mode="absolute"</code> for physical measurements with known precision (body temperature, boiling points). Use <code>mode="percentage"</code> for values that span wide ranges (e.g., <code>NumericTolerance(tolerance=10, mode="percentage")</code> to accept within 10%).</p>
</div>

### Question 4: Regex Checks (Discovery Date)

`TraceRegex` is a verification primitive that runs a regex pattern against the **raw LLM response trace** (the full text the answering model produced, before parsing). The field type must be `bool`, and it is excluded from the parsing schema automatically, so no judge model is needed for these checks. This gives you classical pattern-matching evaluation through karenina's infrastructure.

```python
q4_id = benchmark.add_question(
    question="When was penicillin discovered by Alexander Fleming?",
    raw_answer="September 28, 1928",
)

class Answer(BaseAnswer):
    mentions_discovery_year: bool = VerifiedField(
        description="True if the response mentions the year 1928",
        ground_truth=True,
        verify_with=TraceRegex(pattern=r"\b1928\b"),
    )
    mentions_fleming: bool = VerifiedField(
        description="True if the response mentions Fleming",
        ground_truth=True,
        verify_with=TraceRegex(pattern=r"\bFleming\b"),
    )

benchmark.update_template(q4_id, Answer)
print(f"Q4 added with template: {q4_id[:50]}...")
```

Each `TraceRegex` field is a named check: the `pattern` is applied to the raw trace, and the result is compared against `ground_truth`. Because these fields are excluded from the parsing schema, the pipeline detects a template with only `TraceRegex` fields and skips LLM parsing entirely. No judge model is needed.

`TraceRegex` fields can also be combined with regular parsed fields when you want both structured extraction and pattern matching. In that case, the parsed fields are verified by their own primitives and the `TraceRegex` fields independently check the raw response. See [Answer Templates: Regex Checks](../../core_concepts/answer-templates.md#regex-checks) for a combined example.

### Question 5: Multi-Field with Weighted Scoring (Vaccine Mechanism)

The previous patterns each extract a single value. For questions with multiple dimensions, where you want to check delivery method *and* target protein *and* immune response independently, use multiple `VerifiedField` declarations with `weight` parameters. The pipeline computes a weighted score automatically; no custom `verify()` or `verify_granular()` needed.

```python
q5_id = benchmark.add_question(
    question="How do mRNA vaccines work?",
    raw_answer=(
        "mRNA vaccines deliver genetic instructions for cells to produce a "
        "spike protein, which triggers an immune response producing antibodies."
    ),
)

class Answer(BaseAnswer):
    delivery_mechanism: str = VerifiedField(
        description=(
            "The primary mechanism by which the vaccine delivers instructions "
            "to cells, as described in the response (e.g., 'mRNA instructions', "
            "'messenger RNA', 'genetic code'). Normalize to lowercase. Extract "
            "the delivery method itself, not subsequent biological processes "
            "(protein production, immune response). If the response describes "
            "multiple steps, extract the initial delivery mechanism."
        ),
        ground_truth="mrna",
        verify_with=ExactMatch(normalize=["lowercase", "strip"]),
        weight=2.0,
    )
    target_protein: str = VerifiedField(
        description=(
            "The protein that cells are instructed to produce, as described "
            "in the response (e.g., 'spike protein'). Normalize to lowercase. "
            "If the response names a specific variant (e.g., 'Wuhan spike "
            "protein'), extract the general protein name. If the response "
            "mentions multiple proteins, extract the one identified as the "
            "primary target of the vaccine."
        ),
        ground_truth="spike protein",
        verify_with=ExactMatch(normalize=["lowercase", "strip"]),
        weight=2.0,
    )
    mentions_immune_response: bool = VerifiedField(
        description=(
            "True if the response describes the resulting immune response "
            "(antibody production, T-cell activation, or immune memory). "
            "False if only the protein production step is mentioned without "
            "describing what happens next."
        ),
        ground_truth=True,
        verify_with=BooleanMatch(),
        weight=1.0,
    )

benchmark.update_template(q5_id, Answer)
print(f"Q5 added with template: {q5_id[:50]}...")
```

The `weight` parameter controls how much each field contributes to the overall score. Here, `delivery_mechanism` and `target_protein` each have `weight=2.0` while `mentions_immune_response` has `weight=1.0`, making the two string fields together worth 80% of the total score. The pipeline computes the weighted score automatically from the field weights; you do not need to write `verify()` or `verify_granular()` methods.

---

## Inspect the Benchmark

```python
print(f"Total questions: {benchmark.question_count}")
print(f"With templates:  {len(benchmark.get_finished_templates())}")
print(f"Progress:        {benchmark.get_progress()}%")

# Preview a question
q = benchmark.get_question(q1_id)
print(f"\nQ1: {q['question'][:60]}...")
print(f"Has template: {q.get('finished', False)}")
```

---

## Save the Benchmark

```python
tmpdir = tempfile.mkdtemp()
checkpoint_path = Path(tmpdir) / "biomedical_factual_qa.jsonld"
benchmark.save(checkpoint_path)
print(f"Saved to: {checkpoint_path.name}")
print(f"File exists: {checkpoint_path.exists()}")
```

---

## Reload and Verify Round-Trip

```python
loaded = Benchmark.load(checkpoint_path)

print(f"Name:      {loaded.name}")
print(f"Questions: {loaded.question_count}")
print(f"Templates: {len(loaded.get_finished_templates())}")
print(f"Match:     {loaded.question_count == benchmark.question_count}")
```

---

## Cleanup

```python
import shutil
shutil.rmtree(tmpdir, ignore_errors=True)
```

---

## Template Pattern Summary

The five patterns above form a toolkit for factual verification. Most real benchmarks use a mix: boolean fields for presence checks, string fields for extractable values, and numeric fields for measurements. The multi-field pattern (Q5) is the most common in practice because real questions usually have multiple verifiable dimensions. Start with the simplest pattern that handles your ground truth, and reach for more complex ones only when needed.

| Pattern | VerifiedField + Primitive | Use When | Don't Use When | Example |
|---------|--------------------------|----------|----------------|---------|
| Boolean check | `VerifiedField` + `BooleanMatch()` | Known concept, multiple synonyms | You need the extracted value | Gene identification, pathway presence |
| String extraction | `VerifiedField` + `ExactMatch(normalize=[...])` | Strict matching with normalization | Only need presence check (use boolean) | Blood types, gene symbols |
| Numeric tolerance | `VerifiedField` + `NumericTolerance(tolerance=..., mode=...)` | Measurements or counts | Value is categorical, not numeric | Body temperature, chromosome counts |
| Regex checks | `VerifiedField` + `TraceRegex(pattern=...)` | Pattern must appear in raw response; no LLM judge needed | Only need to normalize parsed field values | Year mentions, keyword presence, citation counts |
| Multi-field + weights | Multiple `VerifiedField` with `weight=` | Multiple dimensions, want weighted scoring | Single-answer questions | Mechanisms, multi-part descriptions |

---

## Next Steps

- [Full Evaluation Benchmark](full-evaluation-benchmark.ipynb): Add rubric traits for quality assessment
- [Quality Assessment](quality-assessment-benchmark.ipynb): Rubric-only evaluation without templates
- [Scaled Authoring](scaled-authoring.ipynb): Bulk workflows and auto-generation
- [Answer Templates](../core_concepts/answer-templates.ipynb): Deep dive into template concepts
