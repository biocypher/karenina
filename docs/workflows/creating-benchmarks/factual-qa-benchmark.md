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

When your questions have definitive correct answers (a specific name, a numeric value, a known date), you can verify correctness entirely through template logic. This notebook introduces **answer templates**, the core mechanism in karenina for structuring and verifying LLM responses. You define a `verify()` method that checks the parsed response against ground truth, and the pipeline handles the rest.

This notebook walks through five template patterns for factual verification, progressing from simple boolean checks to multi-field partial credit. If your evaluation also needs subjective quality assessment (clarity, completeness, reasoning quality), see [Full Evaluation Benchmark](full-evaluation-benchmark.ipynb) which combines templates with rubric traits.

**What you'll learn:**

- Boolean check: check concept presence without string matching
- String extraction: extract values and verify with programmatic logic
- Numeric tolerance: accept answers within a specified range
- Regex checks: define `self.regex` patterns checked against the raw response trace
- Multi-field with partial credit: evaluate multiple dimensions with `verify_granular()`

```python tags=["hide-cell"]
# Setup cell: hidden in rendered documentation.
# No mocking needed. All examples create Benchmark objects locally.
import re
import tempfile
from pathlib import Path

from pydantic import Field

from karenina.schemas.entities import BaseAnswer
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

Each question below demonstrates a different template pattern, ordered from simplest to most expressive. We start with a boolean check (the minimal approach), then progress through string extraction, numeric comparison, and regex handling, culminating in multi-field partial credit that combines several techniques. Each pattern builds on the previous. If you find a simpler pattern sufficient for your use case, you likely don't need the more complex ones.

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
    identifies_tp53: bool = Field(
        description=(
            "True if the response identifies TP53 (including p53, TP53, or "
            "'tumor protein p53') as the single most commonly mutated gene "
            "in human cancers. False if TP53 is mentioned as one of several "
            "commonly mutated genes without being singled out as the most "
            "frequent (e.g., 'TP53, KRAS, and PIK3CA are commonly mutated')."
        )
    )

    def verify(self) -> bool:
        return self.identifies_tp53

benchmark.update_template(q1_id, Answer)
print(f"Q1 added with template: {q1_id[:50]}...")
```

A boolean check avoids string matching entirely. Instead of extracting "TP53" as text and comparing it, we ask the judge "Did the response identify TP53 as the most common?" This is more reliable because the judge handles synonyms (TP53, p53, tumor protein p53) through its description.

The key design decision is in the field description: it must be specific enough to disambiguate edge cases. The `identifies_tp53` description explicitly states that merely listing TP53 alongside other cancer genes should return False — the response must single it out as the *most commonly mutated*. Without this, the judge would likely return True for "TP53, KRAS, and PIK3CA are frequently mutated in cancer," which mentions TP53 but doesn't actually answer the question. Invest time in writing precise descriptions; they are the instructions your judge LLM follows.

<div class="admonition tip">
<p class="admonition-title">When to use boolean checks</p>
<p>Prefer boolean fields when you need to check for the presence of a concept and the concept has multiple valid surface forms. The judge handles normalization so <code>verify()</code> stays trivial. For checking multiple concepts independently with partial credit, see <a href="#question-5-multi-field-with-partial-credit-vaccine-mechanism">Multi-Field with Partial Credit</a>.</p>
</div>

### Question 2: String Extraction (Blood Type Identification)

Using `str` fields tells the judge to extract a specific piece of text from the response and populate the template with it. In this role the judge acts more like a *parser* than an evaluator: the field description guides what to look for and helps the model produce clean output, but the judge chooses the value based on what the response actually says.

Crucially, `str` fields let you design verification tests without exposing the ground truth to the judge. This matters because a judge that sees the expected answer may anchor on it rather than faithfully extracting what the response contains. With boolean fields, it is common practice (though not required) to embed the ground truth in the description itself ("True if the response identifies BCL2..."), which can introduce this anchoring effect. String fields avoid this: the description specifies *what kind of value* to extract, not what the correct value is. The actual correctness check happens in `verify()`, which the judge never sees.

The tradeoff: since free text is far more variable than a boolean, you will typically need normalization logic in `verify()` to handle formatting differences in the extracted string.

```python
q2_id = benchmark.add_question(
    question="What is the most common blood type worldwide?",
    raw_answer="O+",
)

class Answer(BaseAnswer):
    blood_type: str = Field(
        description=(
            "The blood type stated in the response as the most common worldwide. "
            "Use standard notation: uppercase letter followed by '+' or '-' "
            "(e.g., 'O+' not 'O positive' or 'type O'). If the response uses "
            "the full name ('O positive'), normalize to shorthand ('O+')."
        )
    )

    def ground_truth(self):
        self.correct = {"blood_type": "O+"}

    def verify(self) -> bool:
        extracted = self.blood_type.strip().upper().replace(" ", "")
        # Normalize common variations
        extracted = (
            extracted
            .replace("POSITIVE", "+")
            .replace("NEGATIVE", "-")
            .replace("POS", "+")
            .replace("NEG", "-")
        )
        return extracted == self.correct["blood_type"]

benchmark.update_template(q2_id, Answer)
print(f"Q2 added with template: {q2_id[:50]}...")
```

In the example above, the field description asks for "the blood type stated in the response" without ever mentioning that "O+" is correct. The judge extracts whatever it finds; `verify()` then checks the extracted value against ground truth. Note the defensive normalization chain: even though the description asks for standard notation, the judge may return "O positive" or "o+". Because text output is inherently variable, handling these variations in `verify()` is expected and keeps the correctness logic deterministic.

<div class="admonition tip">
<p class="admonition-title">Boolean vs. string: ground truth exposure</p>
<p>With <strong>boolean</strong> fields, it is common practice to include the ground truth in the description ("True if BCL2 is identified"), though this is not strictly required. With <strong>string</strong> fields, the description specifies what to extract without revealing the expected value, keeping the judge unbiased. Use string when you want to prevent anchoring, need programmatic control over matching, or need the extracted value for downstream analysis.</p>
</div>

### Question 3: Numeric Tolerance (Normal Body Temperature)

Because karenina verification happens through templates, the `verify()` method is just Python code, which means you can implement any verification logic you need. Numeric tolerance is a good example: the judge extracts a raw number, and your code decides what counts as "close enough." A body temperature of 37.2°C is clinically normal even though the textbook value is 37.0°C. Rather than trying to encode tolerance rules in a prompt, you write a programmatic comparison with an explicit margin.

```python
q3_id = benchmark.add_question(
    question="What is the normal human body temperature in degrees Celsius?",
    raw_answer="37.0",
)

class Answer(BaseAnswer):
    temperature_celsius: float = Field(
        description=(
            "The normal body temperature stated in the response, in degrees "
            "Celsius. If the response gives the value in Fahrenheit (e.g., "
            "98.6°F), extract the Celsius equivalent. If a range is given "
            "(e.g., 36.5-37.5), extract the midpoint."
        )
    )

    def ground_truth(self):
        self.correct = {"temperature_celsius": 37.0}
        self.tolerance = 0.5  # Accept 36.5-37.5, reflecting natural variation

    def verify(self) -> bool:
        return abs(self.temperature_celsius - self.correct["temperature_celsius"]) <= self.tolerance

benchmark.update_template(q3_id, Answer)
print(f"Q3 added with template: {q3_id[:50]}...")
```

For exact counts where only one value is correct, set `tolerance = 0`. Here is the same pattern applied to a chromosome count with an `int` field:

```python
# Example: numeric tolerance with exact match
class Answer(BaseAnswer):
    pair_count: int = Field(
        description=(
            "The number of chromosome pairs in a normal human somatic cell, "
            "as a whole number. If the response gives the total count (e.g., 46), "
            "extract the number of pairs (23)."
        )
    )

    def ground_truth(self):
        self.correct = {"pair_count": 23}
        self.tolerance = 0  # Exact match: only 23 is correct

    def verify(self) -> bool:
        return abs(self.pair_count - self.correct["pair_count"]) <= self.tolerance

print("Exact-match example defined (not added to benchmark)")
```

Note: this chromosome example is a standalone illustration, not added to the benchmark. The pattern is identical to Q3; only the tolerance value and field type (`int` vs `float`) change.

<div class="admonition tip">
<p class="admonition-title">Choosing tolerance values</p>
<p>Use <code>tolerance = 0</code> for exact counts (chromosomes, electrons). Use absolute tolerance for physical measurements with known precision (body temperature, boiling points). Use percentage-based tolerance for values that span wide ranges (e.g., <code>tolerance_pct = 10</code> to accept within 10%).</p>
</div>

### Question 4: Regex Checks (Discovery Date)

Templates can define regex patterns via `self.regex` that the pipeline runs automatically against the **raw LLM response trace** (the full text the answering model produced, before parsing). When a template has **no user-defined fields** — only `self.regex` — the pipeline detects this automatically, skips LLM parsing entirely, and no judge model is needed. This gives you classical pattern-matching evaluation through karenina's infrastructure.

```python
q4_id = benchmark.add_question(
    question="When was penicillin discovered by Alexander Fleming?",
    raw_answer="September 28, 1928",
)

class Answer(BaseAnswer):
    def ground_truth(self):
        self.regex = {
            "mentions_discovery_year": {
                "pattern": r"\b1928\b",
                "expected": "1928",
                "match_type": "exact",
            },
            "mentions_fleming": {
                "pattern": r"\bFleming\b",
                "expected": "Fleming",
                "match_type": "exact",
            },
        }

benchmark.update_template(q4_id, Answer)
print(f"Q4 added with template: {q4_id[:50]}...")
```

No fields, no `verify()`, no judge LLM — the regex patterns are the entire evaluation. Each entry in `self.regex` is a named check with a `pattern` (applied via `re.findall()` on the raw trace), an `expected` value, and a `match_type`:

- **`"exact"`**: exactly one match, equal to `expected` (str). Use for specific terms.
- **`"contains"`**: `expected` (str) is among the matches. Use when a pattern has alternations like `r"\b(activates|inhibits|blocks)\b"` and you want a specific one.
- **`"count"`**: number of matches equals `expected` (int). Use for counting occurrences like citations.
- **`"all"`**: all items in `expected` (list) are found in matches. Use when multiple terms must all appear.

Regex-only templates can also be combined with fields when you want both structured extraction and pattern matching. In that case, `verify()` checks the parsed fields and `self.regex` independently checks the raw response — both must pass. See [Answer Templates: Regex Checks](../../core_concepts/answer-templates.md#regex-checks) for a combined example.

### Question 5: Multi-Field with Partial Credit (Vaccine Mechanism)

The previous patterns each extract a single value. For questions with multiple dimensions, where you want to check delivery method *and* target protein *and* immune response independently, use multiple fields with both `verify()` (all-or-nothing) and `verify_granular()` (partial credit). This is the most expressive pattern and combines techniques from the earlier examples.

```python
q5_id = benchmark.add_question(
    question="How do mRNA vaccines work?",
    raw_answer=(
        "mRNA vaccines deliver genetic instructions for cells to produce a "
        "spike protein, which triggers an immune response producing antibodies."
    ),
)

class Answer(BaseAnswer):
    delivery_mechanism: str = Field(
        description=(
            "How the vaccine delivers its payload, as described in the response "
            "(e.g., 'mRNA instructions', 'messenger RNA', 'genetic code'). "
            "Normalize to lowercase."
        )
    )
    target_protein: str = Field(
        description=(
            "The protein that cells are instructed to produce (e.g., 'spike "
            "protein'). Normalize to lowercase. If the response names a specific "
            "variant, extract the general protein name."
        )
    )
    mentions_immune_response: bool = Field(
        description=(
            "True if the response describes the resulting immune response "
            "(antibody production, T-cell activation, or immune memory). "
            "False if only the protein production step is mentioned without "
            "describing what happens next."
        )
    )

    def ground_truth(self):
        self.correct = {
            "delivery_mechanism": "mrna",
            "target_protein": "spike protein",
            "mentions_immune_response": True,
        }

    def _check_delivery(self) -> bool:
        mechanism = self.delivery_mechanism.strip().lower()
        return "mrna" in mechanism or "messenger rna" in mechanism

    def _check_target(self) -> bool:
        return "spike" in self.target_protein.strip().lower()

    def _check_immune_response(self) -> bool:
        return self.mentions_immune_response == self.correct["mentions_immune_response"]

    def verify(self) -> bool:
        return (
            self._check_delivery()
            and self._check_target()
            and self._check_immune_response()
        )

    def verify_granular(self) -> float:
        checks = [
            self._check_delivery(),
            self._check_target(),
            self._check_immune_response(),
        ]
        return sum(checks) / len(checks)

benchmark.update_template(q5_id, Answer)
print(f"Q5 added with template: {q5_id[:50]}...")
```

The pipeline calls `verify()` for the pass/fail result used in scoring. `verify_granular()` provides a 0.0 to 1.0 score for finer-grained analysis. Getting 2 out of 3 checks correct yields 0.67 instead of a flat failure.

Extracting each check into a private `_check_*` method keeps both `verify()` and `verify_granular()` readable and ensures the logic is defined once. This pattern is recommended whenever you have three or more fields to verify.

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

| Pattern | Use When | Don't Use When | Example |
|---------|----------|----------------|---------|
| Boolean check | Known concept, multiple synonyms | You need the extracted value | Gene identification, pathway presence |
| String extraction | Strict matching with programmatic control | Only need presence check (use boolean) | Blood types, gene symbols |
| Numeric tolerance | Measurements or counts | Value is categorical, not numeric | Body temperature, chromosome counts |
| Regex checks (`self.regex`) | Pattern must appear in raw response; no fields = no LLM judge needed | Only need to normalize parsed field values | Year mentions, keyword presence, citation counts |
| Multi-field + partial credit | Multiple dimensions, want granular scoring | Single-answer questions | Mechanisms, multi-part descriptions |

---

## Next Steps

- [Full Evaluation Benchmark](full-evaluation-benchmark.ipynb): Add rubric traits for quality assessment
- [Quality Assessment](quality-assessment-benchmark.ipynb): Rubric-only evaluation without templates
- [Scaled Authoring](scaled-authoring.ipynb): Bulk workflows and auto-generation
- [Answer Templates](../core_concepts/answer-templates.ipynb): Deep dive into template concepts
