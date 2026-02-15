---
jupyter:
  jupytext:
    formats: docs/workflows/creating-benchmarks//md,docs/notebooks/creating-benchmarks//ipynb
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

# Factual QA Benchmark

This scenario creates a benchmark that evaluates factual correctness using templates alone. Each question has a custom Answer template with a `verify()` method that checks the parsed response against ground truth. This is the simplest evaluation strategy — ideal when questions have definitive correct answers.

**What you'll learn:**

- Boolean decomposition — check concept presence without string matching
- String normalization — extract and normalize values before comparison
- Numeric tolerance — accept answers within a specified range
- Regex in `verify()` — handle variable formats in judge output
- Multi-field with partial credit — evaluate multiple dimensions with `verify_granular()`

```python tags=["hide-cell"]
# Setup cell: hidden in rendered documentation.
# No mocking needed — all examples create Benchmark objects locally.
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

Each question below demonstrates a different template pattern. For notebook compatibility, we use a two-step approach: first add the question, then define and attach the template class using `update_template()`.

### Question 1: Boolean Decomposition — Drug Target Identification

Boolean fields check whether a concept is present in the response, delegating synonym handling to the judge through the field description. No string matching needed.

```python
# First, add the question
q1_id = benchmark.add_question(
    question="What is the approved pharmacological target of venetoclax?",
    raw_answer="BCL2",
)

# Then define and attach the template
class Answer(BaseAnswer):
    identifies_bcl2: bool = Field(
        description=(
            "True if the response identifies BCL2 (including Bcl-2, BCL-2, or "
            "B-cell lymphoma 2) as the direct pharmacological target of venetoclax. "
            "False if BCL2 is mentioned only as a pathway member or a different "
            "protein is identified as the primary target."
        )
    )
    mentions_inhibition: bool = Field(
        description=(
            "True if the response describes inhibition as the mechanism of action "
            "(including phrases like 'blocks', 'suppresses', or 'prevents activity of'). "
            "False if only downstream effects are mentioned without the mechanism."
        )
    )

    def model_post_init(self, __context):
        self.correct = {"identifies_bcl2": True, "mentions_inhibition": True}

    def verify(self) -> bool:
        return all(
            getattr(self, field) == self.correct[field]
            for field in self.correct
        )

    def verify_granular(self) -> float:
        matches = sum(
            1 for field in self.correct
            if getattr(self, field) == self.correct[field]
        )
        return matches / len(self.correct)

benchmark.update_template(q1_id, Answer)
print(f"Q1 added with template: {q1_id[:50]}...")
```

Boolean decomposition avoids string matching entirely. Instead of extracting "BCL2" as text and comparing it, we ask the judge "Did the response identify BCL2?" This is more reliable because the judge handles synonyms (Bcl-2, BCL-2, B-cell lymphoma 2) through its description.

!!! tip "When to use boolean decomposition"
    Prefer boolean fields when you have a known set of concepts to check for and the concepts have multiple valid surface forms. Each `bool` field is an independent check — the judge handles normalization, and `verify()` stays trivial.

### Question 2: String Normalization — Gene Symbol Extraction

When you need the actual extracted value (not just presence), use a `str` field with normalization in `verify()`.

```python
q2_id = benchmark.add_question(
    question="What gene is most commonly mutated in human cancers?",
    raw_answer="TP53",
)

class Answer(BaseAnswer):
    gene_symbol: str = Field(
        description=(
            "The official HGNC gene symbol mentioned in the response as the most "
            "commonly mutated gene in human cancers. Use uppercase without hyphens "
            "(e.g., 'TP53' not 'p53' or 'tp-53'). If the response uses an alias "
            "or alternate name, normalize to the standard symbol."
        )
    )

    def model_post_init(self, __context):
        self.correct = {"gene_symbol": "TP53"}

    def verify(self) -> bool:
        extracted = self.gene_symbol.strip().upper().replace("-", "")
        expected = self.correct["gene_symbol"].upper()
        return extracted == expected

benchmark.update_template(q2_id, Answer)
print(f"Q2 added with template: {q2_id[:50]}...")
```

Use string extraction when you need the value itself (for downstream analysis or display), or when the question has a single extractable answer with a canonical form. Boolean decomposition is better when you only care about presence and the concept has many synonyms.

### Question 3: Numeric Tolerance — Chromosome Count

Numeric fields use comparison with tolerance in `verify()`. For exact counts, set tolerance to zero; for measurements or estimates, use an appropriate margin.

```python
q3_id = benchmark.add_question(
    question="How many chromosome pairs are in a normal human somatic cell?",
    raw_answer="23",
)

class Answer(BaseAnswer):
    pair_count: int = Field(
        description=(
            "The number of chromosome pairs stated in the response, as a whole "
            "number. If the response gives the total count (e.g., 46), extract "
            "the number of pairs (23). If the response says 'twenty-three', "
            "extract the numeric value 23."
        )
    )

    def model_post_init(self, __context):
        self.correct = {"pair_count": 23}
        self.tolerance = 0  # Exact match for this question

    def verify(self) -> bool:
        return abs(self.pair_count - self.correct["pair_count"]) <= self.tolerance

benchmark.update_template(q3_id, Answer)
print(f"Q3 added with template: {q3_id[:50]}...")
```

For measurements where approximate answers are acceptable, increase the tolerance. Here is the same pattern applied to a temperature question with a `float` field:

```python
# Example: numeric tolerance with float
class Answer(BaseAnswer):
    temperature: float = Field(
        description=(
            "The boiling point temperature stated in the response, in degrees "
            "Celsius. If the response gives the value in Fahrenheit or Kelvin, "
            "extract the Celsius equivalent."
        )
    )

    def model_post_init(self, __context):
        self.correct = {"temperature": 100.0}
        self.tolerance = 0.5  # Accept within +/-0.5 degrees C

    def verify(self) -> bool:
        return abs(self.temperature - self.correct["temperature"]) <= self.tolerance

print("Tolerance example defined (not added to benchmark)")
```

!!! tip "Choosing tolerance values"
    Use `tolerance = 0` for exact counts (chromosomes, electrons). Use absolute tolerance for physical measurements with known precision (temperature, mass). Use percentage-based tolerance for values that span wide ranges (e.g., `tolerance_pct = 10` to accept within 10%).

### Question 4: Regex in verify() — PubMed ID Format

When the judge might return a value in various formats, use `re.search()` in `verify()` to extract the relevant portion before comparison.

```python
q4_id = benchmark.add_question(
    question="What is the PubMed ID for the original CRISPR-Cas9 paper by Jinek et al. (2012)?",
    raw_answer="PMID: 22745249",
)

class Answer(BaseAnswer):
    pubmed_id: str = Field(
        description=(
            "The PubMed identifier for the paper, in the format 'PMID: ' followed "
            "by digits (e.g., 'PMID: 22745249'). If the response gives only the "
            "numeric ID without the PMID prefix, extract just the digits."
        )
    )

    def model_post_init(self, __context):
        self.correct = {"pubmed_id": "22745249"}

    def verify(self) -> bool:
        # Extract digits from whatever format the judge returned
        match = re.search(r"\d{6,9}", self.pubmed_id)
        if not match:
            return False
        return match.group() == self.correct["pubmed_id"]

benchmark.update_template(q4_id, Answer)
print(f"Q4 added with template: {q4_id[:50]}...")
```

Using `re.search()` in `verify()` is useful when the judge might extract the value in various formats. Here, whether the judge returns "22745249", "PMID: 22745249", or "PMID:22745249", the regex extracts the numeric portion for comparison.

### Question 5: Multi-Field with Partial Credit — Drug Mechanism

For questions with multiple dimensions to evaluate, use several fields with both `verify()` (all-or-nothing) and `verify_granular()` (partial credit). Extract each check into a private helper method for clarity.

```python
q5_id = benchmark.add_question(
    question="Describe the mechanism of action of imatinib.",
    raw_answer="Imatinib is a tyrosine kinase inhibitor that selectively targets BCR-ABL.",
)

class Answer(BaseAnswer):
    drug_class: str = Field(
        description=(
            "The pharmacological class of imatinib as described in the response "
            "(e.g., 'tyrosine kinase inhibitor'). Normalize to lowercase."
        )
    )
    primary_target: str = Field(
        description=(
            "The primary molecular target of imatinib as stated in the response. "
            "Use the standard gene symbol in uppercase (e.g., 'BCR-ABL')."
        )
    )
    mentions_selectivity: bool = Field(
        description=(
            "True if the response mentions that imatinib is selective or specific "
            "for its target, or describes which kinases it does/does not inhibit. "
            "False if selectivity is not discussed."
        )
    )

    def model_post_init(self, __context):
        self.correct = {
            "drug_class": "tyrosine kinase inhibitor",
            "primary_target": "BCR-ABL",
            "mentions_selectivity": True,
        }

    def _check_drug_class(self) -> bool:
        return "tyrosine kinase inhibitor" in self.drug_class.strip().lower()

    def _check_target(self) -> bool:
        extracted = self.primary_target.strip().upper().replace(" ", "")
        return extracted in ("BCR-ABL", "BCRABL", "BCR-ABL1", "BCRABL1")

    def _check_selectivity(self) -> bool:
        return self.mentions_selectivity == self.correct["mentions_selectivity"]

    def verify(self) -> bool:
        return (
            self._check_drug_class()
            and self._check_target()
            and self._check_selectivity()
        )

    def verify_granular(self) -> float:
        checks = [
            self._check_drug_class(),
            self._check_target(),
            self._check_selectivity(),
        ]
        return sum(checks) / len(checks)

benchmark.update_template(q5_id, Answer)
print(f"Q5 added with template: {q5_id[:50]}...")
```

The pipeline calls `verify()` for the pass/fail result used in scoring. `verify_granular()` provides a 0.0--1.0 score for finer-grained analysis — here, getting 2 out of 3 checks correct yields 0.67 instead of a flat failure.

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

| Pattern | Use When | Example |
|---------|----------|---------|
| Boolean decomposition | Known concepts to check for | Drug targets, pathway members |
| String normalization | Single extractable value | Gene symbols, chemical names |
| Numeric tolerance | Measurements or counts | Boiling points, chromosome counts |
| Regex in `verify()` | Variable formats in judge output | IDs, codes, formatted values |
| Multi-field + partial credit | Multiple dimensions to evaluate | Drug mechanisms, multi-part answers |

---

## Next Steps

- [Full Evaluation Benchmark](full-evaluation-benchmark.md) — Add rubric traits for quality assessment
- [Quality Assessment](quality-assessment-benchmark.md) — Rubric-only evaluation without templates
- [Scaled Authoring](scaled-authoring.md) — Bulk workflows and auto-generation
- [Answer Templates](../../core_concepts/answer-templates.md) — Deep dive into template concepts
