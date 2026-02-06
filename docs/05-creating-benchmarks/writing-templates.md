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

# Writing Custom Templates

Custom templates give you full control over evaluation logic. While [automatic generation](generating-templates.md) works well for straightforward questions, custom templates are essential when you need complex comparison logic, domain-specific tolerance, or multi-step verification.

This page covers practical patterns for writing templates by hand. For the conceptual foundation (what templates are, field types, `verify()` basics), see [Answer Templates](../04-core-concepts/answer-templates.md).

```python tags=["hide-cell"]
# Setup cell: ensures examples execute without live API keys.
# This cell is hidden in rendered documentation.
from karenina.schemas.entities import BaseAnswer
from pydantic import Field
```

## When to Write Custom Templates

Write templates manually when:

- **Fuzzy matching** is needed (e.g., "BCL2" should match "Bcl-2" and "BCL-2")
- **Numeric tolerance** applies (e.g., within 5% of the expected value)
- **List comparison** requires set semantics, not exact ordering
- **Conditional logic** depends on which fields are populated
- **Domain-specific normalization** is needed before comparison

## Case-Insensitive String Matching

The most common pattern normalizes strings before comparison:

```python
class Answer(BaseAnswer):
    gene_symbol: str = Field(
        description="The gene symbol mentioned in the response"
    )

    def model_post_init(self, __context):
        self.correct = {"gene_symbol": "TP53"}

    def verify(self) -> bool:
        return self.gene_symbol.strip().upper() == self.correct["gene_symbol"].upper()


# Handles variations: "tp53", " TP53 ", "Tp53"
for variant in ["tp53", " TP53 ", "Tp53"]:
    parsed = Answer(gene_symbol=variant)
    print(f"{variant!r:12s} → verify(): {parsed.verify()}")
```

## Numeric Tolerance

For questions involving measurements or estimates, exact comparison is often too strict:

```python
class Answer(BaseAnswer):
    temperature: float = Field(
        description="The boiling point temperature in degrees Celsius"
    )

    def model_post_init(self, __context):
        self.correct = {"temperature": 100.0}
        self.tolerance = 0.5  # Accept within ±0.5°C

    def verify(self) -> bool:
        return abs(self.temperature - self.correct["temperature"]) <= self.tolerance


# Exact and approximate values both pass
for temp in [100.0, 99.8, 100.3, 101.0]:
    parsed = Answer(temperature=temp)
    print(f"{temp:6.1f}°C → verify(): {parsed.verify()}")
```

## Percentage-Based Tolerance

For values that span wide ranges, use relative tolerance instead of absolute:

```python
class Answer(BaseAnswer):
    population: int = Field(
        description="The estimated population of the city"
    )

    def model_post_init(self, __context):
        self.correct = {"population": 8_336_817}
        self.tolerance_pct = 10  # Accept within 10%

    def verify(self) -> bool:
        expected = self.correct["population"]
        threshold = expected * (self.tolerance_pct / 100)
        return abs(self.population - expected) <= threshold


# Values within 10% of 8,336,817 pass
for pop in [8_336_817, 8_000_000, 9_000_000, 7_000_000]:
    parsed = Answer(population=pop)
    print(f"{pop:>10,} → verify(): {parsed.verify()}")
```

## Set-Based List Comparison

When order doesn't matter, compare lists as sets:

```python
class Answer(BaseAnswer):
    symptoms: list[str] = Field(
        description="The symptoms of the condition listed in the response"
    )

    def model_post_init(self, __context):
        self.correct = {"symptoms": ["fever", "cough", "fatigue"]}

    def verify(self) -> bool:
        extracted = {s.strip().lower() for s in self.symptoms}
        expected = {s.lower() for s in self.correct["symptoms"]}
        return extracted == expected


# Order doesn't matter; normalization handles case
parsed = Answer(symptoms=["Fatigue", "fever", "Cough"])
print(f"Extracted: {parsed.symptoms}")
print(f"verify():  {parsed.verify()}")

# Missing or extra items fail
parsed2 = Answer(symptoms=["fever", "cough"])
print(f"\nMissing item: {parsed2.symptoms}")
print(f"verify():     {parsed2.verify()}")
```

## Subset Matching

Sometimes you want to check that the response includes at least the expected items, but extra items are acceptable:

```python
class Answer(BaseAnswer):
    proteins: list[str] = Field(
        description="Proteins involved in the signaling pathway mentioned in the response"
    )

    def model_post_init(self, __context):
        self.correct = {"required_proteins": ["EGFR", "RAS", "RAF"]}

    def verify(self) -> bool:
        extracted = {p.strip().upper() for p in self.proteins}
        required = {p.upper() for p in self.correct["required_proteins"]}
        return required.issubset(extracted)


# Extra proteins are fine as long as required ones are present
parsed = Answer(proteins=["EGFR", "RAS", "RAF", "MEK", "ERK"])
print(f"Extracted: {parsed.proteins}")
print(f"verify():  {parsed.verify()}")

# Missing a required protein fails
parsed2 = Answer(proteins=["EGFR", "MEK"])
print(f"\nMissing required: {parsed2.proteins}")
print(f"verify():         {parsed2.verify()}")
```

## Multi-Field with Partial Credit

For complex templates with multiple attributes, implement both `verify()` (all-or-nothing) and `verify_granular()` (partial credit):

```python
class Answer(BaseAnswer):
    drug_name: str = Field(
        description="The name of the drug mentioned in the response"
    )
    target: str = Field(
        description="The protein target of the drug"
    )
    mechanism: str = Field(
        description="The mechanism of action (e.g., inhibitor, agonist)"
    )

    def model_post_init(self, __context):
        self.correct = {
            "drug_name": "venetoclax",
            "target": "BCL2",
            "mechanism": "inhibitor",
        }

    def _check_drug_name(self) -> bool:
        return self.drug_name.strip().lower() == self.correct["drug_name"].lower()

    def _check_target(self) -> bool:
        # Normalize common gene symbol variations
        extracted = self.target.strip().upper().replace("-", "").replace("_", "")
        expected = self.correct["target"].upper().replace("-", "").replace("_", "")
        return extracted == expected

    def _check_mechanism(self) -> bool:
        return self.mechanism.strip().lower() == self.correct["mechanism"].lower()

    def verify(self) -> bool:
        return self._check_drug_name() and self._check_target() and self._check_mechanism()

    def verify_granular(self) -> float:
        checks = [self._check_drug_name(), self._check_target(), self._check_mechanism()]
        return sum(checks) / len(checks)


# 2 out of 3 correct → verify() fails, verify_granular() gives partial credit
parsed = Answer(drug_name="Venetoclax", target="Bcl-2", mechanism="agonist")
print(f"Drug:      {parsed._check_drug_name()}")
print(f"Target:    {parsed._check_target()}")
print(f"Mechanism: {parsed._check_mechanism()}")
print(f"verify():          {parsed.verify()}")
print(f"verify_granular(): {parsed.verify_granular():.2f}")
```

## Boolean Attribute Pattern

For rigorous evaluation, use boolean fields that check for concept presence rather than extracting text. This avoids string matching pitfalls entirely:

```python
class Answer(BaseAnswer):
    mentions_bcl2: bool = Field(
        description="True if the response identifies BCL2 (or BCL-2, Bcl-2) as the target"
    )
    mentions_inhibition: bool = Field(
        description="True if the response describes inhibition as the mechanism of action"
    )
    mentions_apoptosis: bool = Field(
        description="True if the response mentions apoptosis or programmed cell death"
    )

    def model_post_init(self, __context):
        self.correct = {
            "mentions_bcl2": True,
            "mentions_inhibition": True,
            "mentions_apoptosis": True,
        }

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


parsed = Answer(mentions_bcl2=True, mentions_inhibition=True, mentions_apoptosis=False)
print(f"verify():          {parsed.verify()}")
print(f"verify_granular(): {parsed.verify_granular():.2f}")
```

## Adding Templates to a Benchmark

Templates are added to benchmarks as **code strings** via `add_question()` or `add_answer_template()`:

```python
from karenina import Benchmark

benchmark = Benchmark.create(name="Custom Templates Example")

# Option 1: Provide template when adding the question
template_code = '''class Answer(BaseAnswer):
    target: str = Field(description="The protein target mentioned")

    def model_post_init(self, __context):
        self.correct = {"target": "BCL2"}

    def verify(self) -> bool:
        return self.target.strip().upper() == self.correct["target"].upper()
'''

question_id = benchmark.add_question(
    question="What is the approved drug target of Venetoclax?",
    raw_answer="BCL2",
    answer_template=template_code,
)
print(f"Question added with template: {question_id[:40]}...")

# Option 2: Add template to an existing question
q2_id = benchmark.add_question(
    question="How many chromosomes are in a human somatic cell?",
    raw_answer="46",
)

count_template = '''class Answer(BaseAnswer):
    count: int = Field(description="The number of chromosomes mentioned")

    def model_post_init(self, __context):
        self.correct = {"count": 46}

    def verify(self) -> bool:
        return self.count == self.correct["count"]
'''

benchmark.add_answer_template(q2_id, count_template)
print(f"Template added to: {q2_id[:40]}...")

# Check status
print(f"\nTotal questions: {benchmark.question_count}")
print(f"With templates:  {len(benchmark.get_finished_templates())}")
```

!!! tip "Code strings vs class objects"
    In notebooks, always use **code strings** for templates. Passing a class object requires `inspect.getsource()` which doesn't work reliably in notebook environments. Code strings work everywhere.

## Template Design Guidelines

**Keep verify() deterministic.** The `verify()` method should always produce the same result for the same input. Avoid randomness, network calls, or time-dependent logic.

**Normalize before comparing.** Strip whitespace, standardize case, and handle common formatting variations (hyphens, underscores, spaces in gene symbols).

**Use helper methods.** For multi-field templates, extract each field check into a private method (e.g., `_check_target()`). This makes `verify_granular()` easy to implement and simplifies debugging.

**Write clear field descriptions.** The Judge LLM only sees the field name, type, and description. A vague description like "The answer" will produce unreliable parsing. Be specific about what to extract and how.

**Test locally before running verification.** Instantiate your template with sample values and call `verify()` and `verify_granular()` to confirm the logic works before adding it to a benchmark.

## Next Steps

- [Generating Templates](generating-templates.md) — Automatic template generation for common question types
- [Defining Rubrics](defining-rubrics.md) — Add quality assessment alongside correctness checks
- [Saving Benchmarks](saving-benchmarks.md) — Save your benchmark with templates to a checkpoint
- [Answer Templates](../04-core-concepts/answer-templates.md) — Conceptual foundation (field types, naming requirement)
- [Running Verification](../06-running-verification/index.md) — Execute verification with your custom templates
