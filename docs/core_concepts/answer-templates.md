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

# Answer Templates

Answer templates are the primary mechanism for evaluating **correctness** in Karenina. They define what information to extract from an LLM's response and how to verify it against ground truth.

```python tags=["hide-cell"]
# Mock cell: ensures examples execute without live API keys.
# This cell is hidden in rendered documentation.
```

## What Are Templates?

An answer template is a **Pydantic model** that serves two purposes:

1. **Parsing instructions** — Field names and descriptions tell the Judge LLM what to extract from a free-text response
2. **Verification logic** — The `verify()` method programmatically checks extracted values against expected answers

This is the core of Karenina's **LLM-as-judge** approach: the answering model produces natural text, the Judge LLM fills in the template's schema, and the template's code decides if the answer is correct.

```
Response (free text)  →  Judge LLM  →  Filled template  →  verify()  →  True/False
```

## Template Structure

Every template inherits from `BaseAnswer` and must be named `Answer`. A template has three required components:

1. **Fields** with descriptions that guide the Judge LLM
2. **`model_post_init`** to set ground truth values
3. **`verify`** to compare extracted values against ground truth

Here is a simple template that checks whether an LLM correctly identified a drug target:

```python
from pydantic import Field

from karenina.schemas.entities import BaseAnswer


class Answer(BaseAnswer):
    target: str = Field(description="The protein target of the drug mentioned in the response")

    def model_post_init(self, __context):
        self.correct = {"target": "BCL2"}

    def verify(self) -> bool:
        return self.target.strip().upper() == self.correct["target"].upper()
```

Let's walk through what happens when this template is used:

1. The Judge LLM reads the answering model's response and extracts the `target` field based on the description
2. Pydantic creates an `Answer` instance, then `model_post_init` sets the expected value
3. `verify()` compares the extracted target to "BCL2" (case-insensitive)

```python
# Simulate what the Judge LLM produces after parsing a response
parsed = Answer(target="Bcl-2")
print(f"Extracted target: {parsed.target!r}")
print(f"Ground truth:     {parsed.correct['target']!r}")
print(f"Verified:         {parsed.verify()}")
```

## Naming Requirement

All answer template classes **must be named `Answer`**. The pipeline looks for this exact class name when executing template code.

```python
# Correct
class Answer(BaseAnswer):
    value: str = Field(description="The answer value")

    def model_post_init(self, __context):
        self.correct = {"value": "42"}

    def verify(self) -> bool:
        return self.value.strip() == self.correct["value"]


# This works:
a = Answer(value="42")
print(f"verify(): {a.verify()}")
```

## Field Types

Template fields can use any type that Pydantic supports. The field type guides both the Judge LLM's parsing and the verification logic.

| Type | Use Case | Example |
|------|----------|---------|
| `str` | Names, terms, identifiers | Drug target, gene symbol |
| `int` | Counts, quantities | Number of chromosomes |
| `float` | Measurements, scores | Temperature, percentage |
| `bool` | Yes/no judgments | "Does the response mention X?" |
| `list[str]` | Multiple items | List of proteins, symptoms |

Here is a multi-field example that extracts and verifies two pieces of information:

```python
class Answer(BaseAnswer):
    element: str = Field(description="The chemical element name mentioned in the response")
    atomic_number: int = Field(description="The atomic number stated in the response")

    def model_post_init(self, __context):
        self.correct = {"element": "oxygen", "atomic_number": 8}

    def verify(self) -> bool:
        name_ok = self.element.strip().lower() == self.correct["element"]
        number_ok = self.atomic_number == self.correct["atomic_number"]
        return name_ok and number_ok


# Both fields must match for verification to pass
parsed = Answer(element="Oxygen", atomic_number=8)
print(f"Element correct:  {parsed.element.strip().lower() == parsed.correct['element']}")
print(f"Number correct:   {parsed.atomic_number == parsed.correct['atomic_number']}")
print(f"Overall verify(): {parsed.verify()}")
```

## Partial Credit with verify_granular

For multi-field templates, you can implement `verify_granular()` to return a score between 0.0 and 1.0 representing the fraction of correct fields:

```python
class Answer(BaseAnswer):
    capital: str = Field(description="The capital city mentioned")
    population: int = Field(description="The population figure stated")
    continent: str = Field(description="The continent mentioned")

    def model_post_init(self, __context):
        self.correct = {
            "capital": "paris",
            "population": 2161000,
            "continent": "europe",
        }

    def verify(self) -> bool:
        return (
            self.capital.strip().lower() == self.correct["capital"]
            and self.population == self.correct["population"]
            and self.continent.strip().lower() == self.correct["continent"]
        )

    def verify_granular(self) -> float:
        correct_count = 0
        if self.capital.strip().lower() == self.correct["capital"]:
            correct_count += 1
        if self.population == self.correct["population"]:
            correct_count += 1
        if self.continent.strip().lower() == self.correct["continent"]:
            correct_count += 1
        return correct_count / 3


# 2 out of 3 fields correct
parsed = Answer(capital="Paris", population=999, continent="Europe")
print(f"verify():         {parsed.verify()}")
print(f"verify_granular(): {parsed.verify_granular():.2f}")
```

## Embedding Check

When `embedding_check_enabled` is set in `VerificationConfig`, the pipeline runs a **semantic similarity check** (stage 9) after `verify()`. This uses a SentenceTransformer model to compare the raw LLM response against the expected answer, providing a secondary signal when string-based verification is too strict.

The embedding check is configured via three settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `embedding_check_enabled` | `False` | Enable the embedding similarity check |
| `embedding_check_model` | `all-MiniLM-L6-v2` | SentenceTransformer model name |
| `embedding_check_threshold` | `0.85` | Similarity threshold (0.0-1.0) |

The embedding check result is stored alongside the template result — it does not override `verify()` but provides additional context for analysis.

## Writing Good Field Descriptions

### What the Judge Sees

When the Judge LLM parses a response, it receives the JSON schema derived from your template — field names, types, and descriptions. The system prompt tells it: *"Each field's description is authoritative for what and how to extract. Follow field descriptions precisely."* This means your description **is** your specification. A vague description produces unreliable extractions because the judge has no other guidance for what you want.

### Two Strategies for Field Design

There are two equally valid approaches to designing template fields. Choose based on what you need from the evaluation.

**Strategy 1: Boolean presence checks** — Use `bool` fields to check whether a concept appears in the response. This sidesteps string matching and normalization entirely. Best when you care about presence or absence, not the exact extracted text.

```python
class Answer(BaseAnswer):
    identifies_bcl2_as_target: bool = Field(
        description=(
            "True if the response identifies BCL2 (including variants like Bcl-2, "
            "BCL-2, or B-cell lymphoma 2) as the direct pharmacological target of "
            "the drug. False if BCL2 is mentioned only as a pathway member or if "
            "a different protein is identified as the primary target."
        )
    )
```

This description specifies accepted name variants, distinguishes direct target from pathway mention, and tells the judge how to handle the ambiguous case (mentioned but not as primary target).

**Strategy 2: String extraction with format expectations** — Use `str` fields when you need the actual extracted text for downstream analysis or complex verification logic. When using strings, always specify the expected format.

```python
class Answer(BaseAnswer):
    gene_symbol: str = Field(
        description=(
            "The official HGNC gene symbol for the drug's molecular target, as stated "
            "or implied in the response. Use uppercase without hyphens (e.g., 'BCL2' "
            "not 'Bcl-2', 'TP53' not 'p53'). If multiple targets are mentioned, "
            "extract the primary target — the one described as the direct binding partner."
        )
    )
```

This description specifies the format (HGNC standard, uppercase, no hyphens), provides normalization examples, and handles multi-target disambiguation.

### Lists vs Boolean Decomposition

When you expect a set of items (proteins, symptoms, references), you have two approaches.

**`list[str]` extraction** — Use when the set of expected items is open-ended or you need the actual extracted terms for downstream analysis. Requires normalization in `verify()`.

```python
class Answer(BaseAnswer):
    signaling_proteins: list[str] = Field(
        description=(
            "All proteins explicitly named as part of the signaling pathway in the "
            "response. Use standard gene symbols in uppercase (e.g., 'EGFR', 'KRAS'). "
            "Include only proteins the response explicitly associates with the pathway, "
            "not proteins mentioned in other contexts."
        )
    )
```

**Boolean decomposition** (often simpler) — Use when you have a known set of expected items. Each field is an independent, unambiguous check. No string matching is needed in `verify()`, and you get per-item partial credit automatically.

```python
class Answer(BaseAnswer):
    mentions_egfr: bool = Field(
        description=(
            "True if the response names EGFR (or ErbB1, HER1) as part of the "
            "signaling pathway. False if EGFR is not mentioned or is mentioned "
            "only outside the pathway context."
        )
    )
    mentions_kras: bool = Field(
        description=(
            "True if the response names KRAS (or K-Ras, K-RAS) as part of the "
            "signaling pathway. False otherwise."
        )
    )
    mentions_braf: bool = Field(
        description=(
            "True if the response names BRAF (or B-Raf, B-RAF) as part of the "
            "signaling pathway. False otherwise."
        )
    )
```

Each field is self-contained, accepts known synonyms, and `verify()` is trivial — `all(...)` for strict, `sum(...)/len(...)` for partial credit. The boolean approach avoids set comparison, string normalization, and gives you granular per-item results.

### Anatomy of a Good Description

Every field description should address four elements:

- **What to extract**: What specific information, not just "the answer"
- **Format expectations**: How to represent it — case, notation, units, naming standard
- **Scope boundaries**: What counts and what doesn't — context restrictions, inclusion/exclusion rules
- **Disambiguation**: How to handle ambiguity — multiple candidates, indirect mentions, edge cases

## Next Steps

- [Rubrics](rubrics/index.md) — Assess response quality beyond correctness
- [Evaluation Modes](evaluation-modes.md) — Choose between template-only, rubric-only, or both
- [Creating Benchmarks](../05-creating-benchmarks/index.md) — Build benchmarks with templates and questions
- [Philosophy](../home/philosophy.md) — Why the LLM-as-judge approach works
