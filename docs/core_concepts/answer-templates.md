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
2. **`model_post_init`** to set ground truth values (and optionally `self.regex` for pattern-matching checks)
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

## Why These Three Components?

You might wonder why templates need these specific pieces, and why they're shaped this way.

### Why `model_post_init` for ground truth

The Judge LLM receives the template's JSON schema (field names, types, and descriptions) to know what to extract. Ground truth needs to stay out of that schema so the judge acts as a pure parser, not a correctness checker. If the expected answers were regular Pydantic fields, they'd show up in the schema and bias the judge.

`model_post_init` is Pydantic v2's hook that runs right after an instance is created. By the time it fires, the judge's extracted values are already in the fields, so you can safely attach the expected answers alongside them in `self.correct`. This works because `BaseAnswer` uses `extra="allow"`, letting Pydantic store the `correct` dictionary as an extra attribute without it appearing in the schema.

### Why `verify` as a user-defined method

The judge's job is to **extract** information from a response (fill in the schema). Whether that information is *correct* is a separate question, and one that should be answered by deterministic code, not another LLM call. That's what `verify()` does.

Because `verify()` is plain Python, you get:

- **Reproducibility**: the same extracted values always produce the same verdict, no matter what judge model you use
- **Transparency**: anyone can read the code and see exactly what counts as "correct"
- **Testability**: create an instance with sample values, call `verify()`, and check the result locally without any LLM

### Validation

`BaseAnswer` itself doesn't enforce that your template defines `verify()` or `model_post_init`. Instead, the pipeline validates templates before running them: stage 1 (`ValidateTemplate`) checks that `verify()` exists and is callable, and that `model_post_init` sets `self.correct` as a dictionary. If something is missing, you get a clear error before any LLM calls happen. The one exception is **regex-only templates** (no user-defined fields): `verify()` is optional since regex is the sole evaluation.

`verify_granular()` and `verify_regex()` are fully optional. The pipeline only calls them if your template defines `verify_granular()` or sets `self.regex`, respectively.

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

## Template Patterns

Five patterns form a toolkit for factual verification. Most real benchmarks use a mix: boolean fields for presence checks, string fields for extractable values, and numeric fields for measurements. Start with the simplest pattern that handles your ground truth, and reach for more complex ones only when needed.

For step-by-step examples of implementing each pattern inside a benchmark, see [Factual QA Benchmark](../workflows/creating-benchmarks/factual-qa-benchmark.md).

### Boolean Check

A boolean field checks whether a concept is present in the response, delegating synonym handling to the judge through the field description. No string matching needed — `verify()` simply returns the boolean field.

```python
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


parsed = Answer(identifies_tp53=True)
print(f"Correct answer verified: {parsed.verify()}")

parsed_wrong = Answer(identifies_tp53=False)
print(f"Wrong answer verified:   {parsed_wrong.verify()}")
```

A boolean check avoids string matching entirely. Instead of extracting "TP53" as text and comparing it, we ask the judge "Did the response identify TP53 as the most common?" This is more reliable because the judge handles synonyms (TP53, p53, tumor protein p53) through its description.

The key design decision is in the field description: it must be specific enough to disambiguate edge cases. The `identifies_tp53` description explicitly states that merely listing TP53 alongside other cancer genes should return False — the response must single it out as the *most commonly mutated*. Without this, the judge would likely return True for "TP53, KRAS, and PIK3CA are frequently mutated in cancer," which mentions TP53 but doesn't actually answer the question. Invest time in writing precise descriptions; they are the instructions your judge LLM follows.

Because `verify()` returns the field directly, `model_post_init` can be omitted — the expected answer is embedded in the field description itself. This makes boolean the simplest template pattern, but comes with an anchoring tradeoff: the judge sees the expected answer in the description. See [Ground Truth Exposure](#ground-truth-exposure-and-judge-anchoring) for when this matters.

<div class="admonition tip">
<p class="admonition-title">When to use boolean checks</p>
<p>Prefer boolean fields when you need to check for the presence of a concept that has multiple valid surface forms. The judge handles normalization so <code>verify()</code> stays trivial. For checking multiple concepts independently with partial credit, see <a href="#multi-field-with-partial-credit">Multi-Field with Partial Credit</a>.</p>
</div>

### String Extraction

Using `str` fields tells the judge to extract a specific piece of text. The judge acts as a *parser*: the field description guides what to look for and specifies the expected format, but the judge chooses the value based on what the response actually says. The correctness check happens in `verify()`, which the judge never sees.

This keeps the judge unbiased -- the description specifies *what kind of value* to extract without revealing what the correct answer is. The tradeoff: since free text is more variable than a boolean, you need normalization logic in `verify()` to handle formatting differences.

```python
class Answer(BaseAnswer):
    blood_type: str = Field(
        description=(
            "The blood type stated in the response as the most common worldwide. "
            "Use standard notation: uppercase letter followed by '+' or '-' "
            "(e.g., 'O+' not 'O positive' or 'type O'). If the response uses "
            "the full name ('O positive'), normalize to shorthand ('O+')."
        )
    )

    def model_post_init(self, __context):
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


# Various formats the judge might return — normalization handles them
for value in ["O+", "O positive", "o+", "O POSITIVE"]:
    parsed = Answer(blood_type=value)
    print(f"{value!r:>14} -> verify(): {parsed.verify()}")
```

Even though the description asks for standard notation, the judge may return "O positive" or "o+". Handling these variations in `verify()` is expected and keeps the correctness logic deterministic.

<div class="admonition tip">
<p class="admonition-title">When to use string extraction</p>
<p>Prefer string fields when you need the actual value for downstream analysis, want to prevent judge anchoring on ground truth, or need programmatic control over matching (regex, substring, normalization). See <a href="#ground-truth-exposure-and-judge-anchoring">Ground Truth Exposure</a> for the tradeoff with boolean fields.</p>
</div>

### Numeric Tolerance

Because `verify()` is plain Python, you can implement any verification logic you need. Numeric tolerance is a good example: the judge extracts a raw number, and your code decides what counts as "close enough." A body temperature of 37.2 degrees C is clinically normal even though the textbook value is 37.0 degrees C. Rather than trying to encode tolerance rules in a prompt, you write a programmatic comparison with an explicit margin.

```python
class Answer(BaseAnswer):
    temperature_celsius: float = Field(
        description=(
            "The normal body temperature stated in the response, in degrees "
            "Celsius. If the response gives the value in Fahrenheit (e.g., "
            "98.6 F), extract the Celsius equivalent. If a range is given "
            "(e.g., 36.5-37.5), extract the midpoint."
        )
    )

    def model_post_init(self, __context):
        self.correct = {"temperature_celsius": 37.0}
        self.tolerance = 0.5  # Accept 36.5-37.5, reflecting natural variation

    def verify(self) -> bool:
        return abs(self.temperature_celsius - self.correct["temperature_celsius"]) <= self.tolerance


# Various temperatures within and outside tolerance
for temp in [37.0, 36.8, 37.5, 36.0, 38.0]:
    parsed = Answer(temperature_celsius=temp)
    print(f"{temp} C -> verify(): {parsed.verify()}")
```

For exact counts where only one value is correct, set `tolerance = 0`:

```python
class Answer(BaseAnswer):
    pair_count: int = Field(
        description=(
            "The number of chromosome pairs in a normal human somatic cell, "
            "as a whole number. If the response gives the total count (e.g., 46), "
            "extract the number of pairs (23)."
        )
    )

    def model_post_init(self, __context):
        self.correct = {"pair_count": 23}
        self.tolerance = 0  # Exact match: only 23 is correct

    def verify(self) -> bool:
        return abs(self.pair_count - self.correct["pair_count"]) <= self.tolerance


parsed = Answer(pair_count=23)
print(f"23 pairs -> verify(): {parsed.verify()}")
parsed_wrong = Answer(pair_count=46)
print(f"46 pairs -> verify(): {parsed_wrong.verify()}")
```

<div class="admonition tip">
<p class="admonition-title">Choosing tolerance values</p>
<p>Use <code>tolerance = 0</code> for exact counts (chromosomes, electrons). Use absolute tolerance for physical measurements with known precision (body temperature, boiling points). Use percentage-based tolerance for values that span wide ranges (e.g., <code>tolerance_pct = 10</code> to accept within 10%).</p>
</div>

### Regex Checks

Templates can define regex patterns via `self.regex` that the pipeline checks directly against the **raw LLM response trace** -- the full text the answering model produced, before any parsing.

Sometimes you may not want -- or may not be able to -- use an LLM judge for evaluation. Classical benchmarks often rely on static pattern matching: regex against raw outputs, with no model-based parsing involved. `self.regex` brings this approach into karenina: instead of writing standalone regex scripts outside the framework, you declare named patterns inside the template and the pipeline runs them as part of the standard verification flow. This lets you run classical pattern-matching benchmarks through karenina's pipeline, benefiting from its infrastructure (checkpoints, results, DataFrames, multi-model comparison) without requiring a judge LLM.

When used alongside `verify()`, the pipeline ANDs both results -- both `verify()` and `verify_regex()` must pass for the template to succeed. When used alone (with `verify()` returning `True`), regex checks become the sole evaluation mechanism, giving you a pure pattern-matching benchmark.

**Regex-only templates** (no user-defined fields) are detected automatically: the pipeline skips LLM parsing entirely, so no parsing model is needed. Since there are no fields to check, `verify()` is optional -- if omitted, field verification defaults to `True` and regex is the sole evaluation.

Set `self.regex` in `model_post_init` (alongside `self.correct` if your template also has fields). Each entry is a named check with a `pattern`, an `expected` value, and a `match_type` that controls comparison:

```python
class Answer(BaseAnswer):
    discovery_date: str = Field(
        description=(
            "The date of penicillin's discovery as stated in the response. "
            "Extract the full date if available, otherwise the year alone."
        )
    )

    def model_post_init(self, __context):
        self.correct = {"discovery_date": "1928"}
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

    def verify(self) -> bool:
        return "1928" in self.discovery_date


# verify() checks the parsed field
parsed = Answer(discovery_date="1928")
print(f"verify(): {parsed.verify()}")
```

The pipeline calls `verify_regex()` automatically during the VerifyTemplate stage, passing the raw response trace. The result is ANDed with `verify()` -- both must pass for the template to succeed. If no `self.regex` is defined, the check passes trivially. In the example above, `verify()` passes because the parsed field contains "1928", but the template would still fail if the raw response didn't match both regex patterns (the year and Fleming's name).

Four match types are available:

| `match_type` | `expected` type | Passes when |
|-------------|----------------|-------------|
| `"exact"` | `str` | Exactly one match, equals `expected` |
| `"contains"` | `str` | `expected` is among the matches (for alternation patterns) |
| `"count"` | `int` | Number of matches equals `expected` |
| `"all"` | `list[str]` | All items in `expected` found in matches |

```python
class Answer(BaseAnswer):
    def model_post_init(self, __context):
        self.regex = {
            # contains: check which alternation matched
            "has_mechanism_keyword": {
                "pattern": r"\b(activates|inhibits|blocks)\b",
                "expected": "inhibits",
                "match_type": "contains",
            },
            # count: verify citation count
            "has_three_citations": {
                "pattern": r"\[\d+\]",
                "expected": 3,
                "match_type": "count",
            },
        }


# No fields, no verify() — regex is the sole evaluation
# Pipeline skips LLM parsing automatically
parsed = Answer()
print(f"verify_regex(): {parsed.verify_regex('The drug inhibits the target [1] [2] [3]')}")
```

<div class="admonition tip">
<p class="admonition-title">Regex checks vs. regex in verify()</p>
<p><code>self.regex</code> checks patterns in the <strong>raw response trace</strong> -- what the answering model actually said. <code>re.search()</code> in <code>verify()</code> normalizes <strong>parsed field values</strong> -- what the judge extracted. Use <code>self.regex</code> when you need to verify that certain patterns appear in the original response. Use <code>re.search()</code> in <code>verify()</code> when the judge returns a value in variable format and you need to extract the relevant portion before comparison.</p>
</div>

### Multi-Field with Partial Credit

For questions with multiple dimensions -- where you want to check delivery method *and* target protein *and* immune response independently -- use multiple fields with both `verify()` (all-or-nothing) and `verify_granular()` (partial credit scoring from 0.0 to 1.0).

Extracting each check into a private `_check_*` method keeps both `verify()` and `verify_granular()` readable and ensures the logic is defined once. This pattern is recommended whenever you have three or more fields to verify.

```python
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

    def model_post_init(self, __context):
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


# All fields correct
parsed = Answer(
    delivery_mechanism="mRNA instructions",
    target_protein="spike protein",
    mentions_immune_response=True,
)
print(f"verify():         {parsed.verify()}")
print(f"verify_granular(): {parsed.verify_granular():.2f}")

# 2 out of 3 correct — verify fails, but granular gives partial credit
parsed2 = Answer(
    delivery_mechanism="mRNA instructions",
    target_protein="wrong protein",
    mentions_immune_response=True,
)
print(f"\nverify():         {parsed2.verify()}")
print(f"verify_granular(): {parsed2.verify_granular():.2f}")
```

The pipeline calls `verify()` for the pass/fail result used in scoring. `verify_granular()` provides a 0.0 to 1.0 score for finer-grained analysis -- getting 2 out of 3 checks correct yields 0.67 instead of a flat failure.

### Pattern Summary

| Pattern | Use When | Don't Use When | Example |
|---------|----------|----------------|---------|
| Boolean check | Known concept, multiple synonyms | You need the extracted value | Gene identification, pathway presence |
| String extraction | Strict matching with programmatic control | Only need presence check (use boolean) | Blood types, gene symbols |
| Numeric tolerance | Measurements or counts | Value is categorical, not numeric | Body temperature, chromosome counts |
| Regex checks (`self.regex`) | Pattern must appear in raw response; no fields = no LLM judge needed | Only need to normalize parsed field values | Year mentions, keyword presence, citation counts |
| Multi-field + partial credit | Multiple dimensions, want granular scoring | Single-answer questions | Mechanisms, multi-part descriptions |

## Writing Good Field Descriptions

### What the Judge Sees

When the Judge LLM parses a response, it receives the JSON schema derived from your template -- field names, types, and descriptions. The system prompt tells it: *"Each field's description is authoritative for what and how to extract. Follow field descriptions precisely."* This means your description **is** your specification. A vague description produces unreliable extractions because the judge has no other guidance for what you want.

### Ground Truth Exposure and Judge Anchoring

The choice between boolean and string fields has a subtle but important implication for how the judge LLM behaves.

With **boolean fields**, it is common practice to embed the ground truth in the description: "True if the response identifies BCL2 as the target." This is convenient (no normalization needed, trivial `verify()`), but the judge sees the expected answer and may **anchor** on it rather than faithfully assessing what the response actually says.

With **string fields**, the description tells the judge *what kind of value* to extract ("The protein target mentioned in the response") without revealing what the correct answer is. The judge acts as a **parser**: it extracts what it finds, and `verify()` handles the correctness check programmatically. This keeps the judge unbiased, but requires normalization in `verify()` since free text is more variable than a boolean.

This is a design tradeoff, not a right-or-wrong choice. Boolean fields are faster to write and produce simpler templates. String fields provide a cleaner separation between extraction (judge) and evaluation (`verify()`). Being aware of this distinction helps you make an informed choice for each field in your template.

### Lists vs Individual Boolean Fields

When you expect a set of items (proteins, symptoms, references), you have two approaches.

**`list[str]` extraction** -- Use when the set of expected items is open-ended or you need the actual extracted terms for downstream analysis. Requires normalization in `verify()`.

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

**Individual boolean fields** (often simpler) -- Use when you have a known set of expected items. Each field is an independent, unambiguous check. No string matching is needed in `verify()`, and you get per-item partial credit via `verify_granular()`.

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

Each field is self-contained, accepts known synonyms, and `verify()` is trivial -- `all(...)` for strict, `sum(...)/len(...)` for partial credit. The boolean approach avoids set comparison, string normalization, and gives you granular per-item results.

### Anatomy of a Good Description

Every field description should address four elements:

- **What to extract**: What specific information, not just "the answer"
- **Format expectations**: How to represent it -- case, notation, units, naming standard
- **Scope boundaries**: What counts and what doesn't -- context restrictions, inclusion/exclusion rules
- **Disambiguation**: How to handle ambiguity -- multiple candidates, indirect mentions, edge cases

## Embedding Check

When `embedding_check_enabled` is set in `VerificationConfig`, the pipeline runs a **semantic similarity check** (stage 9) after `verify()`. This uses a SentenceTransformer model to compare the raw LLM response against the expected answer, providing a secondary signal when string-based verification is too strict.

The embedding check is configured via three settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `embedding_check_enabled` | `False` | Enable the embedding similarity check |
| `embedding_check_model` | `all-MiniLM-L6-v2` | SentenceTransformer model name |
| `embedding_check_threshold` | `0.85` | Similarity threshold (0.0-1.0) |

The embedding check result is stored alongside the template result -- it does not override `verify()` but provides additional context for analysis.

## Next Steps

- [Rubrics](rubrics/index.md) -- Assess response quality beyond correctness
- [Evaluation Modes](evaluation-modes.md) -- Choose between template-only, rubric-only, or both
- [Factual QA Benchmark](../workflows/creating-benchmarks/factual-qa-benchmark.md) -- Step-by-step implementation of these patterns in a benchmark
- [Philosophy](../home/philosophy.md) -- Why the LLM-as-judge approach works
