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

### Think of It Like a Form

If you have ever filled out a standardized form — a lab report, a clinical intake sheet, a grant application — you already understand how templates work. The analogy maps directly:

| Form Concept | Template Equivalent | Who Is Responsible |
|-------------|---------------------|-------------------|
| The blank form with labeled fields and instructions | The template's fields and their descriptions | You, the benchmark author |
| The person reading a letter and filling in the form | The Judge LLM | The system (automatic) |
| The answer key stored in a locked drawer | `model_post_init` setting `self.correct` | You, the benchmark author |
| The clerk who checks the filled form against the answer key | The `verify()` method | You, the benchmark author |

This analogy captures something important: **the person filling in the form and the person checking it are not the same**. The form-filler (the judge) reads a document and writes down what they find. The checker (`verify()`) compares what was written down against known correct answers. Neither one needs to do the other's job.

This separation is deliberate. It keeps each step simple, reliable, and independently testable. We will come back to this idea throughout the document, because it is the key to writing good templates.

### Three Parties, Three Jobs

A template is a contract between three participants, each with a single, well-defined responsibility:

1. **You (the benchmark author)** design the form: what fields to extract, what the correct answers are, and how to check them
2. **The Judge LLM** reads the answering model's response and fills in the form — it extracts information, nothing more
3. **The `verify()` method** compares the filled-in values against the answer key and returns a pass/fail verdict

No participant oversteps its role. The judge never sees the answer key. The `verify()` method never reads the original response. You, the author, define the rules but never need to run the evaluation yourself. This is what makes the system reproducible: swap out the judge model, and the same form with the same `verify()` logic still produces a deterministic result.

With this mental model in place, let's look at how templates are built in practice.

## Template Structure

Every template inherits from `BaseAnswer` and must be named `Answer`. A template has three components:

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

> All subsequent examples assume these imports:
> ```python
> from pydantic import Field
> from karenina.schemas.entities import BaseAnswer
> ```

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

Now that you have seen what a template looks like, let's step back and understand *why* it is built this way. Each component exists because of a specific design choice, and understanding those choices will help you write better templates. We will use the form analogy from earlier to keep things concrete.

### The answer key stays hidden: `model_post_init`

Remember the form analogy: the answer key is stored in a locked drawer. The person filling in the form (the judge) should never see it — otherwise they might just copy the expected answers instead of honestly reading the document.

In technical terms: the Judge LLM receives the template's JSON schema (field names, types, and descriptions) to know what to extract. If the correct answers were regular fields, they would appear in that schema and the judge could anchor on them instead of faithfully parsing the response. Ground truth needs to stay *out* of the schema so the judge acts as a pure reader, not a correctness checker.

`model_post_init` is the mechanism that makes this possible. It is a hook that runs right after an instance is created. By the time it fires, the judge's extracted values are already in the fields, so you can safely attach the expected answers alongside them in `self.correct`. The answers are there when `verify()` needs them, but invisible to the judge when it is doing its work.

If you are not a Python developer, the key takeaway is simple: **put your correct answers in `model_post_init`, not in the field definitions.** The system handles the rest.

<div class="admonition info">
<p class="admonition-title">What is <code>model_post_init</code>?</p>
<p><code>model_post_init</code> is a standard Pydantic v2 lifecycle method — it runs automatically after the instance is created and all fields are set. The <code>__context</code> parameter is required by Pydantic's signature but unused in templates; treat it as boilerplate you must include. You do not need deep Pydantic knowledge to use it: just define the method, set <code>self.correct</code> (and optionally <code>self.regex</code>), and the framework handles the rest.</p>
</div>

### Extraction and checking are separate jobs: `verify()`

This is the most important design idea in karenina's template system, and where the form analogy is most useful.

Think about what happens with a real standardized form. One person reads the source document and fills in the blanks. A different person — or an automated system — checks the filled-in values against records. The form-filler does not decide whether the answers are correct. The checker does not go back and re-read the original document. Each actor has exactly one job.

Templates work the same way:

- **The Judge LLM fills in the form** — it reads the answering model's response and extracts values into the template's fields. That is *all* it does. It does not evaluate, score, or judge correctness.
- **`verify()` checks the form** — it compares the extracted values against `self.correct` and returns True or False. It never sees the original response. It works with the structured data the judge already extracted.

Why go through this trouble instead of just asking the judge "Was this answer correct?" Because deterministic code is more reliable than asking an LLM to make judgment calls. By separating extraction from evaluation, you get:

- **Reproducibility**: the same extracted values always produce the same verdict, no matter which judge model you use. Run it today, run it next year — same result.
- **Transparency**: anyone can read the `verify()` code and see exactly what counts as "correct." No prompt engineering mysteries.
- **Testability**: you can create a template instance with sample values, call `verify()`, and check the result on your laptop without any LLM, API key, or network connection.

### Validation

You do not need to worry about forgetting a required component. `BaseAnswer` itself does not enforce that your template defines `verify()` or `model_post_init` — but the pipeline validates templates before running them. Stage 1 (`ValidateTemplate`) checks that `verify()` exists and is callable, and that `model_post_init` sets `self.correct` as a dictionary. If something is missing, you get a clear error message before any LLM calls happen.

For regex-only templates and optional methods like `verify_granular()`, see [Regex Checks](#regex-checks) and [Multi-Field with Partial Credit](#multi-field-with-partial-credit).

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
| `Literal[...]` | Fixed categories | Classification labels, mutation types |

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

`Literal` constrains the judge to a fixed set of values — Pydantic generates an `enum` in the JSON schema, so the judge can only return one of the allowed options:

```python
from typing import Literal


class Answer(BaseAnswer):
    mutation_type: Literal["missense", "nonsense", "frameshift", "silent"] = Field(
        description="The type of point mutation described in the response."
    )

    def model_post_init(self, __context):
        self.correct = {"mutation_type": "missense"}

    def verify(self) -> bool:
        return self.mutation_type == self.correct["mutation_type"]


parsed = Answer(mutation_type="missense")
print(f"verify(): {parsed.verify()}")
```

## Template Patterns

Five patterns form a toolkit for factual verification. Most real benchmarks use a mix: boolean fields for presence checks, string fields for extractable values, and numeric fields for measurements. Start with the simplest pattern that handles your ground truth, and reach for more complex ones only when needed.

For step-by-step examples of implementing each pattern inside a benchmark, see [Factual QA Benchmark](../workflows/creating-benchmarks/factual-qa-benchmark.md).

### Boolean Check

In form terms, this is a checkbox: "Check this box if the letter mentions X." The judge reads the response and ticks yes or no. No string matching needed — `verify()` compares the boolean against `self.correct`. The judge handles synonym recognition through the field description, so it can recognize that "p53," "TP53," and "tumor protein p53" all refer to the same thing.

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

    def model_post_init(self, __context):
        self.correct = {"identifies_tp53": True}

    def verify(self) -> bool:
        return self.identifies_tp53 == self.correct["identifies_tp53"]


parsed = Answer(identifies_tp53=True)
print(f"Correct answer verified: {parsed.verify()}")

parsed_wrong = Answer(identifies_tp53=False)
print(f"Wrong answer verified:   {parsed_wrong.verify()}")
```

A boolean check avoids string matching entirely. Instead of extracting "TP53" as text and comparing it, we ask the judge "Did the response identify TP53 as the most common?" This is more reliable because the judge handles synonyms (TP53, p53, tumor protein p53) through its description.

The key design decision is in the field description: it must be specific enough to disambiguate edge cases. The `identifies_tp53` description explicitly states that merely listing TP53 alongside other cancer genes should return False — the response must single it out as the *most commonly mutated*. Without this, the judge would likely return True for "TP53, KRAS, and PIK3CA are frequently mutated in cancer," which mentions TP53 but doesn't actually answer the question. Invest time in writing precise descriptions; they are the instructions your judge LLM follows.

Boolean templates follow the same three-component structure as other patterns. The difference is in `verify()`: instead of string normalization or numeric tolerance, it compares the extracted boolean against `self.correct` directly. This makes boolean the simplest template pattern, but it comes with an anchoring tradeoff: the judge sees the expected answer in the field description. See [Ground Truth Exposure](#ground-truth-exposure-and-judge-anchoring) for when this matters.

<div class="admonition tip">
<p class="admonition-title">When to use boolean checks</p>
<p>Prefer boolean fields when you need to check for the presence of a concept that has multiple valid surface forms. The judge handles normalization so <code>verify()</code> stays trivial. For checking multiple concepts independently with partial credit, see <a href="#multi-field-with-partial-credit">Multi-Field with Partial Credit</a>.</p>
</div>

### String Extraction

In form terms, this is a blank text field: "Write down the blood type mentioned in the letter." The judge writes what it finds without knowing what the correct answer should be. The answer key stays in the locked drawer, and the clerk (`verify()`) does the checking later.

This is the cleanest separation between extraction and evaluation. The description tells the judge *what kind of value* to look for and the expected format, but the judge chooses the value based on what the response actually says. The correctness check happens in `verify()`, which the judge never sees.

The tradeoff: since free text is more variable than a yes/no checkbox, you need normalization logic in `verify()` to handle formatting differences (uppercase vs lowercase, "O positive" vs "O+", and so on).

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

The judge extracts a raw number; `verify()` decides whether it falls within an acceptable range.

This pattern shows the power of having `verify()` as plain code. A body temperature of 37.2 degrees C is clinically normal even though the textbook value is 37.0 degrees C. Rather than trying to encode tolerance rules in a prompt and hoping the LLM applies them consistently, you write a simple programmatic comparison with an explicit margin.

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

Regex patterns search the **raw LLM response trace** — the full text the answering model produced, before any parsing.

Sometimes you may not want — or may not be able to — use an LLM judge for evaluation. Classical benchmarks often rely on static pattern matching: regex against raw outputs, with no model-based parsing involved. `self.regex` brings this approach into karenina: instead of writing standalone regex scripts outside the framework, you declare named patterns inside the template and the pipeline runs them as part of the standard verification flow. This lets you run classical pattern-matching benchmarks through karenina's pipeline, benefiting from its infrastructure (checkpoints, results, DataFrames, multi-model comparison) without requiring a judge LLM.

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

<div class="admonition warning">
<p class="admonition-title">Capture groups change matching behavior</p>
<p>Python's <code>re.findall()</code> returns captured group content instead of full matches when the pattern contains parenthesized groups. For alternation like <code>r"\b(activates|inhibits|blocks)\b"</code>, this works correctly. But for patterns like <code>r"(\d+)\s*(mg|g)"</code>, it returns tuples (<code>[("500", "mg")]</code>), breaking string comparisons. Use non-capturing groups <code>(?:...)</code> when you need grouping but want the full match returned.</p>
</div>

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
result = parsed.verify_regex("The drug inhibits the target [1] [2] [3]")
print(f"verify_regex() passed: {result['success']}")
print(f"Individual checks:     {result['results']}")
```

<div class="admonition tip">
<p class="admonition-title">Regex checks vs. regex in verify()</p>
<p><code>self.regex</code> checks patterns in the <strong>raw response trace</strong> -- what the answering model actually said. <code>re.search()</code> in <code>verify()</code> normalizes <strong>parsed field values</strong> -- what the judge extracted. Use <code>self.regex</code> when you need to verify that certain patterns appear in the original response. Use <code>re.search()</code> in <code>verify()</code> when the judge returns a value in variable format and you need to extract the relevant portion before comparison.</p>
</div>

### Multi-Field with Partial Credit

Sometimes a single question asks about multiple things at once — and getting two out of three right is better than zero. You want to know *which* parts were correct.

For questions with multiple dimensions — where you want to check delivery method *and* target protein *and* immune response independently — use multiple fields with both `verify()` (all-or-nothing) and `verify_granular()` (partial credit scoring from 0.0 to 1.0).

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

## Ground Truth Exposure and Judge Anchoring

The choice between boolean and string fields has a subtle but important implication for how honestly the judge fills in your form.

Think of it this way: imagine your form has a checkbox that says "Check this box if the letter mentions Paris as the capital of France." The form-filler now knows what the expected answer is. They might check the box even if the letter only vaguely alludes to Paris, because the form's wording nudged them. This is **anchoring** — the expected answer influences the extraction.

Now imagine the form has a blank that says "Write down the city named as France's capital." The form-filler writes what they find, without knowing whether you expect Paris, Lyon, or Marseille. The checking happens later, when the clerk compares the filled-in value against the answer key.

This maps directly to template design:

- **Boolean fields** often embed the ground truth in the description: "True if the response identifies BCL2 as the target." This is convenient (no normalization needed, trivial `verify()`), but the judge sees the expected answer and may anchor on it rather than faithfully assessing what the response actually says.
- **String fields** tell the judge *what kind of value* to extract ("The protein target mentioned in the response") without revealing what the correct answer is. The judge acts as a pure reader, and `verify()` handles the correctness check separately. This keeps the judge unbiased, but requires normalization logic in `verify()` since free text is more variable than a boolean.

This is a design tradeoff, not a right-or-wrong choice. Boolean fields are faster to write and produce simpler templates. String fields provide a cleaner separation between extraction (the judge) and evaluation (`verify()`). Being aware of this distinction helps you make an informed choice for each field in your template.

| | Boolean field | String field |
|--|--------------|-------------|
| Judge sees correct answer? | Yes (in description) | No |
| Anchoring risk | Higher | Lower |
| `verify()` complexity | Simple equality check | Normalization required |
| Best for | Known concepts with synonyms | Rigorous benchmarks, unbiased evaluation |

## Writing Good Field Descriptions

### What the Judge Sees

Remember: the judge is the person reading the letter and filling in your form. The *only* guidance it has is the form itself — the field names, types, and descriptions from your template's JSON schema. The system prompt tells the judge: *"Each field's description is authoritative for what and how to extract. Follow field descriptions precisely."*

This means your field description **is** your specification. If the instructions next to a blank on your form are vague ("write the answer here"), the form-filler will not know what you want. If they are precise ("write the patient's systolic blood pressure in mmHg, as stated in the clinical note"), the form-filler knows exactly what to look for and how to write it down. Invest time in your descriptions — they are the single most important part of your template, because they are the only thing the judge has to work with.

For the tradeoff between boolean and string fields, see [Ground Truth Exposure and Judge Anchoring](#ground-truth-exposure-and-judge-anchoring) above.

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

    def model_post_init(self, __context):
        self.correct = {"signaling_proteins": {"EGFR", "KRAS", "BRAF"}}

    def verify(self) -> bool:
        extracted = {p.strip().upper() for p in self.signaling_proteins}
        return extracted == self.correct["signaling_proteins"]
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

Think of a field description as the instructions printed next to a blank on a well-designed form. Good instructions leave no room for guessing. Bad instructions lead to inconsistent or wrong answers — not because the form-filler is careless, but because they were not told what you needed.

Every field description should address four elements:

- **What to extract**: What specific information are you looking for? Not just "the answer," but "the protein target of the drug" or "the year penicillin was discovered"
- **Format expectations**: How should the form-filler write it down? Uppercase, lowercase, specific notation? ("Use standard ABO notation: 'O+' not 'O positive'")
- **Scope boundaries**: What counts and what does not? Are you looking for something mentioned anywhere in the response, or only in a specific context? ("Only proteins the response explicitly associates with the pathway, not proteins mentioned in passing")
- **Disambiguation**: What should the form-filler do when the response is ambiguous? Multiple candidates, indirect mentions, edge cases — spell out how to handle them

## Embedding Check

When `embedding_check_enabled` is set in `VerificationConfig`, the pipeline runs a **semantic similarity check** (stage 9) after `verify()`. This uses a SentenceTransformer model to compare the raw LLM response against the expected answer, providing a secondary signal when string-based verification is too strict.

The embedding check is configured via three settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `embedding_check_enabled` | `False` | Enable the embedding similarity check |
| `embedding_check_model` | `all-MiniLM-L6-v2` | SentenceTransformer model name |
| `embedding_check_threshold` | `0.85` | Similarity threshold (0.0-1.0) |

The embedding check result is stored alongside the template result -- it does not override `verify()` but provides additional context for analysis.

## Common Pitfalls

### What happens when things go wrong

| Scenario | What happens | Result |
|----------|-------------|--------|
| `verify()` raises an exception | Pipeline catches it, marks context as error | Not pass or fail — recorded as **error** |
| Judge returns `None` for a `str` field | Pydantic validation fails at parse time | Parsing failure; `verify()` never runs |
| `verify_granular()` raises | Caught and logged as warning | Does NOT fail verification; granular score is absent |
| Template validation fails (missing `verify()`, bad `self.correct`) | Stage 1 error; all subsequent stages skip | Error before any LLM calls |

**Guard against `None` in `verify()`** — If the judge might fail to extract a value, check before using string methods:

```python
def verify(self) -> bool:
    if self.target is None:
        return False
    return self.target.strip().upper() == self.correct["target"].upper()
```

## Next Steps

- [Rubrics](rubrics/index.md) -- Assess response quality beyond correctness
- [Evaluation Modes](evaluation-modes.md) -- Choose between template-only, rubric-only, or both
- [Factual QA Benchmark](../workflows/creating-benchmarks/factual-qa-benchmark.md) -- Step-by-step implementation of these patterns in a benchmark
- [Philosophy](../home/philosophy.md) -- Why the LLM-as-judge approach works
