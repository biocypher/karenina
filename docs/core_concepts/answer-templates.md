---
jupyter:
  jupytext:
    formats: docs/core_concepts//md,docs/notebooks/core_concepts//ipynb
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

# Answer Templates

Answer templates are the primary mechanism for evaluating **correctness** in Karenina. They define what information to extract from an LLM's response and how to verify it against ground truth.

```python tags=["hide-cell"]
# Mock cell: ensures examples execute without live API keys.
# This cell is hidden in rendered documentation.
```

## 1. What Are Templates?

An answer template is a **Pydantic model** that serves two purposes:

1. **Parsing instructions**: Field names and descriptions tell the Judge LLM what to extract from a free-text response
2. **Verification logic**: The `verify()` method programmatically checks extracted values against expected answers

<div class="admonition info">
<p class="admonition-title">New to Pydantic?</p>
<p><a href="https://docs.pydantic.dev/">Pydantic</a> is a Python library for defining structured data as classes with typed fields. When you write <code>target: str = Field(description="...")</code>, Pydantic knows that <code>target</code> must be a string and carries the description as metadata. Pydantic also generates a <strong>JSON schema</strong>: a machine-readable description of your fields, their types, and their descriptions. This JSON schema is what the Judge LLM receives as its extraction instructions. You do not need deep Pydantic knowledge to write templates.</p>
</div>

This is the core of Karenina's **LLM-as-judge** approach: the answering model produces natural text, the Judge LLM fills in the template's schema, and the template's code decides if the answer is correct.

```
Response (free text)  →  Judge LLM  →  Filled template  →  verify()  →  True/False
```

<div class="admonition note">
<p class="admonition-title">This page teaches how templates work, not how to run them</p>
<p>The code examples below instantiate templates manually and call <code>verify()</code> directly. This is for illustration only. It shows what the Judge LLM and verification pipeline do under the hood. In practice, you never call <code>verify()</code> yourself. Instead, you embed your template in a benchmark and run the <a href="../workflows/creating-benchmarks/factual-qa-benchmark.md">verification pipeline</a>, which handles answer generation, judge parsing, and verification automatically. However, the <code>verify()</code> method itself must be written by the user (see <a href="#template-structure">Template Structure</a> below).</p>
</div>

### 1.1. Think of It Like a Form

If you have ever filled out a standardized form (a lab report, a clinical intake sheet, a grant application), you already understand how templates work. The analogy maps directly:

| Form Concept | Template Equivalent | Who Is Responsible |
|-------------|---------------------|-------------------|
| The blank form with labeled fields and instructions | The template's fields and their descriptions | You, the benchmark author |
| The person reading a letter and filling in the form | The Judge LLM | The system (automatic) |
| The answer key stored in a locked drawer | `ground_truth` setting `self.correct` | You, the benchmark author |
| The clerk who checks the filled form against the answer key | The `verify()` method | You, the benchmark author |

This analogy captures something important: **the person filling in the form and the person checking it are not the same**. The form-filler (the judge) reads a document and writes down what they find. The checker (`verify()`) compares what was written down against known correct answers.

In practice, though, the analogy is not perfectly clean. Some form fields ask the judge to simply copy a value ("write down the city named as France's capital"), but others require genuine evaluation ("check this box if the response identifies BCL2 as the pharmacological target"). In the second case the judge is not merely parsing; it is making a judgment call. The judging problem and evaluation are intrinsically intermingled, and the boundary between "extracting information" and "performing an assessment" shifts depending on field design (see [Ground Truth Exposure](#ground-truth-exposure-and-judge-anchoring) below).

What the architecture *does* guarantee is this: once the judge has filled in the form, everything that follows is deterministic, code-driven verification. The `verify()` method never consults the LLM again. That is the separation that matters, and it is the key to writing good templates. We will come back to this idea throughout the document.

### 1.2. Three Parties, Three Jobs

A template is a contract between three participants, each with a single, well-defined responsibility:

1. **You (the benchmark author)** design the form: what fields to extract, what the correct answers are, and how to check them
2. **The Judge LLM** reads the answering model's response and fills in the form
3. **The `verify()` method** compares the filled-in values against the answer key and returns a pass/fail verdict

The `verify()` method never reads the original response, and the judge never runs `verify()`. How much of the expected answer the judge sees depends on your field design: string fields keep it fully hidden, while boolean fields often reveal it in their descriptions (see [Ground Truth Exposure](#ground-truth-exposure-and-judge-anchoring)). Once the form is filled, evaluation is deterministic code. Swap out the judge model, and the same `verify()` logic still produces the same verdict from the same extracted values.

### 1.3. How It Unfolds: A Walkthrough

To make this concrete, here is how the three participants interact during a single verification. Suppose the benchmark asks: *"How many pairs of chromosomes does a normal human somatic cell have?"*

**Step 1: You define the question and its template.**

The question text goes to the answering model. The template specifies what to extract and how to check it:

```
Template fields:    pair_count (int): "The number of chromosome pairs in a normal human somatic cell"
Ground truth:       {"pair_count": 23}
verify():           self.pair_count == self.correct["pair_count"]
```

**Step 2: The answering model produces a response.**

The pipeline sends the question to the model being evaluated. It replies with free text:

```
"Human somatic cells are diploid, containing 46 chromosomes organized
into 23 pairs. Of these, 22 are autosomal pairs and one is the sex
chromosome pair (XX or XY)..."
```

**Step 3: The judge extracts structured data.**

The Judge LLM receives the response and the template's JSON schema (field names, types, descriptions). It reads the response and fills in the fields:

```
Judge input:   response text + schema {"pair_count": {"type": "integer", "description": "The number of chromosome pairs..."}}
Judge output:  {"pair_count": 23}
```

The judge does not see `ground_truth` or `verify()`. It acts as a reader, not an evaluator.

**Step 4: `verify()` runs the programmatic check.**

The filled template is instantiated, `ground_truth` attaches the expected values, and `verify()` compares:

```
Extracted:   23
Expected:    23
Result:      23 == 23  →  True
```

Note that the response mentions both 46 (total chromosomes) and 23 (pairs). The field description guides the judge to extract the right number. If the judge had extracted 46 instead, `verify()` would return False, correctly flagging a mismatch. The template author controls both what to ask for (the description) and what counts as correct (`verify()`).

With this mental model in place, let's look at how templates are built in practice.

## 2. Template Structure

Every template inherits from `BaseAnswer` and must be named `Answer`. A template has three components:

1. **Fields** with descriptions that guide the Judge LLM
2. **`ground_truth`** to set ground truth values (and optionally `self.regex` for pattern-matching checks)
3. **`verify`** to compare extracted values against ground truth

Here is a simple template that checks whether an LLM correctly identified a drug target:

```python
from typing import Literal

from pydantic import Field

from karenina.schemas.entities import BaseAnswer
```

```python
class Answer(BaseAnswer):
    target: str = Field(description="The protein target of the drug mentioned in the response")

    def ground_truth(self):
        self.correct = {"target": "BCL2"}

    def verify(self) -> bool:
        return self.target.strip().upper().replace("-", "") == self.correct["target"].upper()
```

Let's walk through what happens when this template is used:

1. The Judge LLM reads the answering model's response and extracts the `target` field based on the description
2. Pydantic creates an `Answer` instance, then `ground_truth` sets the expected value
3. `verify()` compares the extracted target to "BCL2" (case-insensitive)

Because `BaseAnswer` uses Pydantic's `extra="allow"` configuration, you can attach any attribute to `self` inside `ground_truth`. The framework uses this for `self.correct` (expected values) and `self.regex` (pattern checks), and you can use it for custom attributes like `self.tolerance`. None of these appear in the JSON schema sent to the judge.

```python
# An answering model might produce a response like:
#   "Venetoclax (ABT-199) is a selective inhibitor of the Bcl-2 protein. By
#    binding directly to Bcl-2, it displaces pro-apoptotic proteins and
#    restores the cell's ability to undergo programmed cell death."
#
# The Judge LLM would extract target="Bcl-2" from that response.
# Here we populate the field directly to demonstrate verify().
parsed = Answer(target="Bcl-2")
print(f"Extracted target: {parsed.target!r}")
print(f"Ground truth:     {parsed.correct['target']!r}")
print(f"Verified:         {parsed.verify()}")
```

Each component exists for a specific reason. The `ground_truth` method keeps the programmatic answer key out of the JSON schema that the judge receives. If correct answers were regular Pydantic fields, they would appear in the schema and could influence the judge's extraction. Because `ground_truth` runs after the instance is created, you can attach expected values to `self.correct` without them ever appearing in the schema. The degree of exposure depends on your field type; see [Ground Truth Exposure](#ground-truth-exposure-and-judge-anchoring) for the full tradeoff.

<div class="admonition info">
<p class="admonition-title">What is <code>ground_truth</code>?</p>
<p><code>ground_truth</code> is a karenina lifecycle method on <code>BaseAnswer</code>. It runs automatically after the instance is created and all fields are set. Unlike the underlying Pydantic hook it wraps, <code>ground_truth</code> takes no extra parameters: just <code>self</code>. You do not need deep Pydantic knowledge to use it: just define the method, set <code>self.correct</code> (and optionally <code>self.regex</code>), and the framework handles the rest.</p>
</div>

The [separation between filling in the form and checking it](#think-of-it-like-a-form) is the most important design idea in karenina's template system. Because everything after the judge is deterministic code, you get reproducibility (the same extracted values always produce the same verdict), transparency (anyone can read `verify()` and see what counts as correct), and testability (you can call `verify()` on your laptop without any LLM or API key).

The three-component structure also means each template is a self-contained evaluation unit that carries its own extraction schema and verification logic. Real benchmarks contain heterogeneous questions: one might use a boolean check, another a string extraction, a third regex only. The pipeline orchestrates execution, but each template decides what to extract and what "correct" means for its question. Add a new question type by writing a new template class; the pipeline runs it without modification.

The pipeline validates templates before running them. Stage 1 (`ValidateTemplate`) checks that `verify()` exists and is callable, and that `ground_truth` sets `self.correct` as a dictionary. If something is missing, you get a clear error message before any LLM calls happen. For regex-only templates and optional methods like `verify_granular()`, see [Regex Checks](#regex-checks) and [Multi-Field with Partial Credit](#multi-field-with-partial-credit).

### 2.1. Naming Requirement

All answer template classes **must be named `Answer`**. The pipeline looks for this exact class name when executing template code. If you add templates through the Python API (e.g., `benchmark.templates.add_answer_template(q_id, MyCustomAnswer)`), the framework automatically renames your class to `Answer` internally. The naming requirement applies when writing template source code directly.

```python
# Correct
class Answer(BaseAnswer):
    value: str = Field(description="The answer value")

    def ground_truth(self):
        self.correct = {"value": "42"}

    def verify(self) -> bool:
        return self.value.strip() == self.correct["value"]


# The Judge LLM would extract this from the answering model's response.
# Here we populate the field directly to demonstrate verify().
a = Answer(value="42")
print(f"verify(): {a.verify()}")
```

## 3. Field Types

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
# An answering model might produce a response like:
#   "Oxygen is the eighth element on the periodic table, with an atomic
#    number of 8. It is essential for aerobic respiration in most living
#    organisms."


class Answer(BaseAnswer):
    element: str = Field(description="The chemical element name mentioned in the response")
    atomic_number: int = Field(description="The atomic number stated in the response")

    def ground_truth(self):
        self.correct = {"element": "oxygen", "atomic_number": 8}

    def verify(self) -> bool:
        name_ok = self.element.strip().lower() == self.correct["element"]
        number_ok = self.atomic_number == self.correct["atomic_number"]
        return name_ok and number_ok


# The Judge LLM would extract element="Oxygen" and atomic_number=8 from
# that response. Here we populate them directly to demonstrate verify().
parsed = Answer(element="Oxygen", atomic_number=8)
print(f"Element correct:  {parsed.element.strip().lower() == parsed.correct['element']}")
print(f"Number correct:   {parsed.atomic_number == parsed.correct['atomic_number']}")
print(f"Overall verify(): {parsed.verify()}")
```

`Literal` constrains the judge to a fixed set of values. Pydantic generates an `enum` in the JSON schema, so the judge can only return one of the allowed options:

```python
# An answering model might produce a response like:
#   "The SNP rs28897696 in the BRCA1 gene results in a missense mutation,
#    where a single nucleotide change causes a different amino acid to be
#    incorporated into the protein."


class Answer(BaseAnswer):
    mutation_type: Literal["missense", "nonsense", "frameshift", "silent"] = Field(
        description="The type of point mutation described in the response."
    )

    def ground_truth(self):
        self.correct = {"mutation_type": "missense"}

    def verify(self) -> bool:
        return self.mutation_type == self.correct["mutation_type"]


# The Judge LLM would extract mutation_type="missense" from that response.
# Here we populate the field directly to demonstrate verify().
parsed = Answer(mutation_type="missense")
print(f"verify(): {parsed.verify()}")
```

## 4. Template Patterns

Five patterns form a toolkit for factual verification. Most real benchmarks use a mix: boolean fields for presence checks, string fields for extractable values, and numeric fields for measurements. Start with the simplest pattern that handles your ground truth, and reach for more complex ones only when needed.

For step-by-step examples of implementing each pattern inside a benchmark, see [Factual QA Benchmark](../workflows/creating-benchmarks/factual-qa-benchmark.md).

### 4.1. Boolean Check

In form terms, this is a checkbox: "Check this box if the letter mentions X." The judge reads the response and ticks yes or no. No string matching needed; `verify()` compares the boolean against `self.correct`. The judge handles synonym recognition through the field description, so it can recognize that "p53," "TP53," and "tumor protein p53" all refer to the same thing.

```python
# An answering model might produce a response like:
#   "TP53 is widely regarded as the single most commonly mutated gene in
#    human cancers. Mutations in p53 are found in over 50% of all tumor
#    types, making it a central focus of cancer genomics research."


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

    def ground_truth(self):
        self.correct = {"identifies_tp53": True}

    def verify(self) -> bool:
        return self.identifies_tp53 == self.correct["identifies_tp53"]


# The Judge LLM would extract identifies_tp53=True from that response.
# Here we populate the field directly to demonstrate verify().
parsed = Answer(identifies_tp53=True)
print(f"Correct answer verified: {parsed.verify()}")

parsed_wrong = Answer(identifies_tp53=False)
print(f"Wrong answer verified:   {parsed_wrong.verify()}")
```

A boolean check avoids string matching entirely. Instead of extracting "TP53" as text and comparing it, we ask the judge "Did the response identify TP53 as the most common?" This is more reliable because the judge handles synonyms (TP53, p53, tumor protein p53) through its description.

The key design decision is in the field description: it must be specific enough to disambiguate edge cases. The `identifies_tp53` description explicitly states that merely listing TP53 alongside other cancer genes should return False; the response must single it out as the *most commonly mutated*. Without this, the judge would likely return True for "TP53, KRAS, and PIK3CA are frequently mutated in cancer," which mentions TP53 but doesn't actually answer the question. Invest time in writing precise descriptions; they are the instructions your judge LLM follows.

Boolean templates follow the same three-component structure as other patterns. The difference is in `verify()`: instead of string normalization or numeric tolerance, it compares the extracted boolean against `self.correct` directly. This makes boolean the simplest template pattern, but it comes with an anchoring tradeoff: the judge sees the expected answer in the field description.

<div class="admonition tip">
<p class="admonition-title">When to use boolean checks</p>
<p>Boolean fields trade some judge independence for simpler verification. See <a href="#ground-truth-exposure-and-judge-anchoring">Ground Truth Exposure</a> for when this matters and how to choose between boolean and string fields. For checking multiple concepts independently with partial credit, see <a href="#multi-field-with-partial-credit">Multi-Field with Partial Credit</a>.</p>
</div>

### 4.2. String Extraction

In form terms, this is a blank text field: "Write down the blood type mentioned in the letter." The judge writes what it finds without knowing what the correct answer should be. The answer key stays in the locked drawer, and the clerk (`verify()`) does the checking later.

This is the cleanest separation between extraction and evaluation. The description tells the judge *what kind of value* to look for and the expected format, but the judge chooses the value based on what the response actually says. The correctness check happens in `verify()`, which the judge never sees.

The tradeoff: since free text is more variable than a yes/no checkbox, you need normalization logic in `verify()` to handle formatting differences (uppercase vs lowercase, "O positive" vs "O+", and so on).

```python
# An answering model might produce a response like:
#   "The most common blood type worldwide is O positive, found in roughly
#    38% of the global population. It is especially prevalent in Central
#    and South America."


class Answer(BaseAnswer):
    blood_type: str = Field(
        description=(
            "The blood type stated in the response as the most common worldwide. "
            "Use standard notation: uppercase letter followed by '+' or '-' "
            "(e.g., 'O+' not 'O positive' or 'type O'). If the response uses "
            "the full name ('O positive'), normalize to shorthand ('O+'). "
            "If the response names multiple blood types, extract the one "
            "identified as the most common overall. If the response distinguishes "
            "by region, extract the type stated as most common globally."
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


# The Judge LLM would extract blood_type="O positive" from that response.
# Here we populate the field directly, simulating various formats the judge
# might return. Normalization in verify() handles them all.
for value in ["O+", "O positive", "o+", "O POSITIVE"]:
    parsed = Answer(blood_type=value)
    print(f"{value!r:>14} -> verify(): {parsed.verify()}")
```

Even though the description asks for standard notation, the judge may return "O positive" or "o+". Handling these variations in `verify()` is expected and keeps the correctness logic deterministic.

<div class="admonition tip">
<p class="admonition-title">When to use string extraction</p>
<p>Prefer string fields when you need the actual value for downstream analysis, want to prevent judge anchoring on ground truth, or need programmatic control over matching (regex, substring, normalization). See <a href="#ground-truth-exposure-and-judge-anchoring">Ground Truth Exposure</a> for the tradeoff with boolean fields.</p>
</div>

### 4.3. Numeric Tolerance

The judge extracts a raw number; `verify()` decides whether it falls within an acceptable range.

This pattern shows the power of having `verify()` as plain code. A body temperature of 37.2 degrees C is clinically normal even though the textbook value is 37.0 degrees C. Rather than trying to encode tolerance rules in a prompt and hoping the LLM applies them consistently, you write a simple programmatic comparison with an explicit margin.

```python
# An answering model might produce a response like:
#   "Normal human body temperature is approximately 37.0 degrees Celsius
#    (98.6 degrees Fahrenheit), though it can vary slightly depending on
#    the time of day and measurement site."


class Answer(BaseAnswer):
    temperature_celsius: float = Field(
        description=(
            "The normal body temperature stated in the response, in degrees "
            "Celsius. Extract the value stated as the standard or average "
            "normal temperature, not extreme or atypical values. If the "
            "response gives the value in Fahrenheit (e.g., 98.6 F), extract "
            "the Celsius equivalent. If a range is given (e.g., 36.5-37.5), "
            "extract the midpoint. If the response gives temperatures for "
            "different measurement sites (oral, rectal, axillary), extract "
            "the oral temperature or the one presented as the primary value."
        )
    )

    def ground_truth(self):
        self.correct = {"temperature_celsius": 37.0}
        self.tolerance = 0.5  # Accept 36.5-37.5, reflecting natural variation

    def verify(self) -> bool:
        return abs(self.temperature_celsius - self.correct["temperature_celsius"]) <= self.tolerance


# The Judge LLM would extract temperature_celsius=37.0 from that response.
# Here we populate the field directly to demonstrate verify().
for temp in [37.0, 36.8, 37.5, 36.0, 38.0]:
    parsed = Answer(temperature_celsius=temp)
    print(f"{temp} C -> verify(): {parsed.verify()}")
```

The `self.tolerance` attribute is not a built-in framework concept; it is a custom attribute you define for your own `verify()` logic. This works because `BaseAnswer` allows arbitrary attributes (see [Template Structure](#template-structure)).

For exact counts where only one value is correct, set `tolerance = 0`:

```python
# An answering model might produce a response like:
#   "A normal human somatic cell contains 46 chromosomes, organized into
#    23 pairs: 22 pairs of autosomes and one pair of sex chromosomes."


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


# The Judge LLM would extract pair_count=23 from that response.
# Here we populate the field directly to demonstrate verify().
parsed = Answer(pair_count=23)
print(f"23 pairs -> verify(): {parsed.verify()}")
parsed_wrong = Answer(pair_count=46)
print(f"46 pairs -> verify(): {parsed_wrong.verify()}")
```

<div class="admonition tip">
<p class="admonition-title">Choosing tolerance values</p>
<p>Use <code>tolerance = 0</code> for exact counts (chromosomes, electrons). Use absolute tolerance for physical measurements with known precision (body temperature, boiling points). Use percentage-based tolerance for values that span wide ranges (e.g., <code>tolerance_pct = 10</code> to accept within 10%).</p>
</div>

### 4.4. Regex Checks

Regex patterns search the **raw LLM response trace**: the full text the answering model produced, before any parsing.

In form terms: sometimes you do not need a form-filler at all. Instead of having someone read the letter and fill in blanks, you search the letter directly for specific words or patterns. That is what regex checks do.

Sometimes you may not want, or may not be able to, use an LLM judge for evaluation. Classical benchmarks often rely on static pattern matching: regex against raw outputs, with no model-based parsing involved. `self.regex` brings this approach into karenina: instead of writing standalone regex scripts outside the framework, you declare named patterns inside the template and the pipeline runs them as part of the standard verification flow. This lets you run classical pattern-matching benchmarks through karenina's pipeline, benefiting from its infrastructure (checkpoints, results, DataFrames, multi-model comparison) without requiring a judge LLM.

When used alongside `verify()`, the pipeline ANDs both results: both `verify()` and `verify_regex()` must pass for the template to succeed. When used alone (with `verify()` returning `True`), regex checks become the sole evaluation mechanism, giving you a pure pattern-matching benchmark.

**Regex-only templates** (no user-defined fields) are detected automatically: the pipeline skips LLM parsing entirely, so no parsing model is needed. Since there are no fields to check, `verify()` is optional. If omitted, field verification defaults to `True` and regex is the sole evaluation.

Set `self.regex` in `ground_truth` (alongside `self.correct` if your template also has fields). Each entry is a named check with a `pattern`, an `expected` value, and a `match_type` that controls comparison:

```python
# An answering model might produce a response like:
#   "Penicillin was discovered in 1928 by Alexander Fleming, who noticed
#    that a mold of the genus Penicillium inhibited bacterial growth on an
#    agar plate."


class Answer(BaseAnswer):
    discovery_date: str = Field(
        description=(
            "The date of penicillin's discovery as stated in the response. "
            "Extract the full date if available, otherwise the year alone."
        )
    )

    def ground_truth(self):
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


# The Judge LLM would extract discovery_date="1928" from that response.
# The regex checks also run against the raw response, matching "1928" and
# "Fleming". Here we populate the field directly to demonstrate verify().
parsed = Answer(discovery_date="1928")
print(f"verify(): {parsed.verify()}")
```

The pipeline calls `verify_regex()` automatically during the VerifyTemplate stage, passing the raw response trace. The result is ANDed with `verify()`: both must pass for the template to succeed. If no `self.regex` is defined, the check passes trivially. In the example above, `verify()` passes because the parsed field contains "1928", but the template would still fail if the raw response didn't match both regex patterns (the year and Fleming's name).

Four match types are available:

| `match_type` | `expected` type | Passes when |
|-------------|----------------|-------------|
| `"exact"` | `str` | Exactly one match, equals `expected` |
| `"contains"` | `str` | `expected` is among the matches (for alternation patterns) |
| `"count"` | `int` | Number of matches equals `expected` |
| `"all"` | `list[str]` | All items in `expected` found in matches |

Note: the `"exact"` match type requires exactly one match in the text. If your pattern might match multiple times (e.g., a year appearing in several sentences), the check will fail even if the matched value is correct. Use `"contains"` instead when multiple matches are possible.

<div class="admonition warning">
<p class="admonition-title">Capture groups change matching behavior</p>
<p>Python's <code>re.findall()</code> returns captured group content instead of full matches when the pattern contains parenthesized groups. For alternation like <code>r"\b(activates|inhibits|blocks)\b"</code>, this works correctly. But for patterns like <code>r"(\d+)\s*(mg|g)"</code>, it returns tuples (<code>[("500", "mg")]</code>), breaking string comparisons. Use non-capturing groups <code>(?:...)</code> when you need grouping but want the full match returned.</p>
</div>

```python
# An answering model might produce a response like:
#   "The drug inhibits the target [1] [2] [3]"
#
# No fields to extract; regex checks run directly against the raw response.


class Answer(BaseAnswer):
    def ground_truth(self):
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


# No fields, no verify(). Regex is the sole evaluation; the pipeline skips
# LLM parsing automatically. In production, verify_regex() receives the
# answering model's raw response. Here we pass a string directly.
parsed = Answer()
result = parsed.verify_regex("The drug inhibits the target [1] [2] [3]")
print(f"verify_regex() passed: {result['success']}")
print(f"Individual checks:     {result['results']}")
```

`verify_regex()` returns a dictionary with three keys: `success` (bool, True if all checks passed), `results` (dict mapping check names to True/False), and `details` (dict mapping check names to diagnostic info including `matches_found`, `match_count`, and `failure_reason`). The `details` key is useful for debugging failing regex checks.

<div class="admonition tip">
<p class="admonition-title">Regex checks vs. regex in verify()</p>
<p><code>self.regex</code> checks patterns in the <strong>raw response trace</strong>: what the answering model actually said. <code>re.search()</code> in <code>verify()</code> normalizes <strong>parsed field values</strong>: what the judge extracted. Use <code>self.regex</code> when you need to verify that certain patterns appear in the original response. Use <code>re.search()</code> in <code>verify()</code> when the judge returns a value in variable format and you need to extract the relevant portion before comparison.</p>
</div>

### 4.5. Multi-Field with Partial Credit

Sometimes a single question asks about multiple things at once, and getting two out of three right is better than zero. You want to know *which* parts were correct.

For questions with multiple dimensions, where you want to check delivery method *and* target protein *and* immune response independently, use multiple fields with both `verify()` (all-or-nothing) and `verify_granular()` (partial credit scoring from 0.0 to 1.0).

`verify_granular()` is not inherited from `BaseAnswer`; it is an optional method you define yourself. The pipeline detects it via `hasattr()` at runtime. If present, the pipeline calls it and records the score; if absent, only the `verify()` pass/fail result is used.

Extracting each check into a private `_check_*` method keeps both `verify()` and `verify_granular()` readable and ensures the logic is defined once. This pattern is recommended whenever you have three or more fields to verify.

```python
# An answering model might produce a response like:
#   "mRNA vaccines work by delivering messenger RNA instructions into
#    cells, which then produce the spike protein found on the surface of
#    SARS-CoV-2. The immune system recognizes this foreign protein and
#    mounts a defensive response, generating antibodies and memory T cells."


class Answer(BaseAnswer):
    delivery_mechanism: str = Field(
        description=(
            "The primary mechanism by which the vaccine delivers instructions "
            "to cells, as described in the response (e.g., 'mRNA instructions', "
            "'messenger RNA', 'genetic code'). Normalize to lowercase. Extract "
            "the delivery method itself, not subsequent biological processes "
            "(protein production, immune response). If the response describes "
            "multiple steps, extract the initial delivery mechanism."
        )
    )
    target_protein: str = Field(
        description=(
            "The protein that cells are instructed to produce, as described "
            "in the response (e.g., 'spike protein'). Normalize to lowercase. "
            "If the response names a specific variant (e.g., 'Wuhan spike "
            "protein'), extract the general protein name. If the response "
            "mentions multiple proteins, extract the one identified as the "
            "primary target of the vaccine."
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


# The Judge LLM would extract delivery_mechanism="mRNA instructions",
# target_protein="spike protein", and mentions_immune_response=True from
# that response. Here we populate them directly to demonstrate verify().
parsed = Answer(
    delivery_mechanism="mRNA instructions",
    target_protein="spike protein",
    mentions_immune_response=True,
)
print(f"verify():         {parsed.verify()}")
print(f"verify_granular(): {parsed.verify_granular():.2f}")

# 2 out of 3 fields match ground truth; verify() fails, but granular gives partial credit
parsed2 = Answer(
    delivery_mechanism="mRNA instructions",
    target_protein="wrong protein",
    mentions_immune_response=True,
)
print(f"\nverify():         {parsed2.verify()}")
print(f"verify_granular(): {parsed2.verify_granular():.2f}")
```

The pipeline calls `verify()` for the pass/fail result used in scoring. `verify_granular()` provides a 0.0 to 1.0 score for finer-grained analysis. Getting 2 out of 3 checks correct yields 0.67 instead of a flat failure.

### 4.6. Pattern Summary

| Pattern | Use When | Don't Use When | Example |
|---------|----------|----------------|---------|
| Boolean check | Known concept, multiple synonyms | You need the extracted value | Gene identification, pathway presence |
| String extraction | Strict matching with programmatic control | Only need presence check (use boolean) | Blood types, gene symbols |
| Numeric tolerance | Measurements or counts | Value is categorical, not numeric | Body temperature, chromosome counts |
| Regex checks (`self.regex`) | Pattern must appear in raw response; no fields = no LLM judge needed | Only need to normalize parsed field values | Year mentions, keyword presence, citation counts |
| Multi-field + partial credit | Multiple dimensions, want granular scoring | Single-answer questions | Mechanisms, multi-part descriptions |

### 4.7. Choosing Between Patterns: Lists vs Individual Booleans

When you expect a set of items (proteins, symptoms, references), you have two approaches.

**`list[str]` extraction**: Use when the set of expected items is open-ended or you need the actual extracted terms for downstream analysis. Requires normalization in `verify()`.

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

    def ground_truth(self):
        self.correct = {"signaling_proteins": {"EGFR", "KRAS", "BRAF"}}

    def verify(self) -> bool:
        extracted = {p.strip().upper() for p in self.signaling_proteins}
        return extracted == self.correct["signaling_proteins"]
```

**Individual boolean fields** (often simpler): Use when you have a known set of expected items. Each field is an independent, unambiguous check. No string matching is needed in `verify()`, and you get per-item partial credit via `verify_granular()`.

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

Each field is self-contained, accepts known synonyms, and `verify()` is trivial: `all(...)` for strict, `sum(...)/len(...)` for partial credit. The boolean approach avoids set comparison, string normalization, and gives you granular per-item results.

## 5. Writing Good Field Descriptions

### 5.1. What the Judge Sees

Remember: the judge is the person reading the letter and filling in your form. The *only* guidance it has is the form itself: the field names, types, and descriptions from your template's JSON schema. The system prompt tells the judge: *"Each field's description is authoritative for what and how to extract. Follow field descriptions precisely."*

This means your field description **is** your specification. If the instructions next to a blank on your form are vague ("write the answer here"), the form-filler will not know what you want. If they are precise ("write the patient's systolic blood pressure in mmHg, as stated in the clinical note"), the form-filler knows exactly what to look for and how to write it down. Invest time in your descriptions. They are the single most important part of your template, because they are the only thing the judge has to work with.

For the tradeoff between boolean and string fields, see [Ground Truth Exposure and Judge Anchoring](#ground-truth-exposure-and-judge-anchoring) below.

### 5.2. Ground Truth Exposure and Judge Anchoring

Earlier we noted that boolean fields often reveal the expected answer in their descriptions, while string fields keep it hidden. This section explores that tradeoff in detail, because it affects how reliably the judge extracts information from the response.

Think of it this way: imagine your form has a checkbox that says "Check this box if the letter mentions Paris as the capital of France." The form-filler now knows what the expected answer is. They might check the box even if the letter only vaguely alludes to Paris, because the form's wording nudged them. This is **anchoring**: the expected answer influences the extraction.

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

### 5.3. Anatomy of a Good Description

The judge receives a JSON schema generated from your template's fields. Each field's `description` is the judge's sole instruction for what to extract and how to format it. The field name and type provide structural hints, but the description carries the semantic weight. A vague description produces unpredictable extractions: the judge fills in whatever seems relevant, and `verify()` fails for reasons that have nothing to do with the answering model's correctness.

The three examples below show how ambiguous descriptions cause extraction failures for the most common field types, and how rewriting them eliminates the ambiguity.

#### 5.3.1. String Fields: Specify the Target, Not Just the Topic

A common mistake with string fields is describing the topic area rather than the specific value to extract.

**Before:**

```python
class Answer(BaseAnswer):
    gene: str = Field(description="The gene mentioned in the response")
```

Suppose the answering model responds: *"Mutations in TP53 are found in over 50% of human cancers. Other frequently mutated genes include KRAS, PIK3CA, and PTEN. The p53 protein functions as a tumor suppressor..."*

The judge sees one instruction: extract "the gene mentioned." But the response mentions four genes (TP53, KRAS, PIK3CA, PTEN). The judge has to guess which one you want. It might pick KRAS. It might extract "p53" instead of the HGNC symbol "TP53." Each run may produce a different result, and `verify()` fails intermittently for reasons that look like extraction bugs but are actually description bugs.

**After:**

```python
class Answer(BaseAnswer):
    gene: str = Field(
        description=(
            "The gene that the response identifies as the single most frequently "
            "mutated gene in human cancers. Extract the official HGNC gene symbol "
            "in uppercase (e.g., 'TP53', not 'p53' or 'tumor protein p53'). "
            "If the response names multiple genes without singling one out as the "
            "most frequent, extract the first gene mentioned."
        )
    )
```

The rewritten description addresses four things the judge needs:

1. **What to extract**: not any gene, but the one singled out as the most frequently mutated
2. **Format**: HGNC symbol, uppercase, with explicit counter-examples of wrong formats
3. **Scope**: only the gene identified as the most frequent, not all genes in the response
4. **Disambiguation**: a fallback rule (first mentioned) for when the response does not single one out

The judge can now make a consistent choice. It reads the response, finds TP53 singled out as appearing in "over 50% of human cancers," and extracts "TP53" in the requested format. The other genes are mentioned but not as the most frequent, so the scope constraint excludes them.

#### 5.3.2. Boolean Fields: Define the Threshold, Not Just the Question

Boolean fields look simple, but their descriptions carry a hidden burden: the judge must evaluate whether a condition is met, not just locate a value. A vague boolean description forces the judge to invent its own threshold for what counts as True.

**Before:**

```python
class Answer(BaseAnswer):
    has_interaction: bool = Field(
        description="True if the response mentions a drug-drug interaction"
    )
```

Suppose the answering model responds: *"In vitro studies suggest a potential interaction between Drug A and Drug B via CYP3A4 inhibition, but clinical trials have not confirmed this finding. The theoretical risk is considered low, and current guidelines do not recommend dose adjustment."*

Does this response "mention a drug-drug interaction"? Technically yes: it discusses one. But the interaction is unconfirmed, theoretical, and clinically insignificant. The judge has no guidance on where to draw the line. It will likely return True, because an interaction is in fact mentioned. If your ground truth expects False (no clinically confirmed interaction), `verify()` fails. The judge extracted exactly what it was asked for; the description just asked the wrong question.

**After:**

```python
class Answer(BaseAnswer):
    has_clinically_confirmed_interaction: bool = Field(
        description=(
            "True if the response describes a drug-drug interaction that has been "
            "confirmed in human clinical trials or is recognized in current "
            "prescribing guidelines (e.g., listed as a contraindication or "
            "requiring dose adjustment). False if the interaction is described as "
            "theoretical, observed only in vitro or in animal models, or "
            "explicitly called clinically insignificant. If the response describes "
            "both confirmed and unconfirmed interactions, return True only if at "
            "least one interaction meets the clinical confirmation threshold."
        )
    )
```

The rewritten description gives the judge a clear decision boundary. "Theoretical" and "in vitro only" map to False. "Confirmed in clinical trials" or "recognized in guidelines" map to True. The judge no longer invents a threshold; you defined one. The field name itself (`has_clinically_confirmed_interaction` instead of `has_interaction`) reinforces the intent, though the description is what the judge actually follows.

#### 5.3.3. List Fields: Control What Gets Included

List fields invite a specific failure mode: over-extraction. Without clear boundaries, the judge collects everything that looks relevant, including items from different contexts, different entities, or different levels of certainty.

**Before:**

```python
class Answer(BaseAnswer):
    symptoms: list[str] = Field(description="The symptoms mentioned in the response")
```

Suppose the answering model responds: *"The patient presented with fever, fatigue, and joint pain. The differential diagnosis includes lupus (which can also cause a malar rash and photosensitivity) and rheumatoid arthritis (characterized by morning stiffness). The patient denied any history of headaches."*

"Symptoms mentioned in the response" could mean anything. The judge might return `["fever", "fatigue", "joint pain", "malar rash", "photosensitivity", "morning stiffness", "headaches"]`, mixing the patient's presenting symptoms with symptoms of differential diagnoses and a denied symptom. If your ground truth expects only the presenting complaints, `verify()` fails because the extracted list is a superset of the correct answer.

**After:**

```python
class Answer(BaseAnswer):
    presenting_symptoms: list[str] = Field(
        description=(
            "The symptoms the patient presented with at the time of evaluation, "
            "as explicitly stated in the response. Include only symptoms attributed "
            "directly to the patient, not symptoms of differential diagnoses or "
            "conditions mentioned for comparison. Exclude symptoms the patient "
            "denied or that are described as absent. Return each symptom as a "
            "separate item in lowercase (e.g., ['fever', 'joint pain'], not "
            "['fever and joint pain']). If the response distinguishes between "
            "signs and symptoms, include both."
        )
    )
```

Three changes make this description effective:

1. **Scope**: "attributed directly to the patient" excludes symptoms of differential diagnoses. "Exclude symptoms the patient denied" handles the negation case.
2. **Format**: "each symptom as a separate item in lowercase" prevents the judge from concatenating multiple symptoms into a single string or using inconsistent casing.
3. **Disambiguation**: "if the response distinguishes between signs and symptoms, include both" prevents the judge from silently filtering out physical signs.

The judge now returns `["fever", "fatigue", "joint pain"]`: the patient's actual presenting symptoms, cleanly separated and consistently formatted.

#### 5.3.4. Quick Reference

Every field description should address four elements. The weight of each varies by field type, but omitting any of them invites unpredictable extractions.

| Element | What It Does | Most Critical For |
|---------|-------------|-------------------|
| **What to extract** | Names the specific value, not just the topic. "The gene identified as most frequent," not "the gene mentioned." | All field types |
| **Format** | Tells the judge how to write the value: casing, notation, symbol conventions, one item per entry for lists. | String and list fields |
| **Scope** | Draws a boundary around what counts. Which mentions are in, which are out. | Boolean and list fields |
| **Disambiguation** | Provides a fallback rule for edge cases: ambiguous mentions, multiple candidates, negated references. | All field types |

## 6. Embedding Check

While not part of template authoring itself, the embedding check is a pipeline feature that template authors should be aware of. It only runs when `verify()` fails.

When `embedding_check_enabled` is set in `VerificationConfig`, the pipeline runs a **semantic similarity check** (stage 9) after `verify()`. This uses a SentenceTransformer model to compare the judge's parsed response fields against the expected answer fields, providing a secondary signal when string-based verification is too strict. The embedding check only runs when field verification (`verify()`) has failed; if `verify()` passes, this stage is skipped entirely.

The embedding check is configured via three settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `embedding_check_enabled` | `False` | Enable the embedding similarity check |
| `embedding_check_model` | `all-MiniLM-L6-v2` | SentenceTransformer model name |
| `embedding_check_threshold` | `0.85` | Similarity threshold (0.0-1.0) |

The embedding check result is stored alongside the template result. When semantic similarity exceeds the threshold and `verify()` has failed, the embedding check can override the field verification result, upgrading it to a pass. The override is tracked via the `embedding_override_applied` flag in results.

## 7. Common Pitfalls

### 7.1. What happens when things go wrong

| Scenario | What happens | Result |
|----------|-------------|--------|
| `verify()` raises an exception | Pipeline catches it, marks context as error | Not pass or fail; recorded as **error** |
| Judge returns `None` for a `str` field | Pydantic validation fails at parse time | Parsing failure; `verify()` never runs |
| `verify_granular()` raises | Caught and logged as warning | Does NOT fail verification; granular score is absent |
| Template validation fails (missing `verify()`, bad `self.correct`) | Stage 1 error; all subsequent stages skip | Error before any LLM calls |

### 7.2. Reserved field names

`BaseAnswer` defines an `id` field (`id: str | None = None`) used internally for question ID injection. Do not use `id` as a field name in your templates. Your custom fields will shadow the inherited `id` field and may cause unexpected behavior during pipeline execution.

**Guard against `None` in `verify()`**: If you declare a field as optional (`target: str | None = Field(...)`), the judge may return `None` when it cannot extract a value. Guard against it before calling string methods:

```python
def verify(self) -> bool:
    if self.target is None:
        return False
    return self.target.strip().upper() == self.correct["target"].upper()
```

## 8. Next Steps

- [Rubrics](rubrics/index.md): Assess response quality beyond correctness
- [Evaluation Modes](../notebooks/core_concepts/evaluation-modes.ipynb): Choose between template-only, rubric-only, or both
- [Factual QA Benchmark](../workflows/creating-benchmarks/factual-qa-benchmark.md): Step-by-step implementation of these patterns in a benchmark
- [Philosophy](../home/philosophy.md): Why the LLM-as-judge approach works
