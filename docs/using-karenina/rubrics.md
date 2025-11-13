# Rubrics

Rubrics provide qualitative evaluation criteria beyond the basic template verification. They enable assessment of answer traits like clarity, conciseness, safety, and domain-specific requirements.

**Quick Navigation:**

- [What Are Rubrics?](#what-are-rubrics) - Core concepts and capabilities
- [Why Use Rubrics?](#why-use-rubrics) - Quality assessment, domain validation, compliance
- [Rubric Scope](#rubric-scope-global-vs-question-specific) - Global vs question-specific rubrics
- [Three Types of Rubric Traits](#three-types-of-rubric-traits) - LLM-based, regex-based, metric-based
- [Understanding Metric Traits](#understanding-metric-traits-two-evaluation-modes) - TP-only vs full-matrix evaluation
- [Creating a Global Rubric](#creating-a-global-rubric) - Apply traits to all questions
- [Creating Question-Specific Rubrics](#creating-question-specific-rubrics) - Apply traits to specific questions
- [Combining Global and Question-Specific](#combining-global-and-question-specific-rubrics) - Use both in one benchmark
- [Working with Rubric Results](#working-with-rubric-results) - Access and analyze evaluation results
- [Rubric Best Practices](#rubric-best-practices) - Effective rubric design guidelines
- [Complete Example](#complete-example) - End-to-end workflow with all trait types

---

## What Are Rubrics?

**Rubrics** are collections of evaluation traits that assess qualitative aspects of LLM responses:

- **Qualitative assessment** - Evaluate traits like clarity, completeness, and style
- **Supplement template verification** - Templates check factual correctness, rubrics assess traits that usually don't have a ground truth or simply characterize the answer style
- **Multiple trait types** - LLM-based, regex-based, and metric-based evaluation
- **Flexible scope** - Apply globally to all questions or to specific questions only

Unlike templates which focus on extracting and verifying structured data, rubrics evaluate broader characteristics of responses that require judgment or pattern matching.

## Why Use Rubrics?

Rubrics are essential for comprehensive evaluation:

1. **Quality Beyond Correctness**: Assess traits like clarity and conciseness that aren't captured by factual verification
2. **Domain Validation**: Check for required terminology or concepts in specialized domains
3. **Safety and Compliance**: Ensure responses meet safety standards or avoid prohibited content
4. **Quantitative Metrics**: Assessing the completeness of an answer by measuring precision, recall, and F1 scores with respect to a list of terms or concepts that we expect to be present/absent in the answer.
5. **Consistent Standards**: Apply uniform evaluation criteria across question sets
6. **Agent Workflow Evaluation**: Use rubrics with [TaskEval](../advanced/task-eval.md) to evaluate pre-logged agent workflow outputs across multiple quality dimensions

## Rubric Scope: Global vs Question-Specific

Rubrics can be applied at two different scopes:

### Global Rubrics - Apply to ALL Questions

**Global rubrics** are evaluated for **every question** in your benchmark. Use global rubrics for traits that should be assessed universally.

**Best for:**

- General quality traits (clarity, conciseness, completeness)
- Safety requirements that apply to all responses
- Style guidelines that should be consistent throughout

**Example use case:** You want to ensure **all** answers in your genomics benchmark are clear and concise, regardless of the specific question.

### Question-Specific Rubrics - Apply to ONE Question

**Question-specific rubrics** are evaluated for **a single question only**. Use question-specific rubrics for domain validation or specialized requirements.

**Best for:**

- Domain-specific terminology checks
- Question-specific validation requirements
- Classification or categorization metrics

**Example use case:** You want to check that the answer to "What is the approved drug target of Venetoclax?" mentions BH3 proteins, but this check only makes sense for that particular question.

---

## Three Types of Rubric Traits

Karenina supports three types of evaluation traits:

### 1. LLM-Based Traits (`RubricTrait`)

**What they are:** AI-evaluated traits where the parsing model uses its judgment to assess subjective qualities of answers. The LLM reads your trait description and applies it to each answer, returning either a score or binary result.

**When to use:**

- Assessing **subjective qualities** that require human-like judgment (clarity, completeness, tone)
- Evaluating **nuanced aspects** that can't be captured by pattern matching
- Measuring **qualities** rather than extracting specific facts
- When you need **flexible evaluation** that adapts to different answer styles

**Two Evaluation Modes:**

**1. Score Mode (1-5 scale):**

- Provides **nuanced assessment** with gradations of quality
- Scale: 1 (Poor) → 2 (Below Average) → 3 (Average) → 4 (Good) → 5 (Excellent)
- Best for: Qualities that exist on a spectrum (clarity, conciseness, thoroughness)

```python
from karenina.schemas import RubricTrait

# Score-based trait: Measures conciseness on a 1-5 scale
RubricTrait(
    name="Conciseness",
    description="Rate the conciseness of the answer on a scale from 1 (very verbose) to 5 (extremely concise).",
    kind="score"
)

# Score-based trait: Measures clarity
RubricTrait(
    name="Clarity",
    description="Rate how clear and understandable the answer is, from 1 (confusing) to 5 (crystal clear).",
    kind="score"
)
```

**2. Binary Mode (pass/fail):**

- Provides **yes/no judgment** on whether criteria are met
- Returns: `true` (pass) or `false` (fail)
- Best for: Questions with clear criteria that either are or aren't satisfied

```python
# Binary trait: Checks for safety concerns
RubricTrait(
    name="Safety Concerns",
    description="Does the answer include any mention of safety concerns or warnings?",
    kind="binary"
)

# Binary trait: Conciseness of the first sentence
RubricTrait(
    name="Conciseness of the first sentence",
    description="The first sentence of the answer should be a concise summary of the answer.",
    kind="binary"
)
```

**How LLM-Based Evaluation Works:**

1. **Prompt Construction**: The parsing model receives:

    - The original question
    - The model's answer
    - Your trait description
    - Scoring instructions (for score mode) or binary criteria (for binary mode)

2. **LLM Judgment**: The parsing model analyzes the answer against your criteria.

3. **Structured Output**: The LLM returns:

    - **Score mode**: Integer from 1–5
    - **Binary mode**: Boolean (true/false)


**Example Scenario:**

```
Question: "What is the approved drug target of Venetoclax?"

Model Answer: "Venetoclax is a BCL-2 inhibitor that works by binding to the BCL-2 protein, which is an anti-apoptotic protein often overexpressed in certain cancers like chronic lymphocytic leukemia. By inhibiting BCL-2, venetoclax promotes apoptosis in cancer cells."

Trait: Conciseness (score, 1-5)
Trait Description: "Rate the conciseness of the answer on a scale from 1 (very verbose) to 5 (extremely concise)."

LLM Evaluation Process:
- Analyzes the answer length and detail
- Considers if extra information (mechanism, cancer types) is necessary
- Compares to an ideal concise answer: "BCL-2"

Result: Score = 2 (verbose, includes unnecessary detail beyond the question)
```

**Best Practices for LLM-Based Traits:**

**Be specific in descriptions**:

- Bad: "Is the answer good?"
- Good: "Rate how clearly the answer explains the concept, from 1 (confusing) to 5 (crystal clear)."

**Provide clear scale anchors** for score mode:

- Include what each extreme means (1 = very verbose, 5 = extremely concise)
- Give context for middle values when helpful

**Use binary mode for yes/no criteria**:

- "Does the answer contain citations?"
- "Is the tone professional?"
- "Does it address safety concerns?"

**Use score mode for gradable qualities**:

- Clarity, conciseness, completeness
- Organization, coherence, depth
- Accuracy, relevance, specificity

**Avoid using LLM traits for**:

- Exact keyword matching (use regex traits instead)
- Factual extraction (use templates instead)
- Classification metrics (use metric traits instead)

### 2. Regex-Based Traits (`ManualRubricTrait`)

**What they are:** Deterministic pattern-matching traits that check if answers match (or don't match) specific regex patterns. These provide **100% reproducible** validation without any LLM judgment.

**When to use:**

- Checking for **required terminology** or keywords
- Validating **format compliance** (dates, gene symbols, IDs)
- Detecting **prohibited content** (profanity, inappropriate terms)
- Ensuring **specific patterns** are present or absent
- When you need **deterministic evaluation** without LLM variability

**Structure:**
```python
from karenina.schemas import ManualRubricTrait

ManualRubricTrait(
    name="Trait Name",
    description="What this pattern checks for",
    pattern=r"regex_pattern",        # Python regex pattern
    case_sensitive=False,              # Case-sensitive matching?
    invert=False                       # Invert the result?
)
```

**Key Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Descriptive name for the trait |
| `description` | `str` | Explains what the pattern validates |
| `pattern` | `str` | Python regex pattern to match |
| `case_sensitive` | `bool` | `True` for case-sensitive, `False` for case-insensitive (default: `False`) |
| `invert` | `bool` | `True` to **fail** if pattern matches, `False` to **pass** if pattern matches (default: `False`) |

**Common Use Cases:**

**1. Required Keyword Check:**

Check that an answer mentions specific required terms:

```python
# Answer must mention "BH3 proteins"
ManualRubricTrait(
    name="Mentions BH3 Proteins",
    description="Answer must mention BH3 proteins (the mechanism of BCL2 inhibition)",
    pattern=r"\bBH3\b",
    case_sensitive=False,
    invert=False  # Pass if pattern found
)

# Answer must mention either "chromosome" or "chromosomes"
ManualRubricTrait(
    name="Mentions Chromosomes",
    description="Answer must use the term 'chromosome' or 'chromosomes'",
    pattern=r"\bchromosomes?\b",  # ? makes 's' optional
    case_sensitive=False,
    invert=False
)
```

**2. Format Validation:**

Ensure answers follow required formats:

```python
# Answer must include a gene symbol (all caps, 3-6 letters)
ManualRubricTrait(
    name="Contains Gene Symbol Format",
    description="Answer must include a gene symbol in standard format (e.g., BCL2, TP53)",
    pattern=r"\b[A-Z]{3,6}\d?\b",  # 3-6 uppercase letters, optional digit
    case_sensitive=True,
    invert=False
)

# Answer must include a numeric value
ManualRubricTrait(
    name="Includes Numeric Answer",
    description="Answer must contain a number",
    pattern=r"\b\d+\b",
    case_sensitive=False,
    invert=False
)
```

**3. Prohibited Content Detection:**

Check that answers DON'T contain unwanted patterns:

```python
# Answer must NOT contain URLs
ManualRubricTrait(
    name="No URLs",
    description="Answer should not contain web URLs",
    pattern=r"https?://[^\s]+",
    case_sensitive=False,
    invert=True  # FAIL if pattern is found
)

# Answer must NOT use informal language
ManualRubricTrait(
    name="No Informal Language",
    description="Answer must avoid informal terms like 'basically', 'kinda', 'sorta'",
    pattern=r"\b(basically|kinda|sorta|like\s+um)\b",
    case_sensitive=False,
    invert=True  # FAIL if pattern is found
)
```

**4. Citation or Reference Check:**

Verify that answers include proper citations:

```python
# Answer must include a citation in square brackets
ManualRubricTrait(
    name="Contains Citation",
    description="Answer must include at least one citation [reference]",
    pattern=r"\[[^\]]+\]",  # Matches [anything inside]
    case_sensitive=False,
    invert=False
)
```

**Using `invert` Parameter:**

The `invert` parameter changes the pass/fail logic:

| `invert` | Pattern Matches | Result |
|----------|----------------|--------|
| `False` (default) | Pattern **found** in answer | ✅ Pass |
| `False` (default) | Pattern **NOT found** in answer | ❌ Fail |
| `True` | Pattern **found** in answer | ❌ Fail |
| `True` | Pattern **NOT found** in answer | ✅ Pass |

**Example:**

```python
# Normal: Pass if "BCL2" is found
ManualRubricTrait(
    name="Mentions BCL2",
    description="Answer must mention BCL2",
    pattern=r"\bBCL2\b",
    invert=False
)
# "The drug targets BCL2" → ✅ Pass (pattern found)
# "The drug targets TP53" → ❌ Fail (pattern not found)

# Inverted: Pass if "BCL2" is NOT found
ManualRubricTrait(
    name="No Mention of BCL2",
    description="Answer must not mention BCL2",
    pattern=r"\bBCL2\b",
    invert=True
)
# "The drug targets BCL2" → ❌ Fail (pattern found)
# "The drug targets TP53" → ✅ Pass (pattern not found)
```

**Regex Pattern Tips:**

**Word Boundaries (`\b`):**
```python
# Without word boundaries: matches partial words
pattern=r"cell"
# Matches: "cell", "cellular", "multicellular"

# With word boundaries: matches complete words only
pattern=r"\bcell\b"
# Matches: "cell"
# Doesn't match: "cellular", "multicellular"
```

**Optional Characters (`?`):**
```python
# Optional 's' for plural
pattern=r"\bchromosomes?\b"
# Matches: "chromosome" or "chromosomes"

# Optional digit at end
pattern=r"\bBCL\d?\b"
# Matches: "BCL" or "BCL2" or "BCL6"
```

**Alternation (`|`):**
```python
# Match any of several options
pattern=r"\b(BCL2|BCL-2|BCL 2)\b"
# Matches: "BCL2", "BCL-2", or "BCL 2"
```

**Character Classes (`[]`):**
```python
# Match uppercase gene symbols
pattern=r"\b[A-Z]{3,6}\b"
# Matches: "TP53", "BCL2", "BRCA1"

# Match numbers
pattern=r"\b\d+\b"
# Matches: "46", "123", "4"
```

**Best Practices for Regex Traits:**

**Use for deterministic checks**:

- Exact keyword presence/absence
- Format validation
- Pattern compliance

**Test your regex patterns**:
```python
import re

pattern = r"\bBCL2\b"
test_cases = [
    "BCL2 is the target",     # Should match
    "BCL2-family proteins",   # Should NOT match (has hyphen)
    "bcl2 variant",           # Should match if case_insensitive=False
]

for test in test_cases:
    match = re.search(pattern, test, re.IGNORECASE if not case_sensitive else 0)
    print(f"'{test}': {'Match' if match else 'No match'}")
```

**Use word boundaries** to avoid partial matches:

- ❌ `pattern=r"cell"` matches "cellular" and "multicellular"
- ✅ `pattern=r"\bcell\b"` only matches "cell"

**Handle term variations**:
```python
# Accept multiple valid formats
pattern=r"\b(BCL2|BCL-2|BCL 2)\b"  # All three formats valid
```

**Avoid using regex traits for:**

- Subjective quality assessment (use LLM traits instead)
- Complex semantic matching (use embedding check instead)
- Classification accuracy (use metric traits instead)

### When to Use Each Trait Type

| Need | Use This Trait Type |
|------|---------------------|
| Subjective quality assessment | LLM-Based (`RubricTrait`) |
| Exact keyword or format validation | Regex-Based (`ManualRubricTrait`) |
| Classification accuracy metrics | Metric-Based (`MetricRubricTrait`) |
| Nuanced scoring (1-5) | LLM-Based (`RubricTrait`, kind="score") |
| Yes/no judgment | LLM-Based (`RubricTrait`, kind="binary") |
| Deterministic pattern matching | Regex-Based (`ManualRubricTrait`) |
| Precision/recall/F1 computation | Metric-Based (`MetricRubricTrait`) |

### 3. Metric-Based Traits (`MetricRubricTrait`)

Confusion matrix-based traits for quantitative classification evaluation.

**Available Metrics:**

- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **Specificity**: TN / (TN + FP)

**Structure:**
```python
from karenina.schemas import MetricRubricTrait

MetricRubricTrait(
    name="Disease Classification",
    description="Evaluate accuracy of identifying inflammatory diseases",
    metrics=["precision", "recall", "f1"],
    tp_instructions=[
        "asthma",      # Should be identified as inflammatory
        "bronchitis",
        "pneumonia"
    ],
    fp_instructions=[
        "emphysema",            # Should NOT be identified as inflammatory
        "pulmonary fibrosis"
    ]
)
```

---

## Understanding Metric Traits: Two Evaluation Modes

Metric-based traits deserve special attention because they operate in two distinct modes, each suited for different evaluation scenarios.

### The Confusion Matrix Concept

Metric traits evaluate classification accuracy using a **confusion matrix**. For any classification task, there are four possible outcomes:

| Category | Definition | Example (Inflammatory Diseases) |
|----------|------------|--------------------------------|
| **TP (True Positive)** | Should be identified AND is identified | Model says "asthma" is inflammatory ✓ |
| **FP (False Positive)** | Should NOT be identified BUT is identified | Model says "emphysema" is inflammatory ✗ |
| **TN (True Negative)** | Should NOT be identified AND is not identified | Model doesn't list "emphysema" ✓ |
| **FN (False Negative)** | Should be identified BUT is NOT identified | Model misses "bronchitis" ✗ |

From these four categories, we compute classification metrics:

- **Precision** = TP / (TP + FP) - "Of what the model identified, how many were correct?"
- **Recall** = TP / (TP + FN) - "Of what should be identified, how many did the model find?"
- **F1 Score** = 2 × (Precision × Recall) / (Precision + Recall) - "Harmonic mean of precision and recall"
- **Accuracy** = (TP + TN) / (TP + TN + FP + FN) - "Overall correctness rate"

### Mode 1: TP-Only Evaluation

**When to use:** When you're identifying items from a **single category** and want to measure how accurately the model identifies them.

**How it works:**

- You provide **only `tp_instructions`** (items that should be identified)
- You may provide **`fp_instructions`** (common incorrect items in the same domain)
- Karenina automatically infers False Positives from the model's response:
  - Any term the model extracts that's **not in `tp_instructions`** is treated as a False Positive
  - If you provide `fp_instructions`, those are highlighted as expected FPs
- True Negatives (TN) are not considered
- False Negatives (FN) = items in `tp_instructions` that the model didn't mention

**Evaluation Mode**: `"tp_only"` (automatically determined)

**Example use case:** "Identify all inflammatory lung diseases from this list"

```python
from karenina.schemas import MetricRubricTrait

# TP-Only Mode: Identifying inflammatory diseases
inflammatory_trait = MetricRubricTrait(
    name="Inflammatory Disease Identification",
    description="Identify inflammatory lung diseases from a mixed list",
    metrics=["precision", "recall", "f1"],

    # Items that SHOULD be identified (True Positives)
    tp_instructions=[
        "asthma",
        "bronchitis",
        "pneumonia",
        "pleurisy"
    ],

    # Common mistakes: items that should NOT be identified (False Positives)
    fp_instructions=[
        "emphysema",            # Obstructive, not inflammatory
        "pulmonary fibrosis",   # Restrictive, not inflammatory
        "sarcoidosis"           # Granulomatous, not inflammatory
    ]
    # Note: No tn_instructions or fn_instructions needed
)
```

**What happens during evaluation:**

1. The parsing model extracts disease names from the LLM's answer
2. Each extracted disease is checked:
   - **In `tp_instructions`?** → Count as True Positive (TP)
   - **In `fp_instructions`?** → Count as False Positive (FP)
   - **Not in either list?** → Also count as False Positive (FP) - unexpected term
3. Items in `tp_instructions` that were NOT extracted → Count as False Negatives (FN)
4. Compute metrics: Precision, Recall, F1

**Example scenario:**

```
Question: "Which are inflammatory: asthma, bronchitis, pneumonia, emphysema, pulmonary fibrosis, sarcoidosis, pleurisy?"

Model answer: "asthma, bronchitis, emphysema"

Evaluation:
- TP: 2 (asthma ✓, bronchitis ✓)
- FP: 1 (emphysema ✗ - listed in fp_instructions)
- FN: 2 (pneumonia ✗, pleurisy ✗ - should have been identified)

Metrics:
- Precision = 2/(2+1) = 0.67
- Recall = 2/(2+2) = 0.50
- F1 = 2*(0.67*0.50)/(0.67+0.50) = 0.57
```

### Mode 2: Full-Matrix Evaluation

**When to use:** When you're evaluating **binary classification** (yes/no, positive/negative) and need to assess both what should be identified AND what should be excluded.

**How it works:**

- You provide **both `tp_instructions` AND `tn_instructions`**
- You may also provide `fp_instructions` and `fn_instructions`
- Karenina evaluates all four categories of the confusion matrix
- Useful for computing accuracy, specificity, and complete classification performance

**Evaluation Mode**: `"full_matrix"` (automatically determined when both TP and TN are provided)

**Example use case:** "Classify each disease as either inflammatory or non-inflammatory"

```python
from karenina.schemas import MetricRubricTrait

# Full-Matrix Mode: Binary classification of diseases
classification_trait = MetricRubricTrait(
    name="Inflammatory vs Non-Inflammatory Classification",
    description="Classify diseases as inflammatory or non-inflammatory",
    metrics=["precision", "recall", "f1", "accuracy", "specificity"],

    # Items that SHOULD be classified as inflammatory (True Positives)
    tp_instructions=[
        "asthma",
        "bronchitis",
        "pneumonia",
        "pleurisy"
    ],

    # Items that SHOULD be classified as non-inflammatory (True Negatives)
    tn_instructions=[
        "emphysema",
        "pulmonary fibrosis",
        "sarcoidosis",
        "lung cancer",
        "tuberculosis"
    ],

    # Optional: Common errors in classification
    fp_instructions=[
        "emphysema",        # Often wrongly called inflammatory
        "sarcoidosis"       # Granulomatous, not inflammatory
    ],

    fn_instructions=[
        "bronchitis",       # Sometimes missed
        "pleurisy"          # Less commonly recognized
    ]
)
```

**What happens during evaluation:**

1. The parsing model extracts disease names AND their classifications from the LLM's answer
2. For each disease mentioned in the answer, check its classification:
   - **Classified as inflammatory AND in `tp_instructions`?** → True Positive (TP)
   - **Classified as inflammatory BUT in `tn_instructions`?** → False Positive (FP)
   - **Classified as non-inflammatory AND in `tn_instructions`?** → True Negative (TN)
   - **Classified as non-inflammatory BUT in `tp_instructions`?** → False Negative (FN)
3. Compute all confusion matrix metrics

**Example scenario:**

```
Question: "Classify each disease as inflammatory or non-inflammatory: asthma, bronchitis, emphysema, sarcoidosis"

Model answer:
- "Inflammatory: asthma, bronchitis, sarcoidosis"
- "Non-inflammatory: emphysema"

Evaluation:
- TP: 2 (asthma ✓, bronchitis ✓)
- FP: 1 (sarcoidosis ✗ - should be TN)
- TN: 1 (emphysema ✓)
- FN: 0 (none missed)

Metrics:
- Precision = 2/(2+1) = 0.67
- Recall = 2/(2+0) = 1.00
- F1 = 2*(0.67*1.00)/(0.67+1.00) = 0.80
- Accuracy = (2+1)/(2+1+1+0) = 0.75
- Specificity = 1/(1+1) = 0.50
```

### Choosing Between Modes

| Scenario | Mode | Instructions Needed | Best For |
|----------|------|---------------------|----------|
| **Item identification** | TP-Only | `tp_instructions` (required)<br>`fp_instructions` (optional) | "Find all inflammatory diseases"<br>"Extract all gene names"<br>"List positive examples" |
| **Binary classification** | Full-Matrix | `tp_instructions` (required)<br>`tn_instructions` (required)<br>`fp_instructions` (optional)<br>`fn_instructions` (optional) | "Classify as inflammatory or not"<br>"Label as positive or negative"<br>"Categorize as safe or unsafe" |

**Key Differences:**

| Aspect | TP-Only Mode | Full-Matrix Mode |
|--------|--------------|------------------|
| **Question type** | "Which items are X?" | "Classify each item as X or not-X" |
| **TN handling** | Ignored (not tracked) | Explicitly tracked and counted |
| **Accuracy metric** | Not available (requires TN) | Available |
| **Specificity metric** | Not available (requires TN) | Available |
| **FP detection** | Any non-TP term counts as FP | Only terms explicitly classified as positive but are in TN list |
| **Use case** | Selection tasks | Classification tasks |

### Automatic Mode Detection

Karenina automatically determines the evaluation mode based on your instructions:

```python
# TP-Only Mode (automatic)
# Reason: Only tp_instructions provided
trait1 = MetricRubricTrait(
    name="Disease Identification",
    metrics=["precision", "recall", "f1"],
    tp_instructions=["asthma", "bronchitis"],
    fp_instructions=["emphysema"]  # Optional
)
# evaluation_mode: "tp_only" (automatically set)

# Full-Matrix Mode (automatic)
# Reason: Both tp_instructions AND tn_instructions provided
trait2 = MetricRubricTrait(
    name="Disease Classification",
    metrics=["precision", "recall", "f1", "accuracy"],
    tp_instructions=["asthma", "bronchitis"],
    tn_instructions=["emphysema", "sarcoidosis"]
)
# evaluation_mode: "full_matrix" (automatically set)
```

### Best Practices for Metric Traits

**For TP-Only Mode:**

- ✅ Use when the question asks to **select** or **identify** items from a category
- ✅ Provide comprehensive `tp_instructions` (all items that should be found)
- ✅ Optionally provide `fp_instructions` to highlight common mistakes
- ✅ Good for: "List all X", "Identify examples of Y", "Extract Z from the text"

**For Full-Matrix Mode:**
- ✅ Use when the question asks to **classify** or **categorize** items
- ✅ Provide both `tp_instructions` (positive class) and `tn_instructions` (negative class)
- ✅ Include all items that need classification in one of the two lists
- ✅ Good for: "Classify each as X or not-X", "Label as positive/negative", "Categorize into groups"

**General Guidelines:**

- Use `repeated_extraction=True` (default) to remove duplicate mentions
- Be specific with instruction terms to avoid ambiguity
- Consider term variations (e.g., "asthma" vs "bronchial asthma")
- Test your metric traits with sample answers first

---

## Creating a Global Rubric

Global rubrics apply to **all questions** in your benchmark. They're perfect for general quality traits like clarity and conciseness.

### Example: General Quality Assessment

```python
from karenina import Benchmark
from karenina.schemas import Rubric, RubricTrait

# Create benchmark
benchmark = Benchmark.create(name="Genomics Knowledge Benchmark")

# Add questions
benchmark.add_question(
    question="How many chromosomes are in a human somatic cell?",
    raw_answer="46"
)

benchmark.add_question(
    question="What is the approved drug target of Venetoclax?",
    raw_answer="BCL2"
)

benchmark.add_question(
    question="How many protein subunits does hemoglobin A have?",
    raw_answer="4"
)

# Create global rubric with LLM-based traits
# These traits will be evaluated for EVERY question
global_rubric = Rubric(
    name="Answer Quality Assessment",
    traits=[
        RubricTrait(
            name="Conciseness",
            description="Rate the conciseness of the answer on a scale from 1 (very verbose) to 5 (extremely concise).",
            kind="score"
        ),
        RubricTrait(
            name="Clarity",
            description="Rate how clear and understandable the answer is, from 1 (confusing) to 5 (crystal clear).",
            kind="score"
        )
    ]
)

# Set as global rubric - applies to ALL questions
benchmark.set_global_rubric(global_rubric)
```

**What happens during verification:**

- The parsing model evaluates **both** traits (Conciseness and Clarity) for **all three questions**
- Each question receives scores from 1-5 for each trait
- Results show how responses perform on general quality metrics

---

## Creating Question-Specific Rubrics

Question-specific rubrics apply to **a single question only**. They're perfect for domain validation and specialized requirements.

### Example 1: Regex-Based Domain Validation

Check that a specific answer mentions required terminology:

```python
from karenina.schemas import Rubric, ManualRubricTrait

# This rubric is ONLY for the Venetoclax question
venetoclax_rubric = Rubric(
    name="Drug Mechanism Validation",
    traits=[
        ManualRubricTrait(
            name="Mentions BH3 Proteins",
            description="Answer must mention BH3 proteins (the mechanism of BCL2 inhibition)",
            pattern=r"\bBH3\b",
            case_sensitive=False
        )
    ]
)

# Add question with specific rubric
# This rubric ONLY applies to THIS question, not the others
benchmark.add_question(
    question="What is the approved drug target of Venetoclax?",
    raw_answer="BCL2",
    rubric=venetoclax_rubric
)
```

**What happens during verification:**

- The regex pattern checks **only** the Venetoclax answer for "BH3"
- Other questions are NOT checked for this pattern
- Returns `True` if the pattern is found, `False` otherwise

### Example 2: Metric-Based Classification

Evaluate classification accuracy for a question that requires identifying items in categories:

```python
from karenina.schemas import Rubric, MetricRubricTrait

# This rubric is ONLY for the disease classification question
disease_rubric = Rubric(
    name="Inflammatory Disease Classification",
    traits=[
        MetricRubricTrait(
            name="Inflammatory Disease Accuracy",
            description="Evaluate accuracy of identifying inflammatory lung diseases",
            metrics=["precision", "recall", "f1"],
            tp_instructions=[
                "asthma",       # Inflammatory diseases (should be identified)
                "bronchitis",
                "pneumonia",
                "pleurisy"
            ],
            fp_instructions=[
                "emphysema",            # NOT inflammatory (should be excluded)
                "pulmonary fibrosis",
                "sarcoidosis"
            ]
        )
    ]
)

# Add question with classification metric rubric
# This rubric ONLY applies to THIS question
benchmark.add_question(
    question="Which of the following are inflammatory lung diseases: asthma, bronchitis, pneumonia, emphysema, pulmonary fibrosis, sarcoidosis, pleurisy?",
    raw_answer="asthma, bronchitis, pneumonia, pleurisy",
    rubric=disease_rubric
)
```

**What happens during verification:**

- The parsing model extracts disease names from the answer
- Each disease is categorized as TP (correct inflammatory), FP (incorrect inflammatory), or FN (missed inflammatory)
- Precision, recall, and F1 score are calculated
- Other questions do NOT use this metric evaluation

---

## Combining Global and Question-Specific Rubrics

You can use both global and question-specific rubrics in the same benchmark:

```python
from karenina import Benchmark
from karenina.schemas import Rubric, RubricTrait, ManualRubricTrait

# Create benchmark
benchmark = Benchmark.create(name="Genomics Knowledge Benchmark")

# Global rubric: applies to ALL questions
global_rubric = Rubric(
    name="General Quality",
    traits=[
        RubricTrait(
            name="Conciseness",
            description="Rate the conciseness of the answer on a scale from 1 (very verbose) to 5 (extremely concise).",
            kind="score"
        ),
        RubricTrait(
            name="Clarity",
            description="Rate how clear and understandable the answer is, from 1 (confusing) to 5 (crystal clear).",
            kind="score"
        )
    ]
)

benchmark.set_global_rubric(global_rubric)

# Question 1: Uses only global rubric
benchmark.add_question(
    question="How many chromosomes are in a human somatic cell?",
    raw_answer="46"
)

# Question 2: Uses global rubric + question-specific rubric
venetoclax_rubric = Rubric(
    name="Drug Mechanism Validation",
    traits=[
        ManualRubricTrait(
            name="Mentions BH3 Proteins",
            description="Answer must mention BH3 proteins",
            pattern=r"\bBH3\b",
            case_sensitive=False
        )
    ]
)

benchmark.add_question(
    question="What is the approved drug target of Venetoclax?",
    raw_answer="BCL2",
    rubric=venetoclax_rubric  # Question-specific rubric
)

# Question 3: Uses only global rubric
benchmark.add_question(
    question="How many protein subunits does hemoglobin A have?",
    raw_answer="4"
)
```

**Result:**

- **Question 1** (chromosomes): Evaluated for Conciseness and Clarity (global rubric)
- **Question 2** (Venetoclax): Evaluated for Conciseness, Clarity (global rubric) + BH3 mention check (question-specific rubric)
- **Question 3** (hemoglobin): Evaluated for Conciseness and Clarity (global rubric)

---

## Working with Rubric Results

After running verification with rubrics, you can access the results:

```python
from karenina.schemas import ModelConfig, VerificationConfig

# Configure verification
model_config = ModelConfig(
    id="gpt-4.1-mini",
    model_provider="openai",
    model_name="gpt-4.1-mini",
    temperature=0.0,
    interface="langchain"
)

config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config]
)

# Run verification
results = benchmark.run_verification(config)

# Access rubric scores for each question
for result in results:
    print(f"\nQuestion: {result.question.question}")

    # LLM-based trait scores (global rubric)
    if result.rubric_scores:
        print("  Global Rubric Scores:")
        for trait_name, score in result.rubric_scores.items():
            print(f"    {trait_name}: {score}/5")

    # Regex trait results (question-specific)
    if result.manual_rubric_results:
        print("  Regex Trait Results:")
        for trait_name, passed in result.manual_rubric_results.items():
            status = "✓ Pass" if passed else "✗ Fail"
            print(f"    {trait_name}: {status}")

    # Metric trait results (question-specific)
    if result.metric_rubric_results:
        print("  Metric Trait Results:")
        for trait_name, metrics in result.metric_rubric_results.items():
            print(f"    {trait_name}:")
            for metric_name, value in metrics.items():
                print(f"      {metric_name}: {value:.2f}")
```

**Example output:**
```
Question: How many chromosomes are in a human somatic cell?
  Global Rubric Scores:
    Conciseness: 5/5
    Clarity: 5/5

Question: What is the approved drug target of Venetoclax?
  Global Rubric Scores:
    Conciseness: 4/5
    Clarity: 5/5
  Regex Trait Results:
    Mentions BH3 Proteins: ✓ Pass

Question: Which of the following are inflammatory lung diseases...
  Global Rubric Scores:
    Conciseness: 4/5
    Clarity: 4/5
  Metric Trait Results:
    Inflammatory Disease Accuracy:
      precision: 1.00
      recall: 0.75
      f1: 0.86
```

---

## Rubric Best Practices

**Design effective rubrics**:

- Keep trait descriptions clear and specific
- Test rubric traits with sample answers before full evaluation
- Use appropriate trait types for your evaluation needs (see [When to Use Each Trait Type](#when-to-use-each-trait-type))
- Consider both global and question-specific scopes (see [Rubric Scope](#rubric-scope-global-vs-question-specific))

**For LLM-based traits**:

- Provide clear scale anchors for score mode (what 1 and 5 represent)
- Use binary mode for yes/no criteria, score mode for gradable qualities
- Avoid using LLM traits for tasks better suited to regex or metrics

**For regex-based traits**:

- Test your patterns before deployment
- Use word boundaries (`\b`) to avoid partial matches
- Consider case sensitivity requirements carefully
- Use `invert=True` when checking for prohibited content

**For metric-based traits**:

- Choose the correct evaluation mode (TP-only vs full-matrix) based on your question type
- Provide comprehensive instruction lists
- Use `repeated_extraction=True` to handle duplicate mentions
- Consider term variations in your instruction lists

---

## Complete Example

Here's a complete workflow showing both global and question-specific rubrics with all three trait types:

```python
from karenina import Benchmark
from karenina.schemas import (
    Rubric, RubricTrait, ManualRubricTrait, MetricRubricTrait,
    ModelConfig, VerificationConfig
)

# 1. Create benchmark
benchmark = Benchmark.create(
    name="Genomics Knowledge Benchmark",
    description="Testing LLM knowledge of genomics and molecular biology",
    version="1.0.0"
)

# 2. Create global rubric (applies to ALL questions)
global_rubric = Rubric(
    name="General Quality Assessment",
    traits=[
        RubricTrait(
            name="Conciseness",
            description="Rate the conciseness of the answer on a scale from 1 (very verbose) to 5 (extremely concise).",
            kind="score"
        ),
        RubricTrait(
            name="Clarity",
            description="Rate how clear and understandable the answer is, from 1 (confusing) to 5 (crystal clear).",
            kind="score"
        )
    ]
)

benchmark.set_global_rubric(global_rubric)

# 3. Add question with only global rubric
benchmark.add_question(
    question="How many chromosomes are in a human somatic cell?",
    raw_answer="46"
)

# 4. Add question with regex-based question-specific rubric
venetoclax_rubric = Rubric(
    name="Drug Mechanism Validation",
    traits=[
        ManualRubricTrait(
            name="Mentions BH3 Proteins",
            description="Answer must mention BH3 proteins",
            pattern=r"\bBH3\b",
            case_sensitive=False
        )
    ]
)

benchmark.add_question(
    question="What is the approved drug target of Venetoclax?",
    raw_answer="BCL2",
    rubric=venetoclax_rubric
)

# 5. Add question with metric-based question-specific rubric
disease_rubric = Rubric(
    name="Inflammatory Disease Classification",
    traits=[
        MetricRubricTrait(
            name="Inflammatory Disease Accuracy",
            description="Evaluate accuracy of identifying inflammatory lung diseases",
            metrics=["precision", "recall", "f1"],
            tp_instructions=["asthma", "bronchitis", "pneumonia", "pleurisy"],
            fp_instructions=["emphysema", "pulmonary fibrosis", "sarcoidosis"]
        )
    ]
)

benchmark.add_question(
    question="Which of the following are inflammatory lung diseases: asthma, bronchitis, pneumonia, emphysema, pulmonary fibrosis, sarcoidosis, pleurisy?",
    raw_answer="asthma, bronchitis, pneumonia, pleurisy",
    rubric=disease_rubric
)

# 6. Generate templates
model_config = ModelConfig(
    id="gpt-4.1-mini",
    model_provider="openai",
    model_name="gpt-4.1-mini",
    temperature=0.1,
    interface="langchain"
)

benchmark.generate_all_templates(model_config=model_config)

# 7. Run verification with rubrics
config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config]
)

results = benchmark.run_verification(config)

# 8. Analyze results
print(f"✓ Verification complete: {len(results)} questions evaluated")
print(f"✓ All questions assessed for: Conciseness, Clarity")
print(f"✓ Venetoclax question checked for: BH3 mention")
print(f"✓ Disease question evaluated with: Precision, Recall, F1")

# 9. Save benchmark
benchmark.save("genomics_benchmark.jsonld")
```

---

## Next Steps

Once you have rubrics configured:

- [Run verification](verification.md) to apply both template and rubric evaluation
- [Analyze results](verification.md#accessing-verification-results) to understand performance across different criteria
- [Save and load benchmarks](saving-loading.md) to preserve your rubric configurations
- [Export results](saving-loading.md#exporting-verification-results) to CSV or JSON for further analysis

---

## Related Documentation

- [Adding Questions](adding-questions.md) - Populate your benchmark with questions
- [Templates](templates.md) - Structured answer evaluation for factual correctness
- [Verification](verification.md) - Run evaluations with multiple models
- [Quick Start](../quickstart.md) - End-to-end workflow example
