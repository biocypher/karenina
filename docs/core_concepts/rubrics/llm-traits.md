---
jupyter:
  jupytext:
    formats: docs/core_concepts/rubrics//md,docs/notebooks/core_concepts/rubrics//ipynb
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

# LLM Rubric Traits

LLM rubric traits use the **parsing model's judgment** to assess observable qualities of LLM responses that do not require ground truth. They are the most flexible trait type, capable of evaluating nuanced aspects like clarity, safety, and tone that cannot be captured by pattern matching or deterministic logic. For an overview of all trait types and how rubrics fit into the evaluation framework, see the [rubrics index](index.md).

```python tags=["hide-cell"]
# Mock cell: ensures examples execute without live API keys.
# This cell is hidden in rendered documentation.
```

## Overview

An `LLMRubricTrait` sends the original question and the model's response to the parsing model along with the trait definition for that kind. The parsing model then returns a structured result.

Three kinds are available:

| Kind | Returns | Best For |
|------|---------|----------|
| **boolean** | `True` / `False` | Binary pass/fail judgments (safety, presence of citations) |
| **score** | `int` in a configurable range | Gradable qualities on a scale (clarity 1-5, conciseness 1-5) |
| **literal** | `int` (class index) | Categorical classification with named classes (quality tiers, response types, tone levels) |

## Why the `description` Field Matters for Boolean and Score Traits

LLM traits are evaluated during [stage 11 (RubricEvaluation)](../verification-pipeline.md) of the [verification pipeline](../verification-pipeline.md). During evaluation, the [prompt assembler](../../advanced-pipeline/prompt-assembly.md) builds a message for the parsing model containing: a **system prompt** assigning the evaluator role, and a **user prompt** with the original question text, the model's full response (trace), and your trait definition.

```
Question + Response Trace + Trait Definition
                    ↓
         ┌──────────────────┐
         │   Parsing Model  │  ← System: evaluator role
         │                  │  ← User: question + trace + trait definition
         └────────┬─────────┘
                  ↓
    Structured result per trait (bool or int)
                  ↓
    VerificationResult.rubric
```

For **boolean** and **score** traits, the `description` field is the main channel through which you tell the parsing model what to evaluate and how. A vague description like "Is the response good?" gives the judge no actionable criteria; a detailed description with concrete examples, edge cases, and domain context produces more reliable judgments.

For **literal** traits, the situation is slightly different: the parsing model receives both the top-level `description` and the per-class descriptions in `classes`. In practice, the class descriptions do most of the work, while the top-level description provides scope and context.

Two factors control evaluation quality:

1. **Trait-definition quality**: For boolean and score traits, this mostly means the `description`. For literal traits, it means the combination of the top-level description and the class descriptions. Detailed, observable criteria with explicit boundary cases produce more consistent judgments.
2. **Model capability**: More capable parsing models interpret nuanced descriptions more faithfully. If you need fine-grained distinctions (e.g., distinguishing "names the target" from "explains the mechanism"), use a stronger model. Simpler traits (e.g., "is a citation present?") are reliable even with smaller models.

If multiple LLM traits exist on the same rubric, they can be evaluated in a single parsing call (batch strategy) or individually (sequential strategy), depending on the evaluation strategy and runtime configuration. Results are stored in `VerificationResult.rubric` and become available for analysis and DataFrame export.

## Boolean Kind

Boolean traits answer a yes/no question about the response. The parsing model reads the trait description and returns `True` or `False`.

**When to use:**

- Safety or compliance checks: *"Is this response safe and appropriate?"*
- Presence checks: *"Does the answer include citations?"*
- Style requirements: *"Is the tone professional?"*

### Writing Boolean Descriptions

A good boolean description defines exactly what evidence makes the answer True or False, including the cases that are hardest to classify. Describe what's observable in the text, not abstract qualities.

```python
LLMRubricTrait(
    name="Mechanistic Explanation",
    description=(
        "Answer True if the response explains the biological mechanism of action — "
        "specifically HOW the drug interacts with its target at the molecular level "
        "(e.g., binds to the BH3 domain, inhibits kinase activity, blocks receptor "
        "dimerization). Simply naming the target without explaining the interaction "
        "mechanism counts as False. Mentioning downstream effects (e.g., 'induces "
        "apoptosis') without explaining the direct molecular interaction also counts "
        "as False."
    ),
    kind="boolean",
    higher_is_better=True,
)
```

This description provides concrete examples of what True looks like and explicitly addresses the boundary cases (naming without explaining, downstream effects without mechanism) that would otherwise be ambiguous.

```python
LLMRubricTrait(
    name="Cites Primary Literature",
    description=(
        "Answer True if the response references at least one specific published study "
        "by providing author names, journal names, or publication years (e.g., "
        "'Tsujimoto et al., 1985' or 'published in Nature'). Generic references like "
        "'studies have shown' or 'research suggests' without specific attributions "
        "count as False."
    ),
    kind="boolean",
    higher_is_better=True,
)
```

This description defines what "cites" means concretely (specific names, journals, or years) and draws a clear boundary against generic attributions.

### Creating a Boolean Trait

```python
from karenina.schemas import LLMRubricTrait

safety_trait = LLMRubricTrait(
    name="Safe Response",
    description=(
        "Answer True if the response is safe and appropriate for a general audience, "
        "with no harmful, misleading, or offensive content. Answer False otherwise."
    ),
    kind="boolean",
    higher_is_better=True,  # True = safe = good
)

print(f"Trait: {safety_trait.name}")
print(f"Kind: {safety_trait.kind}")
print(f"higher_is_better: {safety_trait.higher_is_better}")
```

### Boolean Trait with Inverted Directionality

Sometimes a `True` result indicates a negative outcome. Set `higher_is_better=False` to signal that `True` is bad:

```python
hallucination_trait = LLMRubricTrait(
    name="Contains Hallucination",
    description=(
        "Answer True if the response contains fabricated facts, invented citations, "
        "or information not supported by the question context. Answer False otherwise."
    ),
    kind="boolean",
    higher_is_better=False,  # True = hallucination found = bad
)

print(f"Trait: {hallucination_trait.name}")
print(f"higher_is_better: {hallucination_trait.higher_is_better}")
# Analysis tools know that True here means worse performance
```

## Score Kind

Score traits rate a quality on a numeric scale. The default range is 1-5, but you can customize it with `min_score` and `max_score`.

**When to use:**

- Gradable qualities: *"Rate clarity from 1 (confusing) to 5 (crystal clear)"*
- Spectrum assessment: *"How concise is the response?"*
- Comparative evaluation: where you want to distinguish between adequate and excellent responses

### Writing Score Descriptions

A good score description anchors the scale with concrete examples at key points (low, middle, high). This gives the judge observable criteria for each level rather than an abstract number to pick.

```python
LLMRubricTrait(
    name="Explanation Depth",
    description=(
        "Rate how deeply the response explains the underlying biology. "
        "1 = surface-level, only states the conclusion (e.g., 'BCL2 is important in cancer'). "
        "3 = provides supporting detail (e.g., names the pathway and its role). "
        "5 = thorough mechanistic explanation with molecular-level detail and context."
    ),
    kind="score",
    higher_is_better=True,
)
```

Anchoring at 1, 3, and 5 with concrete examples leaves room for the judge to interpolate. Use score when the quality is genuinely continuous or when distinct levels aren't cleanly separable. If you can define named levels with clear boundaries, use a [literal trait](#literal-kind) instead.

### Creating a Score Trait

```python
clarity_trait = LLMRubricTrait(
    name="Clarity",
    description=(
        "Rate how clear and understandable the response is. "
        "1 = very confusing, hard to follow. "
        "3 = adequate, understandable but could be clearer. "
        "5 = exceptionally clear and well-articulated."
    ),
    kind="score",
    higher_is_better=True,  # Higher score = better clarity
)

print(f"Trait: {clarity_trait.name}")
print(f"Kind: {clarity_trait.kind}")
print(f"Score range: {clarity_trait.min_score}-{clarity_trait.max_score}")
```

### Custom Score Range

The default range is 1-5. You can change it:

```python
detail_trait = LLMRubricTrait(
    name="Detail Level",
    description=(
        "Rate the level of detail in the response. "
        "1 = extremely brief, missing key information. "
        "5 = moderate detail, covers the basics. "
        "10 = comprehensive, covers all relevant aspects with examples."
    ),
    kind="score",
    min_score=1,
    max_score=10,
    higher_is_better=True,
)

print(f"Score range: {detail_trait.min_score}-{detail_trait.max_score}")
```

### Score Validation

The `validate_score` method checks whether a given value is valid for a trait:

```python
# Score trait: accepts integers in [min_score, max_score]
print(clarity_trait.validate_score(3))     # True - valid score
print(clarity_trait.validate_score(6))     # False - above max_score
print(clarity_trait.validate_score(True))  # False - booleans rejected for score traits
```

## Literal Kind

Literal traits perform **categorical classification**. Instead of a binary yes/no or a numeric scale, the parsing model classifies the response into one of several predefined categories. The stored score is the **class index**, an integer indicating which category was selected.

Classification into named categories is often more reliable than numeric scoring because the judge has concrete labels to choose from rather than an abstract scale. Literal traits work well in two common situations:

- You have **distinct categories** and care mainly about the class label.
- You have **ordered levels** and want the class index to carry ordinal meaning.

When your quality dimension has distinct, observable levels, literal traits often produce more consistent results than score traits.

A literal trait is created by setting `kind="literal"` on `LLMRubricTrait` and providing a `classes` dictionary:

| Field | Type | Description |
|-------|------|-------------|
| `kind` | `"literal"` | Must be `"literal"` for categorical classification |
| `classes` | `dict[str, str]` | Class name to description mapping (2-20 classes, order matters) |
| `higher_is_better` | `bool` | Whether higher class indices indicate better performance |

Key characteristics:

- **Ordered**: Dictionary order determines indices (0, 1, 2, ...)
- **Auto-ranged**: `min_score` is set to 0, `max_score` to `len(classes) - 1` automatically
- **Descriptive**: Each class has a name and a description to guide the parsing model
- **2-20 classes**: Must have at least 2 and at most 20 categories

Even when your classes are conceptually distinct rather than "better" or "worse," Karenina still stores them as indices because that is how literal traits are represented internally. In those cases, treat the class **label** as the primary result and the numeric index as a stable storage convention.

### Writing Good Class Descriptions

The quality of class descriptions directly affects how well the parsing model classifies responses. Each class description should be:

- **Mutually exclusive**: clearly distinct from other classes
- **Observable**: describe what the model should look for in the response text
- **Ordered consistently**: if using `higher_is_better`, ensure the natural ordering matches

**Good** (clear criteria the model can evaluate):

    "poor": "Incorrect, misleading, or largely irrelevant to the question"
    "acceptable": "Broadly correct but missing important details or nuance"
    "good": "Correct and well-structured with adequate supporting detail"
    "excellent": "Comprehensive, precise, well-organized, and addresses edge cases"

**Weak** (vague or overlapping):

    "bad": "A bad answer"
    "ok": "An okay answer"
    "good": "A good answer"

Here is a more detailed example with five classes. Each class describes what's observable in the text, and the boundaries are clear: "names specific indications" separates `specific_clinical` from `general_clinical`.

```python
LLMRubricTrait(
    name="Clinical Relevance",
    description="Rate how clinically relevant the response is for a practicing oncologist.",
    kind="literal",
    classes={
        "theoretical": (
            "Purely theoretical with no connection to clinical practice — discusses "
            "only molecular biology, pathway diagrams, or in-vitro findings"
        ),
        "general_clinical": (
            "Mentions clinical context but lacks specifics — e.g., 'used in cancer "
            "treatment' without naming indications, patient populations, or regimens"
        ),
        "specific_clinical": (
            "Names specific indications (e.g., CLL, AML) or patient populations, "
            "but does not discuss treatment protocols or clinical data"
        ),
        "practice_level": (
            "Discusses dosing, treatment lines, combination regimens, or compares "
            "to standard-of-care options"
        ),
        "evidence_based": (
            "Integrates clinical trial data, response rates, survival outcomes, "
            "or references treatment guidelines (e.g., NCCN)"
        ),
    },
    higher_is_better=True,
)
```

### How Literal Evaluation Works

The parsing model receives the trait description, class names, and class descriptions as part of the evaluation prompt. It classifies the response by selecting the best-matching class name, which is then converted to an integer index:

```
Question + Response + Trait Description + Class Definitions
                ↓
         Parsing Model
                ↓
    Selects class name (e.g., "good")
                ↓
    Converted to index (e.g., 2)
```

This two-step process (classify by name, then map to index) is more reliable than asking the model to pick a number directly.

### Distinct Categories Example

```python
response_type_trait = LLMRubricTrait(
    name="Response Type",
    description="Classify what kind of answer this is.",
    kind="literal",
    classes={
        "factual": "Response mainly presents objective facts or data",
        "opinion": "Response mainly expresses subjective views or preferences",
        "speculative": "Response mainly discusses possibilities, uncertainty, or hypotheticals",
        "refusal": "Response declines to answer or redirects the question",
    },
    higher_is_better=False,
)

print(f"Kind: {response_type_trait.kind}")
print(f"Classes: {list(response_type_trait.classes.keys())}")
print(f"Score range: {response_type_trait.min_score} to {response_type_trait.max_score}")
```

The parsing model receives the class names and descriptions, then selects the one that best fits the response. The result is the class index:

- `0` → "factual"
- `1` → "opinion"
- `2` → "speculative"
- `3` → "refusal"

This is a good example of **distinct categories** rather than a quality ladder. There is no intrinsic "best" class here. The order is still stored numerically, but that ordering is mainly a convention for storage and downstream tooling. For display and interpretation, the label is usually more important than the index.

### Quality Tiers

A second common pattern is defining quality levels where order is meaningful:

```python
quality_trait = LLMRubricTrait(
    name="Answer Quality",
    description="Rate the overall quality of this answer.",
    kind="literal",
    classes={
        "poor": "Incorrect, misleading, or largely irrelevant",
        "acceptable": "Broadly correct but missing important details",
        "good": "Correct and well-structured with adequate detail",
        "excellent": "Comprehensive, precise, and well-organized",
    },
    higher_is_better=True,  # Higher index = better quality
)

print(f"Classes: {list(quality_trait.classes.keys())}")
print(f"Score range: {quality_trait.min_score} to {quality_trait.max_score}")
print(f"higher_is_better: {quality_trait.higher_is_better}")
```

Here `higher_is_better=True` because later classes (higher indices) represent better quality:

- `0` → "poor" (worst)
- `1` → "acceptable"
- `2` → "good"
- `3` → "excellent" (best)

### Working with Class Names

`LLMRubricTrait` provides helper methods for converting between class names and indices:

```python
# Get the ordered list of class names
print(f"Class names: {quality_trait.get_class_names()}")

# Get the index for a specific class
print(f"Index of 'good': {quality_trait.get_class_index('good')}")
print(f"Index of 'poor': {quality_trait.get_class_index('poor')}")

# Invalid class names return -1
print(f"Index of 'unknown': {quality_trait.get_class_index('unknown')}")
```

The `get_class_index()` method returns `-1` for unrecognized class names. This value is also accepted by `validate_score()` as a valid error state for literal traits.

### Literal Score Validation

Literal traits validate scores the same way as score traits: the value must be an integer within the auto-derived range:

```python
# Valid scores
print(f"Is 0 valid? {quality_trait.validate_score(0)}")   # First class
print(f"Is 3 valid? {quality_trait.validate_score(3)}")   # Last class
print(f"Is -1 valid? {quality_trait.validate_score(-1)}")  # Error state

# Invalid scores
print(f"Is 4 valid? {quality_trait.validate_score(4)}")    # Out of range
print(f"Is True valid? {quality_trait.validate_score(True)}")  # Boolean rejected
```

Literal traits are evaluated through the standard classification path. The [deep judgment](#deep-judgment-optional) guidance below is intended for boolean and score traits.

## The `higher_is_better` Field

This required field tells analysis tools how to interpret results:

| Kind | `higher_is_better=True` | `higher_is_better=False` |
|------|------------------------|--------------------------|
| boolean | `True` = positive outcome | `True` = negative outcome |
| score | Higher scores = better | Higher scores = worse |
| literal | Higher class indices are interpreted as better | Higher class indices are interpreted as worse |

Most traits use `higher_is_better=True`. Use `False` for traits where a positive detection is bad (e.g., hallucination detected, contains prohibited content).

For literal traits, `higher_is_better` does **not** affect classification itself. It only affects how downstream tooling interprets the numeric indices. If your classes are distinct categories rather than a quality ladder, choose a stable convention and rely on the class labels for human interpretation.

## Deep Judgment (Optional)

Deep judgment enhances LLM trait evaluation by extracting **evidence** from the response to support the judgment. It is currently best suited to **boolean** and **score** traits, where the final output is naturally a boolean or numeric score.

Although the deep-judgment fields exist on `LLMRubricTrait` generally, the current deep-judgment pipeline extracts booleans or numeric scores and does not consume literal class definitions during scoring. In practice, use standard evaluation for literal traits.

Instead of just returning a score or boolean, the parsing model also identifies specific text passages (excerpts) that justify its assessment.

During the pipeline, deep judgment adds stages as a post-processing layer after the initial judgment in stage 11. When enabled, [stage 12 (DeepJudgmentRubric)](../verification-pipeline.md) runs the following sequence:

1. **Judgment** (already completed in stage 11): the parsing model returns a score or boolean
2. **Excerpt extraction**: the parsing model identifies verbatim passages from the response that support its judgment
3. **Fuzzy match validation**: extracted excerpts are validated against the actual response text using fuzzy string matching to ensure they are real quotes (not hallucinated)
4. **Search fallback** (optional): if fuzzy matching fails, a search-based approach attempts to locate the excerpt in the response

This layered approach means you get both the judgment and the evidence trail. The threshold (default 0.85) controls how closely an excerpt must match the original text; higher values require near-exact matches.

**When to use deep judgment:**

- Transparency and auditability are important
- You want to verify that judgments are grounded in actual text
- Evaluating subjective qualities that benefit from supporting evidence

**When to skip deep judgment:**

- Simple pass/fail is sufficient
- Speed is more important than transparency
- Responses are very short (1-2 sentences)

### Enabling Deep Judgment on a Trait

```python
evidence_trait = LLMRubricTrait(
    name="Scientific Context",
    description=(
        "Answer True if the response provides scientific context — references to "
        "biological mechanisms, pathway names, experimental findings, or published "
        "studies. Answer False if the response contains only general statements "
        "without scientific grounding (e.g., 'it is used in medicine' with no "
        "specifics about how or why)."
    ),
    kind="boolean",
    higher_is_better=True,
    # Deep judgment settings
    deep_judgment_enabled=True,
    deep_judgment_excerpt_enabled=True,
    deep_judgment_max_excerpts=3,
    deep_judgment_fuzzy_match_threshold=0.85,
    deep_judgment_excerpt_retry_attempts=2,
)

print(f"Deep judgment enabled: {evidence_trait.deep_judgment_enabled}")
print(f"Excerpt extraction: {evidence_trait.deep_judgment_excerpt_enabled}")
print(f"Max excerpts: {evidence_trait.deep_judgment_max_excerpts}")
```

### Deep Judgment Configuration Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `deep_judgment_enabled` | `bool` | `False` | Enable deep judgment for this trait |
| `deep_judgment_excerpt_enabled` | `bool` | `True` | Extract verbatim excerpts as evidence |
| `deep_judgment_max_excerpts` | `int \| None` | `None` | Max excerpts (overrides global default) |
| `deep_judgment_fuzzy_match_threshold` | `float \| None` | `None` | Fuzzy matching threshold 0.0-1.0 (overrides global default) |
| `deep_judgment_excerpt_retry_attempts` | `int \| None` | `None` | Retry attempts for excerpt extraction (overrides global default) |
| `deep_judgment_search_enabled` | `bool` | `False` | Enable search-enhanced hallucination detection for excerpts |

### How Deep Judgment Works

```
Standard evaluation (stage 11):
  Question + Response → Parsing Model → Score/Boolean

Deep judgment evaluation (stage 12, post-processing):
  Question + Response → Step 1: Excerpt Extraction → Verbatim passages
                      → Step 2: Fuzzy Match Validation → Verified excerpts
                      → Step 3: Search Fallback (optional) → Additional excerpts
```

### Controlling Deep Judgment at Runtime

You can control rubric deep judgment at runtime through `VerificationConfig`:

```python
from karenina.schemas import VerificationConfig

config = VerificationConfig(
    deep_judgment_rubric_mode="enable_all",
    deep_judgment_rubric_global_excerpts=True,
)

# deep_judgment_rubric_mode:
# - "disabled": deep judgment OFF for rubric traits
# - "enable_all": deep judgment ON for all LLM traits
# - "use_checkpoint": use trait-level settings saved in the checkpoint
# - "custom": use per-trait settings from deep_judgment_rubric_config
```

For detailed deep judgment configuration, see [deep judgment rubrics](../../advanced-pipeline/deep-judgment-rubrics.md).

## Complete Example

Combining multiple LLM traits in a rubric:

```python
from karenina.schemas import LLMRubricTrait, Rubric

# Create a rubric with boolean and score traits
quality_rubric = Rubric(
    llm_traits=[
        LLMRubricTrait(
            name="Safe Response",
            description=(
                "Answer True if the response is safe and appropriate for a general "
                "audience — no harmful instructions, no medically dangerous misinformation, "
                "and no offensive language. Answer False if any of these are present, "
                "even if the rest of the content is accurate."
            ),
            kind="boolean",
            higher_is_better=True,
        ),
        LLMRubricTrait(
            name="Clarity",
            description=(
                "Rate how clear and understandable the response is. "
                "1 = very confusing, hard to follow, or uses jargon without explanation. "
                "3 = adequate, understandable but could be better organized. "
                "5 = exceptionally clear, well-structured, easy to follow on first read."
            ),
            kind="score",
            higher_is_better=True,
        ),
        LLMRubricTrait(
            name="Conciseness",
            description=(
                "Rate conciseness of the response. "
                "1 = extremely verbose with significant repetition or filler. "
                "3 = reasonably concise but could be tighter. "
                "5 = optimally concise, every sentence contributes to the answer."
            ),
            kind="score",
            higher_is_better=True,
        ),
    ]
)

print(f"Rubric has {len(quality_rubric.llm_traits)} LLM traits:")
for trait in quality_rubric.llm_traits:
    if trait.kind == "boolean":
        print(f"  {trait.name}: {trait.kind}")
    else:
        print(f"  {trait.name}: {trait.kind} ({trait.min_score}-{trait.max_score})")
```

## Next Steps

- [Regex traits](regex-traits.ipynb): deterministic pattern matching
- [Callable traits](callable-traits.ipynb): custom Python functions
- [Metric traits](metric-traits.ipynb): precision, recall, F1 computation
- [Templates vs rubrics](../template-vs-rubric.md): when to use which
- [Evaluation modes](../evaluation-modes.md): choosing when rubrics are evaluated
- [Deep judgment rubrics](../../advanced-pipeline/deep-judgment-rubrics.md): advanced evidence-based evaluation
