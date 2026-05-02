---
jupyter:
  jupytext:
    formats: docs/core_concepts/rubrics//md,docs/notebooks/core_concepts/rubrics//ipynb
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

# LLM Rubric Traits

LLM rubric traits use the **parsing model as an evaluator LLM** to judge observable properties of a model's raw response trace. They are the flexible rubric trait type: use them when the check requires language understanding, interpretation, or classification rather than exact pattern matching or deterministic logic. For an overview of all rubric trait types, see the [rubrics index](../../../../core_concepts/rubrics/).

```python tags=["hide-cell"]
# Mock cell: ensures examples execute without live API keys.
# This cell is hidden in rendered documentation.
```

## 1. What LLM Rubric Traits Are

An `LLMRubricTrait` sends the original question, the model's raw response trace, and the trait definition to the parsing model. The parsing model makes an LLM judgment and returns a structured result.

LLM rubric traits are meant for qualities you can judge by reading the answer itself, without needing ground truth. Typical examples include whether a biomedical answer presents evidence in a usable style rather than a vague one, hedges appropriately when discussing off-label use, or develops its reasoning as a coherent chain rather than a pile of disconnected claims.

Use `LLMRubricTrait` when the evaluation genuinely needs semantic judgment. If the check can be expressed as an exact pattern or a Python-defined rule, prefer [Regex traits](../regex-traits/) or [Callable traits](../callable-traits/). Pure local regex/callable checks are faster, cheaper, and more reproducible.

### 1.1 Philosophy

The most important idea is that the trait definition is the evaluation spec. In particular, the `description` field is not just documentation for humans. Karenina passes it to the evaluator LLM to tell it what to look for, what counts as success, and where the boundaries are.

That means good LLM traits describe **observable evidence in the response text**:

- what the evaluator should look for
- what should count as a positive result
- what similar-looking cases should still count as negative

**The ground-truth boundary.** LLM traits judge *how* the response is written, not *what* it got right. Checking whether the response identified the correct drug target, addressed the right clinical aspect, or provided an accurate mechanism requires knowing the correct answer. That is the template's job: the judge parses the response into structured fields and `verify()` checks them against ground truth. A rubric trait has no access to ground truth, so it can only evaluate qualities that are visible in the text itself.

A useful litmus test: if the evaluator cannot judge the trait without knowing the correct answer, it belongs in the template, not in a rubric trait.

| Needs ground truth (use a template) | Observable in text (use an LLM trait) |
|--------------------------------------|---------------------------------------|
| "Does the response correctly identify the drug target?" | "Does the response support its claims with named trials or specific data?" |
| "Does the response address the requested clinical aspect?" | "Does the response develop reasoning as a sequence of linked steps?" |
| "Is the mechanism of action accurate?" | "Does the response hedge appropriately when discussing off-label use?" |
| "Does the response name the right guideline?" | "Does the response present evidence in a structured, readable format?" |

If you write "Is the response good?", the evaluator has to guess your standard. If you write "Answer True only if the response supports each claim with at least one named study, date, or quantitative result," the evaluator has a concrete rubric to apply.

## 2. Overview

Four kinds are available: three scalar kinds that return a single value per trait, and one template kind that returns a structured multi-field finding.

| Kind | Returns | Best For |
|------|---------|----------|
| **boolean** | `True` / `False` | Binary judgments with explicit pass/fail boundaries |
| **score** | `int` in a configurable range | Gradable qualities on a scale |
| **literal** | `int` (class index) | Categorical classification with named classes |
| **template** (Pydantic `BaseModel` subclass) | `dict` of typed fields (flattened under dotted keys) | Multi-field findings that are naturally evaluated together |

In the scalar cases, the evaluator LLM reads the question, the raw response trace, and the trait definition, then returns a structured result for Karenina to store in `VerificationResult.rubric`. The template case uses the same single parsing call but fills in every field of a user-defined Pydantic schema at once.

Choose the kind based on the shape of the judgment:

- Use **boolean** when the answer either meets the criterion or does not.
- Use **score** when the quality varies along a continuum.
- Use **literal** when you can define distinct named categories.
- Use **[template](#7-template-kind)** when one evaluation naturally produces several related fields that you would otherwise split across multiple scalar traits.

In addition to `name`, `description`, and `kind`, every `LLMRubricTrait` accepts an optional `summary: str | None` field (a short concept label used by the [dynamic rubric](../../../../core_concepts/rubrics/#6-dynamic-rubric) presence check; falls back to `description` when unset).

## 3. Why the `description` Field Matters

LLM traits are evaluated during [RubricEvaluation](../../verification-pipeline/) of the [verification pipeline](../../verification-pipeline/). During evaluation, the [prompt assembler](../../../../advanced-pipeline/prompt-assembly/) builds a message for the parsing model containing: a **system prompt** assigning the evaluator role, and a **user prompt** with the original question text, the model's full response trace, and your trait definition.

```
Previous Stages
                  │
                  ▼
┌─── RubricEvaluation ────────────────────────────────┐
│                                                      │
│  Question + Response Trace + Trait Definitions       │
│                     │                                │
│                     ▼                                │
│          ┌──────────────────┐                        │
│          │   Parsing Model  │  ← System: evaluator   │
│          │    (LLM call)    │     role               │
│          │                  │  ← User: question +    │
│          │                  │    trace + trait defs  │
│          └────────┬─────────┘                        │
│                   ▼                                  │
│  Structured result: bool or int per trait            │
└───────────────────┬──────────────────────────────────┘
                    │
                    ▼
DeepJudgmentRubric auto-fail check (if configured)
                    │
                    ▼
FinalizeResult
```

This chart shows the standard LLM-trait evaluation flow. If deep judgment is enabled, the LLM-trait work inside the RubricEvaluation stage becomes a multi-step evidence-based path; that variant is shown in [Deep Judgment](#9-deep-judgment-optional) below.

For **boolean** and **score** traits, the `description` field is the main channel through which you tell the evaluator LLM what to evaluate and how. In practice, the more concrete the criteria, boundary cases, and domain context, the more reliable the judgment tends to be.

For **literal** traits, the parsing model receives both the top-level `description` and the class descriptions in `classes`. In practice, the class descriptions do most of the work, while the top-level description provides scope and context.

Two factors control evaluation quality:

1. **Trait-definition quality**: For boolean and score traits, this mostly means the `description`. For literal traits, it means the combination of the top-level `description` and the class descriptions. Detailed, observable criteria with explicit boundary cases produce more consistent judgments.
2. **Model capability**: More capable parsing models interpret nuanced descriptions more faithfully. If you need fine-grained distinctions, use a stronger model. Simpler judgments are reliable even with smaller models.

If multiple LLM traits exist on the same rubric, Karenina uses `VerificationConfig.rubric_evaluation_strategy` to decide whether to evaluate them in a single parsing call (`"batch"`, the default) or one by one (`"sequential"`). Results are stored in `VerificationResult.rubric` and become available for analysis and DataFrame export. See the [VerificationConfig reference](../../../../reference/configuration/verification-config/) for the full field definition.

## 4. Boolean Kind

Boolean traits answer a yes/no question about the response. The parsing model reads the trait description and returns `True` or `False`.

**When to use:**

- Presence or absence checks that require interpretation
- Compliance or safety checks with explicit textual evidence
- Style or content requirements with a clear pass/fail boundary

### 4.1 Writing Boolean Descriptions

A good boolean description defines exactly what evidence makes the answer True or False, including the cases that are hardest to classify. Describe what is observable in the text, not abstract quality labels.

```python
from karenina.schemas.entities.rubric import LLMRubricTrait
```

```python
LLMRubricTrait(
    name="Cites Specific Evidence",
    description=(
        "Answer True if the response supports its claims by naming at least one "
        "concrete piece of evidence: a specific trial name, a study author, a "
        "publication year, a quantitative result (e.g., hazard ratio, response rate), "
        "or a guideline reference. Generic phrases such as 'studies suggest' or "
        "'research has shown' do not count. A response that names even one specific "
        "source counts as True. A response that relies entirely on unsourced "
        "assertions counts as False."
    ),
    kind="boolean",
    higher_is_better=True,
)
```

This description works well because the evaluator only needs to scan the response text for named evidence; it never needs to know whether the cited evidence is accurate or relevant to the correct answer.

```python
LLMRubricTrait(
    name="Sequentially Supported Reasoning",
    description=(
        "Answer True if the response presents its reasoning as a sequence of linked "
        "steps, where later statements are supported by earlier ones. A response "
        "counts as False if it jumps between conclusions, lists disconnected claims, "
        "or introduces new assertions without showing how they follow from what was "
        "already said."
    ),
    kind="boolean",
    higher_is_better=True,
)
```

This description stays grounded in the text. The evaluator is judging the internal organization of the reasoning, not whether the underlying biomedical claims are correct.

### 4.2 Boolean Trait with Inverted Directionality

Sometimes a `True` result indicates a negative outcome. Set `higher_is_better=False` to signal that `True` is bad:

```python
overclaim_trait = LLMRubricTrait(
    name="Contains Unsupported Certainty",
    description=(
        "Answer True if the response makes definitive claims (e.g., 'X is the standard "
        "treatment,' 'Y has been proven to') without citing a named study, guideline, "
        "numeric result, or other concrete source in the same sentence or the sentence "
        "immediately following. A single unsupported definitive claim is enough for True. "
        "Answer False if every strong claim is accompanied by a specific reference, or if "
        "the response uses appropriately hedged language throughout (e.g., 'evidence "
        "suggests,' 'some trials indicate')."
    ),
    kind="boolean",
    higher_is_better=False,  # True = unsupported certainty found = bad
)

print(f"Trait: {overclaim_trait.name}")
print(f"higher_is_better: {overclaim_trait.higher_is_better}")
# Analysis tools know that True here means worse performance
```

## 5. Score Kind

Score traits rate a quality on a numeric scale. The default range is 1-5, but you can customize it with `min_score` and `max_score`.

**When to use:**

- Gradable qualities such as clarity, conciseness, or sequential coherence
- Cases where several answers may all be acceptable, but some are stronger than others
- Qualities that vary along a continuum rather than a small set of named categories

### 5.1 Writing Score Descriptions

A good score description anchors the scale with concrete examples at key points (low, middle, high). This gives the judge observable criteria for each level rather than an abstract number to pick.

```python
LLMRubricTrait(
    name="Clinical Clarity",
    description=(
        "Rate how clear the response is for a clinician reading quickly. "
        "1 = disorganized, jargon-heavy, or hard to follow. "
        "3 = understandable but unevenly structured or partially vague. "
        "5 = easy to follow, well organized, and precise without unnecessary jargon."
    ),
    kind="score",
    higher_is_better=True,
)
```

Anchoring at 1, 3, and 5 with concrete criteria leaves room for the evaluator to interpolate. Use score when the quality is genuinely continuous or when distinct levels are not cleanly separable. If you can define named levels with clear boundaries, use a [literal trait](#6-literal-kind) instead.

### 5.2 Creating a Score Trait

```python
conciseness_trait = LLMRubricTrait(
    name="Conciseness",
    description=(
        "Rate how concise the response is for the question asked. "
        "1 = verbose, repetitive, or padded with background that does not help answer "
        "the question. "
        "3 = mostly focused but could be tighter. "
        "5 = direct and efficient, with essentially every sentence contributing useful information."
    ),
    kind="score",
    higher_is_better=True,  # Higher score = better conciseness
)

print(f"Trait: {conciseness_trait.name}")
print(f"Kind: {conciseness_trait.kind}")
print(f"Score range: {conciseness_trait.min_score}-{conciseness_trait.max_score}")
```

### 5.3 Custom Score Range

The default range is 1-5. You can change it:

```python
sequentiality_trait = LLMRubricTrait(
    name="Sequential Reasoning Quality",
    description=(
        "Rate how well the response builds its reasoning step by step. "
        "1 = claims are disconnected or abrupt, with little visible chaining. "
        "5 = some sequential structure is present, but important jumps remain. "
        "10 = reasoning unfolds in a clear chain, with each step supported by the "
        "previous one or by explicitly introduced evidence."
    ),
    kind="score",
    min_score=1,
    max_score=10,
    higher_is_better=True,
)

print(f"Score range: {sequentiality_trait.min_score}-{sequentiality_trait.max_score}")
```

### 5.4 Score Validation

The `validate_score` method checks whether a given value is valid for a trait:

```python
# Score trait: accepts integers in [min_score, max_score]
print(conciseness_trait.validate_score(3))     # True - valid score
print(conciseness_trait.validate_score(6))     # False - above max_score
print(conciseness_trait.validate_score(True))  # False - booleans rejected for score traits
```

## 6. Literal Kind

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
| `higher_is_better` | `bool \| None` | Whether higher class indices indicate better performance. `None` means directionality does not apply. Defaults to `True`. |

Key characteristics:

- **Ordered**: Dictionary order determines indices (0, 1, 2, ...)
- **Auto-ranged**: `min_score` is set to 0, `max_score` to `len(classes) - 1` automatically
- **Descriptive**: Each class has a name and a description to guide the parsing model
- **2-20 classes**: Must have at least 2 and at most 20 categories

Even when your classes are conceptually distinct rather than "better" or "worse," Karenina still stores them as indices because that is how literal traits are represented internally. In those cases, treat the class **label** as the primary result and the numeric index as a stable storage convention.

### 6.1 Writing Good Class Descriptions

The quality of class descriptions directly affects how well the parsing model classifies responses. Each class description should be:

- **Mutually exclusive**: clearly distinct from other classes
- **Observable**: describe what the model should look for in the response text
- **Ordered consistently**: if using `higher_is_better`, ensure the natural ordering matches

**Good** (clear criteria the model can evaluate):

    "unsupported": "Claims are stated without any supporting evidence, citations, or concrete data"
    "anecdotal": "Some claims cite examples or general references, but no quantitative data or specific sources"
    "partially_sourced": "Key claims include specific data points or named sources, but at least one major claim lacks support"
    "well_sourced": "Every major claim is backed by a specific data point, named source, or explicit citation"

**Weak** (vague or overlapping):

    "bad": "A bad answer"
    "ok": "An okay answer"
    "good": "A good answer"

Here is a more detailed example with four classes for a different text-observable quality. Each class describes what is visible in the response text, and the boundaries are explicit.

```python
LLMRubricTrait(
    name="Structural Accessibility",
    description="Classify how well the response uses formatting and signposting to make its content easy to scan and navigate in a biomedical answer.",
    kind="literal",
    classes={
        "dense_wall": (
            "The response is a continuous block of text with no paragraph breaks, "
            "headings, or transition phrases to separate distinct points"
        ),
        "loosely_broken": (
            "The response has some paragraph breaks, but sections lack clear topic "
            "sentences and transitions between points are abrupt or missing"
        ),
        "signposted": (
            "The response uses paragraph breaks with recognizable topic sentences or "
            "transitional phrases, making it possible to skim for specific points"
        ),
        "well_scaffolded": (
            "The response uses clear section markers (headings, numbered lists, or "
            "explicit labels), each section opens with a topic sentence, and "
            "transitions between sections are smooth"
        ),
    },
    higher_is_better=True,
)
```

### 6.2 How Literal Evaluation Works

The parsing model receives the trait description, class names, and class descriptions as part of the evaluation prompt. It classifies the response by selecting the best-matching class name, which is then converted to an integer index:

```
Question + Response + Trait Description + Class Definitions
                ↓
         Parsing Model
                ↓
    Selects class name (e.g., "good_fit")
                ↓
    Converted to index (e.g., 2)
```

This two-step process (classify by name, then map to index) is more reliable than asking the model to pick a number directly.

### 6.3 Distinct Categories Example

```python
evidence_style_trait = LLMRubricTrait(
    name="Response Framing",
    description="Classify the main framing the response chooses for a biomedical answer.",
    kind="literal",
    classes={
        "mechanism_first": "Organizes the answer mainly around biology, pathway, or mechanism explanation",
        "clinical_first": "Organizes the answer mainly around indication, patient use, or practice context",
        "evidence_first": "Organizes the answer mainly around studies, results, or supporting evidence",
        "caveat_first": "Organizes the answer mainly around uncertainty, limitations, or exceptions",
    },
    higher_is_better=False,
)

print(f"Kind: {evidence_style_trait.kind}")
print(f"Classes: {list(evidence_style_trait.classes.keys())}")
print(f"Score range: {evidence_style_trait.min_score} to {evidence_style_trait.max_score}")
```

The parsing model receives the class names and descriptions, then selects the one that best fits the response. The result is the class index:

- `0` → "mechanism_first"
- `1` → "clinical_first"
- `2` → "evidence_first"
- `3` → "caveat_first"

This is a good example of **distinct categories** rather than a quality ladder. There is no intrinsic "best" class here. The order is still stored numerically, but that ordering is mainly a convention for storage and downstream tooling. For display and interpretation, the label is usually more important than the index.

### 6.4 Quality Tiers

A second common pattern is defining quality levels where order is meaningful:

```python
quality_trait = LLMRubricTrait(
    name="Sequential Reasoning Quality",
    description="Classify how well the response builds a step-by-step chain of reasoning in a biomedical answer.",
    kind="literal",
    classes={
        "fragmented": "Presents conclusions or facts as separate pieces with little visible logical chaining",
        "loosely_chained": "Shows some ordering, but several steps are implied rather than clearly connected",
        "well_chained": "Builds a mostly clear sequence where later claims follow from earlier ones",
        "tightly_chained": "Builds a clear, explicit chain where each step is motivated and supports the next",
    },
    higher_is_better=True,  # Higher index = better quality
)

print(f"Classes: {list(quality_trait.classes.keys())}")
print(f"Score range: {quality_trait.min_score} to {quality_trait.max_score}")
print(f"higher_is_better: {quality_trait.higher_is_better}")
```

Here `higher_is_better=True` because later classes (higher indices) represent better quality:

- `0` → "fragmented" (worst)
- `1` → "loosely_chained"
- `2` → "well_chained"
- `3` → "tightly_chained" (best)

### 6.5 Working with Class Names

`LLMRubricTrait` provides helper methods for converting between class names and indices:

```python
# Get the ordered list of class names
print(f"Class names: {quality_trait.get_class_names()}")

# Get the index for a specific class
print(f"Index of 'well_chained': {quality_trait.get_class_index('well_chained')}")
print(f"Index of 'fragmented': {quality_trait.get_class_index('fragmented')}")

# Invalid class names return -1
print(f"Index of 'unknown': {quality_trait.get_class_index('unknown')}")
```

The `get_class_index()` method returns `-1` for unrecognized class names. This value is also accepted by `validate_score()` as a valid error state for literal traits.

### 6.6 Literal Score Validation

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

Literal traits are evaluated through the standard classification path. The [deep judgment](#9-deep-judgment-optional) guidance below is intended for boolean and score traits.

## 7. Template Kind

The scalar kinds ([boolean](#4-boolean-kind), [score](#5-score-kind), [literal](#6-literal-kind)) each return a single value per trait. Template kind lets you pass a Pydantic `BaseModel` subclass as `kind`, and the judge LLM populates the entire schema in one structured-output call, producing a multi-field evaluation finding from a single trait instead of several scalar traits that share the same underlying investigation.

Unlike the [agentic template kind](../agentic-traits/#54-template-kind-agentic-evaluation-with-structured-output), which launches an agent with tool access, the LLM-trait template kind uses the same single parsing call as scalar LLM traits. The parsing model reads the question and the response trace, then fills in every field of the schema at once. There is no agent, no tool use, and no workspace access. Use this variant when the evaluation can be judged from the response text alone but naturally produces several related findings that are awkward to split across individual boolean/score/literal traits (for example, a compliance audit with multiple criteria, a structured summary of rhetorical moves, or a tagged list of cited sources). If you need to inspect files, run code, or explore workspace state, use the [agentic template kind](../agentic-traits/#54-template-kind-agentic-evaluation-with-structured-output) instead.

### 7.1 Constraints

Template kind imposes several constraints to keep the structured output serializable and unambiguous:

- **Field types.** Each field must be a primitive (`bool`, `int`, `float`, `str`) or a `list` of primitives (`list[str]`, `list[int]`, ...). Nested `BaseModel`s, dictionaries, tuples, and non-primitive lists are rejected at trait construction time.
- **Optional fields need explicit defaults.** An `Optional[T]` or `T | None` field must declare a default (commonly `None`), so the JSON Schema round-trips faithfully when the rubric is serialized to a checkpoint.
- **`higher_is_better` must be `None`.** A structured multi-field result has no single direction; analysis tools cannot aggregate it into a "better/worse" score the way they can for scalar kinds. The validator rejects any other value.
- **`deep_judgment_enabled` must be `False`.** [Deep judgment](#9-deep-judgment-optional) produces excerpts and reasoning for a single scalar score; it is mutually exclusive with template kind, which has no scalar score to reason about.
- **`classes`, `min_score`, and `max_score` are ignored.** Those fields apply only to scalar kinds; template kind derives its output shape entirely from the Pydantic schema.

### 7.2 Minimal Example

```python
from pydantic import BaseModel, Field
from karenina.schemas.entities.rubric import LLMRubricTrait, Rubric


class CitationFindings(BaseModel):
    """The judge reads the response and fills this in."""

    cites_named_trial: bool = Field(
        description="True if the response names at least one specific clinical trial."
    )
    citation_count: int = Field(
        description="Number of distinct named sources (trials, authors, guidelines) cited."
    )
    cited_sources: list[str] = Field(
        description="Short labels for each named source (e.g., 'KEYNOTE-189', 'NCCN 2024')."
    )


citation_trait = LLMRubricTrait(
    name="citation_audit",
    description=(
        "Inspect the response for citations of clinical evidence. Record whether any "
        "specific trial is named, count the distinct named sources, and list their labels. "
        "Do not infer sources that are not explicitly named in the response."
    ),
    kind=CitationFindings,
    higher_is_better=None,  # required: no single direction for multi-field output
)

rubric = Rubric(llm_traits=[citation_trait])

print(f"Trait: {citation_trait.name}")
print(f"Template kind: {citation_trait.is_template_kind}")
print(f"Fields: {list(CitationFindings.model_fields.keys())}")
```

### 7.3 Result Shape

Template traits flatten their structured output into `VerificationResultRubric.llm_trait_scores` using dotted keys. Each field of the Pydantic schema becomes an entry `"trait_name.field_name"`, mirroring the convention used for [agentic template traits](../agentic-traits/#54-template-kind-agentic-evaluation-with-structured-output) under `agentic_trait_scores`.

For the example above, a successful evaluation stores:

```text
result.rubric.llm_trait_scores == {
    "citation_audit.cites_named_trial": True,
    "citation_audit.citation_count": 2,
    "citation_audit.cited_sources": ["KEYNOTE-189", "NCCN 2024"],
}
```

The bare trait name (`"citation_audit"`) does **not** appear in the successful case. It is reserved for the failure path: if the parsing call fails or the model returns output that cannot be validated against the schema, the pipeline stores `llm_trait_scores["citation_audit"] = None` instead of the dotted keys. Analysis code should therefore check for the bare key when deciding whether the trait evaluated successfully.

### 7.4 Evaluation Strategy Interaction

Template traits are always evaluated **one call per trait** because each trait contributes a distinct Pydantic schema that cannot share a batched output model with other traits. When a rubric mixes template and scalar traits and `VerificationConfig.rubric_evaluation_strategy="batch"`, template traits still run one at a time (in parallel when async is enabled), while the scalar traits in the same rubric continue to share a single batched call. Setting `rubric_evaluation_strategy="sequential"` affects only the scalar traits; it does not change template-trait behavior.

## 8. The `higher_is_better` Field

This field (type `bool | None`, default `True`) tells analysis tools how to interpret results:

| Kind | `higher_is_better=True` | `higher_is_better=False` | `higher_is_better=None` |
|------|------------------------|--------------------------|------------------------|
| boolean | `True` = positive outcome | `True` = negative outcome | Directionality does not apply |
| score | Higher scores = better | Higher scores = worse | Directionality does not apply |
| literal | Higher class indices are interpreted as better | Higher class indices are interpreted as worse | Directionality does not apply |
| template | Not allowed (validation error) | Not allowed (validation error) | Required |

Most traits use `higher_is_better=True` (the default). Use `False` for traits where a positive detection is bad (for example, scope drift detected or prohibited content present). Use `None` when the trait has no meaningful directionality.

For literal traits, `higher_is_better` does **not** affect classification itself. It only affects how downstream tooling interprets the numeric indices. If your classes are distinct categories rather than a quality ladder, choose a stable convention and rely on the class labels for human interpretation.

For [template kind](#7-template-kind), `higher_is_better` must be `None`; the validator rejects any other value because a multi-field structured result has no single direction to aggregate.

## 9. Deep Judgment (Optional)

Without deep judgment, each LLM trait is evaluated in a single parsing call: the model reads the question, the response trace, and the trait definition, then returns a score or boolean directly. Deep judgment replaces that single call with a multi-step evidence-based evaluation that produces both a judgment and the textual evidence behind it.

Deep judgment is currently best suited to **boolean** and **score** traits. The pipeline extracts booleans or numeric scores and does not consume literal class definitions during scoring. In practice, use standard evaluation for literal traits.

### 9.1 What Happens When Deep Judgment Is On

Deep-judgment traits are evaluated **one at a time** (never batched) during the RubricEvaluation pipeline stage. Any traits on the same rubric that do not have deep judgment enabled are evaluated separately through the standard path. The sequence for each deep-judgment trait depends on whether excerpt extraction is enabled.

**With excerpts** (`deep_judgment_excerpt_enabled=True`, the default):

1. **Excerpt extraction.** The parsing model receives the question, response trace, and trait definition, then returns verbatim quotes from the response that are relevant to the trait. This uses structured output parsing.
2. **Fuzzy-match validation.** Each excerpt is checked against the original response text using `difflib.SequenceMatcher`. An excerpt passes if its similarity score meets the configured threshold (default 0.80). If any excerpt fails validation, the model retries with feedback explaining which excerpts were rejected and why, up to `deep_judgment_excerpt_retry_attempts` times.
3. **Hallucination assessment (optional).** If `deep_judgment_search_enabled=True`, each validated excerpt is checked against web search results (Tavily by default). The result is a per-excerpt risk level ("none", "low", "medium", "high"), with the overall risk set to the maximum across excerpts.
4. **Reasoning generation.** The parsing model generates free-form reasoning based on the excerpts (and hallucination assessment, if present).
5. **Score extraction.** A final structured-output call extracts the boolean or numeric score from the reasoning. If structured parsing fails, Karenina falls back to regex parsing of the plain-text response.

**Without excerpts** (`deep_judgment_excerpt_enabled=False`):

1. **Reasoning generation.** The parsing model generates reasoning directly from the full response (no excerpt step).
2. **Score extraction.** Same as above.

This path is faster (2 LLM calls per trait instead of 3+) but provides less verifiable evidence.

**After all traits are evaluated**, the DeepJudgmentRubric stage (Stage 12) runs an auto-fail check. If any trait exhausted its retries without producing valid excerpts, the entire verification result is auto-failed. The auto-fail is skipped if abstention was already detected (abstention takes priority).

**When to use deep judgment:**

- Transparency and auditability are important
- You want to verify that judgments are grounded in actual text
- Evaluating subjective qualities that benefit from supporting evidence

**When to skip deep judgment:**

- Simple pass/fail is sufficient
- Speed is more important than transparency
- Responses are very short (1-2 sentences)

### 9.2 Enabling Deep Judgment on a Trait

```python
evidence_trait = LLMRubricTrait(
    name="Uses Specific Evidence",
    description=(
        "Answer True if the response supports its biomedical claims with concrete "
        "evidence such as named trials, study authors, publication years, response "
        "rates, hazard ratios, or guideline references. Generic phrases like "
        "'studies have shown' count as False."
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

### 9.3 Deep Judgment Configuration Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `deep_judgment_enabled` | `bool` | `False` | Enable deep judgment for this trait |
| `deep_judgment_excerpt_enabled` | `bool` | `True` | Extract verbatim excerpts as evidence |
| `deep_judgment_max_excerpts` | `int \| None` | `None` | Max excerpts (overrides global default of 7) |
| `deep_judgment_fuzzy_match_threshold` | `float \| None` | `None` | Fuzzy matching threshold 0.0-1.0 (overrides global default of 0.80) |
| `deep_judgment_excerpt_retry_attempts` | `int \| None` | `None` | Retry attempts for excerpt extraction (overrides global default of 2) |
| `deep_judgment_search_enabled` | `bool` | `False` | Enable search-enhanced hallucination detection for excerpts |

Per-trait fields set to `None` fall back to the global defaults on `VerificationConfig`.

### 9.4 Controlling Deep Judgment at Runtime

You can control rubric deep judgment at runtime through `VerificationConfig`:

```python
from karenina.schemas import VerificationConfig
from karenina.schemas.config import ModelConfig

config = VerificationConfig(
    parsing_only=True,
    parsing_models=[
        ModelConfig(
            id="haiku-parser",
            model_name="claude-haiku-4-5",
            model_provider="anthropic",
            interface="langchain",
            temperature=0.0,
        )
    ],
    deep_judgment_rubric_mode="enable_all",
    deep_judgment_rubric_global_excerpts=True,
)

# deep_judgment_rubric_mode:
# - "disabled": deep judgment OFF for rubric traits (default)
# - "enable_all": deep judgment ON for all LLM traits
# - "use_checkpoint": use trait-level settings saved in the checkpoint
# - "custom": use per-trait settings from deep_judgment_rubric_config
```

For detailed configuration (four modes, per-trait overrides, result fields, cost considerations), see [deep judgment rubrics](../../../../advanced-pipeline/deep-judgment-rubrics/).

## 10. Next Steps

- [Regex traits](../regex-traits/): deterministic pattern matching
- [Callable traits](../callable-traits/): custom Python functions
- [Metric traits](../metric-traits/): precision, recall, F1 computation
- [Evaluation modes](../../evaluation-modes/): choosing when rubrics are evaluated
- [Deep judgment rubrics](../../../../advanced-pipeline/deep-judgment-rubrics/): advanced evidence-based evaluation
