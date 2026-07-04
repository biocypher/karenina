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

# Callable Traits

Callable traits evaluate LLM responses using **custom Python functions**. They are the rubric trait type for **author-defined evaluation logic**: use them when the built-in trait types are not the right interface for the assessment you want, or when you want to own the evaluation procedure yourself. They give you full programmatic control over evaluation logic, from simple word count checks to domain-specific scoring.

```python tags=["hide-cell"]
# Mock cell: ensures examples execute without live API keys.
import warnings
warnings.filterwarnings("ignore", message="Deserializing callable")
```

## 1. What Callable Traits Are

A `CallableRubricTrait` wraps a Python function that runs locally during [RubricEvaluation](../../verification-pipeline/) of the [verification pipeline](../../verification-pipeline/). The function receives the configured rubric evaluation input as a single string argument and returns a boolean (pass/fail), a numeric score (int or float), or a string class label (for literal/categorical classification).

Callable traits are meant for checks that need **custom programmatic logic**. Typical examples include minimum length requirements, repetition checks, sentence counts, heuristic term counts, or custom evaluators that combine multiple strategies under one Python entrypoint.

Use `CallableRubricTrait` when the built-in traits do not express the evaluation shape you need cleanly. If the check is an exact text pattern, prefer [Regex traits](../regex-traits/). If the check fits Karenina's built-in semantic judgment path, prefer [LLM traits](../llm-traits/). Reach for `CallableRubricTrait` when you need custom orchestration, validation, or scoring logic beyond those built-in abstractions.

### 1.1 Philosophy

The most important idea is that the Python function is the evaluation spec. In callable traits, the function body defines what counts as success, failure, or score.

That means good callable traits define **explicit code-level rules**:

- one string input
- explicit, inspectable Python logic
- a clearly bounded output type: boolean, numeric (int or float), or string class label

**The abstraction boundary.** Callable traits are best thought of as the escape hatch trait type. In principle, a callable can re-implement the behavior of other trait types or even call an external model. In practice, you should prefer the built-in traits when they already match the assessment you want:

- use [Regex traits](../regex-traits/) for exact textual predicates
- use [LLM traits](../llm-traits/) for Karenina-managed semantic judgment
- use [Metric traits](../metric-traits/) for checklist-style precision/recall evaluation

Reach for `CallableRubricTrait` when your assessment does not fit those built-in shapes cleanly, or when you need custom orchestration across them.

| Better fit for Callable Traits | Usually better fit for built-in tools |
|--------------------------------|---------------------------------------|
| "Run a custom scoring heuristic over the response text" | "Does the response match this exact citation format?" → [Regex trait](../regex-traits/) |
| "Combine several local checks into one programmatic score" | "Does the answer use evidence convincingly?" → [LLM trait](../llm-traits/) |
| "Call a custom evaluator pipeline that Karenina does not model directly" | "Did the parsed answer match the gold structured fields?" → template verification |

A useful litmus test: if you find yourself wanting an assessment that Karenina's built-in trait types do not model cleanly, but you can express it behind a single Python function, a callable trait is probably the right abstraction.

## 2. Overview

Karenina evaluates callable traits by running your Python function locally. The difference from [regex traits](../regex-traits/) is that callable traits can implement arbitrary Python logic while regex traits are limited to pattern matching.

The function is serialized using [cloudpickle](https://github.com/cloudpipe/cloudpickle) so it can be stored in checkpoint files. That makes callable traits portable across Karenina workflows, but it also means deserializing them executes Python code.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | *(required)* | Human-readable identifier |
| `description` | `str \| None` | `None` | What this trait evaluates |
| `summary` | `str \| None` | `None` | Short concept label used by the [dynamic rubric](../../../../core_concepts/rubrics/#6-dynamic-rubric) presence check; falls back to `description` when unset |
| `kind` | `str` | *(required)* | `"boolean"` for pass/fail, `"score"` for numeric, `"literal"` for categorical |
| `callable_code` | `bytes` | *(required)* | Serialized function (cloudpickle) |
| `classes` | `dict[str, str] \| None` | `None` | Class name to description mapping (required if `kind="literal"`) |
| `min_score` | `int \| None` | `None` | Minimum score (required if `kind="score"`, auto-derived for `kind="literal"`) |
| `max_score` | `int \| None` | `None` | Maximum score (required if `kind="score"`, auto-derived for `kind="literal"`) |
| `invert_result` | `bool` | `False` | Invert the boolean result (only for `kind="boolean"`) |
| `higher_is_better` | `bool \| None` | `True` | Whether higher return values indicate better performance. `None` means directionality does not apply. |

**Key characteristics:**

- Returns **boolean**, **int**, **float**, or **class index** depending on `kind`
- Karenina itself makes no evaluator LLM call; it simply runs your function
- Function must accept exactly **one `str` parameter**
- Function is serialized via cloudpickle
- Use `from_callable()` to create traits

Whether evaluation is deterministic, reproducible, cheap, or latency-free depends on what your function does. A pure local function has those properties. A function that calls an external API or another LLM does not.

## 3. How Callable Evaluation Works

During rubric evaluation, Karenina executes the callable locally:

```
Previous Stages
                  │
                  ▼
┌─── RubricEvaluation ─────────────────────────────┐
│                                                   │
│  Rubric evaluation input (str)                    │
│           │                                       │
│           ▼                                       │
│   Your Python function(text)                      │
│           │                                       │
│           ▼                                       │
│   Returns bool, int, float, or str                │
│           │                                       │
│   Apply invert_result (if boolean and set)        │
│           │                                       │
│   Final result                                    │
└───────────┬───────────────────────────────────────┘
            │
            ▼
FinalizeResult → VerificationResult.rubric
```

Callable traits skip stage 12 (DeepJudgmentRubric), which applies only to [LLM traits](../llm-traits/) with [deep judgment](../../../../advanced-pipeline/deep-judgment-rubrics/) enabled.

If your function is pure local code, evaluation is deterministic: the same string input always produces the same result. If your function calls external services, including an LLM, reproducibility, latency, and cost depend on that implementation.

## 4. Why `from_callable()` Matters

Always use `CallableRubricTrait.from_callable()` rather than constructing a trait directly. It is the normal authoring interface because it validates the function signature and handles serialization for you.

```python
from karenina.schemas import CallableRubricTrait

minimum_length_trait = CallableRubricTrait.from_callable(
    name="Minimum Word Count",
    description="Response must contain at least 50 words",
    func=lambda text: len(text.split()) >= 50,
    kind="boolean",
    higher_is_better=True,
)

short_response = "The answer is BCL2."
long_response = "The drug target is BCL2. " * 20

print(minimum_length_trait.evaluate(short_response))  # False
print(minimum_length_trait.evaluate(long_response))   # True
```

If you think of callable traits as "serialized Python functions with metadata," `from_callable()` is the method that turns ordinary Python code into that stored form safely.

## 5. Boolean Callable Traits

Boolean callables are the simplest form: your function returns `True` or `False`.

They work well for explicit pass/fail rules such as:

- minimum or maximum length
- repetition detection
- presence of multiple required conditions
- local heuristics that reduce naturally to pass/fail

```python
repetition_trait = CallableRubricTrait.from_callable(
    name="No Excessive Repetition",
    description="Response should not repeat the same sentence excessively",
    func=lambda text: len({s.strip() for s in text.split(".") if s.strip()})
    < len([s.strip() for s in text.split(".") if s.strip()]) * 0.5,
    kind="boolean",
    invert_result=True,
    higher_is_better=True,
)

print(repetition_trait.evaluate("Unique sentence one. Unique sentence two. Unique sentence three."))  # True
print(repetition_trait.evaluate("Repeat this. Repeat this. Repeat this."))                            # False
```

## 6. Score Callable Traits

Score callables return a numeric value (int or float) within a defined range. You must specify `min_score` and `max_score`. Float return values are preserved (not truncated to int).

```python
def count_sentences(text: str) -> int:
    import re

    sentences = re.split(r"[.!?]+", text.strip())
    return len([s for s in sentences if s.strip()])

sentence_count_trait = CallableRubricTrait.from_callable(
    name="Sentence Count",
    description="Count the number of sentences in the response",
    func=count_sentences,
    kind="score",
    min_score=0,
    max_score=100,
    higher_is_better=True,
)

sample = "BCL2 is a proto-oncogene. It regulates apoptosis. It is located on chromosome 18."
print(f"Score: {sentence_count_trait.evaluate(sample)}")  # Score: 3
```

Use score callables when the rule is still programmatic but the output varies along a numeric scale rather than a simple pass/fail boundary.

## 7. `invert_result` and `higher_is_better`

These two fields solve different problems:

- `invert_result` changes the **boolean evaluation output**
- `higher_is_better` changes the **downstream interpretation**

`invert_result` only applies to boolean callable traits. Use it when your function naturally detects a bad condition but you want the final result to mean "passed the check."

The `higher_is_better` field controls how results are interpreted in aggregate analysis:

| `higher_is_better` | Meaning | Example |
|--------------------|---------|---------|
| `True` | Higher values = better performance | Length check pass, sentence count |
| `False` | Lower values = better performance | Error count, penalty score |

```python
error_count_trait = CallableRubricTrait.from_callable(
    name="Grammar Error Count",
    description="Count potential grammar errors (lower is better)",
    func=lambda text: text.count("  "),
    kind="score",
    min_score=0,
    max_score=50,
    higher_is_better=False,
)

print(f"Errors: {error_count_trait.evaluate('Clean  text with  two double  spaces.')}")  # Errors: 3
```

## 8. Literal Callable Traits

Literal callables classify the response into one of several predefined categories. Your function returns a string matching one of the class names. Karenina converts it to an integer index (0-based, following class order).

You must provide a `classes` dict mapping class names to descriptions (2 to 20 classes). `min_score` and `max_score` are auto-derived from the number of classes.

```python
tone_classifier = CallableRubricTrait.from_callable(
    name="Response Tone",
    description="Classify the tone of the response",
    func=lambda text: "formal" if "therefore" in text.lower() else "casual",
    kind="literal",
    classes={
        "formal": "Academic or professional tone with structured arguments",
        "casual": "Conversational, relaxed tone",
    },
    higher_is_better=True,
)

print(tone_classifier.evaluate("Therefore, the evidence suggests..."))  # 0 (formal)
print(tone_classifier.evaluate("Yeah, I think it's fine"))              # 1 (casual)
```

Use literal callables when you have a programmatic classifier that maps responses to discrete categories. If the classification needs LLM judgment, use an [LLM trait](../llm-traits/) with `kind="literal"` instead.

## 9. Serialization and Security

Callable traits are portable because the function is serialized with cloudpickle when you call `from_callable()`. This is powerful, but it introduces real constraints.

**Authoring constraints:**

- lambda functions are fine for simple logic
- module-level named functions are usually better for more complex logic
- closures over large objects serialize more state than you may expect
- unpicklable state such as file handles or locks cannot be serialized cleanly
- the deserializing environment must have the dependencies your function expects
- external calls inside the function change the trait's runtime behavior, cost, and reproducibility profile

You can retrieve the stored function using `deserialize_callable()`:

```python
length_trait = CallableRubricTrait.from_callable(
    name="Response Length",
    func=lambda text: len(text),
    kind="score",
    min_score=0,
    max_score=10000,
    higher_is_better=True,
)

func = length_trait.deserialize_callable()
print(f"Function result: {func('Hello world')}")  # Function result: 11
print(f"Trait evaluate:  {length_trait.evaluate('Hello world')}")  # Trait evaluate:  11
```

<div class="admonition warning">
<p class="admonition-title">Security Warning</p>
<p>Deserializing callable code can execute arbitrary Python code. Only load <code>CallableRubricTrait</code> instances from <strong>trusted sources</strong>. This is also why <code>CallableRubricTrait</code> cannot be created via the web API.</p>
</div>

## 10. Using Callable Traits in a Rubric

Callable traits combine with other trait types in a `Rubric`:

```python
from karenina.schemas import CallableRubricTrait, Rubric

rubric = Rubric(callable_traits=[
    minimum_length_trait,
    sentence_count_trait,
])

print(f"Rubric has {len(rubric.callable_traits)} callable traits")
for trait in rubric.callable_traits:
    print(f"  - {trait.name} (kind={trait.kind})")
```

## 11. Next Steps

- [LLM traits](../llm-traits/): boolean, score, and literal traits evaluated by an LLM judge
- [Regex traits](../regex-traits/): deterministic pattern matching
- [Metric traits](../metric-traits/): precision, recall, and F1 for extraction tasks
- [Templates vs rubrics](../../template-vs-rubric/): choosing between correctness checks and rubric-style evaluation
