---
jupyter:
  jupytext:
    formats: docs/core_concepts/questions-and-benchmarks//md,docs/notebooks/core_concepts/questions-and-benchmarks//ipynb
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

# Questions

A question is the basic building block of evaluation in Karenina. Each question carries the text sent to an LLM, a reference answer that anchors the evaluation, and intrinsic metadata (author, sources, timestamps, keywords). The `Question` model is self-contained: everything that describes the question itself lives on the object.

```python tags=["hide-cell"]
# Mock cell: ensures examples execute without live API keys.
# This cell is hidden in rendered documentation.
import hashlib
from unittest.mock import MagicMock, patch

mock_modules = {}
for mod in ["sqlalchemy", "sqlalchemy.orm", "sqlalchemy.ext", "sqlalchemy.ext.declarative"]:
    mock_modules[mod] = MagicMock()

with patch.dict("sys.modules", mock_modules):
    from karenina.benchmark import Benchmark
    from karenina.schemas.entities import BaseAnswer
    from karenina.schemas.entities.question import Question
    from pydantic import Field
```

```python
# Standard imports for working with questions
from karenina.schemas.entities.question import Question
from karenina.schemas.entities import BaseAnswer
from karenina.benchmark import Benchmark
```

## What Is a Question?

The `Question` model carries core evaluation data and intrinsic metadata:

**Core fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `question` | `str` | Yes | The question text sent to the LLM (minimum 1 character) |
| `raw_answer` | `str` | Yes | The human-readable reference answer (minimum 1 character) |
| `keywords` | `list[str]` | No | Keywords for filtering and organization (default: empty list) |
| `few_shot_examples` | `list[dict[str, str]] \| None` | No | Example question-answer pairs for parsing guidance |
| `id` | `str` (computed) | Auto | Deterministic MD5 hash of the question text |

**Intrinsic metadata** (carried on the Question object itself):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `date_created` | `str` | Now (ISO) | When the question was created |
| `date_modified` | `str` | Now (ISO) | When the question was last modified |
| `answer_template` | `str \| None` | `None` | Template code attached to this question |
| `author` | `dict \| None` | `None` | Author information (name, affiliation, etc.) |
| `sources` | `list[dict] \| None` | `None` | Source documents or references |
| `custom_metadata` | `dict \| None` | `None` | Arbitrary key-value pairs |
| `question_rubric` | `dict \| None` | `None` | Question-specific rubric traits |

```python
q = Question(
    question="What is the putative target of venetoclax?",
    raw_answer="BCL2 (B-cell lymphoma 2)",
    keywords=["pharmacology", "oncology"],
)

print(f"Question:   {q.question}")
print(f"Raw answer: {q.raw_answer}")
print(f"Keywords:   {q.keywords}")
print(f"ID:         {q.id}")
```

<div class="admonition note">
<p class="admonition-title">Backward compatibility: <code>tags</code> to <code>keywords</code></p>
<p>The field formerly named <code>tags</code> has been renamed to <code>keywords</code>. A model validator accepts the legacy <code>tags</code> key during construction and automatically converts it to <code>keywords</code>, so existing checkpoint files and code using <code>tags=</code> continue to work.</p>
</div>

## Deterministic IDs

Every question gets an ID computed as an MD5 hash of its text. The same question text always produces the same ID, which enables reliable cross-referencing between benchmarks, results, and traces.

```python
# The ID is deterministic: same text produces same ID
q1 = Question(question="What is the capital of France?", raw_answer="Paris")
q2 = Question(question="What is the capital of France?", raw_answer="Paris")
print(f"q1.id: {q1.id}")
print(f"q2.id: {q2.id}")
print(f"Same:  {q1.id == q2.id}")
```

```python
# Under the hood: MD5 of the question text
manual_id = hashlib.md5("What is the capital of France?".encode("utf-8")).hexdigest()
print(f"Manual hash: {manual_id}")
print(f"Matches:     {manual_id == q1.id}")
```

You can override the auto-generated ID by passing `question_id` when adding to a benchmark:

```python
benchmark = Benchmark.create(name="ID Demo", version="0.1.0")
custom_id = benchmark.add_question(
    question="What is the capital of France?",
    raw_answer="Paris",
    question_id="france-capital-001",
)
print(f"Custom ID: {custom_id}")
```

<div class="admonition warning">
<p class="admonition-title">Changing text changes the ID</p>
<p>If you modify a question's text (even whitespace or capitalization), the auto-generated ID changes. This breaks cross-references to results and traces tied to the old ID. If you need to edit question text while preserving the ID, use a custom <code>question_id</code>.</p>
</div>

## `raw_answer` vs Template `ground_truth`

These two concepts are related but serve different purposes:

| Concept | Where It Lives | Who Sees It | Purpose |
|---------|---------------|-------------|---------|
| `raw_answer` | `Question.raw_answer` | Benchmark authors; stored in results for reference | Human-readable reference answer written in plain language |
| `self.correct` | Set inside `ground_truth()` method | Only `verify()` | Programmatic answer key used by the `verify()` method to check correctness |

The relationship: `raw_answer` is the source of truth from which the template author derives `self.correct`. A well-written `raw_answer` describes the expected answer in plain language. The `ground_truth()` method translates that into structured values that `verify()` can compare programmatically.

`raw_answer` is **not** sent to the [Judge LLM](../verification-pipeline.md) during parsing. The Judge receives only the question text, the LLM's response, and the template's JSON schema. The `raw_answer` serves as the author's reference when writing the template's `ground_truth()` method, and it is stored in verification results for human review.

```python
# The raw_answer describes the expected answer in natural language
q = Question(
    question="What is the putative target of venetoclax?",
    raw_answer="BCL2 (B-cell lymphoma 2), an anti-apoptotic protein",
)

# The template's ground_truth() derives structured values from that
# (see Answer Templates for full details on BaseAnswer)
class Answer(BaseAnswer):
    target: str = Field(
        description=(
            "The direct protein target of venetoclax as stated in the response. "
            "Extract the protein name or gene symbol (e.g., 'BCL2', 'Bcl-2', "
            "'B-cell lymphoma 2'). If the response mentions multiple proteins, "
            "extract only the one identified as the direct pharmacological "
            "target, not downstream effectors or pathway members."
        )
    )

    def ground_truth(self):
        # Derived from raw_answer: "BCL2 (B-cell lymphoma 2)"
        self.correct = {"target": "BCL2"}

    def verify(self) -> bool:
        return self.target.strip().upper().replace("-", "") == self.correct["target"]
```

<div class="admonition tip">
<p class="admonition-title">Writing a good <code>raw_answer</code></p>
<p>Although <code>raw_answer</code> is not sent to the Judge LLM, it still matters. It is the reference you (the benchmark author) use when writing the template's <code>ground_truth()</code> method. A specific <code>raw_answer</code> like <code>"BCL2 (B-cell lymphoma 2)"</code> makes it clear what <code>self.correct</code> should contain, while a vague one like <code>"yes"</code> leaves the template author guessing. It also appears in verification results, so reviewers can see at a glance what the expected answer was.</p>
</div>

## Keywords

Keywords provide lightweight categorization for filtering and grouping questions within a benchmark. They are stored on the Question object and can be used with `search_questions()` or custom filters. For operational examples, see [Benchmark Operations](../../workflows/creating-benchmarks/benchmark-operations.md#filtering-and-search).

## Few-Shot Examples

Questions can carry example question-answer pairs that guide the [Judge LLM](../verification-pipeline.md) during parsing. These examples show the judge how similar questions should be parsed, improving accuracy for complex or ambiguous response formats. For configuration modes and best practices, see [Few-Shot](../few-shot.md).

## The `finished` Flag

The `finished` flag determines whether a question enters the verification pipeline. It is tracked separately from the question's intrinsic data because `finished` is a property of the question's membership in a benchmark, not an intrinsic property of the question itself. A question can be finished in one benchmark and unfinished in another.

The default depends on the interface:

- **Python API** (`add_question()`): defaults to `finished=True`, because questions added programmatically are assumed to be complete
- **GUI** (karenina-gui): always passes `finished=False`, prompting the user to review before verification

Only finished questions are included when the verification pipeline runs. If `run_verification()` returns an empty result set, the most common cause is that all questions are marked `finished=False`.

For managing finished status (marking, toggling, batch operations), see [Benchmark Operations](../../workflows/creating-benchmarks/benchmark-operations.md#managing-question-state).

## Next Steps

- [Benchmarks](benchmarks.md): the benchmark as a package, metadata, persistence
- [Answer Templates](../answer-templates.md): how to write templates, field types, `ground_truth()` and `verify()`
- [Evaluation Modes](../evaluation-modes.md): how `finished` status, templates, and rubrics determine what the pipeline runs
- [Benchmark Operations](../../workflows/creating-benchmarks/benchmark-operations.md): adding questions, managing finished status, accessing question data, writing good `raw_answer` values
- [Creating Benchmarks](../../workflows/creating-benchmarks/index.md): step-by-step authoring workflow
- [Checkpoints](../checkpoints.md): how benchmarks persist as JSON-LD files
- [Few-Shot](../few-shot.md): configuring example injection for parsing accuracy
