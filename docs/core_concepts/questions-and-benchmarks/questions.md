---
jupyter:
  jupytext:
    formats: docs/core_concepts/questions-and-benchmarks//md,docs/notebooks/core_concepts/questions-and-benchmarks//ipynb
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

# Questions

A question is the atomic unit of evaluation in Karenina. Each question carries the text sent to an LLM and a reference answer that anchors the evaluation. Questions are lightweight by design: the `Question` Pydantic model has only four fields. Rich metadata (author, sources, timestamps, the `finished` flag) lives on the benchmark's cache layer, not on the Question object itself.

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

## What Is a Question?

The `Question` model defines four fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `question` | `str` | Yes | The question text sent to the LLM (minimum 1 character) |
| `raw_answer` | `str` | Yes | The human-readable reference answer (minimum 1 character) |
| `tags` | `list[str \| None]` | No | Category tags for filtering and organization |
| `few_shot_examples` | `list[dict[str, str]] \| None` | No | Example question-answer pairs for parsing guidance |
| `id` | `str` (computed) | Auto | Deterministic MD5 hash of the question text |

```python
# Create a standalone Question object
q = Question(
    question="What is the putative target of venetoclax?",
    raw_answer="BCL2 (B-cell lymphoma 2)",
    tags=["pharmacology", "oncology"],
)

print(f"Question:   {q.question}")
print(f"Raw answer: {q.raw_answer}")
print(f"Tags:       {q.tags}")
print(f"ID:         {q.id}")
```

<div class="admonition info">
<p class="admonition-title">Lightweight by design</p>
<p>The <code>Question</code> model carries only what is intrinsic to the question itself. Metadata that only makes sense in a benchmark context (author, sources, timestamps, the <code>finished</code> flag, custom properties) lives on the benchmark's cache entry, not on the Question object. This means a <code>Question</code> can exist and be passed around independently of any benchmark. See <a href="#question-metadata-the-extended-profile">Question Metadata</a> below for how to access the full picture.</p>
</div>

## Deterministic IDs

Every question gets an ID computed as an MD5 hash of its text. The same question text always produces the same ID, which enables reliable cross-referencing between benchmarks, results, and traces.

```python
# The ID is deterministic: same text → same ID
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
| `raw_answer` | `Question.raw_answer` | The Judge LLM (as context during parsing) | Human-readable reference; helps the judge understand what a correct response looks like |
| `self.correct` | Set inside `ground_truth()` method | Only `verify()` | Programmatic answer key used by the `verify()` method to check correctness |

The relationship: `raw_answer` is the source of truth from which the template author derives `self.correct`. A well-written `raw_answer` describes the expected answer in plain language. The `ground_truth()` method translates that into structured values that `verify()` can compare programmatically.

```python
# The raw_answer describes the expected answer in natural language
q = Question(
    question="What is the putative target of venetoclax?",
    raw_answer="BCL2 (B-cell lymphoma 2), an anti-apoptotic protein",
)

# The template's ground_truth() derives structured values from that
class Answer(BaseAnswer):
    target: str = Field(
        description="The protein target of the drug mentioned in the response"
    )

    def ground_truth(self):
        # Derived from raw_answer: "BCL2 (B-cell lymphoma 2)"
        self.correct = {"target": "BCL2"}

    def verify(self) -> bool:
        return self.target.strip().upper().replace("-", "") == self.correct["target"]
```

### Writing Good `raw_answer` Values

The Judge LLM sees `raw_answer` as context during the parsing stage. A precise `raw_answer` helps the judge understand what a correct response looks like, which can improve parsing accuracy for ambiguous responses.

| Quality | Example `raw_answer` | Issue |
|---------|---------------------|-------|
| Poor | `"yes"` | Ambiguous; the judge gets no context about what "yes" means |
| Poor | `"BCL2"` | Bare abbreviation; acceptable but provides minimal context |
| Good | `"BCL2 (B-cell lymphoma 2)"` | Full name with abbreviation; clear and unambiguous |
| Good | `"BCL2 (B-cell lymphoma 2), an anti-apoptotic protein overexpressed in CLL"` | Rich context; particularly helpful for domain-specific questions |

<div class="admonition note">
<p class="admonition-title"><code>raw_answer</code> is visible to the Judge</p>
<p>The Judge LLM receives <code>raw_answer</code> as part of the parsing context. This means its phrasing can influence extraction. A vague <code>raw_answer</code> like <code>"yes"</code> gives the judge nothing to anchor on, while a specific one like <code>"BCL2 (B-cell lymphoma 2)"</code> helps it identify the relevant information in the response. The degree of influence depends on the evaluation model and prompt configuration.</p>
</div>

## Tags and Organization

Tags provide lightweight categorization for filtering and grouping:

```python
benchmark = Benchmark.create(name="Tags Demo", version="0.1.0")
q_id = benchmark.add_question(
    question="What is the putative target of venetoclax?",
    raw_answer="BCL2",
)

# Tags are stored on the Question object
q_obj = benchmark.get_question_as_object(q_id)
print(f"Tags: {q_obj.tags}")
```

Tags are primarily useful for organizing large benchmarks. You can filter questions by tags using `search_questions()` or custom filters.

## Few-Shot Examples

Few-shot examples provide parsing guidance to the Judge LLM. They are question-answer pairs that show the judge how similar questions should be parsed, improving accuracy for complex or ambiguous response formats.

```python
q_id = benchmark.add_question(
    question="What are the FDA-approved indications for venetoclax?",
    raw_answer="CLL and AML",
    few_shot_examples=[
        {
            "question": "What are the approved uses of imatinib?",
            "answer": "CML and GIST",
        },
    ],
)
q_data = benchmark.get_question(q_id)
print(f"Few-shot examples: {q_data.get('few_shot_examples')}")
```

Few-shot examples are injected into the parsing prompt at the `few_shot` stage. See [Few-Shot](../few-shot.md) for configuration modes and best practices.

## Question Metadata: The Extended Profile

As described in the [overview](index.md), questions have two layers. When a question is added to a benchmark, the benchmark creates a cache entry with extended metadata fields.

### Accessing the Full Profile

```python
# Create a question with full metadata
bm = Benchmark.create(name="Metadata Demo", version="0.1.0")
q_id = bm.add_question(
    question="What protein does imatinib inhibit?",
    raw_answer="BCR-ABL tyrosine kinase",
    author={"name": "Dr. Smith", "affiliation": "Oncology Dept"},
    sources=[{"title": "Drucker et al. 2001", "url": "https://example.com/paper"}],
    custom_metadata={"difficulty": "easy", "domain": "oncology"},
)

# get_question() returns the raw cache entry
q_data = bm.get_question(q_id)
print(f"Question text:  {q_data['question']}")
print(f"Finished:       {q_data.get('finished')}")

# get_question_metadata() returns a clean summary
metadata = bm.get_question_metadata(q_id)
print(f"\nFull metadata:")
for key, value in metadata.items():
    print(f"  {key}: {value}")
```

### Cache Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Question ID (MD5 hash or custom) |
| `question` | `str` | Question text |
| `raw_answer` | `str` | Reference answer |
| `answer_template` | `str` | Template code (always present; may be a default) |
| `finished` | `bool` | Whether the question is ready for verification |
| `author` | `dict \| None` | Author information (name, affiliation, etc.) |
| `sources` | `list[dict] \| None` | Source documents or references |
| `custom_metadata` | `dict` | Arbitrary key-value pairs |
| `date_created` | `str` | ISO timestamp of when the question was added |
| `date_modified` | `str` | ISO timestamp of last modification |
| `question_rubric` | `dict \| None` | Question-specific rubric traits |
| `keywords` | `list[str] \| None` | Tags/keywords from the Question object |

### Custom Properties

You can attach arbitrary metadata to individual questions:

```python
# Set and get custom properties
bm.set_question_custom_property(q_id, "reviewed_by", "Dr. Jones")
bm.set_question_custom_property(q_id, "confidence", "high")

reviewer = bm.get_question_custom_property(q_id, "reviewed_by")
print(f"Reviewed by: {reviewer}")

# Custom metadata is also accessible via get_question_metadata()
meta = bm.get_question_metadata(q_id)
print(f"Custom metadata: {meta['custom_metadata']}")
```

<div class="admonition note">
<p class="admonition-title">Use <code>get_question_metadata()</code> for the full picture</p>
<p><code>get_question()</code> returns the raw internal cache entry (a dict). <code>get_question_metadata()</code> returns a clean summary that includes computed fields like <code>has_template</code> and <code>has_rubric</code>. For inspecting a question's status, prefer the metadata method.</p>
</div>

## The `finished` Flag and the Pipeline

The `finished` flag is a boolean on each question's cache entry that gates pipeline entry. Only finished questions are included when the verification pipeline calls `get_finished_templates()`.

### Default Behavior

The default depends on the interface:

- **Python API** (`add_question()`): defaults to `finished=True`, because questions added programmatically are assumed to be complete
- **GUI** (karenina-gui): always passes `finished=False`, prompting the user to review before verification

```python
bm = Benchmark.create(name="Finished Demo", version="0.1.0")

# Python API: finished=True by default
q1 = bm.add_question(
    question="What is the target of imatinib?",
    raw_answer="BCR-ABL",
)

# Explicitly set to unfinished
q2 = bm.add_question(
    question="What is the target of trastuzumab?",
    raw_answer="HER2",
    finished=False,
)

print(f"q1 finished: {bm.get_question(q1).get('finished')}")
print(f"q2 finished: {bm.get_question(q2).get('finished')}")
```

### Managing Finished Status

```python
# Mark a question as finished or unfinished
bm.mark_finished(q2)
print(f"q2 after mark_finished: {bm.get_question(q2).get('finished')}")

bm.mark_unfinished(q2)
print(f"q2 after mark_unfinished: {bm.get_question(q2).get('finished')}")

# Toggle
new_status = bm.toggle_finished(q2)
print(f"q2 after toggle: {new_status}")
```

### Impact on Verification

```python
# get_finished_templates() only returns finished questions
finished_templates = bm.get_finished_templates()
print(f"Finished templates: {len(finished_templates)}")

# List unfinished questions
unfinished = bm.get_unfinished_questions(ids_only=True)
print(f"Unfinished question IDs: {unfinished}")
```

<div class="admonition tip">
<p class="admonition-title">Zero results? Check <code>finished</code> first</p>
<p>If <code>run_verification()</code> returns an empty result set, the most common cause is that all questions are marked <code>finished=False</code>. This happens frequently when loading benchmarks created in the GUI, where questions start unfinished by default. Use <code>benchmark.mark_finished_batch(benchmark.get_question_ids())</code> to mark all questions as ready.</p>
</div>

## Three Ways to Add Questions

### 1. String Arguments

The most common approach: pass question text and raw answer directly.

```python
bm = Benchmark.create(name="Add Methods Demo", version="0.1.0")

q_id = bm.add_question(
    question="What is the target of venetoclax?",
    raw_answer="BCL2",
    author={"name": "Dr. Smith"},
)
print(f"Added via strings: {q_id[:16]}...")
```

### 2. Question Object

Pass a `Question` instance. The benchmark uses its auto-generated ID and copies its fields:

```python
q_obj = Question(
    question="What is the mechanism of action of venetoclax?",
    raw_answer="BH3 mimetic",
    tags=["pharmacology"],
)

q_id = bm.add_question(q_obj)
print(f"Added via Question object: {q_id[:16]}...")
```

### 3. Inline with Template

Pass both the question and template in one call:

```python
class DrugClassAnswer(BaseAnswer):
    drug_class: str = Field(description="The pharmacological class of the drug")
    def ground_truth(self):
        self.correct = {"drug_class": "BH3 mimetic"}
    def verify(self) -> bool:
        return "bh3" in self.drug_class.lower()

q_id = bm.add_question(
    question="What class of drug is venetoclax?",
    raw_answer="BH3 mimetic",
    answer_template=DrugClassAnswer,
)
print(f"Added with inline template: {q_id[:16]}...")
print(f"Has template: {bm.has_template(q_id)}")
```

## Common Pitfalls

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Ambiguous `raw_answer` leads to poor parsing | The Judge LLM lacks context to extract the right value | Use descriptive `raw_answer` values: `"BCL2 (B-cell lymphoma 2)"` not just `"BCL2"` |
| Forgot to mark finished; verification returns nothing | Questions default to `finished=True` in Python but `finished=False` in GUI exports | Check `get_unfinished_questions()` and use `mark_finished_batch()` |
| Same question text produces different IDs in different benchmarks | You are comparing custom IDs vs auto-generated IDs, or text differs in whitespace | Use consistent text or explicit `question_id` values |
| Modified question text breaks result cross-references | ID changed because it is computed from text | Use a custom `question_id` when text may evolve |
| `get_template()` raises `ValueError` | Question only has a default template | Check `has_template()` first; add a real template with `add_answer_template()` |

## Next Steps

- [Benchmarks deep dive](../../notebooks/core_concepts/questions-and-benchmarks/benchmarks.ipynb): the facade, readiness checking, default templates, filtering
- [Answer Templates](../../notebooks/core_concepts/answer-templates.ipynb): how to write templates, field types, `ground_truth()` and `verify()`
- [Checkpoints](../checkpoints.md): how benchmarks persist as JSON-LD files
- [Few-Shot](../few-shot.md): configuring example injection for parsing accuracy
- [Creating Benchmarks](../../workflows/creating-benchmarks/index.md): step-by-step authoring workflow
