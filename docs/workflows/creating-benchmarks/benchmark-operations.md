---
jupyter:
  jupytext:
    formats: docs/workflows/creating-benchmarks//md,docs/notebooks/creating-benchmarks//ipynb
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

# Benchmark Operations

This page covers the full operational API for working with `Benchmark` objects: creating, populating, inspecting, querying, and checking readiness. Think of it as a reference for everything you can do with a benchmark before running verification.

For the conceptual overview of what a benchmark *is*, see [Benchmarks (Core Concepts)](../../core_concepts/questions-and-benchmarks/benchmarks.md). For end-to-end authoring scenarios, see the other pages in this section ([Factual QA](factual-qa-benchmark.md), [Full Evaluation](full-evaluation-benchmark.md), [Quality Assessment](quality-assessment-benchmark.md), [Scaled Authoring](scaled-authoring.md)).

```python tags=["hide-cell"]
# Setup cell: hidden in rendered documentation.
import hashlib
from datetime import datetime

from pydantic import Field

from karenina.benchmark import Benchmark
from karenina.schemas.entities import BaseAnswer, Rubric
from karenina.schemas.entities.question import Question
from karenina.schemas.entities.rubric import LLMRubricTrait, RegexRubricTrait
```

---

## Creating a Benchmark

Every benchmark starts with `Benchmark.create()` (or equivalently, the `Benchmark()` constructor):

```python
benchmark = Benchmark.create(
    name="Drug Target Identification",
    description="Evaluate LLM accuracy on identifying drug targets from literature",
    version="1.0.0",
    creator="Pharmacology Team",
)

print(f"Name:        {benchmark.name}")
print(f"Version:     {benchmark.version}")
print(f"Description: {benchmark.description}")
print(f"Creator:     {benchmark.creator}")
print(f"Questions:   {benchmark.question_count}")
```

`Benchmark.create()` and `Benchmark()` are interchangeable. Both accept the same parameters and produce identical results.

To load an existing benchmark from a checkpoint file:

```python
# benchmark = Benchmark.load(Path("my_benchmark.jsonld"))
```

---

## Adding Questions

Add questions one at a time. Each question requires the text sent to the LLM and a reference answer (the ground truth). There are three ways to add a question.

### String arguments

The most common approach: pass question text and raw answer directly.

```python
q1_id = benchmark.add_question(
    question="What is the putative target of venetoclax?",
    raw_answer="BCL2 (B-cell lymphoma 2)",
)

q2_id = benchmark.add_question(
    question="What is the mechanism of action of venetoclax?",
    raw_answer="BH3 mimetic that selectively inhibits BCL2",
)

print(f"Added {benchmark.question_count} questions")
print(f"Q1 ID: {q1_id}")
print(f"Q2 ID: {q2_id}")
```

### Question object

Pass a `Question` instance. The benchmark uses its auto-generated ID and copies its fields:

```python
q_obj = Question(
    question="What are the FDA-approved indications for venetoclax?",
    raw_answer="CLL and AML",
    keywords=["pharmacology"],
    author={"name": "Dr. Smith", "affiliation": "Oncology Dept"},
)

q_extra_id = benchmark.add_question(q_obj)
print(f"Added via Question object: {q_extra_id}")
```

### Inline with template

Pass both the question and an answer template in one call. See [Inline (at question creation time)](#inline-at-question-creation-time) below for the full example.

Question IDs are deterministic: derived from the question text via hashing. The same text always produces the same ID. For details on ID generation and custom IDs, see [Questions (Core Concepts)](../../core_concepts/questions-and-benchmarks/questions.md#deterministic-ids).

---

## Writing Good `raw_answer` Values

The Judge LLM sees `raw_answer` as context during the parsing stage. A precise `raw_answer` helps the judge understand what a correct response looks like, which improves parsing accuracy for ambiguous responses.

| Quality | Example `raw_answer` | Issue |
|---------|---------------------|-------|
| Poor | `"yes"` | Ambiguous; the judge gets no context about what "yes" means |
| Poor | `"BCL2"` | Bare abbreviation; acceptable but provides minimal context |
| Good | `"BCL2 (B-cell lymphoma 2)"` | Full name with abbreviation; clear and unambiguous |
| Good | `"BCL2 (B-cell lymphoma 2), an anti-apoptotic protein overexpressed in CLL"` | Rich context; particularly helpful for domain-specific questions |

For more on the relationship between `raw_answer` and the template's `ground_truth()`, see [Questions (Core Concepts)](../../core_concepts/questions-and-benchmarks/questions.md#raw_answer-vs-template-ground_truth).

---

## Attaching Answer Templates

Templates define how to verify correctness. There are three ways to attach a template to a question.

### Inline (at question creation time)

Define the template as a class and pass it when adding the question. `add_question` accepts both class objects and source strings:

```python
class Answer(BaseAnswer):
    identifies_bh3_mimetic: bool = Field(
        description=(
            "True if the response identifies venetoclax as a BH3 mimetic "
            "(including 'BH3-mimetic', 'BH3 mimetic', or 'Bcl-2 homology 3 "
            "mimetic'). False if the response describes a different mechanism "
            "class (e.g., 'kinase inhibitor') or mentions BH3 only as a "
            "protein domain without connecting it to the drug's mechanism."
        )
    )

    def ground_truth(self):
        self.correct = {"identifies_bh3_mimetic": True}

    def verify(self) -> bool:
        return self.identifies_bh3_mimetic == self.correct["identifies_bh3_mimetic"]

q3_id = benchmark.add_question(
    question="Describe the pharmacological mechanism of venetoclax.",
    raw_answer="BH3 mimetic that selectively inhibits BCL2",
)
benchmark.update_template(q3_id, Answer)
print(f"Has template: {benchmark.has_template(q3_id)}")
```

<div class="admonition note">
<p class="admonition-title">Template classes must be named <code>Answer</code></p>
<p>The pipeline looks for this exact class name. If you pass a class with a different name (e.g., <code>MyTemplate</code>), the benchmark auto-renames it to <code>Answer</code> internally, but naming it <code>Answer</code> from the start is the canonical pattern.</p>
</div>

### After the fact

Add or replace a template on an existing question using `update_template`:

```python
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
        self.correct = {"target": "BCL2"}

    def verify(self) -> bool:
        extracted = self.target.strip().upper().replace("-", "").replace(" ", "")
        return extracted == self.correct["target"]

benchmark.update_template(q1_id, Answer)
print(f"Q1 has template: {benchmark.has_template(q1_id)}")
```

### Global template

Apply a single template to all questions that do not yet have one:

```python
# Returns a list of question IDs that received the template
# applied_ids = benchmark.apply_global_template(template_code)
```

`apply_global_template()` only applies to questions without an existing template. Questions that already have one (even a default placeholder) are skipped.

### Default Templates

When you add a question without specifying a template, the benchmark generates a **minimal default placeholder**. This placeholder contains a single `response` field and a `verify()` that always returns `False`.

The default template exists so the internal data structure stays consistent (every question always has a template string), but `has_template()` returns `False` for defaults. This distinction matters for readiness checking.

```python
# q2 was added without a template
print(f"Q2 has template: {benchmark.has_template(q2_id)}")
```

<div class="admonition warning">
<p class="admonition-title">Default templates always fail verification</p>
<p>The auto-generated default template's <code>verify()</code> returns <code>False</code>. If your benchmark reports 0% accuracy, check whether all questions have real templates using <code>benchmark.get_missing_templates()</code>.</p>
</div>

---

## Attaching Rubric Traits

Rubric traits assess response quality. They attach at two levels.

### Global rubric

Global traits apply to every question in the benchmark:

```python
global_rubric = Rubric(llm_traits=[
    LLMRubricTrait(
        name="conciseness",
        description=(
            "True if the response directly answers the question without "
            "unnecessary preamble, filler phrases, or tangential information. "
            "A concise response may still include relevant context or caveats. "
            "False if the response contains significant padding, repeats the "
            "question back, or includes large blocks of unrequested detail."
        ),
        kind="boolean",
        higher_is_better=True,
    ),
])
benchmark.set_global_rubric(global_rubric)
print(f"Global rubric set: {benchmark.get_global_rubric() is not None}")
```

### Question-specific traits

Question-specific traits apply to a single question. When both global and question-specific traits exist, they merge. If a question-specific trait has the same name as a global trait, the question-specific one takes precedence.

```python
benchmark.add_question_rubric_trait(
    q1_id,
    RegexRubricTrait(
        name="mentions_bcl2_family",
        pattern=r"(?i)bcl[-\s]?2\s+family",
        description="Response mentions the BCL2 protein family.",
        higher_is_better=True,
    ),
)
print(f"Questions with rubric: {[q['id'] for q in benchmark.get_questions_with_rubric()]}")
```

For the full set of trait types (LLM, regex, callable, metric, literal), see [Rubrics](../../core_concepts/rubrics/index.md).

---

## Custom Properties

Benchmarks support arbitrary key-value metadata for domain-specific attributes:

```python
benchmark.set_custom_property("domain", "pharmacology")
benchmark.set_custom_property("target_audience", "drug discovery teams")

print(f"Domain: {benchmark.get_custom_property('domain')}")
print(f"All properties: {benchmark.get_all_custom_properties()}")
```

Custom properties describe the benchmark as a whole. For per-question metadata, see [Managing Question State](#managing-question-state) below.

---

## Managing Question State

### Finished status

The `finished` flag determines whether a question enters the verification pipeline. Only finished questions are processed by `get_finished_templates()`. See [Questions (Core Concepts)](../../core_concepts/questions-and-benchmarks/questions.md#the-finished-flag) for the conceptual explanation.

```python
# Mark individual questions
benchmark.mark_finished(q2_id)
benchmark.mark_unfinished(q2_id)

new_status = benchmark.toggle_finished(q2_id)
print(f"q2 after toggle: {new_status}")

# Batch operations
benchmark.mark_finished_batch(benchmark.get_question_ids())

# Check status
finished_templates = benchmark.get_finished_templates()
unfinished = benchmark.get_unfinished_questions(ids_only=True)
print(f"Finished: {len(finished_templates)}, Unfinished: {len(unfinished)}")
```

### Accessing question data

The benchmark provides several methods for inspecting questions:

```python
# Raw cache entry (intrinsic data only, no finished status)
q_data = benchmark.get_question(q1_id)
print(f"Question text: {q_data['question']}")
print(f"Author:        {q_data.get('author')}")

# Clean summary combining intrinsic data + registry state
metadata = benchmark.get_question_metadata(q1_id)
print(f"Finished:     {metadata['finished']}")
print(f"Has template: {metadata['has_template']}")

# Full Question object with all fields
q_obj = benchmark.get_question_as_object(q1_id)
print(f"Keywords: {q_obj.keywords}")
```

<div class="admonition note">
<p class="admonition-title">Use <code>get_question_metadata()</code> for the full picture</p>
<p><code>get_question()</code> returns the raw internal cache entry (intrinsic data only, no <code>finished</code> status). <code>get_question_metadata()</code> returns a clean summary that merges both layers, including <code>finished</code> from the registry and computed fields like <code>has_template</code> and <code>has_rubric</code>. For inspecting a question's status, prefer the metadata method.</p>
</div>

### Per-question custom properties

Attach arbitrary metadata to individual questions:

```python
benchmark.set_question_custom_property(q1_id, "reviewed_by", "Dr. Jones")
benchmark.set_question_custom_property(q1_id, "confidence", "high")

reviewer = benchmark.get_question_custom_property(q1_id, "reviewed_by")
print(f"Reviewed by: {reviewer}")
```

---

## Readiness Checking (GUI Workflows)

<div class="admonition note">
<p class="admonition-title">Primarily relevant for GUI-authored benchmarks</p>
<p>When you build benchmarks through the Python API, readiness is rarely a concern: <code>add_question()</code> defaults <code>finished=True</code>, and you typically provide templates inline or immediately after. Readiness checking becomes important when benchmarks are authored through the GUI (karenina-gui), where questions start as <code>finished=False</code> and templates are added iteratively through the interface.</p>
</div>

The `check_readiness()` method validates that all pieces are in place before verification:

```python
readiness = benchmark.check_readiness()
print(f"Ready for verification: {readiness['ready_for_verification']}")
print(f"Has questions:          {readiness['has_questions']}")
print(f"All have templates:     {readiness['all_have_templates']}")
print(f"All finished:           {readiness['all_finished']}")
print(f"Templates valid:        {readiness['templates_valid']}")
print(f"Rubrics valid:          {readiness['rubrics_valid']}")
if readiness['missing_templates']:
    print(f"Missing templates:      {readiness['missing_templates_count']} questions")
if readiness['unfinished_questions']:
    print(f"Unfinished:             {int(readiness['unfinished_count'])} questions")
```

| Check | What It Verifies |
|-------|------------------|
| `has_questions` | At least one question exists |
| `all_have_templates` | Every question has a real template (not a default placeholder) |
| `all_finished` | Every question is marked as finished |
| `templates_valid` | All template code is syntactically valid Python |
| `rubrics_valid` | All rubric traits have required fields |

The overall `ready_for_verification` flag is `True` only when all checks pass. In GUI workflows, where questions and templates are built incrementally, this is the signal that the benchmark is complete and can be sent to the verification pipeline.

### Health Report

For a higher-level view, `get_health_report()` returns a scored assessment (0 to 100) with actionable recommendations. This is what the GUI displays to help users understand what still needs work:

```python
report = benchmark.get_health_report()
print(f"Health score:  {report['health_score']}")
print(f"Health status: {report['health_status']}")
if report.get('recommendations'):
    print("Recommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")
```

| Component | Points | Criteria |
|-----------|--------|----------|
| Questions exist | 20 | At least one question added |
| Completion progress | 30 | Based on `get_progress()` percentage |
| Templates valid | 25 | All templates parse as valid Python |
| Rubrics valid | 15 | All rubric traits properly configured |
| All finished | 10 | Every question marked as finished |

Status levels: **excellent** (90+), **good** (75+), **fair** (50+), **poor** (25+), **critical** (below 25).

---

## Filtering and Search

Benchmarks provide several ways to query their questions.

### Filter by attributes

```python
finished_qs = benchmark.filter_questions(finished=True)
templated_qs = benchmark.filter_questions(has_template=True)
ready_qs = benchmark.filter_questions(finished=True, has_template=True)

print(f"Finished questions:  {len(finished_qs)}")
print(f"With templates:      {len(templated_qs)}")
print(f"Ready for pipeline:  {len(ready_qs)}")
```

You can also filter by custom metadata or with a lambda:

```python
# Filter by custom metadata key-value pairs
# benchmark.filter_by_custom_metadata(domain="pharmacology", difficulty="hard")

# Filter with a lambda
# benchmark.filter_questions(
#     custom_filter=lambda q: q.get("custom_metadata", {}).get("priority") == "high"
# )
```

### Search by text

```python
results = benchmark.search_questions("venetoclax")
print(f"Questions mentioning 'venetoclax': {len(results)}")
```

### Aggregate

```python
counts = benchmark.count_by_field("finished")
print(f"By finished status: {counts}")
```

---

## Benchmark as a Collection

`Benchmark` implements Python's collection protocols, so you can use it like a container:

```python
# Length: number of questions
print(f"len(benchmark) = {len(benchmark)}")

# Containment: check if a question ID exists
print(f"q1_id in benchmark: {q1_id in benchmark}")

# Iteration: loop over questions
for question_data in benchmark:
    print(f"  - {question_data['question'][:60]}...")

# Indexing: access by position or ID
first_question = benchmark[0]
print(f"First question type: {type(first_question).__name__}")
```

Slicing is also supported: `benchmark[0:2]` returns a list of `SchemaOrgQuestion` objects.

---

## Common Pitfalls

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| 0% accuracy across all questions | Default templates still in place; `verify()` always returns `False` | Check `benchmark.get_missing_templates()` and add real templates |
| `is_complete` returns `False` | Some questions lack real templates or are not marked finished | Check `benchmark.get_unfinished_questions()` and `benchmark.get_missing_templates()` |
| `check_readiness()` says ready but verification returns no results | Questions have templates but are not marked `finished` | Call `benchmark.mark_finished(question_id)` for each question |
| `validate_templates()` reports errors | Template code has syntax errors or missing imports | Review template code; ensure it inherits from `BaseAnswer` |
| Global template does not appear on a question | `apply_global_template()` only applies to questions without templates | Questions that already have a template (even a default) may need explicit `add_answer_template()` |
| Rubric trait name collision raises `ValueError` | A question-specific trait has the same name as a global trait | Question-specific traits override globals with the same name; ensure names are intentionally unique or use override semantics |
| Ambiguous `raw_answer` leads to poor parsing | The Judge LLM lacks context to extract the right value | Use descriptive `raw_answer` values: `"BCL2 (B-cell lymphoma 2)"` not just `"BCL2"` |
| Same question text produces different IDs across benchmarks | Comparing custom IDs vs auto-generated IDs, or text differs in whitespace | Use consistent text or explicit `question_id` values |
| Modified question text breaks result cross-references | ID changed because it is computed from text | Use a custom `question_id` when text may evolve |
| `get_template()` raises `ValueError` | Question only has a default template | Check `has_template()` first; add a real template with `add_answer_template()` |

---

## Next Steps

- [Factual QA Benchmark](factual-qa-benchmark.md): hand-written templates for factual verification
- [Full Evaluation Benchmark](full-evaluation-benchmark.md): templates combined with rubric traits
- [Quality Assessment](quality-assessment-benchmark.md): rubric-only evaluation for subjective tasks
- [Scaled Authoring](scaled-authoring.md): bulk ingestion and automatic template generation
- [Running Verification](../running-verification/index.md): execute the benchmark against LLMs
