---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Loading a Benchmark

Before running verification, you need to load your benchmark from a checkpoint file (JSON-LD) or a database. This page covers `Benchmark.load()`, inspecting questions, templates, and rubrics, and preparing for verification.

For background on the checkpoint format, see [Checkpoints](../core_concepts/checkpoints.md).

```python tags=["hide-cell"]
# Setup cell (hidden in rendered docs).
# No mocking needed — all examples load from the test checkpoint file.
import os as _os

_os.chdir(_os.path.dirname(_os.path.abspath("__file__")))
```

---

## Loading from a File

Use `Benchmark.load()` to load a checkpoint from a JSON-LD file:

```python
from pathlib import Path
from karenina.benchmark import Benchmark

benchmark = Benchmark.load(Path("test_checkpoint.jsonld"))

print(f"Name:        {benchmark.name}")
print(f"Description: {benchmark.description}")
print(f"Version:     {benchmark.version}")
print(f"Creator:     {benchmark.creator}")
print(f"Questions:   {benchmark.question_count}")
```

`Benchmark.load()` parses the JSON-LD structure, validates it, and rebuilds the internal question/template/rubric caches. The returned `Benchmark` object is ready for inspection and verification.

---

## Loading from a Database

If your benchmark is stored in a database (via `save_to_db()`), load it by name:

```python
# benchmark = Benchmark.load_from_db("My Benchmark", storage="sqlite:///benchmarks.db")
```

The `storage` parameter is a SQLAlchemy connection string. This is useful when benchmarks are managed through the karenina-server or GUI.

---

## Inspecting Questions

### Listing Questions

```python
# Get all question IDs
question_ids = benchmark.get_question_ids()
print(f"Question IDs ({len(question_ids)}):")
for qid in question_ids:
    print(f"  {qid}")
```

### Getting Question Details

Each question is returned as a dictionary with all its fields:

```python
# Get the first question
qid = benchmark.get_question_ids()[0]
question = benchmark.get_question(qid)

print(f"Question:   {question['question']}")
print(f"Answer:     {question['raw_answer']}")
print(f"Finished:   {question.get('finished', False)}")
print(f"Has rubric: {'question_rubric' in question and question['question_rubric'] is not None}")
```

### Getting All Questions

Use `get_all_questions()` for bulk access:

```python
# Full question dictionaries
all_questions = benchmark.get_all_questions()
print(f"Total questions: {len(all_questions)}")
for q in all_questions:
    print(f"  - {q['question'][:60]}...")

# IDs only (lighter)
ids = benchmark.get_all_questions(ids_only=True)
print(f"\nIDs only: {len(ids)} questions")
```

### Question Metadata

For a structured metadata summary including template and rubric status:

```python
qid = benchmark.get_question_ids()[0]
meta = benchmark.get_question_metadata(qid)

print(f"Question:     {meta['question'][:50]}...")
print(f"Has template: {meta['has_template']}")
print(f"Has rubric:   {meta['has_rubric']}")
print(f"Finished:     {meta['finished']}")
print(f"Created:      {meta['date_created']}")
```

---

## Inspecting Templates

Templates define how answers are parsed and verified. Not every question needs a template — questions without templates can still be evaluated using rubric-only mode.

```python
# Check which questions have templates
for qid in benchmark.get_question_ids():
    has_it = benchmark.has_template(qid)
    q = benchmark.get_question(qid)
    print(f"{'[template]' if has_it else '[none]    '} {q['question'][:50]}...")
```

### Viewing Template Code

```python
# Get the template code for a question that has one
qid = benchmark.get_question_ids()[0]
if benchmark.has_template(qid):
    code = benchmark.get_template(qid)
    print(code)
```

### Finished Templates

The `get_finished_templates()` method returns templates that are ready for verification (questions marked as finished with non-default templates):

```python
finished = benchmark.get_finished_templates()
print(f"Templates ready for verification: {len(finished)}")
for ft in finished:
    print(f"  - {ft.question_preview}: {len(ft.template_code)} chars")
```

### Missing Templates

Find questions that still need templates:

```python
missing = benchmark.get_missing_templates()
print(f"Questions without templates: {len(missing)}")
for m in missing:
    print(f"  - {m['question'][:60]}...")
```

---

## Inspecting Rubrics

Rubrics evaluate response quality through traits. They can be **global** (applied to all questions) or **question-specific**.

### Global Rubric

```python
global_rubric = benchmark.get_global_rubric()
if global_rubric:
    print(f"Global LLM traits:      {len(global_rubric.llm_traits)}")
    print(f"Global regex traits:    {len(global_rubric.regex_traits)}")
    print(f"Global callable traits: {len(global_rubric.callable_traits)}")
    print(f"Global metric traits:   {len(global_rubric.metric_traits)}")
else:
    print("No global rubric defined")
```

### Question-Specific Rubrics

```python
for qid in benchmark.get_question_ids():
    q = benchmark.get_question(qid)
    rubric = q.get("question_rubric")
    if rubric:
        trait_count = sum(
            len(rubric.get(k, []))
            for k in ["llm_traits", "regex_traits", "callable_traits", "metric_traits"]
        )
        print(f"{q['question'][:40]}... — {trait_count} trait(s)")
    else:
        print(f"{q['question'][:40]}... — no question rubric")
```

---

## Benchmark Status

Use the status properties to understand the overall state of a loaded benchmark:

```python
print(f"Name:          {benchmark.name}")
print(f"Questions:     {benchmark.question_count}")
print(f"Finished:      {benchmark.finished_count}")
print(f"Empty:         {benchmark.is_empty}")
print(f"Complete:      {benchmark.is_complete}")
print(f"Progress:      {benchmark.get_progress():.1f}%")
```

A benchmark is **complete** when all questions are finished and have templates. The `get_progress()` method returns a percentage based on the ratio of finished questions to total questions.

---

## Next Steps

- [Configure verification settings](verification-config.md) — set up models, evaluation mode, and feature flags
- [Run verification via Python API](python-api.md) — full end-to-end example
- [Run verification via CLI](cli.md) — command-line workflow
- [Analyze results](../07-analyzing-results/index.md) — inspect and export verification outputs
