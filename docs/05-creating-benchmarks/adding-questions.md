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

# Adding Questions

Questions are the core content of any benchmark. Each question pairs a prompt with an expected answer, and optionally includes a template, metadata, and few-shot examples. This page covers the three ways to add questions and how to work with question metadata.

For background on how questions are stored in checkpoint files, see [Checkpoints](../core_concepts/checkpoints.md).

```python tags=["hide-cell"]
# Setup cell (hidden in rendered docs).
# No mocking needed — all examples create Benchmark objects locally.
```

---

## Adding a Question with Strings

The simplest way to add a question is by passing the question text and expected answer as strings:

```python
from karenina import Benchmark

benchmark = Benchmark(name="Science Quiz")

q_id = benchmark.add_question(
    question="What is the chemical symbol for gold?",
    raw_answer="Au",
)

print(f"Question ID: {q_id}")
print(f"Question count: {benchmark.question_count}")
```

The `add_question()` method returns a **question ID** — a deterministic URN based on the question text (e.g., `urn:uuid:question-what-is-the-chemical-...-a1b2c3d4`). This ID can be used to reference the question later.

## Adding a Question with a Template

You can attach an answer template at the same time. Templates can be passed as **Python code strings** or as **Answer classes** (in scripts where `inspect.getsource()` works).

### As a Code String

The most portable approach — works in notebooks and scripts alike:

```python
template_code = '''class Answer(BaseAnswer):
    symbol: str = Field(description="The chemical symbol for gold")

    def verify(self) -> bool:
        return self.symbol.strip().upper() == "AU"
'''

q_id = benchmark.add_question(
    question="What element has the symbol Au?",
    raw_answer="Gold",
    answer_template=template_code,
)

print(f"Question ID: {q_id}")
print(f"Finished count: {benchmark.finished_count}")
```

When you pass an `answer_template`, the question is automatically marked as **finished** (`finished=True`).

### As an Answer Class

In Python scripts (not notebooks), you can pass a class directly. The class name is automatically normalized to `"Answer"` internally:

    from karenina.schemas.entities import BaseAnswer
    from pydantic import Field

    class GoldAnswer(BaseAnswer):
        symbol: str = Field(description="The chemical symbol for gold")

        def verify(self) -> bool:
            return self.symbol.strip().upper() == "AU"

    q_id = benchmark.add_question(
        question="What element has the symbol Au?",
        raw_answer="Gold",
        answer_template=GoldAnswer,
    )

!!! note
    Passing a class requires that `inspect.getsource()` can access the class source code. This works in `.py` files but not in interactive environments (notebooks, REPL). In those contexts, use a code string instead.

For more on writing templates, see [Writing Templates](writing-templates.md) and [Answer Templates](../core_concepts/answer-templates.md).

## Adding a Question with a Question Object

For programmatic workflows, create a `Question` object first and pass it directly:

```python
from karenina.schemas.entities import Question

q_obj = Question(
    question="How many chromosomes are in a human somatic cell?",
    raw_answer="46",
    tags=["genetics", "cell-biology"],
    few_shot_examples=[
        {"question": "How many chromosomes does a fruit fly have?", "answer": "8"},
    ],
)

q_id = benchmark.add_question(q_obj)

print(f"Question ID: {q_id}")
print(f"Total questions: {benchmark.question_count}")
```

When you pass a `Question` object, its `tags` and `few_shot_examples` are automatically extracted and stored in the benchmark.

### Question Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `question` | `str` | Yes | The question text |
| `raw_answer` | `str` | Yes | The expected answer |
| `tags` | `list[str]` | No | Tags for categorization and filtering |
| `few_shot_examples` | `list[dict[str, str]]` | No | Few-shot examples as `{"question": ..., "answer": ...}` pairs |
| `id` | `str` | Auto | MD5 hash of the question text (computed, read-only) |

## Optional Metadata

All three usage patterns support additional metadata parameters:

```python
q_id = benchmark.add_question(
    question="What is the mechanism of action of imatinib?",
    raw_answer="Tyrosine kinase inhibitor targeting BCR-ABL",
    author={"name": "Dr. Smith", "email": "smith@example.com"},
    sources=[
        {"name": "DrugBank", "url": "https://go.drugbank.com/drugs/DB00619"},
    ],
    custom_metadata={"difficulty": "hard", "domain": "oncology"},
)

print(f"Added question with metadata: {q_id}")
```

### Metadata Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `author` | `dict[str, Any]` | Author information (name, email, affiliation, etc.) |
| `sources` | `list[dict[str, Any]]` | Source documents or references |
| `custom_metadata` | `dict[str, Any]` | Arbitrary key-value pairs stored in the checkpoint |
| `question_id` | `str` | Override the auto-generated ID (use with care) |
| `finished` | `bool` | Override the auto-set finished flag |
| `few_shot_examples` | `list[dict[str, str]]` | Few-shot examples (also available via Question object) |

## Extracting Questions from Files

For larger benchmarks, you can extract questions from Excel, CSV, or TSV files using the `extract_questions_from_file` utility:

    from karenina.benchmark.authoring.questions import extract_questions_from_file

    # Extract from a spreadsheet
    questions = extract_questions_from_file(
        file_path="questions.xlsx",
        question_column="Question",
        answer_column="Answer",
        author_name_column="Author",           # optional
        keywords_columns=[                     # optional
            {"column": "Keywords", "separator": ","},
        ],
    )

    # Add extracted questions to benchmark
    for question, metadata in questions:
        benchmark.add_question(question, **metadata)

The function returns a list of `(Question, dict)` tuples. Each tuple contains the `Question` object and a metadata dictionary that can be unpacked directly into `add_question()`.

**Supported formats**: `.xlsx`, `.xls`, `.csv`, `.tsv`

## Working with Questions

After adding questions, you can inspect them:

```python
# List all question IDs
ids = benchmark.get_question_ids()
print(f"Question IDs: {len(ids)} total")

# Access all questions
questions = benchmark.get_all_questions()
for q in questions[:2]:
    print(f"  - {q['question'][:50]}...")

# Get a specific question by ID
first_id = ids[0]
question_data = benchmark.get_question(first_id)
print(f"\nFirst question: {question_data['question'][:60]}...")
print(f"Expected answer: {question_data['raw_answer'][:60]}...")
```

## add_question() Reference

Full method signature:

    benchmark.add_question(
        question,                  # str or Question object (required)
        raw_answer=None,           # str — required if question is str
        answer_template=None,      # str, type, or None
        question_id=None,          # str — auto-generated if None
        finished=<auto>,           # bool — True if template provided, False otherwise
        author=None,               # dict — author information
        sources=None,              # list[dict] — source references
        custom_metadata=None,      # dict — arbitrary metadata
        few_shot_examples=None,    # list[dict] — few-shot examples
    ) -> str                       # returns question ID

**Key behaviors**:

- **Auto-finished**: Providing an `answer_template` automatically sets `finished=True` unless explicitly overridden
- **Deterministic IDs**: Question IDs are generated from the question text using MD5 hashing. Duplicate texts get suffixed (`-1`, `-2`, etc.)
- **Class renaming**: Answer template classes are automatically renamed to `"Answer"` (required by the verification system)
- **Default templates**: If no template is provided, a minimal default template is generated that always returns `False` from `verify()`

---

## Next Steps

- [Writing Templates](writing-templates.md) — define evaluation criteria for your questions
- [Defining Rubrics](defining-rubrics.md) — add quality assessment traits
- [Saving Benchmarks](saving-benchmarks.md) — persist your benchmark to JSON-LD or database
- [Answer Templates](../core_concepts/answer-templates.md) — concept guide for templates
