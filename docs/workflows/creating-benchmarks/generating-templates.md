# Generating Templates

When a benchmark has many questions, writing templates by hand for each one
is time-consuming. Karenina provides **automated template generation** that
uses an LLM to produce `Answer` classes from question/answer pairs.

The generator analyses each question and its reference answer, extracts
ground-truth attributes, and produces a complete Pydantic class with
`verify()` logic — ready to use in verification.

## How It Works

Template generation uses a **two-phase structured approach**:

```
Question + Reference Answer
          │
          ▼
  Phase 1: Ground-Truth Extraction
  LLM identifies attributes, types, and expected values
          │
          ▼
  Phase 2: Field Description Generation
  LLM writes judge instructions for each attribute
          │
          ▼
  Phase 3: Code Generation
  Python/Pydantic class assembled from phases 1–2
          │
          ▼
  Answer class with verify() method
```

**Phase 1** extracts a minimal set of typed attributes from the reference
answer. The generator favours `bool` attributes (for concept presence) and
`Literal` types (for exact values) over free-text strings, keeping
verification deterministic.

**Phase 2** produces field descriptions that instruct the judge LLM on
_how_ to fill in each attribute from a candidate response. These become the
`Field(description=...)` values in the generated class.

**Phase 3** assembles the Pydantic class code, including `model_post_init`
with ground-truth values and a `verify()` method with type-appropriate
comparison logic (equality for booleans/literals, tolerance for floats,
set comparison for lists).

## Generating for a Single Question

Use `generate_template_for_question()` on a `Benchmark` that already has
questions with reference answers:

```python
result = benchmark.generate_template_for_question(
    question_id="urn:uuid:question-abc123",
    model="gpt-4o",
    model_provider="openai",
)

if result["success"]:
    print(result["template_code"])
```

The return value is a dictionary with these keys:

| Key | Type | Description |
|-----|------|-------------|
| `success` | `bool` | Whether generation succeeded |
| `template_code` | `str` | The generated Python class code |
| `error` | `str \| None` | Error message if generation failed |
| `raw_response` | `str \| None` | Raw LLM response (for debugging) |
| `skipped` | `bool` | `True` if a template already exists |

If the question already has a template, the method returns immediately with
`skipped=True`. Pass `force_regenerate=True` to override.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `"gemini-2.0-flash"` | LLM model name |
| `model_provider` | `"google_genai"` | Provider (openai, anthropic, google_genai) |
| `temperature` | `0` | Generation temperature |
| `interface` | `"langchain"` | Adapter interface |
| `force_regenerate` | `False` | Regenerate even if a template exists |
| `endpoint_base_url` | `None` | Custom OpenAI-compatible endpoint URL |
| `endpoint_api_key` | `None` | API key for custom endpoint |

## Generating for Multiple Questions

### Selected Questions

```python
results = benchmark.generate_templates(
    question_ids=["urn:uuid:question-abc123", "urn:uuid:question-def456"],
    model="gpt-4o",
    model_provider="openai",
)

for qid, result in results.items():
    status = "OK" if result["success"] else "FAIL"
    print(f"{qid}: {status}")
```

### All Questions at Once

```python
results = benchmark.generate_all_templates(
    model="gpt-4o",
    model_provider="openai",
    only_missing=True,  # Skip questions that already have templates
)

print(f"Generated {sum(1 for r in results.values() if r['success'])} templates")
```

The `only_missing` parameter (default `True`) skips questions that already
have templates. Set `only_missing=False` combined with
`force_regenerate=True` to regenerate all templates from scratch.

### Progress Tracking

Both `generate_templates()` and `generate_all_templates()` accept a
`progress_callback`:

```python
def on_progress(percentage: float, message: str) -> None:
    print(f"{percentage:.0f}%: {message}")

results = benchmark.generate_all_templates(
    model="gpt-4o",
    model_provider="openai",
    progress_callback=on_progress,
)
```

## Reviewing Generated Templates

Generated templates are automatically added to the benchmark. Inspect them
with the standard template inspection methods:

```python
# View a generated template
template_code = benchmark.get_template("urn:uuid:question-abc123")
print(template_code)
```

A generated template looks like:

```python
class Answer(BaseAnswer):
    mentions_bcl2: bool = Field(
        description="Answer with true if the response mentions BCL2 "
        "or BCL-2 as the target gene; otherwise answer false."
    )
    mentions_apoptosis: bool = Field(
        description="Answer with true if the response refers to "
        "apoptosis or programmed cell death; otherwise answer false."
    )

    def model_post_init(self, __context):
        self.correct = {"mentions_bcl2": True, "mentions_apoptosis": True}

    def verify(self) -> bool:
        return (
            self.mentions_bcl2 == self.correct["mentions_bcl2"] and
            self.mentions_apoptosis == self.correct["mentions_apoptosis"]
        )

    def verify_granular(self) -> float:
        correct_count = 0
        total_count = 2

        if self.mentions_bcl2 == self.correct["mentions_bcl2"]:
            correct_count += 1
        if self.mentions_apoptosis == self.correct["mentions_apoptosis"]:
            correct_count += 1

        return correct_count / total_count
```

Key characteristics of generated templates:

- **Boolean attributes** for concept presence (preferred over string matching)
- **Literal types** for exact value matching (IDs, codes, specific terms)
- **No string attributes** — the generator forbids `str`, `List[str]`, and `Dict[str, str]` types to keep verification deterministic
- **`verify_granular()`** is included automatically when there are multiple attributes, providing partial credit scores
- **Float tolerance** — float comparisons use a default tolerance of `0.001`

## Exporting and Importing Templates

Templates can be exported to a JSON file for review, sharing, or version
control:

```python
from pathlib import Path

# Export all templates to JSON
benchmark.export_generated_templates(Path("templates.json"))

# Import templates into a different benchmark
benchmark2.import_generated_templates(Path("templates.json"))
```

The JSON file maps question IDs to template code strings:

```json
{
  "urn:uuid:question-abc123": "class Answer(BaseAnswer):\n    ...",
  "urn:uuid:question-def456": "class Answer(BaseAnswer):\n    ..."
}
```

`import_generated_templates()` returns a dictionary mapping question IDs
to `bool` indicating success. It skips questions that already have templates
unless `force_overwrite=True` is passed.

## The Standalone Function

For use outside the `Benchmark` API, the `generate_answer_template`
function generates a single template from a question/answer pair:

```python
from karenina.benchmark.authoring import generate_answer_template

template_code = generate_answer_template(
    question="What gene does vemurafenib target?",
    raw_answer="BRAF",
    model="gpt-4o",
    model_provider="openai",
    interface="langchain",
)

print(template_code)
```

This returns the Python code string directly (not wrapped in a result
dictionary). You can also pass a `ModelConfig` object via the `config`
parameter instead of individual model parameters.

## The AnswerBuilder

For cases where you want to construct a template programmatically
_without_ an LLM call, use `AnswerBuilder`. It provides a fluent
interface for defining attributes and regex patterns, then compiles
them into an `Answer` class:

```python
from karenina.benchmark.authoring.answers.builder import AnswerBuilder

builder = (
    AnswerBuilder()
    .add_attribute("mentions_drug", "bool", "Whether the drug is mentioned", True)
    .add_attribute("mentions_target", "bool", "Whether the target is mentioned", True)
    .add_regex("citations", r"\[\d+\]", expected=3, match_type="count")
)

Answer = builder.compile()
benchmark.add_question(
    "What does vemurafenib target?",
    "BRAF",
    answer_template=Answer,
)
```

### Builder Methods

| Method | Description |
|--------|-------------|
| `add_attribute(name, type, description, ground_truth)` | Add a Pydantic field |
| `remove_attribute(name)` | Remove an attribute |
| `add_regex(name, pattern, expected, match_type, description)` | Add a regex pattern |
| `remove_regex(name)` | Remove a regex pattern |
| `compile(class_name="Answer")` | Compile to an executable `Answer` class |

The `match_type` parameter for `add_regex` supports: `"exact"`,
`"contains"`, `"count"`, and `"all"`.

## Model Selection

The default model for template generation is `gemini-2.0-flash` via the
`google_genai` provider. This default is optimized for speed and cost in
bulk generation. For higher-quality templates on complex questions, consider:

| Use Case | Recommended Model | Provider |
|----------|-------------------|----------|
| Bulk generation | `gemini-2.0-flash` | `google_genai` |
| Complex questions | `gpt-4o` | `openai` |
| Custom endpoint | Any compatible model | `openai_endpoint` |

The generator uses structured output (Pydantic schema parsing) internally,
so the chosen model must support structured output through the selected
interface.

## Tips

- **Start with `generate_all_templates(only_missing=True)`** to generate
  templates for all questions that don't already have one
- **Review generated templates** before running verification — the LLM
  may make incorrect assumptions about ground truth
- **Regenerate selectively** using `generate_template_for_question()` with
  `force_regenerate=True` for templates that need improvement
- **Export before regenerating** using `export_generated_templates()` to
  keep a backup of the current templates
- **Use temperature 0** (the default) for deterministic, reproducible
  template generation
- **Write custom templates** for questions where the generated logic is
  insufficient — see [Writing Custom Templates](writing-templates.md)

## Next Steps

- [Writing Custom Templates](writing-templates.md) — hand-craft templates
  for complex verification logic
- [Defining Rubrics](defining-rubrics.md) — add quality evaluation traits
- [Saving Benchmarks](saving-benchmarks.md) — persist your work
- [Running Verification](../06-running-verification/index.md) — execute
  verification with generated templates
- [Answer Templates](../core_concepts/answer-templates.md) — template
  concepts and field types
