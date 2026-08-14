---
name: karenina-qa
description: >
  Build and run single-turn QA benchmarks with karenina. Use when evaluating
  an LLM on factual questions, extraction tasks, or any single-turn Q&A
  evaluation. Covers question definition, template attachment, checkpoint
  management, running verification, and result analysis. Invoke for
  benchmark creation, loading, or iteration.
---

# Single-Turn QA Benchmark

## What This Skill Does

Guides creation and execution of single-turn QA benchmarks: define questions
with known answers, attach answer templates that specify extraction and
verification logic, configure the verification pipeline, run it, and analyze
results.

## Prerequisites

1. karenina installed (`uv pip install -e karenina` from the monorepo root)
2. API key for the provider you plan to use present in the **process
   environment** (e.g. `ANTHROPIC_API_KEY` exported in the shell that runs
   the code). The repo-root `.env` is NOT auto-loaded when running from
   `karenina/` — source it explicitly before launching. Without a key,
   verification fails at the `generate_answer` stage with
   `TypeError: Could not resolve authentication method...`, recorded as
   `FailureCategory.UNEXPECTED_ERROR`. As an alternative to the environment
   variable, pass `ModelConfig(anthropic_api_key=...)` per model.

**If you are unsure about any API (constructor parameters, method names, field types), invoke the `using-karenina` skill and check its `references/` directory for full documentation.** That skill contains the complete karenina docs and is the authoritative source for API details.

## Step-by-Step Procedure

Follow these eight steps in order. Do not skip steps; each builds on the previous.

### Step 1: Initialize or Load a Benchmark

Create a new benchmark or resume from a saved checkpoint.

**New benchmark:**

```python
from karenina.benchmark import Benchmark

benchmark = Benchmark(name="my-qa-benchmark")
```

**Resume from checkpoint:**

```python
from pathlib import Path
from karenina.benchmark import Benchmark

benchmark = Benchmark.load(Path("my-qa-benchmark.jsonld"))
```

Checkpoints persist all questions, templates, and rubrics — not verification
results. Use `Benchmark.load()` to continue where you left off; export
results separately with `ResultsStore.export_to_file()` (see Step 8).

### Step 2: Define Questions

Each question requires `question` (the text sent to the answering LLM) and
`raw_answer` (the ground truth reference). Optional fields include `keywords`
for categorization and `answer_notes` for interpretation guidance.

```python
from karenina.schemas.entities.question import Question

q1 = Question(
    question="What is the primary pharmacological target of venetoclax?",
    raw_answer="BCL2 (B-cell lymphoma 2)",
)

q2 = Question(
    question="What is the mechanism of action of metformin?",
    raw_answer="Metformin activates AMP-activated protein kinase (AMPK)",
)

benchmark.add_question(q1)
benchmark.add_question(q2)
```

For bulk loading, `add_questions()` accepts a list of Question objects or dicts:

```python
benchmark.add_questions([q1, q2])

# Or from a CSV file:
from karenina.benchmark.authoring.questions import extract_questions_from_file

questions = extract_questions_from_file(
    file_path="questions.csv",
    question_column="Question",
    answer_column="Answer",
    keywords_columns=[{"column": "Area", "separator": ","}],
    custom_metadata_columns=["Complexity"],
)
benchmark.add_questions(questions)
```

`Question.id` is the MD5 hash of the question text. Adding a question with
identical text silently overwrites the previous entry.

### Step 3: Create Answer Templates

Each question needs an answer template: a `BaseAnswer` subclass that tells the
judge LLM what to extract and how to verify correctness.

**Ask the user:** "Do you want to auto-generate templates using an LLM, or
author them manually? Auto-generation is recommended for most benchmarks;
manual authoring gives fine-grained control over individual templates."

**If auto-generate:** Ask the user which model to use for generation.

For all questions at once (via Benchmark):

```python
benchmark.generate_all_templates(
    model="claude-sonnet-4-6",       # ask user for model
    model_provider="anthropic",
)
```

Templates are attached automatically. The `only_missing=True` default makes
this safe for incremental workflows: add new questions, re-run, and only the
gaps get filled.

For a single question (standalone, no Benchmark needed):

```python
from karenina.benchmark.authoring.answers import generate_answer_template
from karenina.schemas.config.models import ModelConfig

config = ModelConfig(
    id="gen",
    model_name="claude-sonnet-4-6",  # ask user for model
    model_provider="anthropic",
    temperature=0.0,
)
template_code = generate_answer_template(question_obj=question, config=config)
```

This returns the template as a Python source string. After generation,
individual templates can be refined manually if needed.

**If manual authoring:** Delegate to the `karenina-template-authoring` skill
for detailed guidance on field types, verification primitives, and description
writing. Templates can be attached in two ways:

```python
# Pass the class when adding the question
benchmark.add_question(q1, answer_template=Answer)

# Or attach after the question exists
benchmark.add_answer_template(q1.id, template_code)
```

For a minimal template example, see `assets/benchmark-skeleton.py` in this
skill directory.

### Step 4: (Optional) Define Rubrics

Rubrics assess response quality beyond correctness (safety, conciseness,
citation quality). **You MUST invoke the `karenina-rubric-authoring` skill
before constructing any `Rubric` or trait object.** The rubric API has
type-specific trait lists and required fields that differ by trait type;
the rubric-authoring skill guides you through these correctly.

After creating the rubric, attach it to the benchmark:

```python
benchmark.set_global_rubric(rubric)
```

Attaching a rubric does NOT change the evaluation mode. The default
`evaluation_mode` is `"template_only"`, which runs template verification only
and skips any attached rubric. To evaluate attached rubrics you MUST set
`evaluation_mode="template_and_rubric"` (or `"rubric_only"` for rubric-only
evaluation) explicitly in Step 6.

### Step 5: Save the Checkpoint

Once the benchmark definition is complete (questions, templates, and any
rubrics), save it to disk before running verification:

```python
benchmark.save("my-qa-benchmark.jsonld")
```

The checkpoint stores the benchmark definition only — questions, templates,
and rubrics — not verification results (see Gotcha 1). Results live in memory
and are exported separately in Step 8.

### Step 6: Configure the Verification Pipeline

**Before writing configuration, ask the user which models to use for
answering and parsing (judging).** Do not assume a default model. Example
prompt: "Which model should I use for answering and which for judging? For
example, `claude-sonnet-4-6` for answering and `claude-haiku-4-5` for judging."

**Simple configuration (inline):**

```python
from karenina.schemas.config.models import ModelConfig
from karenina.schemas.verification.config import VerificationConfig

config = VerificationConfig(
    answering_models=[
        ModelConfig(
            id="answering",
            model_provider="anthropic",
            model_name="claude-sonnet-4-6",  # ask user for model
        ),
    ],
    parsing_models=[
        ModelConfig(
            id="judge",
            model_provider="anthropic",
            model_name="claude-haiku-4-5",  # ask user for model
        ),
    ],
)
```

`evaluation_mode` is omitted here, so it defaults to `"template_only"`.

**With rubrics:** attaching a rubric alone does not enable rubric evaluation.
Set the mode explicitly:

```python
config = VerificationConfig(
    answering_models=[...],
    parsing_models=[...],
    evaluation_mode="template_and_rubric",  # required to evaluate attached rubrics
)
```

For complex configurations (guards, deep judgment, replicates, presets, MCP
integration), delegate to the `karenina-verification` skill.

The pipeline reads its behavior from `evaluation_mode`:

- `evaluation_mode="template_only"` (default): only template verification runs; any attached rubric is skipped
- `evaluation_mode="template_and_rubric"`: both template and rubric evaluation run
- `evaluation_mode="rubric_only"`: only rubric evaluation runs (no template needed)

### Step 7: Run Verification

**Python API:**

```python
results = benchmark.run_verification(config)
summary = results.get_summary()
print(summary)
```

**CLI:**

```bash
karenina verify checkpoint.jsonld --preset presets/my-preset.json --output results.json
```

The pipeline assembles its stage sequence dynamically from the configuration:
validate template, generate answer, guards/autofails, optional abstention and
sufficiency checks, parse, verify, embedding check, optional deep judgment,
optional rubric evaluation, then finalize. Which stages run depends on the
enabled features and `evaluation_mode`; there is no fixed stage count.
Progress is reported per question.

### Step 8: Analyze Results

```python
from karenina.schemas.results.failure import Failure, FailureCategory

for result in results:
    q = result.metadata.question_text
    ok = result.metadata.failure is None
    failure = result.metadata.failure          # Failure | None

    # Template (if template evaluation ran)
    if result.template:
        passed = result.template.verify_result      # bool or None
        response = result.template.raw_llm_response  # full LLM text
        parsed = result.template.parsed_llm_response # dict of extracted fields

    # Rubric (if rubric evaluation ran)
    if result.rubric:
        llm_scores = result.rubric.llm_trait_scores       # dict[str, int|bool]
        regex_scores = result.rubric.regex_trait_scores    # dict[str, bool]
        all_scores = result.rubric.get_all_trait_scores()  # flat dict
```

A question failed when `result.metadata.failure is not None`. Inspect
`failure.category` (`FailureCategory`, e.g. `recursion_limit`,
`trace_validation`, `abstention`, `sufficiency`, `parsing`, `timeout`),
`failure.stage` (the stage that failed), and `failure.reason` for the message.

**Do NOT use flat accessors** like `result.question_id` or
`result.verify_result`. Always go through the sub-objects: `result.metadata`,
`result.template`, `result.rubric`.

DataFrame access:

```python
from karenina.benchmark.core import ResultsStore

# Template results as a DataFrame
template_df = results.get_template_results().to_dataframe()

# Rubric results (if evaluation_mode includes rubrics)
rubric_df = results.get_rubrics_results().to_dataframe()

# Export to file (ResultsStore replaces the deprecated Benchmark export)
store = ResultsStore()
store.add(results)                    # returns the auto-generated run name
store.export_to_file("results.json")
```

## Quick-Start Snippet

The complete minimal benchmark below is also available at
`assets/benchmark-skeleton.py`:

```python
"""Minimal karenina QA benchmark skeleton."""
from karenina.benchmark import Benchmark
from karenina.schemas.config.models import ModelConfig
from karenina.schemas.entities import BaseAnswer, VerifiedField
from karenina.schemas.entities.question import Question
from karenina.schemas.primitives import BooleanMatch
from karenina.schemas.verification.config import VerificationConfig

question = Question(
    question="What is the primary pharmacological target of venetoclax?",
    raw_answer="BCL2 (B-cell lymphoma 2)",
)

class Answer(BaseAnswer):
    identifies_target: bool = VerifiedField(
        description=(
            "True if the response identifies BCL2 (including Bcl-2, BCL-2, "
            "or B-cell lymphoma 2) as the primary pharmacological target. "
            "False if BCL2 is not mentioned or a different protein is "
            "identified as the primary target."
        ),
        ground_truth=True,
        verify_with=BooleanMatch(),
    )

benchmark = Benchmark(name="drug-target-extraction")
benchmark.add_question(question, answer_template=Answer)

config = VerificationConfig(
    answering_models=[
        ModelConfig(
            id="answering",
            model_provider="anthropic",
            model_name="claude-sonnet-4-6",  # ask user for model
        ),
    ],
    parsing_models=[
        ModelConfig(
            id="judge",
            model_provider="anthropic",
            model_name="claude-haiku-4-5",  # ask user for model
        ),
    ],
)

results = benchmark.run_verification(config)
print(results.get_summary())
```

## Curation Workflow

To build a benchmark from an expert-authored table and hand it to a curator,
combine the authoring helpers:

```python
from karenina.benchmark import Benchmark, export_curation_workbook
from karenina.benchmark.authoring.questions import extract_questions_from_file
from karenina.benchmark.authoring.answers import generate_answer_template

# 1. Import rows from CSV/TSV/Excel -> list[Question]
questions = extract_questions_from_file(
    file_path="questions.csv", question_column="Question", answer_column="Answer",
    sheet_name=None, author_name_column=None, author_email_column=None,
    author_affiliation_column=None, url_column=None,
    keywords_columns=[{"column": "Area", "separator": ","}],
    answer_notes_column=None, custom_metadata_columns=["Complexity"],
)

# 2. Draft a template for one question (config overrides individual params)
code = generate_answer_template(questions[0], model="claude-sonnet-4-6",
    model_provider="anthropic", temperature=0.0, interface="langchain",
    endpoint_base_url=None, endpoint_api_key=None, config=None)  # -> str

# 3. Draft templates for all missing questions, incrementally
benchmark = Benchmark.create(name="my-benchmark")
benchmark.add_questions(questions, finished=False)
benchmark.generate_all_templates(
    model="claude-sonnet-4-6",
    model_provider="anthropic",
    only_missing=True,   # default; safe to re-run, fills only the gaps
)

# 4. Emit a field-level Excel workbook for human review
summary = export_curation_workbook(benchmark, "curation_review.xlsx")
# summary.output_path, summary.question_count, summary.field_count,
# summary.draft_count, summary.finished_count, summary.template_parse_error_count
```

`generate_all_templates(..., only_missing=True)` is the default and safe to
re-run: only gaps get filled.

## Gotchas

### 1. Checkpoints Save the Benchmark Definition, Not Results

`Benchmark.save()` writes a JSON-LD checkpoint that includes all questions,
templates, and rubrics. Verification results are NOT persisted: they are
held in memory only and are lost when the process exits. To keep results,
export them explicitly via `ResultsStore` — `store.add(results)` then
`store.export_to_file("results.json")` (see Step 8). Call
`Benchmark.load(path)` to resume from a checkpoint. Do not manually edit
checkpoint files; the JSON-LD structure uses internal references that break
if modified by hand.

### 2. Question.id Is MD5 of Question Text

`Question.id` is a computed property: the MD5 hash of the question text.
Two questions with identical text produce the same ID. Adding a duplicate
silently overwrites the previous entry (including its template and results).
Change the question text to get a distinct ID.

### 3. Auto-Fail Means a Guard Triggered

If a question shows `verify_result=False` with no parsed template output, a
pipeline guard (recursion limit, trace validation, abstention, or sufficiency)
triggered before parsing. Check `result.template.recursion_limit_reached`,
`result.template.abstention_detected`, and `result.metadata.failure` (its
`.category` and `.stage`) to identify which guard fired. See the
`karenina-verification` skill for guard configuration.

### 4. evaluation_mode Defaults to template_only

The pipeline defaults to `evaluation_mode="template_only"`. Attaching a rubric
does NOT change the mode: to evaluate rubric traits you MUST set
`evaluation_mode="template_and_rubric"` (or `"rubric_only"`) explicitly. There
is no `"auto"` value. `evaluation_mode="rubric_only"` with no traits attached
produces no scores.

### 5. Template Attachment Is Required for template_only and template_and_rubric

Every question must have an answer template attached before running
verification in `template_only` or `template_and_rubric` mode. Questions
without templates cause a validation failure at the `validate_template` stage.
If you are only using rubrics, set `evaluation_mode="rubric_only"`.

<!-- AUTO:benchmark-api -->
### Benchmark API Quick Reference

> Auto-generated for karenina v0.1.0.

Do not guess method names. If a method is not listed here,
verify it exists with Grep before calling it.

> All karenina Pydantic models use `extra="forbid"`. Passing
> an undefined field raises `ValidationError`.

#### Create / Load / Save

| Method | Returns |
|--------|---------|
| `Benchmark(name, description=, version=, creator=, workspace_root=)` | `Benchmark` |
| `Benchmark.create(name, description=, version=, creator=, workspace_root=)` | `Benchmark` |
| `Benchmark.load(path, workspace_root=)` | `Benchmark` |
| `save(path, save_deep_judgment_config=)` | `None` |

#### Questions

| Method | Returns |
|--------|---------|
| `add_question(question, raw_answer=, answer_template=, question_id=, ...)` | `str` |
| `add_questions(questions_data, finished=)` | `list[str]` |
| `get_question(question_id)` | `dict[str, Any]` |
| `get_question_ids()` | `list[str]` |
| `get_all_questions(ids_only=)` | `list[str] | list[dict[str, Any]]` |
| `get_question_as_object(question_id)` | `Question` |
| `remove_question(question_id)` | `bool` |

#### Templates

| Method | Returns |
|--------|---------|
| `add_answer_template(question_id, template_code)` | `None` |
| `get_template(question_id)` | `str` |
| `has_template(question_id)` | `bool` |
| `update_template(question_id, template_code)` | `None` |
| `apply_global_template(template_code)` | `list[str]` |
| `get_missing_templates(ids_only=)` | `list[str] | list[dict[str, Any]]` |

#### Rubrics

| Method | Returns |
|--------|---------|
| `set_global_rubric(rubric)` | `None` |
| `get_global_rubric()` | `Rubric | None` |
| `set_question_rubric(question_id, rubric)` | `None` |
| `get_question_rubric(question_id)` | `Rubric | None` |
| `add_global_rubric_trait(trait)` | `None` |
| `add_question_rubric_trait(question_id, trait)` | `None` |
| `clear_global_rubric()` | `bool` |
| `remove_question_rubric(question_id)` | `bool` |

#### Verification

| Method | Returns |
|--------|---------|
| `run_verification(config, question_ids=, run_name=, async_enabled=, ...)` | `VerificationResultSet` |
| `resume_verification(state_path, config=, question_ids=, run_name=, ...)` | `VerificationResultSet` |
| `extend_template(prior_results, config, run_name=, question_ids=, ...)` | `VerificationResultSet` |
| `extend_rubric(prior_results, config, run_name=, question_ids=, ...)` | `VerificationResultSet` |

#### Results

| Method | Returns |
|--------|---------|
| `get_verification_results(question_ids=, run_name=)` | `dict[str, VerificationResult]` |
| `get_verification_history(question_id=)` | `dict[str, dict[str, VerificationResult]]` |
| `get_verification_summary(run_name=)` | `dict[str, Any]` |
| `store_verification_results(results, run_name=)` | `None` |
| `export_verification_results_to_file(file_path, question_ids=, run_name=, format=, global_rubric=)` | `None` |
| `clear_verification_results(question_ids=, run_name=)` | `int` |

#### Methods That Do NOT Exist on Benchmark

| Hallucinated Name | Use Instead |
|-------------------|-------------|
| `set_template()` | `add_answer_template() or update_template()` |
| `get_latest_results()` | `ResultsStore.get_latest() (get_verification_results() is deprecated)` |
| `add_rubric()` | `set_global_rubric() or set_question_rubric()` |
| `load_checkpoint()` | `Benchmark.load()` |
| `verify()` | `run_verification()` |

#### Deprecated Benchmark Methods (emit DeprecationWarning)

| Method | Use Instead |
|--------|-------------|
| `store_verification_results()` | `ResultsStore.add()` |
| `get_verification_results()` | `ResultsStore.get_by_run() / get_latest()` |
| `get_verification_history()` | `ResultsStore.get_by_question()` |
| `get_verification_summary()` | `ResultsStore.get_summary()` |
| `export_verification_results_to_file()` | `ResultsStore.export_to_file()` |
| `clear_verification_results()` | `ResultsStore.clear()` |

<!-- /AUTO:benchmark-api -->

### 6. ExactMatch Normalizers Are Limited

`ExactMatch(normalize=[...])` accepts normalizer names `"lowercase"`,
`"strip"`, `"remove_punctuation"`, `"collapse_whitespace"` — or `SynonymMap`
instances (`Normalizer` is `str | SynonymMap`). There is **no `"uppercase"`
normalizer**. The default `["lowercase", "strip"]` is correct for most cases.
An unrecognized name raises `ValueError: Unknown normalizer: ...` from
`apply_normalizer` at verify time — a loud error, not a silent failure.

### 7. `correct` Is a Reserved BaseAnswer Field Name

A `VerifiedField` named `correct` raises `TypeError` at class-definition
time: `Field name(s) {'correct'} reserved by BaseAnswer for internal use.
Please rename your field(s) to avoid collision.` The reserved set is
`{"correct"}` (`BaseAnswer.correct` holds the per-field ground-truth
mapping). Rename the field — e.g. `is_correct` or `identifies_target`.
