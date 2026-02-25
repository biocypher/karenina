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

# Benchmarks

A benchmark is Karenina's self-contained evaluation unit. It bundles questions, answer templates, rubric traits, and metadata into a portable, versioned package that can be saved, shared, loaded, and executed. Everything needed to reproduce an evaluation lives inside the benchmark.

```python tags=["hide-cell"]
# Mock cell: ensures examples execute without live API keys.
# This cell is hidden in rendered documentation.
import hashlib
from datetime import datetime
from unittest.mock import MagicMock, patch

# Minimal mock so Benchmark can be created without a database
mock_modules = {}
for mod in ["sqlalchemy", "sqlalchemy.orm", "sqlalchemy.ext", "sqlalchemy.ext.declarative"]:
    mock_modules[mod] = MagicMock()

with patch.dict("sys.modules", mock_modules):
    from karenina.benchmark import Benchmark
    from karenina.schemas.entities import BaseAnswer, Rubric
    from karenina.schemas.entities.rubric import LLMRubricTrait, RegexTrait
    from pydantic import Field
```

## What Is a Benchmark?

At its core, a benchmark is four things:

1. **Self-contained**: all evaluation data (questions, templates, rubrics, metadata) lives inside the object
2. **Portable**: benchmarks serialize to JSON-LD checkpoint files that can be shared across teams and environments
3. **Versioned**: name, version, description, and timestamps track the benchmark's evolution
4. **Executable**: given a `VerificationConfig`, a benchmark can run its full verification pipeline

```python
# Create a new benchmark
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

## The Facade Pattern

`Benchmark` is a **facade** that delegates to seven specialized managers. You never need to call managers directly; the Benchmark class exposes all their methods on its own surface. Understanding the managers helps when reading source code or debugging:

| Manager | Responsibility | Example Methods |
|---------|---------------|-----------------|
| `MetadataManager` | Name, version, creator, timestamps, custom properties | `get_custom_property()`, `set_custom_property()` |
| `QuestionManager` | CRUD operations, metadata, finished status, filtering | `add_question()`, `get_question()`, `filter_questions()` |
| `TemplateManager` | Answer template storage, validation, default detection | `add_answer_template()`, `has_template()`, `validate_templates()` |
| `RubricManager` | Global and question-specific rubric management | `set_global_rubric()`, `add_question_rubric_trait()` |
| `ResultsManager` | Verification result storage and retrieval | `get_verification_results()`, `get_verification_summary()` |
| `VerificationManager` | Pipeline execution | `run_verification()` |
| `ExportManager` | Serialization, statistics, readiness checks | `check_readiness()`, `get_health_report()`, `to_dict()` |

<div class="admonition note">
<p class="admonition-title">You interact with Benchmark, not the managers</p>
<p>All manager methods are exposed directly on the <code>Benchmark</code> class. For example, <code>benchmark.add_question(...)</code> delegates to <code>QuestionManager.add_question(...)</code> internally. The manager layer exists for code organization, not for you to use directly.</p>
</div>

## Creating vs Loading

There are two ways to get a Benchmark instance:

**Creating a new benchmark:**

```python
# Benchmark.create() is an alias for the constructor
bm = Benchmark.create(name="My Benchmark", version="1.0.0")

# These are equivalent:
bm2 = Benchmark(name="My Benchmark", version="1.0.0")
```

**Loading from a checkpoint:**

```python
# Load a previously saved benchmark
# bm = Benchmark.load(Path("my_benchmark.jsonld"))
```

<div class="admonition info">
<p class="admonition-title"><code>create()</code> is an alias</p>
<p><code>Benchmark.create(name, description, version, creator)</code> is syntactic sugar for the constructor. Both accept the same parameters and produce the same result. Use whichever reads better in your code.</p>
</div>

For checkpoint format details, see [Checkpoints](../checkpoints.md).

## Default Templates: The Invisible Placeholder

When you add a question without specifying an answer template, the benchmark generates a **minimal default template**. This placeholder contains a single `response` field and a `verify()` that always returns `False`.

```python
# Add a question without a template
q_id = benchmark.add_question(
    question="What is the putative target of venetoclax?",
    raw_answer="BCL2 (B-cell lymphoma 2)",
)

# The benchmark has a question, but has_template() returns False
print(f"Question count:  {benchmark.question_count}")
print(f"Has template:    {benchmark.has_template(q_id)}")
```

The default template exists so the benchmark's internal data structure remains consistent (every question always has a template string), but `has_template()` returns `False` for defaults. This distinction matters for readiness checking: a benchmark may have questions but not be "ready" because templates are still placeholders.

```python
# Now add a real template
class Answer(BaseAnswer):
    target: str = Field(
        description="The protein target of the drug mentioned in the response"
    )

    def ground_truth(self):
        self.correct = {"target": "BCL2"}

    def verify(self) -> bool:
        return self.target.strip().upper().replace("-", "") == self.correct["target"]

benchmark.add_answer_template(q_id, Answer)
print(f"Has template (after):  {benchmark.has_template(q_id)}")
```

<div class="admonition tip">
<p class="admonition-title">Default templates always fail verification</p>
<p>The auto-generated default template's <code>verify()</code> returns <code>False</code>. If your benchmark reports 0% accuracy, check whether all questions have real templates using <code>benchmark.get_missing_templates()</code>.</p>
</div>

## Template Attachment: How Templates Connect to Questions

There are three ways to attach a template to a question:

### 1. Inline (at question creation time)

Pass the template class or code string directly when adding the question:

```python
class DrugMechanismAnswer(BaseAnswer):
    mechanism: str = Field(
        description="The mechanism of action described in the response"
    )

    def ground_truth(self):
        self.correct = {"mechanism": "BH3 mimetic"}

    def verify(self) -> bool:
        return "bh3" in self.mechanism.lower() and "mimetic" in self.mechanism.lower()

q2_id = benchmark.add_question(
    question="What is the mechanism of action of venetoclax?",
    raw_answer="BH3 mimetic that selectively inhibits BCL2",
    answer_template=DrugMechanismAnswer,
)
print(f"Has template: {benchmark.has_template(q2_id)}")
```

<div class="admonition tip">
<p class="admonition-title">Auto-renaming to <code>Answer</code></p>
<p>When you pass a class (like <code>DrugMechanismAnswer</code>) as <code>answer_template</code>, the benchmark captures its source code and renames it to <code>Answer</code> automatically. The verification system requires all template classes to be named <code>Answer</code>. You can use any descriptive name in your code; the rename is transparent.</p>
</div>

### 2. After the fact

Add the template to an existing question using its ID:

```python
# benchmark.add_answer_template(question_id, template_code_or_class)
```

### 3. Global template

Apply a single template to all questions that do not yet have one:

```python
# benchmark.apply_global_template(template_code)
# Returns a list of question IDs that received the template
```

## Rubric Attachment and Scope

Rubrics attach at two levels:

- **Global rubrics** apply to every question in the benchmark
- **Question-specific rubrics** apply to a single question

When both exist for a question, they merge. If a question-specific trait has the same name as a global trait, the question-specific one takes precedence.

```python
# Set a global rubric
global_rubric = Rubric(llm_traits=[
    LLMRubricTrait(
        name="conciseness",
        description="True if the response is concise and avoids unnecessary elaboration.",
        kind="boolean",
        higher_is_better=True,
    ),
])
benchmark.set_global_rubric(global_rubric)

# Add a question-specific trait
benchmark.add_question_rubric_trait(
    q_id,
    RegexTrait(
        name="mentions_bcl2_family",
        pattern=r"(?i)bcl[-\s]?2\s+family",
        description="Response mentions the BCL2 protein family.",
        higher_is_better=True,
    ),
)

print(f"Global rubric set:   {benchmark.get_global_rubric() is not None}")
print(f"Q1 has rubric:       {benchmark.get_questions_with_rubric()}")
```

For trait types and detailed rubric authoring, see [Rubrics](../rubrics/index.md).

## Benchmark as a Collection

`Benchmark` implements Python's collection protocols, so you can use it like a container:

```python
# Length: number of questions
print(f"len(benchmark) = {len(benchmark)}")

# Containment: check if a question ID exists
print(f"q_id in benchmark: {q_id in benchmark}")

# Iteration: loop over questions
for question_data in benchmark:
    print(f"  - {question_data['question'][:60]}...")

# Indexing: access by position or ID
first_question = benchmark[0]
print(f"First question type: {type(first_question).__name__}")
```

Slicing is also supported: `benchmark[0:2]` returns a list of `SchemaOrgQuestion` objects.

## Readiness: How Everything Ties Together

Before running verification, the benchmark needs all its pieces in place. The `check_readiness()` method validates this:

| Check | What It Verifies |
|-------|-----------------|
| `has_questions` | At least one question exists |
| `all_have_templates` | Every question has a real template (not a default) |
| `all_finished` | Every question is marked as finished |
| `templates_valid` | All template code is syntactically valid Python |
| `rubrics_valid` | All rubric traits have required fields |

The overall `ready_for_verification` flag is `True` only when all checks pass.

```python
# Check readiness of our partially built benchmark
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

### Building Readiness Incrementally

In practice, you build a benchmark piece by piece and readiness evolves as you go:

```python
# Start fresh
bm = Benchmark.create(name="Readiness Demo", version="0.1.0")

# Step 1: empty benchmark
r = bm.check_readiness()
print(f"Step 1 (empty):       ready={r['ready_for_verification']}")

# Step 2: add a question (gets a default template)
demo_id = bm.add_question(
    question="What protein does imatinib target?",
    raw_answer="BCR-ABL tyrosine kinase",
)
r = bm.check_readiness()
print(f"Step 2 (question):    ready={r['ready_for_verification']}, "
      f"has_templates={r['all_have_templates']}")

# Step 3: add a real template
class Answer(BaseAnswer):
    target: str = Field(description="The protein target")
    def ground_truth(self):
        self.correct = {"target": "BCR-ABL"}
    def verify(self) -> bool:
        return "bcr" in self.target.lower() and "abl" in self.target.lower()

bm.add_answer_template(demo_id, Answer)
r = bm.check_readiness()
print(f"Step 3 (template):    ready={r['ready_for_verification']}, "
      f"has_templates={r['all_have_templates']}")
```

<div class="admonition tip">
<p class="admonition-title">Run <code>check_readiness()</code> before <code>run_verification()</code></p>
<p>If verification produces zero results or unexpected failures, the first thing to check is readiness. Common issues: default templates still in place, questions not marked as finished, or a rubric trait missing its description.</p>
</div>

### Health Report

For a higher-level view, `get_health_report()` returns a scored assessment (0 to 100) with status levels and actionable recommendations:

```python
report = bm.get_health_report()
print(f"Health score:  {report['health_score']}")
print(f"Health status: {report['health_status']}")
if report.get('recommendations'):
    print("Recommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")
```

The scoring breakdown:

| Component | Points | Criteria |
|-----------|--------|----------|
| Questions exist | 20 | At least one question added |
| Completion progress | 30 | Based on `get_progress()` percentage |
| Templates valid | 25 | All templates parse as valid Python |
| Rubrics valid | 15 | All rubric traits properly configured |
| All finished | 10 | Every question marked as finished |

Status levels: **excellent** (90+), **good** (75+), **fair** (50+), **poor** (25+), **critical** (below 25).

## Filtering and Search

Benchmarks provide three categories of query methods for working with questions:

### Filter by Attributes

```python
# Filter by system fields
finished_qs = benchmark.filter_questions(finished=True)
templated_qs = benchmark.filter_questions(has_template=True)

# Combine filters
ready_qs = benchmark.filter_questions(finished=True, has_template=True)
print(f"Finished questions:  {len(finished_qs)}")
print(f"With templates:      {len(templated_qs)}")
print(f"Ready for pipeline:  {len(ready_qs)}")
```

```python
# Filter by custom metadata
# benchmark.filter_by_custom_metadata(domain="pharmacology", difficulty="hard")

# Filter with a lambda
# benchmark.filter_questions(
#     custom_filter=lambda q: q.get("custom_metadata", {}).get("priority") == "high"
# )
```

### Search by Text

```python
# Search question text and answers
results = benchmark.search_questions("venetoclax")
print(f"Questions mentioning 'venetoclax': {len(results)}")
```

### Aggregate

```python
# Count questions grouped by finished status
counts = benchmark.count_by_field("finished")
print(f"By finished status: {counts}")
```

## Benchmark-Level Metadata

Beyond the standard fields (name, version, description, creator), benchmarks support custom properties for domain-specific metadata:

```python
# Set custom properties
benchmark.set_custom_property("domain", "pharmacology")
benchmark.set_custom_property("target_audience", "drug discovery teams")

# Retrieve them
domain = benchmark.get_custom_property("domain")
print(f"Domain: {domain}")

# Get all at once
all_props = benchmark.get_all_custom_properties()
print(f"All properties: {all_props}")
```

Custom properties are distinct from question metadata. Use benchmark-level properties for attributes that characterize the benchmark as a whole (domain, intended use, regulatory context). Use question metadata for per-question attributes (author, sources, difficulty).

## Common Pitfalls

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| 0% accuracy across all questions | Default templates still in place; `verify()` always returns `False` | Check `benchmark.get_missing_templates()` and add real templates |
| `is_complete` returns `False` | Some questions lack real templates or are not marked finished | Check `benchmark.get_unfinished_questions()` and `benchmark.get_missing_templates()` |
| `check_readiness()` says ready but verification returns no results | Questions have templates but are not marked `finished` | Call `benchmark.mark_finished(question_id)` for each question |
| `validate_templates()` reports errors | Template code has syntax errors or missing imports | Review template code; ensure it inherits from `BaseAnswer` |
| Global template does not appear on a question | `apply_global_template()` only applies to questions without templates | Questions that already have a template (even a default) may need explicit `add_answer_template()` |
| Rubric trait name collision raises `ValueError` | A question-specific trait has the same name as a global trait | When both scopes exist, question-specific traits override globals with the same name; ensure names are intentionally unique or use override semantics |

<div class="admonition warning">
<p class="admonition-title">Default templates are invisible but present</p>
<p>Every question always has a template string in the benchmark's internal data. When no template is provided, a default is generated. The <code>has_template()</code> method distinguishes defaults from real templates. Methods like <code>get_template()</code> raise <code>ValueError</code> for questions with only a default, reinforcing that defaults are placeholders, not usable templates.</p>
</div>

## Next Steps

- [Questions deep dive](../../notebooks/core_concepts/questions-and-benchmarks/questions.ipynb): the Question schema, deterministic IDs, `raw_answer`, metadata layers
- [Answer Templates](../../notebooks/core_concepts/answer-templates.ipynb): how to write templates, field types, verification patterns
- [Rubrics](../rubrics/index.md): trait types, global vs question-specific, deep judgment
- [Checkpoints](../checkpoints.md): how benchmarks persist as JSON-LD files
- [Creating Benchmarks](../../workflows/creating-benchmarks/index.md): step-by-step authoring workflow
