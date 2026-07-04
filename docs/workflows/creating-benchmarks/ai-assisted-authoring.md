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

# AI-Assisted Template Authoring

When building benchmarks with many questions, hand-writing every answer template gives full control but scales poorly. AI-assisted generation speeds up the process: generate a draft from the question and its reference answer, review the output, and refine where needed. For cases where you want programmatic control without writing raw class code, `AnswerBuilder` provides a fluent interface for constructing templates from attribute and regex specifications.

This tutorial covers both approaches and the practical workflow pattern that combines them: bulk generate, review each template, then replace problematic ones with hand-written or builder-constructed versions.

**What you'll learn:**

- Generate a template from a question using `generate_answer_template()`
- Understand the two-phase generation process (ground truth extraction, then field descriptions)
- Review and edit generated template code
- Build templates programmatically with `AnswerBuilder`
- Combine `AnswerBuilder` attributes with regex patterns
- Use `generate_all_templates()` for batch generation
- Workflow pattern: generate, review, refine

```python tags=["hide-cell"]
# Setup cell: hidden in rendered documentation.
# Mocks LLM-dependent operations so examples execute without API keys.
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from karenina import Benchmark
from karenina.schemas.entities.question import Question
from karenina.benchmark.authoring.answers.builder import AnswerBuilder
from karenina.benchmark.benchmark_helpers import TemplateProgressEvent

# Realistic generated template code (matches actual generator output format)
_GENERATED_CODE = """from karenina.schemas.entities import BaseAnswer, VerifiedField
from karenina.schemas.primitives import BooleanMatch

class Answer(BaseAnswer):
    identifies_target: bool = VerifiedField(
        description="True if the response identifies BCL2 as the primary pharmacological target of venetoclax.",
        ground_truth=True,
        verify_with=BooleanMatch(),
    )
    mentions_mechanism: bool = VerifiedField(
        description="True if the response describes venetoclax as a selective BCL2 inhibitor that mimics BH3-only proteins.",
        ground_truth=True,
        verify_with=BooleanMatch(),
    )
"""


def _mock_generate_answer_template(question_obj, **kwargs):
    """Return realistic generated template code without calling an LLM."""
    return _GENERATED_CODE


def _mock_generate_all_templates(self, **kwargs):
    """Simulate batch template generation."""
    results = {}
    for qid in self.get_question_ids():
        if not self.has_template(qid) or kwargs.get("force_regenerate"):
            self.add_answer_template(qid, _GENERATED_CODE)
            results[qid] = {"success": True, "template_code": _GENERATED_CODE, "skipped": False}
        else:
            results[qid] = {"success": True, "skipped": True}
        if kwargs.get("progress_callback"):
            processed = len(results)
            total = len(self.get_question_ids())
            kwargs["progress_callback"](
                TemplateProgressEvent(
                    event="task_completed",
                    question_id=qid,
                    processed_count=processed,
                    total_count=total,
                    successful_count=processed,
                    failed_count=0,
                    percentage=processed / total * 100,
                    error=None,
                    template_code=None,
                    task_duration=None,
                )
            )
    return results
```

---

## When to Use Each Tool

Three approaches exist for creating answer templates. Choose based on how much control you need and how many templates you are writing.

| Tool | Best For | Speed | Control |
|------|----------|-------|---------|
| `generate_answer_template()` | Quick drafts, large benchmarks | Fast | Review needed |
| `AnswerBuilder` | Programmatic construction, uniform patterns | Medium | High |
| Hand-written templates | Complex `verify()` logic, custom validation | Slow | Full |

In practice, most large benchmarks use a combination: generate drafts for the majority, replace a few with `AnswerBuilder` or hand-written versions where the generated output needs adjustment.

---

## Generate a Template

`generate_answer_template()` takes a `Question` object and returns Python code for an `Answer(BaseAnswer)` class. The generation uses a two-phase approach: first, an LLM extracts the ground truth fields from the question and reference answer; second, it generates field descriptions that instruct the judge on what to extract from responses.

```python
from karenina.benchmark.authoring.answers.generator import generate_answer_template

# Create a Question object
question = Question(
    question="What is the putative target of venetoclax?",
    raw_answer="BCL2 (B-cell lymphoma 2), an anti-apoptotic protein",
)

# Generate the template (mocked for this tutorial)
with patch(
    "karenina.benchmark.authoring.answers.generator.generate_answer_template",
    _mock_generate_answer_template,
):
    code_string = _mock_generate_answer_template(question, model="claude-haiku-4-5", model_provider="anthropic")

print("Generated template code:")
print(code_string)
```

The two phases happen automatically inside `generate_answer_template()`. Phase 1 (ground truth extraction) determines what fields the template needs and what their correct values are. Phase 2 (field description generation) writes the instructional text that the judge LLM follows when parsing responses. You never call these phases separately; the function returns the final Python code string ready to use.

---

## Review Generated Code

Before attaching a generated template to your benchmark, review it for correctness. Check these elements:

1. **Field types**: Are `bool`, `str`, `float`, and `Literal` types appropriate for each field?
2. **Descriptions**: Do they give the judge enough context to parse correctly?
3. **`ground_truth` values**: Do they match the reference answer?
4. **`verify_with` primitives**: Is the right primitive selected for each field type?

```python
# The generated code above defines:
#   identifies_target: bool   -> VerifiedField with BooleanMatch, ground_truth=True
#   mentions_mechanism: bool  -> VerifiedField with BooleanMatch, ground_truth=True
#   No ground_truth() or verify() methods needed; VerifiedField handles both.

# Inspect the structure
lines = code_string.strip().split("\n")
print(f"Total lines: {len(lines)}")
print(f"Fields defined: identifies_target, mentions_mechanism")
print(f"Ground truth: identifies_target=True, mentions_mechanism=True")
print(f"Verification: BooleanMatch() for both fields")
```

For boolean fields, the most common issue is descriptions that are too vague. A description like "True if the response mentions BCL2" may return True for responses that mention BCL2 in a different context. The generated code above is more precise: it specifies "as the primary pharmacological target of venetoclax."

---

## Add to Benchmark

Once you are satisfied with the generated code, attach it to a benchmark question using `add_answer_template()`, which accepts a code string directly.

```python
benchmark = Benchmark.create(
    name="Drug Target Identification",
    description="Evaluate LLM accuracy on identifying drug targets",
    version="1.0.0",
)

q1_id = benchmark.add_question(
    question="What is the putative target of venetoclax?",
    raw_answer="BCL2 (B-cell lymphoma 2), an anti-apoptotic protein",
)

benchmark.add_answer_template(q1_id, code_string)
print(f"Template added: {benchmark.has_template(q1_id)}")
print(f"Progress: {benchmark.get_progress()}%")
```

---

## Edit Generated Templates

Generated templates are code strings, so you can modify them with standard string operations before attaching. This is useful when a field description needs refinement, a ground truth value needs correction, or you want to swap a primitive for a more appropriate one.

```python
# Refine the identifies_target description to be more precise,
# and add normalization to handle case variations
edited_code = code_string.replace(
    '        description="True if the response identifies BCL2 as the primary pharmacological target of venetoclax.",\n'
    "        ground_truth=True,\n"
    "        verify_with=BooleanMatch(),",
    "        description=(\n"
    '            "True if the response identifies BCL2 (including Bcl-2, BCL-2, or "\n'
    '            "B-cell lymphoma 2) as the direct pharmacological target of venetoclax. "\n'
    '            "False if BCL2 is mentioned only as a pathway member."\n'
    "        ),\n"
    "        ground_truth=True,\n"
    "        verify_with=BooleanMatch(),",
)

# Re-attach the edited version
benchmark.add_answer_template(q1_id, edited_code)
print(f"Template updated for: {q1_id[:50]}...")
print(f"Has template: {benchmark.has_template(q1_id)}")
```

For minor edits (tweaking a description, adjusting a ground truth value, swapping a primitive), string replacement works well. For larger structural changes (adding fields, introducing custom composition logic), consider writing the template by hand or using `AnswerBuilder` instead.

---

## Build with AnswerBuilder

`AnswerBuilder` provides a fluent interface for constructing templates without writing class code. Chain `add_attribute()` calls to define fields, then call `compile()` to produce an executable `Answer` class.

```python
builder = (
    AnswerBuilder()
    .add_attribute(
        "identifies_target",
        "bool",
        "True if the response identifies BCL2 as the primary target of venetoclax.",
        True,
    )
    .add_attribute(
        "mentions_mechanism",
        "bool",
        "True if the response describes venetoclax as a BH3 mimetic.",
        True,
    )
    .add_attribute(
        "specifies_indication",
        "bool",
        "True if the response mentions CLL or AML as an approved indication.",
        True,
    )
)

Answer = builder.compile()
print(f"Compiled class: {Answer.__name__}")
print(f"Builder state:\n{builder}")
```

`add_attribute()` accepts four arguments: the field name, the type as a string (`"bool"`, `"int"`, `"float"`, `"Literal['a', 'b']"`), a description for the judge, and the ground truth value. The builder validates names and types at build time, catching errors before `compile()`.

---

## Add Regex Patterns

`AnswerBuilder` also supports regex patterns via `add_regex()`. Regex checks run against the raw LLM response trace, independent of parsed fields. You can combine attributes and regex patterns in a single builder.

```python
builder_with_regex = (
    AnswerBuilder()
    .add_attribute(
        "identifies_drug_class",
        "bool",
        "True if the response identifies venetoclax as a BH3 mimetic or BCL2 inhibitor.",
        True,
    )
    .add_regex(
        "mentions_bcl2",
        r"(?i)\bBCL[-\s]?2\b",
        expected="BCL2",
        match_type="contains",
        description="Raw response contains a mention of BCL2.",
    )
    .add_regex(
        "has_citation",
        r"\[\d+\]",
        expected=1,
        match_type="count",
        description="Response includes at least one bracketed citation.",
    )
)

Answer = builder_with_regex.compile()
print(f"Compiled class: {Answer.__name__}")
print(f"Builder state:\n{builder_with_regex}")
```

The `match_type` parameter controls how regex results are evaluated:

| `match_type` | `expected` type | Behavior |
|-------------|-----------------|----------|
| `"exact"` | `str` | Exactly one match, equal to `expected` |
| `"contains"` | `str` | `expected` appears among matches |
| `"count"` | `int` | Number of matches equals `expected` |
| `"all"` | `list` | All items in `expected` found in matches |

---

## Batch Generation

For benchmarks with many questions, `generate_all_templates()` generates templates for every question that lacks one. The `only_missing=True` default makes this safe for incremental workflows: add new questions, then run generation to fill in only the gaps.

```python
# Add several questions without templates
q2_id = benchmark.add_question(
    question="What is the mechanism of action of metformin?",
    raw_answer="Activates AMP-activated protein kinase (AMPK) and reduces hepatic glucose production.",
)
q3_id = benchmark.add_question(
    question="What is the half-life of amoxicillin?",
    raw_answer="Approximately 1 hour.",
)
q4_id = benchmark.add_question(
    question="What is the antidote for acetaminophen overdose?",
    raw_answer="N-acetylcysteine (NAC).",
)

print(f"Questions: {benchmark.question_count}")
print(f"With templates: {len(benchmark.get_finished_templates())}")
```

```python
# Generate templates for all questions missing one
with patch.object(type(benchmark), "generate_all_templates", _mock_generate_all_templates):
    results = benchmark.generate_all_templates(
        model="claude-haiku-4-5",
        model_provider="anthropic",
        only_missing=True,
        progress_callback=lambda event: print(f"  {event.percentage:.0f}%: {event.event}"),
    )

generated = sum(1 for r in results.values() if r["success"] and not r.get("skipped"))
skipped = sum(1 for r in results.values() if r.get("skipped"))
print(f"\nGenerated: {generated}")
print(f"Skipped (already had template): {skipped}")
print(f"Progress: {benchmark.get_progress()}%")
```

On each invocation the `progress_callback` receives a single `TemplateProgressEvent` (from `karenina.benchmark.benchmark_helpers`). Read fields like `event.percentage` (0 to 100), `event.event` (the event type, such as `task_completed`), and the per-question counts to display progress in scripts or the GUI. For production use, pass a real model and provider. The function calls `generate_answer_template()` internally for each question.

---

## Workflow: Generate then Refine

The practical workflow for large benchmarks combines all three tools. Start with bulk generation, then review and replace where needed.

```
generate_all_templates(only_missing=True)
    |
    v
Review each generated template
    |
    v
Replace problematic ones:
    - AnswerBuilder for uniform patterns
    - Hand-written code for complex verify() logic
    |
    v
Save checkpoint
```

```python
# Step 1: Bulk generation already done above

# Step 2: Review a generated template
first_id = benchmark.get_question_ids()[0]
template = benchmark.get_template(first_id)
print(f"Reviewing template for: {first_id[:50]}...")
print(f"Template present: {template is not None}")

# Step 3: Replace one with an AnswerBuilder version
metformin_builder = (
    AnswerBuilder()
    .add_attribute("mentions_ampk", "bool", "Whether AMPK activation is mentioned", True)
    .add_attribute("mentions_hepatic", "bool", "Whether hepatic glucose reduction is mentioned", True)
)
benchmark.update_template(q2_id, metformin_builder.compile())
print(f"\nReplaced template for metformin question")

# Step 4: Save
tmpdir = tempfile.mkdtemp()
checkpoint_path = Path(tmpdir) / "drug_targets.jsonld"
benchmark.save(checkpoint_path)
print(f"Saved to: {checkpoint_path.name}")
print(f"Questions: {benchmark.question_count}")
print(f"Templates: {len(benchmark.get_finished_templates())}")
```

This pattern scales to hundreds of questions. The initial bulk generation handles the majority; targeted replacement keeps quality high for questions where the generated output falls short.

---

## Cleanup

```python
import shutil

shutil.rmtree(tmpdir, ignore_errors=True)
```

---

## Next Steps

- [Answer Templates](../../notebooks/core_concepts/answer-templates.ipynb): Deep dive into template concepts, field types, and `verify()` patterns
- [Factual QA Benchmark](../../notebooks/creating-benchmarks/factual-qa-benchmark.ipynb): Hand-written template patterns (boolean, string, numeric, regex)
- [Scaled Authoring](../../notebooks/creating-benchmarks/scaled-authoring.ipynb): Bulk ingestion, ADeLe classification, and few-shot examples
- [Running Verification](../running-verification/index.md): Execute the benchmark against LLMs
