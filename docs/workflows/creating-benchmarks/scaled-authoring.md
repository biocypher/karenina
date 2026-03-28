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

# Scaled Authoring

When building large benchmarks with dozens or hundreds of questions, manual question entry and template writing don't scale. This scenario demonstrates the power-user workflow: bulk question ingestion, automated template generation, programmatic template construction, ADeLe question classification, and few-shot example injection.

**What you'll learn:**

- Bulk question ingestion from files
- Automated template generation with `generate_all_templates()`
- Programmatic templates with `AnswerBuilder`
- ADeLe question classification
- Few-shot examples (per-question and via `FewShotConfig`)
- Progress callbacks

```python tags=["hide-cell"]
# Setup cell: hidden in rendered documentation.
# Mocks LLM-dependent operations so examples execute without API keys.
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Mock for generate_all_templates — returns pre-built template results
_mock_template_results = {}

def _mock_generate_all_templates(self, **kwargs):
    """Return mock template generation results."""
    results = {}
    for qid in self.get_question_ids():
        if not self.get_template(qid) or kwargs.get("force_regenerate"):
            # Create a simple VerifiedField template for each question
            template_code = '''from karenina.schemas.entities import BaseAnswer, BooleanMatch, VerifiedField


class Answer(BaseAnswer):
    is_correct: bool = VerifiedField(
        description="True if the response contains the expected answer.",
        ground_truth=True,
        verify_with=BooleanMatch(),
    )
'''
            self.add_answer_template(qid, template_code)
            results[qid] = {"success": True, "template_code": template_code, "skipped": False}
        else:
            results[qid] = {"success": True, "skipped": True}
        if kwargs.get("progress_callback"):
            kwargs["progress_callback"](
                len(results) / len(self.get_question_ids()) * 100,
                f"Generated {len(results)}/{len(self.get_question_ids())}",
            )
    return results
```

---

## Create the Benchmark

```python
from karenina import Benchmark

benchmark = Benchmark.create(
    name="Pharmacology Knowledge Base",
    description="Large-scale evaluation of LLM pharmacology knowledge",
    version="1.0.0",
)
print(f"Created: {benchmark.name}")
```

---

## Bulk Question Ingestion

For large benchmarks, questions typically come from spreadsheets or CSV files rather than manual entry. The `extract_questions_from_file()` function reads tabular data and returns a list of `Question` objects with all metadata (keywords, author, custom fields) populated directly.

```python
from karenina.benchmark.authoring.questions import extract_questions_from_file
```

The function supports Excel, CSV, and TSV formats:

| Format | Extension | Notes |
|--------|-----------|-------|
| Excel | `.xlsx`, `.xls` | Supports multiple sheets |
| CSV | `.csv` | Comma-separated |
| TSV | `.tsv` | Tab-separated |

```python
# In a real workflow, you'd point to an actual file:
#
# questions = extract_questions_from_file(
#     file_path="questions.xlsx",
#     question_column="Question",
#     answer_column="Expected Answer",
#     author_name_column="Author",
#     keywords_columns=[
#         {"column": "Tags", "separator": ","},
#     ],
#     custom_metadata_columns=["Complexity"],
# )
#
# benchmark.add_questions(questions)
```

For this tutorial, we add questions manually:

```python
questions = [
    ("What is the mechanism of action of metformin?", "Metformin activates AMP-activated protein kinase (AMPK) and reduces hepatic glucose production."),
    ("What is the half-life of amoxicillin?", "Approximately 1 hour."),
    ("Name three common side effects of statins.", "Muscle pain, liver enzyme elevation, and digestive problems."),
    ("What is the antidote for acetaminophen overdose?", "N-acetylcysteine (NAC)."),
    ("How does warfarin prevent blood clots?", "Warfarin inhibits vitamin K epoxide reductase, blocking synthesis of clotting factors II, VII, IX, and X."),
    ("What is the therapeutic index of lithium?", "Narrow — therapeutic range is 0.6-1.2 mEq/L, with toxicity above 1.5 mEq/L."),
    ("What class of drug is omeprazole?", "Proton pump inhibitor (PPI)."),
    ("What is the primary indication for naloxone?", "Reversal of opioid overdose."),
]

for q_text, answer in questions:
    benchmark.add_question(question=q_text, raw_answer=answer)

print(f"Added {benchmark.question_count} questions")
print(f"With templates: {len(benchmark.get_finished_templates())}")
```

---

## Automated Template Generation

With many questions added, `generate_all_templates()` uses an LLM to produce `Answer` classes for every question that lacks a template. The `only_missing=True` default skips questions that already have templates, making this safe for incremental workflows — add new questions, then run generation to fill in only the new ones.

```python
# Generate templates for all questions that don't have one
with patch.object(type(benchmark), "generate_all_templates", _mock_generate_all_templates):
    results = benchmark.generate_all_templates(
        model="claude-haiku-4-5",
        model_provider="anthropic",
        only_missing=True,
        progress_callback=lambda pct, msg: print(f"  {pct:.0f}%: {msg}"),
    )

# Summarize results
generated = sum(1 for r in results.values() if r["success"] and not r.get("skipped"))
skipped = sum(1 for r in results.values() if r.get("skipped"))
failed = sum(1 for r in results.values() if not r["success"])

print(f"\nGenerated: {generated}")
print(f"Skipped:   {skipped}")
print(f"Failed:    {failed}")
print(f"Progress:  {benchmark.get_progress()}%")
```

Review a generated template to check the output:

```python
first_id = benchmark.get_question_ids()[0]
template = benchmark.get_template(first_id)
print(template)
```

---

## Programmatic Templates with AnswerBuilder

For questions where you want precise control without writing template class code by hand, `AnswerBuilder` provides a fluent interface. It builds `Answer` classes using `VerifiedField` with appropriate verification primitives, then compiles them into executable classes. Each attribute you add is mapped to a `VerifiedField` with a matching primitive (`BooleanMatch` for booleans, `ExactMatch` for strings, `NumericTolerance` for floats, and so on).

```python
from karenina.benchmark.authoring.answers.builder import AnswerBuilder

builder = (
    AnswerBuilder()
    .add_attribute("mentions_ampk", "bool", "Whether AMPK activation is mentioned", True)
    .add_attribute("mentions_hepatic", "bool", "Whether hepatic glucose reduction is mentioned", True)
    .add_regex("has_mechanism_keyword", r"\b(activates|inhibits|blocks)\b", expected=True, match_type="contains")
)

Answer = builder.compile()

# Replace the auto-generated template for the metformin question
metformin_id = benchmark.get_question_ids()[0]  # First question
benchmark.update_template(metformin_id, Answer)
print(f"Updated template for: {metformin_id[:50]}...")
```

The compiled class uses `VerifiedField` internally. For example, `add_attribute("mentions_ampk", "bool", ..., True)` produces a field equivalent to:

```python
mentions_ampk: bool = VerifiedField(
    description="Whether AMPK activation is mentioned",
    ground_truth=True,
    verify_with=BooleanMatch(),
)
```

`BaseAnswer` auto-generates `ground_truth()`, `verify()`, and `verify_granular()` from the `VerifiedField` metadata, so the compiled class needs no hand-written verification methods.

When to use each approach:

- **AnswerBuilder**: Quick boolean/regex templates built from data, no string manipulation needed.
- **Class definitions or code strings**: Complex verification logic, custom normalization, multi-step checks that need full Python expressiveness.

---

## ADeLe Question Classification

ADeLe (Annotated Demand Levels) classifies questions across 18 cognitive complexity dimensions, producing scores from 0 (none) to 5 (very high) on each dimension. This helps you understand what your benchmark measures and filter questions by difficulty.

```python
from karenina.integrations.adele import QuestionClassifier
```

Classification requires LLM access. In a real workflow with API keys configured:

```python
# In a real workflow:
#
# classifier = QuestionClassifier(
#     model_name="claude-haiku-4-5",
#     provider="anthropic",
# )
#
# question_pairs = [
#     (qid, benchmark.get_question(qid)["question"])
#     for qid in benchmark.get_question_ids()
# ]
#
# results = classifier.classify_batch(
#     questions=question_pairs,
#     on_progress=lambda done, total: print(f"{done}/{total}"),
# )
#
# for q_id, result in results.items():
#     print(f"{q_id[:30]}... volume={result.scores['volume']}, "
#           f"reasoning={result.scores['logical_reasoning_logic']}")
```

ADeLe provides:

- **18 cognitive complexity dimensions** — volume, reasoning depth, domain specificity, and more
- **Ordinal scores** from 0 (none) to 5 (very high) on each dimension
- **Filtering support** — select question subsets by complexity for targeted evaluation

See [ADeLe Concepts](../../notebooks/core_concepts/adele.ipynb) for the full dimension list and scoring reference.

---

## Adding Few-Shot Examples

Few-shot examples guide the answering model toward the expected response format by prepending question-answer pairs to the prompt during verification.

### Per-Question Examples

Attach examples directly to individual questions when adding them:

```python
# Add few-shot examples to specific questions
benchmark.add_question(
    question="What is the mechanism of action of clopidogrel?",
    raw_answer="Clopidogrel irreversibly inhibits the P2Y12 ADP receptor on platelets.",
    few_shot_examples=[
        {"question": "How does warfarin prevent blood clots?", "answer": "Inhibits vitamin K epoxide reductase"},
        {"question": "How does heparin prevent blood clots?", "answer": "Activates antithrombin III"},
    ],
)
print(f"Added question with {2} few-shot examples")
```

### FewShotConfig for Verification

`FewShotConfig` controls *which* examples are used during verification. Global examples are appended to every question. The `pool_k` parameter limits how many per-question examples are included.

```python
from karenina.schemas import FewShotConfig, QuestionFewShotConfig

few_shot_config = FewShotConfig(
    pool_mode="k-shot",
    pool_k=2,
    global_examples=[
        {"question": "What class of drug is aspirin?", "answer": "NSAID"},
    ],
)

print(f"Mode: {few_shot_config.pool_mode}")
print(f"K: {few_shot_config.pool_k}")
print(f"Global examples: {len(few_shot_config.global_examples)}")
```

See [Few-Shot Configuration](../../notebooks/core_concepts/few-shot.ipynb) for the full configuration reference.

---

## Save and Reload

```python
tmpdir = tempfile.mkdtemp()
checkpoint_path = Path(tmpdir) / "pharmacology_knowledge_base.jsonld"
benchmark.save(checkpoint_path)

loaded = Benchmark.load(checkpoint_path)
print(f"Questions: {loaded.question_count}")
print(f"Templates: {len(loaded.get_finished_templates())}")
print(f"Progress:  {loaded.get_progress()}%")
```

---

## Cleanup

```python
import shutil
shutil.rmtree(tmpdir, ignore_errors=True)
```

---

## Workflow Summary

```
Extract from file (or add manually)
    │
    ▼
generate_all_templates(only_missing=True)
    │
    ▼
Review → Replace with AnswerBuilder or custom code where needed
    │
    ▼
Classify with ADeLe [optional]
    │
    ▼
Add few-shot examples [optional]
    │
    ▼
Save checkpoint
```

---

## Next Steps

- [Factual QA Benchmark](factual-qa-benchmark.ipynb) — Detailed template patterns (boolean, numeric, regex)
- [Full Evaluation Benchmark](full-evaluation-benchmark.ipynb) — Add rubric traits alongside templates
- [Quality Assessment](quality-assessment-benchmark.ipynb) — Rubric-only evaluation
- [Answer Templates](../core_concepts/answer-templates.ipynb) — Template concepts
- [ADeLe Concepts](../../notebooks/core_concepts/adele.ipynb) — Full ADeLe dimension reference
- [Few-Shot Configuration](../../notebooks/core_concepts/few-shot.ipynb) — Advanced few-shot options
