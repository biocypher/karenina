---
jupyter:
  jupytext:
    formats: docs/workflows/running-verification//md,docs/notebooks/running-verification//ipynb
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

# Few-Shot Example Configuration

This tutorial shows how to configure few-shot examples for verification runs. Few-shot examples are prepended to the answering model's prompt, showing it the expected response format before it generates its own answer. The judge LLM never sees these examples. Use few-shot when your answering model produces poorly formatted responses, or when you want to demonstrate the expected output structure.

**What you'll learn:**

- Add few-shot examples when creating questions
- Configure `FewShotConfig` with source and pool modes (all, k-shot, custom)
- Override per-question with `QuestionFewShotConfig` and `inherit` mode
- Add global examples
- Select examples by index with `from_index_selections()`
- Resolve final examples with `resolve_examples_for_question()`
- Attach `FewShotConfig` to `VerificationConfig`

```python tags=["hide-cell"]
# Setup cell: creates a benchmark with questions that have few-shot examples.
# This cell is hidden in the rendered documentation.
# No LLM mocking needed: FewShotConfig is pure configuration and resolve is deterministic.
from karenina import Benchmark
from karenina.schemas.config.models import FewShotConfig, QuestionFewShotConfig, ModelConfig
from karenina.schemas.verification.config import VerificationConfig

_benchmark = Benchmark.create(name="Drug QA", description="Few-shot demo", version="1.0.0")

_examples_q1 = [
    {"question": "What is the target of imatinib?", "answer": "BCR-ABL tyrosine kinase"},
    {"question": "What is the target of trastuzumab?", "answer": "HER2 (ErbB2) receptor"},
    {"question": "What is the target of rituximab?", "answer": "CD20 protein on B cells"},
]
_examples_q2 = [
    {"question": "What is the half-life of aspirin?", "answer": "15 to 20 minutes"},
    {"question": "What is the half-life of metformin?", "answer": "Approximately 6.2 hours"},
    {"question": "What is the half-life of warfarin?", "answer": "20 to 60 hours"},
    {"question": "What is the half-life of amoxicillin?", "answer": "About 1 hour"},
]
_examples_q3 = [
    {"question": "Is metformin FDA-approved for type 2 diabetes?", "answer": "Yes"},
    {"question": "Is aspirin FDA-approved for migraine prevention?", "answer": "No"},
    {"question": "Is lisinopril FDA-approved for hypertension?", "answer": "Yes"},
]
_examples_q4 = [
    {"question": "What class is amoxicillin?", "answer": "Aminopenicillin (beta-lactam)"},
    {"question": "What class is ciprofloxacin?", "answer": "Fluoroquinolone"},
    {"question": "What class is azithromycin?", "answer": "Macrolide"},
    {"question": "What class is doxycycline?", "answer": "Tetracycline"},
]

_q1_id = _benchmark.add_question(
    question="What is the target of venetoclax?", raw_answer="BCL2",
    few_shot_examples=_examples_q1,
)
_q2_id = _benchmark.add_question(
    question="What is the half-life of caffeine in healthy adults?", raw_answer="5 hours",
    few_shot_examples=_examples_q2,
)
_q3_id = _benchmark.add_question(
    question="Is venetoclax FDA-approved for CLL?", raw_answer="Yes",
    few_shot_examples=_examples_q3,
)
_q4_id = _benchmark.add_question(
    question="What drug class does vancomycin belong to?", raw_answer="Glycopeptide",
    few_shot_examples=_examples_q4,
)

_qids = _benchmark.get_question_ids()
_examples_by_qid = {
    _q1_id: _examples_q1, _q2_id: _examples_q2,
    _q3_id: _examples_q3, _q4_id: _examples_q4,
}
```

---

## How Few-Shot Works

Few-shot examples are injected into the answering model's prompt only. The judge, rubric evaluators, and all other pipeline stages never see them:

```
Few-shot examples + Question --> Answering model prompt
Response only (no examples) --> Judge model prompt
```

This means few-shot examples influence how the model responds without biasing evaluation.

---

## Add Examples to Questions

Each example is a dict with `"question"` and `"answer"` keys, passed via `benchmark.add_question()`:

```python
benchmark = Benchmark.create(name="Drug QA", description="Few-shot demo", version="1.0.0")

examples = [
    {"question": "What is the target of imatinib?", "answer": "BCR-ABL tyrosine kinase"},
    {"question": "What is the target of trastuzumab?", "answer": "HER2 (ErbB2) receptor"},
    {"question": "What is the target of rituximab?", "answer": "CD20 protein on B cells"},
]
q_id = benchmark.add_question(
    question="What is the target of venetoclax?",
    raw_answer="BCL2",
    few_shot_examples=examples,
)

q_data = benchmark.get_question(q_id)
print(f"Question: {q_data['question'][:50]}")
print(f"Few-shot examples: {len(q_data['few_shot_examples'])}")
```

---

## Global Mode: All

The default mode uses every example attached to each question:

```python
from karenina.schemas.config.models import FewShotConfig

config_all = FewShotConfig(pool_mode="all")
resolved = config_all.resolve_examples_for_question(
    question_id=_q2_id, available_examples=_examples_by_qid[_q2_id],
)

print(f"Mode: {config_all.pool_mode}")
print(f"Available: {len(_examples_by_qid[_q2_id])}, Resolved: {len(resolved)}")
for ex in resolved:
    print(f"  Q: {ex['question'][:45]}  A: {ex['answer']}")
```

This works well when you have a small, curated set (2 to 5 examples per question).

---

## Global Mode: K-Shot

K-shot randomly samples *k* examples per question, using the question ID as seed for reproducibility:

```python
config_kshot = FewShotConfig(pool_mode="k-shot", pool_k=2)
resolved = config_kshot.resolve_examples_for_question(
    question_id=_q2_id, available_examples=_examples_by_qid[_q2_id],
)

print(f"Mode: {config_kshot.pool_mode}, k={config_kshot.pool_k}")
print(f"Available: {len(_examples_by_qid[_q2_id])}, Resolved: {len(resolved)}")
for ex in resolved:
    print(f"  Q: {ex['question'][:45]}  A: {ex['answer']}")
```

If a question has fewer examples than *k*, all examples are used (no error).

---

## Global Mode: Custom

Custom mode selects specific examples by index:

```python
config_custom = FewShotConfig.from_index_selections({
    _q1_id: [0, 2],       # First and third examples
    _q2_id: [1],           # Second example only
    _q4_id: [0, 1, 3],    # Skip third example
})
resolved_q1 = config_custom.resolve_examples_for_question(
    question_id=_q1_id, available_examples=_examples_by_qid[_q1_id],
)

print(f"Mode: {config_custom.pool_mode}")
print(f"Q1 resolved ({len(resolved_q1)} examples):")
for ex in resolved_q1:
    print(f"  Q: {ex['question'][:45]}  A: {ex['answer']}")
```

---

## Disabling Few-Shot

Setting `source="disabled"` turns off few-shot entirely, returning an empty list for all questions:

```python
config_none = FewShotConfig(source="disabled")
resolved = config_none.resolve_examples_for_question(
    question_id=_q1_id, available_examples=_examples_by_qid[_q1_id],
)

print(f"Source: {config_none.source}")
print(f"Resolved examples: {len(resolved)}")
```

Use `source="disabled"` to establish a zero-shot baseline for comparison.

---

## Per-Question Overrides

Each question can override the global mode via `QuestionFewShotConfig`. Questions without an explicit config inherit the global settings:

```python
from karenina.schemas.config.models import QuestionFewShotConfig

config_mixed = FewShotConfig(
    pool_mode="all",
    question_configs={
        _q1_id: QuestionFewShotConfig(mode="k-shot", k=1),   # Sample 1 example
        _q3_id: QuestionFewShotConfig(mode="custom", selected_examples=[]),  # No examples for q3
        # q2, q4: inherit global "all" mode
    },
)

for qid, label in [(_q1_id, "q1"), (_q2_id, "q2"), (_q3_id, "q3"), (_q4_id, "q4")]:
    effective = config_mixed.get_effective_config(qid)
    resolved = config_mixed.resolve_examples_for_question(
        question_id=qid, available_examples=_examples_by_qid[qid],
    )
    print(f"{label}: mode={effective.mode}, resolved={len(resolved)}")
```

The `inherit` mode (the default) delegates to the global mode and k value. Override it to customize specific questions while leaving the rest unchanged.

---

## Global Examples

Global examples are appended to every question's resolved examples, regardless of mode:

```python
config_global = FewShotConfig(
    pool_mode="k-shot", pool_k=1,
    global_examples=[
        {"question": "What is the capital of France?", "answer": "Paris"},
        {"question": "What is 2 + 2?", "answer": "4"},
    ],
)
resolved = config_global.resolve_examples_for_question(
    question_id=_q1_id, available_examples=_examples_by_qid[_q1_id],
)

print(f"Total resolved: {len(resolved)} (1 from k-shot + 2 global)")
for ex in resolved:
    print(f"  Q: {ex['question'][:45]}  A: {ex['answer']}")
```

Resolution order: stored examples first, then global examples.

---

## Bulk Selection

`from_index_selections()` builds custom selections for multiple questions. `k_shot_for_questions()` creates per-question k values in one call:

```python
config_bulk = FewShotConfig.from_index_selections({
    _q1_id: [0, 1], _q2_id: [0, 2, 3], _q3_id: [1], _q4_id: [0, 3],
})
print(f"Custom bulk (mode={config_bulk.pool_mode}):")
for qid, label in [(_q1_id, "q1"), (_q2_id, "q2"), (_q3_id, "q3"), (_q4_id, "q4")]:
    resolved = config_bulk.resolve_examples_for_question(
        question_id=qid, available_examples=_examples_by_qid[qid],
    )
    print(f"  {label}: {len(resolved)} examples selected")
```

```python
config_varied_k = FewShotConfig.k_shot_for_questions(
    question_k_mapping={_q1_id: 1, _q2_id: 3, _q4_id: 2},
    pool_k=2,
)
print(f"Varied k-shot (mode={config_varied_k.pool_mode}):")
for qid, label in [(_q1_id, "q1"), (_q2_id, "q2"), (_q3_id, "q3"), (_q4_id, "q4")]:
    effective = config_varied_k.get_effective_config(qid)
    print(f"  {label}: k={effective.k}")
```

---

## Resolve Examples

Call `resolve_examples_for_question()` to preview exactly what the answering model will see. This combines the global mode, per-question overrides, and external examples into a final list:

```python
config_preview = FewShotConfig(
    pool_mode="all",
    global_examples=[
        {"question": "Format example", "answer": "Short, precise answer"},
    ],
    question_configs={_q2_id: QuestionFewShotConfig(mode="k-shot", k=2)},
)

resolved_q1 = config_preview.resolve_examples_for_question(
    question_id=_q1_id, available_examples=_examples_by_qid[_q1_id],
)
print(f"Q1 (inherits 'all'): {len(resolved_q1)} examples (3 stored + 1 global)")

resolved_q2 = config_preview.resolve_examples_for_question(
    question_id=_q2_id, available_examples=_examples_by_qid[_q2_id],
)
print(f"Q2 (k-shot, k=2):   {len(resolved_q2)} examples (2 sampled + 1 global)")
```

Use this to verify your configuration before running a full verification pass.

---

## Attach to Verification

Pass the `FewShotConfig` to `VerificationConfig` via the `few_shot_config` field:

```python
from karenina.schemas.verification.config import VerificationConfig
from karenina.schemas.config.models import ModelConfig

few_shot = FewShotConfig(
    pool_mode="k-shot",
    pool_k=2,
    question_configs={_q3_id: QuestionFewShotConfig(mode="custom", selected_examples=[])},
)

config = VerificationConfig(
    answering_models=[
        ModelConfig(id="haiku", model_name="claude-haiku-4-5",
                    model_provider="anthropic", interface="langchain")
    ],
    parsing_models=[
        ModelConfig(id="haiku-parser", model_name="claude-haiku-4-5",
                    model_provider="anthropic", interface="langchain",
                    temperature=0.0)
    ],
    few_shot_config=few_shot,
)

print(f"Source:           {config.few_shot_config.source}")
print(f"Pool mode:        {config.few_shot_config.pool_mode}")
print(f"Pool k:           {config.few_shot_config.pool_k}")
print(f"Per-question:     {len(config.few_shot_config.question_configs)} overrides")
```

When `few_shot_config` is `None` (the default) or `source="disabled"`, no examples are prepended.

---

## Tuning Strategy

- Start with `source="disabled"` to establish a zero-shot baseline
- If the answering model produces poorly formatted responses, add 2 to 3 examples per question
- Use `resolve_examples_for_question()` to preview before running full verification
- Increase *k* incrementally; more examples increase prompt cost without guaranteed improvement
- Use per-question overrides for questions where the global strategy underperforms
- Compare zero-shot and few-shot results side by side to confirm examples actually help

```python tags=["hide-cell"]
# No mocks to restore; FewShotConfig is pure configuration.
print("Done")
```

---

## Next Steps

- [Few-Shot Concepts](../../core_concepts/few-shot.md): Detailed explanation of modes, resolution, and edge cases
- [Prompt Assembly](../../core_concepts/prompt-assembly.md): How few-shot examples are injected into the answering prompt
- [VerificationConfig Reference](../../reference/configuration/verification-config.md): All configuration fields
- [Basic Verification](../../notebooks/running-verification/basic-verification.ipynb): Simplest verification workflow
- [Full Evaluation](../../notebooks/running-verification/full-evaluation.ipynb): Template and rubric evaluation with quality checks
