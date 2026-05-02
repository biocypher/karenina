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

# ADeLe Question Classification

ADeLe (Annotated Demand Levels) classifies benchmark questions across 18 cognitive dimensions, from domain knowledge required to reasoning complexity. Each dimension produces an ordinal score from 0 (none) to 5 (very high). Use classification results to understand your benchmark's difficulty profile, filter questions by complexity, or create rubrics based on ADeLe traits.

This tutorial is useful when you have a set of questions and want to characterize what your benchmark measures before running verification. Classification can also guide question selection: keep only high-reasoning questions for a challenging benchmark, or balance difficulty across categories.

**What you'll learn:**

- Create a `QuestionClassifier` with model configuration
- Classify a single question with `classify_single()`
- Inspect scores, labels, and summary format
- Classify a benchmark in batch with `classify_batch()`
- Store classifications in `custom_metadata` via `to_checkpoint_metadata()`
- Reload classifications with `from_checkpoint_metadata()`
- Create a rubric from ADeLe traits with `create_adele_rubric()`
- Filter questions by cognitive complexity

```python tags=["hide-cell"]
# Setup cell: hidden in rendered documentation.
# Mocks LLM-dependent operations so examples execute without API keys.
from unittest.mock import patch

from karenina.integrations.adele.schemas import QuestionClassificationResult

# The actual 18 ADeLe trait names
_DIMS = [
    "attention_and_scan", "atypicality", "comprehension_complexity",
    "comprehension_evaluation", "conceptualization_and_learning",
    "knowledge_applied_sciences", "knowledge_cultural", "knowledge_formal_sciences",
    "knowledge_natural_sciences", "knowledge_social_sciences",
    "metacognition_relevance", "metacognition_task_planning", "metacognition_uncertainty",
    "mind_modelling", "logical_reasoning_logic", "logical_reasoning_quantitative",
    "spatial_physical_understanding", "volume",
]
_LEVEL_NAMES = ["very_low", "very_low", "low", "moderate", "high", "very_high"]

def _make_result(qid, qtext, raw_scores, ts_suffix="00Z", tokens=700):
    scores = dict(zip(_DIMS, raw_scores))
    labels = {d: _LEVEL_NAMES[min(s, 5)] for d, s in scores.items()}
    return QuestionClassificationResult(
        question_id=qid, question_text=qtext, scores=scores, labels=labels,
        model="claude-haiku-4-5", classified_at=f"2025-02-15T10:30:{ts_suffix}",
        usage_metadata={"input_tokens": tokens - 200, "output_tokens": 200, "total_tokens": tokens},
    )

_mock_single_result = _make_result(
    "q1", "What is the approved pharmacological target of venetoclax?",
    #  AS  AT  CEc CEe CL  KNa KNc KNf KNn KNs MCr MCt MCu MS  QLl QLq SNs VO
    [  2,  1,  2,  3,  1,  4,  0,  1,  4,  0,  2,  1,  1,  0,  2,  0,  0,  3],
)
_mock_partial_result = QuestionClassificationResult(
    question_id="q1",
    question_text="What is the approved pharmacological target of venetoclax?",
    scores={"knowledge_natural_sciences": 4, "logical_reasoning_logic": 2},
    labels={"knowledge_natural_sciences": "high", "logical_reasoning_logic": "low"},
    model="claude-haiku-4-5", classified_at="2025-02-15T10:30:05Z",
    usage_metadata={"input_tokens": 180, "output_tokens": 60, "total_tokens": 240},
)
_mock_batch_results = {
    "q1": _mock_single_result,
    "q2": _make_result(
        "q2", "Compare the mechanisms of action of imatinib and dasatinib.",
        [3, 2, 3, 4, 2, 5, 0, 1, 5, 0, 3, 3, 2, 1, 3, 0, 0, 4], ts_suffix="10Z", tokens=730,
    ),
    "q3": _make_result(
        "q3", "What is the half-life of amoxicillin?",
        [1, 0, 1, 2, 0, 2, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 2], ts_suffix="15Z", tokens=670,
    ),
}

def _mock_classify_single(self, question_text, trait_names=None, question_id=None):
    if trait_names and len(trait_names) < 18:
        return _mock_partial_result
    return _mock_single_result

def _mock_classify_batch(self, questions, trait_names=None, on_progress=None):
    results = {}
    for i, (qid, qtext) in enumerate(questions):
        results[qid] = _mock_batch_results.get(qid, _mock_single_result)
        if on_progress:
            on_progress(i + 1, len(questions))
    return results
```

---

## The 18 Dimensions

ADeLe organizes its dimensions into four categories. Each dimension uses a 0 to 5 ordinal scale, from "very low" to "very high."

| Category | Dimension | What It Measures |
|----------|-----------|-----------------|
| **Attention** | `attention_and_scan` | Visual scanning or search required |
| | `atypicality` | Uncommon or surprising elements |
| **Comprehension** | `comprehension_complexity` | Structural complexity of the question |
| | `comprehension_evaluation` | Evaluation or judgment required |
| | `conceptualization_and_learning` | Concept formation or learning demand |
| **Knowledge** | `knowledge_applied_sciences` | Applied science knowledge needed |
| | `knowledge_cultural` | Cultural or humanities knowledge |
| | `knowledge_formal_sciences` | Math or formal science knowledge |
| | `knowledge_natural_sciences` | Natural science domain expertise |
| | `knowledge_social_sciences` | Social science knowledge needed |
| **Metacognition** | `metacognition_relevance` | Relevance judgment required |
| | `metacognition_task_planning` | Planning or strategy needed |
| | `metacognition_uncertainty` | Uncertainty handling required |
| **Other** | `mind_modelling` | Theory of mind or perspective-taking |
| | `logical_reasoning_logic` | Formal logical deduction |
| | `logical_reasoning_quantitative` | Quantitative reasoning |
| | `spatial_physical_understanding` | Spatial or physical reasoning |
| | `volume` | Amount of content to process |

---

## Create a Classifier

The `QuestionClassifier` wraps an LLM to evaluate questions against ADeLe dimensions. By default it uses `claude-haiku-4-5` for efficiency, since classification runs one LLM call per question in batch mode.

```python
from karenina.integrations.adele import QuestionClassifier

classifier = QuestionClassifier(
    model_name="claude-haiku-4-5",
    provider="anthropic",
)
print(f"Classifier ready: model={classifier._model_name}, mode={classifier._trait_eval_mode}")
```

---

## Classify a Single Question

`classify_single()` evaluates a question against all 18 dimensions and returns a `QuestionClassificationResult` with scores, labels, and usage metadata.

```python
with patch.object(QuestionClassifier, "classify_single", _mock_classify_single):
    result = classifier.classify_single(
        question_text="What is the approved pharmacological target of venetoclax?",
        question_id="q1",
    )

print(f"Question: {result.question_text}")
print(f"Model:    {result.model}")
print(f"Dimensions classified: {len(result.scores)}")
```

The `scores` dict maps each dimension to an integer (0 to 5), while `labels` maps to the corresponding level name:

```python
# Inspect a few scores and labels
for dim in ["knowledge_natural_sciences", "logical_reasoning_logic", "volume"]:
    print(f"  {dim}: score={result.scores[dim]}, label={result.labels[dim]}")
```

The `get_summary()` method produces a compact `"label (score)"` format for each dimension:

```python
summary = result.get_summary()
for dim, value in list(summary.items())[:5]:
    print(f"  {dim}: {value}")
```

---

## Select Specific Traits

When you only need a subset of dimensions, pass `trait_names` to skip the rest. This reduces token usage and latency.

```python
with patch.object(QuestionClassifier, "classify_single", _mock_classify_single):
    partial = classifier.classify_single(
        question_text="What is the approved pharmacological target of venetoclax?",
        question_id="q1",
        trait_names=["knowledge_natural_sciences", "logical_reasoning_logic"],
    )

print(f"Dimensions classified: {len(partial.scores)}")
for dim, score in partial.scores.items():
    print(f"  {dim}: {partial.labels[dim]} ({score})")
```

---

## Batch Classification

For a full benchmark, `classify_batch()` accepts a list of `(question_id, question_text)` tuples and returns a dict mapping each ID to its classification result. The optional `on_progress` callback receives `(completed, total)` counts.

```python
questions = [
    ("q1", "What is the approved pharmacological target of venetoclax?"),
    ("q2", "Compare the mechanisms of action of imatinib and dasatinib."),
    ("q3", "What is the half-life of amoxicillin?"),
]

with patch.object(QuestionClassifier, "classify_batch", _mock_classify_batch):
    batch_results = classifier.classify_batch(
        questions=questions,
        on_progress=lambda done, total: print(f"  Classified {done}/{total}"),
    )

print(f"\nClassified {len(batch_results)} questions")
for qid, res in batch_results.items():
    print(f"  {qid}: knowledge_natural_sciences={res.scores['knowledge_natural_sciences']}, "
          f"logical_reasoning_logic={res.scores['logical_reasoning_logic']}")
```

---

## Store in Checkpoint Metadata

`to_checkpoint_metadata()` converts a classification result into a dict suitable for storing in a question's `custom_metadata` field. This preserves classifications across save/load cycles.

```python
metadata = result.to_checkpoint_metadata()
print("Checkpoint metadata keys:", list(metadata.keys()))
print("Inner keys:", list(metadata["adele_classification"].keys()))
```

In a real workflow, you would update the question's metadata in the benchmark:

```python
# In a real workflow with a benchmark loaded:
#
# for qid, classification in batch_results.items():
#     question = benchmark.get_question(qid)
#     existing_meta = question.get("custom_metadata", {})
#     existing_meta.update(classification.to_checkpoint_metadata())
#     benchmark.update_question_metadata(qid, custom_metadata=existing_meta)
#
# benchmark.save("checkpoint.jsonld")

print("Classifications stored in custom_metadata under 'adele_classification' key")
```

---

## Reload Classifications

`from_checkpoint_metadata()` reconstructs a `QuestionClassificationResult` from stored metadata, completing the round-trip.

```python
from karenina.integrations.adele.schemas import QuestionClassificationResult

# Simulate loading from checkpoint
stored_metadata = result.to_checkpoint_metadata()

reloaded = QuestionClassificationResult.from_checkpoint_metadata(
    metadata=stored_metadata,
    question_id="q1",
    question_text="What is the approved pharmacological target of venetoclax?",
)

print(f"Reloaded: {reloaded.question_id}")
print(f"Scores match: {reloaded.scores == result.scores}")
print(f"Labels match: {reloaded.labels == result.labels}")
print(f"Model: {reloaded.model}")
```

If the metadata does not contain an `adele_classification` key, `from_checkpoint_metadata()` returns `None`:

```python
empty = QuestionClassificationResult.from_checkpoint_metadata({})
print(f"Missing classification returns: {empty}")
```

---

## Filter by Complexity

Use scores to select question subsets by cognitive profile. This is useful for building targeted benchmarks or stratifying evaluation by difficulty.

```python
# Find questions requiring significant logical reasoning (logical_reasoning_logic >= 3)
high_reasoning = {
    qid: res for qid, res in batch_results.items()
    if res.scores.get("logical_reasoning_logic", 0) >= 3
}
print(f"High logical reasoning questions: {len(high_reasoning)}")
for qid in high_reasoning:
    score = high_reasoning[qid].scores["logical_reasoning_logic"]
    print(f"  {qid}: logical_reasoning_logic={score}")

# Find low-knowledge questions (knowledge_natural_sciences <= 2)
low_knowledge = {
    qid: res for qid, res in batch_results.items()
    if res.scores.get("knowledge_natural_sciences", 0) <= 2
}
print(f"\nLow natural science knowledge questions: {len(low_knowledge)}")
for qid in low_knowledge:
    score = low_knowledge[qid].scores["knowledge_natural_sciences"]
    print(f"  {qid}: knowledge_natural_sciences={score}")
```

You can also combine filters for more specific selection:

```python
# Questions requiring high domain knowledge AND logical reasoning
complex_questions = {
    qid: res for qid, res in batch_results.items()
    if res.scores.get("knowledge_natural_sciences", 0) >= 4
    and res.scores.get("logical_reasoning_logic", 0) >= 3
}
print(f"High knowledge + logical reasoning: {len(complex_questions)} questions")
```

---

## Create Rubric from ADeLe Traits

ADeLe traits can be used directly as rubric traits for verification. `create_adele_rubric()` builds a `Rubric` containing the specified ADeLe dimensions as `LLMRubricTrait` objects with `kind="literal"`.

```python
from karenina.integrations.adele import create_adele_rubric, ADELE_TRAIT_NAMES

# Create a rubric with selected traits
rubric = create_adele_rubric(
    trait_names=["knowledge_natural_sciences", "logical_reasoning_logic"]
)

print(f"Rubric traits: {len(rubric.llm_traits)}")
for trait in rubric.llm_traits:
    print(f"  {trait.name}: kind={trait.kind}, classes={len(trait.classes)}")
```

To use all 18 traits, pass `None` (or omit `trait_names`):

```python
full_rubric = create_adele_rubric()
print(f"Full rubric: {len(full_rubric.llm_traits)} traits")
print(f"Available trait names ({len(ADELE_TRAIT_NAMES)}):")
for name in ADELE_TRAIT_NAMES[:6]:
    print(f"  - {name}")
print(f"  ... and {len(ADELE_TRAIT_NAMES) - 6} more")
```

---

## Cleanup

```python
# No temporary files were created in this tutorial.
print("Done.")
```

---

## Next Steps

- [ADeLe Concept Page](../../notebooks/core_concepts/adele.ipynb): Full dimension reference and scoring details
- [Rubrics](../../core_concepts/rubrics/index.md): Deep dive into rubric concepts and trait types
- [Scaled Authoring](../../notebooks/creating-benchmarks/scaled-authoring.ipynb): Bulk workflows, template generation, and classification in context
- [Running Verification](../running-verification/index.md): Execute benchmarks with ADeLe rubrics
