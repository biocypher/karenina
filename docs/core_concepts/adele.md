---
jupyter:
  jupytext:
    formats: docs/core_concepts//md,docs/notebooks/core_concepts//ipynb
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

# ADeLe: Question Classification

**ADeLe** (Annotated Demand Levels) is a framework for classifying *questions* along 18 cognitive dimensions. It characterizes what makes a question hard, not how well a model answered it.

[Answer templates](../answer-templates/) and [rubrics](../../../core_concepts/rubrics/) evaluate responses. ADeLe evaluates the questions themselves: how much reasoning, knowledge, attention, or metacognition a question demands. The output is a difficulty profile (18 scores on a 0-5 ordinal scale) that you can use to filter benchmarks, stratify analyses, or understand your question set before any model runs.

!!! info "Attribution"
    ADeLe was developed by Zhou, Pacchiardi, Martinez-Plumed, Collins et al. at the Leverhulme Centre for the Future of Intelligence (Cambridge) and the Center for Information Technology Policy (Princeton). The 18 demand-level rubrics and ordinal scale used in this integration are derived from their work.

    **Paper**: Zhou, L., Pacchiardi, L., Martinez-Plumed, F., Collins, K.M., et al. (2025). *General Scales Unlock AI Evaluation with Explanatory and Predictive Power*. [arXiv:2503.06378](https://arxiv.org/abs/2503.06378)

    **Project**: [https://kinds-of-intelligence-cfi.github.io/ADELE/](https://kinds-of-intelligence-cfi.github.io/ADELE/)

```python tags=["hide-cell"]
# Mock cell: ensures examples execute without live API keys.
# This cell is hidden in rendered documentation.
from karenina.integrations.adele import (
    QuestionClassifier,
    QuestionClassificationResult,
    ADELE_TRAIT_NAMES,
)

_SCORE_TO_LABEL = {
    0: "none", 1: "very_low", 2: "low",
    3: "intermediate", 4: "high", 5: "very_high",
}

_MOCK_PROFILES = {
    "What is the chemical formula for water?": {
        "knowledge_natural_sciences": 2,
        "comprehension_complexity": 1,
    },
    "What is 2 + 2?": {
        "knowledge_formal_sciences": 1,
        "logical_reasoning_quantitative": 1,
    },
    "Explain the significance of BCL2 in cancer therapy.": {
        "comprehension_complexity": 3,
        "comprehension_evaluation": 3,
        "knowledge_applied_sciences": 4,
        "knowledge_natural_sciences": 3,
        "metacognition_relevance": 2,
        "volume": 2,
    },
    "Write a haiku about recursion.": {
        "comprehension_evaluation": 3,
        "knowledge_formal_sciences": 2,
        "atypicality": 3,
    },
    "Prove that the square root of 2 is irrational.": {
        "logical_reasoning_logic": 4,
        "knowledge_formal_sciences": 3,
        "atypicality": 1,
        "volume": 3,
        "metacognition_task_planning": 3,
    },
}


def _build_mock_result(question_text, trait_names=None, question_id=None):
    names = trait_names or ADELE_TRAIT_NAMES
    profile = _MOCK_PROFILES.get(question_text, {})
    scores = {n: profile.get(n, 0) for n in names}
    labels = {n: _SCORE_TO_LABEL[s] for n, s in scores.items()}
    return QuestionClassificationResult(
        question_id=question_id,
        question_text=question_text,
        scores=scores,
        labels=labels,
        model="claude-haiku-4-5",
        classified_at="2025-06-15T10:30:00",
    )


def _mock_classify_single(self, question_text, trait_names=None, question_id=None):
    return _build_mock_result(question_text, trait_names, question_id)


def _mock_classify_batch(self, questions, trait_names=None, on_progress=None):
    results = {}
    for i, (qid, text) in enumerate(questions):
        results[qid] = _build_mock_result(text, trait_names, qid)
        if on_progress:
            on_progress(i + 1, len(questions))
    return results


QuestionClassifier.classify_single = _mock_classify_single
QuestionClassifier.classify_batch = _mock_classify_batch
```

## 1. What It Is

ADeLe classifies a question into 18 independent dimensions, each scored on a 6-level ordinal scale (0 = none, 5 = expert-level). The result is a structured profile of the cognitive demands the question places on a solver.

**The abstraction boundary**: ADeLe is a question-level annotation tool. It does not participate in the [verification pipeline](../verification-pipeline/), does not evaluate responses, and does not affect how templates or rubrics run. Classifications are metadata: they describe the question, not the evaluation.

| Concept | What it evaluates | When it runs |
|---|---|---|
| [Answer template](../answer-templates/) | Correctness of the response (parsed fields vs ground truth) | During verification pipeline (stages 7-8) |
| [Rubric](../../../core_concepts/rubrics/) | Quality of the response (observable traits of raw text) | During verification pipeline (stage 11) |
| **ADeLe** | Cognitive demands of the question itself | Before or independent of verification |

## 2. Why It Exists

Benchmark questions vary in difficulty, but "difficulty" is not one thing. A question can be hard because it requires deep domain knowledge, or because it demands multi-step reasoning, or because the relevant information is buried among distractors. ADeLe makes these axes of difficulty explicit and measurable.

This matters for benchmark design and analysis:

- **Stratified analysis**: break down verification results by reasoning depth, domain specificity, or novelty to see where a model struggles
- **Targeted filtering**: run expensive verification only on questions above a complexity threshold
- **Benchmark characterization**: understand what your question set tests before spending compute on evaluation
- **Balanced authoring**: identify gaps in your question coverage (too many rote recall questions, too few requiring metacognition)

ADeLe is LLM-based: the `QuestionClassifier` sends each question to an LLM along with the 18 dimension rubrics, and the LLM returns a classification per dimension. This means results are not perfectly deterministic, but the structured rubrics and low temperature (default `0.0`) provide high consistency.

## 3. The 18 Dimensions

ADeLe defines 18 dimensions organized into six categories. Each dimension measures a distinct cognitive demand on a shared 6-level ordinal scale.

### The ordinal scale

Every dimension uses the same scale:

| Score | Label | Meaning |
|---|---|---|
| **0** | `none` | No or minimal requirement |
| **1** | `very_low` | Very low requirement |
| **2** | `low` | Low requirement |
| **3** | `intermediate` | Moderate requirement |
| **4** | `high` | High requirement |
| **5** | `very_high` | Expert-level requirement |

A score of **-1** indicates a classification error (the LLM did not return a valid class for that dimension).

### Dimension reference

| Category | Code | Trait Name | What It Measures |
|---|---|---|---|
| **Attention** | AS | `attention_and_scan` | Level of attention needed to locate specific elements. Ranges from obvious targets to sustained tracking among distractors. |
| **Comprehension** | CEc | `comprehension_complexity` | Difficulty of understanding text or semantic content. Ranges from none to highly convoluted, interconnected concepts. |
| | CEe | `comprehension_evaluation` | Difficulty of generating and articulating ideas. Ranges from no expression to sophisticated, interconnected content. |
| **Conceptualization** | CL | `conceptualization_and_learning` | Need for abstraction, inductive reasoning, or analogical reasoning. |
| | AT | `atypicality` | How novel the task is. Ranges from well-known tasks to fundamentally unique problems. |
| **Knowledge** | KNa | `knowledge_applied_sciences` | Knowledge in medicine, law, education, business, agriculture, or engineering. |
| | KNc | `knowledge_cultural` | Everyday knowledge from daily life, social interactions, and popular media. |
| | KNf | `knowledge_formal_sciences` | Knowledge in mathematics, logic, computer science, or statistics. |
| | KNn | `knowledge_natural_sciences` | Knowledge in physics, chemistry, biology, astronomy, earth sciences, or ecology. |
| | KNs | `knowledge_social_sciences` | Knowledge in history, psychology, sociology, anthropology, literature, art, philosophy, or linguistics. |
| **Metacognition** | MCr | `metacognition_relevance` | Difficulty of identifying what information is necessary to solve the task. |
| | MCt | `metacognition_task_planning` | Difficulty of monitoring and regulating thought processes. |
| | MCu | `metacognition_uncertainty` | Difficulty of knowing what you know and do not know. |
| **Reasoning & Other** | MS | `mind_modelling` | Cognitive demands for modelling others' intentions, emotions, and beliefs. |
| | QLl | `logical_reasoning_logic` | Deductive reasoning demands: matching rules, structured decisions, deriving conclusions. |
| | QLq | `logical_reasoning_quantitative` | Complexity of reasoning about quantities and numerical relationships. |
| | SNs | `spatial_physical_understanding` | Complexity of spatial and physical reasoning. |
| | VO | `volume` | Time a competent human would need. Ranges from under 1 second to over 1,000 minutes. |

The **Code** column shows the original ADeLe identifiers from the paper. The **Trait Name** column shows the snake_case name used throughout the Karenina API.

## 4. How Classification Works

The `QuestionClassifier` is the entry point. It wraps an LLM call that sends the question text together with the ADeLe dimension rubrics, and parses the response into structured scores.

```
Question text
       │
       ▼
 QuestionClassifier
  ├── builds prompt from ADeLe rubric definitions
  ├── sends to LLM (structured output)
  └── parses response
       │
       ▼
 QuestionClassificationResult
  ├── scores: {"attention_and_scan": 0, "logical_reasoning_logic": 3, ...}
  ├── labels: {"attention_and_scan": "none", "logical_reasoning_logic": "intermediate", ...}
  └── metadata (model, timestamp, token usage)
```

### Evaluation modes

The classifier supports two strategies for how it sends dimensions to the LLM:

| Mode | LLM calls per question | Tradeoff | Best for |
|---|---|---|---|
| `batch` (default) | 1 | All 18 dimensions in a single call. Faster and cheaper; dimensions share context. | Most use cases |
| `sequential` | 1 per dimension | Each dimension in a separate call. Each gets full prompt attention; can run in parallel with `async_enabled=True`. | When precision on individual dimensions matters |

### Under the hood

Each ADeLe dimension is implemented as an `LLMRubricTrait` with `kind="literal"` and 6 classes corresponding to the ordinal scale. The classifier uses these traits to build its prompts but does not route through the [verification pipeline](../verification-pipeline/). It calls the LLM directly via `LLMPort.with_structured_output()` to get reliable structured responses. Parsed rubric files are cached with `lru_cache` so repeated classifications do not re-parse the bundled rubric definitions.

## 5. Using the QuestionClassifier

### Classify a single question

```python
from karenina.integrations.adele import QuestionClassifier

# Default: claude-haiku-4-5 via Anthropic, batch mode, temperature 0.0
classifier = QuestionClassifier()

result = classifier.classify_single(
    "What is the chemical formula for water?"
)

# Scores are integer indices (0-5) keyed by trait name
print("Scores (first 5):", dict(list(result.scores.items())[:5]))

# Labels are the ordinal class names
print("Labels (first 5):", dict(list(result.labels.items())[:5]))

# Summary combines both
print("Summary (first 5):", dict(list(result.get_summary().items())[:5]))
```

### Classify a batch of questions

```python
questions = [
    ("q1", "What is 2 + 2?"),
    ("q2", "Explain the significance of BCL2 in cancer therapy."),
    ("q3", "Write a haiku about recursion."),
]

results = classifier.classify_batch(
    questions,
    on_progress=lambda done, total: print(f"{done}/{total}")
)

for qid, r in results.items():
    # Show only non-zero dimensions for readability
    nonzero = {k: v for k, v in r.get_summary().items() if not v.startswith("none")}
    print(f"{qid}: {nonzero}")
```

### Classify a subset of dimensions

You do not have to evaluate all 18 dimensions. Pass `trait_names` to select specific ones:

```python
result = classifier.classify_single(
    "Prove that the square root of 2 is irrational.",
    trait_names=["logical_reasoning_logic", "knowledge_formal_sciences", "atypicality"]
)

# Only the selected dimensions appear in scores and labels
print(result.scores)
print(result.get_summary())
```

### Worked example: filter by difficulty, then verify

A common workflow is to classify questions first, then run verification only on a subset that meets a complexity threshold.

```python
# Classify a set of questions
all_results = classifier.classify_batch([
    ("q1", "What is 2 + 2?"),
    ("q2", "Prove that the square root of 2 is irrational."),
    ("q3", "Explain the significance of BCL2 in cancer therapy."),
])

# Filter to questions requiring high logical reasoning (score >= 4)
complex_ids = [
    qid for qid, r in all_results.items()
    if r.scores.get("logical_reasoning_logic", 0) >= 4
]
print(f"Questions requiring high logical reasoning: {complex_ids}")
```

The filtered `complex_ids` list can then be passed to `benchmark.run_verification(config=config, question_ids=complex_ids)` to run verification only on the hard questions. This avoids spending compute on questions that are too easy to meaningfully differentiate models.

## 6. Working with ADeLe Traits Directly

Because ADeLe dimensions are standard `LLMRubricTrait` objects with `kind="literal"`, you can access them directly for inspection or reuse.

```python
from karenina.integrations.adele import (
    get_adele_trait,
    get_adele_trait_by_code,
    get_all_adele_traits,
    create_adele_rubric,
    ADELE_TRAIT_NAMES,
    ADELE_CODES,
    ADELE_CODE_TO_NAME,
    ADELE_NAME_TO_CODE,
)

# Get a single trait by snake_case name
trait = get_adele_trait("logical_reasoning_logic")
print(f"kind={trait.kind}, classes={len(trait.classes)}, higher_is_better={trait.higher_is_better}")

# Get a trait by its original ADeLe code
trait_by_code = get_adele_trait_by_code("QLl")
print(f"Code 'QLl' -> name='{trait_by_code.name}'")

# Get all 18 traits
all_traits = get_all_adele_traits()
print(f"Total traits: {len(all_traits)}")
print(f"All literal kind: {all(t.kind == 'literal' for t in all_traits)}")
```

```python
# Create a Rubric containing selected ADeLe traits
rubric = create_adele_rubric(["knowledge_formal_sciences", "logical_reasoning_logic"])
print(f"Rubric with {len(rubric.llm_traits)} traits: {[t.name for t in rubric.llm_traits]}")

# Create a Rubric with all 18 traits (pass None or omit argument)
full_rubric = create_adele_rubric()
print(f"Full rubric: {len(full_rubric.llm_traits)} traits")
```

```python
# Look up codes and names
print(f"attention_and_scan -> code: {ADELE_NAME_TO_CODE['attention_and_scan']}")
print(f"QLl -> name: {ADELE_CODE_TO_NAME['QLl']}")
print(f"All codes: {ADELE_CODES}")
```

### Trait structure

Each ADeLe trait is an `LLMRubricTrait` with:

| Field | Value |
|---|---|
| `kind` | `"literal"` |
| `classes` | Ordered dict of 6 class names (`"none"` through `"very_high"`) mapped to descriptions from the original ADeLe rubrics |
| `min_score` | `0` |
| `max_score` | `5` |
| `higher_is_better` | `True` (higher index = higher cognitive demand) |
| `deep_judgment_enabled` | `False` (per-trait field for rubric deep judgment) |

```python
# Inspect the class structure of a trait
trait = get_adele_trait("attention_and_scan")
print(f"min_score={trait.min_score}, max_score={trait.max_score}")
print(f"Class names: {list(trait.classes.keys())}")
print(f"First class description (truncated): {list(trait.classes.values())[0][:120]}...")
```

These traits are the same type described in [LLM rubric traits (literal kind)](../rubrics/llm-traits/). The difference is context: when used through the `QuestionClassifier`, they evaluate question difficulty; when attached to a rubric, they would evaluate response properties.

## 7. Storing and Loading Results

Classification results can be persisted in [checkpoint](../../../core_concepts/questions-and-benchmarks/checkpoints/) metadata and reconstructed later.

```python
from karenina.integrations.adele import QuestionClassificationResult

# Construct a result (normally returned by classify_single)
result = QuestionClassificationResult(
    question_id="q1",
    question_text="What is 2 + 2?",
    scores={"attention_and_scan": 0, "knowledge_formal_sciences": 1, "logical_reasoning_quantitative": 1},
    labels={"attention_and_scan": "none", "knowledge_formal_sciences": "very_low", "logical_reasoning_quantitative": "very_low"},
    model="claude-haiku-4-5",
    classified_at="2025-06-15T10:30:00",
)

# Convert to checkpoint metadata format
metadata = result.to_checkpoint_metadata()
print(metadata)
```

The metadata dict can be passed as `custom_metadata` when adding a question to a benchmark via `benchmark.add_question(question=..., raw_answer=..., custom_metadata=metadata)`.

```python
# Reconstruct from checkpoint metadata
loaded = QuestionClassificationResult.from_checkpoint_metadata(
    metadata,
    question_id="q1",
    question_text="What is 2 + 2?"
)
print(f"Loaded scores: {loaded.scores}")
print(f"Loaded model: {loaded.model}")

# Returns None when no classification exists
empty = QuestionClassificationResult.from_checkpoint_metadata(
    {"other_key": "value"},
    question_id="q1",
    question_text="What is 2 + 2?"
)
print(f"From empty metadata: {empty}")
```

## 8. Configuration Reference

### QuestionClassifier constructor

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `model_name` | `str` | `"claude-haiku-4-5"` | Model for classification |
| `provider` | `str` | `"anthropic"` | Model provider |
| `interface` | `str` | `"langchain"` | Interface type (`"langchain"`, `"openrouter"`, or `"openai_endpoint"`) |
| `temperature` | `float` | `0.0` | LLM temperature |
| `trait_eval_mode` | `str` | `"batch"` | `"batch"` (one call) or `"sequential"` (per-dimension) |
| `async_enabled` | `bool \| None` | `None` | Parallel execution for sequential mode. If `None`, reads `KARENINA_ASYNC_ENABLED` env var (default: `True`). |
| `async_max_workers` | `int \| None` | `None` | Max concurrent workers. If `None`, reads `KARENINA_ASYNC_MAX_WORKERS` env var (default: `2`). |
| `model_config` | `ModelConfig \| None` | `None` | Full model configuration (overrides `model_name`, `provider`, `temperature`) |
| `llm` | `LLMPort \| None` | `None` | Pre-initialized LLM instance (skips lazy initialization) |
| `endpoint_base_url` | `str \| None` | `None` | Custom base URL for `openai_endpoint` interface |
| `endpoint_api_key` | `str \| None` | `None` | API key for `openai_endpoint` interface |

The LLM is lazily initialized on first use. If you pass `llm`, it is used directly. If you pass `model_config`, it takes precedence over the individual `model_name`/`provider`/`temperature` parameters.

### Public API summary

| Export | Type | Purpose |
|---|---|---|
| `QuestionClassifier` | Class | Classify questions against ADeLe dimensions |
| `QuestionClassificationResult` | Class | Stores scores, labels, and metadata for one classified question |
| `AdeleTraitInfo` | Class | API-friendly schema describing a single ADeLe trait |
| `get_adele_trait(name)` | Function | Get one trait by snake_case name |
| `get_adele_trait_by_code(code)` | Function | Get one trait by original ADeLe code (e.g., `"QLl"`) |
| `get_all_adele_traits()` | Function | Get all 18 traits as `LLMRubricTrait` objects |
| `create_adele_rubric(trait_names)` | Function | Create a `Rubric` with specified (or all) ADeLe traits |
| `ADELE_TRAIT_NAMES` | `list[str]` | All 18 snake_case trait names |
| `ADELE_CODES` | `list[str]` | All 18 original ADeLe codes |
| `ADELE_CODE_TO_NAME` | `dict[str, str]` | Code to snake_case name mapping |
| `ADELE_NAME_TO_CODE` | `dict[str, str]` | Snake_case name to code mapping |

## 9. Next Steps

- [Scaled Authoring](../../creating-benchmarks/scaled-authoring/): ADeLe classification in the benchmark authoring workflow
- [LLM rubric traits (literal kind)](../rubrics/llm-traits/): the trait type ADeLe dimensions use internally
- [Rubrics](../../../core_concepts/rubrics/): how trait-based evaluation works
- [Running verification](../../../workflows/running-verification/): evaluate your benchmark after filtering by ADeLe scores
- [Evaluation modes](../evaluation-modes/): template, rubric, or both
