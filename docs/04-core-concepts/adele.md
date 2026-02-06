# ADeLe: Question Classification

**ADeLe** (Assessment of Difficulty Level) is a framework for classifying questions along 18 cognitive dimensions. Each dimension measures a different aspect of question difficulty on a 6-level ordinal scale, giving you a detailed profile of what makes a question challenging.

Karenina includes a built-in ADeLe integration that can classify your benchmark questions using an LLM, store the results in checkpoint metadata, and use them for filtering or analysis.

## The 18 Dimensions

ADeLe defines 18 dimensions organized into six categories:

### Attention

| Code | Dimension | What It Measures |
|------|-----------|-----------------|
| **AS** | Attention and Scan | Level of attention required to locate specific elements within information. Ranges from immediately obvious targets to sustained tracking of multiple targets among distractors. |

### Comprehension & Expression

| Code | Dimension | What It Measures |
|------|-----------|-----------------|
| **CEc** | Comprehension Complexity | Difficulty of understanding text, stories, or semantic content. Ranges from no comprehension required to highly convoluted, interconnected concepts. |
| **CEe** | Comprehension Evaluation | Difficulty of generating and articulating ideas or semantic content. Ranges from no meaningful expression to highly sophisticated, interconnected content. |

### Conceptualization

| Code | Dimension | What It Measures |
|------|-----------|-----------------|
| **CL** | Conceptualization and Learning | Need for abstraction, inductive reasoning, or analogical reasoning. Ranges from applying established procedures to generating new analogical frameworks in real time. |
| **AT** | Atypicality | How novel or unique the task is. Ranges from well-known, memorized tasks to fundamentally different tasks not found in standard sources. |

### Knowledge Domains

| Code | Dimension | What It Measures |
|------|-----------|-----------------|
| **KNa** | Applied Sciences | Knowledge in medicine, law, education, business, agriculture, or engineering. |
| **KNc** | Cultural Knowledge | Everyday knowledge from daily life, social interactions, and popular media. |
| **KNf** | Formal Sciences | Knowledge in mathematics, logic, computer science, or statistics. |
| **KNn** | Natural Sciences | Knowledge in physics, chemistry, biology, astronomy, earth sciences, or ecology. |
| **KNs** | Social Sciences | Knowledge in history, psychology, sociology, anthropology, literature, art, philosophy, or linguistics. |

### Metacognition

| Code | Dimension | What It Measures |
|------|-----------|-----------------|
| **MCr** | Relevance Assessment | Difficulty of identifying what information is necessary to solve the task. Ranges from immediately apparent to requiring constant reassessment. |
| **MCt** | Task Planning | Difficulty of monitoring and regulating thought processes. Ranges from no critical thinking to sophisticated metacognitive strategies. |
| **MCu** | Uncertainty Management | Difficulty of knowing what you know and don't know. Ranges from unambiguous to extremely challenging knowledge boundary determination. |

### Reasoning & Other

| Code | Dimension | What It Measures |
|------|-----------|-----------------|
| **MS** | Mind Modelling | Cognitive demands for modelling others' intentions, emotions, and beliefs. Ranges from no mind modelling to managing multiple agents' mental states. |
| **QLl** | Logical Reasoning | Deductive reasoning demands — matching rules, structured decisions, deriving conclusions. Ranges from none to complex nested conditionals. |
| **QLq** | Quantitative Reasoning | Complexity of reasoning about quantities and numerical relationships. Ranges from none to expert-level quantitative analysis. |
| **SNs** | Spatial Understanding | Complexity of spatial and physical reasoning. Ranges from none to multiple simultaneous transformations. |
| **VO** | Volume | Time a competent human would need to complete the task. Ranges from under 1 second to over 1,000 minutes. |

## The 6-Level Ordinal Scale

Every dimension uses the same ordinal scale:

| Score | Label | Meaning |
|-------|-------|---------|
| **0** | none | No or minimal requirement |
| **1** | very_low | Very low requirement |
| **2** | low | Low requirement |
| **3** | intermediate | Moderate requirement |
| **4** | high | High requirement |
| **5** | very_high | Expert-level requirement |

A score of **-1** indicates a classification error (the LLM did not return a valid class for that dimension).

## Using the QuestionClassifier

The `QuestionClassifier` evaluates questions against ADeLe dimensions using an LLM:

```python
from karenina.integrations.adele import QuestionClassifier

# Default: uses claude-3-5-haiku-latest via Anthropic
classifier = QuestionClassifier()

# Classify a single question
result = classifier.classify_single(
    "What is the chemical formula for water?"
)

# View scores and labels
print(result.scores)   # {"attention_and_scan": 0, "knowledge_natural_sciences": 1, ...}
print(result.labels)   # {"attention_and_scan": "none", "knowledge_natural_sciences": "very_low", ...}
print(result.get_summary())  # {"attention_and_scan": "none (0)", ...}
```

### Constructor Options

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `model_name` | str | `"claude-3-5-haiku-latest"` | Model for classification |
| `provider` | str | `"anthropic"` | Model provider |
| `interface` | str | `"langchain"` | Interface type |
| `temperature` | float | `0.0` | LLM temperature (deterministic) |
| `trait_eval_mode` | str | `"batch"` | `"batch"` (one call) or `"sequential"` (per-trait) |
| `async_enabled` | bool \| None | None | Parallel execution for sequential mode |
| `async_max_workers` | int \| None | None | Max concurrent workers |
| `model_config` | ModelConfig \| None | None | Full model configuration (overrides individual params) |
| `llm` | LLMPort \| None | None | Pre-initialized LLM instance |

### Classifying Multiple Questions

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

for qid, result in results.items():
    print(f"{qid}: {result.get_summary()}")
```

### Evaluating a Subset of Dimensions

You can classify against specific dimensions instead of all 18:

```python
result = classifier.classify_single(
    "Prove that the square root of 2 is irrational.",
    trait_names=["logical_reasoning_logic", "knowledge_formal_sciences", "atypicality"]
)
```

## Evaluation Modes

The classifier supports two evaluation strategies:

| Mode | LLM Calls | Speed | Accuracy | Best For |
|------|-----------|-------|----------|----------|
| **batch** (default) | 1 per question | Fast | Good | Most use cases |
| **sequential** | 1 per trait per question | Slower | Potentially better | When precision on individual dimensions matters |

In sequential mode with `async_enabled=True`, trait evaluations run in parallel for faster throughput.

## Storing Results in Checkpoints

Classification results can be saved to and loaded from checkpoint metadata:

```python
# Save classification to checkpoint metadata
metadata = result.to_checkpoint_metadata()
# Returns: {"adele_classification": {"scores": {...}, "labels": {...}, "classified_at": "...", "model": "..."}}

# Store when adding a question
benchmark.add_question(
    question="What is 2 + 2?",
    raw_answer="4",
    custom_metadata=metadata
)

# Reconstruct from checkpoint metadata later
loaded = QuestionClassificationResult.from_checkpoint_metadata(
    metadata,
    question_id="q1",
    question_text="What is 2 + 2?"
)
```

## Working with ADeLe Traits Directly

ADeLe dimensions are implemented as `LLMRubricTrait` objects with `kind="literal"` and 6 classes. You can access them directly:

```python
from karenina.integrations.adele import (
    get_adele_trait,
    get_all_adele_traits,
    create_adele_rubric,
    ADELE_TRAIT_NAMES,
    ADELE_CODES,
    ADELE_CODE_TO_NAME,
    ADELE_NAME_TO_CODE,
)

# Get a single trait by name
trait = get_adele_trait("logical_reasoning_logic")

# Get all 18 traits
all_traits = get_all_adele_traits()

# Create a Rubric with selected ADeLe traits
rubric = create_adele_rubric(["knowledge_formal_sciences", "logical_reasoning_logic"])

# Look up codes
ADELE_NAME_TO_CODE["attention_and_scan"]  # "AS"
ADELE_CODE_TO_NAME["QLl"]                 # "logical_reasoning_logic"
```

## Using Classifications for Filtering

A common pattern is filtering benchmark questions by difficulty before running verification:

```python
# Classify all questions
results = classifier.classify_batch(
    [(qid, q["text"]) for qid, q in benchmark.get_all_questions().items()]
)

# Select only questions requiring high logical reasoning
complex_ids = [
    qid for qid, r in results.items()
    if r.scores.get("logical_reasoning_logic", 0) >= 4
]

# Run verification on the filtered subset
result_set = benchmark.run_verification(
    config=config,
    question_ids=complex_ids
)
```

## Next Steps

- [Classifying questions in your benchmark](../05-creating-benchmarks/classifying-with-adele.md) — step-by-step workflow
- [Literal rubric traits](rubrics/literal-traits.md) — the trait type ADeLe dimensions use
- [Running verification](../06-running-verification/index.md) — evaluate your benchmark
- [Evaluation modes](evaluation-modes.md) — template, rubric, or both
