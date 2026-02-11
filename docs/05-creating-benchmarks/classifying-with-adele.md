# Classifying Questions with ADeLe

ADeLe (Annotated Demand Levels) classifies your benchmark questions by cognitive complexity across 18 dimensions. This helps you understand what your benchmark measures and filter questions by difficulty.

> ADeLe was introduced in [Zhou et al. (2025)](https://arxiv.org/abs/2503.06378). See [ADeLe Concepts](../core_concepts/adele.md) for full attribution.

!!! note "Concept vs Workflow"
    This page covers **how to run classification** in the benchmark creation workflow. For the full list of 18 dimensions and the ordinal scale, see [ADeLe Concepts](../core_concepts/adele.md).

## When to Classify

Classification is **optional** in the benchmark creation workflow. Use it when you want to:

- **Understand your benchmark** — See which cognitive dimensions your questions exercise
- **Filter by complexity** — Select questions that test specific skills (e.g., only reasoning-heavy questions)
- **Balance difficulty** — Ensure your benchmark covers a range of complexity levels
- **Compare question sets** — Quantify how two sets of questions differ

## Setting Up the Classifier

```python
from karenina.integrations.adele import QuestionClassifier

# Default: uses claude-3-5-haiku via Anthropic
classifier = QuestionClassifier()

# Custom model
classifier = QuestionClassifier(
    model_name="gpt-4o-mini",
    provider="openai",
    interface="langchain",
)
```

The classifier uses an LLM to evaluate each question. The default model (`claude-3-5-haiku-latest`) is fast and cost-effective for classification tasks.

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | `"claude-3-5-haiku-latest"` | Model to use for classification |
| `provider` | `str` | `"anthropic"` | LLM provider |
| `interface` | `str` | `"langchain"` | Adapter interface |
| `temperature` | `float` | `0.0` | LLM temperature (0.0 for deterministic results) |
| `trait_eval_mode` | `str` | `"batch"` | `"batch"` (all traits in 1 call) or `"sequential"` (per-trait calls) |
| `async_enabled` | `bool \| None` | `None` | Enable parallel execution (reads `KARENINA_ASYNC_ENABLED` env var if None) |
| `async_max_workers` | `int \| None` | `None` | Max concurrent workers (reads `KARENINA_ASYNC_MAX_WORKERS` env var if None) |
| `model_config` | `ModelConfig \| None` | `None` | Optional `ModelConfig` object (takes precedence over individual params) |

## Classifying a Single Question

```python
result = classifier.classify_single(
    question_text="What is the capital of France?",
    question_id="q1",
)

print(result.scores)
# {"attention_and_scan": 0, "volume": 1, "mind_modelling": 0, ...}

print(result.labels)
# {"attention_and_scan": "none", "volume": "very_low", "mind_modelling": "none", ...}
```

Each dimension receives a score from **0** (none) to **5** (very_high). The `labels` dict provides the human-readable label for each score.

### Classifying a Subset of Traits

If you only need certain dimensions, pass `trait_names` to reduce LLM calls:

```python
result = classifier.classify_single(
    question_text="Analyze the sociological implications of AI on labor markets.",
    trait_names=["volume", "mind_modelling", "logical_reasoning_logic"],
)

# Only the requested traits are scored
print(result.scores)
# {"volume": 4, "mind_modelling": 3, "logical_reasoning_logic": 2}
```

## Classifying Multiple Questions

Use `classify_batch()` for efficient classification of many questions:

```python
questions = [
    ("q1", "What is 2+2?"),
    ("q2", "Explain the relationship between GDP growth and inflation."),
    ("q3", "Write a recursive function to compute Fibonacci numbers."),
]

results = classifier.classify_batch(
    questions=questions,
    on_progress=lambda completed, total: print(f"{completed}/{total}"),
)

# results is a dict: question_id -> QuestionClassificationResult
for q_id, result in results.items():
    print(f"{q_id}: volume={result.scores['volume']}")
```

## Interpreting Results

### The Result Object

`QuestionClassificationResult` contains:

| Field | Type | Description |
|-------|------|-------------|
| `scores` | `dict[str, int]` | Trait name to score (0-5, or -1 for error) |
| `labels` | `dict[str, str]` | Trait name to class label (`"none"` through `"very_high"`) |
| `model` | `str` | Model used for classification |
| `classified_at` | `str` | ISO timestamp |
| `usage_metadata` | `dict[str, Any]` | Token usage information |

### The Summary View

Use `get_summary()` for a compact overview:

```python
summary = result.get_summary()
# {"attention_and_scan": "none (0)", "volume": "very_low (1)", ...}
```

### Understanding Scores

Higher scores mean greater cognitive complexity on that dimension:

| Score | Label | Meaning |
|-------|-------|---------|
| **0** | none | No complexity on this dimension |
| **1** | very_low | Minimal complexity |
| **2** | low | Some complexity |
| **3** | intermediate | Moderate complexity |
| **4** | high | Significant complexity |
| **5** | very_high | Extreme complexity |
| **-1** | *(error)* | Classification failed for this trait |

A simple factual question like *"What is the capital of France?"* will score 0-1 on most dimensions. A complex analytical question will score 3-5 on dimensions like `volume`, `comprehension_complexity`, and relevant knowledge domains.

## Using Results for Filtering

### Filtering by Complexity

After classifying your questions, use the scores to select subsets:

```python
# Classify all questions in a benchmark
benchmark = Benchmark.load("my_benchmark.jsonld")
questions = benchmark.get_all_questions()

question_pairs = [(q["id"], q["text"]) for q in questions]
results = classifier.classify_batch(questions=question_pairs)

# Find high-complexity reasoning questions
reasoning_heavy = [
    q_id for q_id, result in results.items()
    if result.scores.get("logical_reasoning_logic", 0) >= 3
]

# Find quick factual questions (low volume, low reasoning)
quick_factual = [
    q_id for q_id, result in results.items()
    if result.scores.get("volume", 0) <= 1
    and result.scores.get("logical_reasoning_logic", 0) <= 1
]
```

### Storing Classifications in Checkpoint Metadata

Save classification results alongside your questions for later use:

```python
# Convert result to metadata format
metadata = result.to_checkpoint_metadata()

# Store when adding a question
benchmark.add_question(
    question="What is photosynthesis?",
    raw_answer="The process by which plants convert sunlight to energy.",
    custom_metadata=metadata,
)

# Later, retrieve classifications from a loaded benchmark
question = benchmark.get_question(question_id)
restored = QuestionClassificationResult.from_checkpoint_metadata(
    metadata=question.get("custom_metadata", {}),
    question_id=question_id,
    question_text=question["text"],
)
```

### Running Verification on a Subset

Use filtered question IDs with `run_verification()`:

```python
from karenina.schemas.verification import VerificationConfig

config = VerificationConfig(...)
results = benchmark.run_verification(
    config=config,
    question_ids=reasoning_heavy,  # Only verify the filtered subset
)
```

## Evaluation Modes

The classifier supports two evaluation strategies:

| Mode | LLM Calls | Speed | Accuracy | Best For |
|------|-----------|-------|----------|----------|
| `"batch"` | 1 per question | Fast | Good | Bulk classification, cost-sensitive |
| `"sequential"` | 18 per question | Slower | Potentially better | Complex questions, maximum accuracy |

```python
# Batch mode (default) — all 18 traits in one LLM call
classifier = QuestionClassifier(trait_eval_mode="batch")

# Sequential mode — one call per trait, with parallel execution
classifier = QuestionClassifier(
    trait_eval_mode="sequential",
    async_enabled=True,
    async_max_workers=4,
)
```

## Working with ADeLe Traits Directly

The ADeLe integration also provides helper functions for working with traits:

```python
from karenina.integrations.adele import (
    get_adele_trait,
    get_all_adele_traits,
    create_adele_rubric,
    ADELE_TRAIT_NAMES,
)

# Get a single trait as LLMRubricTrait
trait = get_adele_trait("attention_and_scan")
print(trait.kind)     # "literal"
print(trait.classes)  # {"none": "...", "very_low": "...", ...}

# List all 18 trait names
print(ADELE_TRAIT_NAMES)

# Create a Rubric with ADeLe traits (for use in verification pipeline)
rubric = create_adele_rubric(["volume", "mind_modelling"])
print(len(rubric.llm_traits))  # 2
```

!!! tip "ADeLe traits as rubric traits"
    Each ADeLe dimension is a [literal trait](../core_concepts/rubrics/literal-traits.md) with 6 ordered classes. You can use `create_adele_rubric()` to add ADeLe dimensions directly to your benchmark's rubric evaluation.

## Next Steps

- [ADeLe Concepts](../core_concepts/adele.md) — Full list of 18 dimensions with descriptions
- [Generating Templates](generating-templates.md) — Auto-generate answer templates for your questions
- [Defining Rubrics](defining-rubrics.md) — Add rubric traits for quality evaluation
- [Running Verification](../06-running-verification/index.md) — Execute your benchmark
- [Literal Traits](../core_concepts/rubrics/literal-traits.md) — How ADeLe traits work as rubric traits
