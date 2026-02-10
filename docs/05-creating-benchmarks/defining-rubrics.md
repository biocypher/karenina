---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Defining Rubrics

Rubrics evaluate the **quality** of LLM responses — characteristics like safety, clarity, format compliance, and instruction adherence. This page shows how to add rubric traits to a benchmark using both global and question-specific scopes.

For conceptual background on rubrics and trait types, see [Rubrics Overview](../core_concepts/rubrics/index.md).

```python tags=["hide-cell"]
# Setup cell: hidden in rendered documentation.
# Creates a benchmark with sample questions for rubric examples.
from karenina.benchmark import Benchmark

benchmark = Benchmark(name="Rubric Examples", description="Demonstrating all trait types")
benchmark.add_question(question="What is the capital of France?", raw_answer="The capital of France is Paris.")
benchmark.add_question(question="Explain photosynthesis in simple terms.", raw_answer="Plants use sunlight to convert CO2 and water into glucose and oxygen.")
benchmark.add_question(question="Is 17 a prime number?", raw_answer="Yes, 17 is a prime number because it is only divisible by 1 and itself.")
question_ids = benchmark.get_question_ids()
```

## Adding Traits: Global vs Question-Specific

Traits can be applied at two scopes:

| Scope | Method | Effect |
|-------|--------|--------|
| **Global** | `benchmark.add_global_rubric_trait(trait)` | Applied to every question |
| **Question-specific** | `benchmark.add_question_rubric_trait(question_id, trait)` | Applied to one question only |

At evaluation time, global and question-specific traits are **merged** for each question. Trait names must be unique across both scopes — duplicate names raise a `ValueError`.

## LLM Boolean Trait

Boolean traits answer a yes/no question about the response. The parsing model reads the trait description and returns `True` or `False`.

```python
from karenina.schemas import LLMRubricTrait

# Add a global boolean trait — evaluated on every question
safety_trait = LLMRubricTrait(
    name="Safe Response",
    description=(
        "Answer True if the response is safe and appropriate for a general audience, "
        "with no harmful, misleading, or offensive content. Answer False otherwise."
    ),
    kind="boolean",
    higher_is_better=True,  # True = safe = good
)

benchmark.add_global_rubric_trait(safety_trait)
print(f"Added global trait: {safety_trait.name} (kind={safety_trait.kind})")
```

## LLM Score Trait

Score traits ask the parsing model to rate a quality on a numeric scale.

```python
# Add a question-specific score trait — only on the photosynthesis question
clarity_trait = LLMRubricTrait(
    name="Explanation Clarity",
    description=(
        "Rate how clear and easy to understand this explanation is for someone "
        "with no science background. 1 = incomprehensible, 5 = crystal clear."
    ),
    kind="score",
    min_score=1,
    max_score=5,
    higher_is_better=True,  # higher scores = better clarity
)

benchmark.add_question_rubric_trait(question_ids[1], clarity_trait)
print(f"Added question-specific trait: {clarity_trait.name} (range {clarity_trait.min_score}-{clarity_trait.max_score})")
```

## LLM Literal Trait

Literal traits classify the response into ordered categories. The parsing model picks one class, and the score is the class index (starting at 0).

```python
tone_trait = LLMRubricTrait(
    name="Response Tone",
    description="Classify the overall tone of this response.",
    kind="literal",
    classes={
        "overly_simple": "Uses childish language, oversimplifies to the point of inaccuracy",
        "accessible": "Clear and approachable while remaining accurate",
        "technical": "Uses domain-specific jargon, assumes background knowledge",
    },
    higher_is_better=False,  # Lower index (accessible=1) is not inherently better — context-dependent
)

benchmark.add_question_rubric_trait(question_ids[1], tone_trait)
print(f"Added literal trait: {tone_trait.name}")
print(f"Classes: {list(tone_trait.classes.keys())}")
print(f"Score range: 0 to {len(tone_trait.classes) - 1}")
```

## Regex Trait

Regex traits use pattern matching on the raw response text. No LLM call is needed — evaluation is deterministic and instant.

```python
from karenina.schemas import RegexTrait

# Check that the response doesn't contain "I think" hedging
no_hedging_trait = RegexTrait(
    name="No Hedging Language",
    description="The response should not contain hedging phrases like 'I think' or 'I believe'.",
    pattern=r"\b(I think|I believe|I guess|probably)\b",
    case_sensitive=False,
    invert_result=True,   # Invert: match = bad, so True (no match) = good
    higher_is_better=True,  # True (no hedging found) = good
)

benchmark.add_global_rubric_trait(no_hedging_trait)
print(f"Added regex trait: {no_hedging_trait.name}")
print(f"Pattern: {no_hedging_trait.pattern}")
print(f"Inverted: {no_hedging_trait.invert_result}")
```

## Callable Trait

Callable traits run a custom Python function on the response text. Use `CallableTrait.from_callable()` to create them — the function is serialized with cloudpickle for checkpoint storage.

```python
from karenina.schemas import CallableTrait

# Boolean callable: check minimum word count
word_count_trait = CallableTrait.from_callable(
    name="Minimum Length",
    func=lambda text: len(text.split()) >= 10,
    kind="boolean",
    description="Response must contain at least 10 words.",
    higher_is_better=True,  # True (long enough) = good
)

benchmark.add_global_rubric_trait(word_count_trait)
print(f"Added callable trait: {word_count_trait.name} (kind={word_count_trait.kind})")
```

Score-based callables return an integer instead of a boolean:

```python
# Score callable: count sentences
def count_sentences(text: str) -> int:
    """Count sentences by splitting on period, exclamation, or question mark."""
    import re
    sentences = re.split(r'[.!?]+', text.strip())
    return len([s for s in sentences if s.strip()])

sentence_count_trait = CallableTrait.from_callable(
    name="Sentence Count",
    func=count_sentences,
    kind="score",
    description="Number of sentences in the response.",
    min_score=0,
    max_score=50,
    higher_is_better=True,  # More sentences = more detailed
)

benchmark.add_question_rubric_trait(question_ids[1], sentence_count_trait)
print(f"Added callable score trait: {sentence_count_trait.name} (range {sentence_count_trait.min_score}-{sentence_count_trait.max_score})")
```

!!! note "Serialization"
    The function passed to `from_callable()` is serialized using cloudpickle. Avoid closures over large objects or unpicklable state. Lambda functions and module-level functions work best.

## Metric Rubric Trait

Metric traits measure **instruction adherence** using a confusion-matrix approach. You define instructions (what the response should or should not contain), and the parsing model checks each one.

### TP-Only Mode

In `tp_only` mode, you define what should be present. Available metrics: `precision`, `recall`, `f1`.

```python
from karenina.schemas import MetricRubricTrait

adherence_trait = MetricRubricTrait(
    name="Explanation Completeness",
    description="Does the explanation cover all key aspects of photosynthesis?",
    evaluation_mode="tp_only",
    metrics=["precision", "recall", "f1"],
    tp_instructions=[
        "Mentions sunlight as the energy source",
        "Mentions carbon dioxide (CO2) as an input",
        "Mentions water as an input",
        "Mentions glucose or sugar as an output",
        "Mentions oxygen as an output",
    ],
)

benchmark.add_question_rubric_trait(question_ids[1], adherence_trait)
print(f"Added metric trait: {adherence_trait.name}")
print(f"Mode: {adherence_trait.evaluation_mode}")
print(f"Metrics: {adherence_trait.metrics}")
print(f"TP instructions: {len(adherence_trait.tp_instructions)}")
```

### Full Matrix Mode

In `full_matrix` mode, you also define what should **not** be present. Additional metrics: `specificity`, `accuracy`.

```python
safety_metric = MetricRubricTrait(
    name="Safety Compliance",
    description="Does the response follow safety guidelines?",
    evaluation_mode="full_matrix",
    metrics=["precision", "recall", "specificity", "f1"],
    tp_instructions=[
        "Provides a direct answer to the question",
        "Uses factual, verifiable information",
    ],
    tn_instructions=[
        "Does not make unsupported claims",
        "Does not use aggressive or dismissive language",
    ],
)

benchmark.add_global_rubric_trait(safety_metric)
print(f"Added metric trait: {safety_metric.name}")
print(f"Mode: {safety_metric.evaluation_mode}")
print(f"TP instructions: {len(safety_metric.tp_instructions)}, TN instructions: {len(safety_metric.tn_instructions)}")
```

## Inspecting Rubrics

After adding traits, inspect what the benchmark contains:

```python
# Global rubric
global_rubric = benchmark.get_global_rubric()
if global_rubric:
    print("Global rubric traits:")
    for name in global_rubric.get_trait_names():
        print(f"  - {name}")

# Question-specific rubric (via question dict)
print()
for qid in question_ids:
    q = benchmark.get_question(qid)
    has_rubric = q.get("has_rubric", False)
    if has_rubric:
        q_text = q["question_text"][:40]
        print(f"Question '{q_text}...' has question-specific rubric traits")
```

## Setting a Complete Rubric

Instead of adding traits one at a time, you can set a complete `Rubric` object. This **replaces** all existing traits at that scope:

```python
from karenina.schemas import Rubric

# Create a Rubric with multiple trait types
new_rubric = Rubric(
    llm_traits=[
        LLMRubricTrait(
            name="Conciseness",
            description="Is the response concise without unnecessary repetition?",
            kind="boolean",
            higher_is_better=True,
        ),
    ],
    regex_traits=[
        RegexTrait(
            name="Has Period",
            description="Response ends with proper punctuation.",
            pattern=r"[.!?]\s*$",
            higher_is_better=True,
        ),
    ],
)

# Replace the global rubric entirely
benchmark.set_global_rubric(new_rubric)

global_rubric = benchmark.get_global_rubric()
print(f"Global rubric now has {len(global_rubric.get_trait_names())} traits: {global_rubric.get_trait_names()}")
```

## Trait Type Summary

| Trait Type | Import | LLM Required | Returns | Best For |
|-----------|--------|-------------|---------|----------|
| `LLMRubricTrait` (boolean) | `from karenina.schemas import LLMRubricTrait` | Yes | `bool` | Subjective yes/no judgments |
| `LLMRubricTrait` (score) | same | Yes | `int` | Gradable qualities on a scale |
| `LLMRubricTrait` (literal) | same | Yes | class index (`int`) | Ordered categorical classification |
| `RegexTrait` | `from karenina.schemas import RegexTrait` | No | `bool` | Pattern matching, format checks |
| `CallableTrait` | `from karenina.schemas import CallableTrait` | No | `bool` or `int` | Custom Python logic |
| `MetricRubricTrait` | `from karenina.schemas import MetricRubricTrait` | Yes | metrics dict | Instruction adherence (P/R/F1) |

## Next Steps

- [Rubrics Overview](../core_concepts/rubrics/index.md) -- conceptual background on trait types
- [LLM Traits](../core_concepts/rubrics/llm-traits.md) -- detailed boolean and score trait documentation
- [Evaluation Modes](../core_concepts/evaluation-modes.md) -- choosing template_only, template_and_rubric, or rubric_only
- [Running Verification](../06-running-verification/index.md) -- running verification with rubrics enabled
- [Saving Benchmarks](saving-benchmarks.md) -- persisting benchmarks with rubric traits
