# Few-Shot Configuration

Few-shot prompting provides example question-answer pairs to the LLM before asking the main question, guiding responses toward expected formats, styles, and content.

## Why Use Few-Shot Prompting?

Few-shot examples can improve verification results by:

- **Response quality** — Models learn from demonstrated good answers
- **Consistency** — Responses follow patterns shown in examples
- **Format adherence** — Models match the structure of examples
- **Accuracy** — Examples clarify what kind of answer is expected

For example, if your examples show concise gene symbol answers like "BCL2" and "TP53", the model learns to respond with the standard symbol rather than a verbose explanation.

## Where Examples Come From

Few-shot examples are stored on individual questions as `question → answer` pairs. You add them when creating your benchmark:

```python
benchmark.add_question(
    question_text="How many subunits does hemoglobin A have?",
    raw_answer="4",
    few_shot_examples=[
        {"question": "How many chromosomes in a human somatic cell?", "answer": "46"},
        {"question": "How many base pairs in human mitochondrial DNA?", "answer": "16569"},
    ],
)
```

The `FewShotConfig` on `VerificationConfig` controls **which** of these stored examples are actually used during verification and **how many**.

## The Five Modes

FewShotConfig supports five selection modes that control how examples are chosen:

| Mode | Description | Use Case |
|------|-------------|----------|
| `all` | Use all available examples | Small example sets (2-5 examples) |
| `k-shot` | Randomly sample *k* examples | Large example sets where using all is costly |
| `custom` | Select specific examples by index or hash | Curated sets where order/selection matters |
| `none` | No examples used | Disable for specific questions |
| `inherit` | Use parent (global) settings | Per-question default — falls back to global mode |

### `all` Mode

Uses every example attached to a question. This is the **default global mode**.

```python
from karenina.schemas import FewShotConfig

config = FewShotConfig(global_mode="all")
```

Best when questions have a small number of carefully selected examples and you want them all included in the prompt.

### `k-shot` Mode

Randomly samples *k* examples from the available set. Sampling uses the question ID as a seed for reproducibility — the same question always gets the same examples.

```python
config = FewShotConfig(global_mode="k-shot", global_k=3)
```

Use when questions have many examples but you want to limit prompt length and cost. The default *k* is 3.

### `custom` Mode

Selects specific examples by index position or MD5 hash:

```python
from karenina.schemas import QuestionFewShotConfig

# Select by index
config = FewShotConfig(
    global_mode="custom",
    question_configs={
        "question_1": QuestionFewShotConfig(
            mode="custom",
            selected_examples=[0, 2, 4],  # Use examples at these indices
        ),
    },
)

# Select by hash
config = FewShotConfig(
    global_mode="custom",
    question_configs={
        "question_1": QuestionFewShotConfig(
            mode="custom",
            selected_examples=["abc123def456..."],  # MD5 hash of example question text
        ),
    },
)
```

Use when you've identified the most effective examples for each question and want deterministic selection.

### `none` Mode

Disables few-shot examples entirely:

```python
config = FewShotConfig(global_mode="none")
```

Or disable for a specific question while keeping the global mode:

```python
config = FewShotConfig(
    global_mode="all",
    question_configs={
        "question_3": QuestionFewShotConfig(mode="none"),
    },
)
```

### `inherit` Mode

Per-question default. When a question's mode is `inherit`, it uses the global mode and *k* value. This is the default for `QuestionFewShotConfig`, so questions without explicit overrides automatically inherit.

## Per-Question Overrides

Each question can have its own configuration that overrides the global settings:

```python
config = FewShotConfig(
    global_mode="k-shot",
    global_k=3,
    question_configs={
        # This question uses all examples instead of k-shot
        "question_1": QuestionFewShotConfig(mode="all"),
        # This question uses 5 examples instead of the global 3
        "question_2": QuestionFewShotConfig(mode="k-shot", k=5),
        # This question has no examples
        "question_3": QuestionFewShotConfig(mode="none"),
        # This question inherits global settings (same as not listing it)
        "question_4": QuestionFewShotConfig(mode="inherit"),
    },
)
```

You can also exclude specific examples while using another mode:

```python
config = FewShotConfig(
    global_mode="all",
    question_configs={
        "question_1": QuestionFewShotConfig(
            mode="all",
            excluded_examples=[2],  # Skip example at index 2
        ),
    },
)
```

## FewShotConfig Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `True` | Master switch — when `False`, no examples are used regardless of mode |
| `global_mode` | `"all"` \| `"k-shot"` \| `"custom"` \| `"none"` | `"all"` | Default mode for all questions |
| `global_k` | `int` | `3` | Number of examples for `k-shot` mode |
| `question_configs` | `dict[str, QuestionFewShotConfig]` | `{}` | Per-question overrides keyed by question ID |
| `global_external_examples` | `list[dict[str, str]]` | `[]` | External examples appended to every question |

## QuestionFewShotConfig Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | `"all"` \| `"k-shot"` \| `"custom"` \| `"none"` \| `"inherit"` | `"inherit"` | Selection mode for this question |
| `k` | `int \| None` | `None` | Override global *k* for this question |
| `selected_examples` | `list[str \| int] \| None` | `None` | Indices or MD5 hashes for `custom` mode |
| `external_examples` | `list[dict[str, str]] \| None` | `None` | Question-specific external examples |
| `excluded_examples` | `list[str \| int] \| None` | `None` | Indices or hashes of examples to exclude |

## External Examples

You can add examples that don't come from the question's stored examples. These are appended after the resolved examples:

```python
config = FewShotConfig(
    global_mode="k-shot",
    global_k=2,
    # These are added to every question
    global_external_examples=[
        {"question": "What is the capital of France?", "answer": "Paris"},
    ],
    question_configs={
        "question_1": QuestionFewShotConfig(
            mode="k-shot",
            # These are added only to question_1
            external_examples=[
                {"question": "What gene does imatinib target?", "answer": "BCR-ABL1"},
            ],
        ),
    },
)
```

The resolution order is: resolved examples (from stored examples) → question-specific external → global external.

## Convenience Constructors

FewShotConfig provides factory methods for common patterns:

```python
# From index selections
config = FewShotConfig.from_index_selections({
    "question_1": [0, 2, 4],
    "question_2": [1, 3],
})

# From hash selections
config = FewShotConfig.from_hash_selections({
    "question_1": ["abc123...", "def456..."],
})

# Different k per question
config = FewShotConfig.k_shot_for_questions({
    "question_1": 5,
    "question_2": 2,
}, global_k=3)
```

## Using FewShotConfig in Verification

Pass the config to `VerificationConfig`:

```python
from karenina.schemas import VerificationConfig, FewShotConfig

config = VerificationConfig(
    answering_models=[...],
    parsing_models=[...],
    few_shot_config=FewShotConfig(
        global_mode="k-shot",
        global_k=3,
    ),
)

# Check if few-shot is enabled
config.is_few_shot_enabled()  # True

# Get the config
config.get_few_shot_config()  # FewShotConfig(...)
```

## Validation

VerificationConfig validates few-shot settings:

- In `k-shot` mode, the global *k* must be at least 1
- Per-question *k* values must also be at least 1
- You can validate that selections reference valid examples using `validate_selections()`

## Next Steps

- [Adding Few-Shot Examples](../05-creating-benchmarks/few-shot-examples.md) — How to add examples to questions
- [VerificationConfig Tutorial](../06-running-verification/verification-config.md) — Complete configuration setup
- [Answer Templates](answer-templates.md) — What few-shot examples help with
- [VerificationConfig Reference](../10-configuration-reference/verification-config.md) — Exhaustive field reference
