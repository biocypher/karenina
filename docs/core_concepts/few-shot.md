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

# Few-Shot Examples

Few-shot examples teach the **answering model** how to respond by prepending question-answer pairs to the prompt before the main question. They affect only the answering stage of the [verification pipeline](../verification-pipeline/); the [Judge LLM](../answer-templates/), [rubric evaluators](../../../core_concepts/rubrics/), and all other pipeline stages never see them.

```python tags=["hide-cell"]
# Mock cell: ensures examples execute without live API keys.
# This cell is hidden in rendered documentation.
from karenina.schemas import (
    FewShotConfig,
    ModelConfig,
    Question,
    QuestionFewShotConfig,
    VerificationConfig,
)
```

## 1. What Are Few-Shot Examples?

A few-shot example is a `{"question": "...", "answer": "..."}` pair stored on a [question](../questions-and-benchmarks/questions/). When verification runs, Karenina prepends the resolved examples to the prompt so the answering model sees a pattern before it encounters the real question:

```text
Question: How many chromosomes in a human somatic cell?
Answer: 46

Question: How many base pairs in human mitochondrial DNA?
Answer: 16569

Question: How many subunits does hemoglobin A have?
Answer:
```

The answering model receives this as a single user message. The trailing `Answer:` cues the model to continue in the same format.

Few-shot examples are useful when:

- **Responses should follow a specific format**: short gene symbols instead of verbose explanations, numeric values without units, or structured answers
- **Consistency matters**: the model should respond uniformly across related questions
- **The expected style is hard to specify in prose**: showing three good answers communicates the pattern more efficiently than describing it

They are not useful when the answering model already produces responses in the format the template expects, or when the question domain is so varied that examples from one topic would confuse another.

## 2. The Abstraction Boundary

The most important idea: few-shot examples influence **how the answering model responds**, not how Karenina evaluates the response.

| Pipeline component | Sees few-shot examples? | Why |
|--------------------|:-----------------------:|-----|
| **Answering model** (Stage 2: GenerateAnswer) | Yes | Examples are injected into the user message |
| **Judge LLM** (Stage 7: ParseTemplate) | No | Receives the raw response and template schema only |
| **Rubric evaluator** (Stage 11: RubricEvaluation) | No | Evaluates response qualities independently |
| **Template verify()** (Stage 8: VerifyTemplate) | No | Programmatic check on parsed fields |

Few-shot examples are a prompt engineering tool for the model being evaluated. They do not change the evaluation criteria, the parsing instructions, or the rubric definitions.

## 3. How It Works

Few-shot examples flow through three phases: storage, resolution, and injection.

```text
Phase 1: Storage
  Question entity
    └─ few_shot_examples: [{"question": "...", "answer": "..."}, ...]

Phase 2: Resolution (at verification time)
  FewShotConfig.resolve_examples_for_question()
    ├─ Input: available examples from question, question ID, question text
    ├─ Check source (question_pool / global / both / disabled)
    ├─ Apply mode (all / k-shot / custom)
    ├─ Apply exclusions
    └─ Append global examples

Phase 3: Injection (GenerateAnswer stage)
  _construct_few_shot_prompt()
    ├─ Prepend each resolved example as "Question: ...\nAnswer: ..."
    ├─ Append main question as "Question: ...\nAnswer:"
    └─ Send as user message to answering model
```

**Storage** happens at benchmark creation time. You attach examples to individual questions via `benchmark.add_question()`. The examples live on the `Question` entity:

```python
question = Question(
    question="How many subunits does hemoglobin A have?",
    raw_answer="4",
    few_shot_examples=[
        {"question": "How many chromosomes in a human somatic cell?", "answer": "46"},
        {"question": "How many base pairs in human mitochondrial DNA?", "answer": "16569"},
    ],
)
print(f"Stored {len(question.few_shot_examples)} examples on question")
```

**Resolution** happens at verification time. `FewShotConfig` on `VerificationConfig` controls which stored examples are actually used and how many. Resolution is per-question: each question can use a different mode and different k value.

**Injection** happens inside the GenerateAnswer stage. The resolved examples are formatted into a prompt string and sent as the user message. If no examples are resolved (or few-shot is disabled), the question text is sent unmodified.

## 4. The Source Field

`FewShotConfig.source` controls where examples come from. It replaces the old `enabled` toggle with a richer set of options:

| Source | Behavior | Best for |
|--------|----------|----------|
| `"question_pool"` | Use only per-question stored examples | Questions with curated example pools |
| `"global"` | Use only global examples | Uniform examples across all questions |
| `"both"` | Use per-question stored examples and global examples | Combining per-question and shared examples |
| `"disabled"` | No examples used | Establishing a zero-shot baseline |

## 5. The Three Selection Modes

`FewShotConfig.pool_mode` controls which stored examples are selected from the per-question pool. Three modes apply globally; a fourth (`inherit`) is for per-question delegation.

| Mode | Behavior | Best for |
|------|----------|----------|
| `all` | Use every example attached to the question | Small, curated example sets (2 to 5 examples) |
| `k-shot` | Randomly sample *k* examples; uses question ID as seed for reproducibility | Large example pools where using all would be costly |
| `custom` | Select specific examples by index position or MD5 hash | Curated selections where exact control matters |
| `inherit` | Delegate to the global mode and k value | Per-question default; questions without explicit config inherit automatically |

To disable few-shot entirely, set `source="disabled"` instead of using a mode.

### `all` mode (default)

Uses every example attached to the question. This is the default `pool_mode`.

```python
all_config = FewShotConfig(pool_mode="all")
print(f"Mode: {all_config.pool_mode}, source: {all_config.source}")
```

### `k-shot` mode

Randomly samples *k* examples. The question ID seeds the random selection, so the same question always gets the same examples across runs.

```python
k_shot_config = FewShotConfig(pool_mode="k-shot", pool_k=3)
print(f"Mode: {k_shot_config.pool_mode}, k: {k_shot_config.pool_k}")
```

If a question has fewer examples than *k*, all examples are used (no error). The default `pool_k` is `3`.

### `custom` mode

Selects examples by integer index or MD5 hash of the example's question text:

```python
custom_config = FewShotConfig(
    pool_mode="custom",
    question_configs={
        "question_1": QuestionFewShotConfig(
            mode="custom",
            selected_examples=[0, 2],  # indices into the question's example list
        ),
    },
)
print(f"question_1 mode: {custom_config.question_configs['question_1'].mode}")
```

Use `FewShotConfig.generate_example_hash(question_text)` to compute the hash for a given example:

```python
example_hash = FewShotConfig.generate_example_hash("How many chromosomes in a human somatic cell?")
print(f"Hash: {example_hash}")
```

### Disabling few-shot

To disable few-shot entirely (globally or per-question), set `source="disabled"`:

```python
# Globally disabled
none_config = FewShotConfig(source="disabled")

# Disabled for one question via per-question override (source="disabled" at question level)
# For the rest, the global source applies
mixed_config = FewShotConfig(
    pool_mode="all",
    question_configs={
        "question_3": QuestionFewShotConfig(mode="custom", selected_examples=[]),
    },
)
print(f"Source: {none_config.source}")
print(f"question_3 mode: {mixed_config.question_configs['question_3'].mode}")
```

### `inherit` mode

The default for `QuestionFewShotConfig`. When a question's mode is `inherit`, it uses the global mode and global k value. Questions without an entry in `question_configs` implicitly inherit.

```python
default_q = QuestionFewShotConfig()
print(f"Default per-question mode: {default_q.mode}")
```

## 6. Per-Question Overrides

Each question can override the global settings independently. This is how you run different strategies for different parts of the benchmark:

```python
override_config = FewShotConfig(
    pool_mode="k-shot",
    pool_k=3,
    question_configs={
        "question_1": QuestionFewShotConfig(mode="all"),  # use all examples
        "question_2": QuestionFewShotConfig(mode="k-shot", k=5),  # sample 5 instead of 3
        "question_3": QuestionFewShotConfig(mode="custom", selected_examples=[]),  # no examples
        # question_4 inherits: k-shot with k=3
    },
)

# Resolve effective config for each question
for qid in ["question_1", "question_2", "question_3", "question_4"]:
    effective = override_config.get_effective_config(qid)
    print(f"{qid}: mode={effective.mode}, k={effective.k}")
```

You can also exclude specific examples while keeping the mode:

```python
exclusion_config = FewShotConfig(
    pool_mode="all",
    question_configs={
        "question_1": QuestionFewShotConfig(
            mode="all",
            excluded_examples=[2],  # skip the example at index 2
        ),
    },
)
print(f"question_1 exclusions: {exclusion_config.question_configs['question_1'].excluded_examples}")
```

Exclusions accept both integer indices and MD5 hash strings, just like `selected_examples` in custom mode.

## 7. Resolution in Action

`FewShotConfig.resolve_examples_for_question()` takes the stored examples and applies the mode logic. Here is a complete resolution example:

```python
# A question with 5 stored examples
examples = [
    {"question": "What is the gene symbol for tumor protein p53?", "answer": "TP53"},
    {"question": "What is the gene symbol for epidermal growth factor receptor?", "answer": "EGFR"},
    {"question": "What is the gene symbol for vascular endothelial growth factor A?", "answer": "VEGFA"},
    {"question": "What is the gene symbol for breast cancer type 1?", "answer": "BRCA1"},
    {"question": "What is the gene symbol for programmed death-ligand 1?", "answer": "CD274"},
]

# Resolve with "all" mode
config_all = FewShotConfig(pool_mode="all")
resolved = config_all.resolve_examples_for_question("q1", available_examples=examples)
print(f"all mode: {len(resolved)} examples")

# Resolve with k-shot (k=2)
config_k = FewShotConfig(pool_mode="k-shot", pool_k=2)
resolved_k = config_k.resolve_examples_for_question("q1", available_examples=examples)
print(f"k-shot (k=2): {len(resolved_k)} examples -> {[e['answer'] for e in resolved_k]}")

# Same question ID always gives same selection
resolved_k2 = config_k.resolve_examples_for_question("q1", available_examples=examples)
print(f"Reproducible: {resolved_k == resolved_k2}")

# Resolve with source="disabled"
config_none = FewShotConfig(source="disabled")
resolved_none = config_none.resolve_examples_for_question("q1", available_examples=examples)
print(f"disabled source: {len(resolved_none)} examples")
```

## 8. Global Examples

Global examples are question-answer pairs that are appended to every question's resolved examples. They are defined at the `FewShotConfig` level and appear after the per-question stored examples.

```python
global_config = FewShotConfig(
    pool_mode="k-shot",
    pool_k=2,
    global_examples=[
        {"question": "What is the capital of France?", "answer": "Paris"},
    ],
)

resolved = global_config.resolve_examples_for_question("q1", available_examples=examples)
print(f"Total examples for q1: {len(resolved)}")
print(f"Last one (global): {resolved[-1]['answer']}")
```

The resolution order is: resolved stored examples, then global examples.

## 9. Convenience Constructors

`FewShotConfig` provides class methods for common setup patterns:

```python
# Custom selections by index
by_index = FewShotConfig.from_index_selections(
    {
        "question_1": [0, 2, 4],
        "question_2": [1, 3],
    }
)
print(f"from_index_selections: pool_mode={by_index.pool_mode}, questions={list(by_index.question_configs.keys())}")

# Custom selections by hash
by_hash = FewShotConfig.from_hash_selections(
    {
        "question_1": [FewShotConfig.generate_example_hash("example question text")],
    }
)
print(f"from_hash_selections: pool_mode={by_hash.pool_mode}")

# Different k per question
per_q_k = FewShotConfig.k_shot_for_questions(
    {
        "question_1": 5,
        "question_2": 2,
    },
    pool_k=3,
)
print(f"k_shot_for_questions: pool_mode={per_q_k.pool_mode}, pool_k={per_q_k.pool_k}")
```

`from_index_selections` and `from_hash_selections` default to `pool_mode="custom"`; `k_shot_for_questions` defaults to `pool_mode="k-shot"`.

## 10. Using FewShotConfig in Verification

Pass the config to `VerificationConfig` via the `few_shot_config` field:

```python
config = VerificationConfig(
    answering_models=[
        ModelConfig(id="answerer", model_provider="anthropic", model_name="claude-haiku-4-5"),
    ],
    parsing_models=[
        ModelConfig(id="judge", model_provider="anthropic", model_name="claude-haiku-4-5"),
    ],
    few_shot_config=FewShotConfig(
        pool_mode="k-shot",
        pool_k=3,
    ),
)

print(f"Few-shot enabled: {config.is_few_shot_enabled()}")
print(f"Config: {config.get_few_shot_config()}")
```

When `few_shot_config` is `None` (the default) or `source="disabled"`, the pipeline sends the question text directly to the answering model with no examples prepended.

```python
config_no_fs = VerificationConfig(
    answering_models=[
        ModelConfig(id="answerer", model_provider="anthropic", model_name="claude-haiku-4-5"),
    ],
    parsing_models=[
        ModelConfig(id="judge", model_provider="anthropic", model_name="claude-haiku-4-5"),
    ],
)
print(f"Few-shot enabled (no config): {config_no_fs.is_few_shot_enabled()}")
```

### Validation

`VerificationConfig` validates few-shot settings at construction time:

- In `k-shot` mode, `pool_k` must be at least 1
- Per-question `k` values must be at least 1 when specified
- `validate_selections()` checks that index and hash selections reference valid examples (call this explicitly after construction if you want bounds checking)

```python
# validate_selections returns a list of error messages (empty if valid)
fs_config = FewShotConfig(
    pool_mode="custom",
    question_configs={
        "q1": QuestionFewShotConfig(mode="custom", selected_examples=[0, 1, 99]),
    },
)
errors = fs_config.validate_selections({"q1": examples[:3]})
print(f"Validation errors: {errors}")
```

## 11. Choosing the Right Mode

| Situation | Recommended mode | Rationale |
|-----------|-----------------|-----------|
| 2 to 5 hand-picked examples per question | `all` | Small set; include everything |
| 10+ examples per question, cost-sensitive | `k-shot` | Limits prompt length while preserving reproducibility |
| Ablation study: which examples help most? | `custom` | Exact control over which examples appear |
| Baseline comparison without examples | `source="disabled"` | Clean zero-shot baseline |
| Most questions share a strategy, a few differ | Per-question overrides with `inherit` default | Global mode covers the common case; overrides handle exceptions |

**Litmus test for whether few-shot helps**: if the answering model already produces responses that your template can parse and verify, few-shot examples add cost without benefit. Try a zero-shot run first. Add examples when you observe format mismatches, verbose responses where concise ones are expected, or inconsistent answer styles across similar questions.

## 12. FewShotConfig Field Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `source` | `Literal["disabled", "question_pool", "global", "both"]` | `"both"` | Controls where examples come from: per-question pool, global list, both, or disabled entirely |
| `pool_mode` | `Literal["all", "k-shot", "custom"]` | `"all"` | Default selection mode for per-question example pools |
| `pool_k` | `int` | `3` | Number of examples for `k-shot` mode |
| `question_configs` | `dict[str, QuestionFewShotConfig]` | `{}` | Per-question overrides keyed by question ID |
| `global_examples` | `list[dict[str, str]]` | `[]` | Examples appended to every question |

### QuestionFewShotConfig Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | `Literal["all", "k-shot", "custom", "inherit"]` | `"inherit"` | Selection mode for this question |
| `k` | `int \| None` | `None` | Override global k for this question (k-shot mode) |
| `selected_examples` | `list[str \| int] \| None` | `None` | Indices or MD5 hashes for custom mode |
| `excluded_examples` | `list[str \| int] \| None` | `None` | Indices or hashes of examples to exclude |

## 13. Provenance in Results

When few-shot is active, the verification result records provenance metadata so you can trace which results used few-shot prompting:

| Field | Location | Description |
|-------|----------|-------------|
| `few_shot_enabled` | `result.metadata` | `True` if few-shot was active for this verification |
| `few_shot_example_count` | `result.metadata` | Number of resolved examples (0 if disabled) |

These fields enable filtering and grouping in analysis (e.g., comparing accuracy with and without few-shot examples across the same benchmark).

## 14. Next Steps

- [Answer Templates](../answer-templates/): what few-shot examples help the answering model produce
- [Verification Pipeline](../verification-pipeline/): where GenerateAnswer fits in the 13-stage pipeline (with sub-stages 7a/7b and 11a/11b plus the always-on placeholder-retry guard)
- [Running Verification](../../../workflows/running-verification/): complete configuration and execution workflow
- [VerificationConfig Reference](../../../reference/configuration/verification-config/): exhaustive field reference including `few_shot_config`
