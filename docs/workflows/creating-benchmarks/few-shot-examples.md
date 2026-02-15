# Adding Few-Shot Examples

Few-shot examples are question-answer pairs that are prepended to a question's prompt during verification, guiding the answering model toward the expected response format, style, and content.

This page covers how to add examples to your benchmark. For configuration options that control *which* examples are used during verification, see [Few-Shot Configuration](../core_concepts/few-shot.md).

---

## Example Format

Each few-shot example is a dictionary with `question` and `answer` keys:

```python
{"question": "What is the approved drug target of Imatinib?", "answer": "BCR-ABL"}
```

During verification, these are formatted into a simple prompt:

```
Question: What is the approved drug target of Imatinib?
Answer: BCR-ABL

Question: What is the approved drug target of Trastuzumab?
Answer: HER2

Question: What is the approved drug target of Venetoclax?
Answer:
```

The answering model sees the examples before generating its response, learning from their format and content.

---

## Adding Examples When Creating Questions

The most common approach is to provide examples when adding a question to the benchmark:

```python
from karenina import Benchmark

benchmark = Benchmark.create(name="Drug Targets")

benchmark.add_question(
    question="What is the approved drug target of Venetoclax?",
    raw_answer="BCL2",
    few_shot_examples=[
        {"question": "What is the approved drug target of Imatinib?", "answer": "BCR-ABL"},
        {"question": "What is the approved drug target of Trastuzumab?", "answer": "HER2"},
        {"question": "What is the approved drug target of Rituximab?", "answer": "CD20"},
    ],
)
```

You can also pass examples via a `Question` object:

```python
from karenina.schemas.entities import Question

q = Question(
    question="How many chromosomes are in a human somatic cell?",
    raw_answer="46",
    few_shot_examples=[
        {"question": "How many autosomal chromosome pairs are in humans?", "answer": "22"},
        {"question": "How many sex chromosomes are in humans?", "answer": "2"},
    ],
)

benchmark.add_question(q)
```

---

## Per-Question Examples

Each question stores its own list of examples. This lets you tailor examples to the type of question being asked:

```python
# Numerical question — examples demonstrate concise numerical answers
benchmark.add_question(
    question="How many subunits does hemoglobin A have?",
    raw_answer="4",
    few_shot_examples=[
        {"question": "How many subunits does RNA polymerase have?", "answer": "5"},
        {"question": "How many catalytic subunits does DNA polymerase III have?", "answer": "3"},
    ],
)

# Nomenclature question — examples demonstrate standard gene symbols
benchmark.add_question(
    question="What gene does Venetoclax target?",
    raw_answer="BCL2",
    few_shot_examples=[
        {"question": "What gene does Imatinib target?", "answer": "BCR-ABL1"},
        {"question": "What gene does Olaparib target?", "answer": "PARP1"},
    ],
)
```

Not every question needs examples. Questions without `few_shot_examples` simply skip few-shot prompting (even when few-shot is enabled in the verification config).

---

## Choosing Good Examples

Examples are most effective when they match the question's domain and expected answer format:

| Guideline | Why |
|-----------|-----|
| Same domain as the question | Models generalize better from related examples |
| Same answer format | Concise examples produce concise answers; verbose examples produce verbose answers |
| Correct answers only | Incorrect examples teach incorrect patterns |
| 2-5 examples per question | Diminishing returns beyond 5; increased token cost |

For instance, if your question expects a gene symbol like "BCL2", provide examples whose answers are also gene symbols — not full sentences.

---

## External Examples

In addition to per-question examples, you can define external examples that are appended during verification. These are configured on `FewShotConfig`, not on individual questions:

- **Global external examples** — appended to every question
- **Question-specific external examples** — appended to a single question

```python
from karenina.schemas import FewShotConfig, QuestionFewShotConfig

config = FewShotConfig(
    global_mode="k-shot",
    global_k=2,
    # Appended to every question during verification
    global_external_examples=[
        {"question": "What is the target of Erlotinib?", "answer": "EGFR"},
    ],
    question_configs={
        "abc123...": QuestionFewShotConfig(
            # Appended only to this question
            external_examples=[
                {"question": "What is the target of Sorafenib?", "answer": "VEGFR2"},
            ],
        ),
    },
)
```

The final example list for a question is resolved in this order: stored examples (selected by mode) → question-specific external → global external.

See [Few-Shot Configuration](../core_concepts/few-shot.md) for the full configuration reference.

---

## Inspecting Examples

You can check what examples are stored on a question:

```python
benchmark = Benchmark.load("my_benchmark.jsonld")

for q_id in benchmark.questions:
    question = benchmark.get_question_as_object(q_id)
    n = len(question.few_shot_examples or [])
    print(f"{q_id}: {n} examples")
```

---

## Saving and Loading

Few-shot examples are persisted in the JSON-LD checkpoint alongside their question. When you save and reload a benchmark, the examples are preserved:

```python
benchmark.save("my_benchmark.jsonld")

reloaded = Benchmark.load("my_benchmark.jsonld")
q = reloaded.get_question_as_object(q_id)
print(q.few_shot_examples)  # Same examples as before
```

---

## Next Steps

- [Few-Shot Configuration](../core_concepts/few-shot.md) — Control which examples are used during verification (modes, k-shot, custom selection)
- [Running Verification](../06-running-verification/index.md) — Execute verification with few-shot enabled
- [VerificationConfig Tutorial](../06-running-verification/verification-config.md) — Complete configuration setup including `few_shot_config`
