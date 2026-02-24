# Templates vs Rubrics

Karenina evaluates LLM responses using two complementary building blocks: **answer templates** and **rubrics**. Understanding the distinction between them is key to designing effective benchmarks.

## The Core Distinction

| | Answer Templates | Rubrics |
|---|---|---|
| **Question** | *Did the model give the right answer?* | *How well did the model answer?* |
| **Evaluates** | Correctness | Quality |
| **Operates on** | Parsed, structured data (filled schema) | Raw response trace (full text) |
| **Method** | Judge LLM parses response into Pydantic schema, then `verify()` checks against ground truth | Trait evaluators assess qualities of the raw text |
| **Output** | Boolean (pass/fail) | Boolean, integer score, or metrics dict |

In short:

- **Templates** answer: *"Is the extracted information correct?"*
- **Rubrics** answer: *"Does the response have desirable qualities?"*

## How They Work

### Answer Templates: Parse, Then Verify

```
                    ┌──────────────┐
                    │  LLM Answer  │
                    │  (free text) │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Judge LLM   │
                    │  parses into │
                    │ Pydantic     │
                    │ schema       │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  verify()    │
                    │  compares to │
                    │ ground truth │
                    └──────┬───────┘
                           │
                     PASS or FAIL
```

A template is a Pydantic model that defines **what to extract** from the response and **how to check it**. The Judge LLM fills in the schema fields, and the `verify()` method compares them against expected values.

Note that the judge's role varies by field type. With `str` fields, the judge is a pure parser: it extracts values that `verify()` then checks. With `bool` fields, the description often includes the expected answer ("True if TP53 is identified as the most common"), so the judge performs some evaluation during extraction. See [Answer Templates](../notebooks/core_concepts/answer-templates.ipynb) for guidance on this tradeoff.

```python
from pydantic import Field
from karenina.schemas.entities import BaseAnswer

class Answer(BaseAnswer):
    tissue: str = Field(
        description="Tissue where KRAS is most essential"
    )

    def model_post_init(self, __context):
        self.correct = {"tissue": "pancreas"}

    def verify(self) -> bool:
        return self.tissue.strip().lower() == self.correct["tissue"]
```

### Rubrics: Evaluate Traits on the Raw Trace

```
                    ┌──────────────┐
                    │  LLM Answer  │
                    │  (full text)  │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
        ┌─────▼────┐ ┌────▼─────┐ ┌───▼──────┐
        │ LLM Trait│ │Regex     │ │Callable  │
        │ (judge)  │ │Trait     │ │Trait     │
        └─────┬────┘ └────┬─────┘ └───┬──────┘
              │            │            │
           bool/int      bool        bool/int
```

Rubrics evaluate **qualities of the raw response** without parsing it into structured data first. Four trait types are available:

| Trait Type | Returns | LLM Required | Use Case |
|---|---|---|---|
| **LLMRubricTrait** | bool or int | Yes | Subjective assessment (safety, clarity, tone) |
| **RegexTrait** | bool | No | Pattern matching (citations, format compliance) |
| **CallableTrait** | bool or int | No | Custom Python logic (word count, readability) |
| **MetricRubricTrait** | metrics dict | Yes | Precision/recall/F1 over extracted terms |

## When to Use Each

### Use Templates When

- There is a **definitive correct answer** (factual questions, known values)
- You need to **extract and verify specific values** (gene names, dates, numbers)
- **Precision matters** (e.g., distinguishing "BCL2" from "BCL-2")
- **Multiple pieces of information** must be jointly extracted and verified

### Use Rubrics When

- Evaluating **response qualities** independent of correctness (clarity, safety, tone)
- **No single correct answer** exists (open-ended questions)
- You want to measure **how** the model answered, not just **what** it answered
- Checking for **patterns** (citations, disclaimers, format compliance)

### Use Both Together

Templates and rubrics are complementary. A common pattern:

1. **Template**: Verify the model extracted the correct drug target
2. **Rubric traits**: Check that the response was concise, cited sources, and avoided hallucination

This is enabled by the `template_and_rubric` evaluation mode.

## Evaluation Modes

Karenina's three evaluation modes control which building blocks are active:

| Mode | Templates | Rubrics | When to Use |
|---|---|---|---|
| `template_only` | Yes | No | Pure correctness verification (default) |
| `template_and_rubric` | Yes | Yes | Correctness + quality assessment |
| `rubric_only` | No | Yes | Quality-only evaluation (no correct answer needed) |

For details on configuring evaluation modes, see [Evaluation Modes](evaluation-modes.md).

## Conceptual Summary

```
┌─────────────────────────────────────────────────────────┐
│                   LLM Response                          │
│              (natural, free-form text)                   │
└───────────────────────┬─────────────────────────────────┘
                        │
          ┌─────────────┴─────────────┐
          │                           │
   ┌──────▼──────┐            ┌──────▼──────┐
   │  TEMPLATE   │            │   RUBRIC    │
   │             │            │             │
   │ Judge LLM   │            │ Trait       │
   │ parses into │            │ evaluators  │
   │ schema      │            │ assess raw  │
   │      │      │            │ response    │
   │ verify()    │            │             │
   │ checks      │            │ 4 trait     │
   │ correctness │            │ types       │
   └──────┬──────┘            └──────┬──────┘
          │                          │
    Correctness              Quality traits
    (pass/fail)          (bool, score, metrics)
```

---

## Learn More

- [LLM Evaluation Philosophy](../home/philosophy.md) — Why Karenina uses LLMs as judges
- [Answer Templates](../notebooks/core_concepts/answer-templates.ipynb) — Deep dive into template structure, `verify()`, and field types
- [Rubrics Overview](rubrics/index.md) — All four rubric trait types in detail
- [Evaluation Modes](evaluation-modes.md) — How to configure template-only, rubric-only, and combined modes

**Back to**: [Core Concepts](index.md)
