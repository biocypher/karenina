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

# Templates vs Rubrics

Karenina evaluates LLM responses using two complementary building blocks: **answer templates** and **rubrics**. Templates check whether the model gave the *correct* answer. Rubrics check whether the model answered *well*. They operate on different inputs, use different evaluation strategies, and answer fundamentally different questions. Understanding the distinction between them is the foundation for effective benchmark design.

```python tags=["hide-cell"]
# Mock cell: ensures examples execute without live API keys.
# This cell is hidden in rendered documentation.
```

## 1. The Core Distinction

| | Answer Templates | Rubrics |
|---|---|---|
| **Question answered** | *Did the model give the right answer?* | *How well did the model answer?* |
| **Evaluates** | Correctness against ground truth | Observable qualities of the response |
| **Operates on** | Parsed, structured data (filled Pydantic schema) | Raw response trace (full text) |
| **Requires ground truth** | Yes (the `self.correct` dictionary) | No (judges by reading the response alone) |
| **Method** | Judge LLM parses response into schema, then `verify()` checks programmatically | Trait evaluators assess the raw text (LLM, regex, callable, or metric) |
| **Output** | Boolean (pass/fail) | Boolean, integer score, or metrics dict |
| **Pipeline stages** | [ParseTemplate](../notebooks/core_concepts/verification-pipeline.ipynb) (stage 7), [VerifyTemplate](../notebooks/core_concepts/verification-pipeline.ipynb) (stage 8) | [RubricEvaluation](../notebooks/core_concepts/verification-pipeline.ipynb) (stage 11) |

In short:

- **Templates** answer: *"Is the extracted information correct?"*
- **Rubrics** answer: *"Does the response have desirable qualities?"*

These are not alternative ways of doing the same thing. They evaluate orthogonal dimensions. A response can pass its template (correct drug target extracted) while failing a rubric trait (unclear reasoning). Conversely, a response can score well on rubric traits (concise, well-cited) while failing its template (wrong answer).

## 2. How Templates Work: Parse, Then Verify

```
                    ┌──────────────┐
                    │  LLM Answer  │
                    │  (free text) │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Judge LLM   │
                    │  parses into │
                    │  Pydantic    │
                    │  schema      │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  verify()    │
                    │  compares to │
                    │  ground truth│
                    └──────┬───────┘
                           │
                     PASS or FAIL
```

A template is a [Pydantic model](../notebooks/core_concepts/answer-templates.ipynb) that defines **what to extract** from the response and **how to check it**. The Judge LLM fills in the schema fields, then the `verify()` method compares them against expected values.

The judge's role varies by field type. With `str` fields, the judge acts as a pure parser: it extracts values that `verify()` then checks. With `bool` fields, the description often encodes the evaluation criterion ("True if TP53 is identified as the most common"), so the judge performs some evaluation during extraction. See [Answer Templates](../notebooks/core_concepts/answer-templates.ipynb) for guidance on this tradeoff.

```python
from pydantic import Field
from karenina.schemas.entities import BaseAnswer


class Answer(BaseAnswer):
    tissue: str = Field(
        description="Tissue where KRAS is most essential"
    )

    def ground_truth(self):
        self.correct = {"tissue": "pancreas"}

    def verify(self) -> bool:
        return self.tissue.strip().lower() == self.correct["tissue"]


# Simulate what happens after the Judge LLM fills in the schema
parsed = Answer(tissue="Pancreas")
print(f"Extracted: {parsed.tissue}")
print(f"Ground truth: {parsed.correct}")
print(f"Verified: {parsed.verify()}")
```

The verification is entirely programmatic: once the Judge LLM fills in the schema, no further LLM calls are needed for the correctness verdict. This makes template-based evaluation **deterministic and reproducible** given the same parsed output.

## 3. How Rubrics Work: Evaluate Traits on the Raw Trace

```
                    ┌──────────────┐
                    │  LLM Answer  │
                    │  (full text) │
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

[Rubrics](rubrics/index.md) evaluate **qualities of the raw response** without parsing it into structured data first. Unlike templates, rubrics never see a filled schema; they work directly on the response text. Four trait classes are available, and `LLMRubricTrait` supports three evaluation kinds:

| Trait Type | Returns | LLM Required | Use Case |
|---|---|---|---|
| **[LLMRubricTrait](rubrics/llm-traits.md)** (boolean) | `bool` | Yes | Binary quality judgment (safety, conciseness) |
| **[LLMRubricTrait](rubrics/llm-traits.md)** (score) | `int` | Yes | Numeric rating within a configurable range |
| **[LLMRubricTrait](rubrics/llm-traits.md)** (literal) | `int` | Yes | Classification into ordered categories (e.g., tone: formal/casual/technical) |
| **[RegexTrait](rubrics/regex-traits.md)** | `bool` | No | Deterministic pattern matching (citations, format compliance) |
| **[CallableTrait](rubrics/callable-traits.md)** | `bool` or `int` | No | Custom Python logic (word count, readability, structure checks) |
| **[MetricRubricTrait](rubrics/metric-traits.md)** | metrics dict | Yes | Precision/recall/F1 over expected content items |

Rubrics can be attached at the **benchmark level** (applied to every question) or the **question level** (applied to one question). When both are present, Karenina merges them; trait names must be unique across scopes. See [Rubrics](rubrics/index.md) for full details on attachment, scoping, and the `higher_is_better` field.

## 4. The Ground-Truth Boundary

The most important idea for choosing between templates and rubrics is the **ground-truth boundary**: whether the evaluation requires knowing the correct answer.

**Templates** live on the ground-truth side. The `verify()` method compares parsed fields against `self.correct`, which encodes what the benchmark author knows to be true. Without ground truth, template verification has nothing to compare against.

**Rubrics** live on the observable side. The evaluator (whether LLM, regex, or callable) judges properties that are visible in the response text itself, without access to the correct answer. The evaluator LLM receives the question and the response trace, but it never sees `raw_answer`, `self.correct`, or the template's ground truth.

**Litmus test**: if the evaluator cannot make the judgment without knowing the correct answer, it belongs in the template. If the evaluator can make the judgment by reading the response alone, it belongs in a rubric.

| Needs ground truth (use a template) | Observable in the response (use a rubric) |
|---|---|
| "Did the response identify BCL2 as the target?" | "Does the response cite specific trials or data?" |
| "Is the mechanism of action accurate?" | "Is the reasoning presented as a linked chain of steps?" |
| "Did the response correctly list all three approved indications?" | "Does the response hedge appropriately on off-label use?" |

Traits about topical relevance, focus, or "answering the right question" almost always require ground truth and should be template checks, not rubric traits. When in doubt, apply the litmus test.

## 5. What Each Evaluator Sees

Understanding what inputs each evaluation path receives clarifies why the two building blocks are complementary, not interchangeable.

| Input | Template (Judge LLM) | Template (`verify()`) | Rubric (evaluator) |
|---|---|---|---|
| Question text | Yes | No | Yes |
| LLM response | Yes | No (sees parsed fields) | Yes (raw trace) |
| JSON schema (field names, types, descriptions) | Yes | N/A | No |
| Parsed field values | N/A | Yes | No |
| Ground truth (`self.correct`) | No | Yes | No |
| `raw_answer` | No | No | No |
| Trait definition (name, description, kind) | N/A | N/A | Yes |

Key observations:

- The Judge LLM never sees ground truth. Its job is schema extraction, not correctness judgment.
- `verify()` never sees the raw response. It only works with the parsed, structured fields.
- Rubric evaluators never see the template, the schema, or the parsed fields. They work on the raw text.
- `raw_answer` is never sent to any LLM. It exists for human review and for deriving `self.correct`.

These information boundaries are deliberate. They make each evaluation path focused and auditable: you can inspect exactly what each component saw when it made its judgment.

## 6. When to Use Each

### Use templates when:

- There is a **definitive correct answer** (factual questions, known values)
- You need to **extract and verify specific values** (gene names, dates, numbers)
- **Precision matters** (e.g., distinguishing "BCL2" from "BCL-XL")
- **Multiple fields** must be jointly extracted and verified

### Use rubrics when:

- Evaluating **response qualities** independent of correctness (clarity, safety, tone)
- **No single correct answer** exists (open-ended questions)
- You want to measure **how** the model answered, not just **what** it answered
- Checking for **patterns** or **format compliance** (citations, disclaimers, word limits)

### Use both together when:

- You need both a correctness verdict **and** quality assessment
- The same response should be evaluated for factual accuracy (template) and for properties like clarity, conciseness, or citation quality (rubric)

### Decision heuristics

| Situation | Use | Reasoning |
|---|---|---|
| "Is the answer X?" | Template | Requires comparing against a known correct value |
| "Is the response safe?" | Rubric (LLM trait) | Observable quality; no ground truth needed |
| "Does the response contain `[N]` citations?" | Rubric (regex trait) | Deterministic pattern match on raw text |
| "Is the response under 150 words?" | Rubric (callable trait) | Programmatic check on raw text |
| "Did the response mention all expected drug interactions?" | Rubric (metric trait) | Precision/recall over a checklist of expected items |
| "Did the response get the drug target right AND explain it clearly?" | Both | Correctness (template) + quality (rubric) |

**Priority heuristic for rubric trait type selection**: prefer regex or callable traits over LLM traits when possible. They are faster, cheaper, and fully reproducible. Use LLM traits when the evaluation genuinely requires language understanding. See the [decision flowchart](rubrics/index.md#decision-flowchart) in the rubrics docs for detailed trait type selection guidance.

## 7. Evaluation Modes

The `evaluation_mode` field on [VerificationConfig](../../reference/configuration/verification-config.md) controls which building blocks are active during a pipeline run:

| Mode | Templates | Rubrics | Pipeline stages active | When to use |
|---|---|---|---|---|
| `template_only` (default) | Yes | No | Stages 1-10, 13 | Pure correctness verification |
| `template_and_rubric` | Yes | Yes | All 13 stages | Correctness + quality assessment |
| `rubric_only` | No | Yes | Stages 1-2, 11-13 (template stages skipped) | Quality-only evaluation; no correct answer needed |

Setting `evaluation_mode` to `template_and_rubric` or `rubric_only` automatically requires `rubric_enabled=True`. These two fields are validated together; a mismatch raises a `ValueError` at configuration time.

For details on configuring evaluation modes, see [Evaluation Modes](../notebooks/core_concepts/evaluation-modes.ipynb).

## 8. Worked Example: Both Together

Consider the question: *"Which is the putative target of venetoclax?"*

**Template** (correctness): extract the drug target and verify it against ground truth.

```python
class VenetoclaxAnswer(BaseAnswer):
    identifies_bcl2_as_target: bool = Field(
        description=(
            "True if the response identifies BCL2 (including Bcl-2, BCL-2, or "
            "B-cell lymphoma 2) as the direct pharmacological target."
        )
    )

    def ground_truth(self):
        self.correct = {"identifies_bcl2_as_target": True}

    def verify(self) -> bool:
        return self.identifies_bcl2_as_target == self.correct["identifies_bcl2_as_target"]


# Simulate what the Judge LLM would produce after parsing a correct response
parsed = VenetoclaxAnswer(identifies_bcl2_as_target=True)
print(f"Template verdict: {'PASS' if parsed.verify() else 'FAIL'}")
```

**Rubric traits** (quality): assess observable properties of the response.

```python
from karenina.schemas.entities.rubric import LLMRubricTrait, RegexTrait

# LLM trait: does the response explain the mechanism?
mechanism_explanation = LLMRubricTrait(
    name="explains_mechanism",
    description=(
        "True if the response explains how the drug interacts with its target "
        "(e.g., BH3 mimetic, inhibition of anti-apoptotic activity). "
        "False if the target is stated without mechanistic context."
    ),
    kind="boolean",
    higher_is_better=True,
)
print(f"LLM trait: {mechanism_explanation.name} (kind={mechanism_explanation.kind})")

# Regex trait: does the response include citations?
has_citations = RegexTrait(
    name="has_citations",
    description="The response includes at least one numbered citation.",
    pattern=r"\[\d+\]",
    higher_is_better=True,
)

# Regex traits can be evaluated locally without an LLM
sample_response = "Venetoclax targets BCL2 [1], acting as a BH3 mimetic [2]."
print(f"Regex trait '{has_citations.name}' on sample: {has_citations.evaluate(sample_response)}")

sample_no_citations = "Venetoclax targets BCL2, acting as a BH3 mimetic."
print(f"Regex trait '{has_citations.name}' on sample without citations: {has_citations.evaluate(sample_no_citations)}")
```

**What happens during evaluation** (using `template_and_rubric` mode):

1. **Stages 1-2**: The pipeline validates the template and generates an answer (or uses a cached response in TaskEval).
2. **Stages 3-6**: Guard stages run (recursion limit, trace validation, optional abstention/sufficiency checks).
3. **Stage 7 (ParseTemplate)**: The Judge LLM receives the question, the response, and the template's JSON schema. It fills in `identifies_bcl2_as_target`.
4. **Stage 8 (VerifyTemplate)**: `verify()` compares the parsed value against `self.correct`. Result: **PASS** or **FAIL**.
5. **Stage 11 (RubricEvaluation)**: Each trait evaluates the raw response independently. The LLM trait assesses whether the mechanism is explained. The regex trait checks for `[N]` citation patterns.
6. **Stage 13 (FinalizeResult)**: All results are combined into a single `VerificationResult`.

The template verdict and rubric scores are independent. A response could correctly identify BCL2 (template passes) but fail to explain the mechanism (LLM trait returns `False`) and include no citations (regex trait returns `False`).

## 9. Next Steps

- [Answer Templates](../notebooks/core_concepts/answer-templates.ipynb): deep dive into template structure, `verify()`, and field types
- [Rubrics](rubrics/index.md): all trait types, attachment scoping, and the `higher_is_better` field
- [Evaluation Modes](../notebooks/core_concepts/evaluation-modes.ipynb): configuring `template_only`, `rubric_only`, and `template_and_rubric`
- [Verification Pipeline](../notebooks/core_concepts/verification-pipeline.ipynb): the 13-stage engine that executes both evaluation paths
- [LLM Evaluation Philosophy](../home/philosophy.md): why Karenina uses LLMs as judges

**Back to**: [Core Concepts](index.md)
