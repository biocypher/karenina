---
jupyter:
  jupytext:
    formats: docs/core_concepts/questions-and-benchmarks//md,docs/notebooks/core_concepts/questions-and-benchmarks//ipynb
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

# Questions: The Heart of Evaluation

A **Question object** is the fundamental building block and minimal unit of evaluation in Karenina. These objects are the primary units that compose [Benchmarks](benchmarks.md), which act as packages for organizing, versioning, and persisting collections of questions.

It is important to distinguish between the **Question object** (the container) and its **`question` attribute** (the literal text prompt sent to the model).

While the prompt is just one part of the package, a complete Question object anchors the entire [evaluation loop](../verification-pipeline.md)—connecting what you ask an LLM, what you expect as an answer, and how that answer is eventually verified.

Think of a Question object as a self-contained package that carries:
1.  **The Prompt (`question`)**: Exactly what the model sees.
2.  **The Reference (`raw_answer`)**: What the human author knows to be true.
3.  **The Verification (`answer_template`)**: The machine-readable logic required for evaluation (see [Answer Templates](../answer-templates.md)).
4.  **The Rubric (`question_rubric`)**: Optional question-specific traits to augment benchmark-level quality checks (see [Rubrics](../rubrics/index.md)).
5.  **The Context**: Metadata like keywords, sources, and authorship for organization and audit trails.

```python tags=["hide-cell"]
# Mock cell: ensures examples execute without live API keys.
# This cell is hidden in rendered documentation.
import hashlib
from unittest.mock import MagicMock, patch

mock_modules = {}
for mod in ["sqlalchemy", "sqlalchemy.orm", "sqlalchemy.ext", "sqlalchemy.ext.declarative"]:
    mock_modules[mod] = MagicMock()

with patch.dict("sys.modules", mock_modules):
    from karenina.benchmark import Benchmark
    from karenina.schemas.entities import BaseAnswer
    from karenina.schemas.entities.question import Question
    from pydantic import Field
```

```python
# Standard imports for working with questions
from karenina.schemas.entities.question import Question
from karenina.schemas.entities import BaseAnswer
from karenina.benchmark import Benchmark
```

## Anatomy of a Question

At its simplest, a question bridges the gap between a raw prompt and a verifiable result. It only requires the text to send to the model and a human-readable reference answer.

```python
q = Question(
    question="What is the putative target of venetoclax?",
    raw_answer="BCL2 (B-cell lymphoma 2)",
    keywords=["pharmacology", "oncology"],
)

print(f"Question:   {q.question}")
print(f"Raw answer: {q.raw_answer}")
print(f"Keywords:   {q.keywords}")
print(f"ID:         {q.id}")
```

### The Prompt (`question`)
The `question` field is the literal text sent to the model being evaluated. Its behavior in the pipeline depends on the stage:

*   **Answering Stage (Bare Control)**: By default, Karenina sends the `question` as a bare user message. This stage is deliberately unmanaged to ensure the benchmark author has total control. Any background info, domain context, or specific formatting instructions must be included in the `question` text itself. The only framework-level additions are optional: a system prompt from the [`ModelConfig`](../../reference/configuration/model-config.md) or prepended [Few-Shot examples](../few-shot.md).
*   **Parsing & Rubric Stages (Contextual Reference)**: The same `question` text is also sent to the [Judge LLM](../verification-pipeline.md) during parsing and to [rubric evaluators](../rubrics/index.md) when assessing response quality. Unlike the answering stage, these "evaluation" stages are managed by the framework using specialized system prompts and [instruction builders](../prompt-assembly.md) that provide the necessary context for parsing and judgment.

### The Reference (`raw_answer`)
The `raw_answer` is your human-readable "Source of Truth." **It is never sent to the answering LLM and never sent to the Judge LLM.**

Its role is purely programmatic and organizational:
1.  **Template Generation**: It is the primary input for the automatic template generator, which derives structured ground truth and verification logic from it.
2.  **Reference**: It is stored in results and exported in reports so human reviewers can see the expected answer at a glance.
3.  **Ground Truth**: It serves as the source from which you derive the `self.correct` dictionary in your [Answer Template](../answer-templates.md).

### The Answer Template (`answer_template`)
While the `raw_answer` is for humans, the `answer_template` is for the machine. It is a string containing the Python code for a `BaseAnswer` subclass (always named `Answer`).

This code defines:
1.  **The Schema**: What fields the Judge LLM should extract from the model's response.
2.  **The Verification**: The `verify()` method that programmatically compares extracted values against the ground truth.

Every question in a benchmark can have its own unique `answer_template`, allowing you to mix different types of questions (boolean, numeric, text extraction) in the same evaluation run. For a deep dive into writing these, see [Answer Templates](../answer-templates.md).

### The Rubric (`question_rubric`)
While rubrics are typically defined at the benchmark level to evaluate all questions consistently, an individual question can carry its own `question_rubric`. These question-specific traits augment the global benchmark rubric, allowing you to add targeted quality checks (e.g., verifying a specific tone or format constraint) that only apply to this particular prompt. For more details on defining traits, see [Rubrics](../rubrics/index.md).

## A Question's Journey

When you run a benchmark, each question follows a predictable path through the [Verification Pipeline](../verification-pipeline.md):

1.  **Generation**: The `question` text is sent to the answering LLM to produce a response.
2.  **Parsing**: A "Judge" LLM receives a package of context: the original `question`, the model's response, the template's JSON schema, and internal parsing instructions. It extracts specific data points from the response into the structured schema.
3.  **Verification**: The extracted data is checked against the programmatic "Ground Truth" (derived from your `raw_answer`) using the template's `verify()` logic.
4.  **Rubric Evaluation**: If enabled, the `question` is passed to rubric evaluators along with the model's response to assess qualities like safety, conciseness, or citation style.
5.  **Finalization**: The result—Pass or Fail, along with any rubric scores—is saved alongside the question's metadata.

## Managing the Lifecycle

### The `finished` Flag: "Are we ready?"

The `finished` flag determines whether a question enters the [verification pipeline](../verification-pipeline.md). It is a property of the question's membership in a specific benchmark, not an intrinsic property of the question itself (see [Evaluation Modes](../evaluation-modes.md)).

*   **Python API**: Defaults to `True`. Questions added via `add_question()` are assumed to be complete.
*   **GUI**: Defaults to `False`. This ensures you review and "finish" questions before they enter a production verification run.

*Note: If your verification run returns zero results, it almost always means your questions are marked `finished=False`.*

### Deterministic IDs: Content-Addressable Identity

Every question is assigned a unique identity (`id`) that acts as a **content-addressable fingerprint**. This ID is computed as a deterministic **MD5 hash of the `question` string** (UTF-8 encoded), resulting in a 32-character hexadecimal string.

This approach ensures that "the same question" always carries the same identity across different benchmarks or evaluation runs, enabling stable historical tracking.

```python
# The ID is deterministic: same text produces same ID
q1 = Question(question="What is the capital of France?", raw_answer="Paris")
q2 = Question(question="What is the capital of France?", raw_answer="Paris")
print(f"q1.id: {q1.id}")
print(f"q2.id: {q2.id}")
print(f"Same:  {q1.id == q2.id}")

# Under the hood: MD5 of the UTF-8 encoded question text
import hashlib
manual_id = hashlib.md5("What is the capital of France?".encode("utf-8")).hexdigest()
print(f"Manual hash: {manual_id}")
print(f"Matches:     {manual_id == q1.id}")
```

#### Key ID Behaviors:
*   **Prompt-Exclusive**: The ID is derived **only** from the `question` text.
*   **Metadata Independent**: Modifying the `raw_answer`, `answer_template`, `keywords`, or `author` does **not** change the ID. You can refine your evaluation logic without losing the question's historical identity.
*   **Case & Whitespace Sensitive**: "What is BCL2?" and "What is bcl2?" (or adding a trailing space) will produce completely different IDs.

<div class="admonition warning">
<p class="admonition-title">Changing text changes the ID</p>
<p>If you modify a question's text, its fingerprint changes. This breaks historical cross-references in your results. If you need to fix a typo while preserving an existing ID, you can pass a custom <code>question_id</code> when adding the question to a benchmark to override the automatic hashing.</p>
</div>

## `raw_answer` vs Template `ground_truth`

While related, they serve different audiences:

| Concept | Where It Lives | Audience | Purpose |
| :--- | :--- | :--- | :--- |
| **`raw_answer`** | `Question.raw_answer` | Humans / Authors | A plain-language description of the correct answer. |
| **`self.correct`** | `Answer.ground_truth()` | `verify()` method | A structured dictionary of values for programmatic comparison. |

```python
# Example: raw_answer vs structured ground_truth
class Answer(BaseAnswer):
    target: str = Field(description="The protein target.")

    def ground_truth(self):
        # Derived from raw_answer: "BCL2 (B-cell lymphoma 2)"
        self.correct = {"target": "BCL2"}

    def verify(self) -> bool:
        return self.target.strip().upper() == self.correct["target"]
```

## Detailed Reference: Metadata and Special Fields

### Metadata for Organization and Context
*   **`answer_notes`**: A free-text field for edge cases ("Accept 'Bcl2' but not 'BCL-XL'"), reasoning, or reviewer instructions. The automatic [template generator](../../workflows/creating-benchmarks/scaled-authoring.md) uses these notes to produce more robust `verify()` methods.
*   **`keywords`**: Pure metadata for filtering and grouping. They never reach an LLM and have no effect on pipeline execution (see [Benchmark Operations](../../workflows/creating-benchmarks/benchmark-operations.md)).
*   **`author` & `sources`**: Provenance tracking (who wrote it, where the answer came from). Preserved in [checkpoints](checkpoints.md) and exports for audit trails.
*   **`custom_metadata`**: An open dictionary for any domain-specific attributes (e.g., "difficulty": "hard").

### Special Pipeline Fields
*   **`few_shot_examples`**: A list of example question-answer pairs. When enabled in [VerificationConfig](../../reference/configuration/verification-config.md), these are prepended to the question to guide the answering model's format or level of detail. They are not sent to the Judge during parsing (see [Few-Shot](../few-shot.md)).
*   **`answer_template`**: The Python code (subclass of `BaseAnswer`) that defines the parsing schema and `verify()` logic for this specific question.
*   **`question_rubric`**: Question-specific rubric traits that augment the benchmark-level rubric.

### Field Reference Table

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `question` | `str` | Required | The prompt text sent to the LLM. |
| `raw_answer` | `str` | Required | Human-readable reference answer (metadata). |
| `keywords` | `list[str]` | `[]` | Keywords for filtering (formerly `tags`). |
| `id` | `str` | Auto | Deterministic MD5 hash of the question text. |
| `date_created` | `str` | Now (ISO) | Creation timestamp. |
| `date_modified` | `str` | Now (ISO) | Last modification timestamp. |
| `answer_notes` | `str \| None` | `None` | Metadata for edge cases or reasoning. |
| `author` | `dict \| None` | `None` | Author information. |
| `sources` | `list[dict] \| None` | `None` | Source documents or references. |
| `answer_template` | `str \| None` | `None` | Required for evaluation; defines parsing and verification logic. |
| `question_rubric` | `dict \| None` | `None` | Question-specific rubric traits. |
| `custom_metadata` | `dict \| None` | `None` | Arbitrary key-value pairs. |
## Next Steps

*   [Benchmarks](benchmarks.md): How questions are grouped into packages.
*   [Answer Templates](../answer-templates.md): Writing the `verify()` logic for your questions.
*   [Evaluation Modes](../evaluation-modes.md): How `finished` status and templates drive the pipeline.
*   [Benchmark Operations](../../workflows/creating-benchmarks/benchmark-operations.md): Adding, searching, and managing questions at scale.
