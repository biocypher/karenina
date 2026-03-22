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

A **Question object** is the fundamental building block and minimal unit of evaluation in Karenina. These objects are the primary units that compose [Benchmarks](../benchmarks/), which act as packages for organizing, versioning, and persisting collections of questions.

It is important to distinguish between the **Question object** (the container) and its **`question` attribute** (the literal text prompt sent to the model).

While the prompt is just one part of the package, a complete Question object anchors the entire [evaluation loop](../../verification-pipeline/)—connecting what you ask an LLM, what you expect as an answer, and how that answer is eventually verified.

Think of a Question object as a self-contained package that carries:
1.  **The Prompt (`question`)**: Exactly what the model sees.
2.  **The Reference (`raw_answer`)**: What the human author knows to be true.
3.  **The Verification (`answer_template`)**: The machine-readable logic required for evaluation (see [Answer Templates](../../answer-templates/)).
4.  **The Rubric (`question_rubric`)**: Optional question-specific traits to augment benchmark-level quality checks (see [Rubrics](../../../../core_concepts/rubrics/)).
5.  **The Context**: Metadata like keywords, sources, and authorship for organization and audit trails.
6.  **The Workspace (`workspace_path`)**: For [agentic tasks](../agentic-evaluation/), an optional relative path to the directory containing starter code, tests, or data files that the answering agent will operate on.

```python tags=["hide-cell"]
# Mock cell: ensures examples execute without live API keys.
# This cell is hidden in rendered documentation.
import hashlib
from unittest.mock import MagicMock, patch

mock_modules = {}
for mod in [
    "sqlalchemy", "sqlalchemy.orm", "sqlalchemy.ext",
    "sqlalchemy.ext.declarative", "sqlalchemy.engine",
    "sqlalchemy.sql", "sqlalchemy.event",
    "karenina.storage", "karenina.storage.base",
    "karenina.storage.engine", "karenina.storage.db_config",
    "karenina.storage.models", "karenina.storage.generated_models",
    "karenina.storage.auto_mapper", "karenina.storage.operations",
]:
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

## 1. Anatomy of a Question

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

### 1.1. The Prompt (`question`)
The `question` field is the literal text sent to the model being evaluated. Its behavior in the pipeline depends on the stage:

*   **Answering Stage (Bare Control)**: By default, Karenina sends the `question` as a bare user message. This stage is deliberately unmanaged to ensure the benchmark author has total control. Any background info, domain context, or specific formatting instructions must be included in the `question` text itself. The only framework-level additions are optional: a system prompt from the [`ModelConfig`](../../../../reference/configuration/model-config/) or prepended [Few-Shot examples](../../few-shot/).
*   **Parsing & Rubric Stages (Contextual Reference)**: The same `question` text is also sent to the [Judge LLM](../../verification-pipeline/) during parsing and to [rubric evaluators](../../../../core_concepts/rubrics/) when assessing response quality. Unlike the answering stage, these "evaluation" stages are managed by the framework using specialized system prompts and [instruction builders](../../prompt-assembly/) that provide the necessary context for parsing and judgment.

### 1.2. The Reference (`raw_answer`)
The `raw_answer` is your human-readable "Source of Truth." **It is never sent to the answering LLM and never sent to the Judge LLM.**

Its role is purely programmatic and organizational:
1.  **Template Generation**: It is the primary input for the automatic template generator, which derives structured ground truth and verification logic from it.
2.  **Reference**: It is stored in results and exported in reports so human reviewers can see the expected answer at a glance.
3.  **Ground Truth**: It serves as the source from which you derive the `self.correct` dictionary in your [Answer Template](../../answer-templates/).

### 1.3. The Answer Template (`answer_template`)
While the `raw_answer` is for humans, the `answer_template` is for the machine. It is a string containing the Python code for a `BaseAnswer` subclass (always named `Answer`).

This code defines:
1.  **The Schema**: What fields the Judge LLM should extract from the model's response.
2.  **The Verification**: The `verify()` method that programmatically compares extracted values against the ground truth.

Every question in a benchmark can have its own unique `answer_template`, allowing you to mix different types of questions (boolean, numeric, text extraction) in the same evaluation run. For a deep dive into writing these, see [Answer Templates](../../answer-templates/).

### 1.4. The Rubric (`question_rubric`)
While rubrics are typically defined at the benchmark level to evaluate all questions consistently, an individual question can carry its own `question_rubric`. These question-specific traits augment the global benchmark rubric, allowing you to add targeted quality checks (e.g., verifying a specific tone or format constraint) that only apply to this particular prompt. For more details on defining traits, see [Rubrics](../../../../core_concepts/rubrics/).

## 2. A Question's Journey

When you run a benchmark, each question follows a predictable path through the [Verification Pipeline](../../verification-pipeline/):

1.  **Generation**: The `question` text is sent to the answering LLM to produce a response. For [agentic tasks](../agentic-evaluation/), the answering model can run as an agent with tool access in the question's [workspace](#23-workspace-path-for-agentic-tasks), reading files, writing code, and executing scripts.
2.  **Parsing**: A "Judge" LLM receives a package of context: the original `question`, the model's response, the template's JSON schema, and internal parsing instructions. It extracts specific data points from the response into the structured schema. When [agentic parsing](../agentic-evaluation/#4-two-step-agentic-judging-stage-7b) is enabled separately, the judge instead runs as an agent that independently inspects workspace artifacts before extraction. Agentic answering and agentic parsing are independently configurable: an agentic answering model can be paired with a classical parser, or vice versa.
3.  **Verification**: The extracted data is checked against the programmatic "Ground Truth" (derived from your `raw_answer`) using the template's `verify()` logic.
4.  **Rubric Evaluation**: If enabled, the `question` is passed to rubric evaluators along with the model's response to assess qualities like safety, conciseness, or citation style.
5.  **Finalization**: The result, Pass or Fail along with any rubric scores, is saved alongside the question's metadata.

### 2.3. Workspace Path for Agentic Tasks

For coding and data analysis tasks evaluated with the [agentic workflow](../agentic-evaluation/), a question can specify a `workspace_path`: a relative path from the benchmark's `workspace_root` to the directory containing starter code, test files, or data for that question.

```python
# workspace_path resolves to workspace_root / "task_01"
q = Question(
    question="Fix the bug in calculator.py so that division by zero returns an error message.",
    raw_answer="Division by zero handled with try/except",
    workspace_path="task_01",
)
```

When `workspace_path` is set, the pipeline expects a pre-existing directory at `workspace_root / workspace_path`. When it is not set, the pipeline creates an empty directory. The `workspace_root` is set on the [Benchmark](../benchmarks/) (it is where files live on disk), while `workspace_path` is per-question.

By default, the pipeline copies each question's workspace to a sibling working directory before the agent runs, so the original files are preserved for re-runs. This behavior is controlled by the `workspace_copy` setting on [VerificationConfig](../../../../reference/configuration/verification-config/).

`workspace_path` is persisted in [checkpoints](../../../../core_concepts/questions-and-benchmarks/checkpoints/) as a relative path. The `workspace_root` is not persisted (it is a local filesystem path). When loading a checkpoint on a different machine, supply the new workspace root via `Benchmark.load()`.

## 3. Managing the Lifecycle

### 3.1. The `finished` Flag: "Are we ready?"

The `finished` flag determines whether a question enters the [verification pipeline](../../verification-pipeline/). It is a property of the question's membership in a specific benchmark, not an intrinsic property of the question itself (see [Evaluation Modes](../../evaluation-modes/)).

*   **Python API**: Defaults to `True`. Questions added via `add_question()` are assumed to be complete.
*   **GUI**: Defaults to `False`. This ensures you review and "finish" questions before they enter a production verification run.

*Note: If your verification run returns zero results, it almost always means your questions are marked `finished=False`.*

### 3.2. Deterministic IDs: Content-Addressable Identity

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

#### 3.2.1. Key ID Behaviors:
*   **Prompt-Exclusive**: The ID is derived **only** from the `question` text.
*   **Metadata Independent**: Modifying the `raw_answer`, `answer_template`, `keywords`, or `author` does **not** change the ID. You can refine your evaluation logic without losing the question's historical identity.
*   **Case & Whitespace Sensitive**: "What is BCL2?" and "What is bcl2?" (or adding a trailing space) will produce completely different IDs.

<div class="admonition warning">
<p class="admonition-title">Changing text changes the ID</p>
<p>If you modify a question's text, its fingerprint changes. This breaks historical cross-references in your results. If you need to fix a typo while preserving an existing ID, you can pass a custom <code>question_id</code> when adding the question to a benchmark to override the automatic hashing.</p>
</div>

## 4. `raw_answer` vs Template `ground_truth`

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

## 5. Detailed Reference: Metadata and Special Fields

### 5.1. Metadata for Organization and Context
*   **`answer_notes`**: A free-text field for edge cases ("Accept 'Bcl2' but not 'BCL-XL'"), reasoning, or reviewer instructions. The automatic [template generator](../../../creating-benchmarks/scaled-authoring/) uses these notes to produce more robust `verify()` methods.
*   **`keywords`**: Pure metadata for filtering and grouping. They never reach an LLM and have no effect on pipeline execution (see [Benchmark Operations](../../../../workflows/creating-benchmarks/benchmark-operations/)).
*   **`author` & `sources`**: Provenance tracking (who wrote it, where the answer came from). Preserved in [checkpoints](../../../../core_concepts/questions-and-benchmarks/checkpoints/) and exports for audit trails.
*   **`custom_metadata`**: An open dictionary for any domain-specific attributes (e.g., "difficulty": "hard").

### 5.2. Special Pipeline Fields
*   **`few_shot_examples`**: A list of example question-answer pairs. When enabled in [VerificationConfig](../../../../reference/configuration/verification-config/), these are prepended to the question to guide the answering model's format or level of detail. They are not sent to the Judge during parsing (see [Few-Shot](../../few-shot/)).
*   **`answer_template`**: The Python code (subclass of `BaseAnswer`) that defines the parsing schema and `verify()` logic for this specific question.
*   **`question_rubric`**: Question-specific rubric traits that augment the benchmark-level rubric.
*   **`workspace_path`**: For [agentic tasks](../agentic-evaluation/), the relative path from the benchmark's `workspace_root` to this question's workspace directory. See [Section 2.3](#23-workspace-path-for-agentic-tasks).

### 5.3. Field Reference Table

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
| `workspace_path` | `str \| None` | `None` | Relative path from `workspace_root` to this question's workspace directory (for [agentic tasks](../agentic-evaluation/)). |
| `question_rubric` | `dict \| None` | `None` | Question-specific rubric traits. |
| `custom_metadata` | `dict \| None` | `None` | Arbitrary key-value pairs. |

**Strict schema enforcement:** `Question` uses `extra="forbid"`, so any unrecognized fields in the input data will raise a `ValidationError`. This catches typos and prevents silent data loss from misspelled field names.

**Legacy `tags` key:** For backward compatibility, the legacy `tags` key is accepted during construction and automatically converted to `keywords`. If both `tags` and `keywords` are present, `keywords` takes precedence and `tags` is discarded.

## 6. Next Steps

*   [Benchmarks](../benchmarks/): How questions are grouped into packages.
*   [Answer Templates](../../answer-templates/): Writing the `verify()` logic for your questions.
*   [Evaluation Modes](../../evaluation-modes/): How `finished` status and templates drive the pipeline.
*   [Agentic Evaluation](../agentic-evaluation/): Workspace-based evaluation for coding and data analysis tasks.
*   [Benchmark Operations](../../../../workflows/creating-benchmarks/benchmark-operations/): Adding, searching, and managing questions at scale.
