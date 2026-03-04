# Evaluating with TaskEval

This workflow walks through evaluating free text outputs using TaskEval. You log any text or structured traces, attach evaluation criteria (templates, rubrics, or both), run the judge LLM, and inspect results. No question definition or answer generation is needed. For the underlying concepts, see [TaskEval](../../core_concepts/task-eval.md).

## Overview

```
Log outputs → Attach criteria → Evaluate → Inspect results
```

TaskEval evaluates any free text using karenina's two primitives: templates for correctness and rubrics for quality. You supply the outputs; karenina's judge LLM machinery handles the rest.

---

## Step 1: Create a TaskEval Instance

```python
from karenina.benchmark.task_eval import TaskEval

task = TaskEval(
    task_id="agent-drug-target",
    metadata={"model": "claude-sonnet-4", "run_date": "2025-02-15"},
)
```

**Constructor parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_id` | `str \| None` | `None` | Identifier for tracking |
| `metadata` | `dict \| None` | `None` | Arbitrary metadata dict |
| `merge_strategy` | `"concatenate" \| "traces_only"` | `"concatenate"` | How to combine logs before evaluation |
| `callable_registry` | `dict[str, Callable]` | `None` | Registry for callable trait evaluation |

---

## Step 2: Log Outputs

TaskEval accepts two kinds of logged output: plain text and structured `Message` traces.

### Text Logging

Use `log()` for simple text outputs. Each call records the text as potential answer content.

```python
task.log("The primary pharmacological target of venetoclax is BCL2.")
```

With step scoping:

```python
task.log("Found 3 relevant papers on venetoclax.", step_id="retrieval", target="both")
task.log("BCL2 is the direct target.", step_id="synthesis", target="both")
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | (required) | Text content to log |
| `step_id` | `str \| None` | `None` | Step identifier for grouping |
| `target` | `"global" \| "step" \| "both"` | `"both"` | Where to store the log |
| `level` | `"debug" \| "info" \| "warn" \| "error"` | `"info"` | Log level |
| `tags` | `list[str] \| None` | `None` | Optional tags |

### Trace Logging

Use `log_trace()` for structured conversation traces built from `Message` objects (`karenina.ports.messages`).

```python
from karenina.ports.messages import Message

task.log_trace([
    Message.user("What is the target of venetoclax?"),
    Message.assistant("The primary pharmacological target of venetoclax is BCL2."),
])
```

#### Tool Call Traces

For agent workflows with tool use, include `ToolUseContent` blocks and `Message.tool_result()`:

```python
from karenina.ports.messages import Message, ToolUseContent

task.log_trace([
    Message.user("What is the target of venetoclax?"),
    Message.assistant("Let me search the database.", tool_calls=[
        ToolUseContent(id="call_1", name="search_db", input={"query": "venetoclax target"})
    ]),
    Message.tool_result(tool_use_id="call_1", content="BCL2 (B-cell lymphoma 2)"),
    Message.assistant("Based on the search results, the primary pharmacological target of venetoclax is BCL2 (B-cell lymphoma 2)."),
])
```

**Parameters**: Same as `log()`, except the first argument is `messages: list[Message] | str`. If a plain `str` is passed, it is automatically wrapped as `Message.assistant(text)`.

---

## Step 3: Attach Evaluation Criteria

Attach templates, rubrics, or both. TaskEval auto-detects the [evaluation mode](../../core_concepts/evaluation-modes.md) (`template_only`, `rubric_only`, or `template_and_rubric`) based on what you provide. No question definition is needed.

### Templates (correctness)

Pass a `BaseAnswer` subclass directly via `add_template()`. The template's fields tell the judge LLM what to extract; `verify()` checks correctness.

```python
from pydantic import Field
from karenina.schemas.entities import BaseAnswer

class Answer(BaseAnswer):
    identifies_bcl2: bool = Field(
        description="True if the response identifies BCL2 as the primary target of venetoclax."
    )

    def ground_truth(self):
        self.correct = {"identifies_bcl2": True}

    def verify(self) -> bool:
        return self.identifies_bcl2 == self.correct["identifies_bcl2"]

task.add_template(Answer)
```

### Rubrics (quality)

Attach a rubric via `add_rubric()`. Rubric evaluation uses the same infrastructure as Benchmark: all trait types (LLM, regex, callable, metric) and all evaluation options (batch vs sequential strategy, full-trace vs last-message scope) are available.

```python
from karenina.schemas.entities.rubric import LLMRubricTrait, RegexTrait, Rubric

rubric = Rubric(
    llm_traits=[
        LLMRubricTrait(
            name="conciseness",
            kind="boolean",
            description="True if the response answers directly without unnecessary elaboration.",
        ),
    ],
    regex_traits=[
        RegexTrait(
            name="has_citations",
            pattern=r"\[\d+\]",
            description="Response includes numbered citations in bracket notation.",
        ),
    ],
)
task.add_rubric(rubric)
```

### Both Together

Attach a template and rubric to the same TaskEval for combined correctness and quality evaluation:

```python
task.add_template(Answer)
task.add_rubric(Rubric(llm_traits=[
    LLMRubricTrait(
        name="conciseness",
        kind="boolean",
        description="True if the response answers the question directly without unnecessary elaboration.",
    ),
]))
```

---

## Step 4: Configure and Evaluate

Create a `VerificationConfig` with `parsing_only=True` (no answering model needed) and call `evaluate()`.

```python
from karenina.schemas.config.models import ModelConfig
from karenina.schemas.verification.config import VerificationConfig

config = VerificationConfig(
    parsing_models=[ModelConfig(id="haiku", model_provider="anthropic", model_name="claude-haiku-4-5")],
    parsing_only=True,
)
result = task.evaluate(config)
```

### Override Merge Strategy

You can override the instance default per evaluation call:

```python
result = task.evaluate(config, merge_strategy="traces_only")
```

### Evaluate a Specific Step

Pass `step_id` to evaluate only one step's logs and criteria:

```python
result = task.evaluate(config, step_id="retrieval")
```

Without `step_id`, TaskEval runs global evaluation and auto-evaluates all steps that have logs.

---

## Step 5: Inspect Results

`evaluate()` returns a `TaskEvalResult` containing global and per-step results.

### Result Structure

```python
result.task_id        # str | None
result.metadata       # dict
result.global_eval    # StepEval | None
result.per_step       # dict[str, StepEval]
result.timestamp      # str (ISO format)
```

### StepEval

Each `StepEval` contains `verification_results`: a dict mapping question IDs to lists of `VerificationResult` (one per replicate).

```python
step = result.global_eval

# Summary statistics
stats = step.get_summary_stats()
# {"traces_total": 1, "traces_passed": 1, "template_verification_passed": 1, ...}

# Formatted output
print(step.format_verification_results())
```

### Accessing VerificationResult Fields

Each `VerificationResult` uses nested sub-objects:

```python
for qid, vr_list in step.verification_results.items():
    vr = vr_list[0]  # first replicate

    # Template results
    vr.template.verify_result       # bool | None
    vr.template.raw_llm_response    # str

    # Rubric results
    vr.rubric.llm_trait_scores      # dict[str, int | bool]
    vr.rubric.regex_trait_scores    # dict[str, bool]

    # Metadata
    vr.metadata.completed_without_errors  # bool
    vr.metadata.question_id               # str
```

### Display and Export

```python
# Human-readable display
print(result.display())

# One-line summary
print(result.summary())
# "1/1 template verifications passed | 2/2 rubric traits passed"

# JSON export
json_str = result.export_json()

# Markdown export
md_str = result.export_markdown()
```

---

## Complete Example

A full agent-with-tools evaluation combining trace logging, template verification, and rubric evaluation.

```python
from karenina.benchmark.task_eval import TaskEval
from karenina.ports.messages import Message, ToolUseContent
from karenina.schemas.entities import BaseAnswer
from karenina.schemas.entities.rubric import LLMRubricTrait, RegexTrait, Rubric
from karenina.schemas.config.models import ModelConfig
from karenina.schemas.verification.config import VerificationConfig
from pydantic import Field

# 1. Define template
class Answer(BaseAnswer):
    identifies_bcl2: bool = Field(
        description="True if the response identifies BCL2 as the primary target."
    )
    def ground_truth(self):
        self.correct = {"identifies_bcl2": True}
    def verify(self) -> bool:
        return self.identifies_bcl2 == self.correct["identifies_bcl2"]

# 2. Create TaskEval and log agent trace
task = TaskEval(task_id="agent-drug-lookup", metadata={"model": "claude-sonnet-4"})
task.log_trace([
    Message.user("What is the primary pharmacological target of venetoclax?"),
    Message.assistant("I'll search the pharmacology database.", tool_calls=[
        ToolUseContent(id="call_1", name="pharma_search", input={"drug": "venetoclax"})
    ]),
    Message.tool_result(tool_use_id="call_1", content="Venetoclax targets BCL2 [1]."),
    Message.assistant(
        "The primary pharmacological target of venetoclax is BCL2 "
        "(B-cell lymphoma 2), a protein that regulates apoptosis [1]."
    ),
])

# 3. Attach template (correctness) and rubric (quality)
task.add_template(Answer)
task.add_rubric(Rubric(
    llm_traits=[
        LLMRubricTrait(
            name="conciseness",
            kind="boolean",
            description="True if the response answers directly without unnecessary elaboration.",
        ),
    ],
    regex_traits=[
        RegexTrait(
            name="has_citations",
            pattern=r"\[\d+\]",
            description="Response includes numbered citations.",
        ),
    ],
))

# 4. Evaluate
config = VerificationConfig(
    parsing_models=[ModelConfig(id="haiku", model_provider="anthropic", model_name="claude-haiku-4-5")],
    parsing_only=True,
)
result = task.evaluate(config)

# 5. Inspect results
print(result.summary())
stats = result.global_eval.get_summary_stats()
print(f"Template pass rate: {stats['template_verification_passed']}/{stats['template_verification_total']}")
print(f"Rubric pass rate: {stats['rubric_traits_passed']}/{stats['rubric_traits_total']}")
```

---

## Next Steps

- [TaskEval concept page](../../core_concepts/task-eval.md): Merge strategies, pipeline integration, and step-level scoping in depth
- [Answer Templates](../../core_concepts/answer-templates.md): Writing `verify()` methods and field descriptions
- [Rubrics](../../core_concepts/rubrics/index.md): All five trait types (LLM, regex, callable, metric, literal)
- [Verification Pipeline](../../core_concepts/verification-pipeline.md): The 13-stage engine that TaskEval feeds into
