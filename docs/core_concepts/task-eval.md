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

# TaskEval

TaskEval evaluates any free text output using karenina's two evaluation primitives: [templates](answer-templates.md) for correctness and [rubrics](rubrics/index.md) for quality.

```python tags=["hide-cell"]
# Mock cell: ensures examples execute without live API keys.
# This cell is hidden in rendered documentation.
import sys
from unittest.mock import MagicMock
from pydantic import Field
from karenina.schemas.entities import BaseAnswer
from karenina.schemas.entities.rubric import LLMRubricTrait, Rubric

_storage_mock = MagicMock()
for _mod in [
    "karenina.storage",
    "karenina.storage.generated_models",
    "karenina.storage.auto_mapper",
    "karenina.storage.operations",
    "karenina.storage.models",
]:
    sys.modules[_mod] = _storage_mock

from karenina.benchmark.task_eval import TaskEval
from karenina.ports.messages import Message

class Answer(BaseAnswer):
    identifies_bcl2: bool = Field(
        description="True if BCL2 is identified as the primary target."
    )
    def ground_truth(self):
        self.correct = {"identifies_bcl2": True}
    def verify(self) -> bool:
        return self.identifies_bcl2 == self.correct["identifies_bcl2"]

def _mock_add_template(self, template_class, step_id=None):
    self.add_question(
        {"id": template_class.__name__.lower(), "question": "", "raw_answer": "", "answer_template": "mock"},
        step_id=step_id,
    )

TaskEval.add_template = _mock_add_template

task = TaskEval(task_id="my-eval")
```

## What Is TaskEval?

[Benchmark](questions-and-benchmarks/index.md) is a closed-loop workflow: it defines questions, generates responses through the [verification pipeline](verification-pipeline.md), and evaluates them together. TaskEval is the open-loop counterpart. You supply any free text output and evaluate it using karenina's primitives (templates and rubrics), with no question definition or answer generation required.

Use TaskEval whenever you have text that needs structured evaluation:

- **Agent workflows** that produce multi-step traces you want to score for correctness or quality
- **CI pipelines** that capture outputs from production runs and need post-hoc quality checks
- **Any free text** from external systems, user submissions, or one-off experiments that you want to evaluate with karenina's judge LLM machinery

## How TaskEval Works

TaskEval is a recording and evaluation container. The text it evaluates is always produced by external code: an agent, a pipeline, a human annotator. TaskEval never generates responses. Instead, it provides methods to record that text, attach evaluation criteria to it, and run evaluation through karenina's judge LLM.

The workflow has three phases.

### Phase 1: Record

External code produces text. You record it into a TaskEval instance using `log()` for plain text or `log_trace()` for structured conversation traces. Each call appends to the TaskEval's log store.

If your process has distinct phases, you can assign each log entry to a named step via the `step_id` parameter. For example, an agent workflow with a retrieval phase and a synthesis phase can record outputs separately under `step_id="retrieval"` and `step_id="synthesis"`. Logs without a `step_id` go to the global scope.

### Phase 2: Attach evaluation criteria

Once text is recorded, you attach evaluation criteria: [templates](answer-templates.md) for correctness, [rubrics](rubrics/index.md) for quality, or both.

A template is a structured schema that tells a judge LLM what to extract from the text, paired with a `verify()` method that checks the extracted values against ground truth. A rubric defines qualitative dimensions (conciseness, citation quality, logical coherence) that a judge LLM scores independently from factual correctness. Each rubric trait describes one dimension.

Criteria can be attached globally (evaluated against all recorded text) or to a specific step (evaluated against only that step's logs).

### Phase 3: Evaluate

Calling `evaluate()` sends the recorded text through karenina's [verification pipeline](verification-pipeline.md). The pipeline's judge LLM parses the text into any attached templates, runs `verify()`, and scores each rubric trait. No questions to define, no answers to generate: the text you recorded is the response, and the criteria you attached define what to check.

### Putting it together

Suppose an agent workflow looked up a cancer drug and produced this output:

> *"The primary pharmacological target of venetoclax is BCL2 (B-cell lymphoma 2), a protein that inhibits apoptosis. Venetoclax binds selectively to BCL2, displacing pro-apoptotic proteins and triggering programmed cell death in cancer cells [1]."*

You want to evaluate this along two dimensions. Did the response identify BCL2 as the primary target? This is a correctness check, expressed as a [template](answer-templates.md) with a boolean field and a `verify()` method. Is the response concise and well cited? These are quality checks, expressed as [rubric](rubrics/index.md) traits.

You would record this text with `log()`, attach a template for the BCL2 check and a rubric for conciseness and citation quality, then call `evaluate()`. The pipeline handles the rest.

See the [workflow page](../notebooks/task-eval/index.ipynb) for the complete code walkthrough.

### Object structure

A TaskEval instance holds two parallel scopes: global and per-step. Each scope stores logs (the text to evaluate), templates (correctness criteria), and rubrics (quality criteria).

```python
from karenina.benchmark.task_eval import TaskEval

task = TaskEval(
    task_id="agent-drug-target",
    metadata={"model": "claude-haiku-4-5"},
)
```

```
┌───────────────────────────────────────────────────────┐
│                      TaskEval                         │
│                                                       │
│  task_id: "agent-drug-lookup"                         │
│  metadata: {"model": "claude-sonnet-4"}               │
│  merge_strategy: "concatenate"                        │
│                                                       │
│  ┌─────────────────────────────────────────────────┐  │
│  │ Global Scope                                    │  │
│  │                                                 │  │
│  │  logs        ← log(), log_trace()               │  │
│  │  templates   ← add_template(Answer)             │  │
│  │  rubrics     ← add_rubric(Rubric)               │  │
│  └─────────────────────────────────────────────────┘  │
│                                                       │
│  ┌──────────────────┐  ┌──────────────────┐           │
│  │ Step: retrieval  │  │ Step: synthesis  │  ...      │
│  │                  │  │                  │           │
│  │  logs            │  │  logs            │           │
│  │  templates       │  │  templates       │           │
│  │  rubrics         │  │  rubrics         │           │
│  └──────────────────┘  └──────────────────┘           │
└───────────────────────────────────────────────────────┘
```

The global scope evaluates all logs together. Per-step scopes evaluate only the logs, templates, and rubrics attached to that step. When you call `evaluate()` without a `step_id`, TaskEval runs global evaluation first, then auto-evaluates each step that has data.

## Benchmark vs TaskEval

| Dimension | Benchmark | TaskEval |
|-----------|-----------|----------|
| Workflow | Closed-loop: generate + evaluate | Open-loop: evaluate only |
| Starting point | Questions (define what to ask) | Traces (record what happened) |
| Answer generation | Pipeline generates responses | You supply pre-recorded outputs |
| Question required | Yes | No |
| Best for | Controlled model comparison | Evaluating any free text output |

## Attaching Evaluation Criteria

Pass a `BaseAnswer` subclass via `add_template()` for correctness checks, and a `Rubric` via `add_rubric()` for quality traits. The [evaluation mode](evaluation-modes.md) is auto-detected from what you attach: `template_only`, `rubric_only`, or `template_and_rubric`. Both can be scoped globally or to a specific step via the `step_id` parameter.

```python
# Correctness: judge LLM parses the response into Answer, then verify() runs
task.add_template(Answer)

# Quality: each trait is scored independently by the judge LLM
task.add_rubric(Rubric(llm_traits=[
    LLMRubricTrait(name="conciseness", kind="boolean",
                   description="True if the response is direct and avoids unnecessary elaboration.")
]))
```

See the [workflow](../notebooks/task-eval/index.ipynb#step-3-attach-evaluation-criteria) for complete examples including rubric trait types and step-scoped criteria.

## Logging

Use `log()` for plain text and `log_trace()` for structured `Message` traces from `karenina.ports.messages`. Both accept `step_id` to group output under a named step and `target` to control where the log is stored (`"global"`, `"step"`, or `"both"`).

```python
# Plain text
task.log("BCL2 is the primary pharmacological target of venetoclax.")
task.log("Found 3 relevant papers.", step_id="retrieval")

# Structured trace
task.log_trace([
    Message.user("What is the target of venetoclax?"),
    Message.assistant("The primary pharmacological target of venetoclax is BCL2."),
])
```

See the [workflow](../notebooks/task-eval/index.ipynb#step-2-log-outputs) for complete parameter reference, tool call traces, and scoping examples.

## Merge Strategies

When evaluation runs, TaskEval combines logged outputs into a single response for the pipeline. Two strategies control this merging:

**`concatenate`** (default): Combines text and trace logs into a single response. Text logs are wrapped as `Message.assistant()` objects; trace messages are included as-is. All messages are serialized together.

**`traces_only`**: Uses only logs that contain structured `Message` traces. Text-only logs are ignored. Use this when your text logs are debugging output that should not be part of the evaluated response.

The strategy is set at construction time via `merge_strategy` and can be overridden per `evaluate()` call.

## Step-Specific Evaluation

TaskEval supports both global and step-level scoping for logs, questions, and rubrics. This enables per-phase analysis of multi-step agent workflows.

When you call `evaluate()` without a `step_id`, TaskEval runs global evaluation and then auto-evaluates all steps that have logs. Results are available in `result.global_eval` and `result.per_step`. Pass `step_id` to evaluate only one step's logs and criteria.

See the [workflow](../notebooks/task-eval/index.ipynb#step-4-configure-and-evaluate) for step-scoped evaluation examples.
