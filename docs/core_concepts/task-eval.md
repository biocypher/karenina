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

TaskEval applies Karenina's evaluation engine ([templates](../answer-templates/) for correctness, [rubrics](../../../core_concepts/rubrics/) for quality) to text you supply, without generating responses.

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

## 1. What Is TaskEval?

Karenina has two evaluation workflows. [Benchmark](../../../core_concepts/questions-and-benchmarks/) is a closed-loop workflow: it defines questions, sends them to an answering model through the [verification pipeline](../verification-pipeline/), and evaluates the generated responses. TaskEval is the open-loop counterpart. You supply pre-existing text (an agent trace, a pipeline output, a human-written response) and evaluate it using the same pipeline, the same templates, the same rubrics, and the same judge LLM. No question definition or answer generation is required.

**The abstraction boundary**: TaskEval handles recording text, attaching evaluation criteria, and routing evaluation through the verification pipeline. It does not generate responses, manage questions, or provide persistence. It is a stateless evaluation container: you populate it, evaluate, and read the results.

Use TaskEval whenever you have text that already exists and needs structured evaluation:

- **Agent workflows** that produce multi-step traces you want to score for correctness or quality
- **CI pipelines** that capture outputs from production runs and need post-hoc quality checks
- **One-off experiments** where you want to evaluate a response against a template or rubric without building a full benchmark

## 2. Why TaskEval Exists

Many evaluation scenarios start with text that already exists. A coding agent produced a solution. A RAG pipeline returned a synthesis. A production system logged a response. In all these cases, the text is already produced; what you need is a way to evaluate it.

Benchmark assumes it controls the full loop: it sends a question to a model, captures the response, and evaluates it. If your text was produced by a different system, at a different time, or through a process Benchmark cannot orchestrate (a multi-step agent, a human annotator, an external API), you need a way to feed that text directly into the evaluation engine.

The most important idea is that **TaskEval reuses the same verification pipeline as Benchmark**. Internally, it injects your text as `cached_answer_data`, bypassing the answer generation stage while running every subsequent stage identically: template validation, judge parsing, `verify()`, rubric evaluation, abstention detection, and metric computation. The evaluation quality is the same; only the source of the text being evaluated differs.

```
Benchmark mode:                  TaskEval mode:
  Questions + Answering Model      Pre-existing text
          │                              │
          ▼                              ▼
  Generate Answer (Stage 2)        cached_answer_data
          │                              │
          └───────────┬──────────────────┘
                      ▼
            Stages 3-13 (identical)
            Parse → Verify → Rubric
                      │
                      ▼
               VerificationResult
```

## 3. How It Works

TaskEval is a recording and evaluation container with a three-phase lifecycle.

### 3.1 Phase 1: Record

External code produces text. You record it into a TaskEval instance using `log()` for plain strings or `log_trace()` for structured conversation traces (lists of `Message` objects from `karenina.ports.messages`). Each call appends to the TaskEval's internal log store.

If your process has distinct phases, you can assign each log entry to a named **step** via the `step_id` parameter. An agent workflow with a retrieval phase and a synthesis phase can record outputs separately under `step_id="retrieval"` and `step_id="synthesis"`. Logs without a `step_id` go to the global scope.

### 3.2 Phase 2: Attach Evaluation Criteria

Once text is recorded, you attach evaluation criteria: [templates](../answer-templates/) for correctness, [rubrics](../../../core_concepts/rubrics/) for quality, or both.

A template is a structured Pydantic schema that tells the judge LLM what to extract from the text, paired with a `verify()` method that checks the extracted values against ground truth. A rubric defines qualitative dimensions (conciseness, citation quality, logical coherence) that the judge LLM scores independently from factual correctness.

Criteria can be attached **globally** (evaluated against all recorded text) or to a **specific step** (evaluated against only that step's logs). The [evaluation mode](../evaluation-modes/) is auto-detected from what you attach: `template_only`, `rubric_only`, or `template_and_rubric`.

### 3.3 Phase 3: Evaluate

Calling `evaluate()` sends the recorded text through Karenina's [verification pipeline](../verification-pipeline/). The pipeline's judge LLM parses the text into any attached templates, runs `verify()`, and scores each rubric trait. TaskEval returns a `TaskEvalResult` containing verification outcomes for both global and per-step scopes.

### 3.4 Walkthrough

Suppose an agent workflow looked up a cancer drug and produced this output:

> *"The primary pharmacological target of venetoclax is BCL2 (B-cell lymphoma 2), a protein that inhibits apoptosis. Venetoclax binds selectively to BCL2, displacing pro-apoptotic proteins and triggering programmed cell death in cancer cells [1]."*

You want to evaluate this along two dimensions:

1. **Correctness**: Did the response identify BCL2 as the primary target? This is a template check: a boolean field (`identifies_bcl2`) and a `verify()` method that compares it against ground truth.
2. **Quality**: Is the response concise and well cited? These are rubric traits, scored by the judge LLM.

You would record this text with `log()`, attach a template for the BCL2 check and a rubric for conciseness, then call `evaluate()`. The verification pipeline handles judge parsing, `verify()`, and trait scoring. The [workflow page](../../task-eval/) shows the complete code for this scenario.

## 4. Benchmark vs TaskEval

| Dimension | Benchmark | TaskEval |
|-----------|-----------|----------|
| Workflow | Closed-loop: generate + evaluate | Open-loop: evaluate only |
| Starting point | Questions (define what to ask) | Text (record what happened) |
| Answer generation | Pipeline generates responses | You supply pre-recorded text |
| Question definition | Required | Not required |
| Pipeline stages used | All 13 stages (with sub-stages 7a/7b and 11a/11b plus the always-on placeholder-retry guard) | Same pipeline; generation stage bypassed |
| Result type | `VerificationResult` (per question/model) | `TaskEvalResult` (global + per-step) |
| Best for | Controlled model comparison | Evaluating existing outputs |

**When to use which**:

- **Use Benchmark** when you control the question-and-answer loop: you have questions, you want to send them to one or more models, and you want to compare the results.
- **Use TaskEval** when the text already exists: an agent produced a trace, a pipeline captured an output, or you are working with outputs from a system Benchmark cannot orchestrate.

A useful litmus test: if you need `answering_models` in your `VerificationConfig`, you need Benchmark. If you only need `parsing_models` (with `parsing_only=True`), TaskEval is the right tool.

## 5. Object Structure

A TaskEval instance holds two parallel scopes: **global** and **per-step**. Each scope stores logs (the text to evaluate), templates (correctness criteria), and rubrics (quality criteria).

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
│  task_id: "agent-drug-target"                         │
│  metadata: {"model": "claude-haiku-4-5"}              │
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

The global scope evaluates all logs together. Per-step scopes evaluate only the logs, templates, and rubrics attached to that step. When you call `evaluate()` without a `step_id`, TaskEval runs global evaluation first, then automatically evaluates each step that has data.

The constructor accepts two additional parameters: `merge_strategy` (see [Merge Strategies](#8-merge-strategies)) and `callable_registry`, a `dict[str, Callable]` for pre-registering functions used by [callable rubric traits](../rubrics/callable-traits/).

## 6. Attaching Evaluation Criteria

Pass a `BaseAnswer` subclass via `add_template()` for correctness checks, and a `Rubric` via `add_rubric()` for quality traits. Both can be scoped globally or to a specific step via the `step_id` parameter.

The evaluation mode is auto-detected from what you attach:

| You attach | Detected mode | Pipeline behavior |
|------------|---------------|-------------------|
| Template only | `template_only` | Judge parses response into template; `verify()` runs |
| Rubric only | `rubric_only` | A synthetic pass-through template is created internally; rubric traits are scored |
| Both | `template_and_rubric` | Full evaluation: template parsing, `verify()`, and rubric scoring |

```python
# Correctness: judge LLM parses the response into Answer, then verify() runs
task.add_template(Answer)

# Quality: each trait is scored independently by the judge LLM
task.add_rubric(Rubric(llm_traits=[
    LLMRubricTrait(name="conciseness", kind="boolean",
                   description="True if the response is direct and avoids unnecessary elaboration.")
]))
```

<div class="admonition note">
<p class="admonition-title"><code>add_template()</code> requires inspectable source</p>
<p><code>add_template()</code> extracts the source code of your <code>BaseAnswer</code> subclass using <code>inspect.getsource()</code>. The class must be defined in a <code>.py</code> file or Jupyter notebook, not constructed dynamically at runtime. For dynamically defined templates, use <code>add_question()</code> with template source code passed as a string in the <code>answer_template</code> field.</p>
</div>

If your rubric includes [callable traits](../rubrics/callable-traits/), register the callable functions before calling `evaluate()`:

```python
task.register_callable("under_150_words", lambda text: len(text.split()) < 150)
```

When multiple rubrics are attached to the same scope, TaskEval merges them into a single rubric. Trait names must be unique across all rubrics in the same scope; duplicate names raise a `ValueError`.

See the [workflow](../../task-eval/#step-3-attach-evaluation-criteria) for complete examples including all trait types and step-scoped criteria.

## 7. Logging

Use `log()` for plain text and `log_trace()` for structured `Message` traces from `karenina.ports.messages`. Both accept `step_id` to scope output to a named step and `target` to control where the log is stored.

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

The `target` parameter controls routing when a `step_id` is provided:

| `target` value | Global scope | Step scope | Use when |
|----------------|-------------|------------|----------|
| `"both"` (default) | Yes | Yes (requires `step_id`) | The log is relevant to both overall and step-specific evaluation |
| `"global"` | Yes | No | The `step_id` is for bookkeeping, but the text should only appear in global evaluation |
| `"step"` | No | Yes (requires `step_id`) | The text is relevant only to this step, not the overall evaluation |

Without a `step_id`, only the global scope receives the log regardless of the `target` value.

`log_trace()` also accepts a plain string, which is automatically wrapped as a single `Message.assistant()` object. When structured traces contain tool calls, the full message structure (including tool use and tool results) is preserved for the pipeline.

See the [workflow](../../task-eval/#step-2-log-outputs) for complete parameter reference and tool call trace examples.

## 8. Merge Strategies

When evaluation runs, TaskEval combines logged outputs into a single response for the verification pipeline. The `merge_strategy` parameter controls this merging:

| Strategy | Behavior | Use when |
|----------|----------|----------|
| `"concatenate"` (default) | Text logs are wrapped as `Message.assistant()` objects; trace messages are included as-is. All messages are serialized together into a single response string. | You want all logged output evaluated together |
| `"traces_only"` | Only logs that contain structured `Message` traces are used. Text-only logs are ignored. | Your text logs are debugging output or metadata that should not be part of the evaluated response |

Set the strategy at construction time or override it per `evaluate()` call via `task.evaluate(config, merge_strategy="traces_only")`.

## 9. Step-Specific Evaluation

TaskEval supports both global and step-level scoping for logs, templates, and rubrics. This enables per-phase analysis of multi-step agent workflows.

When you call `evaluate()` without a `step_id`, TaskEval:

1. Runs global evaluation against all global-scope logs and criteria
2. Automatically evaluates every step that has logs, templates, or rubrics defined

Results are structured as:

- `result.global_eval`: a `StepEval` containing global verification results
- `result.per_step`: a `dict[str, StepEval]` mapping step IDs to their evaluation results

To evaluate only one step, pass `step_id` to `evaluate()`.

See the [workflow](../../task-eval/#step-4-configure-and-evaluate) for step-scoped evaluation examples.

## 10. Result Objects

`evaluate()` returns a `TaskEvalResult` that wraps all evaluation outcomes. Each scope (global or per-step) is represented by a `StepEval`, which stores `VerificationResult` objects keyed by question ID.

| Method on `TaskEvalResult` | Returns |
|----------------------------|---------|
| `summary()` | One-line summary: "2/3 template verifications passed \| 4/5 rubric traits passed" |
| `summary_compact()` | Compact one-liner with task ID and success rate |
| `display()` | Formatted multi-line report with all verification details |
| `export_json()` | Full results serialized as a JSON string |
| `export_markdown()` | Results formatted as Markdown |

`StepEval` provides `get_summary_stats()` for aggregate statistics (traces passed, template verification counts, rubric trait counts, success rate) and `aggregate_rubric_results()` for averaging rubric scores across replicates. For replicate support, set `replicate_count` on your `VerificationConfig`.

## 11. The `taskeval` Interface

TaskEval is wired through a sentinel value of `ModelConfig.interface`: `"taskeval"`. When the verification pipeline encounters this interface, it routes around answer generation entirely and consumes the pre-collected text supplied via `log()` / `log_trace()` instead.

The interface is registered on `AdapterRegistry` with `llm_factory=None`, `parser_factory=None`, and `agent_factory=None`, plus `requires_provider=False`. There is no LLM, parser, or agent backing it: this interface is a routing signal, not an adapter that calls a model.

```python
from karenina.schemas.config import ModelConfig

answering_model = ModelConfig(
    id="taskeval-answerer",
    model_name="recorded-output",   # arbitrary sentinel name
    interface="taskeval",
)
```

`model_name` is required by `ModelConfig` but its value is treated as a label (it identifies the recorded output set, not a live model). `model_provider` is not required because `requires_provider=False`. Parsing and rubric models in the same `VerificationConfig` still use a real adapter (typically `langchain` or `claude_tool`) for judge calls.

For the registry-side details and the full interface matrix, see [Available Adapters](../../../advanced-adapters/available-adapters/#taskeval-taskeval-no-op-interface).

## 12. Next Steps

- [TaskEval workflow](../../task-eval/): step-by-step guide with complete code
- [Answer templates](../answer-templates/): writing templates for correctness evaluation
- [Rubrics](../../../core_concepts/rubrics/): defining quality traits
- [Verification pipeline](../verification-pipeline/): the 13-stage pipeline (with sub-stages 7a/7b and 11a/11b plus the always-on placeholder-retry guard) TaskEval routes through
- [Evaluation modes](../evaluation-modes/): `template_only`, `rubric_only`, `template_and_rubric`
- [Available adapters](../../../advanced-adapters/available-adapters/): the full interface matrix, including the `taskeval` row
