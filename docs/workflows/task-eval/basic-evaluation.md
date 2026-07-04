---
jupyter:
  jupytext:
    formats: docs/workflows/task-eval//md,docs/notebooks/task-eval//ipynb
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

# Basic TaskEval Evaluation

This tutorial walks through evaluating free text outputs using TaskEval. You log any text or structured traces, attach evaluation criteria (templates, rubrics, or both), run the judge LLM, and inspect results. No question definition or answer generation is needed. For the underlying concepts, see [TaskEval](../../core_concepts/task-eval.md).

**What you'll learn:**

- Create a `TaskEval` instance with tracking metadata
- Log text output with `log()` and structured traces with `log_trace()`
- Attach a correctness template with `add_template()`
- Attach a quality rubric with `add_rubric()`
- Configure `VerificationConfig` with `parsing_only=True`
- Run evaluation and inspect `TaskEvalResult`
- Use `summary()`, `display()`, `get_summary_stats()`, and field-level access
- Export results to JSON and Markdown

```python tags=["hide-cell"]
# Mock cell: replays captured LLM responses from docs/data/workflow-taskeval/ so
# the workflow executes without live API keys. The full pipeline logic runs;
# only the raw model calls are mocked.
# This cell is hidden in the rendered documentation.
import hashlib
import json
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

# Resolve fixtures directory (works from notebook, markdown, and repo root CWDs)
_FIXTURES_DIR = None
for _candidate in [
    Path("data/workflow-taskeval"),
    Path("../data/workflow-taskeval"),
    Path("../../data/workflow-taskeval"),
    Path("docs/data/workflow-taskeval"),
]:
    if _candidate.is_dir():
        _FIXTURES_DIR = _candidate
        break
assert _FIXTURES_DIR is not None, "Could not find data/workflow-taskeval fixtures directory"

# Load fixtures indexed by prompt hash for order-independent matching
_fixtures_by_hash: dict[str, dict] = {}
for _p in _FIXTURES_DIR.glob("*.json"):
    _data = json.loads(_p.read_text())
    _fixtures_by_hash[_data["prompt_hash"]] = _data


def _hash_messages(messages) -> str:
    """Compute the same hash used during capture for fixture matching."""
    normalized = []
    for msg in messages:
        content = msg.content if hasattr(msg, "content") else str(msg)
        msg_type = msg.type if hasattr(msg, "type") else "unknown"
        if isinstance(content, str):
            normalized.append(f"{msg_type}:{content}")
        elif isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    text_parts.append(str(block.get("text", block.get("input", ""))))
                else:
                    text_parts.append(str(block))
            normalized.append(f"{msg_type}:{'|'.join(text_parts)}")
    return hashlib.sha256("|".join(normalized).encode()).hexdigest()[:16]


# Save original ainvoke for restoration
_original_ainvoke = BaseChatModel.ainvoke


async def _replaying_ainvoke(self, input, config=None, **kwargs):
    """Return the captured LLM response matching this request's prompt hash."""
    messages = input if isinstance(input, list) else [input]
    prompt_hash = _hash_messages(messages)
    fixture = _fixtures_by_hash.get(prompt_hash)
    if fixture is None:
        raise ValueError(f"No fixture for prompt hash {prompt_hash}")
    resp = fixture["response"]
    return AIMessage(
        content=resp["content"],
        id=resp.get("id", "fixture"),
        tool_calls=resp.get("tool_calls", []),
        response_metadata=resp.get("response_metadata", {}),
        usage_metadata=resp.get("usage_metadata"),
    )


BaseChatModel.ainvoke = _replaying_ainvoke

# Patch add_template to handle nbconvert execution where inspect.getsource()
# cannot retrieve source code from cell-defined classes.
from karenina.benchmark.task_eval import TaskEval as _TaskEval

_ANSWER_TEMPLATE_CODE = """\
from karenina.schemas.entities import BaseAnswer
from pydantic import Field
from typing import Any

class Answer(BaseAnswer):
    identifies_bcl2: bool = Field(
        description=(
            "True if the response identifies BCL2 (including Bcl-2, BCL-2, or "
            "B-cell lymphoma 2) as the direct pharmacological target of venetoclax. "
            "False if BCL2 is mentioned only as a pathway member or a different "
            "protein is identified as the primary target."
        )
    )

    def ground_truth(self):
        self.correct = {"identifies_bcl2": True}

    def verify(self) -> bool:
        return self.identifies_bcl2 == self.correct["identifies_bcl2"]
"""

_original_add_template = _TaskEval.add_template


def _safe_add_template(self, template_class, step_id=None):
    try:
        _original_add_template(self, template_class, step_id=step_id)
    except (TypeError, OSError):
        self.add_question(
            {
                "id": template_class.__name__.lower(),
                "question": "",
                "raw_answer": "",
                "answer_template": _ANSWER_TEMPLATE_CODE,
            },
            step_id=step_id,
        )


_TaskEval.add_template = _safe_add_template
```

---

## Complete Example

Four lines to go from logged text to evaluation results:

```python
from karenina.benchmark.task_eval import TaskEval
from karenina.schemas.config.models import ModelConfig
from karenina.schemas.verification.config import VerificationConfig
from karenina.schemas.entities import BaseAnswer
from pydantic import Field

task = TaskEval(task_id="quick-demo")
task.log("The primary pharmacological target of venetoclax is BCL2.")


class Answer(BaseAnswer):
    identifies_bcl2: bool = Field(description="True if BCL2 is identified as the target.")

    def ground_truth(self):
        self.correct = {"identifies_bcl2": True}

    def verify(self) -> bool:
        return self.identifies_bcl2 == self.correct["identifies_bcl2"]


task.add_template(Answer)
config = VerificationConfig(
    parsing_models=[
        ModelConfig(
            id="haiku",
            model_name="claude-haiku-4-5",
            model_provider="anthropic",
            interface="langchain",
            temperature=0.0,
        )
    ],
    parsing_only=True,
)
result = task.evaluate(config)
print(result.summary())
```

---

## Create a TaskEval Instance

TaskEval is a recording and evaluation container. You log text into it, attach evaluation criteria, then run evaluation. Create one with an optional task ID and metadata for tracking.

```python
from karenina.benchmark.task_eval import TaskEval

task = TaskEval(
    task_id="agent-drug-target",
    metadata={"model": "claude-haiku-4-5", "run_date": "2025-02-15"},
)

print(f"Created TaskEval: {task.task_id}")
```

`task_id` and `metadata` are optional tracking fields that appear in results and exports. They have no effect on evaluation behavior.

**Constructor parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_id` | `str \| None` | `None` | Identifier for tracking |
| `metadata` | `dict \| None` | `None` | Arbitrary metadata dict |
| `merge_strategy` | `"concatenate" \| "traces_only"` | `"concatenate"` | How to combine logs before evaluation (see [Merge Strategies](#merge-strategies)) |
| `callable_registry` | `dict[str, Callable]` | `None` | Registry for callable trait evaluation (see [Rubrics](../../core_concepts/rubrics/index.md)) |

---

## Log Outputs

TaskEval accepts two kinds of logged output: plain text and structured `Message` traces. Multiple calls accumulate: each `log()` or `log_trace()` call appends to the TaskEval's log store. When evaluation runs, the [merge strategy](#merge-strategies) controls how these logs combine into the response that the judge LLM sees.

### Text Logging

Use `log()` when you have a plain string output to evaluate. This is the simplest path: pass any text and TaskEval records it as potential answer content.

```python
task.log(
    "The approved drug target of venetoclax is BCL2 (B-cell lymphoma 2). "
    "Venetoclax is a selective BCL2 inhibitor that works by displacing pro-apoptotic "
    "proteins, triggering programmed cell death in cancer cells [1]."
)

print(f"Logged {len(task.global_logs)} event(s)")
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | (required) | Text content to log |
| `step_id` | `str \| None` | `None` | Step identifier for grouping (see [Multi-Step Evaluation](../../notebooks/task-eval/multi-step-evaluation.ipynb)) |
| `target` | `"global" \| "step" \| "both"` | `"both"` | Where to store the log |
| `level` | `"debug" \| "info" \| "warn" \| "error"` | `"info"` | Log level |
| `tags` | `list[str] \| None` | `None` | Optional tags |

### Trace Logging

Use `log_trace()` when the judge LLM needs to see the full conversation structure: user messages, assistant responses, tool calls, and tool results. This matters for rubric traits that assess tool usage, multi-turn reasoning, or conversation flow. Without trace structure, the judge only sees a flat string.

```python
from karenina.ports.messages import Message

trace_demo = TaskEval(task_id="trace-demo")
trace_demo.log_trace(
    [
        Message.user("What is the target of venetoclax?"),
        Message.assistant("The primary pharmacological target of venetoclax is BCL2."),
    ]
)

print(f"Logged {len(trace_demo.global_logs)} trace event(s)")
```

#### Tool Call Traces

For agent workflows with tool use, include `ToolUseContent` blocks and `Message.tool_result()`. The judge LLM receives the full trace, so tool call context can influence both template parsing and rubric scoring.

```python
from karenina.ports.messages import ToolUseContent

trace_demo.log_trace(
    [
        Message.user("What is the target of venetoclax?"),
        Message.assistant(
            "Let me search the database.",
            tool_calls=[ToolUseContent(id="call_1", name="search_db", input={"query": "venetoclax target"})],
        ),
        Message.tool_result(tool_use_id="call_1", content="BCL2 (B-cell lymphoma 2)"),
        Message.assistant(
            "Based on the search results, the primary pharmacological target of venetoclax is BCL2 (B-cell lymphoma 2)."
        ),
    ]
)

print(f"Total trace events: {len(trace_demo.global_logs)}")
```

**Parameters**: Same as `log()`, except the first argument is `messages: list[Message] | str`. If a plain `str` is passed, it is automatically wrapped as `Message.assistant(text)`.

### When to Use Each

| Situation | Use | Why |
|-----------|-----|-----|
| Single text output (API response, summary, generated answer) | `log()` | Simple, no conversation structure needed |
| Agent trace with tool calls | `log_trace()` | Preserves tool use context for the judge |
| Multi-turn conversation | `log_trace()` | Preserves turn structure and speaker roles |
| Mix of debug output and structured responses | Both | Text for notes, traces for evaluable content (use `traces_only` merge strategy) |

---

## Merge Strategies

Before evaluation, TaskEval combines all logged outputs into a single response for the [verification pipeline](../../core_concepts/verification-pipeline.md). The merge strategy controls how `log()` and `log_trace()` calls are combined.

### `concatenate` (default)

Every log contributes. Text logs are wrapped as `Message.assistant(text)`. Trace logs contribute their messages directly. All messages are serialized together in the order you logged them.

Use `concatenate` when all logged content is part of the response being evaluated.

```python
concat_demo = TaskEval(task_id="concat-demo")
concat_demo.log("BCL2 is the primary target of venetoclax.")
concat_demo.log("It inhibits the anti-apoptotic protein, triggering cell death.")

print(f"Logged {len(concat_demo.global_logs)} events (both contribute with 'concatenate')")
```

### `traces_only`

Only logs that contain structured `Message` traces are used. Text-only `log()` calls are silently ignored. No error is raised if all logs are text (the pipeline receives an empty string).

Use `traces_only` when your text logs are debug output, status messages, or intermediate notes that should not reach the judge LLM.

```python
traces_demo = TaskEval(task_id="traces-demo", merge_strategy="traces_only")
traces_demo.log("Debug: starting retrieval phase")
traces_demo.log_trace([Message.assistant("BCL2 is the primary target.")])

print(f"Total logs: {len(traces_demo.global_logs)}")
print("With traces_only: only the trace log reaches the judge")
print("The text log ('Debug: starting retrieval phase') is silently ignored")
```

### Setting the Strategy

Set at construction time via `merge_strategy`, or override per `evaluate()` call:

```python
custom_strategy = TaskEval(task_id="custom-strategy", merge_strategy="traces_only")
print(f"Instance default: {custom_strategy.merge_strategy}")
# Override per call: task.evaluate(config, merge_strategy="concatenate")
```

The per-call override takes precedence over the instance default.

---

## Attach Evaluation Criteria

Attach [templates](../../core_concepts/answer-templates.md), [rubrics](../../core_concepts/rubrics/index.md), or both. TaskEval auto-detects the [evaluation mode](../../notebooks/core_concepts/evaluation-modes.ipynb) based on what you provide:

| You attach | Evaluation mode | What happens |
|------------|----------------|--------------|
| Template only | `template_only` | Judge parses response into template fields, `verify()` checks correctness |
| Rubric only | `rubric_only` | Each trait scored independently (a synthetic minimal template is created internally) |
| Template + rubric | `template_and_rubric` | Both: template verification and rubric scoring |

If you attach neither, `evaluate()` raises `ValueError`.

### Templates (correctness)

Pass a `BaseAnswer` subclass directly via `add_template()`. The template's fields tell the judge LLM what to extract; `verify()` checks correctness programmatically.

```python
from pydantic import Field

from karenina.schemas.entities import BaseAnswer


class Answer(BaseAnswer):
    identifies_bcl2: bool = Field(
        description=(
            "True if the response identifies BCL2 (including Bcl-2, BCL-2, or "
            "B-cell lymphoma 2) as the direct pharmacological target of venetoclax. "
            "False if BCL2 is mentioned only as a pathway member or a different "
            "protein is identified as the primary target."
        )
    )

    def ground_truth(self):
        self.correct = {"identifies_bcl2": True}

    def verify(self) -> bool:
        return self.identifies_bcl2 == self.correct["identifies_bcl2"]


task.add_template(Answer)

print("Attached answer template with 1 verification field")
```

See [Answer Templates](../../core_concepts/answer-templates.md) for field types, descriptions, and writing `verify()` methods.

### Rubrics (quality)

Attach a rubric via `add_rubric()`. Each trait evaluates one dimension of the response independently. TaskEval supports all the same trait types as Benchmark.

```python
from karenina.schemas.entities.rubric import LLMRubricTrait, RegexRubricTrait, Rubric

rubric = Rubric(
    llm_traits=[
        LLMRubricTrait(
            name="conciseness",
            kind="boolean",
            description=(
                "True if the response answers directly without unnecessary "
                "elaboration. False if the response is verbose or includes "
                "tangential information."
            ),
            higher_is_better=True,
        ),
    ],
    regex_traits=[
        RegexRubricTrait(
            name="has_citations",
            pattern=r"\[\d+\]",
            description="Response includes numbered citations in bracket notation.",
            case_sensitive=False,
            higher_is_better=True,
        ),
    ],
)

task.add_rubric(rubric)

print(f"Added rubric with {len(rubric.llm_traits)} LLM trait(s) and {len(rubric.regex_traits)} regex trait(s)")
```

### Other Trait Types

TaskEval also supports callable traits (custom Python functions via cloudpickle), metric traits (precision/recall/F1 from confusion matrices), and literal traits (exact expected values). See [Rubrics](../../core_concepts/rubrics/index.md) for all five trait types and their setup.

**Note**: Trait names must be unique across all trait types within a rubric. If you add multiple rubrics (e.g., global + step-scoped), they are merged before evaluation, and duplicate trait names raise `ValueError`.

---

## Configure and Evaluate

Create a [VerificationConfig](../../reference/configuration/verification-config.md) with `parsing_only=True` and call `evaluate()`.

`parsing_only=True` tells the pipeline to skip answer generation (stage 2), since you already supplied the text via `log()`. The pipeline still runs all relevant evaluation stages: template parsing (stage 7), template verification (stage 8), rubric evaluation (stage 11), and finalization (stage 13).

```python
from karenina.schemas.config.models import ModelConfig
from karenina.schemas.verification.config import VerificationConfig

config = VerificationConfig(
    parsing_models=[
        ModelConfig(
            id="claude-haiku-4-5",
            model_name="claude-haiku-4-5",
            model_provider="anthropic",
            interface="langchain",
            temperature=0.0,
        )
    ],
    parsing_only=True,
)

result = task.evaluate(config)

print(f"Evaluation complete: {result.summary()}")
```

### Evaluation Scope

Without `step_id`, `evaluate()` runs global evaluation first, then auto-evaluates every step that has logged data. Pass `step_id` to evaluate only one step's logs and criteria:

```python
# Evaluate only one step (shown in Multi-Step Evaluation):
# result = task.evaluate(config, step_id="retrieval")
print("Without step_id: evaluates global + all steps")
print("With step_id: evaluates only that step")
```

---

## Inspect Results

`evaluate()` returns a `TaskEvalResult` containing global and per-step results.

### Quick Overview

Start with `display()` for a formatted report and `summary()` for a one-liner:

```python
print(result.display())
```

```python
print(result.summary())
```

### Result Structure

```python
print(f"task_id:    {result.task_id}")
print(f"metadata:   {result.metadata}")
print(f"global_eval: {'present' if result.global_eval else 'None'}")
print(f"per_step:   {list(result.per_step.keys())}")
print(f"timestamp:  {result.timestamp}")
```

`global_eval` contains evaluation results for globally-scoped logs and criteria. `per_step` maps step IDs to their respective `StepEval` objects.

### Summary Statistics

Each `StepEval` contains `verification_results`: a dict mapping question IDs to lists of [VerificationResult](../analyzing-results/verification-result.md) (one per replicate).

`get_summary_stats()` returns a dict with these keys:

| Key | Type | Description |
|-----|------|-------------|
| `traces_total` | `int` | Number of evaluated traces (question IDs) |
| `traces_passed` | `int` | Traces where at least one replicate passed verification |
| `template_verification_total` | `int` | Results where template verification was performed |
| `template_verification_passed` | `int` | Of those, how many passed |
| `rubric_traits_total` | `int` | Sum of all trait counts (LLM + regex + callable + metric) |
| `rubric_traits_passed` | `int` | Traits with score `True` or `> 0` |
| `success_rate` | `float` | `traces_passed / traces_total * 100`, or `0.0` if no traces |

```python
step = result.global_eval
stats = step.get_summary_stats()

for key, value in stats.items():
    print(f"  {key}: {value}")
```

### Formatted Results

For a detailed text breakdown of each verification result, including the full response, individual trait scores, and any error messages:

```python
print(step.format_verification_results())
```

### Accessing VerificationResult Fields

Each [VerificationResult](../analyzing-results/verification-result.md) uses nested sub-objects for structured access:

```python
for qid, vr_list in step.verification_results.items():
    vr = vr_list[0]  # first replicate

    print(f"Question: {qid}")
    print(f"  Template pass: {vr.template.verify_result}")

    if vr.rubric:
        print(f"  LLM traits: {vr.rubric.llm_trait_scores}")
        print(f"  Regex traits: {vr.rubric.regex_trait_scores}")
```

### Export

Export results as JSON (optionally including raw logs) or Markdown for downstream analysis or archiving:

```python
json_str = result.export_json()
print(json_str[:300])
```

```python
md_str = result.export_markdown()
print(md_str[:300])
```

For downstream DataFrame analysis, see [Analyzing Results](../analyzing-results/index.md).

---

## Cleanup

```python tags=["hide-cell"]
# Restore original LLM behavior and add_template
BaseChatModel.ainvoke = _original_ainvoke
_TaskEval.add_template = _original_add_template
```

---

## Next Steps

- [Quality Assessment with Rubrics](../../notebooks/task-eval/quality-assessment.ipynb): Rubric-only evaluation with all trait types
- [Multi-Step Agent Evaluation](../../notebooks/task-eval/multi-step-evaluation.ipynb): Step-scoped evaluation for agent workflows
- [TaskEval concept page](../../core_concepts/task-eval.md): Object structure, pipeline integration, and merge strategies in depth
- [Answer Templates](../../core_concepts/answer-templates.md): Writing `verify()` methods and field descriptions
- [Rubrics](../../core_concepts/rubrics/index.md): All five trait types (LLM, regex, callable, metric, literal)
