---
name: karenina-task-eval
description: >
  Evaluate pre-recorded LLM outputs with karenina's TaskEval API. Use when
  you have existing model responses, chatbot logs, or agent traces and want
  to assess them against structured criteria. Covers logging outputs,
  template attachment, multi-step evaluation, and merge strategies.
  Invoke when evaluating pre-collected data without running an LLM.
---

# TaskEval: Pre-Recorded Output Evaluation

## What This Skill Does

TaskEval evaluates pre-recorded outputs using the same template/rubric engine
as Benchmark mode, but skips answer generation. You supply the model responses
(text or structured traces), attach an answer template and optional rubrics,
then run the verification pipeline as a judge-only pass.

## Prerequisites

- karenina installed (`uv pip install -e karenina`)
- An API key for the judge model (e.g., `ANTHROPIC_API_KEY` for Claude)

## Procedure

Follow these seven steps when building a TaskEval evaluation.

### Step 1: Create a TaskEval Instance

```python
from karenina.benchmark.task_eval import TaskEval

task = TaskEval(task_id="my-evaluation")
```

Optional constructor parameters:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `task_id` | `None` | Identifier for tracking results |
| `metadata` | `{}` | Arbitrary metadata dict attached to results |
| `callable_registry` | `{}` | Named callables for `CallableRubricTrait` evaluation |
| `merge_strategy` | `"concatenate"` | How logs are merged before evaluation |

**Do NOT pass `name=` to TaskEval.** There is no `name` parameter; use `task_id` instead.

### Step 2: Log Outputs

Use `.log()` for plain text and `.log_trace()` for structured conversation
traces (Message objects).

```python
# Plain text logging
task.log("BCL2 is the primary pharmacological target of venetoclax.")

# With step targeting
task.log("Step 1 result", step_id="extraction", level="info", tags=["step1"])

# Structured trace logging
from karenina.ports.messages import Message
task.log_trace([
    Message.assistant("The answer is BCL2."),
])
```

**Parameters for `.log()` and `.log_trace()`:**

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `text` / `messages` | `str` / `list[Message]` | (required) | The output to evaluate |
| `step_id` | `str \| None` | `None` | Associates log with a named step |
| `target` | `"global" \| "step" \| "both"` | `"both"` | Where the log is stored |
| `level` | `"debug" \| "info" \| "warn" \| "error"` | `"info"` | Log severity |
| `tags` | `list[str] \| None` | `None` | Categorization tags |

**Advanced: `add_question()`** accepts a dict or `Question` object as the first
positional argument (`question_obj`), NOT a keyword argument named `question`:

```python
# Correct
task.add_question({"id": "q1", "question": "What is 2+2?", "raw_answer": "4"})

# Wrong: "question" is not a keyword argument
task.add_question(question="What is 2+2?")
```

### Step 3: Author a Template

Define a `BaseAnswer` subclass with `VerifiedField` definitions. This tells
the judge LLM what to extract and how to verify it.

For detailed template authoring guidance, delegate to `karenina-template-authoring`.

```python
from karenina.schemas.entities import BaseAnswer, VerifiedField
from karenina.schemas.primitives import BooleanMatch

class Answer(BaseAnswer):
    identifies_bcl2: bool = VerifiedField(
        description=(
            "True if BCL2 is identified as the primary target of venetoclax. "
            "False otherwise."
        ),
        ground_truth=True,
        verify_with=BooleanMatch(),
    )
```

Attach the template to the task. Call this AFTER logging outputs:

```python
task.add_template(Answer)
```

`add_template()` converts the class source to a synthetic question internally.
It requires the class to be defined in a `.py` file or notebook (not
dynamically generated via `type()`).

### Step 4: (Optional) Add Rubrics

For qualitative assessment beyond template verification, add rubrics.

For detailed rubric authoring guidance, delegate to `karenina-rubric-authoring`.

```python
from karenina.schemas.entities.rubric import LLMRubricTrait, Rubric

rubric = Rubric(llm_traits=[
    LLMRubricTrait(
        name="cites_evidence",
        description=(
            "True if the response cites specific clinical trials, studies, "
            "or data to support its claims. False if claims are unsupported."
        ),
        kind="boolean",
    ),
])
task.add_rubric(rubric)
```

Both `add_rubric()` and `add_dynamic_rubric()` accept an optional `step_id`
parameter for step-scoped evaluation.

### Step 5: Configure

Create a `VerificationConfig` with `parsing_only=True`. This is mandatory;
without it the pipeline attempts to generate answers from a non-existent
answering LLM.

```python
from karenina.schemas.config.models import ModelConfig
from karenina.schemas.verification.config import VerificationConfig

config = VerificationConfig(
    parsing_models=[
        ModelConfig(
            id="judge",
            model_provider="anthropic",
            model_name="claude-haiku-4-5",
        ),
    ],
    parsing_only=True,
)
```

You do NOT set `evaluation_mode` for TaskEval. With `parsing_models` +
`parsing_only=True` (as above), `TaskEval.evaluate()` detects the evaluation
mode internally from what you attached: `template_only` when you added a
template, `rubric_only` when you added only rubrics, and `template_and_rubric`
when you added both. If neither a template nor a rubric was attached,
`evaluate()` raises `ValueError`. Attach or detach templates and rubrics to
change the mode — there is no config value to override it.

### Step 6: Evaluate

```python
result = task.evaluate(config)
```

`evaluate()` accepts optional overrides:

| Parameter | Type | Purpose |
|-----------|------|---------|
| `step_id` | `str \| None` | Evaluate a specific step instead of global |
| `merge_strategy` | `"concatenate" \| "traces_only" \| None` | Override instance default |
| `answering_model` | `ModelConfig \| None` | Identity recorded for the answering stage; a sentinel with `interface="taskeval"` is used when omitted |
| `run_name` | `str \| None` | Run name for result tracking; defaults to `taskeval_<uuid8>` when omitted |

When called without `step_id`, evaluation runs globally and then automatically
evaluates all steps that have logs, questions, or rubrics. If the global
context has no templates or rubrics but steps do, the global pass is skipped
(`result.global_eval` is `None`) and only the steps are evaluated.

### Step 7: Analyze Results

```python
# One-line summary
print(result.summary())

# Formatted display
print(result.display())

# Structured access
if result.global_eval:
    stats = result.global_eval.get_summary_stats()
    print(f"Success rate: {stats['success_rate']:.0f}%")

# Export
print(result.export_json())
print(result.export_markdown())
```

For advanced results analysis, delegate to `karenina-results`.

## Quick-Start Snippet

```python
"""Evaluate a single pre-recorded response."""
from karenina.benchmark.task_eval import TaskEval
from karenina.schemas.config.models import ModelConfig
from karenina.schemas.entities import BaseAnswer, VerifiedField
from karenina.schemas.primitives import BooleanMatch
from karenina.schemas.verification.config import VerificationConfig

task = TaskEval(task_id="my-evaluation")
task.log(
    "BCL2 is the primary pharmacological target of venetoclax. "
    "It works by inhibiting the anti-apoptotic BCL-2 protein."
)

class Answer(BaseAnswer):
    identifies_bcl2: bool = VerifiedField(
        description=(
            "True if BCL2 is identified as the primary target of venetoclax. "
            "False otherwise."
        ),
        ground_truth=True,
        verify_with=BooleanMatch(),
    )

task.add_template(Answer)

config = VerificationConfig(
    parsing_models=[
        ModelConfig(
            id="judge",
            model_provider="anthropic",
            model_name="claude-haiku-4-5",
        ),
    ],
    parsing_only=True,
)

result = task.evaluate(config)
print(result.summary())
```


## Gotchas

### 1. `parsing_only=True` Is Required

Without `parsing_only=True` in `VerificationConfig`, the pipeline tries to
invoke an answering LLM to generate responses. TaskEval has no answering
model, so this fails. Always set `parsing_only=True`.

### 2. Log Order Matters

The judge sees the full concatenated trace in the order logs were added.
If you log outputs out of order, the judge may misinterpret the conversation
flow. **Template parsing, however, reads only the FINAL AI message by default**
(`use_full_trace_for_template` defaults to `False`): multiple `.log()` calls
produce multiple trace messages, and a template whose fields are spread across
them will parse only the last one. Either put all text for one question in a
single `.log()` call, or set `VerificationConfig(use_full_trace_for_template=True)`
to hand the judge the full trace (honored by TaskEval).

### 3. Multi-Step Evaluation

Two approaches:

- **Single TaskEval with `step_id`**: log outputs to named steps, add
  questions/rubrics per step, and call `task.evaluate(config)`. Global
  evaluation runs first, then all steps with data are evaluated automatically.
  Steps-only evaluation works: when the global context has no templates or
  rubrics but step contexts do, the global pass is skipped and only the steps
  are evaluated (`result.global_eval` is `None`). `evaluate()` raises
  `ValueError` only when no context anywhere — global or step — has a
  template or rubric.

- **Separate TaskEval per step**: create independent `TaskEval` instances for
  each step. Simpler when steps have unrelated evaluation criteria.

Use `step_id` when steps share context (e.g., agent traces where later steps
build on earlier ones). Use separate instances when steps are independent.

### 4. Merge Strategy

The `merge_strategy` parameter controls how logged data is combined before
the judge sees it:

| Strategy | Behavior |
|----------|----------|
| `"concatenate"` (default) | Text logs are wrapped as assistant Messages, then combined with trace Messages. All logs contribute. |
| `"traces_only"` | Only LogEvents with `trace_messages` are used. Plain text logs are ignored. |

Override at call time: `task.evaluate(config, merge_strategy="traces_only")`.
`merge_strategy` is validated at construction and on the `evaluate()` override —
an unknown value raises `ValueError: merge_strategy must be one of
('concatenate', 'traces_only'), got ...`.

### 5. Guards Are Opt-In

Pipeline guards (abstention check, sufficiency check) do NOT run during
TaskEval by default: `abstention_enabled` and `sufficiency_enabled` both
default to `False` in `VerificationConfig`, and TaskEval passes the config
flags through unchanged. To turn a guard on, set it explicitly:

```python
config = VerificationConfig(
    parsing_models=[...],
    parsing_only=True,
    abstention_enabled=True,
    sufficiency_enabled=True,
)
```

Each enabled guard adds one extra judge LLM call per question, so enable
only the checks you need.

### 6. `add_template()` Must Follow `.log()` Calls

`add_template()` converts the class source to a synthetic question. It does
not capture logged outputs at call time. However, the convention is to log
outputs first and attach templates second, matching the conceptual flow of
"here is the data, here is how to evaluate it."

Template classes must be fully self-contained: `add_template()` captures only
the subclass's own source, so inheriting from another user template class
fails validation with `name 'X' is not defined`.

### 7. Evaluation Mode Is Detected Internally

`TaskEval.evaluate()` auto-detects the evaluation mode from what you attached.
There is no `evaluation_mode` to set on `VerificationConfig` for TaskEval — the
mode is chosen internally and cannot be overridden via config:

- `template_only` when only a template was added (`add_template()` or a
  question with an `answer_template`)
- `rubric_only` when only rubrics were added (`add_rubric()` /
  `add_dynamic_rubric()`)
- `template_and_rubric` when both are present

If neither a template nor a rubric was attached, `evaluate()` raises
`ValueError`. To change the mode, attach or detach templates/rubrics.

### 8. Dynamically Defined Classes Require Source Files

`add_template()` uses `inspect.getsource()` to capture the class definition.
Classes created with `type()` or in a REPL without a source file will raise
`TypeError`. For dynamic templates, pass template source code as a string
via `add_question()` instead.

### 9. TracePrimitive Templates Need Field Defaults

TracePrimitive-only templates must set `default=` on every `VerifiedField`:
the offline fast path instantiates `Answer()` bare, so a missing default
raises `ValidationError` and the question lands in `failed_questions`.

## Validation

Run the validation script to check a TaskEval script before execution:

```bash
uv run skills/karenina-task-eval/scripts/validate_task_eval.py path/to/script.py
```

The validator checks (AST-based, no imports required):
1. `TaskEval()` instantiation exists
2. `.log()` calls exist
3. A `BaseAnswer` subclass is defined
4. `VerificationConfig` includes `parsing_only=True`

## Reference Pointers

- **Template authoring**: `karenina-template-authoring` skill
- **Rubric authoring**: `karenina-rubric-authoring` skill
- **Results analysis**: `karenina-results` skill
- **TaskEval API source**: `karenina/src/karenina/benchmark/task_eval/task_eval.py`
- **Skeleton script**: `assets/task-eval-skeleton.py`
- **Validation script**: `scripts/validate_task_eval.py`
- **Full karenina docs**: The `using-karenina` skill contains the complete karenina documentation in its `references/` directory. Consult it for API details not covered here.
