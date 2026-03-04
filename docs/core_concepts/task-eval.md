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

_storage_mock = MagicMock()
_storage_mock.DBConfig = MagicMock()
_storage_mock.load_benchmark = MagicMock()
_storage_mock.save_benchmark = MagicMock()
for _mod in [
    "karenina.storage",
    "karenina.storage.generated_models",
    "karenina.storage.auto_mapper",
    "karenina.storage.operations",
    "karenina.storage.models",
]:
    sys.modules[_mod] = _storage_mock

from karenina.benchmark.task_eval import TaskEval, TaskEvalResult, StepEval
from karenina.ports.messages import Message, ToolUseContent
from karenina.schemas.entities import BaseAnswer
from karenina.schemas.entities.rubric import LLMRubricTrait, RegexTrait, Rubric
from karenina.schemas.config.models import ModelConfig
from karenina.schemas.verification.config import VerificationConfig
from karenina.schemas.verification import VerificationResult
from karenina.schemas.verification.result_components import (
    VerificationResultMetadata,
    VerificationResultTemplate,
    VerificationResultRubric,
)
from karenina.schemas.verification.model_identity import ModelIdentity
from pydantic import Field
from datetime import datetime

_mock_identity = ModelIdentity(interface="mock", model_name="mock-model")

def _make_vr(qid, verify_result=True, llm_scores=None, regex_scores=None, response=""):
    vr = VerificationResult(
        metadata=VerificationResultMetadata(
            question_id=qid, template_id="t1", completed_without_errors=True,
            question_text="(mock)", answering=_mock_identity, parsing=_mock_identity,
            execution_time=0.05, timestamp=datetime.now().isoformat(), result_id="abcdef0123456789",
        ),
        template=VerificationResultTemplate(
            verify_result=verify_result, template_verification_performed=True,
            raw_llm_response=response or "(trace)",
        ),
    )
    if llm_scores or regex_scores:
        vr.rubric = VerificationResultRubric(
            rubric_evaluation_performed=True,
            llm_trait_scores=llm_scores or {}, regex_trait_scores=regex_scores or {},
        )
    return vr

def _mock_evaluate(self, config, step_id=None, merge_strategy=None):
    def _build_step(questions, logs, rubrics):
        step = StepEval()
        merged_text = " ".join(l.text for l in logs if l.text)
        llm_scores, regex_scores = {}, {}
        if rubrics:
            for r in rubrics:
                for t in r.llm_traits:
                    llm_scores[t.name] = True if t.kind == "boolean" else 4
                for t in r.regex_traits:
                    regex_scores[t.name] = True
        for q in questions:
            qid = q.get("id", q.get("question", "q")[:8]) if isinstance(q, dict) else getattr(q, "id", "q")
            step.verification_results[qid] = [
                _make_vr(qid, verify_result=True, llm_scores=llm_scores or None,
                         regex_scores=regex_scores or None, response=merged_text[:200])
            ]
        if not questions and rubrics:
            step.verification_results["synthetic"] = [
                _make_vr("synthetic", verify_result=True, llm_scores=llm_scores or None,
                         regex_scores=regex_scores or None, response=merged_text[:200])
            ]
        return step
    result = TaskEvalResult(task_id=self.task_id, metadata=self.metadata)
    if step_id:
        logs = self.step_logs.get(step_id, [])
        qs = self.step_questions.get(step_id, self.global_questions)
        rs = self.step_rubrics.get(step_id, self.global_rubrics)
        result.per_step[step_id] = _build_step(qs, logs, rs)
    else:
        if self.global_questions or self.global_rubrics:
            result.global_eval = _build_step(self.global_questions, self.global_logs, self.global_rubrics)
        for sid in self._get_available_step_ids():
            logs = self.step_logs.get(sid, [])
            qs = self.step_questions.get(sid, [])
            rs = self.step_rubrics.get(sid, [])
            if qs or rs:
                result.per_step[sid] = _build_step(
                    qs or self.global_questions, logs, rs or self.global_rubrics
                )
    return result

_orig_add_template = TaskEval.add_template


def _mock_add_template(self, template_class, step_id=None):
    question_dict = {
        "id": template_class.__name__.lower(),
        "question": "",
        "raw_answer": "",
        "answer_template": "mock_template",
    }
    self.add_question(question_dict, step_id=step_id)


TaskEval.add_template = _mock_add_template

TaskEval.evaluate = _mock_evaluate
```

```python
from karenina.benchmark.task_eval import TaskEval
from karenina.schemas.entities import BaseAnswer
from karenina.schemas.entities.rubric import LLMRubricTrait, Rubric
from pydantic import Field
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

See the [workflow page](../workflows/task-eval/index.md) for the complete code walkthrough.

### Object structure

A TaskEval instance holds two parallel scopes: global and per-step. Each scope stores logs (the text to evaluate), templates (correctness criteria), and rubrics (quality criteria).

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

TaskEval accepts two types of evaluation criteria: [answer templates](answer-templates.md) for correctness and [rubrics](rubrics/index.md) for quality. The [evaluation mode](evaluation-modes.md) is auto-detected based on what you attach.

### Templates (correctness)

Pass a `BaseAnswer` subclass directly via `add_template()`. No question text is needed; the template's fields and `verify()` method define what the judge LLM extracts and how correctness is determined.

```python
from karenina.schemas.entities import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    identifies_bcl2: bool = Field(
        description="True if the response identifies BCL2 as the primary target of venetoclax."
    )
    def ground_truth(self):
        self.correct = {"identifies_bcl2": True}
    def verify(self) -> bool:
        return self.identifies_bcl2 == self.correct["identifies_bcl2"]

task = TaskEval(task_id="my-eval")
task.log("BCL2 is the primary pharmacological target of venetoclax.")
task.add_template(Answer)
```

### Rubrics (quality)

Attach a `Rubric` via `add_rubric()`. Rubric evaluation uses the same infrastructure as Benchmark: LLM traits, regex traits, callable traits, and metric traits. All Benchmark evaluation options apply, including batch vs sequential strategy and full-trace vs last-message scope.

```python
task = TaskEval(task_id="my-eval")
task.log("BCL2 is the primary pharmacological target of venetoclax.")
task.add_rubric(Rubric(llm_traits=[
    LLMRubricTrait(name="conciseness", kind="boolean", description="True if the response is concise and direct.")
]))
```

### Both together

Attach a template and rubric to the same TaskEval for combined correctness and quality evaluation:

```python
task = TaskEval(task_id="my-eval")
task.log("BCL2 is the primary pharmacological target of venetoclax.")
task.add_template(Answer)
task.add_rubric(Rubric(llm_traits=[
    LLMRubricTrait(name="conciseness", kind="boolean", description="True if the response is concise and direct.")
]))
```

## Logging

TaskEval provides two logging methods for recording agent outputs.

### Text Logging

Use `log()` for plain text outputs. The text is treated as potential answer content for evaluation.

```python
task = TaskEval(task_id="my-eval")
task.log("The primary target of venetoclax is BCL2.")
task.log("Additional context about the drug mechanism.", step_id="reasoning")
```

### Trace Logging

Use `log_trace()` for structured `Message` traces from `karenina.ports.messages`. This preserves the full conversation structure, including tool calls and results.

```python
from karenina.ports.messages import Message, ToolUseContent

task = TaskEval(task_id="my-eval")
task.log_trace([
    Message.user("What is the target of venetoclax?"),
    Message.assistant("Let me look that up.", tool_calls=[
        ToolUseContent(id="call_1", name="search_db", input={"query": "venetoclax target"})
    ]),
    Message.tool_result(tool_use_id="call_1", content="BCL2 (B-cell lymphoma 2)"),
    Message.assistant("The primary pharmacological target of venetoclax is BCL2."),
])
```

### Scoping Parameters

Both methods accept `step_id` and `target` parameters:

- **`step_id`**: Groups logs under a named step for per-phase evaluation
- **`target`**: Controls where logs are stored: `"global"`, `"step"`, or `"both"` (default). When `target="both"` and a `step_id` is provided, the log is stored in both the global and step-specific stores.

See the [workflow page](../workflows/task-eval/index.md) for full examples.

## Merge Strategies

When evaluation runs, TaskEval combines logged outputs into a single response for the pipeline. Two strategies control this merging:

**`concatenate`** (default): Combines text and trace logs into a single response. Text logs are wrapped as `Message.assistant()` objects; trace messages are included as-is. All messages are serialized together.

**`traces_only`**: Uses only logs that contain structured `Message` traces. Text-only logs are ignored. Use this when your text logs are debugging output that should not be part of the evaluated response.

```python
# Use traces_only when text logs are just debug output
task = TaskEval(task_id="my-eval", merge_strategy="traces_only")
task.log("Debug: starting search")  # ignored during evaluation
task.log_trace([Message.assistant("BCL2 is the target.")])  # used
```

## Step-Specific Evaluation

TaskEval supports both global and step-level scoping for logs, questions, and rubrics. This enables per-phase analysis of multi-step agent workflows.

```python
task = TaskEval(task_id="agent-run")

# Log outputs for each phase
task.log("Found 3 relevant papers.", step_id="retrieval")
task.log("BCL2 is the primary target based on the evidence.", step_id="synthesis")

# Attach step-specific rubrics
task.add_rubric(
    Rubric(llm_traits=[
        LLMRubricTrait(name="retrieval_quality", kind="boolean", description="True if relevant sources were found.")
    ]),
    step_id="retrieval",
)
task.add_rubric(
    Rubric(llm_traits=[
        LLMRubricTrait(name="synthesis_quality", kind="boolean", description="True if the synthesis is well-supported.")
    ]),
    step_id="synthesis",
)
```

When you call `evaluate()` without a `step_id`, TaskEval runs global evaluation and then auto-evaluates all steps that have logs. Results are available in `result.global_eval` and `result.per_step`.

## How TaskEval Uses the Pipeline

TaskEval injects logged outputs as `cached_answer_data` into the [verification pipeline](verification-pipeline.md). This skips answer generation (stage 2) but runs all other relevant stages: template parsing (stage 7), template verification (stage 8), rubric evaluation (stage 11), and finalization (stage 13).

Use `VerificationConfig(parsing_only=True)` since no answering model is needed; only a parsing (judge) model is required.

```python
from karenina.schemas.config.models import ModelConfig
from karenina.schemas.verification.config import VerificationConfig

config = VerificationConfig(
    parsing_models=[ModelConfig(id="haiku", model_provider="anthropic", model_name="claude-haiku-4-5")],
    parsing_only=True,
)
result = task.evaluate(config)
```

The pipeline treats logged output exactly like a live LLM response. All template parsing, verification, and rubric evaluation stages operate identically to the standard Benchmark workflow.

<div class="admonition note">
<p class="admonition-title">Pipeline Reuse</p>
<p>TaskEval reuses the same verification pipeline as Benchmark. The only difference is that stage 2 (answer generation) is skipped because the response is already provided via <code>cached_answer_data</code>.</p>
</div>
