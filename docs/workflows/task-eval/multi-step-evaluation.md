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

# Multi-Step Agent Evaluation

This tutorial shows how to evaluate agent workflows with distinct phases using TaskEval's step-scoped evaluation. Each step gets its own logs, templates, and rubric traits, evaluated independently. Use this when your agent performs sequential phases (retrieval, reasoning, synthesis) and you want to assess each phase separately: did retrieval find relevant sources? Did synthesis accurately summarize them?

**What you'll learn:**

- Log outputs to named steps with `step_id`
- Control log routing with `target` ("global", "step", "both")
- Attach step-specific templates and rubrics
- Log structured agent traces with tool calls using `log_trace()`
- Evaluate individual steps with `evaluate(config, step_id=...)`
- Evaluate globally (all steps combined)
- Compare per-step results via `TaskEvalResult.per_step`
- Use `traces_only` merge strategy to filter debug output

```python tags=["hide-cell"]
# Mock cell: replays captured LLM responses from docs/data/workflow-taskeval-multistep/ so
# the workflow executes without live API keys. The full pipeline logic runs;
# only the raw model calls are mocked.
# This cell is hidden in the rendered documentation.
import hashlib
import json
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

# Resolve fixtures directory
_FIXTURES_DIR = None
for _candidate in [
    Path("data/workflow-taskeval-multistep"),
    Path("../data/workflow-taskeval-multistep"),
    Path("../../data/workflow-taskeval-multistep"),
    Path("docs/data/workflow-taskeval-multistep"),
]:
    if _candidate.is_dir():
        _FIXTURES_DIR = _candidate
        break

# Track call order for sequential fixture replay
_call_counter = 0
_fixtures_by_sequence: dict[int, dict] = {}

if _FIXTURES_DIR is not None:
    for _p in sorted(_FIXTURES_DIR.glob("*.json")):
        _data = json.loads(_p.read_text())
        _fixtures_by_sequence[_data["sequence"]] = _data

# Save original ainvoke for restoration
_original_ainvoke = BaseChatModel.ainvoke


async def _replaying_ainvoke(self, input, config=None, **kwargs):
    """Return captured LLM responses in sequence order."""
    global _call_counter
    _call_counter += 1
    fixture = _fixtures_by_sequence.get(_call_counter)
    if fixture is None:
        return AIMessage(
            content=[{
                "type": "tool_use",
                "id": f"fixture_{_call_counter}",
                "name": "RubricEvaluationResult",
                "input": {"reasoning": "Fixture response", "abstention_detected": False, "score": True},
                "caller": {"type": "direct"},
            }],
            id=f"fixture-{_call_counter}",
            tool_calls=[{
                "name": "RubricEvaluationResult",
                "args": {"reasoning": "Fixture response", "abstention_detected": False, "score": True},
                "id": f"fixture_{_call_counter}",
                "type": "tool_call",
            }],
            response_metadata={"model_name": "claude-haiku-4-5-20251001", "stop_reason": "tool_use"},
            usage_metadata={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )
    resp = fixture["response"]
    return AIMessage(
        content=resp["content"],
        id=resp.get("id", "fixture"),
        tool_calls=resp.get("tool_calls", []),
        response_metadata=resp.get("response_metadata", {}),
        usage_metadata=resp.get("usage_metadata"),
    )


BaseChatModel.ainvoke = _replaying_ainvoke

# Patch add_template for nbconvert compatibility
from karenina.benchmark.task_eval import TaskEval as _TaskEval

_original_add_template = _TaskEval.add_template


_MULTISTEP_TEMPLATE_CODE = '''\
from karenina.schemas.entities import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    identifies_bcl2: bool = Field(
        description=(
            "True if the combined output identifies BCL2 (including Bcl-2, BCL-2, or "
            "B-cell lymphoma 2) as the direct pharmacological target of venetoclax."
        )
    )

    def ground_truth(self):
        self.correct = {"identifies_bcl2": True}

    def verify(self) -> bool:
        return self.identifies_bcl2 == self.correct["identifies_bcl2"]
'''

def _safe_add_template(self, template_class, step_id=None):
    try:
        _original_add_template(self, template_class, step_id=step_id)
    except (TypeError, OSError):
        self.add_question(
            {"id": template_class.__name__.lower(), "question": "", "raw_answer": "",
             "answer_template": _MULTISTEP_TEMPLATE_CODE},
            step_id=step_id,
        )


_TaskEval.add_template = _safe_add_template
```

---

## Complete Example

A compact multi-step evaluation end-to-end:

```python
from karenina.benchmark.task_eval import TaskEval
from karenina.schemas.config.models import ModelConfig
from karenina.schemas.entities.rubric import LLMRubricTrait, Rubric
from karenina.schemas.verification.config import VerificationConfig

task = TaskEval(task_id="multi-step-demo")

task.log("Found 3 relevant papers on venetoclax.", step_id="retrieval")
task.log("BCL2 is the direct target of venetoclax.", step_id="synthesis")

task.add_rubric(Rubric(llm_traits=[
    LLMRubricTrait(name="source_quality", kind="boolean", higher_is_better=True,
                   description="True if relevant sources were identified."),
]), step_id="retrieval")

task.add_rubric(Rubric(llm_traits=[
    LLMRubricTrait(name="accuracy", kind="boolean", higher_is_better=True,
                   description="True if the synthesis is factually accurate."),
]), step_id="synthesis")

config = VerificationConfig(
    parsing_models=[ModelConfig(id="haiku", model_name="claude-haiku-4-5",
                                model_provider="anthropic", interface="langchain", temperature=0.0)],
    parsing_only=True,
)

for sid in ["retrieval", "synthesis"]:
    step_result = task.evaluate(config, step_id=sid)
    stats = step_result.per_step[sid].get_summary_stats()
    print(f"Step '{sid}': {stats['rubric_traits_passed']}/{stats['rubric_traits_total']} traits passed")
```

---

## Set Up the Agent Workflow

This tutorial uses a three-phase drug target analysis scenario:

1. **Retrieval**: Search databases for relevant sources
2. **Reasoning**: Analyze retrieved information and identify the target
3. **Synthesis**: Produce a final answer with supporting evidence

Each phase generates different output that should be evaluated against different criteria.

```python
from karenina.benchmark.task_eval import TaskEval

task = TaskEval(task_id="drug-target-agent")

print(f"Created TaskEval: {task.task_id}")
print("Phases: retrieval → reasoning → synthesis")
```

---

## Log Step Outputs

Pass `step_id` to `log()` to assign output to a named step. The `target` parameter controls where the log is stored:

| Target | Step scope | Global scope | Default when |
|--------|-----------|-------------|-------------|
| `"both"` | Yes | Yes | `step_id` is set |
| `"step"` | Yes | No | Never (must specify explicitly) |
| `"global"` | No | Yes | `step_id` is not set |

```python
# Retrieval phase outputs
task.log(
    "Database search returned 3 results for 'venetoclax mechanism of action': "
    "(1) Souers et al. 2013, Nat Med; (2) Roberts et al. 2016, NEJM; "
    "(3) Anderson et al. 2016, Blood.",
    step_id="retrieval",
)

# Reasoning phase outputs
task.log(
    "Analysis: All three sources identify BCL2 as the direct binding target. "
    "Souers et al. describe the BH3 mimetic mechanism. Roberts et al. confirm "
    "clinical efficacy via BCL2 inhibition in CLL patients.",
    step_id="reasoning",
)

# Synthesis phase outputs
task.log(
    "The approved pharmacological target of venetoclax is BCL2 (B-cell lymphoma 2). "
    "Venetoclax is a selective BH3 mimetic that binds BCL2, displacing pro-apoptotic "
    "proteins and triggering apoptosis [Souers 2013, Roberts 2016].",
    step_id="synthesis",
)

print(f"Global logs: {len(task.global_logs)} (all steps contribute with target='both')")
```

### Step-Only Logging

Use `target="step"` when a log should not appear in global evaluation. This is useful for intermediate debug output that is relevant to one step but should not influence the overall assessment.

```python
task.log(
    "DEBUG: search latency was 1.2s, 3/50 results passed relevance filter",
    step_id="retrieval",
    target="step",
    level="debug",
)

print(f"Global logs unchanged: {len(task.global_logs)}")
```

---

## Log Structured Traces

Use `log_trace()` when you need to preserve conversation structure, tool calls, and speaker roles. This matters for rubric traits that assess tool usage or multi-turn reasoning.

```python
from karenina.ports.messages import Message, ToolUseContent

task.log_trace([
    Message.user("What is the approved pharmacological target of venetoclax?"),
    Message.assistant("Let me search the drug database.", tool_calls=[
        ToolUseContent(id="call_1", name="search_drugs", input={"query": "venetoclax target"})
    ]),
    Message.tool_result(tool_use_id="call_1", content="BCL2 (B-cell lymphoma 2); selective BH3 mimetic"),
    Message.assistant(
        "Based on the database search, the approved pharmacological target of "
        "venetoclax is BCL2 (B-cell lymphoma 2). Venetoclax acts as a selective "
        "BH3 mimetic that directly binds to the BCL2 protein."
    ),
], step_id="retrieval")

print(f"Global logs after trace: {len(task.global_logs)}")
```

---

## Attach Step-Scoped Criteria

Pass `step_id` to `add_rubric()` to attach criteria to a specific step. Each step can have different traits, tuned to what that phase should accomplish.

```python
from karenina.schemas.entities.rubric import LLMRubricTrait, RegexRubricTrait, Rubric

# Retrieval: did it find relevant sources?
task.add_rubric(Rubric(
    llm_traits=[
        LLMRubricTrait(
            name="source_relevance",
            kind="boolean",
            higher_is_better=True,
            description=(
                "True if the retrieved sources are directly relevant to the query about "
                "venetoclax's pharmacological target. False if sources are tangential or unrelated."
            ),
        ),
        LLMRubricTrait(
            name="source_count",
            kind="score",
            min_score=1,
            max_score=5,
            higher_is_better=True,
            description=(
                "Rate the number and diversity of sources retrieved. "
                "1: no sources or only one. 3: adequate coverage. 5: comprehensive multi-source retrieval."
            ),
        ),
    ],
), step_id="retrieval")

print("Added retrieval rubric (2 traits)")
```

```python
# Reasoning: is the analysis sound?
task.add_rubric(Rubric(
    llm_traits=[
        LLMRubricTrait(
            name="reasoning_quality",
            kind="boolean",
            higher_is_better=True,
            description=(
                "True if the reasoning step correctly synthesizes information from "
                "retrieved sources. False if it introduces unsupported claims or "
                "ignores contradictory evidence."
            ),
        ),
    ],
), step_id="reasoning")

print("Added reasoning rubric (1 trait)")
```

```python
# Synthesis: is the final answer accurate and well-supported?
task.add_rubric(Rubric(
    llm_traits=[
        LLMRubricTrait(
            name="factual_accuracy",
            kind="boolean",
            higher_is_better=True,
            description="True if the final answer is factually correct.",
        ),
    ],
    regex_traits=[
        RegexRubricTrait(
            name="has_citations",
            pattern=r"\[.+?\d{4}\]|\(.+?\d{4}\)",
            description="Response includes author-year citations.",
            higher_is_better=True,
        ),
    ],
), step_id="synthesis")

print("Added synthesis rubric (1 LLM + 1 regex trait)")
```

---

## Attach a Global Template

A global template evaluates the combined output from all steps. This checks whether the overall workflow produced the correct answer, regardless of how individual steps performed.

```python
from pydantic import Field
from karenina.schemas.entities import BaseAnswer


class Answer(BaseAnswer):
    identifies_bcl2: bool = Field(
        description=(
            "True if the combined output identifies BCL2 (including Bcl-2, BCL-2, or "
            "B-cell lymphoma 2) as the direct pharmacological target of venetoclax."
        )
    )

    def ground_truth(self):
        self.correct = {"identifies_bcl2": True}

    def verify(self) -> bool:
        return self.identifies_bcl2 == self.correct["identifies_bcl2"]


task.add_template(Answer)

print("Added global template for correctness verification")
```

---

## Evaluate Per Step

Evaluate each step individually by passing `step_id`. Each step evaluation uses only that step's logs and criteria.

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

for step_id in ["retrieval", "reasoning", "synthesis"]:
    step_result = task.evaluate(config, step_id=step_id)
    step_eval = step_result.per_step[step_id]
    stats = step_eval.get_summary_stats()
    print(f"Step '{step_id}': "
          f"{stats['rubric_traits_passed']}/{stats['rubric_traits_total']} rubric traits passed")
```

---

## Evaluate Globally

Call `evaluate()` without `step_id` to run global evaluation. Global evaluation uses all globally-scoped logs (from steps with `target="both"` or `target="global"`) and globally-attached criteria (the template in this case).

```python
global_result = task.evaluate(config)

if global_result.global_eval:
    stats = global_result.global_eval.get_summary_stats()
    print(f"Global: {stats['traces_passed']}/{stats['traces_total']} passed template verification")
    print(f"Global: {stats['rubric_traits_passed']}/{stats['rubric_traits_total']} rubric traits passed")
```

---

## Inspect Per-Step Results

The `TaskEvalResult.per_step` dict maps step IDs to `StepEval` objects. Each `StepEval` contains `verification_results` and provides `get_summary_stats()`.

```python
print(f"Steps evaluated: {list(global_result.per_step.keys())}")

for step_id, step_eval in global_result.per_step.items():
    stats = step_eval.get_summary_stats()
    print(f"\n--- {step_id} ---")
    print(f"  Traces: {stats['traces_total']}")
    print(f"  Rubric traits: {stats['rubric_traits_passed']}/{stats['rubric_traits_total']}")

    for qid, vr_list in step_eval.verification_results.items():
        vr = vr_list[0]
        if vr.rubric:
            for name, score in vr.rubric.llm_trait_scores.items():
                print(f"  {name}: {score}")
            for name, score in vr.rubric.regex_trait_scores.items():
                print(f"  {name}: {score}")
```

---

## Merge Strategies in Multi-Step

The `traces_only` merge strategy is particularly useful in multi-step workflows. When steps produce both debug output (via `log()`) and structured traces (via `log_trace()`), `traces_only` ensures only the structured content reaches the judge.

```python
filtered_task = TaskEval(task_id="filtered-demo", merge_strategy="traces_only")

# Debug output: excluded from evaluation
filtered_task.log("DEBUG: starting retrieval, timeout=30s", step_id="retrieval")
filtered_task.log("DEBUG: 3 results returned in 1.2s", step_id="retrieval")

# Structured trace: included in evaluation
filtered_task.log_trace([
    Message.assistant("Found 3 papers on venetoclax: Souers 2013, Roberts 2016, Anderson 2016."),
], step_id="retrieval")

print(f"Total logs: {len(filtered_task.global_logs)}")
print("With traces_only: only the Message trace reaches the judge")
print("Both debug log() calls are silently excluded")
```

You can also override the merge strategy per `evaluate()` call:

```python
# Override just for one evaluation
# result = task.evaluate(config, merge_strategy="traces_only")
print("Per-call override: task.evaluate(config, merge_strategy='traces_only')")
```

---

## Cleanup

```python tags=["hide-cell"]
# Restore original LLM behavior and add_template
BaseChatModel.ainvoke = _original_ainvoke
_TaskEval.add_template = _original_add_template
```

---

## Next Steps

- [Basic Evaluation](../../notebooks/task-eval/basic-evaluation.ipynb): Single-step evaluation with templates and rubrics
- [Quality Assessment](../../notebooks/task-eval/quality-assessment.ipynb): Rubric-only evaluation with all trait types
- [TaskEval concept page](../../core_concepts/task-eval.md): Object structure, pipeline integration, merge strategies
- [Rubrics](../../core_concepts/rubrics/index.md): All five trait types in depth
- [Verification Pipeline](../../core_concepts/verification-pipeline.md): The 13-stage engine (with sub-stages 7a/7b and 11a/11b) that TaskEval feeds into
