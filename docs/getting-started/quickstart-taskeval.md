---
jupyter:
  jupytext:
    formats: getting-started//md,notebooks//ipynb
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Quick Start: TaskEval

TaskEval evaluates textual outputs that come from an external source: an LLM response, a paper excerpt, a chat transcript, or any other text you already have. You supply the text; Karenina's judge LLM handles the evaluation. This guide walks you through logging outputs, attaching evaluation criteria, running evaluation, and inspecting results.

By the end you will have evaluated a pre-recorded LLM response for both **correctness** (via an answer template) and **quality** (via rubric traits), with no question definition or answer generation required.

---

## Prerequisites

- **Python 3.11+**
- **Karenina installed** (see [Installation](installation.md))
- **API key** for the judge LLM provider:

> ```bash
> export ANTHROPIC_API_KEY="sk-ant-..."
> ```

---

```python tags=["hide-cell"]
# Mock cell: replays captured LLM responses from docs/data/quickstart-taskeval/ so
# the quickstart executes without live API keys. The full pipeline logic runs;
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
    Path("data/quickstart-taskeval"),
    Path("../data/quickstart-taskeval"),
    Path("docs/data/quickstart-taskeval"),
]:
    if _candidate.is_dir():
        _FIXTURES_DIR = _candidate
        break
assert _FIXTURES_DIR is not None, "Could not find data/quickstart-taskeval fixtures directory"

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

_ANSWER_TEMPLATE_CODE = '''\
from karenina.schemas.entities import BaseAnswer, VerifiedField
from karenina.schemas.primitives import BooleanMatch

class Answer(BaseAnswer):
    identifies_bcl2_as_target: bool = VerifiedField(
        description=(
            "True if the response identifies BCL2 (including Bcl-2, BCL-2, or "
            "B-cell lymphoma 2) as the direct pharmacological target of venetoclax. "
            "False if BCL2 is mentioned only as a pathway member or a different "
            "protein is identified as the primary target."
        ),
        ground_truth=True,
        verify_with=BooleanMatch(),
    )
    mentions_mechanism: bool = VerifiedField(
        description=(
            "True if the response explains the mechanism of action (e.g., inhibiting "
            "BCL2 to trigger apoptosis). False if only the target is named without "
            "any mechanistic explanation."
        ),
        ground_truth=True,
        verify_with=BooleanMatch(),
    )
'''

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

## Step 1: Create a TaskEval Instance

TaskEval is a recording and evaluation container. You log text into it, attach evaluation criteria, then run evaluation. Create one with an optional task ID and metadata for tracking.

```python
from karenina.benchmark.task_eval import TaskEval

task = TaskEval(
    task_id="drug-target-eval",
    metadata={"model": "claude-haiku-4-5", "scenario": "pharmacology"},
)

print(f"Created TaskEval: {task.task_id}")
```

> **Learn more**: [TaskEval Concepts](../core_concepts/task-eval.md)

---

## Step 2: Log the Output to Evaluate

TaskEval evaluates text that you supply. Use `log()` to record any string as content for evaluation.

Here we log an LLM response about a drug target. This could come from any source: an agent run, a CI pipeline, an external API, or a manual experiment.

```python
task.log(
    "The approved drug target of venetoclax is BCL2 (B-cell lymphoma 2). "
    "Venetoclax is a selective BCL2 inhibitor that works by displacing pro-apoptotic "
    "proteins, triggering programmed cell death in cancer cells [1]."
)

print(f"Logged {len(task.global_logs)} event(s)")
```

> **Learn more**: [Logging methods](../notebooks/task-eval/index.ipynb#step-2-log-outputs) · [Structured trace logging](../notebooks/task-eval/index.ipynb#trace-logging)

---

## Step 3: Define an Answer Template

An answer template is a Pydantic schema that defines what to extract from the logged output and how to verify it. Each field uses `VerifiedField` to declare what to extract, the correct value, and a verification primitive that checks the result.

```python
from karenina.schemas.entities import BaseAnswer, VerifiedField
from karenina.schemas.primitives import BooleanMatch


class Answer(BaseAnswer):
    identifies_bcl2_as_target: bool = VerifiedField(
        description=(
            "True if the response identifies BCL2 (including Bcl-2, BCL-2, or "
            "B-cell lymphoma 2) as the direct pharmacological target of venetoclax. "
            "False if BCL2 is mentioned only as a pathway member or a different "
            "protein is identified as the primary target."
        ),
        ground_truth=True,
        verify_with=BooleanMatch(),
    )
    mentions_mechanism: bool = VerifiedField(
        description=(
            "True if the response explains the mechanism of action (e.g., inhibiting "
            "BCL2 to trigger apoptosis). False if only the target is named without "
            "any mechanistic explanation."
        ),
        ground_truth=True,
        verify_with=BooleanMatch(),
    )


task.add_template(Answer)

print("Attached answer template with 2 verification fields")
```

> **Learn more**: [Answer Templates](../core_concepts/answer-templates.md) · [Template Authoring skill](/karenina-template-authoring)

---

## Step 4: Add Rubric Traits

While templates verify **correctness**, rubrics assess **quality**. Each trait evaluates one dimension of the response independently. Here we add an LLM-judged score trait and a regex pattern trait.

```python
from karenina.schemas.entities.rubric import LLMRubricTrait, RegexRubricTrait, Rubric

rubric = Rubric(
    llm_traits=[
        LLMRubricTrait(
            name="Conciseness",
            description=(
                "Rate how concise the response is on a scale of 1-5, where "
                "1 is very verbose and 5 is extremely concise."
            ),
            kind="score",
        ),
    ],
    regex_traits=[
        RegexRubricTrait(
            name="Has Citations",
            description="The response includes numbered citations in bracket notation (e.g., [1]).",
            pattern=r"\[\d+\]",
            case_sensitive=False,
        ),
    ],
)

task.add_rubric(rubric)

print(f"Added rubric with {len(rubric.llm_traits)} LLM trait(s) and {len(rubric.regex_traits)} regex trait(s)")
```

> **Learn more**: [Rubrics](../core_concepts/rubrics/index.md) · [All trait types](../core_concepts/rubrics/index.md): LLM, regex, callable, metric, literal

---

## Step 5: Run Evaluation

Configure the judge LLM and run evaluation. TaskEval uses `parsing_only=True` since no answering model is needed.

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

> **Learn more**: [Evaluation modes](../notebooks/core_concepts/evaluation-modes.ipynb) · [Model Config](../reference/configuration/model-config.md)

---

## Step 6: Inspect Results

`evaluate()` returns a `TaskEvalResult`. The quickest way to see what happened is `display()`, which prints a formatted report with template pass/fail status and rubric scores:

```python
print(result.display())
```

For a one-line overview, use `summary()`:

```python
print(result.summary())
```

You can also export the full results as JSON or Markdown for downstream analysis or archiving:

```python
print(result.export_json()[:300])
```

For programmatic access to individual fields (template verdicts, rubric scores, metadata), see the [results inspection reference](../notebooks/task-eval/index.ipynb#step-5-inspect-results).

> **Learn more**: [Results inspection](../notebooks/task-eval/index.ipynb#step-5-inspect-results) · [DataFrame analysis](../07-analyzing-results/dataframe-analysis.md)

---

## Bonus: Multi-Step Evaluation

TaskEval supports step-scoped evaluation for multi-phase agent workflows. Each step gets its own logs, templates, and rubric traits, evaluated independently.

```python
multi_task = TaskEval(task_id="multi-step-agent")

# Log per-step outputs
multi_task.log(
    "Found 3 relevant papers on venetoclax mechanism of action.",
    step_id="retrieval",
)
multi_task.log(
    "BCL2 is the primary target of venetoclax. It selectively inhibits BCL2, "
    "triggering apoptosis in CLL cells.",
    step_id="synthesis",
)

# Step-specific rubrics
multi_task.add_rubric(
    Rubric(llm_traits=[
        LLMRubricTrait(
            name="retrieval_quality",
            kind="boolean",
            description="True if relevant sources were found for the query.",
        )
    ]),
    step_id="retrieval",
)
multi_task.add_rubric(
    Rubric(llm_traits=[
        LLMRubricTrait(
            name="synthesis_accuracy",
            kind="boolean",
            description="True if the synthesis accurately reflects the retrieved information.",
        )
    ]),
    step_id="synthesis",
)

# Evaluate each step individually
for step_id in ["retrieval", "synthesis"]:
    step_result = multi_task.evaluate(config, step_id=step_id)
    step_eval = step_result.per_step[step_id]
    stats = step_eval.get_summary_stats()
    print(f"Step '{step_id}': {stats['rubric_traits_passed']}/{stats['rubric_traits_total']} traits passed")
```

> **Learn more**: [Multi-step evaluation](../notebooks/task-eval/index.ipynb#step-4-configure-and-evaluate) · [Step scoping](../core_concepts/task-eval.md#step-specific-evaluation)

---

## Next Steps

- **[TaskEval Concepts](../core_concepts/task-eval.md)**: Merge strategies, object structure, pipeline integration
- **[Answer Templates](../core_concepts/answer-templates.md)**: Field types, descriptions, and writing `verify()` methods
- **[Rubrics](../core_concepts/rubrics/index.md)**: All five trait types (LLM, regex, callable, metric, literal)
- **[Benchmark Quick Start](quickstart.md)**: If you need Karenina to generate responses and evaluate them

```python tags=["hide-cell"]
# Restore original LLM behavior and add_template
BaseChatModel.ainvoke = _original_ainvoke
_TaskEval.add_template = _original_add_template
```
