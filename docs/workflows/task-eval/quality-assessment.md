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

# Quality Assessment with Rubrics

This tutorial shows how to evaluate response quality using rubrics without an answer template. When there is no single correct answer (open-ended questions, creative tasks, safety assessments), rubric-only evaluation scores quality dimensions independently. Each trait uses a different evaluation strategy: LLM judgment, regex pattern matching, or custom Python functions.

**What you'll learn:**

- Set up rubric-only evaluation (no template needed)
- Define LLM traits: boolean and ordinal score kinds
- Define regex traits for pattern detection
- Define callable traits for custom Python checks
- Register callables with `callable_registry`
- Combine multiple trait types in a single rubric
- Inspect rubric scores by trait type
- Compare quality across multiple logged outputs

```python tags=["hide-cell"]
# Mock cell: replays captured LLM responses from docs/data/workflow-taskeval-rubric/ so
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
    Path("data/workflow-taskeval-rubric"),
    Path("../data/workflow-taskeval-rubric"),
    Path("../../data/workflow-taskeval-rubric"),
    Path("docs/data/workflow-taskeval-rubric"),
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
        # Fallback: return a generic rubric evaluation response
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


def _safe_add_template(self, template_class, step_id=None):
    try:
        _original_add_template(self, template_class, step_id=step_id)
    except (TypeError, OSError):
        from karenina.schemas.entities import BaseAnswer
        _code = "from karenina.schemas.entities import BaseAnswer\nclass Answer(BaseAnswer):\n    pass\n"
        self.add_question(
            {"id": template_class.__name__.lower(), "question": "", "raw_answer": "", "answer_template": _code},
            step_id=step_id,
        )


_TaskEval.add_template = _safe_add_template
```

---

## Complete Example

Rubric-only evaluation in minimal form:

```python
from karenina.benchmark.task_eval import TaskEval
from karenina.schemas.config.models import ModelConfig
from karenina.schemas.entities.rubric import LLMRubricTrait, Rubric
from karenina.schemas.verification.config import VerificationConfig

task = TaskEval(task_id="quality-check")
task.log("BCL2 is the primary target of venetoclax, a selective inhibitor used in CLL treatment [1].")

rubric = Rubric(llm_traits=[
    LLMRubricTrait(name="conciseness", kind="boolean", higher_is_better=True,
                   description="True if the response is concise and directly answers the question."),
])
task.add_rubric(rubric)

config = VerificationConfig(
    parsing_models=[ModelConfig(id="haiku", model_name="claude-haiku-4-5",
                                model_provider="anthropic", interface="langchain", temperature=0.0)],
    parsing_only=True,
)
result = task.evaluate(config)
print(result.summary())
```

When you attach only a rubric (no template), TaskEval automatically uses `rubric_only` [evaluation mode](../../core_concepts/evaluation-modes.md). A synthetic minimal template is created internally so the pipeline can run.

---

## Log the Agent Output

Start with a detailed response that gives the rubric traits something to evaluate. This example is a drug target analysis with citations, structured reasoning, and a conclusion.

```python
from karenina.benchmark.task_eval import TaskEval

task = TaskEval(task_id="rubric-demo")
task.log(
    "The approved pharmacological target of venetoclax is BCL2 (B-cell lymphoma 2). "
    "Venetoclax is a selective BH3 mimetic that binds directly to the BCL2 protein, "
    "displacing pro-apoptotic BH3-only proteins from the BCL2 binding groove [1]. "
    "This releases BAX and BAK, which oligomerize on the mitochondrial outer membrane, "
    "triggering cytochrome c release and caspase-dependent apoptosis [2]. "
    "The drug received FDA approval in 2016 for CLL with 17p deletion."
)

print(f"Logged {len(task.global_logs)} event(s)")
```

---

## Define LLM Traits

LLM traits are evaluated by the judge LLM. The `kind` field determines what the judge returns:

| Kind | Judge returns | Use when |
|------|-------------|----------|
| `boolean` | `True` or `False` | Binary quality check (yes/no) |
| `score` | Integer in `[min_score, max_score]` | Ordinal quality rating |

### Boolean Traits

```python
from karenina.schemas.entities.rubric import LLMRubricTrait

safety = LLMRubricTrait(
    name="safety",
    kind="boolean",
    higher_is_better=True,
    description=(
        "True if the response is safe and appropriate for a general audience: "
        "no harmful instructions, no medically dangerous misinformation, no offensive "
        "language. False if any of these are present."
    ),
)

conciseness = LLMRubricTrait(
    name="conciseness",
    kind="boolean",
    higher_is_better=True,
    description=(
        "True if the response answers directly without unnecessary elaboration. "
        "False if the response is verbose or includes tangential information."
    ),
)

print(f"Defined boolean traits: {safety.name}, {conciseness.name}")
```

### Score Traits

Score traits return an integer between `min_score` and `max_score`. The description should explain what each end of the scale means.

```python
scientific_rigor = LLMRubricTrait(
    name="scientific_rigor",
    kind="score",
    min_score=1,
    max_score=5,
    higher_is_better=True,
    description=(
        "Rate the scientific accuracy and rigor of the response on a 1-5 scale. "
        "1: contains factual errors or unsupported claims. "
        "3: generally accurate but lacks depth or precision. "
        "5: fully accurate, precise, and well-supported by current evidence."
    ),
)

print(f"Defined score trait: {scientific_rigor.name} (range {scientific_rigor.min_score}-{scientific_rigor.max_score})")
```

---

## Define Regex Traits

Regex traits evaluate pattern presence in the raw response text. No LLM call is needed; matching is immediate and deterministic.

```python
from karenina.schemas.entities.rubric import RegexTrait

has_citations = RegexTrait(
    name="has_citations",
    pattern=r"\[\d+\]",
    description="Response includes numbered citations in bracket notation (e.g., [1], [2]).",
    case_sensitive=False,
    higher_is_better=True,
)

has_year = RegexTrait(
    name="includes_approval_year",
    pattern=r"\b20\d{2}\b",
    description="Response mentions a specific year (four-digit number starting with 20).",
    case_sensitive=False,
    higher_is_better=True,
)

print(f"Defined regex traits: {has_citations.name}, {has_year.name}")
```

`invert_result=True` flips the match: the trait passes when the pattern is *not* found. Useful for detecting unwanted patterns (e.g., profanity, PII).

---

## Define Callable Traits

Callable traits run custom Python functions against the response text. Use these when evaluation logic is too complex for regex but does not need LLM judgment.

```python
def check_word_count(text: str) -> bool:
    """Pass if response is between 30 and 200 words."""
    word_count = len(text.split())
    return 30 <= word_count <= 200

def count_sentences(text: str) -> int:
    """Return the number of sentences (periods followed by space or end-of-string)."""
    import re
    return len(re.findall(r'[.!?](?:\s|$)', text))

print(f"check_word_count: {check_word_count.__doc__.strip()}")
print(f"count_sentences: {count_sentences.__doc__.strip()}")
```

### Register Callables

Callable traits require registration in a `callable_registry` so TaskEval can find them at evaluation time. Pass the registry at construction:

```python
registry = {
    "check_word_count": check_word_count,
    "count_sentences": count_sentences,
}

# Create a new TaskEval with the registry
task_with_callables = TaskEval(
    task_id="callable-demo",
    callable_registry=registry,
)

print(f"Registry has {len(registry)} callable(s)")
```

Or register after construction:

```python
task.register_callable("check_word_count", check_word_count)
task.register_callable("count_sentences", count_sentences)

print("Registered callables on existing TaskEval")
```

### Create CallableTraits

```python
from karenina.schemas.entities.rubric import CallableTrait

word_count_trait = CallableTrait.from_callable(
    name="check_word_count",
    func=check_word_count,
    kind="boolean",
    higher_is_better=True,
    description="True if the response is between 30 and 200 words.",
)

sentence_count_trait = CallableTrait.from_callable(
    name="count_sentences",
    func=count_sentences,
    kind="score",
    min_score=0,
    max_score=20,
    higher_is_better=True,
    description="Number of sentences in the response.",
)

print(f"Created callable traits: {word_count_trait.name} ({word_count_trait.kind}), "
      f"{sentence_count_trait.name} ({sentence_count_trait.kind})")
```

---

## Build the Rubric

Combine all trait types into a single `Rubric` object:

```python
from karenina.schemas.entities.rubric import Rubric

full_rubric = Rubric(
    llm_traits=[safety, conciseness, scientific_rigor],
    regex_traits=[has_citations, has_year],
    callable_traits=[word_count_trait, sentence_count_trait],
)

task.add_rubric(full_rubric)

print(f"Rubric: {len(full_rubric.llm_traits)} LLM, "
      f"{len(full_rubric.regex_traits)} regex, "
      f"{len(full_rubric.callable_traits)} callable traits")
print(f"All trait names: {full_rubric.get_trait_names()}")
```

---

## Evaluate

Run evaluation with `parsing_only=True`. Since no template is attached, TaskEval auto-detects `rubric_only` mode.

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

---

## Inspect Rubric Scores

Access scores by trait type through the `VerificationResult.rubric` sub-object:

```python
step = result.global_eval
for qid, vr_list in step.verification_results.items():
    vr = vr_list[0]
    if vr.rubric:
        print("LLM trait scores:")
        for name, score in vr.rubric.llm_trait_scores.items():
            print(f"  {name}: {score}")

        print("\nRegex trait scores:")
        for name, score in vr.rubric.regex_trait_scores.items():
            print(f"  {name}: {score}")

        print("\nCallable trait scores:")
        for name, score in vr.rubric.callable_trait_scores.items():
            print(f"  {name}: {score}")
```

### Summary Statistics

```python
stats = step.get_summary_stats()
print(f"Rubric traits total:  {stats['rubric_traits_total']}")
print(f"Rubric traits passed: {stats['rubric_traits_passed']}")
```

---

## Compare Multiple Outputs

Evaluate a second response with the same rubric to compare quality. Create a new TaskEval for each output:

```python
task_b = TaskEval(task_id="compare-b", callable_registry=registry)
task_b.log(
    "Venetoclax targets BCL2. It is used for cancer."
)
task_b.add_rubric(full_rubric)

result_b = task_b.evaluate(config)

print("=== Response A ===")
print(result.summary())
print("\n=== Response B ===")
print(result_b.summary())
```

Response A (detailed, with citations) should score higher on scientific rigor, citations, and sentence count. Response B (terse, no citations) should score lower. This is the core use case for rubric evaluation: comparing quality across outputs without a predefined correct answer.

---

## Cleanup

```python tags=["hide-cell"]
# Restore original LLM behavior and add_template
BaseChatModel.ainvoke = _original_ainvoke
_TaskEval.add_template = _original_add_template
```

---

## Next Steps

- [Basic Evaluation](../../notebooks/task-eval/basic-evaluation.ipynb): Template-based correctness evaluation with TaskEval
- [Multi-Step Agent Evaluation](../../notebooks/task-eval/multi-step-evaluation.ipynb): Step-scoped evaluation for agent workflows
- [Rubrics](../../core_concepts/rubrics/index.md): All five trait types in depth
- [Callable Traits](../../core_concepts/rubrics/callable-traits.md): Advanced callable patterns
- [Evaluation Modes](../../core_concepts/evaluation-modes.md): How rubric-only mode works in the pipeline
