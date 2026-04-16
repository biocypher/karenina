---
jupyter:
  jupytext:
    formats: docs/core_concepts/rubrics//md,docs/notebooks/core_concepts/rubrics//ipynb
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

# Agentic Rubric Traits

Agentic rubric traits deploy an **agent with tool access** to investigate workspace artifacts before producing a structured score. They are the rubric trait type to use when evaluation requires examining files, running code, or navigating a workspace directory, not just reading the response text. For an overview of all rubric trait types, see the [rubrics index](../../../../core_concepts/rubrics/).

```python tags=["hide-cell"]
# Mock cell: builds pre-computed result objects for the examples below.
# This cell is hidden in rendered documentation.
from karenina.schemas.verification import VerificationResult
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.schemas.verification.result_components import (
    VerificationResultMetadata,
    VerificationResultRubric,
    VerificationResultTemplate,
)

_answering = ModelIdentity(model_name="claude-sonnet-4-20250514", interface="claude_agent_sdk")
_parsing = ModelIdentity(model_name="claude-sonnet-4-20250514", interface="claude_agent_sdk")
_ts = "2026-03-15T12:00:00+00:00"
_qid = "q_logistic_regression"
_rid = VerificationResultMetadata.compute_result_id(_qid, _answering, _parsing, _ts)

_mock_result = VerificationResult(
    metadata=VerificationResultMetadata(
        question_id=_qid,
        template_id="tmpl_agentic",
        failure=None,
        caveats=[],
        question_text="Write a logistic regression classifier for the Iris dataset.",
        answering=_answering,
        parsing=_parsing,
        execution_time=14.3,
        timestamp=_ts,
        result_id=_rid,
    ),
    template=VerificationResultTemplate(
        raw_llm_response="Here is a logistic regression classifier using statsmodels...",
        template_verification_performed=True,
        verify_result=True,
    ),
    rubric=VerificationResultRubric(
        agentic_trait_scores={
            "logistic_regression_library": 0,
            "code_runs_without_error": True,
        },
        agentic_trait_investigation_traces={
            "logistic_regression_library": (
                "Investigation: Scanned workspace for Python files. Found main.py. "
                "Checked imports: `import statsmodels.api as sm`. The code uses "
                "statsmodels.Logit for logistic regression rather than scikit-learn. "
                "Classification: statsmodels."
            ),
            "code_runs_without_error": (
                "Investigation: Executed `python main.py` in the workspace. "
                "Exit code 0, no tracebacks. The script ran to completion and printed "
                "accuracy results. Result: True."
            ),
        },
    ),
)
```

## 1. What Agentic Rubric Traits Are

An `AgenticRubricTrait` launches an agent that can read files, execute code, and explore the workspace before the system extracts a structured score from the investigation trace. This is fundamentally different from [LLM traits](../llm-traits/), which evaluate qualities visible in the response text through a single parsing call with no tool access.

Use `AgenticRubricTrait` when the evaluation depends on **artifacts the answering model produced** (generated files, code output, database entries) or on **workspace state that cannot be inferred from the response text alone**. If the check can be made by reading the response, prefer an [LLM trait](../llm-traits/) or a local check ([regex](../regex-traits/), [callable](../callable-traits/)).

### 1.1 Scope Boundary

This page covers how to define, configure, and interpret agentic rubric traits. It does not cover the pipeline stage mechanics or advanced configuration of evaluation strategies; those are documented in [Agentic Rubric Evaluation](../../../../advanced-pipeline/agentic-rubric-evaluation/) (forthcoming).

| Evaluates response text (use LLM traits) | Evaluates workspace artifacts (use agentic traits) |
|-------------------------------------------|----------------------------------------------------|
| "Does the response cite specific trials?" | "Does the generated code import scikit-learn?" |
| "Is the reasoning presented as linked steps?" | "Does the output CSV contain the expected columns?" |
| "Does the response hedge on off-label use?" | "Does `python main.py` exit without errors?" |

## 2. Core Idea: Investigate, Then Extract

The most important idea is the **two-step pattern**: investigate first, extract second.

1. **Investigation.** An agent (with tool access) receives the question, optionally the response trace, and optionally a workspace directory path. It investigates by reading files, running commands, or any action its tools allow. The output is a free-form investigation trace.

2. **Extraction.** A parser reads the investigation trace and extracts a structured score (boolean, integer, or literal class index) matching the trait's `kind`.

This separation matters because investigation is open-ended and may require multiple tool calls, while extraction is a single structured-output parse. The agent does the hard work; the parser standardizes the result.

```
Question + Response Trace + Workspace Path
                │
                ▼
┌─── Investigation ──────────────────────────┐
│  Agent with tools:                         │
│    reads files, runs code, navigates dirs  │
│                                            │
│  Output: free-form investigation trace     │
└────────────────┬───────────────────────────┘
                 │
                 ▼
┌─── Extraction ─────────────────────────────┐
│  Parser (structured output):               │
│    reads investigation trace               │
│    returns bool / int / class name         │
└────────────────┬───────────────────────────┘
                 │
                 ▼
        Stored in VerificationResult.rubric
          .agentic_trait_scores
          .agentic_trait_investigation_traces
```

## 3. Overview: Field Reference

`AgenticRubricTrait` shares the same `kind` system as `LLMRubricTrait` (boolean, score, literal) but adds agent-specific configuration fields.

### 3.1 Core Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | `str` | Yes | | Human-readable identifier (must be unique within the rubric) |
| `description` | `str` | Yes | | Evaluation instructions for the agent. This is the primary channel for telling the agent what to investigate and what counts as evidence |
| `kind` | `"boolean"` / `"score"` / `"literal"` / `type[BaseModel]` | Yes | | Shape of the result. String literals produce scalar scores; a `BaseModel` subclass (template kind) produces structured multi-field findings (see Section 5.4) |
| `higher_is_better` | `bool \| None` | Yes | | Whether higher values indicate better performance. Must be `None` for template kind, because structured results have no single direction |
| `min_score` | `int \| None` | No | `1` | Lower bound for score traits. Auto-derived for literal |
| `max_score` | `int \| None` | No | `5` | Upper bound for score traits. Auto-derived for literal |
| `classes` | `dict[str, str] \| None` | Literal only | `None` | Class name to description mapping. Required when `kind="literal"` |

### 3.2 Agent Configuration Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `context_mode` | `"workspace_only"` / `"trace_and_workspace"` / `"trace_only"` | `"trace_and_workspace"` | What context the agent receives (see Section 4.1) |
| `max_turns` | `int` | `15` | Maximum agent think-act cycles before the investigation stops |
| `timeout_seconds` | `int` | `120` | Wall-clock timeout for the investigation step |
| `materialize_trace` | `bool` | `False` | Write the answering agent trace to a file in the workspace instead of inlining it in the investigation prompt. The investigation agent receives the file path and can use grep/search tools on it. Requires `context_mode` that includes the trace (`"trace_only"` or `"trace_and_workspace"`). See Section 5.5 |
| `persist_trace` | `bool` | `False` | When `True`, the materialized trace file is kept after evaluation. When `False` (default), cleaned up after evaluation completes |
| `model_override` | `ModelConfig \| None` | `None` | Use a specific model for this trait instead of the pipeline's parsing model. The model's interface must have a registered `agent_factory` |

## 4. How It Works

### 4.1 Context Modes

The `context_mode` field controls what information the agent receives at the start of the investigation.

| Mode | Agent sees | Agent does NOT see | Best for |
|------|-----------|-------------------|----------|
| `"trace_and_workspace"` (default) | Question, response trace, workspace path | Nothing hidden | Most evaluations: the agent can cross-reference the response with workspace artifacts |
| `"workspace_only"` | Question, workspace path | Response trace | Evaluations that should judge only the produced artifacts, ignoring how the model described its work |
| `"trace_only"` | Question, response trace | Workspace path | Evaluations where the agent needs tool access for analysis (e.g., running code it writes itself) but should not browse the workspace |

### 4.2 Execution Lifecycle

During the [verification pipeline](../../verification-pipeline/), agentic rubric traits are evaluated in Stage 11b (AgenticRubricEvaluation), which runs after the standard RubricEvaluation stage (Stage 11) and before DeepJudgmentRubric (Stage 12). The stage only runs when the rubric contains at least one `AgenticRubricTrait`.

For each agentic trait, the pipeline:

1. Resolves the model: uses `trait.model_override` if set, otherwise falls back to the pipeline's parsing model.
2. Validates that the resolved model's interface has an `agent_factory` registered. If not, the trait is skipped with a warning.
3. Runs the investigation step (agent with tools, bounded by `max_turns` and `timeout_seconds`).
4. Runs the extraction step (structured parser call on the investigation trace).
5. Stores the score in `VerificationResult.rubric.agentic_trait_scores` and the trace in `VerificationResult.rubric.agentic_trait_investigation_traces`.

### 4.3 Evaluation Strategies

`VerificationConfig.agentic_rubric_strategy` controls how multiple agentic traits are evaluated.

| Strategy | Behavior | When to use |
|----------|----------|-------------|
| `"individual"` (default) | One agent session per trait. Each trait can use a different model via `model_override` | Robust, isolated. Use when traits require different tools or models, or when you want independent investigation traces |
| `"shared"` | One shared agent investigates all traits in a single session, then per-trait extraction pulls individual scores | Faster when all traits use the same model. Falls back to individual if traits resolve to different models |

## 5. Patterns and Worked Examples

### 5.1 Boolean Trait: Code Executes Without Error

A boolean agentic trait checks a binary condition by running code in the workspace.

```python
from karenina.schemas.entities.rubric import AgenticRubricTrait

runs_cleanly = AgenticRubricTrait(
    name="code_runs_without_error",
    description=(
        "Execute the main Python script in the workspace directory. "
        "Answer True if the script exits with code 0 and produces no "
        "tracebacks or unhandled exceptions. Answer False if execution "
        "fails, raises an exception, or exits with a non-zero code."
    ),
    kind="boolean",
    higher_is_better=True,
    context_mode="trace_and_workspace",
    max_turns=10,
    timeout_seconds=60,
)

print(f"Trait: {runs_cleanly.name}")
print(f"Kind: {runs_cleanly.kind}")
print(f"Context mode: {runs_cleanly.context_mode}")
```

### 5.2 Literal Trait: Library Detection

A literal agentic trait classifies the response into one of several named categories. This example detects which machine learning library the generated code uses by inspecting actual import statements in workspace files.

```python
library_trait = AgenticRubricTrait(
    name="logistic_regression_library",
    description=(
        "Inspect the Python files in the workspace directory. Identify which "
        "library the code uses for logistic regression. Check import statements "
        "and function calls. Classify based on the primary library used for the "
        "logistic regression model itself, not for data loading or plotting."
    ),
    kind="literal",
    classes={
        "statsmodels": "Uses statsmodels (e.g., statsmodels.api.Logit or sm.Logit)",
        "sklearn": "Uses scikit-learn (e.g., sklearn.linear_model.LogisticRegression)",
        "pytorch": "Uses PyTorch (e.g., torch.nn.Linear with sigmoid activation)",
        "other": "Uses a different library or a from-scratch implementation",
    },
    higher_is_better=False,  # No intrinsic ordering; labels matter more than indices
    context_mode="workspace_only",  # Only inspect files, ignore the response text
    max_turns=10,
    timeout_seconds=60,
)

print(f"Trait: {library_trait.name}")
print(f"Classes: {list(library_trait.classes.keys())}")
print(f"Score range: {library_trait.min_score} to {library_trait.max_score}")
```

### 5.3 Score Trait: Test Coverage Quality

A score agentic trait rates a quality on a numeric scale after investigating the workspace.

```python
test_quality = AgenticRubricTrait(
    name="test_coverage_quality",
    description=(
        "Examine the test files in the workspace. Rate the quality of test "
        "coverage for the main module on a 1-5 scale. "
        "1 = no tests or only trivial smoke tests. "
        "3 = tests cover the main functionality but miss edge cases or error paths. "
        "5 = comprehensive tests covering normal paths, edge cases, and error handling."
    ),
    kind="score",
    min_score=1,
    max_score=5,
    higher_is_better=True,
    context_mode="workspace_only",
    max_turns=15,
    timeout_seconds=90,
)

print(f"Trait: {test_quality.name}")
print(f"Score range: {test_quality.min_score}-{test_quality.max_score}")
```

### 5.4 Template Kind: Agentic Evaluation with Structured Output

The scalar kinds (boolean, score, literal) force each trait to produce a single value. Template kind removes that constraint: you pass a Pydantic `BaseModel` subclass as `kind`, and the investigation agent populates the entire schema on the fly. The result is a structured, multi-field evaluation output whose shape you define.

This is the same investigate-then-extract pattern as scalar kinds, but the extraction step parses the investigation trace into your Pydantic class rather than into a single boolean, integer, or class index. You get the full power of agentic evaluation (tool use, multi-step reasoning, workspace access) combined with the expressiveness of a Pydantic template.

The potential is broad: any evaluation question that naturally produces multiple related findings can be captured in a single trait instead of being split across several scalar traits or lost to free-text summaries. Compliance audits that need to flag several criteria at once, code reviews that track multiple quality dimensions, trace analyses that count and categorize patterns: all become a single trait with a schema that matches the evaluation's natural shape.

```python
from pydantic import BaseModel, Field
from karenina.schemas.entities.rubric import AgenticRubricTrait


class CodeQualityFindings(BaseModel):
    """The agent investigates the workspace and fills this in."""

    has_type_hints: bool = Field(
        description="True if the code uses type annotations on function signatures."
    )
    test_count: int = Field(
        description="Number of test functions found in test files."
    )
    external_dependencies: list[str] = Field(
        description="Third-party packages imported by the code."
    )


trait = AgenticRubricTrait(
    name="code_quality",
    description=(
        "Examine the Python files in the workspace. Check whether functions "
        "have type annotations. Count the test functions. List every "
        "third-party package that is imported."
    ),
    kind=CodeQualityFindings,
    higher_is_better=None,  # required: no single direction for multi-field output
    context_mode="trace_and_workspace",
)

print(f"Trait: {trait.name}")
print(f"Template kind: {trait.is_template_kind}")
print(f"Fields: {list(CodeQualityFindings.model_fields.keys())}")
```

Because the output has multiple fields with potentially different meanings, `higher_is_better` is set to `None`. Results are stored as flat dot-notation keys in `agentic_trait_scores` (`code_quality.has_type_hints`, `code_quality.test_count`, `code_quality.external_dependencies`), making them easy to access in DataFrames and downstream analysis.

### 5.5 Trace Materialization

Agent traces from multi-turn agentic workflows can be very large. Rather than embedding the entire trace in the investigation prompt, `materialize_trace=True` writes it to a file and gives the investigation agent the file path. The agent can then use file tools (grep, search, read) to examine the trace selectively, which is both more efficient and more effective for targeted analysis.

Set `persist_trace=True` to keep the file after evaluation for inspection or debugging; by default it is cleaned up.

```python
materialized = AgenticRubricTrait(
    name="trace_review",
    description="Review the agent trace for tool call patterns.",
    kind="boolean",
    higher_is_better=True,
    context_mode="trace_only",
    materialize_trace=True,
    persist_trace=True,  # keep the file for inspection after evaluation
)
```

Trace materialization requires a context mode that includes the trace (`"trace_only"` or `"trace_and_workspace"`). It pairs naturally with template kind: the investigation agent greps a large trace for specific patterns, and the structured Pydantic output captures exactly the findings you care about.

### 5.6 Reading Results

After verification, agentic trait scores and investigation traces are stored in `VerificationResult.rubric`. The following cells use the `_mock_result` from the hidden setup cell.

```python
# Access agentic trait scores
rubric = _mock_result.rubric
print("Agentic trait scores:", rubric.agentic_trait_scores)
```

```python
# Access investigation traces
for trait_name, trace in rubric.agentic_trait_investigation_traces.items():
    print(f"\n--- {trait_name} ---")
    print(trace)
```

```python
# Use get_all_trait_scores() to get a flat dict across all trait types
all_scores = rubric.get_all_trait_scores()
print("All trait scores:", all_scores)
```

```python
# Look up a single trait by name
found = rubric.get_trait_by_name("logistic_regression_library")
if found:
    value, trait_type = found
    print(f"Value: {value}, Type: {trait_type}")
```

### 5.7 Score Validation

Like `LLMRubricTrait`, agentic traits provide a `validate_score` method that checks whether a given value is valid for the trait's kind and range.

```python
# Boolean trait: only accepts bool
print(runs_cleanly.validate_score(True))    # True
print(runs_cleanly.validate_score(1))       # False (int rejected for boolean)

# Score trait: accepts int in [min_score, max_score]
print(test_quality.validate_score(3))       # True
print(test_quality.validate_score(6))       # False (above max_score)

# Literal trait: accepts int in [0, len(classes)-1] or -1 (error state)
print(library_trait.validate_score(0))      # True (statsmodels)
print(library_trait.validate_score(-1))     # True (error state)
print(library_trait.validate_score(4))      # False (out of range)
```

## 6. When to Use Agentic Traits

**Litmus test:** can the evaluation be done from the response text alone?

- **Yes** (the quality is visible in what the model wrote): use an [LLM trait](../llm-traits/), [regex trait](../regex-traits/), or [callable trait](../callable-traits/).
- **No** (the evaluation requires inspecting files, running code, or checking workspace state): use an `AgenticRubricTrait`.

| Scenario | Trait type | Why |
|----------|-----------|-----|
| "Does the response cite specific evidence?" | LLM trait | Visible in the text |
| "Does the generated code use scikit-learn?" | Agentic trait | Requires reading workspace files |
| "Does the response contain a URL?" | Regex trait | Pattern match on text |
| "Does `python main.py` produce correct output?" | Agentic trait | Requires executing code |
| "Is the response under 200 words?" | Callable trait | Deterministic local check |
| "Does the output CSV have the right columns?" | Agentic trait | Requires reading a generated file |

**Cost and latency considerations.** Agentic traits are more expensive than other trait types because each evaluation involves at least one agent session (multiple LLM calls with tool use) plus an extraction call. Use them selectively for checks that genuinely require workspace access.

## 7. Next Steps

- [LLM traits](../llm-traits/): text-based evaluation with the parsing model
- [Regex traits](../regex-traits/): deterministic pattern matching
- [Callable traits](../callable-traits/): custom Python functions
- [Metric traits](../metric-traits/): precision, recall, F1 computation
- [Rubrics index](../../../../core_concepts/rubrics/): overview of all trait types and how to choose
- [Evaluation modes](../../evaluation-modes/): template_only, template_and_rubric, rubric_only
