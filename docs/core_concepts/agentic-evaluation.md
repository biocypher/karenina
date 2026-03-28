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

# Agentic Evaluation

Karenina's classical workflow evaluates factual Q&A: the answering model produces text, the judge parses it into structured fields, and verification primitives compare those fields against ground truth. No tools are needed on either side. Agentic evaluation extends this to coding and data analysis tasks, where both the answering model and the judge need tool access to work in a real workspace.

```python tags=["hide-cell"]
# Setup cell: provides mock objects for executable examples.
# This cell is hidden in rendered documentation.
import datetime
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock

from karenina import Benchmark
from karenina.schemas.config import ModelConfig
from karenina.schemas.entities.question import Question
from karenina.schemas.results import VerificationResultSet
from karenina.schemas.verification import (
    ModelIdentity,
    VerificationConfig,
    VerificationResult,
    VerificationResultMetadata,
    VerificationResultTemplate,
)

# Create a mock benchmark with workspace support
_workspace_root = Path(tempfile.mkdtemp()) / "tasks"
_workspace_root.mkdir(parents=True, exist_ok=True)
(_workspace_root / "workspace").mkdir(exist_ok=True)

_benchmark = Benchmark.create(
    name="Coding Tasks",
    description="Agentic evaluation of coding tasks",
    version="1.0.0",
    workspace_root=_workspace_root,
)
_benchmark.add_question(
    question="Fix the bug in calculator.py so that division by zero returns an error message.",
    raw_answer="Division by zero handled with try/except",
)
_qids = _benchmark.get_question_ids()

# Build a mock VerificationResult with agentic fields
_answering = ModelIdentity(
    model_name="claude-sonnet-4-20250514", interface="claude_agent_sdk"
)
_parsing = ModelIdentity(
    model_name="claude-sonnet-4-20250514", interface="claude_agent_sdk"
)
_ts = datetime.datetime.now(tz=datetime.UTC).isoformat()
_rid = VerificationResultMetadata.compute_result_id(_qids[0], _answering, _parsing, _ts)

_mock_result = VerificationResult(
    metadata=VerificationResultMetadata(
        question_id=_qids[0],
        template_id="tmpl_agentic",
        completed_without_errors=True,
        question_text="Fix the bug in calculator.py so that division by zero returns an error message.",
        raw_answer="Division by zero handled with try/except",
        answering=_answering,
        parsing=_parsing,
        execution_time=8.5,
        timestamp=_ts,
        result_id=_rid,
    ),
    template=VerificationResultTemplate(
        raw_llm_response="I fixed the division by zero bug by adding a try/except block.",
        verify_result=True,
        template_verification_performed=True,
        agentic_parsing_performed=True,
        investigation_trace="Agent examined workspace: read calculator.py, found try/except block for ZeroDivisionError, ran test suite (3/3 passed).",
        parsed_gt_response={"bug_fixed": True, "tests_passing": True},
        parsed_llm_response={"bug_fixed": True, "tests_passing": True},
    ),
)
_mock_result_set = VerificationResultSet(results=[_mock_result])

# Save benchmark so Benchmark.load works
_tmp_path = Path(tempfile.mkdtemp()) / "coding-tasks.jsonld"
_benchmark.save(str(_tmp_path))

# Patch Benchmark.load to return our mock benchmark
_orig_load = Benchmark.load


def _patched_load(path, workspace_root=None):
    _benchmark.set_workspace_root(workspace_root or _workspace_root)
    return _benchmark


Benchmark.load = classmethod(lambda cls, path, workspace_root=None: _patched_load(path, workspace_root))

# Patch run_verification to return mock results
_orig_run = Benchmark.run_verification


def _patched_run(self, config, **kwargs):
    return _mock_result_set


Benchmark.run_verification = _patched_run

# Create a Question object for the workspace_path example
question = _benchmark.get_question_as_object(_qids[0])
```

This page explains the concepts behind agentic evaluation: how workspaces work, how adapters differ in their agent support, how the judge independently verifies artifacts, and how configuration is split between `Benchmark` and `VerificationConfig`.

## 1. Two Kinds of Evaluation Tasks

### 1.1. Factual Q&A Tasks

The answering model receives a question and produces a text response. The [judge LLM](adapters.md) parses that response into the structured fields of an [answer template](answer-templates.md), and [verification primitives](verification-primitives.md) check each field against ground truth. Neither the answering model nor the judge needs tools. This is karenina's classical workflow, covered in the [verification pipeline](verification-pipeline.md) documentation.

### 1.2. Coding and Data Analysis Tasks

The answering model needs tool access to work in a workspace: reading files, writing code, executing scripts, running tests. A text response alone is insufficient because the meaningful output is artifacts on disk (files created, tests passed, code compiled), not the text of the conversation.

The judge also needs tool access to independently verify those artifacts. Parsing the answering agent's trace is not enough because the judge cannot confirm that the workspace actually reflects what the trace describes. The agent may have produced the right conversational output while leaving broken code on disk, or it may have silently overwritten files the trace does not mention.

Agentic evaluation addresses both needs. The answering model runs as an agent with tool access in a workspace. The judge runs as a separate agent (also with tool access) that independently inspects the workspace, re-runs tests, and checks file contents before extracting structured data into the [answer template](answer-templates.md).

## 2. Workspaces

A workspace is a directory on the local filesystem where the answering agent and the judge operate. The workspace belongs to the benchmark (it is where the task's code, data, and test files live), not to the verification configuration.

### 2.1. Setting Up Workspaces

Set the workspace root on the benchmark. Each question can optionally reference a subdirectory within the root:

```python
from pathlib import Path
from karenina.benchmark import Benchmark

benchmark = Benchmark.load("coding-tasks.jsonld", workspace_root=Path("/data/tasks"))

# Or set it after loading
benchmark.set_workspace_root(Path("/data/tasks"))
```

Per-question subdirectories are set via `Question.workspace_path`, which is a relative path from the workspace root:

```python
question.workspace_path = "task_01"  # Resolves to /data/tasks/task_01
```

When `workspace_path` is set, the pipeline expects a pre-existing directory at `workspace_root / workspace_path` containing the starter code, tests, or other artifacts for the task. When `workspace_path` is not set, the pipeline creates an empty directory for the question.

### 2.2. Workspace Copying

By default, the pipeline copies each question's workspace to a sibling working directory before the agent runs (`workspace_copy=True` on [VerificationConfig](../reference/configuration/verification-config/)). This protects the original workspace for re-runs. The copy receives a unique suffix based on timestamp, process ID, and replicate index, making it safe for parallel execution.

When `workspace_copy=False`, the pipeline operates directly in the original directory. This is destructive: the agent may modify files that cannot be recovered.

After the run completes, `workspace_cleanup=True` (the default) deletes the working copy. It never deletes the original source directory. Set `workspace_cleanup=False` to preserve working copies for post-hoc inspection.

### 2.3. Workspace Paths in Checkpoints

`Question.workspace_path` is persisted in checkpoints as a relative path. `Benchmark.workspace_root` is not persisted (it is a local filesystem path that varies between machines). When loading a checkpoint on a different machine, supply the new workspace root:

```python
benchmark = Benchmark.load("checkpoint.jsonld", workspace_root=Path("/new/machine/tasks"))
```

## 3. Deep Agent vs. Tool Loop Adapters

Not all [adapters](adapters.md) are equal when it comes to agent capabilities. Karenina distinguishes two tiers via the `agent_tier` field on `AdapterSpec`:

### 3.1. Tool Loop Adapters (`agent_tier="tool_loop"`)

Tool loop adapters (LangChain, Claude Tool) build agent behavior by orchestrating individual LLM calls. The adapter explicitly manages the tool call loop: it sends a message, checks for tool calls in the response, executes the tools, sends the results back, and repeats. Every intermediate step is visible to the adapter and captured in the trace.

### 3.2. Deep Agent Adapters (`agent_tier="deep_agent"`)

Deep agent adapters wrap a runtime that is itself an agent with built-in tools (e.g., Claude Code via `claude_agent_sdk`). The runtime manages its own tool call loop internally. The adapter hands off a prompt and receives a completed trace.

This distinction matters for the pipeline. For deep agent adapters, the pipeline always uses `AgentPort` (not `LLMPort`) to generate answers, because the `LLMPort` path cannot capture the intermediate tool calls that the runtime executes internally. Using `LLMPort` on a deep agent adapter would produce a response but lose the full tool call trace.

### 3.3. The `agent_tier` Field

The field is declared on `AdapterSpec` in the [adapter registry](../../src/karenina/adapters/registry.py):

```python
@dataclass
class AdapterSpec:
    """Adapter specification in the registry."""
    agent_tier: str = "tool_loop"
```

When `agent_tier="deep_agent"`, `GenerateAnswerStage` (Stage 2) automatically selects the `AgentPort` path even when no MCP servers are configured. This is the same path used for MCP-enabled answering models; the tier simply makes it the default for adapters that need it.

Currently, `claude_agent_sdk` is the only adapter that sets `agent_tier="deep_agent"`.

## 4. Two-Step Agentic Judging (Stage 7b)

When `agentic_parsing=True` in [VerificationConfig](../reference/configuration/verification-config/), the pipeline replaces the classical `ParseTemplateStage` (Stage 7a) with `AgenticParseTemplateStage` (Stage 7b). The two-step process separates investigation from extraction:

### 4.1. Step 1: Investigation

An agent (`AgentPort`) with tool access examines the workspace. It can read files, execute code, run tests, and check outputs. The investigation agent receives:

- The question text (always)
- The answering agent's trace (if `agentic_judge_context` includes it)
- The workspace path (if `agentic_judge_context` includes it)
- The answer template's JSON schema (as a target format for its findings)

The `agentic_judge_context` setting controls what context the investigation agent receives:

| `agentic_judge_context` | Question | Answering trace | Workspace path | Use case |
|-------------------------|:--------:|:---------------:|:--------------:|----------|
| `"workspace_only"` (default) | Yes | No | Yes | Maximum independence: judge verifies from scratch |
| `"trace_and_workspace"` | Yes | Yes | Yes | Judge can cross-reference trace claims against workspace state |
| `"trace_only"` | Yes | Yes | No | Equivalent to classical Stage 7a parsing (no workspace access) |

### 4.2. Step 2: Extraction

A parser (`ParserPort`) with structured output support extracts the investigation findings into the [answer template](answer-templates.md) schema. This step uses the same parser infrastructure as classical Stage 7a: it receives the investigation trace as input and produces a filled `Answer` instance as output.

Separating investigation from extraction keeps each step focused. The investigation agent explores freely with tools; the extraction parser produces clean structured output from the investigation report.

### 4.3. Pipeline Integration

`AgenticParseTemplateStage` sits in the same position as `ParseTemplateStage` in the stage sequence. The [StageOrchestrator](../../src/karenina/benchmark/verification/stages/core/orchestrator.py) selects one or the other based on the `agentic_parsing` flag:

```
Stage 6: SufficiencyCheck (optional)
                │
    ┌───────────┴───────────┐
    ▼                       ▼
Stage 7a:              Stage 7b:
ParseTemplate          AgenticParseTemplate
(classical)            (agentic)
    │                       │
    └───────────┬───────────┘
                ▼
Stage 8: VerifyTemplate
```

All subsequent stages (VerifyTemplate, EmbeddingCheck, RubricEvaluation, etc.) operate identically regardless of which parsing stage ran. The parsed answer is the same type (`BaseAnswer` subclass) in both cases.

### 4.4. Operational Limits

The investigation agent has two safety limits:

- `agentic_parsing_max_turns` (default: 15): Maximum number of tool call turns before the investigation terminates.
- `agentic_parsing_timeout` (default: 120 seconds): Wall-clock timeout for the investigation step.

Both are configurable on `VerificationConfig`.

## 5. Ground Truth Protection

The [answer template](answer-templates.md)'s ground truth values are stored in `__verification__` metadata on each `VerifiedField`. This metadata must never reach the judge, because exposing ground truth would let the judge trivially fill in correct values without actually evaluating the response.

`BaseAnswer.model_json_schema()` strips all `__verification__` metadata from the JSON schema before returning it. This override happens at the base class level, so it protects every adapter and code path universally, including the agentic parsing stage's investigation and extraction steps. Neither the investigation agent nor the extraction parser ever sees ground truth values.

The stripping is implemented in `karenina/schemas/entities/answer.py` and applies recursively to nested schemas.

## 6. Configuration Split

Agentic evaluation settings are split between `Benchmark`, `Question`, and `VerificationConfig` based on what each setting represents:

| Setting | Lives on | Type | Why |
|---------|----------|------|-----|
| `workspace_root` | `Benchmark` | `Path \| None` | The workspace is benchmark data: where the code and test files live on disk |
| `workspace_path` | `Question` | `str \| None` | Per-question subdirectory within the root (relative path) |
| `workspace_copy` | `VerificationConfig` | `bool` (default `True`) | Operational setting for how verification handles workspaces |
| `workspace_cleanup` | `VerificationConfig` | `bool` (default `True`) | Operational setting for post-run cleanup |
| `agentic_parsing` | `VerificationConfig` | `bool` (default `False`) | Verification strategy: use Stage 7b instead of 7a |
| `agentic_judge_context` | `VerificationConfig` | `Literal` (default `"workspace_only"`) | What context the investigation agent receives |
| `agentic_parsing_max_turns` | `VerificationConfig` | `int` (default `15`) | Safety limit on investigation agent turns |
| `agentic_parsing_timeout` | `VerificationConfig` | `float` (default `120.0`) | Wall-clock timeout for investigation |

The principle: benchmark data (where things live) belongs on `Benchmark` and `Question`. Operational settings (how to run verification) belong on `VerificationConfig`. This separation means the same benchmark checkpoint can be verified with different operational settings without modification.

## 7. Validation Rules

`VerificationConfig` enforces several constraints when `agentic_parsing=True`:

1. **AgentPort support required**: Every parsing model's interface must have an `agent_factory` registered in the [adapter registry](adapters.md). If the interface does not support `AgentPort`, validation raises a `ValueError`.

2. **Incompatible with `rubric_only`**: Agentic parsing operates on the template path (Stages 7 and 8). In `rubric_only` mode, template stages are omitted entirely, so enabling `agentic_parsing` raises a `ValueError`.

3. **Warning for `trace_only`**: When `agentic_judge_context="trace_only"`, the investigation agent has no workspace access. This makes it functionally equivalent to classical Stage 7a parsing with extra overhead. The config logs a warning but does not reject this combination.

## 8. End-to-End Example

```python
from pathlib import Path
from karenina.benchmark import Benchmark
from karenina.schemas.verification.config import VerificationConfig
from karenina.schemas.config.models import ModelConfig

# Load benchmark with workspace
benchmark = Benchmark.load(
    "coding-tasks.jsonld",
    workspace_root=Path("/data/coding-benchmark"),
)

# Configure agentic evaluation
config = VerificationConfig(
    answering_models=[
        ModelConfig(
            id="coder",
            model_name="claude-sonnet-4-20250514",
            interface="claude_agent_sdk",
        ),
    ],
    parsing_models=[
        ModelConfig(
            id="judge",
            model_name="claude-sonnet-4-20250514",
            interface="claude_agent_sdk",
        ),
    ],
    agentic_parsing=True,
    agentic_judge_context="workspace_only",
    workspace_copy=True,
    workspace_cleanup=False,  # Keep working copies for inspection
)

results = benchmark.run_verification(config)
```

In this example:

1. The answering model (`coder`) runs as a deep agent adapter (`claude_agent_sdk`) with access to the workspace. It reads files, writes code, and runs tests.
2. The pipeline copies each question's workspace before the answering agent runs.
3. After the answering agent finishes, `AgenticParseTemplateStage` launches a separate investigation agent with tool access to the (now modified) workspace copy.
4. The investigation agent independently verifies the artifacts. It receives only the question and the workspace path (not the answering agent's trace, because `agentic_judge_context="workspace_only"`).
5. A parser extracts the investigation findings into the answer template schema.
6. `VerifyTemplate` checks the parsed fields against ground truth, just as in the classical workflow.

## 9. Agentic Rubric Evaluation

Agentic evaluation extends beyond template verification to rubric evaluation as well. `AgenticRubricTrait` deploys an agent to investigate workspace artifacts and extract quality scores, using the same investigate-then-extract pattern as Stage 7b.

This is particularly powerful for coding benchmarks where quality signals live in the code files (library choices, error handling, documentation quality) rather than in the response trace.

See [Agentic Rubric Traits](rubrics/agentic-traits.md) for trait definition and usage, and [Stage 11b Internals](../../advanced-pipeline/agentic-rubric-evaluation.md) for the pipeline mechanics.

## 10. Next Steps

- [Verification Pipeline](verification-pipeline.md): The 13-stage engine that hosts both classical and agentic parsing
- [Answer Templates](answer-templates.md): Writing the schemas that both classical and agentic judges fill in
- [Adapters](adapters.md): Understanding port protocols and the adapter registry
- [MCP Overview](mcp-overview.md): Tool-augmented evaluation for the answering model
- [Running Verification](../workflows/running-verification/): End-to-end verification workflow including agentic scenarios
- [Agentic Rubric Traits](rubrics/agentic-traits.md): Agent-investigated quality evaluation
