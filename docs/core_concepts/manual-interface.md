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

# Manual Interface

The manual interface is an [adapter](../../../core_concepts/adapters/) that replays pre-recorded LLM responses through the [verification pipeline](../verification-pipeline/) instead of calling a live model. The pipeline evaluates these traces identically to live responses: parsing, template verification, and rubric evaluation all run the same way. The only stage that changes behavior is answer generation, which reads from a local trace store instead of making an API call.

The most important idea is: **the manual interface decouples answer generation from evaluation.** You can generate answers once, then re-evaluate them many times under different templates, rubrics, parsing models, or configurations, without repeating the expensive generation step.

```python tags=["hide-cell"]
# Mock cell: ensures examples execute without live API keys.
# This cell is hidden in rendered documentation.
import datetime
import hashlib
import tempfile
from pathlib import Path

from karenina import Benchmark
from karenina.adapters.manual import ManualTraces
from karenina.schemas.config import ModelConfig
from karenina.schemas.results import VerificationResultSet
from karenina.schemas.verification import VerificationConfig, VerificationResult
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.schemas.verification.result_components import (
    VerificationResultMetadata,
    VerificationResultTemplate,
)

# Create a benchmark with the questions used in visible examples
_bm = Benchmark.create(name="Math QA", description="Arithmetic", version="1.0.0")
for _q, _a in [("What is 2+2?", "4"), ("What is 3+3?", "6")]:
    _bm.add_question(question=_q, raw_answer=_a)

# Patch Benchmark.load to return the pre-built benchmark
_orig_load = Benchmark.load
Benchmark.load = staticmethod(lambda path, **kw: _bm)

# Patch run_verification to return mock results
_qids = _bm.get_question_ids()
_questions_data = [("What is 2+2?", "4"), ("What is 3+3?", "6")]
_answering_id = ModelIdentity(model_name="manual", interface="manual")
_parsing_id = ModelIdentity(model_name="claude-haiku-4-5", interface="langchain")
_ts = datetime.datetime.now(tz=datetime.UTC).isoformat()

def _make_result(qid, q_text, raw_ans):
    rid = VerificationResultMetadata.compute_result_id(qid, _answering_id, _parsing_id, _ts)
    return VerificationResult(
        metadata=VerificationResultMetadata(
            question_id=qid, template_id="tmpl",
            completed_without_errors=True, question_text=q_text,
            raw_answer=raw_ans, answering=_answering_id, parsing=_parsing_id,
            execution_time=0.3, timestamp=_ts, result_id=rid,
        ),
        template=VerificationResultTemplate(
            raw_llm_response=f"The answer is {raw_ans}.",
            verify_result=True, template_verification_performed=True,
            parsed_gt_response={"answer": raw_ans},
            parsed_llm_response={"answer": raw_ans},
        ),
    )

_mock_results = VerificationResultSet(
    results=[_make_result(qid, q, a) for qid, (q, a) in zip(_qids, _questions_data)]
)
_orig_run_verification = Benchmark.run_verification
Benchmark.run_verification = lambda self, config, **kw: _mock_results
```

<div class="admonition info">
<p class="admonition-title">Manual interface vs TaskEval</p>
<p>Both let you evaluate pre-recorded text, but they address different situations. The manual interface operates <strong>inside the Benchmark pipeline</strong>: it replaces the answering model with a trace lookup, so all 13 pipeline stages still run. <a href="../task-eval/">TaskEval</a> operates <strong>outside the Benchmark pipeline</strong>: you feed free text directly into the evaluation engine without needing a benchmark, questions, or checkpoints. See <a href="#6-manual-interface-vs-taskeval">Section 6</a> for detailed guidance on choosing between them.</p>
</div>


## 1. What the Manual Interface Is

The manual interface is one of several adapter backends registered in the [adapter system](../../../core_concepts/adapters/). When `ModelConfig.interface` is set to `"manual"`, the adapter factory returns a `ManualAgentAdapter` that reads from a thread-safe trace store (`ManualTraceManager`) instead of calling an LLM provider. The pipeline stage `GenerateAnswer` uses this adapter through the same `AgentPort` protocol it uses for every other backend; it does not contain manual-specific branching.

### 1.1. The Abstraction Boundary

What the manual interface handles:

- Looking up a pre-recorded trace by question hash (MD5 of the question text)
- Returning that trace in the same `AgentResult` format as live adapters
- Extracting agent metrics (tool calls, iterations, failures) from message-list traces
- Converting port `Message` lists and LangChain message lists to harmonized string traces

What the manual interface does **not** handle:

- Parsing the trace into a template schema (a separate **parsing model** is always required)
- Rubric evaluation (uses the parsing model, not the answering model)
- Prompt construction, stage sequencing, or result storage (handled by other pipeline components)

### 1.2. Three Adapters, One Functional

The manual adapter registers three port implementations, but only one does real work:

| Adapter | Port | Behavior | Purpose |
|---------|------|----------|---------|
| `ManualAgentAdapter` | `AgentPort` | Looks up trace from `ManualTraceManager`, returns `AgentResult` | Answer generation |
| `ManualLLMAdapter` | `LLMPort` | Raises `ManualInterfaceError` if invoked | Safety net: ensures no accidental LLM calls |
| `ManualParserAdapter` | `ParserPort` | Raises `ManualInterfaceError` if invoked | Safety net: parsing uses the separate parsing model |

The LLM and parser adapters exist because the factory always returns an adapter for each port (never `None`). They act as guardrails: if pipeline code accidentally tries to call the answering model's LLM or parser port, it gets an immediate error rather than a silent failure.


## 2. Why It Exists

Answer generation is the slowest and most expensive step in the verification pipeline. When you are iterating on templates, adjusting rubric traits, tuning parsing models, or comparing how different judges handle the same responses, regenerating answers every time wastes time and money.

The manual interface solves this by making the trace store a first-class adapter backend. Because it plugs into the same port/adapter architecture as live backends, every pipeline stage after `GenerateAnswer` runs identically. Results from manual runs are directly comparable to live runs.

| Scenario | How the manual interface helps |
|----------|-------------------------------|
| **Template iteration** | Fix a `verify()` method or add a field, then re-evaluate the same answers instantly |
| **Rubric refinement** | Adjust trait descriptions or add new traits without regenerating responses |
| **Parsing model comparison** | Run the same traces through different judge models to compare extraction quality |
| **Cost control** | Generate answers once with an expensive model, then iterate on evaluation cheaply |
| **External outputs** | Evaluate responses from systems outside Karenina (other frameworks, human-written answers, production logs) |
| **Controlled testing** | Test templates and rubrics with known answers before running full benchmarks |


## 3. How It Works

### 3.1. Pipeline Flow

```
Normal flow:     Question ──► Answering LLM ──► Trace ──► Parsing ──► Verify ──► Rubric
Manual flow:     Question ──► Trace Store    ──► Trace ──► Parsing ──► Verify ──► Rubric
                               (ManualTraceManager)
```

During `GenerateAnswer`, the pipeline computes the MD5 hash of the question text and passes it to the `ManualAgentAdapter` via `AgentConfig.question_hash`. The adapter looks up the corresponding trace in the global `ManualTraceManager` singleton and returns it as an `AgentResult` with zero token usage and `actual_model="manual"`.

Every subsequent stage (parsing, template verification, embedding check, rubric evaluation, finalization) runs without modification. The parsing model (a separate, live LLM) receives the pre-recorded trace and extracts structured data from it exactly as it would from a live response.

### 3.2. Trace Lookup by Question Hash

Traces are indexed by **question hash**: the MD5 hash of the question text, which matches the deterministic ID used throughout Karenina's question system. The pipeline computes this hash automatically during verification; you do not need to compute it manually unless you are building a trace file by hand.

```python
import hashlib

question_hash = hashlib.md5("What is 2+2?".encode("utf-8")).hexdigest()
print(question_hash)
```

If no trace is found for a question hash, the pipeline raises `ManualTraceNotFoundError` with the hash and the count of loaded traces, making mismatches easy to diagnose.

### 3.3. Session Management

The `ManualTraceManager` is a global singleton with session-based cleanup. Traces are automatically cleared after one hour of inactivity (configurable via `session_timeout_seconds`). Individual traces also expire after the session timeout. This prevents memory leaks in long-running server processes. For short-lived scripts, the timeout is effectively irrelevant.


## 4. Trace Formats

`ManualTraces.register_trace()` accepts three input formats. All are stored internally as plain strings.

| Format | Input Type | When to Use |
|--------|-----------|-------------|
| **String** | `str` | Simple text answers with no tool call history |
| **Port message list** | `list[Message]` | Traces captured using Karenina's native message format (`karenina.ports.messages.Message`) |
| **LangChain message list** | `list[AIMessage \| ToolMessage \| ...]` | Traces captured from LangChain agent runs (requires `langchain-core`) |

For message lists, `ManualTraces` automatically:

1. Detects the format (port or LangChain)
2. Converts LangChain messages to port `Message` objects if needed
3. Extracts agent metrics (iterations, tool call counts, failure counts)
4. Harmonizes the message list to a string trace, filtering out system messages and the initial user question
5. Stores the trace string and metrics separately in the `ManualTraceManager`

The extracted agent metrics are preserved in the `AgentResult` and flow through to the verification result, so manual runs of agent traces still report tool usage statistics.

### 4.1. JSON File Format

The CLI and `load_manual_traces_from_file()` expect a JSON file mapping question hashes to trace strings:

```json
{
    "936dbc8755f623c951d96ea2b03e13bc": "The answer is 4.",
    "8f2e2b1e4d5c6a7b8c9d0e1f2a3b4c5d": "The answer is 6."
}
```

Validation rules:

- The file must contain a JSON object (not an array or scalar)
- Keys must be valid 32-character hexadecimal MD5 hashes
- Values must be non-empty strings


## 5. Using the Manual Interface

### 5.1. Python API

Register pre-recorded traces with a benchmark:

```python
from karenina.benchmark import Benchmark
from karenina.adapters.manual import ManualTraces

# Load benchmark
benchmark = Benchmark.load("checkpoint.jsonld")

# Create ManualTraces linked to the benchmark
manual_traces = ManualTraces(benchmark)

# Register by question text (map_to_id=True converts text to hash)
manual_traces.register_trace(
    "What is 2+2?",
    "The answer is 4. I computed this by adding 2 and 2.",
    map_to_id=True,
)

# Or register by MD5 hash directly
manual_traces.register_trace(
    "936dbc8755f623c951d96ea2b03e13bc",
    "The answer is 4.",
)

# Or batch register
manual_traces.register_traces({
    "What is 2+2?": "The answer is 4.",
    "What is 3+3?": "The answer is 6.",
}, map_to_id=True)

print(f"Registered traces for {len(benchmark.get_question_ids())} questions")
```

Then configure and run verification:

```python
from karenina.schemas.config import ModelConfig
from karenina.schemas.verification import VerificationConfig

# Answering model: manual (reads from trace store)
manual_config = ModelConfig(
    interface="manual",
    manual_traces=manual_traces,
)

# Parsing model: a live LLM (required for template extraction)
judge_config = ModelConfig(
    id="claude-haiku",
    model_provider="anthropic",
    model_name="claude-haiku-4-5",
    interface="langchain",
)

config = VerificationConfig(
    answering_models=[manual_config],
    parsing_models=[judge_config],
)

results = benchmark.run_verification(config)
print(f"Verified {len(results)} questions")
```

<div class="admonition tip">
<p class="admonition-title">Automatic defaults for manual ModelConfig</p>
<p>When <code>interface="manual"</code>, <code>ModelConfig</code> automatically sets <code>id</code> and <code>model_name</code> to <code>"manual"</code> if you leave them unset. You do not need to specify <code>model_provider</code>.</p>
</div>

### 5.2. CLI

```
karenina verify checkpoint.jsonld \
    --interface manual \
    --manual-traces traces/my_traces.json \
    --parsing-model claude-haiku-4-5 \
    --parsing-provider anthropic
```

The CLI automatically sets the parsing interface to `"langchain"` when `--interface manual` is specified. The `--manual-traces` flag is required with `--interface manual`.

### 5.3. Registration Behaviors

| Registration method | `map_to_id` | Identifier type | Notes |
|---------------------|-------------|-----------------|-------|
| `register_trace(hash, trace)` | `False` (default) | MD5 hash | Hash format validated (32 hex chars) |
| `register_trace(text, trace, map_to_id=True)` | `True` | Question text | Text must match a question in the benchmark exactly (case-sensitive, whitespace-sensitive) |
| `register_traces(dict, map_to_id=True)` | `True` | Question text | Batch version; calls `register_trace()` for each entry |

When using `map_to_id=True`, `ManualTraces` builds a lazy index of question text to hash on first use, so subsequent lookups are O(1).


## 6. Manual Interface vs TaskEval

Both mechanisms evaluate pre-recorded text, but they serve different workflows.

| Dimension | Manual Interface | [TaskEval](../task-eval/) |
|-----------|-----------------|--------------------------|
| **Operates within** | Benchmark pipeline (all 13 stages run) | Standalone evaluation engine (no benchmark required) |
| **Requires** | Benchmark with questions, checkpoint, `ManualTraces` | Just a template and/or rubric |
| **Input** | Trace per question, keyed by question hash | Free text logged via `task.log()` |
| **Use case** | Re-evaluate benchmark answers under new configs | Evaluate arbitrary text (production logs, human writing, one-off outputs) |
| **Results stored in** | Benchmark results (database, JSON-LD checkpoint) | TaskEval result object |

**Use the manual interface when** you already have a benchmark and want to re-run evaluation on captured traces (template iteration, parsing model comparison, rubric refinement).

**Use TaskEval when** you have text to evaluate but no benchmark context: one-off evaluations, production output spot-checks, or evaluating text from systems that do not produce Karenina checkpoints.


## 7. Constraints

| Constraint | Reason |
|------------|--------|
| **No MCP support** | Configuring `mcp_urls_dict` on a manual `ModelConfig` raises `ValueError`. Pre-recorded traces cannot invoke live tools. |
| **No live LLM calls from the answering model** | `ManualLLMAdapter` and `ManualParserAdapter` raise `ManualInterfaceError` if invoked. Parsing and rubric evaluation use the separate parsing model. |
| **Traces excluded from serialization** | `manual_traces` is `Field(exclude=True)` on `ModelConfig`. When loading a preset with a manual config, traces must be re-registered. |
| **Exact text matching** | When using `map_to_id=True`, question text must match the benchmark exactly (case-sensitive, including whitespace). |
| **No tools or structured output on the answering model** | The manual adapter spec has `supports_mcp=False` and `supports_tools=False`. |


## 8. Next Steps

- [Manual Interface Workflow](../../running-verification/manual-interface-workflow/): Step-by-step walkthrough with executable examples
- [Adapters](../../../core_concepts/adapters/): How the manual interface fits into the port/adapter architecture
- [TaskEval](../task-eval/): Evaluating free text without a benchmark
- [Running Verification](../../../workflows/running-verification/): Overview of all verification methods

```python tags=["hide-cell"]
# Cleanup: restore patched methods
Benchmark.load = _orig_load
Benchmark.run_verification = _orig_run_verification
```
