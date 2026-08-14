---
name: karenina-manual
description: >
  Run karenina evaluation with manually provided responses using the
  ManualAdapter. Use when you want the full verification pipeline but
  provide answers yourself instead of having an LLM generate them.
  Differs from TaskEval: manual uses the full pipeline with ManualAdapter
  intercepting generation; TaskEval skips the pipeline for generation entirely.
---

# Manual Interface Evaluation

The manual interface replays pre-recorded LLM responses through karenina's
full verification pipeline instead of calling a live model. The pipeline
evaluates these traces identically to live responses: parsing, template
verification, and rubric evaluation all run the same way. Only the answer
generation stage changes, reading from a local trace store instead of making
an API call.

**Source of truth**: `karenina/src/karenina/adapters/manual/` (adapter
implementation), `karenina/src/karenina/schemas/config/models.py`
(ModelConfig with `interface="manual"`).

---

## Manual vs TaskEval Decision Guide

Both evaluate pre-recorded text. Choose based on your context:

| Question | Answer | Use |
|----------|--------|-----|
| Do you have a benchmark with questions and checkpoints? | Yes | **Manual interface** |
| Do you want the full pipeline (validate, generate, autofails, optional abstention/sufficiency, parse, verify, rubric, finalize) to run? | Yes | **Manual interface** |
| Are you iterating on templates/rubrics for existing benchmark questions? | Yes | **Manual interface** |
| Do you have free text to evaluate without a benchmark? | Yes | **TaskEval** |
| Are you spot-checking production logs or human-written text? | Yes | **TaskEval** |
| Do you need benchmarks, checkpoints, or question hashes? | No | **TaskEval** |

**In short**: manual interface operates inside the benchmark pipeline
(validate, generate, autofails, optional abstention/sufficiency, parse, verify,
rubric, finalize); TaskEval operates outside it (no benchmark required).

---

## Procedure

### Step 1: Set Up the Benchmark

You need a benchmark with questions. The manual interface looks up traces by
question hash (MD5 of the question text).

```python
from karenina.benchmark import Benchmark

benchmark = Benchmark.create(
    name="Drug Knowledge",
    description="Pharmaceutical knowledge evaluation",
    version="1.0.0",
)

benchmark.add_question(
    question="What is the mechanism of action of aspirin?",
    raw_answer="Irreversible inhibition of cyclooxygenase (COX) enzymes",
)

benchmark.add_question(
    question="What is the primary target of venetoclax?",
    raw_answer="BCL-2",
)
```

### Step 2: Define Templates

Delegate to `karenina-template-authoring` for each question. Templates define
what to extract and how to verify correctness. The manual interface does not
change how templates work.

### Step 3: Register Traces

Provide pre-recorded responses using `ManualTraces`:

```python
from karenina.adapters.manual import ManualTraces

# Create ManualTraces linked to the benchmark
manual_traces = ManualTraces(benchmark)

# Register by question text (recommended)
manual_traces.register_traces({
    "What is the mechanism of action of aspirin?":
        "Aspirin works by irreversibly inhibiting cyclooxygenase (COX) "
        "enzymes, specifically COX-1 and COX-2.",
    "What is the primary target of venetoclax?":
        "Venetoclax selectively targets BCL-2, a protein that prevents "
        "apoptosis in cancer cells.",
}, map_to_id=True)
```

Three trace formats are accepted:

| Format | Input Type | When to Use |
|--------|-----------|-------------|
| String | `str` | Simple text answers |
| Port message list | `list[Message]` | Traces from karenina's native format |
| LangChain message list | `list[AIMessage, ...]` | Traces from LangChain agent runs |

Registration methods:

| Method | Key Type | Notes |
|--------|----------|-------|
| `register_trace(question_identifier, trace)` | MD5 hash | Hash format validated (32 hex chars) |
| `register_trace(question_identifier, trace, map_to_id=True)` | Question text | Text must match exactly |
| `register_traces(traces_dict, map_to_id=True)` | Question text | Batch version |

### Step 4: Configure and Run

Set `interface="manual"` on the answering ModelConfig. A separate parsing
model (live LLM) is always required.

```python
from karenina.schemas.config import ModelConfig
from karenina.schemas.verification import VerificationConfig

config = VerificationConfig(
    answering_models=[
        ModelConfig(
            interface="manual",
            manual_traces=manual_traces,
        )
    ],
    parsing_models=[
        ModelConfig(
            id="judge",
            model_provider="anthropic",
            model_name="claude-haiku-4-5",
        )
    ],
)

results = benchmark.run_verification(config)
```

When `interface="manual"`, ModelConfig automatically sets `id` and
`model_name` to `"manual"` if you leave them unset. You do not need to
specify `model_provider`.

### Step 5: Analyze Results

Delegate to `karenina-results`. Results from manual runs are structured
identically to live runs and are directly comparable.

```python
for r in results:
    print(f"{r.metadata.question_id}: {r.template.verify_result}")
    print(f"  Model: {r.metadata.answering.display_string}")
    print(f"  Parsed: {r.template.parsed_llm_response}")
```

---

## CLI Usage

```
karenina verify checkpoint.jsonld \
    --interface manual \
    --manual-traces traces/my_traces.json \
    --parsing-model claude-haiku-4-5 \
    --parsing-provider anthropic
```

The CLI automatically sets parsing interface to `"langchain"` when
`--interface manual` is specified.

`--output <file>` writes results to a `.json` or `.csv` file, and `--verbose`
shows a progress bar. Trace-file keys are validated as 32-character
hexadecimal MD5 hashes, case-insensitively — though MD5 hexdigests are
lowercase, so lowercase keys are the norm.

### JSON Trace File Format

```json
{
    "385a58a826db8f732c488dafa37433f8": "Aspirin works by irreversibly inhibiting COX.",
    "2a32bfdb18e3e9df6e86e9225bb22615": "Venetoclax targets BCL-2."
}

```
Keys are MD5 hashes of question text. Values are non-empty strings.

---

## Pipeline Behavior with Manual Interface

The pipeline is assembled dynamically from configuration (validate, generate,
autofails, optional abstention/sufficiency, parse, verify, rubric, finalize).
The manual adapter only intercepts the `GenerateAnswer` stage:

```
Normal:  Question -> Answering LLM -> Trace -> Parsing -> Verify -> Rubric
Manual:  Question -> Trace Store    -> Trace -> Parsing -> Verify -> Rubric
```

Guards still apply per question:

- **Abstention check**: Runs on the pre-recorded trace. If the trace looks
  like an abstention, `verify_result` is set to `False`.
- **Sufficiency check**: Runs on the pre-recorded trace.
- **Recursion limit**: Not applicable (manual traces have no agent loop).
- **Trace validation**: Applies only to message-list traces.

---

## Constraints

| Constraint | Reason |
|------------|--------|
| No MCP support | `mcp_urls_dict` on a manual ModelConfig raises `ValueError` |
| No live LLM calls from answering model | `ManualLLMAdapter` and `ManualParserAdapter` raise `ManualInterfaceError` if invoked |
| Traces excluded from serialization | `manual_traces` is `Field(exclude=True)` on ModelConfig; re-register after loading presets |
| Exact text matching | `map_to_id=True` requires exact question text match (case-sensitive, whitespace-sensitive) |
| No tools or structured output | The manual adapter supports neither tools nor MCP by design; traces are precomputed, so there is no tool loop or structured-output schema |
| Parsing model required | A separate live LLM must be configured for template extraction |
---

## Common Workflows

### Template Iteration

Generate answers once, then iterate on templates:

1. Run a live benchmark to capture responses
2. Export traces to JSON
3. Modify template `verify()` logic or add fields
4. Re-run with manual interface (fast: no LLM generation cost)
5. Compare results

### Parsing Model Comparison

Same traces, different judge models:

```python
for judge in ["claude-haiku-4-5", "gpt-4.1-mini"]:
    config = VerificationConfig(
        answering_models=[ModelConfig(interface="manual", manual_traces=traces)],
        parsing_models=[ModelConfig(id=judge, model_provider="...", model_name=judge)],
    )
    results = benchmark.run_verification(config)
    # Compare extraction quality across judges
```

### External Output Evaluation

Evaluate responses from systems outside karenina:

1. Create a benchmark with the same questions
2. Collect responses from the external system
3. Register as manual traces
4. Run pipeline to evaluate with karenina's templates and rubrics

---

## Key Imports

```python
from karenina.benchmark import Benchmark
from karenina.adapters.manual import ManualTraces
from karenina.schemas.config import ModelConfig
from karenina.schemas.verification import VerificationConfig
```

## Assets

- [assets/manual-eval-skeleton.py](assets/manual-eval-skeleton.py): Minimal manual evaluation example

**Full karenina docs**: The `using-karenina` skill contains the complete karenina documentation in its `references/` directory. Consult it for API details not covered here.
