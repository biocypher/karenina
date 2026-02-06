---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Running Verification with the Python API

The Python API is the most flexible way to run verification. It gives you full
control over configuration, lets you filter which questions to verify, and
provides rich result objects for analysis.

This page walks through the complete workflow: load a benchmark, configure
verification, run it, and work with results.

```python tags=["hide-cell"]
# Mock cell: patches run_verification so examples execute without live API keys.
# This cell is hidden in the rendered documentation.
import datetime
import os
from unittest.mock import patch

from karenina.schemas.results import VerificationResultSet
from karenina.schemas.verification import VerificationConfig, VerificationResult
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.schemas.verification.result_components import (
    VerificationResultMetadata,
    VerificationResultRubric,
    VerificationResultTemplate,
)

# Change to notebooks directory so test_checkpoint.jsonld is found
os.chdir(os.path.dirname(os.path.abspath("__file__")))


def _mock_run_verification(self, config, question_ids=None, **kwargs):
    """Return realistic mock results for documentation examples."""
    qids = question_ids or self.get_question_ids()
    mock_results = []
    answers = {
        "capital of France": ("Paris", True),
        "6 multiplied by 7": ("42", True),
        "atomic number 8": ("Oxygen (O)", True),
        "17 a prime": ("True", True),
        "machine learning": ("Machine learning is a subset of AI", None),
    }
    for qid in qids:
        q = self.get_question(qid)
        question_text = q["question"]
        response, verified = ("Mock response", True)
        for key, (resp, ver) in answers.items():
            if key in question_text.lower():
                response, verified = resp, ver
                break
        answering = ModelIdentity(model_name="gpt-4o", interface="langchain")
        parsing = ModelIdentity(model_name="gpt-4o", interface="langchain")
        ts = datetime.datetime.now(tz=datetime.UTC).isoformat()
        result_id = VerificationResultMetadata.compute_result_id(
            qid, answering, parsing, ts
        )
        template_result = None
        if verified is not None:
            template_result = VerificationResultTemplate(
                raw_llm_response=response,
                verify_result=verified,
                template_verification_performed=True,
            )
        rubric_result = None
        if "capital" in question_text.lower():
            rubric_result = VerificationResultRubric(
                rubric_evaluation_performed=True,
                llm_trait_scores={"Is the response concise?": True},
            )
        result = VerificationResult(
            metadata=VerificationResultMetadata(
                question_id=qid,
                template_id="mock_template" if verified is not None else "no_template",
                completed_without_errors=True,
                question_text=question_text,
                raw_answer=q.get("raw_answer"),
                answering=answering,
                parsing=parsing,
                execution_time=1.2,
                timestamp=ts,
                result_id=result_id,
            ),
            template=template_result,
            rubric=rubric_result,
        )
        mock_results.append(result)
    return VerificationResultSet(results=mock_results)


_patcher_run = patch(
    "karenina.benchmark.benchmark.Benchmark.run_verification",
    _mock_run_verification,
)
_patcher_validate = patch.object(
    VerificationConfig, "_validate_config", lambda self: None
)
_patcher_run.start()
_patcher_validate.start()
```

---

## Complete Example

Here is a minimal end-to-end verification in four lines of code:

```python
from karenina import Benchmark
from karenina.schemas.verification import VerificationConfig
from karenina.schemas.config import ModelConfig

# 1. Load benchmark
benchmark = Benchmark.load("test_checkpoint.jsonld")

# 2. Configure
config = VerificationConfig(
    answering_models=[ModelConfig(id="gpt-4o", model_name="gpt-4o", interface="langchain")],
    parsing_models=[ModelConfig(id="gpt-4o", model_name="gpt-4o", interface="langchain")],
)

# 3. Run
results = benchmark.run_verification(config)

# 4. Inspect
print(f"Verified {len(results)} results across {benchmark.question_count} questions")
```

The rest of this page explains each step in detail.

---

## Step 1: Load a Benchmark

```python
from karenina import Benchmark

benchmark = Benchmark.load("test_checkpoint.jsonld")
print(f"Benchmark: {benchmark.name}")
print(f"Questions: {benchmark.question_count}")
print(f"Complete:  {benchmark.is_complete}")
```

`Benchmark.load()` reads a JSON-LD checkpoint file and returns a `Benchmark`
object. See [Loading a Benchmark](loading-benchmark.md) for details on
inspecting questions, templates, and rubrics before running verification.

---

## Step 2: Configure Verification

Configuration controls which models to use, what evaluation mode to apply,
and which optional features to enable.

```python
from karenina.schemas.verification import VerificationConfig
from karenina.schemas.config import ModelConfig

config = VerificationConfig(
    # Models
    answering_models=[
        ModelConfig(id="gpt-4o", model_name="gpt-4o", interface="langchain"),
    ],
    parsing_models=[
        ModelConfig(id="gpt-4o", model_name="gpt-4o", interface="langchain"),
    ],
    # Evaluation mode
    evaluation_mode="template_and_rubric",
    rubric_enabled=True,
    # Optional features
    abstention_check_enabled=True,
    embedding_check_enabled=False,
)
print(config)
```

See [VerificationConfig](verification-config.md) for a full tutorial on all
configuration options, including deep judgment, async execution, and MCP
settings.

### Quick Configuration with `from_overrides`

For simple setups, `from_overrides()` creates a config with sensible defaults:

```python
config = VerificationConfig.from_overrides(
    answering_id="gpt-4o",
    answering_model="gpt-4o",
    parsing_id="gpt-4o",
    parsing_model="gpt-4o",
)
print(f"Evaluation mode: {config.evaluation_mode}")
```

---

## Step 3: Run Verification

### Basic Run

```python
results = benchmark.run_verification(config)
print(f"Completed: {len(results)} verifications")
```

### Verifying Specific Questions

Pass `question_ids` to verify only a subset of questions:

```python
# Verify only the first two questions
question_ids = benchmark.get_question_ids()[:2]
partial_results = benchmark.run_verification(config, question_ids=question_ids)
print(f"Verified {len(partial_results)} of {benchmark.question_count} questions")
```

### Method Signature

`run_verification()` accepts the following parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `VerificationConfig` | *(required)* | Full verification configuration |
| `question_ids` | `list[str] \| None` | `None` | Specific questions to verify (all if `None`) |
| `run_name` | `str \| None` | `None` | Optional name for this run |
| `async_enabled` | `bool \| None` | `None` | Override async execution (uses config default) |
| `progress_callback` | `Callable[[float, str], None] \| None` | `None` | Progress callback `(percentage, message)` |

---

## Step 4: Inspect Results

`run_verification()` returns a `VerificationResultSet` — a container with
filtering, grouping, and analysis methods.

### Iterating Over Results

Each result is a `VerificationResult` with nested sections:

```python
for result in results:
    meta = result.metadata
    q_text = meta.question_text[:50]

    # Template result (correctness)
    if result.template and result.template.verify_result is not None:
        status = "PASS" if result.template.verify_result else "FAIL"
    else:
        status = "N/A"

    # Rubric result (quality)
    rubric_info = ""
    if result.rubric and result.rubric.rubric_evaluation_performed:
        scores = result.rubric.llm_trait_scores or {}
        rubric_info = f" | rubric traits: {len(scores)}"

    print(f"  [{status}] {q_text}{rubric_info}")
```

### Result Structure

Each `VerificationResult` contains up to four nested sections:

| Section | Field | Contains |
|---------|-------|----------|
| **Metadata** | `result.metadata` | Question ID, models, timing, completion status |
| **Template** | `result.template` | Pass/fail, raw LLM response, embedding similarity |
| **Rubric** | `result.rubric` | LLM trait scores, regex scores, callable scores |
| **Deep Judgment** | `result.deep_judgment` | Extracted excerpts, reasoning, hallucination risk |

Access fields through the nested structure:

    result.metadata.question_id      # Which question
    result.metadata.answering_model  # Which model answered
    result.metadata.execution_time   # How long it took
    result.template.verify_result    # True/False/None
    result.rubric.llm_trait_scores   # {"trait_name": True/False or int}

See [VerificationResult Structure](../07-analyzing-results/verification-result.md)
for complete field documentation.

### Summary Statistics

```python
summary = results.get_summary()
print(f"Total results:  {summary['num_results']}")
print(f"Completed:      {summary['num_completed']}")
print(f"With template:  {summary['num_with_template']}")
print(f"With rubric:    {summary['num_with_rubric']}")
print(f"Unique models:  {summary['num_models']}")
```

### Filtering Results

```python
# Filter to only completed results that have template verification
filtered = results.filter(completed_only=True, has_template=True)
print(f"Filtered: {len(filtered)} results with template verification")
```

### Grouping Results

```python
# Group by question to see per-question outcomes
by_question = results.group_by_question()
for qid, group in by_question.items():
    first = group.results[0]
    q_text = first.metadata.question_text[:40]
    print(f"  {q_text}: {len(group)} result(s)")
```

---

## Error Handling

Karenina uses a structured exception hierarchy rooted at `KareninaError`.
Errors are caught per-question during verification — a single question
failure does not abort the entire run.

### Checking for Errors

Results that encountered errors have `completed_without_errors=False`:

    for result in results:
        if not result.metadata.completed_without_errors:
            print(f"Error on: {result.metadata.question_text[:50]}")

### Exception Hierarchy

When running verification programmatically, you can catch specific error types:

    from karenina.exceptions import KareninaError
    from karenina.ports import PortError, ParseError, AdapterUnavailableError

    try:
        results = benchmark.run_verification(config)
    except KareninaError as e:
        print(f"Verification failed: {e}")

Key exception types:

| Exception | When It Occurs |
|-----------|---------------|
| `KareninaError` | Base for all karenina errors |
| `PortError` | Adapter/port layer failure |
| `AdapterUnavailableError` | Requested backend not available |
| `ParseError` | Judge LLM couldn't parse response into template |
| `AgentExecutionError` | Agent runtime failure |
| `AgentTimeoutError` | Agent hit turn/time limit |
| `McpError` | MCP server communication failure |

Most errors during verification are caught internally and recorded in the
result metadata. Exceptions that escape to your code typically indicate
configuration problems (wrong model name, missing API key) rather than
per-question failures.

---

## Async Execution

For large benchmarks, enable async execution to verify multiple questions
in parallel:

    config = VerificationConfig(
        answering_models=[ModelConfig(id="gpt-4o", model_name="gpt-4o", interface="langchain")],
        parsing_models=[ModelConfig(id="gpt-4o", model_name="gpt-4o", interface="langchain")],
        async_enabled=True,
        async_max_workers=4,
    )
    results = benchmark.run_verification(config)

You can also override the async setting per-call:

    # Force async even if config says sync
    results = benchmark.run_verification(config, async_enabled=True)

Async execution is controlled by two settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `async_enabled` | `False` | Enable parallel question verification |
| `async_max_workers` | `4` | Maximum concurrent verifications |

Both can also be set via environment variables (`KARENINA_ASYNC_ENABLED`,
`KARENINA_ASYNC_MAX_WORKERS`).

---

## Progress Tracking

For long-running verifications, pass a callback to monitor progress:

    def on_progress(percentage: float, message: str):
        print(f"[{percentage:.0f}%] {message}")

    results = benchmark.run_verification(config, progress_callback=on_progress)

The callback receives a percentage (0.0–100.0) and a human-readable status
message at each step.

---

## Next Steps

```python tags=["hide-cell"]
# Clean up the mocks
_ = _patcher_run.stop()
_ = _patcher_validate.stop()
```

- [VerificationConfig](verification-config.md) — Full configuration tutorial
- [Analyzing Results](../07-analyzing-results/index.md) — Filtering, grouping, and DataFrame analysis
- [VerificationResult Structure](../07-analyzing-results/verification-result.md) — Complete field reference
- [CLI Verification](cli.md) — Running verification from the command line
- [Multi-Model Evaluation](multi-model.md) — Comparing multiple LLMs on the same benchmark
