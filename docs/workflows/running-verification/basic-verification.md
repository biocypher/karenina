---
jupyter:
  jupytext:
    formats: docs/workflows/running-verification//md,docs/notebooks/running-verification//ipynb
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

# Basic Verification

This scenario runs verification using template-only evaluation — the simplest path. You load a saved benchmark, configure a `VerificationConfig`, run verification, and inspect results. No rubrics, deep judgment, or MCP tools.

**What you'll learn:**

- Load a benchmark and inspect its contents
- Configure verification with `VerificationConfig` and `from_overrides`
- Run verification on all questions or a subset
- Iterate, filter, and summarize results
- Export results to JSON/CSV and auto-save to a database
- Use the CLI for the same workflow
- Handle errors and run asynchronously

```python tags=["hide-cell"]
# Setup cell: creates a mock benchmark and patches run_verification.
# This cell is hidden in the rendered documentation.
import datetime
import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from karenina import Benchmark
from karenina.schemas.config import ModelConfig
from karenina.schemas.results import VerificationResultSet
from karenina.schemas.verification import VerificationConfig, VerificationResult
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.schemas.verification.result_components import (
    VerificationResultMetadata,
    VerificationResultTemplate,
)
from karenina.schemas.results.failure import Failure, FailureCategory

# Create a small benchmark for documentation
_benchmark = Benchmark.create(
    name="Biomedical Factual QA",
    description="Template-only evaluation of biomedical facts",
    version="1.0.0",
)

_questions = [
    ("What is the approved pharmacological target of venetoclax?", "BCL2"),
    ("How many chromosome pairs do humans have?", "23"),
    ("What is the primary neurotransmitter in the sympathetic nervous system?", "Norepinephrine"),
    ("What organ produces insulin?", "Pancreas"),
    ("What is the half-life of caffeine in healthy adults?", "5 hours"),
]
for q, a in _questions:
    _benchmark.add_question(question=q, raw_answer=a)

_tmp = Path(tempfile.mkdtemp()) / "benchmark.jsonld"
_benchmark.save(str(_tmp))

# Build mock results
_answering = ModelIdentity(model_name="claude-haiku-4-5", interface="langchain")
_parsing = ModelIdentity(model_name="claude-haiku-4-5", interface="langchain")
_ts = datetime.datetime.now(tz=datetime.UTC).isoformat()
_qids = _benchmark.get_question_ids()

_pass_fail = [True, True, True, False, None]  # 3 pass, 1 fail, 1 error


def _make(qid, q_text, raw_ans, verified, error=False):
    rid = VerificationResultMetadata.compute_result_id(qid, _answering, _parsing, _ts)
    template = None if error else VerificationResultTemplate(
        raw_llm_response=f"The answer is {raw_ans}.",
        verify_result=verified,
        template_verification_performed=True,
        parsed_gt_response={"answer": raw_ans},
        parsed_llm_response={"answer": raw_ans if verified else "unknown"},
    )
    return VerificationResult(
        metadata=VerificationResultMetadata(
            question_id=qid,
            template_id="tmpl_" + qid[:8],
            failure=Failure(category=FailureCategory.UNEXPECTED_ERROR, stage="unknown", reason="Template class not found") if error else None,
            caveats=[],
            question_text=q_text,
            raw_answer=raw_ans,
            answering=_answering,
            parsing=_parsing,
            execution_time=1.2,
            timestamp=_ts,
            result_id=rid,
        ),
        template=template,
    )


_mock_results = []
for i, (qid, (q, a)) in enumerate(zip(_qids, _questions)):
    v = _pass_fail[i]
    _mock_results.append(_make(qid, q, a, v, error=(v is None)))

_mock_result_set = VerificationResultSet(results=_mock_results)
_orig_run = Benchmark.run_verification


def _patched_run(self, config, **kwargs):
    qids = kwargs.get("question_ids")
    if qids:
        return VerificationResultSet(
            results=[r for r in _mock_results if r.metadata.question_id in qids]
        )
    return _mock_result_set


Benchmark.run_verification = _patched_run
```

---

## Complete Example

Four lines to go from checkpoint to results:

```python
from karenina import Benchmark
from karenina.schemas.verification import VerificationConfig
from karenina.schemas.config import ModelConfig

benchmark = Benchmark.load(str(_tmp))
config = VerificationConfig.from_overrides(
    answering_id="claude-haiku-4-5", answering_model="claude-haiku-4-5",
    parsing_id="claude-haiku-4-5", parsing_model="claude-haiku-4-5",
)
results = benchmark.run_verification(config)
print(f"Verified {len(results)} questions")
```

---

## Load the Benchmark

Load a saved checkpoint and inspect what's inside:

```python
benchmark = Benchmark.load(str(_tmp))

print(f"Name:      {benchmark.name}")
print(f"Questions: {benchmark.question_count}")
print(f"Complete:  {benchmark.is_complete}")
```

The `is_complete` flag indicates whether all questions have answer templates attached. Verification still runs on questions without templates, but those results will have `template=None`.

---

## Configure Verification

### Full Configuration

`VerificationConfig` controls every aspect of a verification run. For template-only evaluation, the minimum is answering and parsing models:

```python
config = VerificationConfig(
    answering_models=[
        ModelConfig(
            id="claude-haiku-4-5",
            model_name="claude-haiku-4-5",
            model_provider="anthropic",
            interface="langchain",
        )
    ],
    parsing_models=[
        ModelConfig(
            id="claude-haiku-4-5",
            model_name="claude-haiku-4-5",
            model_provider="anthropic",
            interface="langchain",
            temperature=0.0,
        )
    ],
    evaluation_mode="template_only",
)

print(f"Answering models: {len(config.answering_models)}")
print(f"Evaluation mode:  {config.evaluation_mode}")
```

### Quick Configuration with `from_overrides`

`from_overrides` is a shortcut that creates a config with sensible defaults:

```python
config = VerificationConfig.from_overrides(
    answering_id="claude-haiku-4-5",
    answering_model="claude-haiku-4-5",
    answering_provider="anthropic",
    parsing_id="claude-haiku-4-5",
    parsing_model="claude-haiku-4-5",
    parsing_provider="anthropic",
)

print(f"Mode: {config.evaluation_mode}")
```

See [VerificationConfig Reference](../../reference/configuration/verification-config.md) for all fields.

---

## Run Verification

### All Questions

```python
results = benchmark.run_verification(config)
print(f"Total results: {len(results)}")
```

### Subset of Questions

Pass `question_ids` to verify specific questions:

```python
subset_ids = benchmark.get_question_ids()[:2]
subset_results = benchmark.run_verification(config, question_ids=subset_ids)
print(f"Subset results: {len(subset_results)}")
```

---

## Inspect Results

### Iterate Over Results

Each result contains `metadata`, `template` (if template exists), `rubric` (if rubric enabled), and `deep_judgment` (if DJ enabled):

```python
for result in results:
    meta = result.metadata
    if result.template:
        status = "PASS" if result.template.verify_result else "FAIL"
    else:
        status = "NO TEMPLATE"
    print(f"[{status}] {meta.question_text[:60]}")
```

### Summary Statistics

```python
summary = results.get_summary()
print(f"Total:     {summary['num_results']}")
print(f"Completed: {summary['num_completed']}")
print(f"Errors:    {summary['num_results'] - summary['num_completed']}")
```

### Filter Results

```python
completed = results.filter(completed_only=True)
print(f"Completed results: {len(completed)}")

passed = results.filter(completed_only=True, has_template=True)
print(f"Results with templates: {len(passed)}")
```

### Group by Question

```python
by_question = results.group_by_question()
for qid, group in list(by_question.items())[:3]:
    print(f"Question {qid}: {len(group)} results")
```

---

## Save Results

### Export to JSON

```python
# Export is available on the VerificationResultSet
export_data = results.model_dump()
print(f"Exported {len(export_data['results'])} results")
```

### Database Auto-Save

Include a `DBConfig` in your `VerificationConfig` to auto-save results after each run:

```python
from karenina.storage import DBConfig

config_with_db = VerificationConfig(
    answering_models=[
        ModelConfig(id="claude-haiku-4-5", model_name="claude-haiku-4-5",
                    model_provider="anthropic", interface="langchain")
    ],
    parsing_models=[
        ModelConfig(id="claude-haiku-4-5", model_name="claude-haiku-4-5",
                    model_provider="anthropic", interface="langchain",
                    temperature=0.0)
    ],
    db_config=DBConfig(storage_url="sqlite:///results.db"),
)

print(f"DB config: {config_with_db.db_config.storage_url}")
```

Auto-save is non-blocking — if the database write fails, verification still succeeds. Control with the `AUTOSAVE_DATABASE` environment variable. See [Database Persistence](../../reference/configuration/verification-config.md) for `DBConfig` fields.

---

## CLI Equivalent

The `karenina verify` command mirrors the Python API:

```python
# Basic verification with preset
# karenina verify checkpoint.jsonld --preset default.json

# With model overrides
# karenina verify checkpoint.jsonld --preset default.json \
#   --answering-model claude-haiku-4-5

# Verify specific questions
# karenina verify checkpoint.jsonld --preset default.json \
#   --questions 0,1,2

# Progressive save (checkpoint progress during long runs)
# karenina verify checkpoint.jsonld --preset default.json \
#   --output results.json --progressive-save

# Resume an interrupted run
# karenina verify --resume results.json.state
print("CLI commands shown in comments above")
```

See [CLI Reference: verify](../../reference/cli/verify.md) for all options.

---

## Error Handling

Individual question failures do not abort the entire run. Both sequential and parallel executors collect errors as they occur, continue processing the remaining questions, and return partial results. If any questions failed, the executor raises `VerificationBatchError` after all questions have been attempted. This exception carries two attributes: `partial_results` (a dict of question ID to `VerificationResult` for questions that succeeded) and `errors` (a list of `(question_id, exception)` pairs for questions that failed).

Catch `VerificationBatchError` to access partial results instead of losing them:

```python
from karenina.exceptions import VerificationBatchError

try:
    results = benchmark.run_verification(config)
except VerificationBatchError as e:
    print(f"{len(e.errors)} questions failed, {len(e.partial_results)} succeeded")
    for question_id, error in e.errors:
        print(f"  {question_id}: {error}")
    # Partial results are still usable
    results = e.partial_results
```

When all questions succeed, `run_verification` returns normally. You can also inspect per-question errors on individual results:

```python
for result in results:
    if result.metadata.failure is not None:
        print(f"Error in {result.metadata.question_id}: {result.metadata.failure.reason if result.metadata.failure else None}")
```

In parallel mode, `VerificationBatchError` is also raised if the batch exceeds the configured timeout (`ExecutorConfig.timeout_seconds`, default 600 seconds). The `partial_results` attribute contains whichever questions completed before the timeout.

| Exception | When |
|-----------|------|
| `KareninaError` | Base exception for all karenina errors |
| `VerificationBatchError` | One or more questions failed, or parallel batch timed out. Carries `partial_results` and `errors`. |
| `PortError` | LLM/agent/parser port failures |
| `AdapterUnavailableError` | Requested adapter not installed |
| `ParseError` | Judge model returned unparseable output |
| `AgentExecutionError` | MCP agent execution failure |
| `AgentTimeoutError` | MCP agent exceeded time limit |
| `McpError` | MCP server connection or tool failure |

Use `metadata.failure` on each result to inspect run health: `failure` is `None` on success, or a structured `Failure` otherwise.

---

## Async Execution

Enable async for parallel question processing:

```python
config_async = VerificationConfig.from_overrides(
    answering_id="claude-haiku-4-5", answering_model="claude-haiku-4-5",
    parsing_id="claude-haiku-4-5", parsing_model="claude-haiku-4-5",
    async_execution=True,
    async_workers=4,
)

print(f"Async: {config_async.async_enabled}, Workers: {config_async.async_max_workers}")
```

Also configurable via `KARENINA_ASYNC_ENABLED` and `KARENINA_ASYNC_MAX_WORKERS` environment variables.

---

## Related Pages

- [Full Evaluation](full-evaluation.ipynb) — Add rubrics, quality checks, and presets
- [Multi-Model Comparison](multi-model-comparison.ipynb) — Compare models side-by-side
- [VerificationConfig Reference](../../reference/configuration/verification-config.md) — All configuration fields
- [CLI Reference: verify](../../reference/cli/verify.md) — Full CLI options
- [Analyzing Results](../analyzing-results/index.md) — DataFrame analysis and export

```python tags=["hide-cell"]
# Cleanup
Benchmark.run_verification = _orig_run
```
