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

# Progressive Save and Resume

This tutorial shows how to checkpoint verification progress so you can resume interrupted runs. For long verification runs (many questions, expensive models, multiple replicates), progressive save writes results incrementally. If the run stops for any reason, you resume from the last checkpoint instead of re-evaluating everything.

**What you'll learn:**

- Enable progressive save with `--progressive-save` (CLI)
- Resume an interrupted run with `--resume`
- Check run status with `karenina verify-status`
- Use `ProgressiveSaveManager` directly in Python
- Understand the `.tmp` and `.state` file pair
- Verify configuration compatibility before resuming
- Read intermediate results from `.tmp` files during a run

```python tags=["hide-cell"]
# Setup cell: creates mock objects for progressive save demonstration.
# This cell is hidden in the rendered documentation.
import datetime
import json
import tempfile
from pathlib import Path

from karenina.schemas.verification.config import VerificationConfig
from karenina.schemas.config.models import ModelConfig
from karenina.schemas.verification import VerificationResult
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.schemas.verification.result_components import (
    VerificationResultMetadata,
    VerificationResultTemplate,
)
from karenina.utils.progressive_save import (
    ProgressiveSaveManager,
    inspect_state_file,
    ProgressiveJobStatus,
    TaskIdentifier,
)

_tmp_dir = Path(tempfile.mkdtemp())
_output_path = _tmp_dir / "results.json"
_benchmark_path = str(_tmp_dir / "benchmark.jsonld")

_answering = ModelIdentity(model_name="claude-haiku-4-5", interface="langchain")
_parsing = ModelIdentity(model_name="claude-haiku-4-5", interface="langchain")
_ts = datetime.datetime.now(tz=datetime.UTC).isoformat()

_questions = [
    ("q1", "What is the primary target of venetoclax?", "BCL2"),
    ("q2", "How many chromosome pairs do humans have?", "23"),
    ("q3", "What organ produces insulin?", "Pancreas"),
]


def _make_result(qid, q_text, raw_ans, verified):
    rid = VerificationResultMetadata.compute_result_id(qid, _answering, _parsing, _ts)
    return VerificationResult(
        metadata=VerificationResultMetadata(
            question_id=qid,
            template_id="tmpl_" + qid,
            completed_without_errors=True,
            question_text=q_text,
            raw_answer=raw_ans,
            answering=_answering,
            parsing=_parsing,
            execution_time=1.5,
            timestamp=_ts,
            result_id=rid,
        ),
        template=VerificationResultTemplate(
            raw_llm_response=f"The answer is {raw_ans}.",
            verify_result=verified,
            template_verification_performed=True,
            parsed_gt_response={"answer": raw_ans},
            parsed_llm_response={"answer": raw_ans if verified else "unknown"},
        ),
    )
```

---

## CLI Workflow

Three commands cover the full progressive save lifecycle:

```python
# 1. Start a verification run with progressive save enabled:
# karenina verify checkpoint.jsonld --preset default.json \
#   --output results.json --progressive-save

# 2. Check progress while a run is active (or after interruption):
# karenina verify-status results.json.state

# 3. Resume from where the run stopped:
# karenina verify --resume results.json.state

print("CLI commands shown in comments above")
```

The `--progressive-save` flag tells the runner to write results incrementally to a `.tmp` file and track progress in a `.state` file. If the process is interrupted, `--resume` picks up from the last completed task. The `verify-status` command reads the `.state` file and reports how many tasks are pending.

---

## How It Works

Progressive save maintains two sidecar files alongside your output path:

```
verify --progressive-save
    │
    ├── results.json.tmp   (accumulated results)
    ├── results.json.state (progress tracking)
    │
    ▼ (on completion)
    results.json           (final output)
```

The `.tmp` file stores results in standard export format, so you can read intermediate results at any time. The `.state` file tracks the task manifest (every task that needs to run), the set of completed task IDs, and a config hash for compatibility checks. Both files use atomic writes to prevent corruption if the process terminates mid-write.

On successful completion, `finalize()` removes the `.tmp` and `.state` files and writes the final output.

---

## Python API: Initialize

Create a `ProgressiveSaveManager`, define the task manifest, and call `initialize()`:

```python
config = VerificationConfig.from_overrides(
    answering_id="claude-haiku-4-5",
    answering_model="claude-haiku-4-5",
    parsing_id="claude-haiku-4-5",
    parsing_model="claude-haiku-4-5",
)

manager = ProgressiveSaveManager(
    output_path=_output_path,
    config=config,
    benchmark_path=_benchmark_path,
)

# Build task manifest from TaskIdentifier objects
task_ids = []
for qid, _, _ in _questions:
    tid = TaskIdentifier(
        question_id=qid,
        answering_canonical_key="claude-haiku-4-5",
        parsing_canonical_key="claude-haiku-4-5",
        replicate=0,
    )
    task_ids.append(tid.to_key())

manager.initialize(task_ids)
print(f"Initialized with {manager.total_tasks} tasks")
print(f"Completed so far: {manager.completed_count}")
print(f"State file: {manager.state_path.name}")
print(f"Tmp file:   {manager.tmp_path.name}")
```

The task manifest is a list of string keys. Each key uniquely identifies one verification task by question ID, answering model, parsing model, and replicate number.

---

## Python API: Add Results

As verification completes each task, call `add_result()` to persist it:

```python
# Simulate completing the first two questions
result_1 = _make_result("q1", "What is the primary target of venetoclax?", "BCL2", True)
result_2 = _make_result("q2", "How many chromosome pairs do humans have?", "23", True)

manager.add_result(result_1)
manager.add_result(result_2)

print(f"Completed: {manager.completed_count}/{manager.total_tasks}")
print(f"Pending:   {len(manager.get_pending_task_ids())} tasks remain")
```

Each `add_result()` call atomically updates both the `.tmp` and `.state` files. If the process crashes after this call, the completed results are safely on disk.

---

## Python API: Inspect State

Use `inspect_state_file()` to check progress without loading the full manager:

```python
status = inspect_state_file(manager.state_path)

print(f"Total tasks:    {status.total_tasks}")
print(f"Completed:      {status.completed_count}")
print(f"Pending:        {status.pending_count}")
print(f"Progress:       {status.progress_percent:.1f}%")
print(f"Tmp file exists:{status.tmp_file_exists}")
print(f"Tmp file size:  {status.tmp_file_size} bytes")
```

`ProgressiveJobStatus` is a lightweight dataclass. It reads only the `.state` JSON file, so it works even while another process is writing results.

---

## Python API: Resume

To resume an interrupted run, load the manager from the `.state` file:

```python
resumed = ProgressiveSaveManager.load_for_resume(manager.state_path)

print(f"Resumed with {resumed.completed_count}/{resumed.total_tasks} already done")

pending = resumed.get_pending_task_ids()
print(f"Pending task IDs: {len(pending)}")

# Complete the remaining task
result_3 = _make_result("q3", "What organ produces insulin?", "Pancreas", True)
resumed.add_result(result_3)

print(f"After adding q3: {resumed.completed_count}/{resumed.total_tasks}")
```

`load_for_resume()` reconstructs the full manager state: it reloads the config, task manifest, completed set, and all previously saved results from the `.tmp` file.

---

## Python API: Finalize

Once all tasks are complete, call `finalize()` to clean up the sidecar files:

```python
state_path = resumed.state_path
tmp_path = resumed.tmp_path

print(f"Before finalize: .state exists = {state_path.exists()}")
print(f"Before finalize: .tmp exists   = {tmp_path.exists()}")

resumed.finalize()

print(f"After finalize:  .state exists = {state_path.exists()}")
print(f"After finalize:  .tmp exists   = {tmp_path.exists()}")
```

After `finalize()`, only the final output file remains. The runner writes the complete results to the output path before calling `finalize()`, so the `.tmp` and `.state` files are no longer needed.

---

## Configuration Compatibility

When resuming, the manager checks that the config and benchmark path match the original run. This prevents accidental result mixing:

```python
# Create a fresh manager to test compatibility
fresh_manager = ProgressiveSaveManager(
    output_path=_output_path,
    config=config,
    benchmark_path=_benchmark_path,
)
fresh_manager.initialize(task_ids)

# Same config and benchmark: compatible
compatible, reason = fresh_manager.is_compatible(config, _benchmark_path)
print(f"Same config:      compatible={compatible}")

# Different config: incompatible
different_config = VerificationConfig.from_overrides(
    answering_id="claude-sonnet-4-20250514",
    answering_model="claude-sonnet-4-20250514",
    parsing_id="claude-haiku-4-5",
    parsing_model="claude-haiku-4-5",
)
compatible, reason = fresh_manager.is_compatible(different_config, _benchmark_path)
print(f"Different config: compatible={compatible}")
print(f"Reason: {reason}")

# Different benchmark path: incompatible
compatible, reason = fresh_manager.is_compatible(config, "/other/benchmark.jsonld")
print(f"Different path:   compatible={compatible}")
print(f"Reason: {reason}")
```

The config hash is computed from the full `VerificationConfig` JSON (excluding manual traces). Any change to models, evaluation mode, or pipeline settings will be detected.

---

## Cleanup

```python tags=["hide-cell"]
# Cleanup temporary files
import shutil
fresh_manager.finalize()
shutil.rmtree(_tmp_dir, ignore_errors=True)
```

---

## Next Steps

- [Basic Verification](../../notebooks/running-verification/basic-verification.ipynb): Single-model template-only evaluation
- [Full Evaluation](../../notebooks/running-verification/full-evaluation.ipynb): Template and rubric evaluation with quality checks
- [CLI Reference: verify](../../reference/cli/verify.md): All `karenina verify` options
- [CLI Reference: verify-status](../../reference/cli/verify-status.md): Inspect progressive save state
