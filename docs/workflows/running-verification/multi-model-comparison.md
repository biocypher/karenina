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

# Multi-Model Comparison

This scenario compares multiple answering models on the same benchmark. You configure several models, leverage answer caching, use replicates for statistical robustness, and analyze results with grouping and DataFrames.

**What you'll learn:**

- Configure multiple answering models in one run
- Understand answer caching and cost savings
- Group and compare results by model
- Use replicates for variance measurement
- Analyze results with DataFrames

```python tags=["hide-cell"]
# Setup cell: creates mock results for 2 models x 5 questions.
# This cell is hidden in the rendered documentation.
import datetime
import tempfile
from pathlib import Path

from karenina import Benchmark
from karenina.schemas.config import ModelConfig
from karenina.schemas.results import VerificationResultSet
from karenina.schemas.verification import VerificationConfig, VerificationResult
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.schemas.verification.result_components import (
    VerificationResultMetadata,
    VerificationResultRubric,
    VerificationResultTemplate,
)

_benchmark = Benchmark.create(
    name="General Knowledge QA",
    description="Multi-model comparison benchmark",
    version="1.0.0",
)
_questions = [
    ("What is the capital of Japan?", "Tokyo"),
    ("What year did World War II end?", "1945"),
    ("What is the chemical formula for water?", "H2O"),
    ("Who wrote Romeo and Juliet?", "William Shakespeare"),
    ("What is the speed of light in m/s?", "299792458"),
]
for q, a in _questions:
    _benchmark.add_question(question=q, raw_answer=a)

_tmp = Path(tempfile.mkdtemp()) / "benchmark.jsonld"
_benchmark.save(str(_tmp))

_models = {
    "claude-haiku": ModelIdentity(model_name="claude-haiku-4-5", interface="langchain"),
    "claude-sonnet": ModelIdentity(model_name="claude-sonnet-4-5", interface="langchain"),
}
_parsing = ModelIdentity(model_name="claude-haiku-4-5", interface="langchain")
_ts = datetime.datetime.now(tz=datetime.UTC).isoformat()
_qids = _benchmark.get_question_ids()

# Pass/fail profiles per model (True=pass, False=fail)
_profiles = {
    "claude-haiku": [True, True, True, True, False],
    "claude-sonnet": [True, True, True, True, True],
}


def _make(qid, q_text, raw_ans, model_key, verified, replicate=None):
    answering = _models[model_key]
    rid = VerificationResultMetadata.compute_result_id(qid, answering, _parsing, _ts, replicate)
    return VerificationResult(
        metadata=VerificationResultMetadata(
            question_id=qid,
            template_id="tmpl_" + qid[:8],
            completed_without_errors=True,
            question_text=q_text,
            raw_answer=raw_ans,
            answering=answering,
            parsing=_parsing,
            execution_time=1.5,
            timestamp=_ts,
            result_id=rid,
            replicate=replicate,
        ),
        template=VerificationResultTemplate(
            raw_llm_response=f"The answer is {raw_ans}.",
            verify_result=verified,
            template_verification_performed=True,
            parsed_gt_response={"answer": raw_ans},
            parsed_llm_response={"answer": raw_ans if verified else "unknown"},
        ),
    )


# 10 results: 2 models x 5 questions
_mock_results = []
for model_key, profile in _profiles.items():
    for i, (qid, (q, a)) in enumerate(zip(_qids, _questions)):
        _mock_results.append(_make(qid, q, a, model_key, profile[i]))

# Also build replicate results (3 replicates x 5 questions for claude-haiku)
_replicate_results = []
import random
random.seed(42)
for rep in range(1, 4):
    for i, (qid, (q, a)) in enumerate(zip(_qids, _questions)):
        passed = random.random() > 0.2  # ~80% pass rate
        _replicate_results.append(_make(qid, q, a, "claude-haiku", passed, replicate=rep))

_mock_result_set = VerificationResultSet(results=_mock_results)
_replicate_result_set = VerificationResultSet(results=_replicate_results)
_call_count = 0
_orig_run = Benchmark.run_verification


def _patched_run(self, config, **kw):
    global _call_count
    _call_count += 1
    if kw.get("_replicate"):
        return _replicate_result_set
    return _mock_result_set


Benchmark.run_verification = _patched_run
```

---

## Configure Multiple Models

Specify multiple models in the `answering_models` list. Each question is verified once per answering model:

```python
from karenina import Benchmark
from karenina.schemas.verification import VerificationConfig
from karenina.schemas.config import ModelConfig

benchmark = Benchmark.load(str(_tmp))

config = VerificationConfig(
    answering_models=[
        ModelConfig(id="claude-haiku", model_name="claude-haiku-4-5",
                    model_provider="anthropic", interface="langchain"),
        ModelConfig(id="claude-sonnet", model_name="claude-sonnet-4-5",
                    model_provider="anthropic", interface="langchain"),
    ],
    parsing_models=[
        ModelConfig(id="haiku-parser", model_name="claude-haiku-4-5",
                    model_provider="anthropic", interface="langchain",
                    temperature=0.0)
    ],
    evaluation_mode="template_only",
)

print(f"Answering models: {len(config.answering_models)}")
print(f"Total verifications: {len(config.answering_models)} models x {benchmark.question_count} questions = {len(config.answering_models) * benchmark.question_count}")
```

---

## Answer Caching

When multiple parsing models evaluate the same answering model's response, karenina caches the answer generation. Each answering model generates each response only once, regardless of how many parsing models evaluate it.

```
2 answering models x 5 questions = 10 answer generations
1 parsing model x 10 answers = 10 parse calls
Total LLM calls: 20

Without caching (if parsing models varied):
2 answering x 5 questions x 2 parsers = 20 answers (10 cached)
```

This makes multi-model evaluation cost-efficient — the expensive answering step is never duplicated.

---

## Run and Compare

```python
results = benchmark.run_verification(config)
print(f"Total results: {len(results)}")
```

### Group by Model

```python
by_model = results.group_by_model()
for model_key, model_results in by_model.items():
    passed = sum(1 for r in model_results if r.template and r.template.verify_result)
    total = len(model_results)
    print(f"{model_key}: {passed}/{total} passed ({100*passed/total:.0f}%)")
```

### Group by Question

See which questions are hardest across models:

```python
by_question = results.group_by_question()
for qid, q_results in list(by_question.items())[:3]:
    q_text = q_results[0].metadata.question_text[:40]
    passed = sum(1 for r in q_results if r.template and r.template.verify_result)
    print(f"{q_text}... — {passed}/{len(q_results)} models passed")
```

### Filter by Model

```python
haiku_only = results.filter(answering_models=["claude-haiku-4-5"])
print(f"Claude Haiku results: {len(haiku_only)}")
```

---

## DataFrame Analysis

Convert results to a DataFrame for richer analysis:

```python
template_results = results.get_template_results()
df = template_results.to_dataframe()
print(f"DataFrame shape: {df.shape}")
print(df[["question_id", "answering_model", "verify_result"]].head(10))
```

See [DataFrame Analysis](../analyzing-results/dataframe-analysis.md) for advanced pivot tables and visualizations.

---

## Replicates

Use `replicate_count` to run each model-question pair multiple times, measuring variance:

```python
config_with_replicates = VerificationConfig(
    answering_models=[
        ModelConfig(id="claude-sonnet", model_name="claude-sonnet-4-5",
                    model_provider="anthropic", interface="langchain"),
    ],
    parsing_models=[
        ModelConfig(id="haiku-parser", model_name="claude-haiku-4-5",
                    model_provider="anthropic", interface="langchain",
                    temperature=0.0)
    ],
    evaluation_mode="template_only",
    replicate_count=3,
)

print(f"Replicates: {config_with_replicates.replicate_count}")
print(f"Total verifications: {config_with_replicates.replicate_count} x {benchmark.question_count} = {config_with_replicates.replicate_count * benchmark.question_count}")
```

### Analyze Replicate Variance

```python
replicate_results = benchmark.run_verification(config_with_replicates, _replicate=True)

by_question = replicate_results.group_by_question()
for qid, q_results in list(by_question.items())[:3]:
    q_text = q_results[0].metadata.question_text[:40]
    passes = sum(1 for r in q_results if r.template and r.template.verify_result)
    print(f"{q_text}... — {passes}/{len(q_results)} replicates passed")
```

---

## CLI Equivalent

Run multi-model comparison from the command line with sequential calls:

```python
# Compare models using separate runs with the same preset:
# karenina verify benchmark.jsonld --preset base.json --answering-model claude-haiku-4-5 --output results_haiku.json
# karenina verify benchmark.jsonld --preset base.json --answering-model claude-sonnet-4-5 --output results_sonnet.json

# With replicates:
# karenina verify benchmark.jsonld --preset base.json --replicate-count 3

print("CLI: sequential runs with --answering-model for each model")
```

---

## Related Pages

- [Basic Verification](basic-verification.md) — Single-model verification walkthrough
- [Full Evaluation](full-evaluation.md) — Add rubric evaluation to multi-model runs
- [DataFrame Analysis](../analyzing-results/dataframe-analysis.md) — Advanced analysis with pandas
- [VerificationConfig Reference](../../reference/configuration/verification-config.md) — All configuration fields

```python tags=["hide-cell"]
# Cleanup
Benchmark.run_verification = _orig_run
```
