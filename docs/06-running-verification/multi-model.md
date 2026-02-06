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

# Multi-Model Evaluation

Karenina supports running the same benchmark across multiple answering and parsing
models in a single call. This lets you compare model performance, measure
inter-judge agreement, and study result variance — all while using **answer
caching** to avoid redundant LLM calls.

```python tags=["hide-cell"]
# Mock cell: patches run_verification so examples execute without live API keys.
# This cell is hidden in the rendered documentation.
import datetime
import os
from unittest.mock import patch

from karenina.schemas.verification import VerificationConfig, VerificationResult
from karenina.schemas.verification.result_components import (
    VerificationResultMetadata,
    VerificationResultTemplate,
)
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.schemas.results import VerificationResultSet

os.chdir(os.path.dirname(os.path.abspath("__file__")))


def _mock_run_verification(self, config, question_ids=None, **kwargs):
    """Return realistic mock results for multi-model documentation."""
    qids = question_ids or self.get_question_ids()
    mock_results = []

    # Model performance profiles — different models have different strengths
    model_profiles = {
        "gpt-4o": {"capital of France": True, "6 multiplied by 7": True,
                    "atomic number 8": True, "prime number": True},
        "claude-sonnet-4-5-20250514": {"capital of France": True, "6 multiplied by 7": True,
                         "atomic number 8": True, "prime number": True},
        "gemini-2.0-flash": {"capital of France": True, "6 multiplied by 7": True,
                             "atomic number 8": False, "prime number": True},
    }

    for qid in qids:
        q = self.get_question(qid)
        q_text = q.get("question", "")
        has_template = self.has_template(qid)

        for ans_model in config.answering_models:
            model_name = ans_model.model_name
            profile = model_profiles.get(model_name, {})

            for parse_model in config.parsing_models:
                # Determine pass/fail from the profile
                passed = any(
                    key in q_text and profile.get(key, False)
                    for key in profile
                )
                if not has_template:
                    passed = None

                answering = ModelIdentity(
                    interface=ans_model.interface,
                    model_name=ans_model.model_name,
                )
                parsing = ModelIdentity(
                    interface=parse_model.interface,
                    model_name=parse_model.model_name,
                )

                template_result = None
                if has_template:
                    template_result = VerificationResultTemplate(
                        raw_llm_response=f"Mock answer from {model_name}",
                        verify_result=passed,
                        template_verification_performed=True,
                    )

                metadata = VerificationResultMetadata(
                    question_id=qid,
                    template_id=self.get_template(qid)[:10] + "..." if has_template else "no_template",
                    completed_without_errors=True,
                    question_text=q_text,
                    answering=answering,
                    parsing=parsing,
                    execution_time=1.2,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    result_id=f"mock-{qid[:8]}-{model_name[:4]}-{parse_model.model_name[:4]}",
                    run_name=kwargs.get("run_name"),
                )

                mock_results.append(
                    VerificationResult(metadata=metadata, template=template_result)
                )

    return VerificationResultSet(results=mock_results)


_patcher1 = patch(
    "karenina.benchmark.benchmark.Benchmark.run_verification",
    _mock_run_verification,
)
_patcher2 = patch(
    "karenina.schemas.verification.config.VerificationConfig._validate_config",
    lambda self: None,
)
_ = _patcher1.start()
_ = _patcher2.start()
```

## How It Works

When you provide multiple answering or parsing models, Karenina creates a
**combinatorial task queue**:

```
Questions × Answering Models × Parsing Models × Replicates = Total Tasks
```

For example, 5 questions × 2 answering models × 1 parsing model × 1 replicate
= 10 verification tasks. Each task runs through the full pipeline independently,
and all results are returned in a single `VerificationResultSet`.

## Configuring Multiple Models

Provide lists for `answering_models` and `parsing_models`:

```python
from karenina.benchmark import Benchmark
from karenina.schemas import ModelConfig, VerificationConfig

benchmark = Benchmark.load("test_checkpoint.jsonld")

config = VerificationConfig(
    answering_models=[
        ModelConfig(
            id="gpt4o",
            model_name="gpt-4o",
            model_provider="openai",
            interface="langchain",
        ),
        ModelConfig(
            id="claude-sonnet",
            model_name="claude-sonnet-4-5-20250514",
            model_provider="anthropic",
            interface="langchain",
        ),
        ModelConfig(
            id="gemini-flash",
            model_name="gemini-2.0-flash",
            model_provider="google_genai",
            interface="langchain",
        ),
    ],
    parsing_models=[
        ModelConfig(
            id="parser",
            model_name="gpt-4o-mini",
            model_provider="openai",
            interface="langchain",
        ),
    ],
)

results = benchmark.run_verification(config)
print(f"Total results: {len(results.results)}")
print(f"Questions: {len(set(r.metadata.question_id for r in results.results))}")
print(f"Answering models: {len(config.answering_models)}")
```

This runs each of the 5 questions against all 3 answering models, parsed by 1
judge — 15 total verification tasks.

## Comparing Models

### Group by Answering Model

```python
by_model = results.group_by_model(by="answering")

for model_name, model_results in by_model.items():
    summary = model_results.get_summary()
    pass_info = summary.get("template_pass_overall", {})
    passed = pass_info.get("passed", 0)
    total = pass_info.get("total", 0)
    pct = pass_info.get("pass_pct", 0)
    print(f"{model_name}: {passed}/{total} passed ({pct:.0f}%)")
```

### Filter to a Specific Model

```python
gpt4_results = results.filter(answering_models=["langchain:gpt-4o"])
print(f"GPT-4o results: {len(gpt4_results.results)}")
```

!!! note
    Model filter values use the `interface:model_name` format (e.g.,
    `"langchain:gpt-4o"`). This is the display string from `ModelIdentity`.

### Group by Question

Compare how each model performed on the same question:

```python
by_question = results.group_by_question()

for qid, q_results in list(by_question.items())[:2]:
    print(f"\nQuestion: {qid[:30]}...")
    for r in q_results.results:
        model = r.metadata.answering.model_name
        passed = r.template.verify_result if r.template else "N/A"
        print(f"  {model}: {passed}")
```

## Result Summary

The `get_summary()` method provides a comprehensive breakdown including token
usage and pass rates by model combination:

```python
summary = results.get_summary()
print(f"Total results: {summary['num_results']}")
print(f"Completed: {summary['num_completed']}")
print(f"Models: {summary['num_models']}")
print(f"Overall pass rate: {summary.get('template_pass_overall', {}).get('pass_pct', 0):.0f}%")
```

The summary also includes `template_pass_by_combo` which breaks down pass rates
per model combination — useful for identifying which answering/parsing pair works
best for your benchmark.

## Answer Caching

When you use multiple **parsing models** with the same answering model, Karenina
automatically caches the answering model's response. This avoids generating the
same answer multiple times.

### How It Works

Consider this configuration:

    1 answering model (GPT-4o) × 3 parsing models × 5 questions = 15 tasks

Without caching, the answering model would be called 15 times. With caching, it
is called only **5 times** — once per question. The 10 remaining tasks reuse the
cached answer and only call the parsing model.

The cache key is: `{question_id}_{answering_model_id}_{replicate}`

When a task finds a cached answer:
1. It skips the answer generation stage entirely
2. It uses the cached response (including token usage and MCP metrics)
3. It runs only the parsing and evaluation stages

### Cost Savings

Answer caching is most impactful when:

| Scenario | Without Cache | With Cache | Savings |
|----------|--------------|------------|---------|
| 1 answering × 3 judges | 15 answer calls | 5 answer calls | 67% fewer answering calls |
| 2 answering × 2 judges | 20 answer calls | 10 answer calls | 50% fewer answering calls |
| 1 answering × 1 judge | 5 answer calls | 5 answer calls | 0% (no sharing) |

Caching is **automatic** — you do not need to configure it. It activates
whenever the same answering model + question + replicate combination appears
in multiple tasks (i.e., when you have multiple parsing models).

### Cache Key and Scope

- Answers are cached **per verification run** (not across runs)
- The cache is **thread-safe** for parallel execution
- If answer generation fails, the cache allows retry by subsequent tasks
- Cache statistics are available via the executor but not exposed in results

## Using `from_overrides` for Model Comparison

A common pattern is to run multiple verification passes from the same base
configuration, overriding just the model:

```python
models_to_compare = [
    ("gpt-4o", "openai"),
    ("claude-sonnet-4-5-20250514", "anthropic"),
    ("gemini-2.0-flash", "google_genai"),
]

all_results = {}
for model_name, provider in models_to_compare:
    config = VerificationConfig.from_overrides(
        answering_model=model_name,
        answering_provider=provider,
        answering_id=f"ans-{model_name}",
        parsing_model="gpt-4o-mini",
        parsing_provider="openai",
        parsing_id="parser",
    )
    run_results = benchmark.run_verification(config, run_name=model_name)
    all_results[model_name] = run_results

for model_name, model_results in all_results.items():
    summary = model_results.get_summary()
    pass_info = summary.get("template_pass_overall", {})
    print(f"{model_name}: {pass_info.get('passed', 0)}/{pass_info.get('total', 0)} passed")
```

This approach gives each model a separate `VerificationResultSet`, which can be
useful when you want to export or analyze results independently. The `run_name`
parameter tags each run for identification.

!!! tip
    The multi-model list approach (multiple `answering_models`) is simpler and
    enables answer caching. The `from_overrides` loop approach gives you separate
    result sets per model. Choose based on your analysis needs.

## Replicates

Run each model combination multiple times to measure variance:

```python
config = VerificationConfig(
    answering_models=[
        ModelConfig(
            id="gpt4o", model_name="gpt-4o",
            model_provider="openai", interface="langchain",
        ),
    ],
    parsing_models=[
        ModelConfig(
            id="parser", model_name="gpt-4o-mini",
            model_provider="openai", interface="langchain",
        ),
    ],
    replicate_count=3,
)

results = benchmark.run_verification(config)
print(f"Total results: {len(results.results)} (5 questions × 3 replicates)")
```

Group by replicate to analyze variance:

```python
by_replicate = results.group_by_replicate()
for rep_num, rep_results in sorted(by_replicate.items()):
    summary = rep_results.get_summary()
    pass_info = summary.get("template_pass_overall", {})
    print(f"Replicate {rep_num}: {pass_info.get('passed', 0)}/{pass_info.get('total', 0)} passed")
```

## Async Execution

Multi-model runs benefit from parallel execution:

```python
config = VerificationConfig(
    answering_models=[
        ModelConfig(
            id="gpt4o", model_name="gpt-4o",
            model_provider="openai", interface="langchain",
        ),
        ModelConfig(
            id="claude-sonnet", model_name="claude-sonnet-4-5-20250514",
            model_provider="anthropic", interface="langchain",
        ),
    ],
    parsing_models=[
        ModelConfig(
            id="parser", model_name="gpt-4o-mini",
            model_provider="openai", interface="langchain",
        ),
    ],
    async_enabled=True,
    async_max_workers=4,
)
```

Parallel execution works seamlessly with answer caching — the cache uses
thread-safe locking. When a task encounters an in-progress cache entry, it is
requeued rather than blocked, keeping all workers busy.

## Next Steps

- [Verification Result structure](../07-analyzing-results/verification-result.md) — full result hierarchy
- [DataFrame analysis](../07-analyzing-results/dataframe-analysis.md) — convert results to pandas DataFrames
- [Python API verification](python-api.md) — single-model workflow
- [VerificationConfig reference](../10-configuration-reference/verification-config.md) — all configuration fields
- [CLI verification](cli.md) — run multi-model from the command line

```python tags=["hide-cell"]
# Cleanup mocks
_ = _patcher1.stop()
_ = _patcher2.stop()
```
