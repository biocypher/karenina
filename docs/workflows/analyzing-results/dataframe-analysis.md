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

# DataFrame Analysis

Karenina provides a DataFrame-first approach for analyzing verification results.
By converting results to pandas DataFrames, you can use the full power of pandas
for filtering, grouping, aggregation, and visualization.

```python tags=["hide-cell"]
# Setup cell: creates mock VerificationResult objects for documentation examples.
# This cell is hidden in the rendered documentation.
import datetime

from karenina.schemas.results import VerificationResultSet
from karenina.schemas.verification import VerificationResult
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.schemas.verification.result_components import (
    VerificationResultMetadata,
    VerificationResultRubric,
    VerificationResultTemplate,
)

# Build mock results representing a verification run with template + rubric evaluation
_answering_gpt4o = ModelIdentity(model_name="gpt-4o", interface="langchain")
_answering_claude = ModelIdentity(model_name="claude-sonnet-4-20250514", interface="claude_agent_sdk")
_parsing = ModelIdentity(model_name="gpt-4o-mini", interface="langchain")
_ts = datetime.datetime.now(tz=datetime.UTC).isoformat()


def _make_result(
    qid, question_text, answering, verified, response,
    rubric_scores=None, regex_scores=None, callable_scores=None,
    parsed_gt=None, parsed_llm=None, replicate=None,
):
    rid = VerificationResultMetadata.compute_result_id(
        qid, answering, _parsing, _ts, replicate
    )
    template = VerificationResultTemplate(
        raw_llm_response=response,
        verify_result=verified,
        template_verification_performed=True,
        parsed_gt_response=parsed_gt or {"answer": response},
        parsed_llm_response=parsed_llm or {"answer": response},
    )
    rubric = None
    if rubric_scores or regex_scores or callable_scores:
        rubric = VerificationResultRubric(
            rubric_evaluation_performed=True,
            llm_trait_scores=rubric_scores,
            regex_trait_scores=regex_scores,
            callable_trait_scores=callable_scores,
        )
    return VerificationResult(
        metadata=VerificationResultMetadata(
            question_id=qid,
            template_id="tmpl_" + qid[:8],
            completed_without_errors=True,
            question_text=question_text,
            answering=answering,
            parsing=_parsing,
            execution_time=1.5,
            timestamp=_ts,
            result_id=rid,
            replicate=replicate,
        ),
        template=template,
        rubric=rubric,
    )


# Create results for two models across 3 questions
_mock_results = [
    # GPT-4o results
    _make_result("q1", "What is the capital of France?", _answering_gpt4o, True, "Paris",
                 rubric_scores={"clarity": 4, "conciseness": True},
                 regex_scores={"no_hedging": True},
                 parsed_gt={"capital": "Paris"}, parsed_llm={"capital": "Paris"}),
    _make_result("q2", "What is 6 multiplied by 7?", _answering_gpt4o, True, "42",
                 rubric_scores={"clarity": 5, "conciseness": True},
                 parsed_gt={"result": "42"}, parsed_llm={"result": "42"}),
    _make_result("q3", "What element has atomic number 8?", _answering_gpt4o, False, "Nitrogen",
                 rubric_scores={"clarity": 3, "conciseness": False},
                 parsed_gt={"element": "Oxygen"}, parsed_llm={"element": "Nitrogen"}),
    # Claude results
    _make_result("q1", "What is the capital of France?", _answering_claude, True, "Paris",
                 rubric_scores={"clarity": 5, "conciseness": True},
                 regex_scores={"no_hedging": True},
                 parsed_gt={"capital": "Paris"}, parsed_llm={"capital": "Paris"}),
    _make_result("q2", "What is 6 multiplied by 7?", _answering_claude, True, "42",
                 rubric_scores={"clarity": 5, "conciseness": True},
                 parsed_gt={"result": "42"}, parsed_llm={"result": "42"}),
    _make_result("q3", "What element has atomic number 8?", _answering_claude, True, "Oxygen",
                 rubric_scores={"clarity": 4, "conciseness": True},
                 parsed_gt={"element": "Oxygen"}, parsed_llm={"element": "Oxygen"}),
]

results = VerificationResultSet(results=_mock_results)
```

## Overview

After running verification, you receive a `VerificationResultSet` containing all
results. The result set provides three specialized accessors that convert results
to pandas DataFrames:

| Accessor | Returns | Rows Represent |
|----------|---------|----------------|
| `get_template_results()` | `TemplateResults` | One row per **parsed field** comparison |
| `get_rubrics_results()` | `RubricResults` | One row per **rubric trait** evaluated |
| `get_judgment_results()` | `JudgmentResults` | One row per **(attribute x excerpt)** pair |

Each accessor returns a wrapper object with a `.to_dataframe()` method plus
filtering, grouping, and aggregation helpers.

## Getting Started

The basic workflow is: extract a result type, convert to DataFrame, analyze
with pandas.

```python
# Extract template results and convert to DataFrame
template_results = results.get_template_results()
df = template_results.to_dataframe()

print(f"DataFrame shape: {df.shape}")
print(f"Columns: {list(df.columns[:8])}...")
```

## Template DataFrames

`TemplateResults` provides three DataFrame methods:

| Method | Exploded By | Use Case |
|--------|-------------|----------|
| `to_dataframe()` | Parsed fields | Field-level pass/fail analysis |
| `to_regex_dataframe()` | Regex patterns | Format compliance analysis |
| `to_usage_dataframe()` | Token usage stages | Cost analysis |

### Field-Level Analysis

The main template DataFrame creates one row per parsed field, enabling
field-level comparison between ground truth and LLM-extracted values.

```python
template_results = results.get_template_results()
df = template_results.to_dataframe()

# Key columns for field-level analysis
print("Field comparison columns:")
print(df[["question_id", "answering_model", "field_name", "gt_value", "llm_value", "field_match"]].to_string(index=False))
```

### Pass Rate by Model

A common analysis pattern: calculate template verification pass rates grouped
by answering model.

```python
# Use the built-in aggregation helper
pass_rates = template_results.aggregate_pass_rate(by="answering_model")
print("Pass rate by model:")
for model, rate in pass_rates.items():
    print(f"  {model}: {rate:.0%}")
```

### Pass Rate by Question

```python
pass_rates_by_q = template_results.aggregate_pass_rate(by="question_id")
print("Pass rate by question:")
for qid, rate in pass_rates_by_q.items():
    print(f"  {qid}: {rate:.0%}")
```

### Filtering Results

`TemplateResults` supports filtering before DataFrame conversion:

```python
# Filter to only failed results
failed = template_results.filter(failed_only=True)
df_failed = failed.to_dataframe()
print(f"Failed results: {len(failed)} (fields in DataFrame: {len(df_failed)})")

# Filter by model (use the full display string: "interface:model_name")
gpt_results = template_results.filter(answering_models=["langchain:gpt-4o"])
print(f"GPT-4o results: {len(gpt_results)}")
```

### Summary Statistics

```python
summary = template_results.get_template_summary()
print(f"Total results: {summary['num_results']}")
print(f"Passed: {summary['num_passed']}, Failed: {summary['num_failed']}")
print(f"Pass rate: {summary['pass_rate']:.0%}")
print(f"Unique questions: {summary['num_questions']}")
```

## Rubric DataFrames

`RubricResults` converts rubric evaluation scores to DataFrames, with one row
per trait evaluated. It supports filtering by trait type.

### Trait Type Filtering

The `to_dataframe()` method accepts a `trait_type` parameter:

| Value | Includes |
|-------|----------|
| `"all"` | All trait types (default) |
| `"llm"` | All LLM traits (score + binary + literal) |
| `"llm_score"` | LLM score traits only (1-5 scale) |
| `"llm_binary"` | LLM binary traits only (True/False) |
| `"llm_literal"` | LLM literal traits only (categorical) |
| `"regex"` | Regex traits |
| `"callable"` | Callable traits |
| `"metric"` | Metric traits (exploded by metric name) |

```python
rubric_results = results.get_rubrics_results()
df_all = rubric_results.to_dataframe()

print("All rubric traits:")
print(df_all[["question_id", "answering_model", "trait_name", "trait_score", "trait_type"]].to_string(index=False))
```

### Filtering by Trait Type

```python
# Get only LLM score traits (numeric 1-5 scale)
df_scores = rubric_results.to_dataframe(trait_type="llm_score")
print(f"\nLLM score traits: {len(df_scores)} rows")
if len(df_scores) > 0:
    print(df_scores[["question_id", "answering_model", "trait_name", "trait_score"]].to_string(index=False))
```

### Aggregating Trait Scores

```python
# Average LLM trait scores by model
avg_by_model = rubric_results.aggregate_llm_traits(
    strategy="mean", by="answering_model"
)
print("Average LLM trait scores by model:")
for model, traits in avg_by_model.items():
    print(f"  {model}:")
    for trait, score in traits.items():
        print(f"    {trait}: {score:.1f}")
```

### Trait Summary

```python
trait_summary = rubric_results.get_trait_summary()
print(f"Results with rubric data: {trait_summary['num_results']}")
print(f"LLM traits: {trait_summary['llm_traits']}")
print(f"Regex traits: {trait_summary['regex_traits']}")
print(f"Callable traits: {trait_summary['callable_traits']}")
```

## Deep Judgment DataFrames

`JudgmentResults` handles deep judgment data, creating one row per
(attribute x excerpt) pair. This is the most granular DataFrame — use it
when deep judgment is enabled in your verification configuration.

```python
# Access judgment results (empty if deep judgment was not enabled)
judgment_results = results.get_judgment_results()
print(f"Results with deep judgment: {len(judgment_results.get_results_with_judgment())}")
```

When deep judgment is enabled, the DataFrame provides columns for excerpt text,
confidence scores, similarity scores, hallucination risk, and reasoning traces.

### Including Deep Judgment in Rubric DataFrames

You can also include deep judgment columns in rubric DataFrames:

```python
# Include trait reasoning and excerpts in rubric DataFrame
rubric_with_dj = results.get_rubrics_results(include_deep_judgment=True)
df = rubric_with_dj.to_dataframe()
# When deep judgment is enabled, additional columns appear:
# trait_reasoning, trait_excerpts, trait_hallucination_risk
print(f"Rubric DataFrame columns: {len(df.columns)}")
```

## Common Analysis Patterns

### Model Comparison

Compare template pass rates and rubric scores across models using pandas:

```python
# Template pass rates by model
template_df = results.get_template_results().to_dataframe()
model_pass = (
    template_df.drop_duplicates(subset=["result_index"])
    .groupby("answering_model")["verify_result"]
    .mean()
)
print("Template pass rate by model:")
print(model_pass.to_string())
```

### Question Difficulty

Identify which questions are hardest by looking at pass rates across all models:

```python
question_pass = (
    template_df.drop_duplicates(subset=["result_index"])
    .groupby("question_id")["verify_result"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "pass_rate", "count": "num_runs"})
    .sort_values("pass_rate")
)
print("\nQuestion difficulty (sorted by pass rate):")
print(question_pass.to_string())
```

### Exporting to CSV

```python
# Export template results to CSV
import os
import tempfile

with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
    template_df.to_csv(f.name, index=False)
    print(f"Exported {len(template_df)} rows to CSV")
    os.unlink(f.name)
```

## Result Access Methods Summary

All three result types share a consistent interface:

| Method | TemplateResults | RubricResults | JudgmentResults |
|--------|:---:|:---:|:---:|
| `to_dataframe()` | field-level | trait-level | attribute x excerpt |
| `filter()` | by model, question, pass/fail | by model, question | by model, question, search |
| `group_by_question()` | dict of TemplateResults | dict of RubricResults | dict of JudgmentResults |
| `group_by_model()` | dict of TemplateResults | dict of RubricResults | dict of JudgmentResults |
| `get_*_summary()` | template stats | trait inventory | judgment stats |

The `VerificationResultSet` itself provides higher-level operations:

- `filter()` — filter by question IDs, models, completion status, etc.
- `group_by_question()` / `group_by_model()` / `group_by_replicate()` — group results
- `get_summary()` — comprehensive statistics including pass rates, token usage, and tool usage

## Next Steps

- [VerificationResult Structure](verification-result.md) — understand the complete result hierarchy
- [Exporting Results](exporting.md) — save results to JSON, CSV, or files
- [Iterating on Benchmarks](iterating.md) — use analysis to improve templates and rubrics
- [Running Verification](../06-running-verification/python-api.md) — how to generate results
