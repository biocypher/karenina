# Analyzing Results

This section covers how to work with verification results — from understanding the result structure to building DataFrames for analysis, exporting data, and iterating on your benchmark based on findings.

## Workflow Overview

```
Run verification (returns VerificationResultSet)
    │
    ▼
Explore result structure (metadata, template, rubric, deep judgment)
    │
    ▼
Filter and group results (by question, model, replicate)
    │
    ▼
Build DataFrames for analysis (template fields, rubric traits, judgment excerpts)
    │
    ▼
Export results (JSON, CSV, file) ─── and/or ─── Iterate (fix templates, improve rubrics, re-run)
```

Each step has a dedicated page with detailed instructions and examples.

---

## Workflow Steps

### 1. Understand the Result Structure

Every call to `run_verification()` returns a `VerificationResultSet` — a collection of `VerificationResult` objects, one per question × model × replicate combination:

```python
results = benchmark.run_verification(config)

# Iterate over individual results
for result in results:
    meta = result.metadata
    print(f"Q: {meta.question_id} | Model: {meta.answering.model_name}")
```

Each `VerificationResult` contains four sections:

| Section | Contains | Present When |
|---------|----------|--------------|
| **metadata** | Question ID, model info, timing, execution status | Always |
| **template** | Parsed answers, verify result, regex results, embedding similarity | Template evaluation ran |
| **rubric** | Trait scores by type (LLM, regex, callable, metric) | Rubric evaluation ran |
| **deep_judgment** | Excerpts, reasoning, hallucination risk | Deep judgment enabled |

[Understand the full result structure →](verification-result.md)

### 2. Filter and Group Results

`VerificationResultSet` provides built-in methods for slicing results:

```python
# Filter to specific questions or models
subset = results.filter(
    question_ids=["q1", "q2"],
    completed_only=True
)

# Group by question, model, or replicate
by_question = results.group_by_question()
by_model = results.group_by_model()
by_replicate = results.group_by_replicate()
```

You can also get a summary of all results:

```python
summary = results.get_summary()
print(f"Total: {summary['num_results']}, Completed: {summary['num_completed']}")
```

### 3. Build DataFrames for Analysis

Convert results into pandas DataFrames for detailed analysis using three specialized builders:

| Builder | Rows | Best For |
|---------|------|----------|
| **TemplateDataFrameBuilder** | One row per parsed field | Field-by-field comparison, pass/fail analysis |
| **RubricDataFrameBuilder** | One row per rubric trait | Trait score distributions, quality analysis |
| **JudgmentDataFrameBuilder** | One row per excerpt | Deep judgment inspection, hallucination review |

```python
# Template results as a DataFrame
template_results = results.get_template_results()
df = template_results.to_dataframe()

# Rubric results as a DataFrame
rubric_results = results.get_rubrics_results()
df_rubric = rubric_results.to_dataframe()
```

Each builder also supports filtering and aggregation with standard pandas operations:

```python
# Pass rate by model
df.groupby("answering_model_name")["field_match"].mean()

# Trait scores by question
df_rubric.groupby("question_id")["trait_score"].mean()
```

[Analyze results with DataFrames →](dataframe-analysis.md)

### 4. Export Results

Save results for sharing, external analysis, or archival:

```python
# Export as JSON string
json_str = benchmark.export_verification_results(format="json")

# Export to file (format inferred from extension)
benchmark.export_verification_results_to_file("results.json")
benchmark.export_verification_results_to_file("results.csv")

# Export DataFrames directly
df.to_csv("template_analysis.csv", index=False)
```

[Export results →](exporting.md)

### 5. Iterate on Your Benchmark

Use analysis findings to improve your benchmark:

- **Failing templates** — Identify questions where `verify_result` is `False` and refine template logic or field descriptions
- **Low rubric scores** — Find traits with consistently low scores and adjust descriptions or thresholds
- **Re-run verification** — After making changes, re-run to measure improvement

[Iterate and improve →](iterating.md)

---

## Result Access Patterns

The `VerificationResultSet` provides specialized accessors for different analysis needs:

| Accessor | Returns | Use Case |
|----------|---------|----------|
| `get_template_results()` | `TemplateResults` | Template field comparisons, regex matches, token usage |
| `get_rubrics_results()` | `RubricResults` | Rubric trait scores by type |
| `get_judgment_results()` | `JudgmentResults` | Deep judgment excerpts and reasoning |
| `get_rubric_judgments_results()` | `RubricJudgmentResults` | Per-trait deep judgment details |
| `filter(...)` | `VerificationResultSet` | Subset by question, model, or completion status |
| `group_by_question()` | `dict[str, VerificationResultSet]` | Per-question analysis |
| `group_by_model()` | `dict[str, VerificationResultSet]` | Cross-model comparison |
| `group_by_replicate()` | `dict[int, VerificationResultSet]` | Replicate consistency |
| `get_summary()` | `dict` | Aggregate statistics (counts, pass rates, timing) |

---

## What You Get by Evaluation Mode

The data available in results depends on which [evaluation mode](../04-core-concepts/evaluation-modes.md) you used:

| Evaluation Mode | Template Results | Rubric Results | Deep Judgment |
|----------------|-----------------|----------------|---------------|
| **template_only** | Parsed fields, verify result, regex, embedding | — | If enabled |
| **template_and_rubric** | Parsed fields, verify result, regex, embedding | Trait scores for all trait types | If enabled |
| **rubric_only** | — | Trait scores for all trait types | If enabled (rubric only) |

---

## Next Steps

- [Running Verification](../06-running-verification/index.md) — If you haven't run verification yet
- [Creating Benchmarks](../05-creating-benchmarks/index.md) — If you need to build or modify a benchmark
