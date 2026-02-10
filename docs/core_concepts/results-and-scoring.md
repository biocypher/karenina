# Results and Scoring

The verification pipeline produces a **`VerificationResult`** for each question evaluated. This page explains the structure of results, what each component contains, and how results vary by evaluation mode.

## What Verification Produces

Each question-model combination produces one `VerificationResult`. If you evaluate 10 questions with 2 answering models and 3 replicates, you get 60 results.

A result contains everything that happened during evaluation:

```
VerificationResult
├── metadata              ← Always present (question ID, model info, timing)
├── template              ← Present when template evaluation ran
├── rubric                ← Present when rubric evaluation ran
├── deep_judgment         ← Present when deep judgment enabled (templates)
└── deep_judgment_rubric  ← Present when deep judgment enabled (rubrics)
```

## Metadata (Always Present)

Every result includes metadata regardless of evaluation mode:

| Field | Type | Description |
|-------|------|-------------|
| `question_id` | `str` | URN identifier for the question |
| `question_text` | `str` | Full question text |
| `raw_answer` | `str` | Ground truth from the checkpoint |
| `answering` | `ModelIdentity` | Answering model name, provider, interface |
| `parsing` | `ModelIdentity` | Parsing model name, provider, interface |
| `execution_time` | `float` | Pipeline execution time in seconds |
| `timestamp` | `str` | ISO timestamp of when the result was produced |
| `completed_without_errors` | `bool` | Whether the pipeline ran successfully |
| `error` | `str \| None` | Error message if something went wrong |
| `result_id` | `str` | Deterministic 16-character hash for this result |
| `template_id` | `str` | MD5 hash of the template code, or `"no_template"` |

## Template Results

Present when template evaluation ran (`template_only` or `template_and_rubric` modes). Contains the core correctness assessment:

| Field | Description |
|-------|-------------|
| `raw_llm_response` | The answering model's full text response |
| `trace_messages` | Structured message trace (for MCP agent runs) |
| `parsed_llm_response` | Fields extracted by the Judge LLM |
| `parsed_gt_response` | Ground truth parsed into the same template fields |
| `verify_result` | Boolean — did `verify()` pass? |
| `verify_granular_result` | Per-field verification (if `verify_granular()` is implemented) |
| `abstention_detected` | Whether the model refused to answer |
| `sufficiency_check_performed` | Whether the sufficiency check was attempted |
| `embedding_check_performed` | Whether the embedding check was attempted |
| `embedding_similarity_score` | Similarity score, 0.0 to 1.0 (if enabled) |
| `regex_overall_success` | Overall regex validation result (if applicable) |
| `usage_metadata` | Token counts and LLM usage statistics |
| `agent_metrics` | MCP agent metrics (tool calls, iterations) |

### The Key Field: `verify_result`

This is the primary correctness signal — a boolean indicating whether the template's `verify()` method returned `True`. When template evaluation did not run (e.g., `rubric_only` mode), this is `None`.

## Rubric Results

Present when rubric evaluation ran (`template_and_rubric` or `rubric_only` modes). Contains per-trait scores organized by trait type:

| Field | Type | Description |
|-------|------|-------------|
| `llm_trait_scores` | `dict[str, int \| bool]` | LLM trait scores (boolean or numeric) |
| `llm_trait_labels` | `dict[str, str]` | Literal trait class labels |
| `regex_trait_scores` | `dict[str, bool]` | Regex trait pass/fail results |
| `callable_trait_scores` | `dict[str, int \| bool]` | Callable trait results |
| `metric_trait_scores` | `dict[str, dict]` | Metric trait confusion matrix counts |
| `metric_trait_confusion_lists` | `dict[str, dict[str, list]]` | Detailed confusion lists (what was found/missed) |

Each dictionary is keyed by trait name. For example:

```python
# LLM trait (boolean)
result.rubric.llm_trait_scores["Safe Response"]  # True

# LLM trait (score)
result.rubric.llm_trait_scores["Clarity"]  # 4

# Regex trait
result.rubric.regex_trait_scores["Has Citations"]  # True

# Metric trait
result.rubric.metric_trait_scores["Reference Coverage"]
# {"tp": 2, "fn": 1, "fp": 0, "precision": 0.67, "recall": 0.67, "f1": 0.67}
```

## Deep Judgment Results (Optional)

When deep judgment is enabled, additional evidence-based results are available:

### Template Deep Judgment

| Field | Description |
|-------|-------------|
| `extracted_excerpts` | Per-attribute verbatim passages from the response |
| `attribute_reasoning` | LLM reasoning for each attribute's extraction |
| `hallucination_risk_assessment` | Risk level per attribute (low/medium/high) |

### Rubric Deep Judgment

| Field | Description |
|-------|-------------|
| `extracted_rubric_excerpts` | Per-trait excerpts with confidence levels |
| `rubric_trait_reasoning` | LLM reasoning for each trait assessment |
| `deep_judgment_rubric_scores` | Scores from deep judgment evaluation |
| `standard_rubric_scores` | Original rubric scores for comparison |
| `traits_without_valid_excerpts` | Traits where evidence was insufficient |

## How Results Vary by Mode

| Component | `template_only` | `template_and_rubric` | `rubric_only` |
|-----------|:---------------:|:---------------------:|:-------------:|
| `metadata` | Yes | Yes | Yes |
| `template` | Yes | Yes | `None` |
| `template.verify_result` | `bool` | `bool` | N/A |
| `rubric` | `None` | Yes | Yes |
| `deep_judgment` | Optional | Optional | `None` |
| `deep_judgment_rubric` | `None` | Optional | Optional |

In `rubric_only` mode, `template` is `None` because no template parsing occurs. The rubric trait scores are the primary output.

## Result Collections

Multiple results are collected into a `VerificationResultSet`, which provides:

- Iteration over individual results
- Filtering by question, model, or replicate
- Aggregation into `TemplateResults` and `RubricResults` for analysis
- DataFrame export for tabular analysis

```python
# After running verification
result_set = benchmark.run_verification(config)

# Access individual results
for result in result_set:
    print(f"{result.metadata.question_id}: {result.template.verify_result}")

# Build DataFrames for analysis
template_results = result_set.get_template_results()
df = template_results.to_dataframe()
```

## Next Steps

- [Verification Pipeline](verification-pipeline.md) — The 13 stages that produce results
- [Evaluation Modes](evaluation-modes.md) — How modes affect which results are available
- [Analyzing Results](../07-analyzing-results/index.md) — Inspecting, aggregating, and exporting results
