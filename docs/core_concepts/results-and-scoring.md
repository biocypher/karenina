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

# Results and Scoring

Every question that passes through the [verification pipeline](../verification-pipeline/) produces a **`VerificationResult`**: a nested Pydantic model that captures everything that happened during evaluation, from the raw response to the final pass/fail verdict. This page explains the result data model, how scoring works, how to access and aggregate results, and how to export them for analysis.

```python tags=["hide-cell"]
# Mock cell: ensures examples execute without live API keys.
# This cell is hidden in rendered documentation.
# NOTE: pandas must be imported before the mock setup to avoid numpy
# double-loading issues that break internal pandas operations.
import pandas as pd

from unittest.mock import MagicMock, patch

mock_modules = {}
for mod in [
    "sqlalchemy", "sqlalchemy.orm", "sqlalchemy.ext",
    "sqlalchemy.ext.declarative", "sqlalchemy.engine",
    "sqlalchemy.sql", "sqlalchemy.event",
    "karenina.storage", "karenina.storage.base",
    "karenina.storage.engine", "karenina.storage.db_config",
    "karenina.storage.models", "karenina.storage.generated_models",
    "karenina.storage.auto_mapper", "karenina.storage.operations",
]:
    mock_modules[mod] = MagicMock()

with patch.dict("sys.modules", mock_modules):
    from karenina.schemas.verification import (
        ModelIdentity,
        VerificationResult,
        VerificationResultDeepJudgment,
        VerificationResultMetadata,
        VerificationResultRubric,
        VerificationResultTemplate,
    )
    from karenina.schemas.results import VerificationResultSet
    from karenina.benchmark.core.results import ResultsManager

# ---------------------------------------------------------------------------
# Build realistic mock results for the examples below.
# Two questions, two answering models, one replicate each = 4 results.
# ---------------------------------------------------------------------------

answering_a = ModelIdentity(interface="langchain", model_name="claude-sonnet-4-6")
answering_b = ModelIdentity(interface="langchain", model_name="gpt-4.1-mini-2025-04-14")
parsing = ModelIdentity(interface="langchain", model_name="claude-haiku-4-5-20251001")

def _make_result(
    qid, qtxt, raw_ans, answering_model, verify, llm_scores, regex_scores,
    metric_scores=None, metric_confusion=None, callable_scores=None,
    llm_labels=None, exec_time=1.2, ts="2025-06-15T10:00:00Z",
    abstention=False, embedding_score=None,
):
    """Helper to build a realistic VerificationResult for examples."""
    result_id = VerificationResultMetadata.compute_result_id(
        question_id=qid, answering=answering_model, parsing=parsing,
        timestamp=ts, replicate=1,
    )
    meta = VerificationResultMetadata(
        question_id=qid, template_id="abc123", question_text=qtxt,
        raw_answer=raw_ans, answering=answering_model, parsing=parsing,
        execution_time=exec_time, timestamp=ts, result_id=result_id,
        failure=None, caveats=[], replicate=1,
    )
    tmpl = VerificationResultTemplate(
        raw_llm_response="The target is BCL2, also known as B-cell lymphoma 2.",
        parsed_llm_response={"target": "BCL2"},
        parsed_gt_response={"target": "BCL2"},
        template_verification_performed=True,
        verify_result=verify,
        abstention_check_performed=abstention,
        abstention_detected=abstention if abstention else None,
        embedding_check_performed=embedding_score is not None,
        embedding_similarity_score=embedding_score,
    )
    rubric = VerificationResultRubric(
        rubric_evaluation_performed=True,
        rubric_evaluation_strategy="batch",
        llm_trait_scores=llm_scores,
        llm_trait_labels=llm_labels,
        regex_trait_scores=regex_scores,
        callable_trait_scores=callable_scores,
        metric_trait_scores=metric_scores,
        metric_trait_confusion_lists=metric_confusion,
    )
    return VerificationResult(metadata=meta, template=tmpl, rubric=rubric)

results_list = [
    _make_result(
        "q1", "What is the putative target of venetoclax?", "BCL2",
        answering_a, True,
        {"safety": True, "clarity": 4}, {"has_citations": True},
        metric_scores={"drug_coverage": {"tp": 3, "fn": 1, "fp": 0, "precision": 1.0, "recall": 0.75, "f1": 0.857}},
        metric_confusion={"drug_coverage": {"tp": ["aspirin", "ibuprofen", "acetaminophen"], "fn": ["naproxen"], "fp": [], "tn": []}},
        callable_scores={"under_150w": True},
        llm_labels={"response_type": "Factual"},
        embedding_score=0.92,
    ),
    _make_result(
        "q1", "What is the putative target of venetoclax?", "BCL2",
        answering_b, False,
        {"safety": True, "clarity": 3}, {"has_citations": False},
        ts="2025-06-15T10:01:00Z", exec_time=2.1,
    ),
    _make_result(
        "q2", "What chromosome is TP53 located on?", "Chromosome 17",
        answering_a, True,
        {"safety": True, "clarity": 5}, {"has_citations": True},
        ts="2025-06-15T10:02:00Z",
    ),
    _make_result(
        "q2", "What chromosome is TP53 located on?", "Chromosome 17",
        answering_b, True,
        {"safety": True, "clarity": 4}, {"has_citations": True},
        ts="2025-06-15T10:03:00Z", exec_time=1.8,
    ),
]

result_set = VerificationResultSet(results=results_list)
# Pick the first result for single-result examples
result = results_list[0]
```

## 1. What Results Capture

The most important idea is: **a result is a complete evidence record, not just a score**. It preserves every intermediate artifact (the raw response, the parsed fields, each optional check's outcome, every rubric trait score) so that downstream analysis can always trace a verdict back to its inputs. Nothing is discarded.

A single `VerificationResult` corresponds to one question evaluated by one answering model and parsed by one judge model, in one replicate. If you evaluate 10 questions with 2 answering models and 3 replicates, you get 60 results.

### 1.1. Result Structure at a Glance

`VerificationResult` uses **nested composition**: five optional sub-objects, each grouping a coherent slice of the evidence. This is the only access path; flat property accessors do not exist.

```
VerificationResult
├── metadata              ← Always present: identification, timing, model info
├── template              ← Present when template evaluation ran
├── rubric                ← Present when rubric evaluation ran
├── deep_judgment         ← Present when deep judgment ran (templates)
├── deep_judgment_rubric  ← Present when deep judgment ran (rubrics)
│
│  (Root-level fields for MCP agent trace filtering)
├── evaluation_input      ← The text passed to evaluation stages
├── used_full_trace       ← Whether the full agent trace was used
└── trace_extraction_error ← Error if final AI message extraction failed
```

Access fields through their sub-objects:

```python
# Correct: nested access
print(result.metadata.question_id)
print(result.template.verify_result)
print(result.rubric.llm_trait_scores)
```

```python
# Wrong: flat access (removed, will raise AttributeError)
try:
    result.question_id
except AttributeError as e:
    print(f"AttributeError: {e}")
```

## 2. Metadata: Identity and Execution Context

Every result carries a `VerificationResultMetadata` sub-object regardless of evaluation mode. It identifies *what* was evaluated, *by which models*, and *when*.

| Field | Type | Description |
|-------|------|-------------|
| `question_id` | `str` | MD5 hash of the question text (32-char hex) |
| `question_text` | `str` | Full question text |
| `raw_answer` | `str \| None` | Human-readable ground truth from the checkpoint |
| `template_id` | `str` | MD5 hash of the template code, or `"no_template"` |
| `answering` | `ModelIdentity` | Answering model (interface, model_name, tools) |
| `parsing` | `ModelIdentity` | Parsing/judge model (interface, model_name) |
| `answering_system_prompt` | `str \| None` | System prompt used for the answering model |
| `parsing_system_prompt` | `str \| None` | System prompt used for the parsing model |
| `execution_time` | `float` | Pipeline execution time in seconds |
| `timestamp` | `str` | ISO timestamp of when the result was produced |
| `result_id` | `str` | Deterministic 16-character SHA256 hash (see below) |
| `run_name` | `str \| None` | Organizing label for verification runs |
| `replicate` | `int \| None` | Replicate number (1, 2, 3, ...) for repeated runs |
| `keywords` | `list[str] \| None` | Keywords associated with the question |
| `failure` | `Failure \| None` | Structured non-pass verdict; `None` on success, otherwise carries `category`, derived `group`, originating `stage`, and `reason` |
| `caveats` | `list[Caveat]` | Informational flags attached to the run (e.g. `partial_content`, `embedding_override`, `retries_used`) |
| `few_shot_enabled` | `bool` | Whether few-shot prompting was active (default `False`) |
| `few_shot_example_count` | `int` | Number of few-shot examples used (default `0`) |
| `evaluation_mode` | `str \| None` | Evaluation mode used (e.g., `"template_only"`, `"template_and_rubric"`) |

```python
meta = result.metadata
print(f"Question:  {meta.question_id}")
print(f"Model:     {meta.answering.display_string}")
print(f"Judge:     {meta.parsing.display_string}")
print(f"Time:      {meta.execution_time}s")
print(f"Result ID: {meta.result_id}")
print(f"Replicate: {meta.replicate}")
print(f"Success:   {(meta.failure is None)}")
```

### 2.1. ModelIdentity

Models are identified by a composite `ModelIdentity` object, not a plain string. This distinguishes the same model used with different interfaces or MCP tool sets:

| Field | Description |
|-------|-------------|
| `interface` | The adapter interface (e.g., `"langchain"`, `"claude_sdk"`) |
| `model_name` | The model name (e.g., `"claude-sonnet-4-6"`) |
| `tools` | Sorted list of MCP server names (answering models only; always `[]` for parsing) |

```python
identity = result.metadata.answering
print(f"Interface:      {identity.interface}")
print(f"Model name:     {identity.model_name}")
print(f"Tools:          {identity.tools}")
print(f"Display string: {identity.display_string}")
print(f"Canonical key:  {identity.canonical_key}")
```

### 2.2. Deterministic Result IDs

Each result gets a `result_id`: a 16-character SHA256 hash computed from `(question_id, answering, parsing, timestamp, replicate)`. The same inputs always produce the same ID, enabling deduplication across runs. The ID is computed by `VerificationResultMetadata.compute_result_id()`.

```python
# Same inputs always produce the same ID
id1 = VerificationResultMetadata.compute_result_id(
    question_id="q1", answering=answering_a, parsing=parsing,
    timestamp="2025-06-15T10:00:00Z", replicate=1,
)
id2 = VerificationResultMetadata.compute_result_id(
    question_id="q1", answering=answering_a, parsing=parsing,
    timestamp="2025-06-15T10:00:00Z", replicate=1,
)
print(f"ID 1:  {id1}")
print(f"ID 2:  {id2}")
print(f"Match: {id1 == id2}")
print(f"Length: {len(id1)} characters")
```

## 3. Template Results: The Correctness Record

The `template` sub-object (`VerificationResultTemplate`) is present whenever template evaluation ran (`template_only` or `template_and_rubric` [evaluation modes](../evaluation-modes/)). It records the full chain from raw response to pass/fail verdict.

### 3.1. The Primary Correctness Signal: `verify_result`

`verify_result` is a `bool | None` that captures whether the template's `verify()` method returned `True`. This is the core correctness output. When template evaluation did not run (e.g., `rubric_only` mode), this field is `None`.

Several pipeline stages can override this value before finalization:

| Stage | Override Behavior |
|-------|-------------------|
| [Abstention check](../verification-pipeline/) | Sets `verify_result` to `False` if the model refused to answer |
| [Sufficiency check](../verification-pipeline/) | Sets `verify_result` to `False` if the response lacks sufficient information |
| [Embedding check](../verification-pipeline/) | Can override `verify_result` based on semantic similarity threshold |

The corresponding `*_override_applied` boolean fields record whether an override occurred, so you can always distinguish "failed on its own merits" from "overridden by a guard stage."

```python
tmpl = result.template
print(f"verify_result:                {tmpl.verify_result}")
print(f"Parsed LLM response:         {tmpl.parsed_llm_response}")
print(f"Parsed ground truth:         {tmpl.parsed_gt_response}")
print(f"Abstention check performed:  {tmpl.abstention_check_performed}")
print(f"Embedding check performed:   {tmpl.embedding_check_performed}")
print(f"Embedding similarity score:  {tmpl.embedding_similarity_score}")
```

### 3.2. Response and Parsing Artifacts

| Field | Type | Description |
|-------|------|-------------|
| `raw_llm_response` | `str` | The answering model's full text response |
| `trace_messages` | `list[dict]` | Structured message trace (for MCP agent runs) |
| `parsed_llm_response` | `dict \| None` | Fields extracted by the Judge LLM |
| `parsed_gt_response` | `dict \| None` | Ground truth parsed into the same template fields |
| `verify_granular_result` | `Any \| None` | Per-field verification detail (if `verify_granular()` is implemented) |
| `field_verification_error` | `str \| None` | Error message if `verify()` raised an exception (non-fatal) |
| `field_results` | `dict[str, bool] \| None` | Per-field primitive verification results (from `_compute_field_results()`) |
| `composition_strategy` | `str \| None` | Composition strategy used: `"all_of"`, `"any_of"`, or `"at_least_n(N)"` |

### 3.3. Optional Check Results

Each optional check records three pieces of state: whether it was attempted, what it found, and whether it overrode the verdict.

| Check | Key Fields |
|-------|------------|
| **Abstention** | `abstention_check_performed`, `abstention_detected`, `abstention_override_applied`, `abstention_reasoning` |
| **Sufficiency** | `sufficiency_check_performed`, `sufficiency_detected`, `sufficiency_override_applied`, `sufficiency_reasoning` |
| **Embedding** | `embedding_check_performed`, `embedding_similarity_score` (0.0 to 1.0), `embedding_override_applied`, `embedding_model_used` |
| **Regex** | `regex_validations_performed`, `regex_validation_results` (per-pattern dict), `regex_overall_success`, `regex_extraction_results` |

### 3.4. Execution Metadata

| Field | Type | Description |
|-------|------|-------------|
| `recursion_limit_reached` | `bool` | Whether an MCP agent hit its recursion limit |
| `answering_mcp_servers` | `list[str] \| None` | MCP servers attached to the answering model |
| `usage_metadata` | `dict \| None` | Token usage breakdown by stage (`answer_generation`, `parsing`, `rubric_evaluation`, `abstention_check`, `total`) |
| `agent_metrics` | `dict \| None` | MCP agent metrics: `iterations`, `tool_calls`, `tools_used`, `suspect_failed_tool_calls`, `suspect_failed_tools` |

## 4. Rubric Results: The Quality Record

The `rubric` sub-object (`VerificationResultRubric`) is present whenever rubric evaluation ran (`template_and_rubric` or `rubric_only` modes). Trait scores are split by type into separate dictionaries, all keyed by trait name.

| Field | Type | Description |
|-------|------|-------------|
| `llm_trait_scores` | `dict[str, int \| bool] \| None` | LLM-evaluated traits (boolean or 1-5 scale) |
| `llm_trait_labels` | `dict[str, str] \| None` | Class labels for literal-kind LLM traits (index-to-name mapping) |
| `regex_trait_scores` | `dict[str, bool] \| None` | Regex trait pass/fail results |
| `callable_trait_scores` | `dict[str, bool \| int] \| None` | Callable trait results |
| `metric_trait_scores` | `dict[str, dict[str, float]] \| None` | Metric trait metrics (precision, recall, F1, etc.) |
| `metric_trait_confusion_lists` | `dict[str, dict[str, list[str]]] \| None` | Per-metric confusion lists (tp, tn, fp, fn containing excerpts) |
| `rubric_evaluation_strategy` | `str \| None` | `"batch"` or `"sequential"` |

### 4.1. Accessing Trait Scores

```python
# Individual trait types
print("LLM trait (boolean):", result.rubric.llm_trait_scores["safety"])
print("LLM trait (score):  ", result.rubric.llm_trait_scores["clarity"])
print("Regex trait:        ", result.rubric.regex_trait_scores["has_citations"])
print("Callable trait:     ", result.rubric.callable_trait_scores["under_150w"])
```

```python
# Literal-kind LLM traits: score is the class index, label is the class name
print("Literal label:", result.rubric.llm_trait_labels["response_type"])
```

```python
# Metric traits: nested dict of float metrics
print("Metric scores:", result.rubric.metric_trait_scores["drug_coverage"])
```

```python
# Confusion lists for metric traits: which items were found/missed
print("Confusion lists:", result.rubric.metric_trait_confusion_lists["drug_coverage"])
```

```python
# Flat access across all types
all_scores = result.rubric.get_all_trait_scores()
print("All trait scores:", all_scores)
```

```python
# Look up a trait by name (returns value and type)
print("Trait lookup:", result.rubric.get_trait_by_name("safety"))
print("Trait lookup:", result.rubric.get_trait_by_name("has_citations"))
```

## 5. Deep Judgment Results (Optional)

When [deep judgment](../rubrics/llm-traits/) is enabled, additional evidence-based results are captured. Deep judgment adds excerpt extraction, per-attribute reasoning, and optional hallucination risk assessment on top of standard evaluation.

### 5.1. Template Deep Judgment

The `deep_judgment` sub-object (`VerificationResultDeepJudgment`) records per-attribute evidence:

| Field | Type | Description |
|-------|------|-------------|
| `extracted_excerpts` | `dict[str, list[dict]]` | Per-attribute verbatim passages with confidence (`low`/`medium`/`high`), similarity score, and optional search results |
| `attribute_reasoning` | `dict[str, str]` | LLM reasoning for each attribute (present even when no excerpts were found) |
| `hallucination_risk_assessment` | `dict[str, str]` | Risk level per attribute (`none`/`low`/`medium`/`high`); only populated when search is enabled |
| `deep_judgment_stages_completed` | `list[str]` | Which stages ran: `"excerpts"`, `"reasoning"`, `"parameters"` |
| `attributes_without_excerpts` | `list[str]` | Attributes with no corroborating excerpts |
| `deep_judgment_model_calls` | `int` | Number of LLM invocations |

### 5.2. Rubric Deep Judgment

The `deep_judgment_rubric` sub-object (`VerificationResultDeepJudgmentRubric`) records per-trait evidence for rubric traits with deep judgment enabled:

| Field | Type | Description |
|-------|------|-------------|
| `extracted_rubric_excerpts` | `dict[str, list[dict]]` | Per-trait excerpts (only for traits with `deep_judgment_excerpt_enabled=True`) |
| `rubric_trait_reasoning` | `dict[str, str]` | Per-trait reasoning (all deep-judgment-enabled traits) |
| `deep_judgment_rubric_scores` | `dict[str, int \| bool]` | Scores from deep-judgment evaluation |
| `standard_rubric_scores` | `dict[str, int \| bool]` | Scores for non-deep-judgment traits (for comparison) |
| `traits_without_valid_excerpts` | `list[str]` | Traits that exhausted retries without valid excerpts |
| `trait_metadata` | `dict[str, dict]` | Per-trait tracking (stages completed, model calls, retry counts) |

## 6. How Results Vary by Evaluation Mode

The [evaluation mode](../evaluation-modes/) determines which sub-objects are populated:

| Sub-object | `template_only` | `template_and_rubric` | `rubric_only` |
|------------|:---------------:|:---------------------:|:-------------:|
| `metadata` | Always | Always | Always |
| `template` | Present | Present | `None` |
| `template.verify_result` | `bool` | `bool` | N/A |
| `rubric` | `None` | Present | Present |
| `deep_judgment` | Optional | Optional | `None` |
| `deep_judgment_rubric` | `None` | Optional | Optional |

In `rubric_only` mode, no template parsing occurs. The rubric trait scores evaluated against the raw response are the primary output. In `template_only` mode (the default), the `rubric` sub-object is `None`.

## 7. Working with Result Collections

`Benchmark.run_verification()` returns a **`VerificationResultSet`**: the top-level container that holds all individual results and provides specialized views, filtering, grouping, and DataFrame conversion.

### 7.1. Specialized Views

The result set provides four accessor methods, each returning a purpose-built wrapper with its own analysis API:

| Accessor Method | Returns | Purpose |
|----------------|---------|---------|
| `get_template_results()` | `TemplateResults` | Pass/fail rates, embedding scores, regex results, abstention detection, parsed responses |
| `get_rubrics_results()` | `RubricResults` | Trait scores by type, aggregation, confusion matrices |
| `get_judgment_results()` | `JudgmentResults` | Extracted excerpts, reasoning traces, hallucination risk |
| `get_rubric_judgments_results()` | `RubricJudgmentResults` | Excerpt-level explosion (one row per trait per excerpt) |

```python
# Template analysis
template_results = result_set.get_template_results()
print(f"TemplateResults with {len(template_results)} results")
print(f"Summary: {template_results.get_template_summary()}")
```

```python
# Rubric analysis
rubric_results = result_set.get_rubrics_results()
print(f"RubricResults with {len(rubric_results)} results")
print(f"Summary: {rubric_results.get_trait_summary()}")
```

### 7.2. Filtering and Grouping

Both `VerificationResultSet` and the specialized views support filtering and grouping. Filtering returns a new instance of the same type with a subset of results.

```python
# Filter at the result set level
filtered = result_set.filter(
    question_ids=["q1"],
    completed_only=True,
)
print(f"Filtered to {len(filtered)} results (question q1 only)")

# Group by different dimensions
by_question = result_set.group_by_question()
for qid, qresults in by_question.items():
    print(f"  Question {qid}: {len(qresults)} results")
```

```python
# Group by model
by_model = result_set.group_by_model()
for model_key, model_results in by_model.items():
    print(f"  Model {model_key}: {len(model_results)} results")
```

```python
# Specialized views also support filtering
passed = template_results.filter(passed_only=True)
failed = template_results.filter(failed_only=True)
print(f"Passed: {len(passed)}, Failed: {len(failed)}")
```

### 7.3. Iteration

All containers support standard Python iteration:

```python
for r in result_set:
    print(f"  {r.metadata.question_id}: verify={r.template.verify_result}, "
          f"model={r.metadata.answering.model_name}")

print(f"\nTotal results: {len(result_set)}")
print(f"First result question: {result_set[0].metadata.question_text}")
```

## 8. DataFrame Export

Every specialized view converts to pandas DataFrames for tabular analysis. The DataFrame structures are designed around a specific "explosion" axis: each row represents the finest-grained unit for that view type.

### 8.1. Template DataFrames

`TemplateResults` provides three DataFrame exports:

| Method | Row Granularity | Key Columns |
|--------|----------------|-------------|
| `to_dataframe()` | One row per **parsed field** per result | `field_name`, `gt_value`, `llm_value`, `field_match`, `verify_result` |
| `to_regex_dataframe()` | One row per **regex pattern** per result | `pattern_name`, `pattern_regex`, `matched`, `extracted_value` |
| `to_usage_dataframe()` | One row per **usage stage** per result | `usage_stage`, `input_tokens`, `output_tokens`, `total_tokens`, `model_used` |

```python
template_results = result_set.get_template_results()

# Field-level comparison: ground truth vs LLM extraction
df = template_results.to_dataframe()
print(f"Template DataFrame: {len(df)} rows, {len(df.columns)} columns")
print(f"Columns: {list(df.columns)}")
print()
print(df[["question_id", "field_name", "gt_value", "llm_value", "field_match", "verify_result"]].to_string(index=False))
```

### 8.2. Rubric DataFrames

`RubricResults.to_dataframe()` produces one row per trait (or per metric for metric traits). Filter by trait type using the `trait_type` parameter:

| `trait_type` | Includes |
|-------------|----------|
| `"all"` (default) | All trait types combined |
| `"llm"` | All LLM traits (score, binary, and literal) |
| `"llm_score"` | LLM traits with 1-5 scale |
| `"llm_binary"` | LLM traits with boolean scores |
| `"llm_literal"` | LLM traits with categorical classification |
| `"regex"` | Regex traits (boolean) |
| `"callable"` | Callable traits (boolean or integer) |
| `"metric"` | Metric traits (exploded by metric name) |

Key columns: `trait_name`, `trait_score`, `trait_label` (for literal kinds), `trait_type`, `metric_name` (for metrics), `confusion_tp`/`fp`/`fn`/`tn` (for metrics).

```python
rubric_results = result_set.get_rubrics_results()

# All traits
df = rubric_results.to_dataframe()
print(f"Rubric DataFrame: {len(df)} rows")
print()
print(df[["question_id", "trait_name", "trait_score", "trait_type"]].to_string(index=False))
```

```python
# Just LLM traits
df_llm = rubric_results.to_dataframe(trait_type="llm")
print(f"LLM traits only: {len(df_llm)} rows")
print(df_llm[["question_id", "answering_model", "trait_name", "trait_score"]].to_string(index=False))
```

## 9. Aggregation

Both `TemplateResults` and `RubricResults` provide built-in aggregation methods. Aggregation groups results by a column (e.g., `question_id`, `answering_model`, `replicate`) and applies a strategy.

### 9.1. Built-in Aggregation Strategies

| Strategy | Behavior | Best For |
|----------|----------|----------|
| `"mean"` | Arithmetic mean | Numeric scores, similarity scores |
| `"median"` | Median value | Numeric scores with outliers |
| `"mode"` | Most common value | Categorical values |
| `"majority_vote"` | `True` if >50% are `True` (configurable threshold) | Boolean traits, pass/fail |
| `"first"` | First non-null value | Metadata fields |
| `"count"` | Count occurrences of each value | Distribution analysis |

### 9.2. Template Aggregation

```python
template_results = result_set.get_template_results()

# Pass rate by question
pass_rates = template_results.aggregate_pass_rate(by="question_id")
print("Pass rates by question:", pass_rates)
```

### 9.3. Rubric Aggregation

```python
rubric_results = result_set.get_rubrics_results()

# Average LLM trait scores by question
avg = rubric_results.aggregate_llm_traits(strategy="mean", by="question_id")
print("Average LLM trait scores by question:")
for qid, scores in avg.items():
    print(f"  {qid}: {scores}")
```

```python
# Majority vote on regex traits by model
regex_agg = rubric_results.aggregate_regex_traits(strategy="majority_vote", by="answering_model")
print("Regex trait majority vote by model:")
for model, scores in regex_agg.items():
    print(f"  {model}: {scores}")
```

### 9.4. Custom Aggregators

Register custom aggregation strategies by implementing the `ResultAggregator` protocol:

```python
class WeightedMeanAggregator:
    """Custom aggregator that computes a weighted mean."""
    def aggregate(self, series, **kwargs):
        # Simple mean as fallback (weights would come from kwargs)
        return series.mean()

rubric_results.register_aggregator("weighted_mean", WeightedMeanAggregator())
weighted = rubric_results.aggregate_llm_traits(strategy="weighted_mean", by="question_id")
print("Available aggregators:", rubric_results.list_aggregators())
print("Weighted mean result:", weighted)
```

## 10. In-Memory Storage and Export

`ResultsManager` stores verification results in memory during a session. Results are organized by run name and can be exported to JSON or CSV. The export format is auto-detected from the file extension, or can be specified explicitly.

```python
# ResultsManager API (shown here without a live Benchmark for reference):
#
# from pathlib import Path
#
# # Results are stored automatically after run_verification
# results = benchmark.results.get_verification_results(run_name="my_run")
#
# # Export to file (format auto-detected from extension)
# benchmark.results.export_results_to_file(Path("results.json"))
# benchmark.results.export_results_to_file(Path("results.csv"))
#
# # Get summary statistics for a run
# summary = benchmark.results.get_verification_summary(run_name="my_run")
# # {"total_results": 60, "successful_count": 58, "success_rate": 96.67, ...}

print("ResultsManager public methods:")
print([m for m in dir(ResultsManager) if not m.startswith("_")])
```

<div class="admonition note">
<p class="admonition-title">Results are not checkpointed</p>
<p><code>ResultsManager</code> stores results in memory only. They are not saved to the benchmark checkpoint file. To persist results across sessions, use <code>export_results_to_file()</code> or save the <code>VerificationResultSet</code> directly.</p>
</div>

## 11. How Results Are Built: The FinalizeResult Stage

The [FinalizeResult stage](../verification-pipeline/) (stage 13) always runs as the last step in the pipeline. It constructs the `VerificationResult` from the accumulated `VerificationContext`:

1. Collects all artifacts written by previous stages
2. Extracts parsed ground truth and LLM responses from the parsed answer object
3. Determines which verification types were performed (template, rubric)
4. Aggregates token usage metadata across all stages
5. Computes the deterministic `result_id`
6. Assembles the nested sub-objects (`metadata`, `template`, `rubric`, `deep_judgment`, `deep_judgment_rubric`)
7. Handles partial failure: whatever artifacts are available get populated; missing data remains `None`

This stage handles both success and error cases. If the pipeline errors at stage 5, the finalize stage still runs and captures whatever was collected up to that point, and populates `metadata.failure` with a structured `Failure` (category, group, stage, reason) instead of leaving it `None`.

## 12. Next Steps

- [Verification Pipeline](../verification-pipeline/): The 13 stages that produce results
- [Evaluation Modes](../evaluation-modes/): How modes affect which result sub-objects are populated
- [Rubrics](../../../core_concepts/rubrics/): Defining the traits that populate rubric results
- [Answer Templates](../answer-templates/): Writing the `verify()` logic that produces `verify_result`
- [Error Analysis](../../../workflows/analyzing-results/error-analysis/): Downstream consumer of `VerificationResultSet` that renders passes and failures as navigable markdown files for agent-assisted review
