# VerificationResult Structure

Every call to `run_verification()` returns a `VerificationResultSet` — a collection of `VerificationResult` objects, one per question verified. This page documents the complete structure of a `VerificationResult` so you know exactly what data is available for analysis.

## Overview

A `VerificationResult` has five top-level sections:

```
VerificationResult
├── metadata                    # Always present
│   ├── question_id, template_id, result_id
│   ├── answering (ModelIdentity), parsing (ModelIdentity)
│   ├── execution_time, timestamp
│   └── completed_without_errors, error
├── template                    # Present when template evaluation ran
│   ├── raw_llm_response, trace_messages
│   ├── parsed_llm_response, parsed_gt_response
│   ├── verify_result, verify_granular_result
│   ├── embedding_*, regex_*, abstention_*, sufficiency_*
│   └── usage_metadata, agent_metrics
├── rubric                      # Present when rubric evaluation ran
│   ├── llm_trait_scores, llm_trait_labels
│   ├── regex_trait_scores, callable_trait_scores
│   └── metric_trait_scores, metric_trait_confusion_lists
├── deep_judgment               # Present when deep judgment enabled (templates)
│   ├── extracted_excerpts, attribute_reasoning
│   └── hallucination_risk_assessment
└── deep_judgment_rubric        # Present when deep judgment enabled (rubrics)
    ├── extracted_rubric_excerpts, rubric_trait_reasoning
    ├── deep_judgment_rubric_scores, standard_rubric_scores
    └── trait_metadata, traits_without_valid_excerpts
```

Plus three shared trace-filtering fields at the root level:

| Field | Type | Description |
|-------|------|-------------|
| `evaluation_input` | `str \| None` | The input text passed to evaluation (full trace or final AI message) |
| `used_full_trace` | `bool` | Whether the full trace was used (`True`) or only the final AI message (`False`) |
| `trace_extraction_error` | `str \| None` | Error message if final AI message extraction failed |

---

## Accessing Fields

Fields are accessed through their nested section objects, not directly on the result:

```python
# Correct: access through section
result.metadata.question_id
result.template.verify_result
result.rubric.llm_trait_scores

# Optional sections need a None check
if result.template:
    print(result.template.verify_result)
if result.rubric:
    print(result.rubric.get_all_trait_scores())
```

---

## Metadata

The `metadata` section is **always present** on every result. It identifies the question, models, and execution context.

### Identification Fields

| Field | Type | Description |
|-------|------|-------------|
| `question_id` | `str` | Question identifier (URN format) |
| `template_id` | `str` | MD5 hash of the template code, or `"no_template"` if none |
| `result_id` | `str` | Deterministic 16-char SHA256 hash computed from question, models, timestamp, and replicate |
| `question_text` | `str` | Full text of the question |
| `raw_answer` | `str \| None` | Ground truth answer from the checkpoint (if provided) |
| `keywords` | `list[str] \| None` | Keywords associated with the question |
| `run_name` | `str \| None` | Optional name for this verification run |
| `replicate` | `int \| None` | Replicate number (1, 2, 3, ...) for repeated runs |

### Model Information

| Field | Type | Description |
|-------|------|-------------|
| `answering` | `ModelIdentity` | Identity of the answering model |
| `parsing` | `ModelIdentity` | Identity of the parsing (judge) model |
| `answering_system_prompt` | `str \| None` | System prompt used for the answering model |
| `parsing_system_prompt` | `str \| None` | System prompt used for the parsing model |

A `ModelIdentity` contains:

| Field | Type | Description |
|-------|------|-------------|
| `interface` | `str` | Adapter interface (e.g., `"langchain"`, `"claude_agent_sdk"`) |
| `model_name` | `str` | Model name (e.g., `"gpt-4.1"`, `"claude-sonnet-4-5-20250929"`) |
| `tools` | `list[str]` | MCP server names (only for answering models; empty for parsing models) |

Convenience properties:

- `metadata.answering_model` — Returns `answering.display_string` (e.g., `"langchain:gpt-4.1"`)
- `metadata.parsing_model` — Returns `parsing.display_string`

### Execution Fields

| Field | Type | Description |
|-------|------|-------------|
| `completed_without_errors` | `bool` | Whether verification completed successfully |
| `error` | `str \| None` | Error message if verification failed |
| `execution_time` | `float` | Execution time in seconds |
| `timestamp` | `str` | ISO timestamp of when verification was run |

---

## Template Results

The `template` section is present when template evaluation was performed (evaluation mode `template_only` or `template_and_rubric`). Access via `result.template`.

### Answer Generation

| Field | Type | Description |
|-------|------|-------------|
| `raw_llm_response` | `str` | Raw text response from the answering model |
| `trace_messages` | `list[dict]` | Full message trace (for multi-turn/agent interactions) |

### Parsed Responses

| Field | Type | Description |
|-------|------|-------------|
| `parsed_llm_response` | `dict \| None` | Fields extracted by the judge LLM (excludes `id` and `correct`) |
| `parsed_gt_response` | `dict \| None` | Ground truth values from the template's `correct` field |

Example:

```python
if result.template and result.template.parsed_llm_response:
    print("Extracted:", result.template.parsed_llm_response)
    # {"tissue": "pancreas", "gene": "KRAS"}

    print("Expected:", result.template.parsed_gt_response)
    # {"tissue": "pancreas", "gene": "KRAS"}
```

### Verification Outcomes

| Field | Type | Description |
|-------|------|-------------|
| `template_verification_performed` | `bool` | Whether `verify()` was executed |
| `verify_result` | `bool \| None` | Template verification result (`True`/`False`, or `None` if skipped) |
| `verify_granular_result` | `Any \| None` | Granular verification result from `verify_granular()` (e.g., `0.67` for partial credit) |

### Embedding Check

| Field | Type | Description |
|-------|------|-------------|
| `embedding_check_performed` | `bool` | Whether embedding check was attempted |
| `embedding_similarity_score` | `float \| None` | Similarity score between 0.0 and 1.0 |
| `embedding_override_applied` | `bool` | Whether the embedding check overrode the verify result |
| `embedding_model_used` | `str \| None` | Name of the embedding model used |

### Regex Validations

| Field | Type | Description |
|-------|------|-------------|
| `regex_validations_performed` | `bool` | Whether regex validation was attempted |
| `regex_validation_results` | `dict[str, bool] \| None` | Per-pattern pass/fail results |
| `regex_validation_details` | `dict[str, dict] \| None` | Detailed match information per pattern |
| `regex_overall_success` | `bool \| None` | Overall regex validation result |
| `regex_extraction_results` | `dict[str, Any] \| None` | What the regex patterns actually extracted |

### Abstention Detection

| Field | Type | Description |
|-------|------|-------------|
| `abstention_check_performed` | `bool` | Whether abstention check was attempted |
| `abstention_detected` | `bool \| None` | Whether the model refused or abstained from answering |
| `abstention_override_applied` | `bool` | Whether abstention check overrode the result |
| `abstention_reasoning` | `str \| None` | LLM's reasoning for the abstention determination |

### Sufficiency Detection

| Field | Type | Description |
|-------|------|-------------|
| `sufficiency_check_performed` | `bool` | Whether sufficiency check was attempted |
| `sufficiency_detected` | `bool \| None` | Whether the response has sufficient information (`True` = sufficient) |
| `sufficiency_override_applied` | `bool` | Whether sufficiency check overrode the result |
| `sufficiency_reasoning` | `str \| None` | LLM's reasoning for the sufficiency determination |

### MCP and Agent Metrics

| Field | Type | Description |
|-------|------|-------------|
| `recursion_limit_reached` | `bool` | Whether the agent hit its recursion limit |
| `answering_mcp_servers` | `list[str] \| None` | Names of MCP servers attached to the answering model |
| `agent_metrics` | `dict \| None` | MCP agent execution metrics (see structure below) |

Agent metrics structure (only present when an agent was used):

```python
{
    "iterations": 3,               # Number of agent think-act cycles
    "tool_calls": 5,               # Total tool invocations
    "tools_used": ["mcp__brave_search", "mcp__read_resource"],
    "suspect_failed_tool_calls": 2, # Tool calls with error-like output
    "suspect_failed_tools": ["mcp__brave_search"]
}
```

### Token Usage

| Field | Type | Description |
|-------|------|-------------|
| `usage_metadata` | `dict \| None` | Token usage breakdown by verification stage |

Usage metadata structure:

```python
{
    "answer_generation": {
        "input_tokens": 150, "output_tokens": 200, "total_tokens": 350,
        "model": "claude-haiku-4-5"
    },
    "parsing": {"input_tokens": 200, "output_tokens": 80, "total_tokens": 280},
    "rubric_evaluation": {...},
    "abstention_check": {...},
    "total": {"input_tokens": 600, "output_tokens": 360, "total_tokens": 960}
}
```

---

## Rubric Results

The `rubric` section is present when rubric evaluation was performed (evaluation mode `template_and_rubric` or `rubric_only`). Access via `result.rubric`.

### Evaluation Status

| Field | Type | Description |
|-------|------|-------------|
| `rubric_evaluation_performed` | `bool` | Whether rubric evaluation was executed |
| `rubric_evaluation_strategy` | `str \| None` | Strategy used: `"batch"` or `"sequential"` |

### Trait Scores by Type

Scores are split by trait type for type-safe access:

| Field | Type | Description |
|-------|------|-------------|
| `llm_trait_scores` | `dict[str, int \| bool] \| None` | LLM-evaluated traits — boolean (`True`/`False`) for boolean kind, integer score for score kind, class index for literal kind |
| `llm_trait_labels` | `dict[str, str] \| None` | Human-readable class names for literal kind traits (e.g., `{"tone": "Professional"}`) |
| `regex_trait_scores` | `dict[str, bool] \| None` | Regex-based traits (boolean pass/fail) |
| `callable_trait_scores` | `dict[str, bool \| int] \| None` | Callable-based traits (boolean or integer score) |
| `metric_trait_scores` | `dict[str, dict[str, float]] \| None` | Metric traits with nested metrics (e.g., `{"extraction": {"precision": 1.0, "recall": 0.8, "f1": 0.89}}`) |

### Metric Trait Details

| Field | Type | Description |
|-------|------|-------------|
| `metric_trait_confusion_lists` | `dict[str, dict[str, list[str]]] \| None` | Confusion matrix lists per metric trait |

Confusion lists structure:

```python
{
    "feature_extraction": {
        "tp": ["feature_A", "feature_B"],  # True positives
        "tn": ["irrelevant_1"],            # True negatives
        "fp": ["hallucinated_1"],          # False positives
        "fn": ["missed_feature"]           # False negatives
    }
}
```

### Convenience Methods

The `VerificationResultRubric` provides helper methods for working with trait scores:

| Method | Returns | Description |
|--------|---------|-------------|
| `get_all_trait_scores()` | `dict` | All trait scores across all types in a flat dictionary |
| `get_trait_by_name(name)` | `tuple \| None` | Look up a trait by name — returns `(value, trait_type)` or `None` |
| `get_llm_trait_labels()` | `dict[str, str]` | Class labels for literal kind LLM traits |

Example:

```python
if result.rubric:
    # Get all scores at once
    all_scores = result.rubric.get_all_trait_scores()
    # {"clarity": 4, "has_citations": True, "extraction": {"precision": 1.0, ...}}

    # Look up a specific trait
    match = result.rubric.get_trait_by_name("clarity")
    if match:
        value, trait_type = match  # (4, "llm")

    # Get literal trait labels
    labels = result.rubric.get_llm_trait_labels()
    # {"response_type": "Factual", "tone": "Professional"}
```

---

## Deep Judgment (Templates)

The `deep_judgment` section is present when deep judgment was enabled for template evaluation. It provides multi-stage parsing with excerpts and reasoning for each template attribute. Access via `result.deep_judgment`.

### Status Fields

| Field | Type | Description |
|-------|------|-------------|
| `deep_judgment_enabled` | `bool` | Whether deep judgment was configured |
| `deep_judgment_performed` | `bool` | Whether deep judgment was successfully executed |
| `deep_judgment_stages_completed` | `list[str] \| None` | Stages completed: `["excerpts", "reasoning", "parameters"]` |
| `deep_judgment_model_calls` | `int` | Number of LLM invocations for deep judgment |
| `deep_judgment_excerpt_retry_count` | `int` | Number of retries for excerpt validation |
| `attributes_without_excerpts` | `list[str] \| None` | Attributes with no corroborating excerpts found |

### Excerpts and Reasoning

| Field | Type | Description |
|-------|------|-------------|
| `extracted_excerpts` | `dict[str, list[dict]] \| None` | Extracted excerpts per attribute |
| `attribute_reasoning` | `dict[str, str] \| None` | Reasoning traces per attribute |

Excerpt structure:

```python
{
    "tissue": [
        {
            "text": "KRAS is most essential in the pancreas",
            "confidence": "high",           # "low", "medium", or "high"
            "similarity_score": 0.92,
            # Only when search enabled:
            "search_results": "External validation text...",
            "hallucination_risk": "none",   # "none", "low", "medium", or "high"
            "hallucination_justification": "Strong external evidence supports this claim"
        }
    ]
}
```

An empty list `[]` for an attribute indicates no excerpts were found (e.g., the model refused to answer or no corroborating evidence exists). Reasoning can still exist for attributes without excerpts, explaining why none were found.

### Search-Enhanced Fields

| Field | Type | Description |
|-------|------|-------------|
| `deep_judgment_search_enabled` | `bool` | Whether search enhancement was enabled |
| `hallucination_risk_assessment` | `dict[str, str] \| None` | Per-attribute hallucination risk (`"none"`, `"low"`, `"medium"`, `"high"`) |

---

## Deep Judgment (Rubrics)

The `deep_judgment_rubric` section is present when deep judgment was enabled for rubric trait evaluation. It provides per-trait excerpts, reasoning, and scores. Access via `result.deep_judgment_rubric`.

### Status and Scores

| Field | Type | Description |
|-------|------|-------------|
| `deep_judgment_rubric_performed` | `bool` | Whether deep judgment rubric evaluation was executed |
| `deep_judgment_rubric_scores` | `dict[str, int \| bool] \| None` | Scores for traits evaluated with deep judgment |
| `standard_rubric_scores` | `dict[str, int \| bool] \| None` | Scores for traits evaluated without deep judgment (in the same rubric) |

### Per-Trait Excerpts and Reasoning

| Field | Type | Description |
|-------|------|-------------|
| `extracted_rubric_excerpts` | `dict[str, list[dict]] \| None` | Extracted excerpts per trait (same structure as template excerpts) |
| `rubric_trait_reasoning` | `dict[str, str] \| None` | Reasoning text per trait explaining how the score was determined |

### Per-Trait Metadata

| Field | Type | Description |
|-------|------|-------------|
| `trait_metadata` | `dict[str, dict] \| None` | Detailed tracking per trait |

Trait metadata structure:

```python
{
    "clarity": {
        "stages_completed": ["excerpt_extraction", "reasoning_generation", "score_extraction"],
        "model_calls": 3,
        "had_excerpts": True,
        "excerpt_retry_count": 1,
        "excerpt_validation_failed": False
    }
}
```

### Auto-Fail and Search Fields

| Field | Type | Description |
|-------|------|-------------|
| `traits_without_valid_excerpts` | `list[str] \| None` | Trait names that failed to extract valid excerpts after all retries (triggers auto-fail) |
| `rubric_hallucination_risk_assessment` | `dict[str, dict] \| None` | Per-trait hallucination risk assessment |

Hallucination risk structure:

```python
{
    "clarity": {
        "overall_risk": "low",
        "per_excerpt_risks": ["none", "low"]
    }
}
```

### Aggregate Statistics

| Field | Type | Description |
|-------|------|-------------|
| `total_deep_judgment_model_calls` | `int` | Total LLM calls across all deep judgment traits |
| `total_traits_evaluated` | `int` | Number of traits evaluated with deep judgment |
| `total_excerpt_retries` | `int` | Total retry attempts across all traits |

---

## Field Count Summary

For verification against source code, here are the field counts per section:

| Section | Fields | Convenience Methods |
|---------|--------|---------------------|
| Root level | 3 (evaluation_input, used_full_trace, trace_extraction_error) | — |
| `metadata` | 16 fields + 2 properties | `answering_model`, `parsing_model` |
| `template` | 28 fields | — |
| `rubric` | 8 fields | `get_all_trait_scores()`, `get_trait_by_name()`, `get_llm_trait_labels()` |
| `deep_judgment` | 10 fields | — |
| `deep_judgment_rubric` | 11 fields | — |

---

## Next Steps

- [DataFrame Analysis](dataframe-analysis.md) — Convert results to pandas DataFrames for deeper analysis
- [Exporting Results](exporting.md) — Save results as JSON or CSV
- [Iterating](iterating.md) — Improve templates and rubrics based on results
- [Running Verification](../06-running-verification/python-api.md) — If you haven't run verification yet
