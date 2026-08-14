# DataFrame Column Definitions

Source: `karenina/src/karenina/schemas/dataframes/template.py` and
`karenina/src/karenina/schemas/dataframes/rubric.py`.

## Template Field DataFrame

Produced by `TemplateResults.to_dataframe()` /
`TemplateDataFrameBuilder.build_field_dataframe()`.

One row per **parsed field** per verification result.

### Status Columns

| Column | Type | Description |
|--------|------|-------------|
| `success` | bool | `True` when `metadata.failure is None` |
| `failure_category` | str or None | `Failure.category` (content, timeout, abstention, parsing, ...) |
| `failure_group` | str or None | `Failure.group` (content, autofail, retry, abstained, system) |
| `failure_stage` | str or None | `Failure.stage` — pipeline stage where it failed |
| `failure_reason` | str or None | `Failure.reason` — human-readable explanation |
| `caveats` | str | Comma-joined `Caveat` names (empty string when none) |
| `recursion_limit_reached` | bool | Whether an MCP agent hit its recursion limit |

### Identification Columns

| Column | Type | Description |
|--------|------|-------------|
| `question_id` | str | MD5 hash of question text |
| `template_id` | str | MD5 hash of template code |
| `question_text` | str | Full question text |
| `keywords` | list[str] or None | Question keywords |
| `replicate` | int or None | Replicate number |
| `answering_mcp_servers` | list[str] or None | MCP servers used for answering |

### Scenario Metadata Columns

| Column | Type | Description |
|--------|------|-------------|
| `scenario_id` | str or None | Scenario identifier |
| `scenario_node` | str or None | Current scenario node |
| `scenario_turn` | int or None | Current turn number |
| `scenario_path` | list[str] or None | Path of node ids through the scenario |


### Model Configuration Columns

| Column | Type | Description |
|--------|------|-------------|
| `answering_model` | str | Answering model display string |
| `parsing_model` | str | Parsing model display string |
| `answering_system_prompt` | str or None | System prompt for answering model |
| `parsing_system_prompt` | str or None | System prompt for parsing model |

### Template Response Columns

| Column | Type | Description |
|--------|------|-------------|
| `raw_llm_response` | str or None | Full text response from answering LLM |

### Field Comparison Columns (Exploded Dimension)

| Column | Type | Description |
|--------|------|-------------|
| `field_name` | str or None | Name of the parsed field |
| `gt_value` | any | Ground truth value for this field |
| `llm_value` | any | LLM-extracted value for this field |
| `field_match` | bool or None | Whether this field passed verification. Uses stored primitive results (`field_results`) when available; falls back to naive equality comparison for older results |
| `field_score` | float or None | Per-field graded score (1.0/0.0 for non-graded primitives; partial credit for graded primitives) |
| `field_type` | str or None | Python type name of the value |

### Verification Check Columns

| Column | Type | Description |
|--------|------|-------------|
| `verify_result` | bool or None | Template verify() pass/fail |
| `verify_granular_result` | any or None | Granular verification result from `verify_granular()` |
| `embedding_check_performed` | bool | Whether embedding check ran |
| `embedding_similarity_score` | float or None | Cosine similarity (0.0 to 1.0) |
| `embedding_model_used` | str or None | Embedding model name |
| `embedding_override_applied` | bool | Whether embedding overrode verify_result |
| `abstention_check_performed` | bool | Whether abstention check ran |
| `abstention_detected` | bool or None | Whether abstention was detected |
| `abstention_reasoning` | str or None | Reasoning for abstention detection |
| `abstention_override_applied` | bool | Whether abstention overrode verify_result |
| `sufficiency_check_performed` | bool | Whether sufficiency check ran |
| `sufficiency_detected` | bool or None | Whether the answer was judged insufficient |
| `sufficiency_reasoning` | str or None | Reasoning for sufficiency determination |
| `sufficiency_override_applied` | bool | Whether sufficiency overrode verify_result |
| `regex_validations_performed` | bool | Whether regex checks ran |
| `regex_overall_success` | bool or None | Whether all regex patterns matched |

### Execution Metadata Columns

| Column | Type | Description |
|--------|------|-------------|
| `execution_time` | float | Pipeline execution time in seconds |
| `timestamp` | str | Timestamp as stored (`YYYY-MM-DD HH:MM:SS`, space separator, no timezone) |
| `run_name` | str or None | Verification run name |
| `result_index` | int | Index into the results list |

---

## Template Regex DataFrame

Produced by `TemplateResults.to_regex_dataframe()` /
`TemplateDataFrameBuilder.build_regex_dataframe()`.

One row per **regex pattern** per verification result.

| Column | Type | Description |
|--------|------|-------------|
| `success` | bool | `True` when `metadata.failure is None` |
| `failure_category` | str or None | `Failure.category` |
| `failure_group` | str or None | `Failure.group` |
| `failure_stage` | str or None | `Failure.stage` |
| `failure_reason` | str or None | `Failure.reason` |
| `caveats` | str | Comma-joined `Caveat` names (empty string when none) |
| `question_id` | str | Question hash |
| `template_id` | str | Template hash |
| `replicate` | int or None | Replicate number |
| `answering_model` | str | Answering model |
| `parsing_model` | str | Parsing model |
| `pattern_name` | str | Name of the regex pattern |
| `pattern_regex` | str or None | The regex pattern string |
| `matched` | bool | Whether the pattern matched |
| `extracted_value` | str or None | Extracted value (if capture group) |
| `match_start` | int or None | Start position of match |
| `match_end` | int or None | End position of match |
| `full_match` | str or None | Full matched text |
| `raw_llm_response` | str | Full LLM response |
| `timestamp` | str | Timestamp as stored (`YYYY-MM-DD HH:MM:SS`, space separator, no timezone) |
| `run_name` | str or None | Run name |

---

## Template Usage DataFrame

Produced by `TemplateResults.to_usage_dataframe()` /
`TemplateDataFrameBuilder.build_usage_dataframe()`.

One row per **usage stage** per result (or one row per result with
`totals_only=True`).

| Column | Type | Description |
|--------|------|-------------|
| `success` | bool | `True` when `metadata.failure is None` |
| `failure_category` | str or None | `Failure.category` |
| `failure_group` | str or None | `Failure.group` |
| `failure_stage` | str or None | `Failure.stage` |
| `failure_reason` | str or None | `Failure.reason` |
| `caveats` | str | Comma-joined `Caveat` names (empty string when none) |
| `question_id` | str | Question hash |
| `template_id` | str | Template hash |
| `replicate` | int or None | Replicate number |
| `answering_model` | str | Answering model |
| `parsing_model` | str | Parsing model |
| `usage_stage` | str or None | Stage name (None when totals_only) |
| `input_tokens` | int or None | Input token count |
| `output_tokens` | int or None | Output token count |
| `total_tokens` | int or None | Total token count |
| `model_used` | str or None | Model used for this stage |
| `input_audio_tokens` | int or None | Audio input tokens |
| `input_cache_read_tokens` | int or None | Cache-read input tokens |
| `output_audio_tokens` | int or None | Audio output tokens |
| `output_reasoning_tokens` | int or None | Reasoning output tokens |
| `agent_iterations` | int or None | MCP agent loop iterations |
| `agent_tool_calls` | int or None | Total tool calls |
| `agent_tools_used` | list or None | Tool names used |
| `agent_suspected_failures` | int or None | Suspected failed tool calls |
| `timestamp` | str | Timestamp as stored (`YYYY-MM-DD HH:MM:SS`, space separator, no timezone) |
| `run_name` | str or None | Run name |

Usage stages: `answer_generation`, `parsing`, `rubric_evaluation`,
`abstention_check`, `sufficiency_check`. The `total` stage aggregates all.

---

## Rubric DataFrame

Produced by `RubricResults.to_dataframe()` /
`RubricDataFrameBuilder.build_dataframe()`.

One row per **trait** per result (metric traits exploded by metric name).

### Status Columns

| Column | Type | Description |
|--------|------|-------------|
| `success` | bool | `True` when `metadata.failure is None` |
| `failure_category` | str or None | `Failure.category` |
| `failure_group` | str or None | `Failure.group` |
| `failure_stage` | str or None | `Failure.stage` |
| `failure_reason` | str or None | `Failure.reason` |
| `caveats` | str | Comma-joined `Caveat` names (empty string when none) |

### Identification Columns

| Column | Type | Description |
|--------|------|-------------|
| `question_id` | str | Question hash |
| `template_id` | str | Template hash |
| `question_text` | str | Full question text |
| `keywords` | list[str] or None | Question keywords |
| `replicate` | int or None | Replicate number |
| `scenario_id` | str or None | Scenario identifier |
| `scenario_node` | str or None | Current scenario node |
| `scenario_turn` | int or None | Current turn number |
| `scenario_path` | list[str] or None | Path of node ids through the scenario |

### Model Configuration Columns

| Column | Type | Description |
|--------|------|-------------|
| `answering_model` | str | Answering model |
| `parsing_model` | str | Parsing model |
| `answering_system_prompt` | str or None | Answering system prompt |
| `parsing_system_prompt` | str or None | Parsing system prompt |

### Rubric Evaluation Metadata Columns

| Column | Type | Description |
|--------|------|-------------|
| `rubric_evaluation_performed` | bool | Whether rubric evaluation ran |
| `rubric_evaluation_strategy` | str or None | `batch` or `sequential` |

### Rubric Data Columns

| Column | Type | Description |
|--------|------|-------------|
| `trait_name` | str or None | Name of the rubric trait |
| `trait_score` | int, bool, float, or None | Trait score value |
| `trait_label` | str or None | Class label for literal-kind LLM traits. Column exists only when the run includes LLM traits (`None` on non-literal LLM rows); absent entirely otherwise |
| `trait_type` | str or None | `llm_score`, `llm_binary`, `llm_literal`, `regex`, `callable`, `metric`, `agentic` |
| `evaluation_method` | str or None | `llm`, `regex`, `callable`, `metric`, `agentic` |
| `metric_name` | str or None | Metric name (for metric traits: `precision`, `recall`, `f1`, etc.). Column exists only when the run includes metric traits |
| `investigation_trace` | str or None | Raw agent investigation trace (agentic traits only). Column exists only when the run includes agentic traits |

Column presence is trait-type driven: each row creator emits only its own
columns, and `_get_column_order` keeps columns that actually exist. A run
with only regex and callable traits yields 29 columns with none of
`trait_label`, `metric_name`, `investigation_trace`, or `confusion_*`.

### Confusion Matrix Columns (Metric Traits Only)

Present only when the run includes metric traits (absent — not null-valued —
otherwise):

| Column | Type | Description |
|--------|------|-------------|
| `confusion_tp` | list[str] or None | True positive items |
| `confusion_fp` | list[str] or None | False positive items |
| `confusion_fn` | list[str] or None | False negative items |
| `confusion_tn` | list[str] or None | True negative items |

### Execution Metadata Columns

| Column | Type | Description |
|--------|------|-------------|
| `execution_time` | float | Pipeline execution time in seconds |
| `timestamp` | str | Timestamp as stored (`YYYY-MM-DD HH:MM:SS`, space separator, no timezone) |
| `run_name` | str or None | Run name |
| `trait_provenance` | str or None | Trait source: `global`, `question_specific`, or `dynamic` |

### Deep Judgment Columns (Optional)

Present when `include_deep_judgment=True` on the builder:

| Column | Type | Description |
|--------|------|-------------|
| `trait_reasoning` | str or None | LLM reasoning for the trait score |
| `trait_excerpts` | str (JSON) | JSON-serialized list of excerpts |
| `trait_hallucination_risk` | str or None | Risk assessment (`none`, `low`, `medium`, `high`) |

### Dynamic Rubric Columns (Conditional)

When dynamic rubrics are used, additional `{trait_name}_skipped` boolean
columns appear, one per dynamic trait. `True` = skipped, `False` = promoted,
`NaN` = not applicable.
