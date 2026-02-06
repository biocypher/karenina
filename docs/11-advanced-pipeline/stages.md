# 13 Stages in Detail

This reference documents every stage in the verification pipeline. Each stage includes its purpose, when it runs, what it does, which configuration controls it, and how it affects the `VerificationResult`.

For an overview of how stages fit together, see [Advanced Pipeline Overview](index.md).

## Quick Reference

| # | Stage | Category | Included When | Runtime Skip Conditions |
|---|-------|----------|---------------|------------------------|
| 1 | [ValidateTemplate](#1-validatetemplate) | Setup | Template modes | — |
| 2 | [GenerateAnswer](#2-generateanswer) | LLM Call | Always | — |
| 3 | [RecursionLimitAutoFail](#3-recursionlimitautofail) | Guard | Always | Recursion not reached |
| 4 | [TraceValidationAutoFail](#4-tracevalidationautofail) | Guard | Always | Non-MCP or manual trace |
| 5 | [AbstentionCheck](#5-abstentioncheck) | Pre-Parse Check | `abstention_enabled` | Recursion limit hit |
| 6 | [SufficiencyCheck](#6-sufficiencycheck) | Pre-Parse Check | `sufficiency_enabled` | Recursion, trace fail, or abstention |
| 7 | [ParseTemplate](#7-parsetemplate) | LLM Call | Template modes | Recursion, trace fail, abstention, or insufficient |
| 8 | [VerifyTemplate](#8-verifytemplate) | Verification | Template modes | Recursion or abstention |
| 9 | [EmbeddingCheck](#9-embeddingcheck) | Enhancement | Template modes | Field verification passed |
| 10 | [DeepJudgmentAutoFail](#10-deepjudgmentautofail) | Enhancement | `deep_judgment_enabled` | Deep judgment not performed or no missing excerpts |
| 11 | [RubricEvaluation](#11-rubricevaluation) | Evaluation | Rubric configured + mode | Trace validation failed (filtered mode) |
| 12 | [DeepJudgmentRubricAutoFail](#12-deepjudgmentrubricautofail) | Enhancement | Rubric configured + mode | Deep judgment rubric not performed or no missing excerpts |
| 13 | [FinalizeResult](#13-finalizeresult) | Finalization | Always | Never skips |

---

## 1. ValidateTemplate

**Category**: Setup | **Class**: `ValidateTemplateStage`

### Purpose

Validates the Python template code and prepares the `Answer` class for downstream stages. This ensures the template compiles, defines a valid Pydantic model with a `verify()` method, and has the question ID injected for ground-truth comparison.

### When It Runs

- **Included**: `template_only` and `template_and_rubric` modes
- **Excluded**: `rubric_only` mode (no template to validate)
- **Always runs** when included (no runtime skip conditions)

### What It Does

1. Calls `validate_answer_template(template_code)` to parse and compile the Python code
2. Checks for a valid Pydantic `Answer` class with a `verify()` method
3. Stores the pre-injection class as `RawAnswer` (used by ParseTemplate for schema extraction)
4. Injects the question ID into the Answer class via `inject_question_id_into_answer_class()`
5. Stores the post-injection class as `Answer`

### Error Behavior

If validation fails, sets `context.error` which halts all subsequent stages (except FinalizeResult).

### Result Fields

| Field | Location | Description |
|-------|----------|-------------|
| `template_validation_error` | `template` | Error message if validation failed (None on success) |

### Configuration

None — always runs when included.

---

## 2. GenerateAnswer

**Category**: LLM Call | **Class**: `GenerateAnswerStage`

### Purpose

Invokes the answering LLM (or MCP-enabled agent) to generate a response to the question. This is the stage that actually calls the model being evaluated.

### When It Runs

- **Included**: All evaluation modes
- **Cache shortcut**: If `cached_answer_data` is provided (from answer caching in multi-model runs), injects cached data and skips the LLM call

### What It Does

1. Checks for cached answer data — if found, injects cached response and returns
2. Selects adapter based on configuration:
   - **AgentPort** (tool calling): if `answering_model.mcp_urls_dict` is configured
   - **LLMPort** (simple generation): otherwise
3. Builds messages: system prompt (if provided) + user message (question + few-shot examples if enabled)
4. Invokes the adapter and captures the response
5. Creates a `UsageTracker` and records token counts and agent metrics
6. Extracts structured `trace_messages` (list of Message objects) for MCP agents

### Error Behavior

Captures exception details (type, message, traceback) and marks `context.error`. All subsequent stages except FinalizeResult are skipped.

### Result Fields

| Field | Location | Description |
|-------|----------|-------------|
| `raw_llm_response` | `template` | Full text response from the model |
| `recursion_limit_reached` | `template` | True if agent hit max turns |
| `usage_metadata` | `template` | Token counts per LLM call |
| `agent_metrics` | `template` | MCP agent metrics (tool calls, turns) |
| `answering_mcp_servers` | `template` | List of MCP server names used |
| `trace_messages` | `template` | Structured Message list (MCP agents) |

### Configuration

| Field | Purpose |
|-------|---------|
| `answering_models` | Model(s) to invoke |
| `few_shot_config` | Whether to include few-shot examples |

---

## 3. RecursionLimitAutoFail

**Category**: Guard | **Class**: `RecursionLimitAutoFailStage`

### Purpose

Auto-fails verification if the answering agent exceeded its maximum recursion (turn) limit during response generation. This catches cases where an MCP agent got stuck in a loop.

### When It Runs

- **Included**: All evaluation modes (always after GenerateAnswer)
- **Runtime skip**: Only executes if `recursion_limit_reached` is True

### What It Does

1. Checks if `recursion_limit_reached` is True
2. If true: sets `verify_result = False` and `field_verification_result = False`
3. Does **not** set `completed_without_errors=False` — the trace and tokens are preserved for analysis

### Result Fields

Modifies `verify_result` and `field_verification_result` on the template section (no new fields added).

### Configuration

None — always included, but only triggers when the recursion limit was actually hit.

---

## 4. TraceValidationAutoFail

**Category**: Guard | **Class**: `TraceValidationAutoFailStage`

### Purpose

For MCP agent traces, validates that the trace ends with a valid AI message containing the final answer. Regular LLM responses and manual traces skip this validation.

### When It Runs

- **Included**: All evaluation modes (always after RecursionLimitAutoFail)
- **Runtime skip**: Non-MCP responses (`mcp_urls_dict` is None) or manual interface traces

### What It Does

1. Checks if this is an MCP run (`answering_model.mcp_urls_dict` is set)
2. Checks if interface is manual (manual traces are trusted)
3. For MCP traces: calls `extract_final_ai_message()` on the trace
4. If extraction fails: auto-fails by setting `verify_result = False`
5. Stores validation status and error message

### Result Fields

| Field | Location | Description |
|-------|----------|-------------|
| `trace_validation_failed` | internal | Whether validation failed |
| `trace_validation_error` | internal | Error message if failed |
| `mcp_enabled` | internal | Whether MCP was configured |

### Configuration

Determined by `answering_model.mcp_urls_dict` and `answering_model.interface`.

---

## 5. AbstentionCheck

**Category**: Pre-Parse Check | **Class**: `AbstentionCheckStage`

### Purpose

Detects whether the model refused to answer or abstained from responding. Runs before template parsing to skip expensive LLM calls when the model clearly didn't provide an answer.

### When It Runs

- **Included**: When `abstention_enabled=True` (all evaluation modes)
- **Runtime skip**: If recursion limit was reached (response is unreliable)

### What It Does

1. Sends the raw response to the parsing LLM with an abstention detection prompt
2. The judge decides whether the response contains a genuine refusal or abstention
3. If abstention detected: sets `verify_result = False` and signals downstream stages to skip parsing

### Detection Patterns

Detects:

- Explicit refusals ("I cannot answer")
- Evasive responses (responding to a different question)
- Deflection ("You should consult an expert")

Does NOT flag:

- Qualified answers with caveats
- Partial answers
- Approximate or estimated values

### Result Fields

| Field | Location | Description |
|-------|----------|-------------|
| `abstention_check_performed` | `template` | Whether check was attempted |
| `abstention_detected` | `template` | Whether abstention was found |
| `abstention_override_applied` | `template` | Whether verify_result was set to False |
| `abstention_reasoning` | `template` | Judge LLM's reasoning |

### Configuration

| Field | Purpose |
|-------|---------|
| `abstention_enabled` | Enable the check (default: False) |
| `prompt_config.abstention_detection` | Custom instructions for the abstention judge |

### Error Behavior

Non-fatal: if the check itself fails, logs a warning and continues the pipeline.

---

## 6. SufficiencyCheck

**Category**: Pre-Parse Check | **Class**: `SufficiencyCheckStage`

### Purpose

Detects whether the response contains sufficient information to populate the template schema. If insufficient, skips expensive template parsing.

### When It Runs

- **Included**: When `sufficiency_enabled=True` (template modes only — not `rubric_only`)
- **Runtime skip**: If recursion limit reached, trace validation failed, or abstention detected

### What It Does

1. Extracts the JSON schema from the `Answer` class
2. Sends the raw response + schema to the parsing LLM with a sufficiency detection prompt
3. The judge decides whether the response contains enough information to fill the schema
4. If insufficient: sets `verify_result = False` and signals downstream stages to skip parsing

### Result Fields

| Field | Location | Description |
|-------|----------|-------------|
| `sufficiency_check_performed` | `template` | Whether check was attempted |
| `sufficiency_detected` | `template` | Whether response is sufficient |
| `sufficiency_override_applied` | `template` | Whether verify_result was set to False |
| `sufficiency_reasoning` | `template` | Judge LLM's reasoning |

### Configuration

| Field | Purpose |
|-------|---------|
| `sufficiency_enabled` | Enable the check (default: False) |
| `prompt_config.sufficiency_detection` | Custom instructions for the sufficiency judge |

### Error Behavior

Non-fatal: if the check itself fails or schema extraction fails, logs a warning and continues.

!!! note
    SufficiencyCheck requires a template (it needs the schema to judge against). It is not available in `rubric_only` mode.

---

## 7. ParseTemplate

**Category**: LLM Call | **Class**: `ParseTemplateStage`

### Purpose

Uses the parsing (judge) LLM to extract structured data from the raw response into the Pydantic `Answer` class. This is where the "LLM-as-judge" pattern is executed — the judge fills in the template schema by interpreting the answering model's response.

### When It Runs

- **Included**: `template_only` and `template_and_rubric` modes
- **Runtime skip**: If recursion limit reached, trace validation failed, abstention detected, or sufficiency check found insufficient response

### What It Does

1. Creates a `TemplateEvaluator` with the parsing model, Answer class, and prompt config
2. If deep judgment is enabled, builds a config dict with excerpt extraction parameters
3. Calls `evaluator.parse_response()` which:
   - Sends the response + template schema to the judge LLM
   - Receives a populated `Answer` Pydantic object
   - Optionally extracts supporting text excerpts (deep judgment)
4. Stores the parsed answer and all deep-judgment metadata

### Deep Judgment Parsing

When `deep_judgment_enabled=True`, parsing includes excerpt extraction:

- For each template attribute, the judge extracts text spans from the response that support its answer
- Excerpts are validated using fuzzy matching against the original response
- If an excerpt can't be found after retry attempts, the attribute is flagged

### Result Fields

| Field | Location | Description |
|-------|----------|-------------|
| `parsed_gt_response` | `template` | Ground truth values from the template |
| `parsed_llm_response` | `template` | Judge's interpretation of the response |
| Deep judgment fields | `deep_judgment` | See below |

Deep judgment fields (when enabled):

| Field | Location | Description |
|-------|----------|-------------|
| `deep_judgment_performed` | `deep_judgment` | Whether deep judgment ran |
| `extracted_excerpts` | `deep_judgment` | Text spans per attribute |
| `attribute_reasoning` | `deep_judgment` | Judge's reasoning per attribute |
| `deep_judgment_stages_completed` | `deep_judgment` | Stages completed |
| `deep_judgment_model_calls` | `deep_judgment` | Total LLM calls made |
| `deep_judgment_excerpt_retry_count` | `deep_judgment` | Retries used |
| `attributes_without_excerpts` | `deep_judgment` | Attributes missing excerpts |
| `hallucination_risk_assessment` | `deep_judgment` | Risk per attribute (if search enabled) |

### Configuration

| Field | Purpose |
|-------|---------|
| `deep_judgment_enabled` | Enable excerpt extraction during parsing |
| `deep_judgment_max_excerpts_per_attribute` | Max excerpts per field |
| `deep_judgment_fuzzy_match_threshold` | Similarity threshold (0.0-1.0) |
| `deep_judgment_excerpt_retry_attempts` | Retry count for failed excerpts |
| `deep_judgment_search_enabled` | Enable web search for hallucination checking |
| `deep_judgment_search_tool` | Search tool name (e.g., "tavily") |
| `use_full_trace_for_template` | Use full trace vs. extracted AI message |
| `prompt_config.parsing` | Custom instructions for the parsing judge |

### Error Behavior

Fatal: if parsing fails, sets `context.error` which halts subsequent stages.

---

## 8. VerifyTemplate

**Category**: Verification | **Class**: `VerifyTemplateStage`

### Purpose

Runs the template's programmatic verification: compares parsed data against ground truth (field verification) and checks regex patterns against the raw response (regex verification).

### When It Runs

- **Included**: `template_only` and `template_and_rubric` modes
- **Runtime skip**: If recursion limit reached or abstention detected

### What It Does

1. **Field verification**: Calls `evaluator.verify_fields(parsed_answer)` which runs the template's `verify()` method comparing parsed values to ground truth
2. **Regex verification**: Calls `evaluator.verify_regex(parsed_answer, raw_llm_response)` which runs any regex traits defined on the template
3. **Combination**: `verify_result = field_success AND regex_success`
4. **Granular verification**: If the template defines `verify_granular()`, calls it to get a partial credit score (0.0-1.0)

### Result Fields

| Field | Location | Description |
|-------|----------|-------------|
| `template_verification_performed` | `template` | Whether verification ran |
| `verify_result` | `template` | Combined pass/fail (field AND regex) |
| `verify_granular_result` | `template` | Partial credit score (0.0-1.0, if available) |
| `field_verification_result` | internal | Field-only pass/fail |
| `regex_validations_performed` | `template` | Whether regex checks ran |
| `regex_validation_results` | `template` | Per-pattern pass/fail |
| `regex_validation_details` | `template` | Per-pattern detail messages |
| `regex_overall_success` | `template` | Combined regex success |
| `regex_extraction_results` | `template` | Actual regex match strings |

### Configuration

None — uses the template's `verify()` and regex trait definitions.

---

## 9. EmbeddingCheck

**Category**: Enhancement | **Class**: `EmbeddingCheckStage`

### Purpose

Provides a semantic similarity fallback when field verification failed. If the parsed response is semantically equivalent to ground truth despite failing exact comparison, the embedding check can override the field result to pass.

### When It Runs

- **Included**: `template_only` and `template_and_rubric` modes (always included when template stages are present)
- **Runtime skip**: If `field_verification_result` is True (no need to override a passing result)

### What It Does

1. Extracts ground truth and LLM response strings from the parsed answer
2. Computes embedding similarity between the two
3. Sends both to the parsing LLM for a semantic equivalence judgment
4. If the judge determines equivalence:
   - Overrides `field_verification_result` to True
   - Recalculates `verify_result = True AND regex_success`

### Result Fields

| Field | Location | Description |
|-------|----------|-------------|
| `embedding_check_performed` | `template` | Whether check was attempted |
| `embedding_similarity_score` | `template` | Cosine similarity (0.0-1.0) |
| `embedding_model_used` | `template` | Embedding model name |
| `embedding_override_applied` | `template` | Whether result was overridden |

### Configuration

| Field | Purpose |
|-------|---------|
| `embedding_check_enabled` | Enable the check (env: `EMBEDDING_CHECK`) |
| `embedding_check_model` | Embedding model (env: `EMBEDDING_CHECK_MODEL`, default: `all-MiniLM-L6-v2`) |
| `embedding_check_threshold` | Similarity threshold (env: `EMBEDDING_CHECK_THRESHOLD`, default: 0.85) |

!!! note
    The embedding check only overrides field verification failures — it never overrides regex failures.

---

## 10. DeepJudgmentAutoFail

**Category**: Enhancement | **Class**: `DeepJudgmentAutoFailStage`

### Purpose

Auto-fails verification if deep-judgment parsing found attributes without corroborating text excerpts after all retry attempts. This enforces the requirement that every claimed answer must be grounded in the actual response text.

### When It Runs

- **Included**: When `deep_judgment_enabled=True` (template modes only)
- **Runtime skip**: If deep judgment was not performed, no attributes are missing excerpts, or abstention was detected

### What It Does

1. Checks if `deep_judgment_performed` is True and `attributes_without_excerpts` is non-empty
2. If conditions met: sets `verify_result = False` and `field_verification_result = False`
3. Logs a warning listing the problematic attributes and retry counts

### Result Fields

Modifies `verify_result` and `field_verification_result` — no new fields added.

### Configuration

No direct configuration — auto-included when `deep_judgment_enabled=True`.

---

## 11. RubricEvaluation

**Category**: Evaluation | **Class**: `RubricEvaluationStage`

### Purpose

Evaluates the response against qualitative rubric criteria. This stage is independent of template verification — it operates on the raw response trace and can run even if template verification failed.

### When It Runs

- **Included**: When evaluation mode is `template_and_rubric` or `rubric_only` AND a rubric with traits is configured
- **Runtime skip**: If trace validation failed and `use_full_trace_for_rubric=False` (can't extract the AI message)

### What It Does

1. Selects the evaluation input:
   - Full trace (if `use_full_trace_for_rubric=True`)
   - Extracted final AI message only (if False)
2. Resolves deep judgment configuration for LLM traits (per-trait settings based on mode)
3. Evaluates each trait type:

| Trait Type | Evaluation Method | Returns |
|-----------|------------------|---------|
| LLM traits (boolean/score/literal) | Judge LLM evaluates against trait description | bool, int, or class index |
| Regex traits | Pattern matching against response text | bool |
| Callable traits | Custom Python function execution | bool or int |
| Metric traits | Judge LLM classifies items into confusion matrix buckets | precision, recall, F1 |

4. If LLM traits have deep judgment enabled, extracts supporting excerpts for each trait score
5. Tracks all LLM usage

### Result Fields

| Field | Location | Description |
|-------|----------|-------------|
| `rubric_evaluation_performed` | `rubric` | Whether evaluation ran |
| `rubric_evaluation_strategy` | `rubric` | "batch" or "sequential" |
| `llm_trait_scores` | `rubric` | Scores for boolean/score LLM traits |
| `llm_trait_labels` | `rubric` | Labels for literal LLM traits |
| `regex_trait_scores` | `rubric` | Results for regex traits |
| `callable_trait_scores` | `rubric` | Results for callable traits |
| `metric_trait_scores` | `rubric` | Precision/recall/F1 per metric trait |
| `metric_trait_confusion_lists` | `rubric` | TP/FN/FP/TN lists per metric trait |

Deep judgment rubric fields (when applicable):

| Field | Location | Description |
|-------|----------|-------------|
| `deep_judgment_rubric_performed` | `deep_judgment_rubric` | Whether deep judgment ran for rubric |
| `extracted_rubric_excerpts` | `deep_judgment_rubric` | Text spans per trait |
| `rubric_trait_reasoning` | `deep_judgment_rubric` | Judge reasoning per trait |
| `deep_judgment_rubric_scores` | `deep_judgment_rubric` | Scores from deep judgment path |
| `standard_rubric_scores` | `deep_judgment_rubric` | Scores from standard path (comparison) |
| `traits_without_valid_excerpts` | `deep_judgment_rubric` | Traits missing excerpts |

### Configuration

| Field | Purpose |
|-------|---------|
| `evaluation_mode` | Must be `template_and_rubric` or `rubric_only` |
| `rubric_evaluation_strategy` | "batch" (all traits together) or "sequential" (one at a time) |
| `use_full_trace_for_rubric` | Use full trace vs. extracted AI message |
| `deep_judgment_rubric_mode` | "disabled", "enable_all", "use_checkpoint", "custom" |
| `prompt_config.rubric_evaluation` | Custom instructions for rubric evaluation |

### Error Behavior

Non-fatal: if evaluation fails, sets `rubric_result=None`, logs a warning, and continues the pipeline.

---

## 12. DeepJudgmentRubricAutoFail

**Category**: Enhancement | **Class**: `DeepJudgmentRubricAutoFailStage`

### Purpose

Auto-fails verification if any deep-judgment-enabled rubric traits failed to extract valid supporting excerpts after all retry attempts.

### When It Runs

- **Included**: When rubric evaluation is included (same conditions as RubricEvaluation)
- **Runtime skip**: If deep judgment rubric was not performed, no traits are missing excerpts, or abstention was detected

### What It Does

1. Checks if `deep_judgment_rubric_performed` is True and `traits_without_valid_excerpts` is non-empty
2. If conditions met: sets `verify_result = False`
3. Logs a warning listing the problematic traits and retry metadata

### Result Fields

Modifies `verify_result` — no new fields added.

### Configuration

No direct configuration — auto-included when rubric evaluation is included.

---

## 13. FinalizeResult

**Category**: Finalization | **Class**: `FinalizeResultStage`

### Purpose

Constructs the final `VerificationResult` object from all accumulated context artifacts and result fields. This is the only stage that produces the result object returned to the caller.

### When It Runs

- **Included**: Always (all evaluation modes)
- **Never skips**: Runs even when `context.error` is set — this is an explicit exception to the normal error-halting behavior

### What It Does

1. Collects timing metadata (execution time, timestamp)
2. Builds `ModelIdentity` objects for answering and parsing models
3. Extracts ground truth and LLM parsed responses from the parsed answer (if available)
4. Converts structured `trace_messages` to serializable format
5. Aggregates all usage tracking data
6. Constructs the four result sub-objects:

| Sub-Object | Contents |
|-----------|----------|
| `VerificationResultMetadata` | Question ID, template ID, model identities, timestamps, error state |
| `VerificationResultTemplate` | Raw response, parsed data, verification results, check results, usage |
| `VerificationResultRubric` | Trait scores by type, labels, confusion lists, strategy |
| `VerificationResultDeepJudgment` | Excerpts, reasoning, model calls, retry counts, hallucination risk |
| `VerificationResultDeepJudgmentRubric` | Rubric excerpts, trait reasoning, per-trait metadata |

7. Assembles the final `VerificationResult` with root-level trace filtering fields

### Error Behavior

Always succeeds — handles both success and error cases. If previous stages failed, the result still contains whatever data was collected, with `completed_without_errors=False` and the error message.

### Configuration

None — always runs.

---

## Stage Interaction Patterns

### Skip Cascades

When a check stage detects a problem, it can trigger skip cascades in downstream stages:

```
AbstentionCheck detects refusal
  → SufficiencyCheck: skips (abstention takes priority)
  → ParseTemplate: skips (nothing useful to parse)
  → VerifyTemplate: skips (no parsed data)
  → EmbeddingCheck: skips (no field verification result)
  → DeepJudgmentAutoFail: skips (abstention detected)
  → DeepJudgmentRubricAutoFail: skips (abstention detected)
  → RubricEvaluation: still runs (evaluates raw trace independently)
  → FinalizeResult: always runs
```

```
SufficiencyCheck detects insufficient response
  → ParseTemplate: skips (not enough info to parse)
  → VerifyTemplate: skips (no parsed data)
  → Remaining template stages: skip accordingly
  → RubricEvaluation: still runs (independent of template)
  → FinalizeResult: always runs
```

### Override Semantics

Several stages can override the `verify_result`:

| Stage | Direction | When |
|-------|-----------|------|
| AbstentionCheck | Pass → Fail | Refusal detected |
| SufficiencyCheck | Pass → Fail | Insufficient response |
| RecursionLimitAutoFail | Pass → Fail | Agent hit max turns |
| TraceValidationAutoFail | Pass → Fail | MCP trace malformed |
| DeepJudgmentAutoFail | Pass → Fail | Missing excerpts for attributes |
| DeepJudgmentRubricAutoFail | Pass → Fail | Missing excerpts for traits |
| EmbeddingCheck | Fail → Pass | Semantic equivalence detected |

All overrides are logged at WARNING level to indicate the result was changed.

### Error Containment

Errors are contained per-question:

- Each question runs through the pipeline independently
- If one question fails, other questions continue unaffected
- Fatal errors (template validation, parsing failure) halt the pipeline for that question only
- Non-fatal errors (rubric evaluation failure) log a warning and continue
- FinalizeResult always runs, ensuring every question produces a `VerificationResult`

## Related

- [Advanced Pipeline Overview](index.md) — Stage ordering and evaluation mode matrix
- [Deep Judgment: Templates](deep-judgment-templates.md) — Excerpt extraction and fuzzy matching details
- [Deep Judgment: Rubrics](deep-judgment-rubrics.md) — Per-trait deep judgment configuration
- [Prompt Assembly System](prompt-assembly.md) — How prompts are constructed for LLM calls
- [VerificationConfig Reference](../10-configuration-reference/verification-config.md) — All 33 configuration fields
- [VerificationResult Structure](../07-analyzing-results/verification-result.md) — Complete result hierarchy
