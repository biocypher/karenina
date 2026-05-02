# Failure and Caveats Taxonomy

Every [`VerificationResult`](../../core_concepts/results-and-scoring.md) carries a structured non-pass verdict in `metadata.failure` and a list of informational flags in `metadata.caveats`. Both fields are populated by the [`finalize_result`](../../core_concepts/verification-pipeline.md) stage at the end of the pipeline using a single classifier, so they appear consistently regardless of which stage produced the underlying outcome. This page enumerates the values, the priority rules used to assign them, and the columns through which they reach analysis tools.

## 1. The Unified Failure Model

`Failure` is a Pydantic model with four authored fields (`category`, `stage`, `reason`, `details`) and one computed field (`group`). The `category` is the leaf classification; `group` is derived from `category` via `CATEGORY_TO_GROUP` and is used for high-level aggregation in DataFrames and dashboards. A `Failure` instance is attached to `metadata.failure` for every non-pass verdict; on success the field is `None`.

Caveats are orthogonal to the pass/fail verdict: they fire on conditions worth flagging regardless of whether the run succeeded. A passing result can carry caveats (e.g. `retries_used` after transient errors recovered before the final attempt), and a failing result can carry caveats unrelated to the failure cause.

## 2. Canonical Imports

```python
from karenina.schemas.results.failure import Failure, FailureCategory, FailureGroup, CATEGORY_TO_GROUP
from karenina.schemas.results.caveat import Caveat
```

## 3. FailureCategory: 14 Leaf Values

| Enum | String | Meaning | Emitting Stage | Group |
|------|--------|---------|----------------|-------|
| `CONTENT` | `content` | `verify_template` returned `False` (the parsed answer did not match ground truth) | `verify_template` | `content` |
| `RECURSION_LIMIT` | `recursion_limit` | Agent run hit its recursion limit before producing a final response | `recursion_limit_autofail` | `autofail` |
| `TRACE_VALIDATION` | `trace_validation` | MCP/agent trace did not end with a valid AI message | `trace_validation_autofail` | `autofail` |
| `DEEP_JUDGMENT` | `deep_judgment` | Deep-judgment excerpt validation rejected one or more template attributes | `deep_judgment_autofail` | `autofail` |
| `DEEP_JUDGMENT_RUBRIC` | `deep_judgment_rubric` | Deep-judgment excerpt validation rejected one or more rubric traits | `deep_judgment_rubric_autofail` | `autofail` |
| `TIMEOUT` | `timeout` | Retry budget for `ErrorCategory.TIMEOUT` exhausted | the failing stage (typically `generate_answer`) | `retry` |
| `CONNECTION` | `connection` | Retry budget for `ErrorCategory.CONNECTION` exhausted; also assigned when `PlaceholderRetryAutoFail` fires | the failing stage or `placeholder_retry_autofail` | `retry` |
| `RATE_LIMIT` | `rate_limit` | Retry budget for `ErrorCategory.RATE_LIMIT` exhausted | the failing stage | `retry` |
| `SERVER_ERROR` | `server_error` | Retry budget for `ErrorCategory.SERVER_ERROR` exhausted | the failing stage | `retry` |
| `ABSTENTION` | `abstention` | Abstention guard detected the model refused to answer | `abstention_check` | `abstained` |
| `SUFFICIENCY` | `sufficiency` | Sufficiency guard detected the response lacked information for the template | `sufficiency_check` | `abstained` |
| `TEMPLATE_VALIDATION` | `template_validation` | The answer template failed to compile/validate | `validate_template` | `system` |
| `PARSING` | `parsing` | The parse step (Judge LLM extraction) raised an exception | `parse_template` | `system` |
| `UNEXPECTED_ERROR` | `unexpected_error` | Catchall: any other stage exception not matched by an earlier rule | the stage that raised | `system` |

The `PlaceholderRetryAutoFail` guard is mapped to `CONNECTION` rather than its own category. It fires when LangChain's `ModelRetryMiddleware` emits its exhaustion sentinel ("Model call failed after ..."), which always reflects an in-agent retry budget being spent on infrastructure errors. This mapping makes MCP/agent retry exhaustion show up under `failure_group == 'retry'` alongside top-level retry exhaustion.

## 4. FailureGroup: 5 Aggregation Buckets

| Enum | String | When It Applies | Categories That Map Here |
|------|--------|-----------------|--------------------------|
| `CONTENT` | `content` | The model produced a real answer that did not match ground truth | `CONTENT` |
| `AUTOFAIL` | `autofail` | A guard stage decided the run is unacceptable regardless of `verify_result` | `RECURSION_LIMIT`, `TRACE_VALIDATION`, `DEEP_JUDGMENT`, `DEEP_JUDGMENT_RUBRIC` |
| `RETRY_EXHAUSTED` | `retry` | A retryable infrastructure error category exhausted its retry budget | `TIMEOUT`, `CONNECTION`, `RATE_LIMIT`, `SERVER_ERROR` |
| `ABSTAINED` | `abstained` | The pre-parse guards detected refusal or insufficient response | `ABSTENTION`, `SUFFICIENCY` |
| `SYSTEM` | `system` | Karenina-side or template-side problem (template invalid, parse exception, unexpected error) | `TEMPLATE_VALIDATION`, `PARSING`, `UNEXPECTED_ERROR` |

`FailureGroup.RETRY_EXHAUSTED` is exposed as the string `"retry"` (not `"retry_exhausted"`). All filtering on `failure_group` columns or fields should use the string values shown above.

The mapping is encoded in `CATEGORY_TO_GROUP`, a `dict[FailureCategory, FailureGroup]`. `Failure.group` is a `computed_field` that reads this dict; consumers cannot override the group by passing one to the constructor, and a `model_validator(mode="before")` silently drops any incoming `group` key. This guarantees the leaf category alone determines the bucket.

```python
from karenina.schemas.results.failure import Failure, FailureCategory, FailureGroup

# Incoming `group` is dropped; computed value comes from CATEGORY_TO_GROUP.
f = Failure(category=FailureCategory.PARSING, group="content", stage="parse_template", reason="invalid JSON")
assert f.group is FailureGroup.SYSTEM  # not CONTENT
```

## 5. Caveat: 3 Informational Flags

| Enum | String | Trigger |
|------|--------|---------|
| `PARTIAL_CONTENT` | `partial_content` | The `RESPONSE_TIMEOUT_PARTIAL` artifact is set on the context (typically because a streaming response was cut short and the truncated payload was preserved in `metadata.partial_content`) |
| `EMBEDDING_OVERRIDE` | `embedding_override` | The `EMBEDDING_OVERRIDE_APPLIED` artifact is set, meaning the [embedding-similarity fallback](../../core_concepts/verification-pipeline.md) flipped at least one field's verdict |
| `RETRIES_USED` | `retries_used` | Any entry in the `RETRY_COUNTS` artifact has `used > 0` (a retry attempt was made for some `ErrorCategory`, regardless of whether the run ultimately passed) |

Caveats are emitted by `collect_caveats(ctx)` in append-only order: `PARTIAL_CONTENT`, `EMBEDDING_OVERRIDE`, `RETRIES_USED`. The result is stored as `metadata.caveats: list[Caveat]`. In CSV exports and DataFrame columns the list is rendered as a comma-joined string of the enum values (empty string when no caveats fired).

## 6. Classifier Priority: 8 Rules

`classify_failure(ctx)` evaluates the finalized `VerificationContext` against eight rules in order; the first match wins. This ordering is the single source of truth for how a completed pipeline run is summarised.

| Rule | Condition | Result |
|------|-----------|--------|
| 1 | `failed_stage` matches a known autofail stage (`RecursionLimitAutoFail`, `TraceValidationAutoFail`, `DeepJudgmentAutoFail`, `DeepJudgmentRubricAutoFail`, `PlaceholderRetryAutoFail`) | `Failure` with the mapped category, stage = the autofail stage name |
| 2 | `template_verification_performed` is `True` and `verify_result` is `False` | `Failure(category=CONTENT, stage="verify_template")` (content fail wins over later retry exhaustion) |
| 3 | `error_category` is one of `{TIMEOUT, CONNECTION, RATE_LIMIT, SERVER_ERROR}` and `_retry_exhausted(ctx)` is `True` | `Failure` with the mapped retry category, `details = {"error_message", "retry_counts"}` |
| 4 | `abstention_detected` or `sufficiency_detected` flag set | `Failure(category=ABSTENTION` or `SUFFICIENCY)` with the guard's reasoning as `reason` |
| 5 | `TEMPLATE_VALIDATION_ERROR` artifact present | `Failure(category=TEMPLATE_VALIDATION, stage="validate_template")` |
| 6 | `error_stage == "parse_template"` and `ctx.error` truthy | `Failure(category=PARSING, stage="parse_template", details={"error_message"})` |
| 7 | `ctx.error` truthy (catchall) | `Failure(category=UNEXPECTED_ERROR, stage=error_stage or last_run_stage, details={"error_message"})` |
| 8 | None of the above | `None` (the run passed) |

Two consequences are worth highlighting. First, a content fail (rule 2) takes precedence over retry exhaustion (rule 3): a model that retried once and then produced a wrong answer is reported as `CONTENT`, not as `RETRY_EXHAUSTED`. Second, rules 5-7 only fire when the verify path did not run; once `verify_template` produces a verdict, the result is either `CONTENT` (rule 2) or pass (rule 8).

## 7. `Failure.details` Payload Conventions

The `details` field is an optional `dict[str, Any]` populated by the classifier for categories where the underlying error message and retry context are useful for post-hoc analysis.

| Category | Keys present in `details` |
|----------|---------------------------|
| `CONTENT` | `None` (no details payload) |
| Autofail categories (`RECURSION_LIMIT`, `TRACE_VALIDATION`, `DEEP_JUDGMENT`, `DEEP_JUDGMENT_RUBRIC`) | `None` |
| Retry-exhausted categories (`TIMEOUT`, `CONNECTION`, `RATE_LIMIT`, `SERVER_ERROR`) | `error_message` (str), `retry_counts` (`dict[str, dict[str, int]]` keyed by `ErrorCategory.value`, each entry `{"used", "budget"}`) |
| `ABSTENTION`, `SUFFICIENCY` | `None` |
| `TEMPLATE_VALIDATION` | `None` |
| `PARSING` | `error_message` (str) |
| `UNEXPECTED_ERROR` | `error_message` (str) |

`reason` is always a trimmed string capped at 500 characters. Callers should treat `details` as advisory: it carries the first error message attributed to the stage but does not include every retry attempt's individual exception.

## 8. Public Classifier API

```python
from karenina.benchmark.verification.failure_classifier import classify_failure, collect_caveats
```

Both functions take a `VerificationContext` (the finalized state object passed between pipeline stages) and are called once by the [`FinalizeResult` stage](../../core_concepts/verification-pipeline.md) to populate `metadata.failure` and `metadata.caveats` on every `VerificationResult`. They are public for callers that build their own pipelines on top of the same context shape; in normal usage you read the populated fields off the result, not call the classifier directly.

## 9. How These Fields Reach Your Analysis

Three surfaces expose the taxonomy uniformly:

- **DataFrames**: `success`, `failure_category`, `failure_group`, `failure_stage`, `failure_reason`, and `caveats` appear on every row of [Template, Rubric, and Judgment DataFrames](../../workflows/analyzing-results/dataframe-analysis.md). Enum-typed fields are rendered as their string `.value`; `caveats` is comma-joined.
- **CSV / database persistence**: the same five `failure_*` columns plus `failure_details_json` (a JSON-serialized `details` payload) and the comma-joined `caveats` column are written by [`ResultsIOManager`](../../workflows/analyzing-results/database-persistence.md) and round-tripped by `_row_to_result`.
- **JSON-LD checkpoints and in-memory access**: the `Failure` model is serialized intact (including `group` from the computed field) in `metadata.failure`; `caveats` is a list of enum strings.
