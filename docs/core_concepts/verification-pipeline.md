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

# Verification Pipeline

The **verification pipeline** is the execution engine that turns a question and its evaluation criteria into a structured `VerificationResult`. It runs a fixed sequence of up to 13 stages (with sub-stages 7a/7b and 11a/11b, plus an always-on placeholder-retry guard), each performing one step: validating a template, generating a response, parsing it, checking correctness, evaluating rubric traits, or finalizing the result.

The most important idea is that the pipeline is a **result-producing machine**: it always returns a `VerificationResult`, even when the model fails, abstains, or a stage throws an exception. Individual stages may skip, fail, or override earlier outcomes, but the pipeline itself never crashes without producing an output.

The pipeline does **not** define what correctness means (that is the [answer template](../answer-templates/)'s job), what quality criteria matter (that is the [rubric](../../../core_concepts/rubrics/)'s job), or which models to use (that is [VerificationConfig](../../../reference/configuration/verification-config/)'s job). The pipeline orchestrates those components in a specific order and manages the flow of data between them.

```python tags=["hide-cell"]
# Mock cell: ensures examples execute without live API keys.
# This cell is hidden in rendered documentation.
from unittest.mock import MagicMock, patch

mock_modules = {}
for mod in [
    "sqlalchemy", "sqlalchemy.engine", "sqlalchemy.orm", "sqlalchemy.ext",
    "sqlalchemy.ext.declarative", "sqlalchemy.engine",
    "sqlalchemy.sql", "sqlalchemy.event",
    "karenina.storage", "karenina.storage.base",
    "karenina.storage.engine", "karenina.storage.db_config",
    "karenina.storage.models", "karenina.storage.generated_models",
    "karenina.storage.auto_mapper", "karenina.storage.operations",
]:
    mock_modules[mod] = MagicMock()

with patch.dict("sys.modules", mock_modules):
    from karenina.benchmark.verification.stages import (
        StageOrchestrator,
        ArtifactKeys,
    )
    from karenina.schemas.entities.rubric import (
        Rubric,
        LLMRubricTrait,
        RegexRubricTrait,
    )
```

## 1. Why a Pipeline?

Structuring evaluation as an ordered sequence of stages provides four guarantees:

| Guarantee | Mechanism |
|-----------|-----------|
| **Consistency** | Every question traverses the same stages in the same order |
| **Auditability** | Each stage records its outcome in a shared context, making it clear exactly where a failure or override occurred |
| **Configurability** | Optional stages are included or skipped based on feature flags without affecting the rest of the sequence |
| **Error containment** | Guard stages catch problems early; `FinalizeResult` always runs to assemble whatever data is available into a valid result |

## 2. Anatomy: The 13 Stages

The pipeline groups its stages into functional categories. Within each category, stages execute in the order shown, with one exception: at runtime `AgenticRubricEvaluation` (11b) is appended after `DeepJudgmentRubricAutoFail` (12), so stage 12 executes before 11b even though the numbering lists 11b first. Stages 7a/7b are mutually exclusive sub-stages of one logical step (classical vs agentic parsing). The placeholder-retry guard always runs immediately after `TraceValidationAutoFail`.

```
  Category           Stage                            Always present?
 ─────────────────────────────────────────────────────────────────────
  Setup              1. ValidateTemplate               Template modes only
  Generation         2. GenerateAnswer                 Yes
  Guards             3. RecursionLimitAutoFail          Yes
                     4. TraceValidationAutoFail         Yes
                     -- PlaceholderRetryAutoFail        Yes (always-on guard)
  Pre-parse checks   5. AbstentionCheck                If abstention_enabled
                     6. SufficiencyCheck                If sufficiency_enabled (template modes)
  Template path     7a. ParseTemplate                  Template modes only (classical path)
                    7b. AgenticParseTemplate           Template modes only (if agentic_parsing)
                     8. VerifyTemplate                  Template modes only
  Enhancements       9. EmbeddingCheck                 Template modes only
                    10. DeepJudgmentAutoFail            If deep_judgment_mode != "disabled"
  Rubric path      11a. RubricEvaluation               If rubric has non-agentic traits + mode includes rubric
                   11b. AgenticRubricEvaluation        If rubric has agentic traits
                    12. DeepJudgmentRubricAutoFail      If rubric stages present AND deep judgment enabled
  Finalization      13. FinalizeResult                  Yes (always last)
```

**"Template modes"** means `template_only` or `template_and_rubric`. In `rubric_only` mode, stages 1 and 6 through 10 are omitted entirely at the orchestrator level. See [Evaluation Modes](../evaluation-modes/) for the full matrix and decision guidance.

**PlaceholderRetryAutoFail** is an always-on guard appended by `StageOrchestrator.from_config` between `TraceValidationAutoFail` and `AbstentionCheck`, in every evaluation mode. It catches the case where the final trace message is a `ModelRetryMiddleware` exhaustion placeholder (a sentinel inserted when the adapter's retry policy ran out) and auto-fails the result with a structured `Failure`. This covers both the trivial case (the first model call exhausts retries) and the mid-trace case (the model runs tools successfully but the final synthesis call fails). It has no feature flag and no per-stage config.

## 3. How It Works: Execution Lifecycle

### 3.1. Stage Composition

Before any stage runs, `StageOrchestrator.from_config()` builds the ordered stage list based on five inputs:

- `evaluation_mode`: which categories of stages to include
- `rubric`: whether a non-empty rubric is attached
- `abstention_enabled`, `sufficiency_enabled`, `deep_judgment_enabled`: feature flags

The resulting list is the pipeline's "program." Only stages in this list can execute. You can inspect the resulting stage list directly:

```python
# Build a pipeline for template_only mode (the default, no optional stages)
orch = StageOrchestrator.from_config(evaluation_mode="template_only")
print(f"template_only: {len(orch.stages)} stages")
for s in orch.stages:
    print(f"  {s.name}")
```

Enabling optional features adds stages to the list:

```python
# Same mode with optional pre-parse checks and deep judgment enabled
orch = StageOrchestrator.from_config(
    evaluation_mode="template_only",
    abstention_enabled=True,
    sufficiency_enabled=True,
    deep_judgment_enabled=True,
)
print(f"template_only (all options): {len(orch.stages)} stages")
for s in orch.stages:
    print(f"  {s.name}")
```

### 3.2. Dependency Validation

Before the first stage executes, the orchestrator validates the artifact dependency chain. Every stage declares what it `requires` (artifact keys it reads) and what it `produces` (new artifact keys it creates). The registry walks the stage list in order, confirming that every required artifact is produced by an earlier stage. If validation fails, the pipeline raises a `ValueError` before any work is done.

```python
# Inspect the artifact contract for each stage
orch = StageOrchestrator.from_config(evaluation_mode="template_only")
errors = orch.validate_dependencies()
print(f"Dependency validation: {'PASSED' if not errors else errors}\n")

print("Artifact flow:")
for s in orch.stages:
    print(f"  {s.name}")
    if s.requires:
        print(f"    reads:    {s.requires}")
    if s.produces:
        print(f"    produces: {s.produces}")
```

### 3.3. Sequential Execution

The orchestrator iterates through the stage list:

```
for each stage in the ordered list:
    1. Call stage.should_run(context)
       → If False, skip this stage entirely
    2. Call stage.execute(context)
       → Stage reads from and writes to the shared VerificationContext
    3. If the stage sets context.error:
       → Log a warning, but continue (do not halt)
    4. If the stage throws an exception:
       → Call context.mark_error(...), log the error, and continue
    5. Update execution_time on the context
```

The key implication: **the pipeline never halts early**. When `context.error` is set, subsequent stages skip themselves via their default `should_run()` logic (which returns `False` when an error exists), but `FinalizeResult` always runs because it overrides this default.

### 3.4. The VerificationContext

All stages share a single mutable `VerificationContext` object. It holds:

| Section | Contents | Example |
|---------|----------|---------|
| **Identity** | `question_id`, `template_id`, `question_text`, `template_code` | Fixed at creation |
| **Configuration** | `answering_model`, `parsing_model`, `rubric`, feature flags | Set by the runner |
| **Artifacts** | Stage outputs keyed by name: `raw_llm_response`, `parsed_answer`, `verify_result`, etc. | Grows as stages execute |
| **Result builder** | Fields that will become the final `VerificationResult` | Accumulated by stages, consumed by `FinalizeResult` |
| **Error state** | `failure` (structured `Failure` or `None`), `caveats` (list of informational flags) | Set by `mark_error()` |

Artifacts are the primary mechanism for inter-stage communication. A stage writes an artifact; a later stage reads it. The dependency validation system ensures this contract is satisfied before execution begins.

## 4. Stage Reference

### Setup

**ValidateTemplate** (stage 1) validates the template code, compiles it, and prepares the `Answer` class. If the template has a syntax error, a missing `Answer` class, or invalid fields, the stage sets `context.error` and the pipeline short-circuits to `FinalizeResult`.

### Generation

**GenerateAnswer** (stage 2) sends the question to the answering model and captures the response. It handles three paths:

| Path | When | What happens |
|------|------|--------------|
| **Cached answer** | `cached_answer_data` is set on the context | Reuses a previous response without calling the LLM; used to share one generation across multiple judges |
| **Agent (MCP)** | `answering_model.mcp_urls_dict` is configured | Runs the full agent loop with tool calls via the `AgentPort` adapter |
| **Direct LLM** | Otherwise | Calls `LLMPort.invoke()` directly |

The stage stores the raw response text, trace messages, usage metadata, and a `recursion_limit_reached` flag for the guard stages to inspect.

### Guards

**RecursionLimitAutoFail** (stage 3) fires only when `recursion_limit_reached` is `True`. It sets `verify_result=False` to record that the model got stuck in a loop rather than producing a proper answer. The trace and all metadata are preserved.

**TraceValidationAutoFail** (stage 4) checks that MCP agent traces end with an AI message. Non-MCP responses and manual-interface traces skip this check. When validation fails, it sets `verify_result=False` and stores a diagnostic error.

**PlaceholderRetryAutoFail** (always-on guard, between stages 4 and 5) inspects the response and trace for the `ModelRetryMiddleware` exhaustion sentinel. When the final trace message is that placeholder (the adapter's retry policy ran out before producing real content, whether on the first call or a mid-trace synthesis call), the stage records a structured `Failure` and sets `verify_result=False`. It is unconditionally appended by `StageOrchestrator.from_config` in every evaluation mode and has no feature flag.

### Pre-Parse Checks

Both pre-parse stages share a pattern: they use the parsing model to assess the raw response, and if the check triggers, they set `verify_result=False` and skip downstream parsing.

**AbstentionCheck** (stage 5) detects when the model refuses to answer ("I cannot answer that question"). When abstention is detected, parsing and verification are skipped.

**SufficiencyCheck** (stage 6) evaluates whether the response contains enough information to populate the template's fields. It retrieves the template's JSON schema and asks the parsing model to judge sufficiency. When the response is insufficient, parsing is skipped.

Both stages follow the `use_full_trace_for_template` setting, so by default the judge sees the same input it gets during template parsing (the final AI message, not the full agent trace).

Both stages record four metadata fields: `*_check_performed`, `*_detected`, `*_override_applied`, and `*_reasoning`.

### Template Processing

**ParseTemplate** (stage 7a) sends the response to the Judge LLM along with the template's JSON schema. The Judge extracts structured fields, producing a filled `Answer` instance. When `agentic_parsing=True`, this stage is replaced by `AgenticParseTemplateStage` (stage 7b), which performs investigation + extraction in two LLM calls; see [Agentic Evaluation](agentic-evaluation.md). Two fast paths exist:

- **Regex-only templates** (templates with no LLM-parsed fields) skip the LLM call entirely and create an empty `Answer()` for regex verification.
- **Deep judgment mode**: when `deep_judgment_mode` is `"full"` or `"reasoning_only"`, the stage also performs deep judgment processing on the response for each parsed field, enabling evidence-based verification.

The stage respects `use_full_trace_for_template`: when `False` (the default), only the final AI message is passed to the Judge; when `True`, the full agent trace is passed.

**VerifyTemplate** (stage 8) runs the template's `verify()` method to compare extracted values against ground truth. It also runs regex verification (`verify_regex()`) if the template defines regex patterns. The final `verify_result` is the AND of field verification and regex verification. If `verify_granular()` is implemented, per-field results are also recorded.

### Enhancements

**EmbeddingCheck** (stage 9) is a semantic similarity fallback. It is always included in template-mode pipelines, but its `should_run()` returns `True` only when `field_verification_result` is `False`. When it runs, it computes embedding similarity between the expected and extracted answers, asks the parsing model for a semantic equivalence judgment, and can override a failed field verification to `True` if the answers are semantically equivalent. It does **not** override regex verification failures.

!!! note "Embedding check activation"
    The embedding check stage runs whenever field verification fails. Whether it then computes similarity is governed by the `embedding_check_enabled` config field, which the stage passes through to the underlying computation. When that field is not set explicitly, the `EMBEDDING_CHECK` environment variable supplies its value. If neither is enabled, the stage records `embedding_check_performed=False` and returns without computing similarity.

**DeepJudgmentAutoFail** (stage 10) examines the deep-judgment metadata from `ParseTemplate`. If any parsed attributes lack supporting verbatim excerpts (`attributes_without_excerpts` is non-empty), it overrides `verify_result` to `False`. This stage skips itself if abstention was detected (abstention takes priority).

### Rubric Evaluation

**RubricEvaluation** (stage 11a) evaluates all applicable non-agentic rubric traits on the response. It handles four trait types through specialized evaluators:

| Trait type | Evaluator | LLM required |
|------------|-----------|:------------:|
| `LLMRubricTrait` | Parsing model (batch or sequential) | Yes |
| `RegexRubricTrait` | Pattern matching | No |
| `CallableRubricTrait` | Local Python function | No |
| `MetricRubricTrait` | Parsing model + confusion matrix computation | Yes |

The rubric evaluation input depends on `use_full_trace_for_rubric`: the full trace by default (`True`), or only the final AI message (`False`). Question-specific traits are merged with benchmark-level traits; duplicate names raise a `ValueError`.

When any LLM trait has deep judgment enabled (via the rubric deep judgment mode), the stage switches to a deep-judgment path that extracts verbatim excerpts supporting each trait assessment.

**AgenticRubricEvaluation** (stage 11b) runs when the rubric includes agentic traits. It deploys an agent per trait to investigate workspace artifacts, then extracts a structured score from the investigation findings. See [agentic rubric traits](rubrics/agentic-traits.md) and [Stage 11b internals](../../advanced-pipeline/agentic-rubric-evaluation.md).

**DeepJudgmentRubricAutoFail** (stage 12) mirrors stage 10 for rubric traits. If any trait lacks valid supporting excerpts after deep-judgment evaluation, it overrides `verify_result` to `False`. Abstention detection takes priority and skips this stage.

### Finalization

**FinalizeResult** (stage 13) always runs, even when earlier stages set errors. It assembles the accumulated context data into a structured `VerificationResult` with nested sub-objects:

| Sub-object | Contains |
|------------|----------|
| `result.metadata` | `question_id`, `template_id`, model identities, timing, `result_id` |
| `result.template` | Raw response, trace, parsed fields, verification results, embedding/abstention/sufficiency metadata |
| `result.rubric` | Trait scores (LLM, regex, callable, metric), evaluation strategy |
| `result.deep_judgment` | Template deep-judgment excerpts, reasoning, retry counts |
| `result.deep_judgment_rubric` | Rubric deep-judgment excerpts, reasoning, trait metadata |

## 5. The Three Evaluation Paths

Within a single pipeline run, template evaluation and rubric evaluation are independent paths that share the same generated response but operate on different inputs and produce different outputs.

```
GenerateAnswer
  │
  ├─── Template path (stages 7-10)
  │      Input:  evaluation input (final AI message or full trace)
  │      Judge:  parses into Answer schema
  │      Output: verify_result (bool), parsed fields, optional embeddings/deep judgment
  │
  ├─── Classical rubric path (stage 11)
  │      Input:  rubric evaluation input (full trace or final AI message)
  │      Judge:  evaluates each trait independently
  │      Output: trait scores (booleans, integers, metrics dicts)
  │
  └─── Agentic rubric path (stage 11b)
         Input:  workspace artifacts + optional trace (per context_mode)
         Agent:  investigates workspace, parser extracts score
         Output: agentic trait scores, investigation traces
```

The defaults are intentionally asymmetric:

- **Template parsing** uses the final AI message: `use_full_trace_for_template=False`
- **Rubric evaluation** uses the full trace: `use_full_trace_for_rubric=True`

Templates judge the final answer; rubrics often need to inspect the broader behavior that produced it.

In `template_and_rubric` mode, both paths execute. A rubric failure does not affect `verify_result`, and a template failure does not affect rubric scores. They are reported as separate sections of the same `VerificationResult`.

## 6. How Evaluation Mode Shapes the Pipeline

The [evaluation mode](../evaluation-modes/) controls which stages `StageOrchestrator.from_config()` includes. The rubric must have at least one trait for rubric stages to be added.

```python
# Build a rubric to demonstrate mode differences
rubric = Rubric(
    llm_traits=[
        LLMRubricTrait(
            name="conciseness",
            description="Is the response concise?",
            kind="boolean",
        ),
    ],
    regex_traits=[
        RegexRubricTrait(
            name="has_citations",
            description="Has bracket citations",
            pattern=r"\[\d+\]",
        ),
    ],
)

# Compare stage lists across all three evaluation modes
for mode in ["template_only", "template_and_rubric", "rubric_only"]:
    orch = StageOrchestrator.from_config(
        evaluation_mode=mode,
        rubric=rubric,
    )
    names = [s.name for s in orch.stages]
    print(f"{mode:25s} → {len(names):2d} stages: {names}")
```

The full matrix:

| Stage | `template_only` | `template_and_rubric` | `rubric_only` |
|-------|:---------------:|:---------------------:|:-------------:|
| ValidateTemplate | Yes | Yes | No |
| GenerateAnswer | Yes | Yes | Yes |
| RecursionLimitAutoFail | Yes | Yes | Yes |
| TraceValidationAutoFail | Yes | Yes | Yes |
| PlaceholderRetryAutoFail | Yes (always-on) | Yes (always-on) | Yes (always-on) |
| AbstentionCheck | If enabled | If enabled | If enabled |
| SufficiencyCheck | If enabled | If enabled | No |
| ParseTemplate (7a) / AgenticParseTemplate (7b) | Yes (one path) | Yes (one path) | No |
| VerifyTemplate | Yes | Yes | No |
| EmbeddingCheck | Present (runs conditionally) | Present (runs conditionally) | No |
| DeepJudgmentAutoFail | If enabled | If enabled | No |
| RubricEvaluation (11a) | No | If rubric has non-agentic traits | If rubric has non-agentic traits |
| AgenticRubricEvaluation (11b) | No | If agentic traits | If agentic traits |
| DeepJudgmentRubricAutoFail | No | If rubric stages present AND deep judgment enabled | If rubric stages present AND deep judgment enabled |
| FinalizeResult | Yes | Yes | Yes |

`rubric_only` truly skips template stages at the orchestrator level, not at the stage level. The stages are never instantiated.

## 7. When to Enable Optional Stages

| Feature flag | What it adds | Enable when | Cost |
|--------------|-------------|-------------|------|
| `abstention_enabled` | AbstentionCheck (stage 5) | Evaluating models that may refuse questions (safety, out-of-scope) | One additional LLM call per question to the parsing model |
| `sufficiency_enabled` | SufficiencyCheck (stage 6) | Template has many fields and partial responses are expected | One additional LLM call per question to the parsing model |
| `deep_judgment_mode` | Deep-judgment parsing in ParseTemplate + DeepJudgmentAutoFail (stage 10) | You need evidence-based verification with traceable excerpts, or want hallucination detection | Multiple additional LLM calls (excerpt extraction + validation per attribute) |
| `deep_judgment_rubric_mode` | Deep-judgment evaluation inside RubricEvaluation (stage 11a). The DeepJudgmentRubricAutoFail guard (stage 12) is appended only when the template `deep_judgment_mode` is non-disabled | Same as above, for rubric traits | Additional LLM calls per trait with deep judgment enabled |

**Heuristics**:

- If your benchmark contains only factual questions with clear correct answers, `template_only` with no optional stages is the fastest and cheapest path.
- Enable `abstention_enabled` when you expect models to sometimes refuse. Without it, a refusal is parsed as a normal response, producing a misleading `verify_result=False`.
- Enable `sufficiency_enabled` when your template has many fields and you want to avoid burning parsing tokens on responses that clearly lack the information needed.
- Set `deep_judgment_mode="full"` when you need an audit trail linking each parsed field back to verbatim text in the response.

## 8. Error Containment

The pipeline distinguishes three kinds of non-success:

| Kind | Set by | Effect on remaining stages | `metadata.failure` |
|------|--------|---------------------------|:------------------------:|
| **Auto-fail** | Guard stages (3, 4, 10, 12) | Sets `verify_result=False`; other stages continue normally | populated (group `autofail`) |
| **Override skip** | Pre-parse checks (5, 6) | Sets `verify_result=False`; downstream stages skip via `should_run()` conditions | populated (group `abstained`) |
| **Error** | Any stage setting `context.failure` or raising an exception | All subsequent stages except `FinalizeResult` skip via `should_run()` | populated (group varies) |

`PlaceholderRetryAutoFail` is the exception among the guard stages. Although it is an always-on guard, it marks a connection-category error (via `mark_error`), so its `Failure` lands in group `retry` (category `connection`), not `autofail`. The underlying cause is an exhausted in-agent retry budget rather than detectable bad model behavior. Because it marks an error, its downstream stages skip via `should_run()` like the Error row below, while `FinalizeResult` still runs.

The critical distinction: **auto-fails are expected outcomes** (the model behaved in a detectable bad way), while **errors are infrastructure failures** (a stage threw an exception or encountered an unrecoverable state). Both produce a valid `VerificationResult`; inspecting `metadata.failure.group` tells you which case applies.

The five `FailureGroup` values are:

| Group | When It Applies |
|-------|-----------------|
| `content` | Real answer was produced; `verify_template` returned `False` |
| `autofail` | A guard stage decided the run is unacceptable (recursion limit, trace validation, deep-judgment rejection) |
| `retry` | A retryable infrastructure error category (timeout, connection, rate limit, server error) exhausted its retry budget |
| `abstained` | The pre-parse guards detected refusal or insufficient response |
| `system` | Karenina-side or template-side problem: invalid template, parse exception, unexpected error |

Each group expands into one or more leaf categories (14 in total) that pinpoint the exact failure mode. The full enumeration, the eight-rule classifier priority, and the conventions for `Failure.details` and `Caveat` flags live in the [Failure and Caveats reference](../../reference/api/failure-and-caveats.md).

### How `should_run()` Works

Every stage inherits a default `should_run()` that returns `False` when `context.error` is set. Stages override this with additional conditions:

| Stage | Runs when (in addition to no error) |
|-------|--------------------------------------|
| RecursionLimitAutoFail | `recursion_limit_reached` is `True` |
| TraceValidationAutoFail | `raw_llm_response` artifact exists |
| PlaceholderRetryAutoFail | Trace contains the `ModelRetryMiddleware` exhaustion sentinel; otherwise no-op |
| AbstentionCheck | `abstention_enabled` is `True` and `recursion_limit_reached` is `False` |
| SufficiencyCheck | `sufficiency_enabled` is `True`, no recursion limit, no trace validation failure, no abstention detected |
| ParseTemplate | No prior failures, sufficiency not detected as insufficient, has both response and Answer artifacts |
| EmbeddingCheck | `field_verification_result` is `False` |
| DeepJudgmentAutoFail | Deep judgment was performed and attributes are missing excerpts |
| RubricEvaluation | Rubric has traits; trace validation did not fail (or full trace is being used) |
| AgenticRubricEvaluation | Rubric has agentic traits; no error |
| DeepJudgmentRubricAutoFail | Deep judgment rubric was performed and traits are missing excerpts |
| FinalizeResult | **Always** (overrides the error check) |

## 9. Worked Example: A Question Through the Pipeline

Consider a benchmark question in `template_and_rubric` mode with `abstention_enabled=True`:

> **Question**: "What is the putative target of venetoclax?"

**Stage 1: ValidateTemplate** compiles the answer template code, produces a validated `Answer` class with a `target` field.

**Stage 2: GenerateAnswer** sends the question to the answering model. The model responds: "Venetoclax targets BCL2 (B-cell lymphoma 2), a key anti-apoptotic protein."

**Stage 3: RecursionLimitAutoFail** checks `recursion_limit_reached`. It is `False` (no agent loop). Stage skips itself.

**Stage 4: TraceValidationAutoFail** checks if MCP is enabled. It is not. Stage skips the validation (non-MCP responses are always valid).

**PlaceholderRetryAutoFail** (always-on guard, between stages 4 and 5) inspects the response for the `ModelRetryMiddleware` exhaustion sentinel. The trace contains real content, so the stage skips itself.

**Stage 5: AbstentionCheck** asks the parsing model: "Did this response refuse to answer?" The model says no. `abstention_detected=False`. Pipeline continues.

**Stage 6: SufficiencyCheck** is not enabled in this configuration (`sufficiency_enabled=False`), so it is not present in the stage list. (When enabled, it would ask the parsing model whether the response contains enough information to populate the template's fields.)

**Stage 7a: ParseTemplate** sends the response and the `Answer` schema to the Judge LLM. The Judge extracts: `{"target": "BCL2"}`. A filled `Answer` instance is stored as `parsed_answer`. For coding tasks with `agentic_parsing=True`, this stage is replaced by `AgenticParseTemplateStage` (stage 7b), which uses a two-step process: an investigation agent independently verifies workspace artifacts, then a parser extracts structured data from the investigation findings. See [Agentic Evaluation](agentic-evaluation.md) for details.

**Stage 8: VerifyTemplate** calls `answer.verify()`, which compares `"BCL2"` against the ground truth `"BCL2"`. Field verification passes. Regex verification (if defined) also runs. `verify_result=True`.

**Stage 9: EmbeddingCheck** checks `field_verification_result`. It is `True`. Stage skips itself (embedding fallback is unnecessary).

**Stage 11a: RubricEvaluation** evaluates the attached non-agentic traits. A `RegexRubricTrait` for `has_citations` checks for `\[\d+\]` in the response. No match: `has_citations=False`. An `LLMRubricTrait` for `conciseness` asks the parsing model to judge. Result: `True`. (Stage 11b `AgenticRubricEvaluation` is not present in this rubric, since no agentic traits are defined.)

**Stage 13: FinalizeResult** assembles the `VerificationResult`:
- `result.template.verify_result = True`
- `result.rubric.llm_trait_scores = {"conciseness": True}`
- `result.rubric.regex_trait_scores = {"has_citations": False}`

Here is the stage list for that configuration:

```python
# The worked example pipeline: template_and_rubric with abstention
orch = StageOrchestrator.from_config(
    evaluation_mode="template_and_rubric",
    rubric=rubric,
    abstention_enabled=True,
)
print(f"{len(orch.stages)} stages in the pipeline:\n")
for i, s in enumerate(orch.stages, 1):
    print(f"  {i:2d}. {s.name}")
```

## Scheduling across multiple inference endpoints

`VerificationConfig` offers two knobs for multi-endpoint deployments (for example several vLLM servers, each hosting one answering model):

1. **`task_ordering`** controls the order in which expanded tasks are handed to the executor. The default `"auto"` picks `"distribute_answerers"` when the configured `answering_models` span more than one distinct `ModelIdentity.canonical_key`, and `"prefix_cache"` otherwise. `"distribute_answerers"` round-robins tasks across answerer identities while preserving prefix-cache locality inside each answerer group, so consecutive tasks target different inference endpoints and no single server is serialized. The other values (`"prefix_cache"`, `"generation_order"`, `"random"`) remain available for manual pinning.

2. **`answerer_concurrency_limits`** caps how many tasks may be in flight on a given answerer at once. Pass an `int` to apply the same cap to every configured answerer; pass a `dict[str, int]` keyed by `ModelConfig.id` for per-model caps. Answerers not present in the dict run uncapped. `None` (the default) disables caps.

Ordering and caps compose. Ordering smooths the steady state so the cap rarely engages; caps guarantee the ceiling under retry bursts or cache-hit cascades. Example for a run with three vLLM hosts and `async_max_workers=48`:

```python
VerificationConfig(
    answering_models=[qwen_a, qwen_b, qwen_c],
    parsing_models=[...],
    async_max_workers=48,
    answerer_concurrency_limits=16,  # every answerer capped at 16
    # task_ordering defaults to "auto" -> distribute_answerers
)
```

The cap is applied at task start (around the whole pipeline run for each task), so when the same endpoint hosts both the answerer and the parser the cap implicitly covers both phases. Stage-level caps are not currently supported.

## 10. Next Steps

- [Evaluation Modes](../evaluation-modes/): how the three modes shape which stages run
- [Prompt Assembly](../prompt-assembly/): how prompts are constructed for the Judge LLM and rubric evaluators
- [Results and Scoring](../results-and-scoring/): what the pipeline produces and how to read it
- [Answer Templates](../answer-templates/): writing the `verify()` logic that stage 8 executes
- [Rubrics](../../../core_concepts/rubrics/): defining the traits that stage 11 evaluates
- [Pipeline Internals](../../../advanced-pipeline/): deep dive into each stage, deep judgment, and custom stages
