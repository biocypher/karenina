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

The **verification pipeline** is the execution engine that turns a question and its evaluation criteria into a structured `VerificationResult`. It runs a fixed sequence of up to 13 stages, each performing one step: validating a template, generating a response, parsing it, checking correctness, evaluating rubric traits, or finalizing the result.

The most important idea is that the pipeline is a **result-producing machine**: it always returns a `VerificationResult`, even when the model fails, abstains, or a stage throws an exception. Individual stages may skip, fail, or override earlier outcomes, but the pipeline itself never crashes without producing an output.

The pipeline does **not** define what correctness means (that is the [answer template](../notebooks/core_concepts/answer-templates.ipynb)'s job), what quality criteria matter (that is the [rubric](rubrics/index.md)'s job), or which models to use (that is [VerificationConfig](../reference/configuration/verification-config.md)'s job). The pipeline orchestrates those components in a specific order and manages the flow of data between them.

```python tags=["hide-cell"]
# Mock cell: ensures examples execute without live API keys.
# This cell is hidden in rendered documentation.
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
    from karenina.benchmark.verification.stages import (
        StageOrchestrator,
        ArtifactKeys,
    )
    from karenina.schemas.entities.rubric import (
        Rubric,
        LLMRubricTrait,
        RegexTrait,
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

The pipeline groups its stages into functional categories. Within each category, stages execute in the order shown.

```
  Category           Stage                            Always present?
 ─────────────────────────────────────────────────────────────────────
  Setup              1. ValidateTemplate               Template modes only
  Generation         2. GenerateAnswer                 Yes
  Guards             3. RecursionLimitAutoFail          Yes
                     4. TraceValidationAutoFail         Yes
  Pre-parse checks   5. AbstentionCheck                If abstention_enabled
                     6. SufficiencyCheck                If sufficiency_enabled (template modes)
  Template path      7. ParseTemplate                  Template modes only
                     8. VerifyTemplate                  Template modes only
  Enhancements       9. EmbeddingCheck                 Template modes only
                    10. DeepJudgmentAutoFail            If deep_judgment_enabled
  Rubric path       11. RubricEvaluation               If rubric has traits + mode includes rubric
                    12. DeepJudgmentRubricAutoFail      If rubric stages present
  Finalization      13. FinalizeResult                  Yes (always last)
```

**"Template modes"** means `template_only` or `template_and_rubric`. In `rubric_only` mode, stages 1 and 6 through 10 are omitted entirely at the orchestrator level. See [Evaluation Modes](../notebooks/core_concepts/evaluation-modes.ipynb) for the full matrix and decision guidance.

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
| **Error state** | `error` (message or `None`), `completed_without_errors` (bool) | Set by `mark_error()` |

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

### Pre-Parse Checks

Both pre-parse stages share a pattern: they use the parsing model to assess the raw response, and if the check triggers, they set `verify_result=False` and skip downstream parsing.

**AbstentionCheck** (stage 5) detects when the model refuses to answer ("I cannot answer that question"). When abstention is detected, parsing and verification are skipped.

**SufficiencyCheck** (stage 6) evaluates whether the response contains enough information to populate the template's fields. It retrieves the template's JSON schema and asks the parsing model to judge sufficiency. When the response is insufficient, parsing is skipped.

Both stages record four metadata fields: `*_check_performed`, `*_detected`, `*_override_applied`, and `*_reasoning`.

### Template Processing

**ParseTemplate** (stage 7) sends the response to the Judge LLM along with the template's JSON schema. The Judge extracts structured fields, producing a filled `Answer` instance. Two fast paths exist:

- **Regex-only templates** (templates with no LLM-parsed fields) skip the LLM call entirely and create an empty `Answer()` for regex verification.
- **Deep judgment mode**: when `deep_judgment_enabled` is `True`, the stage also extracts verbatim excerpts from the response for each parsed field, enabling evidence-based verification.

The stage respects `use_full_trace_for_template`: when `False` (the default), only the final AI message is passed to the Judge; when `True`, the full agent trace is passed.

**VerifyTemplate** (stage 8) runs the template's `verify()` method to compare extracted values against ground truth. It also runs regex verification (`verify_regex()`) if the template defines regex patterns. The final `verify_result` is the AND of field verification and regex verification. If `verify_granular()` is implemented, per-field results are also recorded.

### Enhancements

**EmbeddingCheck** (stage 9) is a semantic similarity fallback. It is always included in template-mode pipelines, but its `should_run()` returns `True` only when `field_verification_result` is `False`. When it runs, it computes embedding similarity between the expected and extracted answers, asks the parsing model for a semantic equivalence judgment, and can override a failed field verification to `True` if the answers are semantically equivalent. It does **not** override regex verification failures.

!!! note "Embedding check activation"
    The embedding check stage runs whenever field verification fails. The underlying computation also checks the `EMBEDDING_CHECK` environment variable (or the `embedding_check_enabled` config field, which sets it). If the env var is not enabled, the stage records `embedding_check_performed=False` and returns without computing similarity.

**DeepJudgmentAutoFail** (stage 10) examines the deep-judgment metadata from `ParseTemplate`. If any parsed attributes lack supporting verbatim excerpts (`attributes_without_excerpts` is non-empty), it overrides `verify_result` to `False`. This stage skips itself if abstention was detected (abstention takes priority).

### Rubric Evaluation

**RubricEvaluation** (stage 11) evaluates all applicable rubric traits on the response. It handles four trait types through specialized evaluators:

| Trait type | Evaluator | LLM required |
|------------|-----------|:------------:|
| `LLMRubricTrait` | Parsing model (batch or sequential) | Yes |
| `RegexTrait` | Pattern matching | No |
| `CallableTrait` | Local Python function | No |
| `MetricRubricTrait` | Parsing model + confusion matrix computation | Yes |

The rubric evaluation input depends on `use_full_trace_for_rubric`: the full trace by default (`True`), or only the final AI message (`False`). Question-specific traits are merged with benchmark-level traits; duplicate names raise a `ValueError`.

When any LLM trait has `deep_judgment_enabled=True`, the stage switches to a deep-judgment path that extracts verbatim excerpts supporting each trait assessment.

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

## 5. The Two Evaluation Paths

Within a single pipeline run, template evaluation and rubric evaluation are independent paths that share the same generated response but operate on different inputs and produce different outputs.

```
GenerateAnswer
  │
  ├─── Template path (stages 7-10)
  │      Input:  evaluation input (final AI message or full trace)
  │      Judge:  parses into Answer schema
  │      Output: verify_result (bool), parsed fields, optional embeddings/deep judgment
  │
  └─── Rubric path (stages 11-12)
         Input:  rubric evaluation input (full trace or final AI message)
         Judge:  evaluates each trait independently
         Output: trait scores (booleans, integers, metrics dicts)
```

The defaults are intentionally asymmetric:

- **Template parsing** uses the final AI message: `use_full_trace_for_template=False`
- **Rubric evaluation** uses the full trace: `use_full_trace_for_rubric=True`

Templates judge the final answer; rubrics often need to inspect the broader behavior that produced it.

In `template_and_rubric` mode, both paths execute. A rubric failure does not affect `verify_result`, and a template failure does not affect rubric scores. They are reported as separate sections of the same `VerificationResult`.

## 6. How Evaluation Mode Shapes the Pipeline

The [evaluation mode](../notebooks/core_concepts/evaluation-modes.ipynb) controls which stages `StageOrchestrator.from_config()` includes. The rubric must have at least one trait for rubric stages to be added.

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
        RegexTrait(
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
| AbstentionCheck | If enabled | If enabled | If enabled |
| SufficiencyCheck | If enabled | If enabled | No |
| ParseTemplate | Yes | Yes | No |
| VerifyTemplate | Yes | Yes | No |
| EmbeddingCheck | Present (runs conditionally) | Present (runs conditionally) | No |
| DeepJudgmentAutoFail | If enabled | If enabled | No |
| RubricEvaluation | No | If rubric has traits | If rubric has traits |
| DeepJudgmentRubricAutoFail | No | If rubric stages present | If rubric stages present |
| FinalizeResult | Yes | Yes | Yes |

`rubric_only` truly skips template stages at the orchestrator level, not at the stage level. The stages are never instantiated.

## 7. When to Enable Optional Stages

| Feature flag | What it adds | Enable when | Cost |
|--------------|-------------|-------------|------|
| `abstention_enabled` | AbstentionCheck (stage 5) | Evaluating models that may refuse questions (safety, out-of-scope) | One additional LLM call per question to the parsing model |
| `sufficiency_enabled` | SufficiencyCheck (stage 6) | Template has many fields and partial responses are expected | One additional LLM call per question to the parsing model |
| `deep_judgment_enabled` | Deep-judgment parsing in ParseTemplate + DeepJudgmentAutoFail (stage 10) | You need evidence-based verification with traceable excerpts, or want hallucination detection | Multiple additional LLM calls (excerpt extraction + validation per attribute) |
| `deep_judgment_rubric_mode` | Deep-judgment evaluation in RubricEvaluation + DeepJudgmentRubricAutoFail (stage 12) | Same as above, for rubric traits | Additional LLM calls per trait with deep judgment enabled |

**Heuristics**:

- If your benchmark contains only factual questions with clear correct answers, `template_only` with no optional stages is the fastest and cheapest path.
- Enable `abstention_enabled` when you expect models to sometimes refuse. Without it, a refusal is parsed as a normal response, producing a misleading `verify_result=False`.
- Enable `sufficiency_enabled` when your template has many fields and you want to avoid burning parsing tokens on responses that clearly lack the information needed.
- Enable `deep_judgment_enabled` when you need an audit trail linking each parsed field back to verbatim text in the response.

## 8. Error Containment

The pipeline distinguishes three kinds of non-success:

| Kind | Set by | Effect on remaining stages | `completed_without_errors` |
|------|--------|---------------------------|:------------------------:|
| **Auto-fail** | Guard stages (3, 4, 10, 12) | Sets `verify_result=False`; other stages continue normally | `True` |
| **Override skip** | Pre-parse checks (5, 6) | Sets `verify_result=False`; downstream stages skip via `should_run()` conditions | `True` |
| **Error** | Any stage setting `context.error` or raising an exception | All subsequent stages except `FinalizeResult` skip via `should_run()` | `False` |

The critical distinction: **auto-fails are expected outcomes** (the model behaved in a detectable bad way), while **errors are infrastructure failures** (a stage threw an exception or encountered an unrecoverable state). Both produce a valid `VerificationResult`; the `completed_without_errors` field tells you which case applies.

### How `should_run()` Works

Every stage inherits a default `should_run()` that returns `False` when `context.error` is set. Stages override this with additional conditions:

| Stage | Runs when (in addition to no error) |
|-------|--------------------------------------|
| RecursionLimitAutoFail | `recursion_limit_reached` is `True` |
| TraceValidationAutoFail | `raw_llm_response` artifact exists |
| AbstentionCheck | `abstention_enabled` is `True` and `recursion_limit_reached` is `False` |
| SufficiencyCheck | `sufficiency_enabled` is `True`, no recursion limit, no trace validation failure, no abstention detected |
| ParseTemplate | No prior failures, sufficiency not detected as insufficient, has both response and Answer artifacts |
| EmbeddingCheck | `field_verification_result` is `False` |
| DeepJudgmentAutoFail | Deep judgment was performed and attributes are missing excerpts |
| RubricEvaluation | Rubric has traits; trace validation did not fail (or full trace is being used) |
| DeepJudgmentRubricAutoFail | Deep judgment rubric was performed and traits are missing excerpts |
| FinalizeResult | **Always** (overrides the error check) |

## 9. Worked Example: A Question Through the Pipeline

Consider a benchmark question in `template_and_rubric` mode with `abstention_enabled=True`:

> **Question**: "What is the putative target of venetoclax?"

**Stage 1: ValidateTemplate** compiles the answer template code, produces a validated `Answer` class with a `target` field.

**Stage 2: GenerateAnswer** sends the question to the answering model. The model responds: "Venetoclax targets BCL2 (B-cell lymphoma 2), a key anti-apoptotic protein."

**Stage 3: RecursionLimitAutoFail** checks `recursion_limit_reached`. It is `False` (no agent loop). Stage skips itself.

**Stage 4: TraceValidationAutoFail** checks if MCP is enabled. It is not. Stage skips the validation (non-MCP responses are always valid).

**Stage 5: AbstentionCheck** asks the parsing model: "Did this response refuse to answer?" The model says no. `abstention_detected=False`. Pipeline continues.

**Stage 7: ParseTemplate** sends the response and the `Answer` schema to the Judge LLM. The Judge extracts: `{"target": "BCL2"}`. A filled `Answer` instance is stored as `parsed_answer`.

**Stage 8: VerifyTemplate** calls `answer.verify()`, which compares `"BCL2"` against the ground truth `"BCL2"`. Field verification passes. Regex verification (if defined) also runs. `verify_result=True`.

**Stage 9: EmbeddingCheck** checks `field_verification_result`. It is `True`. Stage skips itself (embedding fallback is unnecessary).

**Stage 11: RubricEvaluation** evaluates the attached traits. A `RegexTrait` for `has_citations` checks for `\[\d+\]` in the response. No match: `has_citations=False`. An `LLMRubricTrait` for `conciseness` asks the parsing model to judge. Result: `True`.

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

## 10. Next Steps

- [Evaluation Modes](../notebooks/core_concepts/evaluation-modes.ipynb): how the three modes shape which stages run
- [Prompt Assembly](prompt-assembly.md): how prompts are constructed for the Judge LLM and rubric evaluators
- [Results and Scoring](results-and-scoring.md): what the pipeline produces and how to read it
- [Answer Templates](../notebooks/core_concepts/answer-templates.ipynb): writing the `verify()` logic that stage 8 executes
- [Rubrics](rubrics/index.md): defining the traits that stage 11 evaluates
- [Pipeline Internals](../advanced-pipeline/index.md): deep dive into each stage, deep judgment, and custom stages
