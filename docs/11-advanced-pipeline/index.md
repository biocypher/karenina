# Advanced: Verification Pipeline

This section covers the internals of karenina's verification pipeline. Understanding these details is useful when you need to debug unexpected results, customize pipeline behavior, or extend the system with new stages.

## When You Need This

Most users can work entirely with the interfaces described in [Running Verification](../06-running-verification/index.md) — load a benchmark, configure, run, and inspect results. The advanced pipeline documentation is for situations where you need to:

- **Debug failures**: Understand why a specific stage failed or was skipped
- **Tune deep judgment**: Configure excerpt extraction, fuzzy matching, and search-enhanced verification
- **Customize prompts**: Understand how the tri-section prompt assembly system works
- **Extend the pipeline**: Write custom verification stages

## The 13-Stage Pipeline

Every verification run executes a subset of 13 stages in a fixed order. The `StageOrchestrator` builds the stage list based on [evaluation mode](../04-core-concepts/evaluation-modes.md) and feature flags.

```
 ┌─────────────────────────────────────────────────────────────┐
 │  1. ValidateTemplate         [always*]     Setup            │
 │  2. GenerateAnswer            [always]     LLM Call         │
 │  3. RecursionLimitAutoFail    [always]     Guard            │
 │  4. TraceValidationAutoFail   [always]     Guard            │
 │  5. AbstentionCheck          [optional]    Pre-Parse Check  │
 │  6. SufficiencyCheck         [optional]    Pre-Parse Check  │
 │  7. ParseTemplate            [always*]     LLM Call         │
 │  8. VerifyTemplate           [always*]     Verification     │
 │  9. EmbeddingCheck           [always*]     Enhancement      │
 │ 10. DeepJudgmentAutoFail     [optional]    Enhancement      │
 │ 11. RubricEvaluation         [optional]    Evaluation       │
 │ 12. DeepJudgmentRubric       [optional]    Enhancement      │
 │ 13. FinalizeResult            [always]     Finalization     │
 └─────────────────────────────────────────────────────────────┘

 * Skipped in rubric_only mode
```

### Stage Categories

| Category | Stages | Purpose |
|----------|--------|---------|
| Setup | ValidateTemplate | Validate template code before execution |
| LLM Calls | GenerateAnswer, ParseTemplate | Call answering and parsing LLMs |
| Guards | RecursionLimitAutoFail, TraceValidationAutoFail | Auto-fail on structural problems |
| Pre-Parse Checks | AbstentionCheck, SufficiencyCheck | Skip parsing when unnecessary |
| Verification | VerifyTemplate | Run the template's `verify()` method |
| Enhancements | EmbeddingCheck, DeepJudgmentAutoFail, DeepJudgmentRubric | Optional verification refinements |
| Evaluation | RubricEvaluation | Evaluate rubric traits on the raw trace |
| Finalization | FinalizeResult | Build the `VerificationResult` object |

### What Each Stage Does

| # | Stage | What It Does | Controlled By |
|---|-------|-------------|---------------|
| 1 | ValidateTemplate | Compiles template code, validates `Answer` class | Always runs (template modes) |
| 2 | GenerateAnswer | Sends question to answering LLM, captures trace | Always runs |
| 3 | RecursionLimitAutoFail | Auto-fails if agent hit recursion limit | Always runs |
| 4 | TraceValidationAutoFail | Auto-fails if trace doesn't end with AI message | Always runs |
| 5 | AbstentionCheck | Detects model refusal/abstention, skips parsing | `abstention_enabled` |
| 6 | SufficiencyCheck | Detects insufficient responses, skips parsing | `sufficiency_enabled` |
| 7 | ParseTemplate | Judge LLM parses response into template schema | Always runs (template modes) |
| 8 | VerifyTemplate | Runs `verify()` and `verify_granular()` | Always runs (template modes) |
| 9 | EmbeddingCheck | Compares embeddings if field verification failed | `embedding_check_enabled` + own logic |
| 10 | DeepJudgmentAutoFail | Excerpt extraction + fuzzy matching for templates | `deep_judgment_enabled` |
| 11 | RubricEvaluation | Evaluates LLM/regex/callable/metric traits | `template_and_rubric` or `rubric_only` mode |
| 12 | DeepJudgmentRubric | Deep judgment for rubric trait scores | Rubric traits with deep judgment config |
| 13 | FinalizeResult | Assembles `VerificationResult` from context | Always runs |

## How Stages Run by Evaluation Mode

The evaluation mode determines which stages are included in the pipeline:

| Stage | `template_only` | `template_and_rubric` | `rubric_only` |
|-------|:---:|:---:|:---:|
| ValidateTemplate | Yes | Yes | — |
| GenerateAnswer | Yes | Yes | Yes |
| RecursionLimitAutoFail | Yes | Yes | Yes |
| TraceValidationAutoFail | Yes | Yes | Yes |
| AbstentionCheck | If enabled | If enabled | If enabled |
| SufficiencyCheck | If enabled | If enabled | — |
| ParseTemplate | Yes | Yes | — |
| VerifyTemplate | Yes | Yes | — |
| EmbeddingCheck | Yes | Yes | — |
| DeepJudgmentAutoFail | If enabled | If enabled | — |
| RubricEvaluation | — | Yes | Yes |
| DeepJudgmentRubric | — | Yes | Yes |
| FinalizeResult | Yes | Yes | Yes |

## Execution Model

The `StageOrchestrator` executes stages sequentially:

1. **Build stage list** — `StageOrchestrator.from_config()` selects stages based on evaluation mode, feature flags, and rubric presence
2. **Validate dependencies** — Each stage declares what it `requires` and `produces`; the orchestrator checks that dependencies can be satisfied
3. **Execute in order** — Each stage's `should_run()` is called first; if it returns `True`, `execute()` runs
4. **Handle errors gracefully** — If a stage sets an error on the context, remaining stages are skipped (except `FinalizeResult`, which always runs)
5. **Build result** — `FinalizeResult` assembles the `VerificationResult` from all accumulated context artifacts

### Error Containment

Errors are contained per-question. If one question's pipeline fails, other questions continue independently. The `FinalizeResult` stage always executes, ensuring every question produces a `VerificationResult` — even if it records `completed_without_errors=False`.

### Conditional Execution

Each stage implements `should_run(context)` to decide at runtime whether to execute. This is separate from the stage list inclusion — a stage can be in the list but skip execution based on runtime state. For example:

- **EmbeddingCheck** only runs if field verification failed (to provide a second opinion)
- **AbstentionCheck** skips if an error was already set
- **FinalizeResult** always runs regardless of error state

## Section Contents

| Page | What It Covers |
|------|---------------|
| [13 Stages in Detail](stages.md) | Each stage's purpose, conditions, behavior, and configuration |
| [Deep Judgment: Templates](deep-judgment-templates.md) | Excerpt extraction, fuzzy matching, retry logic, search-enhanced verification |
| [Deep Judgment: Rubrics](deep-judgment-rubrics.md) | Per-trait deep judgment configuration and modes |
| [Prompt Assembly System](prompt-assembly.md) | Tri-section prompt pattern, `PromptAssembler`, `AdapterInstructionRegistry` |
| [Custom Stages](custom-stages.md) | `BaseVerificationStage` interface, writing and registering new stages |

## Related

- [Running Verification](../06-running-verification/index.md) — User-facing verification workflow
- [VerificationConfig Tutorial](../06-running-verification/verification-config.md) — Configuring pipeline features
- [Evaluation Modes](../04-core-concepts/evaluation-modes.md) — How modes affect stage selection
- [VerificationResult Structure](../07-analyzing-results/verification-result.md) — What the pipeline produces
- [VerificationConfig Reference](../10-configuration-reference/verification-config.md) — All 33 configuration fields
