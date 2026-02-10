# Verification Pipeline

The verification pipeline is the execution engine that evaluates LLM responses. It runs a sequence of **13 stages** in a fixed order, each performing a specific step — from template validation through answer generation, parsing, verification, rubric evaluation, and result finalization.

## Why a Pipeline?

Structuring evaluation as an ordered pipeline provides:

- **Consistency** — Every question goes through the same stages in the same order
- **Auditability** — Each stage records its outcome, making it clear where failures occur
- **Configurability** — Optional stages can be enabled or disabled without affecting the rest
- **Error containment** — Auto-fail stages catch problems early; `FinalizeResult` always runs to produce a result

## The 13 Stages at a Glance

```
 1. ValidateTemplate          ─┐
                               │  Setup
 2. GenerateAnswer            ─┤  Core generation
                               │
 3. RecursionLimitAutoFail    ─┤
 4. TraceValidationAutoFail   ─┤  Guards
                               │
 5. AbstentionCheck           ─┤
 6. SufficiencyCheck          ─┤  Pre-parse checks (optional)
                               │
 7. ParseTemplate             ─┤
 8. VerifyTemplate            ─┤  Template processing
                               │
 9. EmbeddingCheck            ─┤
10. DeepJudgmentAutoFail      ─┤  Enhancements (optional)
                               │
11. RubricEvaluation          ─┤
12. DeepJudgmentRubric        ─┤  Rubric evaluation (optional)
                               │
13. FinalizeResult            ─┘  Finalization (always last)
```

## Stage Categories

### Setup

**ValidateTemplate** (stage 1) — Always runs first. Validates the template code, compiles it, and prepares the `Answer` class for later use. If the template is malformed (syntax error, missing `Answer` class, invalid fields), verification stops here with an error.

### Core Generation

**GenerateAnswer** (stage 2) — Sends the question to the answering model (LLM or agent) and captures the response. For MCP-enabled models, this runs the full agent loop with tool calls. The response trace is stored for all subsequent stages.

### Guards

**RecursionLimitAutoFail** (stage 3) — Auto-fails if the agent hit its recursion or tool-call limit during generation. This indicates the model got stuck in a loop rather than producing a proper answer.

**TraceValidationAutoFail** (stage 4) — Auto-fails if the response trace does not end with an AI message. This catches cases where tool calls were the last action (the model never produced a final answer).

### Pre-Parse Checks (Optional)

**AbstentionCheck** (stage 5) — Detects when the model refuses to answer (e.g., "I cannot answer that question"). When abstention is detected, parsing is skipped and the result is marked as abstained. Enabled via `abstention_enabled` in `VerificationConfig`.

**SufficiencyCheck** (stage 6) — Evaluates whether the response contains enough information to fill the template. If insufficient, parsing is skipped. Enabled via `sufficiency_enabled` in `VerificationConfig`.

### Template Processing

**ParseTemplate** (stage 7) — Sends the response to the parsing model (Judge LLM) along with the template's JSON schema. The Judge LLM extracts structured fields from the free-text response, producing a filled `Answer` instance.

**VerifyTemplate** (stage 8) — Runs the template's `verify()` method to compare extracted values against ground truth. Also runs `verify_granular()` if implemented, producing per-field results. The outcome is a boolean pass/fail.

### Enhancements (Optional)

**EmbeddingCheck** (stage 9) — Computes semantic similarity between the raw response and the expected answer using a SentenceTransformer model. Provides a secondary signal when string-based verification is too strict or too lenient. Enabled via `embedding_check_enabled`.

**DeepJudgmentAutoFail** (stage 10) — Runs deep verification on template results by extracting verbatim excerpts from the response and validating them against the parsed fields. Provides evidence-based verification with hallucination detection. Enabled via `deep_judgment_enabled`.

### Rubric Evaluation (Optional)

**RubricEvaluation** (stage 11) — Evaluates all applicable rubric traits (global + question-specific) on the raw response trace. Each trait type runs its own evaluator: LLM traits use the parsing model, regex traits use pattern matching, callable traits execute Python functions, and metric traits compute confusion matrices. Present in `template_and_rubric` and `rubric_only` modes.

**DeepJudgmentRubric** (stage 12) — Runs deep judgment on rubric trait results, extracting supporting excerpts for LLM trait assessments. Provides evidence-based rubric evaluation. Enabled via `deep_judgment_rubric_mode`.

### Finalization

**FinalizeResult** (stage 13) — Always runs last, even if earlier stages failed. Assembles the `VerificationResult` object from all stage outcomes, captures timing and usage metadata, and stores the result.

## How Evaluation Mode Shapes the Pipeline

The [evaluation mode](evaluation-modes.md) controls which stages are active:

| Stage | `template_only` | `template_and_rubric` | `rubric_only` |
|-------|:---------------:|:---------------------:|:-------------:|
| ValidateTemplate | Yes | Yes | No |
| GenerateAnswer | Yes | Yes | Yes |
| RecursionLimitAutoFail | Yes | Yes | Yes |
| TraceValidationAutoFail | Yes | Yes | Yes |
| AbstentionCheck | Optional | Optional | Optional |
| SufficiencyCheck | Optional | Optional | No |
| ParseTemplate | Yes | Yes | No |
| VerifyTemplate | Yes | Yes | No |
| EmbeddingCheck | Optional | Optional | No |
| DeepJudgmentAutoFail | Optional | Optional | No |
| RubricEvaluation | No | Yes | Yes |
| DeepJudgmentRubric | No | Optional | Optional |
| FinalizeResult | Yes | Yes | Yes |

In `rubric_only` mode, template-related stages are skipped entirely — the pipeline goes from generation and guards directly to rubric evaluation.

## Error Containment

The pipeline is designed to always produce a result, even when things go wrong:

- **Auto-fail stages** (stages 3, 4, 10, 12) set `verify_result=False` and record the reason, but do not crash the pipeline
- **Stage errors** are caught and recorded in the result metadata (`completed_without_errors=False`, `error` field)
- **FinalizeResult always runs** — it assembles whatever data is available into a valid `VerificationResult`

This means you always get a result object back, even for questions where the model failed, abstained, or hit errors. The result fields tell you exactly what happened.

## Next Steps

- [Evaluation Modes](evaluation-modes.md) — How the three modes shape the pipeline
- [Prompt Assembly](prompt-assembly.md) — How prompts are constructed for pipeline LLM calls
- [Results and Scoring](results-and-scoring.md) — What the pipeline produces
- [Pipeline Internals](../11-advanced-pipeline/index.md) — Deep dive into each stage, deep judgment, and custom stages
