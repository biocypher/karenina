# Evaluation Modes

Karenina's **evaluation mode** determines which building blocks --- templates, rubrics, or both --- are active during verification. The mode controls which pipeline stages run and what kind of results you get back.

## The Three Modes

| Mode | Templates | Rubrics | Default |
|---|---|---|---|
| `template_only` | Yes | No | Yes |
| `template_and_rubric` | Yes | Yes | No |
| `rubric_only` | No | Yes | No |

### `template_only` (default)

The default mode. The pipeline generates an answer, parses it into a structured template using a Judge LLM, and runs the `verify()` method to check correctness.

**Use when**: Questions have definitive correct answers and you want to measure factual accuracy.

**Pipeline stages**:

```
ValidateTemplate → GenerateAnswer → RecursionLimitAutoFail →
TraceValidationAutoFail → [AbstentionCheck] → [SufficiencyCheck] →
ParseTemplate → VerifyTemplate → [EmbeddingCheck] →
[DeepJudgmentAutoFail] → FinalizeResult
```

Stages in brackets are optional and controlled by their respective configuration flags.

### `template_and_rubric`

Runs the full template verification pipeline *and* evaluates rubric traits on the raw response. This gives you both correctness results (pass/fail from `verify()`) and quality assessments (trait scores from rubric evaluation).

**Use when**: You want to measure both *what* the model said (correctness) and *how* it said it (quality).

**Pipeline stages**:

```
ValidateTemplate → GenerateAnswer → RecursionLimitAutoFail →
TraceValidationAutoFail → [AbstentionCheck] → [SufficiencyCheck] →
ParseTemplate → VerifyTemplate → [EmbeddingCheck] →
[DeepJudgmentAutoFail] → RubricEvaluation →
[DeepJudgmentRubricAutoFail] → FinalizeResult
```

This mode adds `RubricEvaluation` (and optionally `DeepJudgmentRubricAutoFail`) after all template stages.

### `rubric_only`

Skips template verification entirely. The pipeline generates an answer and evaluates rubric traits directly on the raw LLM response, without parsing into a structured template.

**Use when**: Questions are open-ended with no single correct answer, and you only want to assess response quality (safety, clarity, tone, format compliance).

**Pipeline stages**:

```
GenerateAnswer → RecursionLimitAutoFail → TraceValidationAutoFail →
[AbstentionCheck] → RubricEvaluation → [DeepJudgmentRubricAutoFail] →
FinalizeResult
```

Note that `rubric_only` mode:

- Skips `ValidateTemplate`, `ParseTemplate`, `VerifyTemplate`, and `EmbeddingCheck`
- Does not include `SufficiencyCheck` (sufficiency is about whether the response has enough information to fill a template, which doesn't apply here)
- Still includes `RecursionLimitAutoFail` and `TraceValidationAutoFail` for safety

## Choosing a Mode

| Scenario | Recommended Mode |
|---|---|
| Factual QA with clear correct answers | `template_only` |
| Factual QA where you also want quality metrics | `template_and_rubric` |
| Open-ended questions (essays, summaries, advice) | `rubric_only` |
| Safety/compliance audits on free-form output | `rubric_only` |
| Benchmarks mixing factual and open-ended questions | `template_and_rubric` (questions without templates are skipped by template stages) |

## Configuration

Set the evaluation mode on `VerificationConfig`:

```python
from karenina.schemas import ModelConfig
from karenina.schemas.verification import VerificationConfig

answering = [ModelConfig(id="answerer", model_name="claude-haiku-4-5", model_provider="anthropic")]
parsing = [ModelConfig(id="parser", model_name="claude-haiku-4-5", model_provider="anthropic")]

# Template-only (default)
config = VerificationConfig(
    evaluation_mode="template_only",
    answering_models=answering,
    parsing_models=parsing,
)

# Template + rubric
config = VerificationConfig(
    evaluation_mode="template_and_rubric",
    rubric_enabled=True,
    answering_models=answering,
    parsing_models=parsing,
)

# Rubric-only
config = VerificationConfig(
    evaluation_mode="rubric_only",
    rubric_enabled=True,
    answering_models=answering,
    parsing_models=parsing,
)
```

The `evaluation_mode` and `rubric_enabled` fields must be consistent:

- `template_only` requires `rubric_enabled=False`
- `template_and_rubric` requires `rubric_enabled=True`
- `rubric_only` requires `rubric_enabled=True`

Setting an inconsistent combination raises a `ValueError` at configuration time.

!!! tip "Convenience method"
    When using `VerificationConfig.from_overrides()`, setting `evaluation_mode` automatically sets `rubric_enabled` to the correct value:

    ```python
    config = VerificationConfig.from_overrides(
        evaluation_mode="template_and_rubric",
        answering_model="claude-haiku-4-5",
        answering_provider="anthropic",
        parsing_model="claude-haiku-4-5",
        parsing_provider="anthropic",
        # rubric_enabled is set to True automatically
    )
    ```

## How Results Differ by Mode

| Mode | Template result | Rubric result |
|---|---|---|
| `template_only` | `verify_result` (bool), parsed responses, embedding check | Not included |
| `template_and_rubric` | `verify_result` (bool), parsed responses, embedding check | Per-trait scores (LLM, regex, callable, metric) |
| `rubric_only` | `verify_result` is `None` (template verification skipped) | Per-trait scores (LLM, regex, callable, metric) |

In `rubric_only` mode, `template_verification_performed` is `False` and `verify_result` is `None` since no template parsing occurs. The rubric trait scores in `rubric` are the primary output.

---

## Learn More

- [Template vs Rubric](template-vs-rubric.md) --- The fundamental distinction between correctness and quality evaluation
- [Answer Templates](answer-templates.md) --- How templates parse and verify answers
- [Rubrics Overview](rubrics/index.md) --- The four rubric trait types
- [VerificationConfig Reference](../10-configuration-reference/verification-config.md) --- All configuration fields including evaluation mode

**Back to**: [Core Concepts](index.md)
