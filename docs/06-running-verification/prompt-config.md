# PromptConfig

PromptConfig lets you inject custom instructions into specific LLM calls in the verification pipeline. Each field targets a pipeline stage — your text is appended to the system prompt for that stage's LLM call.

## Why Use PromptConfig?

By default, the pipeline sends carefully designed prompts to each LLM call. PromptConfig lets you customize these prompts without modifying pipeline internals:

- **Improve parsing accuracy** — add domain-specific guidance for the judge LLM
- **Tune rubric evaluation** — clarify scoring criteria or grading standards
- **Adjust quality checks** — refine what counts as an abstention or insufficient response
- **Guide deep judgment** — provide context for excerpt extraction or hallucination detection

## Injection Points

PromptConfig has 6 fields, each targeting a specific category of pipeline LLM calls:

| Field | Pipeline Stage | Purpose |
|-------|---------------|---------|
| `generation` | Answer generation | Instructions for the answering LLM |
| `parsing` | Template parsing | Instructions for the judge LLM parsing responses into templates |
| `abstention_detection` | Abstention check | Instructions for detecting model refusals |
| `sufficiency_detection` | Sufficiency check | Instructions for assessing response completeness |
| `rubric_evaluation` | Rubric evaluation | Fallback instructions for all rubric trait evaluation calls |
| `deep_judgment` | Deep judgment | Fallback instructions for all deep judgment calls |

All fields are `str | None` with a default of `None` (no custom instructions).

### Direct Fields vs Fallback Fields

The first four fields (`generation`, `parsing`, `abstention_detection`, `sufficiency_detection`) map directly to specific pipeline LLM calls — one field, one call type.

The last two fields (`rubric_evaluation`, `deep_judgment`) are **fallback fields** that cover multiple sub-tasks:

**`rubric_evaluation`** applies to:

- `rubric_llm_trait_batch` — batched boolean/score LLM trait evaluation
- `rubric_llm_trait_single` — sequential boolean/score LLM trait evaluation
- `rubric_literal_trait_batch` — batched literal (categorical) trait evaluation
- `rubric_literal_trait_single` — sequential literal trait evaluation
- `rubric_metric_trait` — metric trait confusion matrix extraction

**`deep_judgment`** applies to:

- `dj_template_excerpt_extraction` — extract verbatim excerpts per template attribute
- `dj_template_hallucination` — assess hallucination risk via search
- `dj_template_reasoning` — map excerpts to template attributes
- `dj_rubric_excerpt_extraction` — extract excerpts for rubric traits
- `dj_rubric_hallucination` — per-excerpt hallucination assessment
- `dj_rubric_reasoning` — rubric trait evaluation reasoning
- `dj_rubric_score_extraction` — extract final scores from reasoning

## Fallback Logic

When the pipeline resolves instructions for a task, it follows this order:

1. **Direct match** — look for a PromptConfig field matching the task name exactly
2. **Category fallback** — if no direct match and the task starts with `rubric_`, use `rubric_evaluation`; if it starts with `dj_`, use `deep_judgment`
3. **None** — no custom instructions applied

For example, if you set only `rubric_evaluation`:

```text
rubric_llm_trait_batch   → uses rubric_evaluation (fallback)
rubric_metric_trait      → uses rubric_evaluation (fallback)
parsing                  → None (no match, no fallback category)
```

## Usage

### Python API

Pass `prompt_config` when creating a `VerificationConfig`:

```python
from karenina.schemas.verification import VerificationConfig, PromptConfig

config = VerificationConfig(
    prompt_config=PromptConfig(
        parsing="Focus on extracting exact numerical values. "
                "If the response contains multiple numbers, choose the one "
                "most directly answering the question.",
        rubric_evaluation="Be strict in scoring. Only award full marks "
                          "when the response is comprehensive and well-sourced.",
    ),
    # ... other config fields
)
```

### In Presets

PromptConfig is serialized as a nested JSON object within presets:

```json
{
  "name": "strict-scoring",
  "config": {
    "prompt_config": {
      "parsing": "Focus on extracting exact numerical values.",
      "rubric_evaluation": "Be strict in scoring."
    }
  }
}
```

Only non-null fields need to be included — omitted fields default to `None`.

## Examples

### Improving Template Parsing

When the judge LLM struggles to parse domain-specific terminology:

```python
config = VerificationConfig(
    prompt_config=PromptConfig(
        parsing=(
            "This benchmark evaluates biomedical knowledge. "
            "Gene names may appear in various formats (e.g., BCL2, BCL-2, Bcl-2). "
            "Normalize gene names to their official HGNC symbols when extracting."
        ),
    ),
    # ... other config fields
)
```

### Tuning Rubric Evaluation

When rubric trait scores are too lenient or too strict:

```python
config = VerificationConfig(
    prompt_config=PromptConfig(
        rubric_evaluation=(
            "Apply strict grading standards. A response must explicitly "
            "address the criterion to receive a positive score — implicit "
            "or partial coverage should receive a negative score."
        ),
    ),
    # ... other config fields
)
```

### Guiding Deep Judgment

When deep judgment excerpt extraction misses relevant passages:

```python
config = VerificationConfig(
    prompt_config=PromptConfig(
        deep_judgment=(
            "When extracting excerpts, include surrounding context sentences. "
            "For numerical claims, always extract the sentence containing "
            "the number and any qualifying statements."
        ),
    ),
    # ... other config fields
)
```

## How Instructions Are Applied

PromptConfig is part of the **tri-section prompt assembly** pattern used throughout the pipeline:

```text
┌─────────────────────────────┐
│  1. Task instructions       │  ← Built-in prompts for each stage
├─────────────────────────────┤
│  2. Adapter instructions    │  ← Per-interface adjustments (e.g., Claude Tool)
├─────────────────────────────┤
│  3. User instructions       │  ← Your PromptConfig text (appended last)
└─────────────────────────────┘
```

Your custom text is **appended to the system prompt** after all built-in and adapter instructions. This means:

- You cannot override built-in instructions, only supplement them
- Your instructions are seen by the LLM as additional guidance
- The order ensures your instructions have the highest positional priority in the prompt

## Next Steps

- [VerificationConfig](verification-config.md) — all configuration options including prompt_config
- [Response Quality Checks](response-quality-checks.md) — abstention and sufficiency detection details
- [PromptConfig Reference](../reference/configuration/prompt-config.md) — complete field reference table
- [Prompt Assembly System](../11-advanced-pipeline/prompt-assembly.md) — how the tri-section pattern works internally
- [Using Presets](using-presets.md) — save prompt configurations in reusable presets
