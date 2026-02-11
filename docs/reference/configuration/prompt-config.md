# PromptConfig Reference

This is the exhaustive reference for all `PromptConfig` fields. For a tutorial introduction with examples, see [PromptConfig Tutorial](../../06-running-verification/prompt-config.md).

`PromptConfig` is a Pydantic model with **6 fields** that inject custom user instructions into specific pipeline LLM calls. All fields are optional strings defaulting to `None`.

**Import path:** `from karenina.schemas.verification import PromptConfig`

---

## Fields

| Field | Type | Default | Pipeline Stage | Description |
|-------|------|---------|---------------|-------------|
| `generation` | `str \| None` | `None` | Answer generation | Custom instructions for the answering LLM call. Appended to the system prompt when the answering model generates a response to the benchmark question. |
| `parsing` | `str \| None` | `None` | Template parsing | Custom instructions for the judge LLM parsing call. Appended to the system prompt when the parsing model extracts structured attributes from raw responses into the answer template schema. |
| `abstention_detection` | `str \| None` | `None` | Abstention check | Custom instructions for the abstention detection LLM call. Guides the model when determining whether the answering model refused to answer, evaded the question, or deflected. |
| `sufficiency_detection` | `str \| None` | `None` | Sufficiency check | Custom instructions for the sufficiency detection LLM call. Guides the model when assessing whether the response contains enough information to populate the answer template. |
| `rubric_evaluation` | `str \| None` | `None` | Rubric evaluation | **Fallback** instructions for all `rubric_*` sub-tasks (see [Covered Sub-Tasks](#rubric-evaluation-sub-tasks) below). Applied when no direct field matches the specific rubric task. |
| `deep_judgment` | `str \| None` | `None` | Deep judgment | **Fallback** instructions for all `dj_*` sub-tasks (see [Covered Sub-Tasks](#deep-judgment-sub-tasks) below). Applied when no direct field matches the specific deep judgment task. |

**Configuration:** `extra="forbid"` — PromptConfig rejects any fields not listed above.

---

## Direct vs Fallback Fields

The 6 fields fall into two categories:

**Direct fields** (4) — each maps to exactly one pipeline LLM call type:

- `generation` — answer generation
- `parsing` — template parsing
- `abstention_detection` — abstention check
- `sufficiency_detection` — sufficiency check

**Fallback fields** (2) — each covers multiple sub-tasks within a pipeline stage:

- `rubric_evaluation` — covers 5 rubric evaluation sub-tasks
- `deep_judgment` — covers 7 deep judgment sub-tasks

### Rubric Evaluation Sub-Tasks

The `rubric_evaluation` field serves as a fallback for these 5 `PromptTask` values:

| Sub-Task | Description |
|----------|-------------|
| `rubric_llm_trait_batch` | Evaluates all boolean/score LLM rubric traits in a single batched call |
| `rubric_llm_trait_single` | Evaluates a single boolean/score LLM rubric trait sequentially |
| `rubric_literal_trait_batch` | Evaluates all literal (categorical) rubric traits in a single batched call |
| `rubric_literal_trait_single` | Evaluates a single literal (categorical) rubric trait sequentially |
| `rubric_metric_trait` | Evaluates a metric rubric trait via confusion matrix extraction |

Which sub-tasks run depends on the `rubric_evaluation_strategy` setting in `VerificationConfig`:

- `"batch"` (default) — uses `*_batch` sub-tasks
- `"sequential"` — uses `*_single` sub-tasks

### Deep Judgment Sub-Tasks

The `deep_judgment` field serves as a fallback for these 7 `PromptTask` values:

| Sub-Task | Flow | Description |
|----------|------|-------------|
| `dj_template_excerpt_extraction` | Template | Extracts verbatim excerpts from the response per template attribute |
| `dj_template_hallucination` | Template | Assesses hallucination risk for extracted excerpts via web search |
| `dj_template_reasoning` | Template | Generates reasoning mapping excerpts to template attributes |
| `dj_rubric_excerpt_extraction` | Rubric | Extracts excerpts supporting deep-judgment-enabled rubric traits |
| `dj_rubric_hallucination` | Rubric | Assesses per-excerpt hallucination risk using search results |
| `dj_rubric_reasoning` | Rubric | Generates reasoning explaining trait evaluation based on excerpts |
| `dj_rubric_score_extraction` | Rubric | Extracts the final score from deep judgment reasoning |

---

## Resolution Logic

The `get_for_task(task_value)` method resolves instructions in this order:

1. **Direct match** — if a PromptConfig field name matches `task_value` exactly, return that field's value
2. **Category fallback** — if `task_value` starts with `rubric_`, return `rubric_evaluation`; if it starts with `dj_`, return `deep_judgment`
3. **None** — no custom instructions for this task

```text
get_for_task("parsing")                → self.parsing              (direct match)
get_for_task("rubric_llm_trait_batch") → self.rubric_evaluation    (fallback)
get_for_task("dj_rubric_reasoning")    → self.deep_judgment        (fallback)
get_for_task("unknown_task")           → None                      (no match)
```

---

## All PromptTask Values

The complete list of 17 `PromptTask` enum values recognized by the pipeline, grouped by the PromptConfig field that covers them:

| PromptConfig Field | PromptTask Values |
|--------------------|-------------------|
| `generation` | `generation` |
| `parsing` | `parsing` |
| `abstention_detection` | `abstention_detection` |
| `sufficiency_detection` | `sufficiency_detection` |
| `rubric_evaluation` | `rubric_llm_trait_batch`, `rubric_llm_trait_single`, `rubric_literal_trait_batch`, `rubric_literal_trait_single`, `rubric_metric_trait` |
| `deep_judgment` | `dj_template_excerpt_extraction`, `dj_template_hallucination`, `dj_template_reasoning`, `dj_rubric_excerpt_extraction`, `dj_rubric_hallucination`, `dj_rubric_reasoning`, `dj_rubric_score_extraction` |

**Source:** `karenina.benchmark.verification.prompts.task_types.PromptTask`

---

## How Instructions Are Applied

PromptConfig participates in the **tri-section prompt assembly** pattern. When the pipeline builds a prompt for any LLM call:

```text
┌─────────────────────────────┐
│  1. Task instructions       │  ← Built-in prompts for each stage
├─────────────────────────────┤
│  2. Adapter instructions    │  ← Per-interface adjustments (registered per adapter)
├─────────────────────────────┤
│  3. User instructions       │  ← Your PromptConfig text (appended last)
└─────────────────────────────┘
```

Your custom text is **appended to the system prompt** after all built-in and adapter instructions:

- You cannot override built-in instructions, only supplement them
- Your instructions are seen by the LLM as additional guidance
- The order gives your instructions the highest positional priority in the prompt

See [Prompt Assembly System](../../11-advanced-pipeline/prompt-assembly.md) for implementation details.

---

## Usage in VerificationConfig

PromptConfig is passed as a field on `VerificationConfig`:

```python
from karenina.schemas.verification import VerificationConfig, PromptConfig

config = VerificationConfig(
    prompt_config=PromptConfig(
        parsing="Normalize gene names to official HGNC symbols.",
        rubric_evaluation="Apply strict grading standards.",
    ),
    # ... other config fields
)
```

### In Presets

PromptConfig serializes as a nested JSON object within the preset `"config"` key:

```json
{
  "name": "strict-scoring",
  "config": {
    "prompt_config": {
      "parsing": "Normalize gene names to official HGNC symbols.",
      "rubric_evaluation": "Apply strict grading standards."
    }
  }
}
```

Only non-null fields need to be included — omitted fields default to `None`.

---

## See Also

- [PromptConfig Tutorial](../../06-running-verification/prompt-config.md) — when to use each field, examples, fallback logic explained
- [VerificationConfig Reference](verification-config.md) — the `prompt_config` field on VerificationConfig
- [Prompt Assembly System](../../11-advanced-pipeline/prompt-assembly.md) — how the tri-section pattern works internally
- [Response Quality Checks](../../06-running-verification/response-quality-checks.md) — abstention and sufficiency detection details
