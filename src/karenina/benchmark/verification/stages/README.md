# Verification Pipeline Stages

This package contains the modular stage-based verification pipeline for evaluating LLM responses against answer templates and rubrics.

## Package Structure

```
stages/
├── __init__.py                    # Public API exports
├── base.py                        # Core types: VerificationContext, VerificationStage, ArtifactKeys
├── orchestrator.py                # StageOrchestrator: pipeline execution
├── check_stage_base.py            # BaseCheckStage: abstention/sufficiency base class
├── autofail_stage_base.py         # BaseAutoFailStage: auto-fail stages base class
├── deep_judgment_helpers.py       # Deep judgment config resolution utilities
├── results_exporter.py            # Export to JSON/CSV
└── <13 stage implementations>
```

## Base Classes

### `BaseVerificationStage` (base.py)

Abstract base class that all stages inherit from. Provides:

- **`name`** (abstract property): Human-readable stage identifier
- **`requires`** (property): List of artifact keys this stage needs
- **`produces`** (property): List of NEW artifact keys this stage creates
- **`should_run(context)`**: Whether to execute (default: skip if `context.error`)
- **`execute(context)`** (abstract): The stage's main logic

**Helper methods** available to all stages:
- `get_or_create_usage_tracker(context)` - Retrieve or create UsageTracker
- `set_artifact_and_result(context, key, value)` - Set both artifact and result field

### `BaseCheckStage` (check_stage_base.py)

Specialized base for detection stages (abstention, sufficiency). Subclasses implement:

- **`_artifact_prefix`**: Returns `"abstention"` or `"sufficiency"`
- **`_should_trigger_override(detected, check_performed)`**: When to set verify_result=False
- **`_detect(context)`**: Detection logic returning `(detected, performed, reasoning, usage)`

### `BaseAutoFailStage` (autofail_stage_base.py)

Specialized base for auto-fail stages. Subclasses implement:

- **`_should_skip_due_to_prior_failure(context)`**: Skip if already failed (e.g., abstention)
- **`_get_autofail_reason(context)`**: Human-readable failure reason
- **`_set_additional_failure_fields(context)`** (optional): Stage-specific fields

## Pipeline Execution Order

The 13-stage pipeline executes in this order (stages 5-12 are optional based on config):

| # | Stage | Description | Optional |
|---|-------|-------------|----------|
| 1 | **ValidateTemplateStage** | Validates template syntax/attributes | No |
| 2 | **GenerateAnswerStage** | LLM generates response | No |
| 3 | **RecursionLimitAutoFailStage** | Auto-fail if recursion limit hit | No |
| 4 | **TraceValidationAutoFailStage** | Auto-fail if trace doesn't end with AI | No |
| 5 | **AbstentionCheckStage** | Detect model refusals | Yes |
| 6 | **SufficiencyCheckStage** | Detect insufficient responses | Yes |
| 7 | **ParseTemplateStage** | Parse response into Pydantic schema | No |
| 8 | **VerifyTemplateStage** | Run verify() method | No |
| 9 | **EmbeddingCheckStage** | Semantic similarity fallback | Yes |
| 10 | **DeepJudgmentAutoFailStage** | Excerpt validation for templates | Yes |
| 11 | **RubricEvaluationStage** | Evaluate rubric traits | Yes |
| 12 | **DeepJudgmentRubricAutoFailStage** | Excerpt validation for rubrics | Yes |
| 13 | **FinalizeResultStage** | Build VerificationResult | No |

## ArtifactKeys (Type-Safe Constants)

Use `ArtifactKeys` constants instead of magic strings when accessing artifacts:

```python
from .base import ArtifactKeys

# Instead of: context.get_artifact("raw_llm_response")
raw = context.get_artifact(ArtifactKeys.RAW_LLM_RESPONSE)

# Instead of: context.set_artifact("verify_result", True)
context.set_artifact(ArtifactKeys.VERIFY_RESULT, True)
```

Key constant groups:
- **Core Pipeline**: `RAW_LLM_RESPONSE`, `PARSED_ANSWER`, `USAGE_TRACKER`
- **Verification**: `VERIFY_RESULT`, `FIELD_VERIFICATION_RESULT`, `VERIFY_GRANULAR_RESULT`
- **Auto-Fail**: `RECURSION_LIMIT_REACHED`, `TRACE_VALIDATION_FAILED`
- **Check Stages**: `ABSTENTION_*`, `SUFFICIENCY_*`
- **Embedding**: `EMBEDDING_CHECK_PERFORMED`, `EMBEDDING_SIMILARITY_SCORE`
- **Deep Judgment**: `DEEP_JUDGMENT_*`, `EXTRACTED_EXCERPTS`
- **Rubric**: `RUBRIC_RESULT`, `LLM_TRAIT_LABELS`, `METRIC_*`

## Adding a New Stage

1. **Create stage file** in `stages/` directory
2. **Inherit from appropriate base class**:
   - `BaseVerificationStage` for general stages
   - `BaseCheckStage` for detection/override stages
   - `BaseAutoFailStage` for auto-fail stages
3. **Implement required methods**:
   ```python
   from .base import BaseVerificationStage, VerificationContext, ArtifactKeys

   class MyNewStage(BaseVerificationStage):
       @property
       def name(self) -> str:
           return "MyNewStage"

       @property
       def requires(self) -> list[str]:
           return [ArtifactKeys.RAW_LLM_RESPONSE]

       @property
       def produces(self) -> list[str]:
           return ["my_new_artifact"]  # Only NEW artifacts

       def should_run(self, context: VerificationContext) -> bool:
           if not super().should_run(context):  # Check for errors
               return False
           return context.my_feature_enabled  # Custom condition

       def execute(self, context: VerificationContext) -> None:
           raw = context.get_artifact(ArtifactKeys.RAW_LLM_RESPONSE)
           # ... stage logic ...
           context.set_artifact("my_new_artifact", result)
   ```
4. **Export from `__init__.py`**
5. **Add to `StageOrchestrator.from_config()`** in orchestrator.py

## Logging Convention

All stages follow this logging level convention:

| Level | Use For |
|-------|---------|
| **ERROR** | System/code errors (exceptions, unexpected states) |
| **WARNING** | Auto-fails and overrides (anything changing verify_result) |
| **INFO** | Important flow events (stage completion, feature enabled) |
| **DEBUG** | Normal flow decisions (validation passed, skipping stages) |

## VerificationContext

The shared state object that flows through all stages:

```python
context = VerificationContext(
    question_id="q1",
    template_id="abc123",
    question_text="What is X?",
    template_code="class Answer(BaseAnswer): ...",
    answering_model=ModelConfig(...),
    parsing_model=ModelConfig(...),
    # Optional features
    abstention_enabled=True,
    deep_judgment_enabled=True,
    # ...
)
```

**Key methods**:
- `set_artifact(key, value)` / `get_artifact(key)` - Inter-stage communication
- `set_result_field(key, value)` / `get_result_field(key)` - Final result fields
- `mark_error(message)` - Stop pipeline with error

## Evaluation Modes

The orchestrator supports three modes via `StageOrchestrator.from_config()`:

1. **template_only** (default): Full template verification pipeline
2. **template_and_rubric**: Template verification + rubric evaluation
3. **rubric_only**: Skip template stages, only evaluate rubrics
