# Writing Custom Stages

This page explains how to extend the verification pipeline with custom stages. Custom stages let you add domain-specific checks, additional evaluations, or integration with external systems — all within the standard pipeline execution flow.

For the built-in stage reference, see [13 Stages in Detail](stages.md).

## The Stage Interface

Every stage in the pipeline implements the `VerificationStage` protocol, defined in `karenina.benchmark.verification.stages.core.base`. The protocol uses structural typing (duck typing), so your class does not need to inherit from anything — it just needs to implement these members:

| Member | Type | Description |
|--------|------|-------------|
| `name` | property → `str` | Human-readable stage name (e.g., `"ToxicityCheck"`) |
| `requires` | property → `list[str]` | Artifact keys this stage reads from the context |
| `produces` | property → `list[str]` | Artifact keys this stage creates (not modifies) |
| `should_run()` | method → `bool` | Whether the stage should execute for this context |
| `execute()` | method → `None` | The stage's main logic, modifying the context in-place |

### Minimal Example

```python
from karenina.benchmark.verification.stages.core.base import (
    ArtifactKeys,
    BaseVerificationStage,
    VerificationContext,
)


class WordCountCheckStage(BaseVerificationStage):
    """Fail verification if the response is too short."""

    def __init__(self, min_words: int = 10):
        self.min_words = min_words

    @property
    def name(self) -> str:
        return "WordCountCheck"

    @property
    def requires(self) -> list[str]:
        return [ArtifactKeys.RAW_LLM_RESPONSE]

    @property
    def produces(self) -> list[str]:
        return ["word_count", "word_count_passed"]

    def should_run(self, context: VerificationContext) -> bool:
        if not super().should_run(context):
            return False
        return context.has_artifact(ArtifactKeys.RAW_LLM_RESPONSE)

    def execute(self, context: VerificationContext) -> None:
        response = context.get_artifact(ArtifactKeys.RAW_LLM_RESPONSE)
        word_count = len(response.split())
        passed = word_count >= self.min_words

        context.set_artifact("word_count", word_count)
        context.set_artifact("word_count_passed", passed)
        context.set_result_field("word_count", word_count)
        context.set_result_field("word_count_passed", passed)
```

## BaseVerificationStage

While the protocol allows pure duck typing, inheriting from `BaseVerificationStage` provides useful defaults:

- **`requires`** and **`produces`** default to `[]` — override only when needed
- **`should_run()`** returns `False` when `context.error` is set, preventing execution after fatal errors
- **`get_or_create_usage_tracker()`** retrieves or creates a `UsageTracker` for token tracking
- **`set_artifact_and_result()`** sets the same key/value in both artifacts and the result builder

Always call `super().should_run(context)` first when overriding `should_run()` to inherit the error-checking behavior.

## VerificationContext

The context is the shared state object passed through all stages. Stages read from and write to the context — they do not return values.

### Key Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `question_id` | `str` | Current question identifier |
| `template_id` | `str` | Current template identifier |
| `question_text` | `str` | The question text |
| `template_code` | `str` | Python template source code |
| `answering_model` | `ModelConfig` | Model generating the answer |
| `parsing_model` | `ModelConfig` | Model parsing the answer |
| `rubric` | `Rubric \| None` | Rubric for evaluation (if any) |
| `error` | `str \| None` | Error message (halts subsequent stages) |

### Key Methods

| Method | Description |
|--------|-------------|
| `set_artifact(key, value)` | Store an artifact for downstream stages |
| `get_artifact(key, default=None)` | Retrieve an artifact set by a prior stage |
| `has_artifact(key)` | Check if an artifact exists |
| `set_result_field(key, value)` | Store a value in the result builder (included in final `VerificationResult`) |
| `get_result_field(key, default=None)` | Retrieve a result field |
| `mark_error(message)` | Mark context as failed — subsequent stages skip (except `FinalizeResult`) |

### Artifacts vs Result Fields

- **Artifacts** are working data shared between stages during pipeline execution. They are not directly included in the final result.
- **Result fields** are accumulated by the result builder and used by `FinalizeResult` to construct the `VerificationResult`.

Many stages set both: an artifact for downstream stage access, and a result field for inclusion in the output. Use `set_artifact_and_result()` to do both in one call.

## ArtifactKeys

Use the `ArtifactKeys` constants instead of raw strings to reference artifacts. Key groups:

**Core pipeline:**

| Key | Description |
|-----|-------------|
| `RAW_LLM_RESPONSE` | Raw response text from the answering model |
| `PARSED_ANSWER` | Parsed Pydantic object from the judge model |
| `ANSWER` | Answer class with question ID injected |
| `RAW_ANSWER` | Answer class before question ID injection |
| `USAGE_TRACKER` | Token usage tracker |
| `VERIFY_RESULT` | Boolean verification outcome |
| `VERIFY_GRANULAR_RESULT` | Per-field verification results dict |

**Detection results:**

| Key | Description |
|-----|-------------|
| `ABSTENTION_DETECTED` | Whether abstention was detected |
| `SUFFICIENCY_DETECTED` | Whether the response was sufficient |
| `RECURSION_LIMIT_REACHED` | Whether the agent hit its recursion limit |
| `TRACE_VALIDATION_FAILED` | Whether trace validation failed |

**Rubric:**

| Key | Description |
|-----|-------------|
| `VERIFY_RUBRIC` | Rubric scores dict: `{trait_name: score}` |
| `LLM_TRAIT_LABELS` | Literal trait labels: `{trait_name: class_name}` |
| `METRIC_TRAIT_METRICS` | Metric trait metrics: `{trait_name: metrics_dict}` |

**Deep judgment:**

| Key | Description |
|-----|-------------|
| `DEEP_JUDGMENT_PERFORMED` | Whether deep judgment ran |
| `EXTRACTED_EXCERPTS` | Per-attribute excerpts: `{attr: [excerpt, ...]}` |
| `ATTRIBUTE_REASONING` | Per-attribute reasoning: `{attr: reasoning_text}` |

## Specialized Base Classes

For common patterns, karenina provides two additional base classes.

### BaseCheckStage

For stages that detect a condition and optionally override `verify_result`. Located at `stages.core.check_stage_base`.

Override these:

| Method | Purpose |
|--------|---------|
| `_artifact_prefix` (property) | Prefix for auto-generated artifact keys (e.g., `"toxicity"` creates `toxicity_check_performed`, `toxicity_detected`, etc.) |
| `_detect(context)` | Return `(detected, check_performed, reasoning, usage_metadata)` |
| `_should_trigger_override(detected, check_performed)` | Return `True` if `verify_result` should be set to `False` |

The base class handles artifact storage, result field updates, and usage tracking automatically.

### BaseAutoFailStage

For stages that set `verify_result = False` when a condition is met. Located at `stages.core.autofail_stage_base`.

Override these:

| Method | Purpose |
|--------|---------|
| `_should_skip_due_to_prior_failure(context)` | Return `True` to skip if a prior failure (like abstention) already set the result |
| `_get_autofail_reason(context)` | Return a human-readable reason for the auto-fail |
| `_set_additional_failure_fields(context)` | Optional: set stage-specific result fields |

Auto-fail stages produce no new artifacts (`produces = []`) — they only modify the existing `verify_result`.

## Registering Custom Stages

### Using StageRegistry

The `StageRegistry` validates stage dependencies before execution:

```python
from karenina.benchmark.verification.stages.core.orchestrator import (
    StageRegistry,
)

registry = StageRegistry()
registry.register(my_stage)

# Validate that all stage requirements are satisfiable
errors = registry.validate_dependencies([stage1, stage2, my_stage])
if errors:
    print("Dependency errors:", errors)
```

### Using StageOrchestrator

To insert a custom stage into a pipeline, build the orchestrator manually:

```python
from karenina.benchmark.verification.stages.core.orchestrator import (
    StageOrchestrator,
)

# Get the default stages for your evaluation mode
orchestrator = StageOrchestrator.from_config(
    rubric=rubric,
    evaluation_mode="template_and_rubric",
    abstention_enabled=True,
)

# Insert your stage at a specific position
stages = list(orchestrator.stages)
# Insert after VerifyTemplate (stage 8) and before EmbeddingCheck (stage 9)
for i, stage in enumerate(stages):
    if stage.name == "VerifyTemplate":
        stages.insert(i + 1, WordCountCheckStage(min_words=20))
        break

# Create a new orchestrator with the modified stage list
custom_orchestrator = StageOrchestrator(stages=stages)
```

### Stage Ordering Rules

When inserting custom stages, follow these guidelines:

1. **Declare `requires` accurately** — the orchestrator validates that required artifacts are produced by prior stages
2. **Place stages after their dependencies** — if your stage reads `RAW_LLM_RESPONSE`, it must come after `GenerateAnswer` (stage 2)
3. **Place stages before consumers** — if your stage produces artifacts that `FinalizeResult` should include, insert it before stage 13
4. **`FinalizeResult` must always be last** — it builds the final `VerificationResult` from all accumulated context

### Dependency Validation

The `requires` and `produces` properties enable automatic dependency validation:

```
Stage A produces: ["parsed_answer"]
Stage B requires: ["parsed_answer"]
Stage C requires: ["parsed_answer", "word_count"]

→ A must run before B
→ A and the stage producing "word_count" must run before C
```

If a stage's requirements cannot be met by preceding stages, the registry reports dependency errors. Custom artifact keys (like `"word_count"`) are valid — they just need to be produced by a prior stage.

## Complete Example: Toxicity Check Stage

This example shows a custom check-type stage that uses an external classifier:

```python
import logging

from karenina.benchmark.verification.stages.core.base import (
    ArtifactKeys,
    BaseVerificationStage,
    VerificationContext,
)

logger = logging.getLogger(__name__)


class ToxicityCheckStage(BaseVerificationStage):
    """Check response for toxic content and optionally fail verification."""

    def __init__(self, threshold: float = 0.8, fail_on_toxic: bool = True):
        self.threshold = threshold
        self.fail_on_toxic = fail_on_toxic

    @property
    def name(self) -> str:
        return "ToxicityCheck"

    @property
    def requires(self) -> list[str]:
        return [ArtifactKeys.RAW_LLM_RESPONSE]

    @property
    def produces(self) -> list[str]:
        return ["toxicity_score", "toxicity_passed"]

    def should_run(self, context: VerificationContext) -> bool:
        if not super().should_run(context):
            return False
        return context.has_artifact(ArtifactKeys.RAW_LLM_RESPONSE)

    def execute(self, context: VerificationContext) -> None:
        response = context.get_artifact(ArtifactKeys.RAW_LLM_RESPONSE)

        # Call your toxicity classifier
        score = self._classify(response)
        passed = score < self.threshold

        # Store artifacts for downstream stages
        context.set_artifact("toxicity_score", score)
        context.set_artifact("toxicity_passed", passed)

        # Store result fields for inclusion in VerificationResult
        context.set_result_field("toxicity_score", score)
        context.set_result_field("toxicity_passed", passed)

        if not passed and self.fail_on_toxic:
            # Override verify_result to fail
            context.set_artifact(ArtifactKeys.VERIFY_RESULT, False)
            context.set_result_field(ArtifactKeys.VERIFY_RESULT, False)
            logger.warning(
                "Toxicity check failed for %s (score=%.2f)",
                context.question_id,
                score,
            )

    def _classify(self, text: str) -> float:
        # Replace with your actual toxicity classifier
        toxic_words = {"harmful", "dangerous", "illegal"}
        words = set(text.lower().split())
        return len(words & toxic_words) / max(len(words), 1)
```

## Next Steps

- [13 Stages in Detail](stages.md) — reference for all built-in stages
- [Prompt Assembly](prompt-assembly.md) — how prompts are constructed for LLM-calling stages
- [Deep Judgment Templates](deep-judgment-templates.md) — deep verification internals
- [Adapter Architecture](../12-advanced-adapters/index.md) — the ports and adapters system used by LLM-calling stages
- [VerificationConfig Reference](../10-configuration-reference/verification-config.md) — all configuration fields
