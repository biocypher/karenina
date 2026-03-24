# Dead Code and Design Decisions

**Date**: 2026-03-24
**Scope**: 4 issues involving dead code, silent failures, and misplaced semantics

## Overview

This spec addresses dead code paths, silent overrides, and unclear semantics across four areas: PromptConfig generation, parsing system prompt auto-assignment, FewShotConfig structure, and runner evaluation mode auto-upgrade. Issue 099 (template_converter.py relocation) was already resolved in a prior commit.

## 1. Wire In PromptConfig.generation

**File**: `src/karenina/benchmark/verification/stages/pipeline/generate_answer.py`

### Problem

`PromptConfig` defines a `generation` field for user instructions to the generation stage, but `GenerateAnswerStage.execute()` never reads it. The parsing and rubric stages honor PromptConfig via `PromptAssembler`, but generation constructs messages manually using only `ModelConfig.system_prompt`.

### Change

In `GenerateAnswerStage.execute()`, replace the existing system prompt block (around line 271, before conversation_history injection) with logic that merges `ModelConfig.system_prompt` and `PromptConfig.generation`:

```python
# Replace the existing block:
#   if answering_model.system_prompt:
#       adapter_messages.append(Message.system(answering_model.system_prompt))
# With:
system_parts = []
if answering_model.system_prompt:
    system_parts.append(answering_model.system_prompt)
if context.prompt_config:
    gen_instructions = context.prompt_config.get_for_task("generation")
    if gen_instructions:
        system_parts.append(gen_instructions)
if system_parts:
    adapter_messages.append(Message.system("\n\n".join(system_parts)))
```

The cached answer path (early return at line 202) is unaffected; generation instructions are only relevant for live LLM calls.

This keeps `ModelConfig.system_prompt` as the base and layers `PromptConfig.generation` on top, consistent with how other stages layer PromptConfig user instructions.

### Test Plan

- Test that `PromptConfig(generation="Focus on accuracy")` results in the instruction appearing in the system message sent to the adapter.
- Test that when `PromptConfig.generation` is None, behavior is unchanged (only ModelConfig.system_prompt in system message).
- Test that both `ModelConfig.system_prompt` and `PromptConfig.generation` are joined with double newline when both are set.
- Test that when neither is set, no system message is added.

## 2. Remove Dead Parsing System Prompt + Add Instruction Shortcuts

**Files**: `src/karenina/schemas/verification/config.py`, `src/karenina/schemas/verification/prompt_config.py`

### Problem

`VerificationConfig.__init__` auto-assigns `DEFAULT_PARSING_SYSTEM_PROMPT` to parsing models with `system_prompt=None`. The parsing pipeline never reads `ModelConfig.system_prompt`: `TemplateEvaluator` builds its own system prompt via `TemplatePromptBuilder.build_system_prompt()`. The auto-assignment exists only to pass a non-empty validation check.

User customization of parsing already works via `PromptConfig(parsing="...")`, which is the user instruction layer in the 3-layer prompt assembly architecture.

### Changes

**Part A: Remove dead code**

1. Remove the auto-assignment block for parsing models (the list comprehension at config.py lines 320-330 that copies models with the default system prompt). Keep `DEFAULT_PARSING_SYSTEM_PROMPT` as a constant (it is used by the CLI interactive mode as a UI default).
2. In `_validate_config()` (config.py line 369-380), split the validation loop so the `system_prompt` check only applies to answering models, not parsing models. Parsing models do not need `system_prompt` because `TemplateEvaluator` builds its own prompt.

```python
# Validation for answering models: require system_prompt
for model in self.answering_models:
    if not model.model_name:
        raise ValueError(...)
    # ... provider check ...
    if not model.system_prompt:
        raise ValueError(f"System prompt is required for answering model {model.id}")

# Validation for parsing models: system_prompt is optional
for model in self.parsing_models:
    if not model.model_name:
        raise ValueError(...)
    # ... provider check ...
    # No system_prompt check: TemplateEvaluator builds its own prompt
```

**Files requiring updates for Part A:**

| File | Change |
|------|--------|
| `config.py` lines 320-330 | Remove parsing model auto-assignment |
| `config.py` lines 369-380 | Split validation loop |
| `__init__.py` re-exports | Keep (constant still used by CLI) |

**Part B: Add instruction shortcuts on VerificationConfig**

Add convenience parameters on `VerificationConfig` that wire into `PromptConfig`. Names match PromptConfig field names exactly with `_instructions` suffix:

| Shortcut | Maps to PromptConfig field |
|----------|---------------------------|
| `generation_instructions` | `generation` |
| `parsing_instructions` | `parsing` |
| `abstention_detection_instructions` | `abstention_detection` |
| `sufficiency_detection_instructions` | `sufficiency_detection` |
| `rubric_evaluation_instructions` | `rubric_evaluation` |
| `agentic_parsing_instructions` | `agentic_parsing` |
| `deep_judgment_instructions` | `deep_judgment` |

In `__init__`, if any `*_instructions` shortcut is set and the corresponding `PromptConfig` field is `None`, create or update the `PromptConfig` to include the shortcut value.

**Merge behavior:** When `prompt_config` is provided, each shortcut only fills in PromptConfig fields that are `None`. A `PromptConfig(parsing="X")` combined with `generation_instructions="Y"` produces `PromptConfig(parsing="X", generation="Y")`. If both the shortcut and the PromptConfig field are set for the same field, `PromptConfig` takes precedence.

```python
# Example: VerificationConfig(parsing_instructions="Be strict")
# Equivalent to: VerificationConfig(prompt_config=PromptConfig(parsing="Be strict"))

# Merge example: VerificationConfig(
#     prompt_config=PromptConfig(parsing="X"),
#     generation_instructions="Y"
# )
# Result: PromptConfig(parsing="X", generation="Y")
```

### Test Plan

- Test that removing the parsing model auto-assignment does not break parsing (TemplateEvaluator builds its own prompt).
- Test that parsing models can have `system_prompt=None` without validation failure.
- Test that answering models still require `system_prompt` (validation unchanged for answering).
- Test each `*_instructions` shortcut wires into the corresponding PromptConfig field.
- Test that explicit PromptConfig fields take precedence over shortcuts.
- Test merge: PromptConfig(parsing="X") + generation_instructions="Y" results in both fields set.
- Test that shortcuts are ignored when PromptConfig already has the corresponding field set.

## 3. FewShotConfig Restructure

**File**: `src/karenina/schemas/config/models.py`

### Problem

`FewShotConfig` has unclear semantics: `enabled` is a boolean kill switch, `global_mode` ambiguously controls pool selection (with `"none"` meaning "no examples at all"), `external_examples` are silently dropped when `mode='none'`, and the naming doesn't communicate what each concept does.

### New Field Structure

```python
class FewShotConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Master source selector: what types of examples are active
    source: Literal["disabled", "question_pool", "global", "both"] = "both"

    # Pool selection: how question-specific examples are chosen
    # Only relevant when source includes question_pool ("question_pool" or "both")
    pool_mode: Literal["all", "k-shot", "custom"] = "all"
    pool_k: int = 3

    # Per-question pool overrides
    question_configs: dict[str, QuestionFewShotConfig] = Field(default_factory=dict)

    # Global examples appended to ALL questions
    # Only used when source includes global ("global" or "both")
    global_examples: list[dict[str, str]] = Field(default_factory=list)
```

```python
class QuestionFewShotConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["all", "k-shot", "custom", "inherit"] = "inherit"
    k: int | None = None
    selected_examples: list[str | int] | None = None
    excluded_examples: list[str | int] | None = None
    # external_examples: REMOVED
```

### Removed Fields (Breaking Changes)

| Removed | Replacement |
|---------|-------------|
| `FewShotConfig.enabled` | `source` ("disabled" replaces `False`, "both" replaces `True`) |
| `FewShotConfig.global_mode` | `pool_mode` |
| `FewShotConfig.global_k` | `pool_k` |
| `FewShotConfig.global_external_examples` | `global_examples` |
| `QuestionFewShotConfig.external_examples` | Removed entirely |
| `QuestionFewShotConfig.mode = "none"` | Use `source="disabled"` or `source="global"` instead |

No backwards compatibility shims. Clean break.

### Inheritance Resolution

When `QuestionFewShotConfig.mode = "inherit"`, the question inherits `pool_mode` and `pool_k` from the top-level `FewShotConfig`. Pool examples are only resolved when `source` includes `"question_pool"` (i.e., `source="question_pool"` or `source="both"`). If `source="global"` or `source="disabled"`, the inherited pool_mode is irrelevant since pool selection is skipped.

### Behavior in resolve_examples_for_question

| source | Pool examples | Global examples |
|--------|--------------|-----------------|
| `"disabled"` | No | No |
| `"question_pool"` | Yes (per pool_mode) | No |
| `"global"` | No | Yes |
| `"both"` | Yes (per pool_mode) | Yes |

### Ripple Effects

All callers of the old FewShotConfig API must be updated:

| File | What to change |
|------|----------------|
| `config.py:387-396` | `_validate_config()`: replace `few_shot_config.enabled` with `source != "disabled"`, replace `global_mode` with `pool_mode`, replace `global_k` with `pool_k` |
| `config.py:558` | `__repr__`: replace `few_shot_config.enabled` with `source != "disabled"` |
| `config.py:586-594` | `is_few_shot_enabled()`: replace `config.enabled` with `config.source != "disabled"` |
| `task_helpers.py:111` | `resolve_few_shot_for_task`: replace `few_shot_config.enabled` with `source != "disabled"` |
| `models.py` | Factory methods (`from_index_selections`, `from_hash_selections`, `k_shot_for_questions`): update `global_mode` to `pool_mode`, `global_k` to `pool_k` |
| `models.py` | Mutation methods (`add_selections_by_index`, `add_selections_by_hash`, `add_k_shot_configs`): update field references |
| `models.py` | `get_effective_config()`: update inheritance to use `pool_mode`/`pool_k` |
| `models.py` | `validate_selections()`: remove external_examples references |
| Tests | All test fixtures referencing old field names |
| Docs | `docs/core_concepts/few-shot.md`, notebooks, `docs/workflows/` |
| Server | karenina-server schemas (if they expose FewShotConfig) |

### Test Plan

- Test `source="disabled"` returns no examples.
- Test `source="question_pool"` returns only pool examples (no globals).
- Test `source="global"` returns only global examples (no pool).
- Test `source="both"` returns pool + global examples.
- Test `pool_mode="all"`, `pool_mode="k-shot"`, `pool_mode="custom"` each work correctly.
- Test that old field names (`enabled`, `global_mode`, etc.) raise validation errors (extra="forbid").
- Test factory methods updated to new field names.
- Test `get_effective_config()` with `mode="inherit"` under each `source` value.
- Test `validate_selections()` with new structure.
- Test mutation methods (`add_selections_by_index`, etc.) work with new field names.

## 4. Remove Runner Auto-Upgrade

**File**: `src/karenina/benchmark/verification/runner.py`

### Problem

When `evaluation_mode="template_only"` but rubric traits are present, the runner silently overrides to `"template_and_rubric"` (lines 211-220). No logging occurs. This is a deliberate behavioral change: the auto-upgrade IS reachable through the normal pipeline (via batch_runner calling run_single_model_verification), but the silent override causes confusion. Users who explicitly set `template_only` should get `template_only` behavior with a clear warning about unused rubric traits.

### Change

Remove the silent auto-upgrade (delete the `evaluation_mode = "template_and_rubric"` assignment). Replace with a warning log that preserves the user's explicit `evaluation_mode` choice:

```python
_has_rubric_traits = rubric and (
    rubric.llm_traits
    or rubric.regex_traits
    or rubric.callable_traits
    or rubric.metric_traits
    or rubric.agentic_traits
)
_has_dynamic_rubric_traits = dynamic_rubric is not None and not dynamic_rubric.is_empty()
if (_has_rubric_traits or _has_dynamic_rubric_traits) and evaluation_mode == "template_only":
    logger.warning(
        "Rubric traits were provided but evaluation_mode='template_only'. "
        "Rubric evaluation will be skipped. Set evaluation_mode='template_and_rubric' "
        "to evaluate rubric traits."
    )
```

### Test Plan

- Test that `evaluation_mode="template_only"` with rubric traits emits a warning and does NOT auto-upgrade.
- Test that `evaluation_mode="template_and_rubric"` with rubric traits works normally (no warning).
- Test that `evaluation_mode="template_only"` without rubric traits emits no warning.

## Implementation Order

1. **055 (Runner auto-upgrade)**: Simplest, standalone, no dependencies.
2. **031 (PromptConfig.generation)**: Small change in generate_answer.py.
3. **032 (Parsing system prompt + shortcuts)**: Touches config.py validation, adds new fields.
4. **034 (FewShotConfig restructure)**: Largest change with ripple effects across docs, tests, and factory methods.

## Documentation Updates

After implementation, the following docs need updates:
- `docs/core_concepts/few-shot.md` and paired notebook
- `docs/workflows/running-verification/few-shot-configuration.md` and paired notebook
- `docs/workflows/creating-benchmarks/scaled-authoring.md` and paired notebook
- Skills in `.claude/skills/` that reference FewShotConfig or PromptConfig

Per CLAUDE.md, the user should be asked about skill/doc updates after code changes are committed.
