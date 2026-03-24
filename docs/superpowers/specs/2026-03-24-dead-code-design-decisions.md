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

In `GenerateAnswerStage.execute()`, read `context.prompt_config.get_for_task("generation")` and append it to the system message after `ModelConfig.system_prompt`.

```python
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

This keeps `ModelConfig.system_prompt` as the base and layers `PromptConfig.generation` on top, consistent with how other stages layer PromptConfig user instructions.

### Test Plan

- Test that `PromptConfig(generation="Focus on accuracy")` results in the instruction appearing in the system message sent to the adapter.
- Test that when `PromptConfig.generation` is None, behavior is unchanged.
- Test that both `ModelConfig.system_prompt` and `PromptConfig.generation` are joined when both are set.

## 2. Remove Dead Parsing System Prompt + Add Instruction Shortcuts

**Files**: `src/karenina/schemas/verification/config.py`, `src/karenina/schemas/verification/prompt_config.py`

### Problem

`VerificationConfig.__init__` auto-assigns `DEFAULT_PARSING_SYSTEM_PROMPT` to parsing models with `system_prompt=None`. The parsing pipeline never reads `ModelConfig.system_prompt`: `TemplateEvaluator` builds its own system prompt via `TemplatePromptBuilder.build_system_prompt()`. The auto-assignment exists only to pass a non-empty validation check.

User customization of parsing already works via `PromptConfig(parsing="...")`, which is the user instruction layer in the 3-layer prompt assembly architecture.

### Changes

**Part A: Remove dead code**

1. Remove `DEFAULT_PARSING_SYSTEM_PROMPT` constant from `config.py`.
2. Remove the auto-assignment block for parsing models (the list comprehension at lines 320-330 that copies models with the default system prompt).
3. Relax validation to allow `system_prompt=None` on parsing models. The validation check `_validate_model_configs()` should not require `system_prompt` on parsing models.

**Part B: Add instruction shortcuts on VerificationConfig**

Add convenience parameters on `VerificationConfig` that wire into `PromptConfig`, one for each `PromptConfig` field:

- `generation_instructions: str | None`
- `parsing_instructions: str | None`
- `abstention_instructions: str | None`
- `sufficiency_instructions: str | None`
- `rubric_instructions: str | None`
- `agentic_parsing_instructions: str | None`
- `deep_judgment_instructions: str | None`

In `__init__`, if any `*_instructions` shortcut is set and the corresponding `PromptConfig` field is not already populated, create or update the `PromptConfig` to include the shortcut value.

```python
# Example: VerificationConfig(parsing_instructions="Be strict")
# Equivalent to: VerificationConfig(prompt_config=PromptConfig(parsing="Be strict"))
```

If both the shortcut and the `PromptConfig` field are set, `PromptConfig` takes precedence (the shortcut is a convenience, not an override).

### Test Plan

- Test that removing DEFAULT_PARSING_SYSTEM_PROMPT does not break parsing (TemplateEvaluator builds its own prompt).
- Test that parsing models can have `system_prompt=None` without validation failure.
- Test each `*_instructions` shortcut wires into the corresponding PromptConfig field.
- Test that explicit PromptConfig fields take precedence over shortcuts.
- Test that setting a shortcut when PromptConfig already has the field does not overwrite.

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

### Behavior in resolve_examples_for_question

| source | Pool examples | Global examples |
|--------|--------------|-----------------|
| `"disabled"` | No | No |
| `"question_pool"` | Yes (per pool_mode) | No |
| `"global"` | No | Yes |
| `"both"` | Yes (per pool_mode) | Yes |

### Ripple Effects

All callers of the old FewShotConfig API must be updated:

- `VerificationConfig` (if it references FewShotConfig fields)
- Any factory methods on FewShotConfig (`from_index_selections`, `from_hash_selections`, `k_shot_for_questions`)
- Test fixtures
- Documentation (docs/core_concepts/few-shot.md, notebooks, etc.)
- Server API schemas (if karenina-server exposes FewShotConfig)

### Test Plan

- Test `source="disabled"` returns no examples.
- Test `source="question_pool"` returns only pool examples (no globals).
- Test `source="global"` returns only global examples (no pool).
- Test `source="both"` returns pool + global examples.
- Test `pool_mode="all"`, `pool_mode="k-shot"`, `pool_mode="custom"` each work correctly.
- Test that old field names (`enabled`, `global_mode`, etc.) raise validation errors (extra="forbid").
- Test factory methods updated to new field names.

## 4. Remove Runner Auto-Upgrade

**File**: `src/karenina/benchmark/verification/runner.py`

### Problem

When `evaluation_mode="template_only"` but rubric traits are present, the runner silently overrides to `"template_and_rubric"` (lines 211-220). No logging occurs. The auto-upgrade is unreachable through `Benchmark.run_verification()` but reachable via direct `run_single_model_verification()` calls.

### Change

Remove the silent auto-upgrade. Replace with a warning log:

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
