# Chunk 21: Model Identity, Reproducibility, and Miscellaneous Fixes

## Overview

Ten independent fixes addressing model identity collapsing, k-shot reproducibility, export/import fidelity, boolean edge-case semantics, error handling asymmetry, and five mechanical improvements.

## Decision Outcomes

### 1. ModelIdentity enrichment (was issue: ModelIdentity collapses same-name configs)

**Decision**: Enrich `ModelIdentity` with `config_id` when it differs from `model_name`.

**Current state**: `ModelIdentity` captures `(interface, model_name, tools)`. Two `ModelConfig` objects with the same `model_name` but different `id`, `system_prompt`, `temperature`, or `max_tokens` produce identical identity values. Cache separates them correctly via `ModelConfig.id`; only result grouping is affected.

**Change**:
- Add optional `config_id: str | None = None` field to `ModelIdentity`
- In `from_model_config()`: set `config_id` when `ModelConfig.id != ModelConfig.model_name`
- Update `display_string`: append ` (config_id)` when present
- Update `canonical_key`: include `config_id` segment when present
- Update `_parse_model_identity()` in `results_io.py` to parse the `(config_id)` suffix from `display_string` on CSV reimport
- Note: `group_by_model()` in `verification_result_set.py` constructs its own key format (appending ` + MCP[...]` for tool servers) independently from `canonical_key`. It reads `display_string` via `result.metadata.answering_model`, so the `config_id` suffix propagates automatically into its grouping keys.

**Files**: `schemas/verification/model_identity.py`, `benchmark/core/results_io.py` (`_parse_model_identity`)

### 2. k-shot hash reproducibility (was issue: hash() not reproducible across processes)

**Decision**: Accept seed-breaking change. Swap `hash()` to `hashlib.md5`.

**Change**: In `FewShotConfig.resolve_examples_for_question()`, replace:
```python
random.seed(hash(question_id) & 0x7FFFFFFF)
```
with:
```python
random.seed(int(hashlib.md5(question_id.encode()).hexdigest(), 16) & 0x7FFFFFFF)
```

`hashlib` is already imported in the same class.

**Files**: `schemas/config/models.py` (one line)

### 3. Run name preservation on reimport (was issue: storage run_name lost on bulk export)

**Decision**: On reimport, group results by their `metadata.run_name` instead of the caller-provided `run_name`. Fall back to caller-provided name if `metadata.run_name` is `None`.

**Change**: The grouping logic belongs in `ResultsManager.load_results_from_file()` (not `ResultsIOManager.load_from_json()`), since `load_from_json` is a static utility that returns a flat `dict[str, VerificationResult]`. Changing its return type would be a breaking change. Instead, `load_results_from_file()` (which already accepts `run_name` and stores results via `self._in_memory_results[run_name] = results`) should iterate over the loaded results, group them by `result.metadata.run_name`, and store each group under its own key. The caller-provided `run_name` serves as fallback for results whose `metadata.run_name` is `None`.

**Files**: `benchmark/core/results.py` (`load_results_from_file`)

### 4. BooleanMatch strict None (was issue: BooleanMatch treats None as falsy)

**Decision**: Strict `None` handling. `None` is a non-match against both `True` and `False`.

**Change**: In `BooleanMatch.check()`, add a guard:
```python
if extracted is None or expected is None:
    return extracted is expected
```

This means `None` only matches `None` (identity), not `False` via `bool()` coercion.

**Files**: `schemas/primitives/comparisons.py`

### 5. Deep judgment hallucination assessment non-fatal (was issue: search makes parsing less reliable)

**Decision**: Make Stage 1.5 hallucination assessment non-fatal, consistent with search.

**Change**: The existing `try/except` at line 590/621 already covers the LLM invocation and JSON parsing for hallucination assessment. Change the existing `except (json.JSONDecodeError, Exception)` handler at line 621 from raising `ValueError` (fatal) to logging a warning and continuing without risk scores. Simplify the redundant exception tuple to just `except Exception`. Stage 2 already defaults to `"high"` risk when no per-excerpt data exists (line 708). Additionally, wrap lines 544-633 (the entire hallucination assessment block, including the cleanup code at lines 629-633 which is outside the inner try) in a single outer `try/except Exception` to catch errors in post-processing as well.

**Files**: `benchmark/verification/evaluators/template/deep_judgment.py`

## Mechanical Fixes

### 6. Abstention/sufficiency adapter instructions (was issue: hardcoded JSON format)

Extract `<output_format>` blocks from `ABSTENTION_DETECTION_SYS` and `SUFFICIENCY_DETECTION_SYS` into adapter instruction registrations. The task enum values are `PromptTask.ABSTENTION_DETECTION` (`"abstention_detection"`) and `PromptTask.SUFFICIENCY_DETECTION` (`"sufficiency_detection"`).

**Adapter registrations needed** (4 adapters that make LLM calls for these tasks):
- `langchain`: explicit JSON schema/format instructions (no native structured output)
- `claude_tool`: minimal instructions (native structured output handles formatting)
- `claude_agent_sdk`: instructions matching its capabilities
- `langchain_deep_agents`: delegates to LangChain-style LLM calls, same as langchain

`manual` and `taskeval` adapters do not need registrations (no LLM calls).

**Stripped prompt structure**: Remove the entire `<output_format>...</output_format>` section from each base prompt constant. The remaining prompt describes the task, evaluation criteria, and examples. Each adapter's `prompts/` directory gets a new module (e.g., `abstention.py`, `sufficiency.py`) with a dataclass implementing the `AdapterInstruction` protocol (`system_addition`/`user_addition` properties), a factory function, and a `register()` call. Each adapter's `registration.py` imports the new module to trigger registration.

**Files**: `benchmark/verification/prompts/trace/abstention.py`, `benchmark/verification/prompts/trace/sufficiency.py`, `adapters/{langchain,claude_tool,claude_agent_sdk,langchain_deep_agents}/prompts/` (new modules), `adapters/*/registration.py` (import additions)

### 7. rubric_only validation warning (was issue: rubric_only accepts config with no rubrics)

Add a warning in `runner.py` after the existing evaluation_mode auto-upgrade block:
```python
if evaluation_mode == "rubric_only" and not _has_rubric_traits and not _has_dynamic_rubric_traits:
    logger.warning(
        "evaluation_mode='rubric_only' but no rubric traits provided. "
        "Rubric evaluation will produce no scores."
    )
```

**Files**: `benchmark/verification/runner.py`

### 8. log_retry max_attempts parameter (was issue: log_retry hardcodes max_attempts=3)

Add `max_attempts: int = 3` keyword parameter to `log_retry()`. Update the two callers (abstention and sufficiency evaluators) to pass `max_attempts` via `functools.partial`. Also fix f-string log lines to use `%`-style formatting per coding conventions, both in the standalone `log_retry()` (line 38) and in the internal `_log_retry` closure inside `create_transient_retry()` (line 71).

**Files**: `utils/retry.py`, `benchmark/verification/evaluators/trace/abstention.py`, `benchmark/verification/evaluators/trace/sufficiency.py`

### 9. Embedding check config bypass (was issue: embedding check bypasses VerificationConfig)

Refactor `perform_embedding_check()` to accept `enabled`, `model`, and `threshold` as parameters instead of reading env vars directly. Update `EmbeddingCheckStage.execute()` to pass these values from `VerificationContext`.

Keep `_get_embedding_model_name()` as a private helper: it is also called by `preload_embedding_model()` (line 41) and `compute_embedding_similarity()` (lines 179, 186-188), which are standalone utility functions that legitimately need env-var defaults. Only `_should_use_embedding_check()` and `_get_embedding_threshold()` can be removed, as they are only called from `perform_embedding_check()`.

The `VerificationConfig` already reads env vars as field defaults, so the env-var path is preserved through config initialization.

**Files**: `benchmark/verification/utils/embedding_check.py`, `benchmark/verification/stages/pipeline/embedding_check.py`

### 10. Rename _template_validation.py (was issue: misleading location)

Rename `_template_validation.py` to `_rubric_kind_validation.py`. Update the import in `rubric.py`. The file is only used by rubric trait validation, never by answer template validation.

**Files**: `schemas/entities/_template_validation.py` (rename), `schemas/entities/rubric.py` (import update)

## Testing Strategy

Each fix gets tests written before implementation (TDD). Key test scenarios:

1. **ModelIdentity**: test that distinct config_ids produce distinct display_strings/canonical_keys; test that matching config_id/model_name omits config_id from display; test CSV round-trip via `_parse_model_identity` preserves config_id
2. **k-shot**: test that same question_id produces same seed across calls (determinism)
3. **Run name**: test round-trip export/import preserving multiple distinct run_name values as separate groups
4. **BooleanMatch**: test `check(None, False)` returns `False`; test `check(None, None)` returns `True`
5. **Deep judgment**: test that hallucination assessment failure logs warning and continues
6. **Adapter instructions**: test that abstention/sufficiency tasks have registered adapter instructions
7. **rubric_only warning**: test that warning is logged when rubric_only + no traits
8. **log_retry**: test that log message uses provided max_attempts value
9. **Embedding check**: test that perform_embedding_check uses passed parameters; negative test verifying env vars are not consulted when explicit parameters are provided
10. **Rename**: test that import path resolves correctly after rename

## Verification

```bash
cd karenina && uv run pytest tests/ -x -q
```
