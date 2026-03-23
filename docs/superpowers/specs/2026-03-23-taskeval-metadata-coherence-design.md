# TaskEval Metadata Coherence: Design Spec

**Date**: 2026-03-23
**Scope**: Issues 024, 114, 115, 117, 160, 165, 166, 168, 169, 179 (170 already fixed; 167 deferred to Pass B)
**Primary files**: `task_eval.py`, `models.py`, `adapters/taskeval/registration.py` (new)
**Secondary files**: `utils/checkpoint.py` (Pass B only), `adapters/registry.py` (add taskeval to builtins)

## Problem

TaskEval results are structurally incompatible with Benchmark results for cross-path analysis. Nine metadata fields diverge between the two paths, breaking joins, grouping, and DataFrame concatenation. Additional functional bugs produce incorrect success rates, missing replicate metadata, and swallowed programming errors.

## Design Decisions (Resolved)

| Issue | Decision |
|-------|----------|
| **160** (guard flags) | Pass `config.abstention_enabled` and `config.sufficiency_enabled` through; both default to `False` |
| **166** (model identity) | Optional `answering_model: ModelConfig \| None` on `evaluate()`, fallback to sentinel `ModelConfig(model_name="user-provided", model_provider="user-provided", interface="taskeval")` |
| **117** (exception scope) | Narrow catch to `KareninaError`, `ValueError`, `RuntimeError`; record failures in `StepEval.failed_questions`; let programming bugs propagate |

## Implementation: Pass A (Mechanical Fixes)

### 0. Register `taskeval` Interface

Create `src/karenina/adapters/taskeval/__init__.py` (empty) and `src/karenina/adapters/taskeval/registration.py`:

```python
"""TaskEval adapter registration.

Registers a no-op 'taskeval' interface for use as a sentinel in ModelConfig
when TaskEval evaluates pre-collected outputs with no live LLM invocation.
"""

from karenina.adapters.registry import AdapterAvailability, AdapterRegistry, AdapterSpec


def _check_availability() -> AdapterAvailability:
    return AdapterAvailability(
        available=True,
        reason="TaskEval interface for pre-collected output evaluation",
    )


_taskeval_spec = AdapterSpec(
    interface="taskeval",
    description="No-op interface for TaskEval pre-collected output evaluation",
    llm_factory=None,
    parser_factory=None,
    agent_factory=None,
    availability_checker=_check_availability,
    fallback_interface=None,
    routes_to=None,
    supports_mcp=False,
    supports_tools=False,
    requires_provider=False,
)

AdapterRegistry.register(_taskeval_spec)
```

Add `"karenina.adapters.taskeval.registration"` to `_load_builtins()` in `adapters/registry.py`.

### 1. `evaluate()` Signature Expansion

Add two optional parameters to `TaskEval.evaluate()`:

```python
def evaluate(
    self,
    config: VerificationConfig,
    step_id: str | None = None,
    merge_strategy: Literal["concatenate", "traces_only"] | None = None,
    answering_model: ModelConfig | None = None,
    run_name: str | None = None,
) -> TaskEvalResult:
```

At the top of `evaluate()`, resolve defaults:

```python
from uuid import uuid4

if answering_model is None:
    answering_model = ModelConfig(
        id="taskeval_user_provided",
        model_name="user-provided",
        model_provider="user-provided",
        interface="taskeval",
    )

if run_name is None:
    run_name = f"taskeval_{uuid4().hex[:8]}"
```

Also emit FewShotConfig warning (issue 179):

```python
if config.is_few_shot_enabled():
    logger.debug("FewShotConfig has no effect in TaskEval mode")
```

Thread `answering_model` and `run_name` through the full call chain:

| Method | New Parameters Added |
|--------|---------------------|
| `evaluate()` | `answering_model`, `run_name` |
| `_evaluate_global()` | `answering_model`, `run_name` |
| `_evaluate_step()` | `answering_model`, `run_name` |
| `_evaluate_step_internal()` | `answering_model`, `run_name` |
| `_run_evaluation_loop()` | `answering_model`, `run_name` |
| `_evaluate_and_store()` | `answering_model`, `run_name`, `replicate` |
| `_evaluate()` | `answering_model`, `run_name`, `replicate`, `abstention_enabled`, `sufficiency_enabled` |

### 2. Replicate Tracking (Issue 114)

In `_run_evaluation_loop()`, change:

```python
# Before
for _ in range(replicate_count):

# After
for rep_idx in range(replicate_count):
    replicate = None if replicate_count == 1 else rep_idx + 1
```

Thread `replicate` through `_evaluate_and_store()` and `_evaluate()` to `run_single_model_verification(replicate=replicate)`.

### 3. Guard Flag Threading (Issue 160)

Extract from config in `_run_evaluation_loop()` or `_evaluate()`:

```python
abstention_enabled = config.abstention_enabled  # defaults to False
sufficiency_enabled = config.sufficiency_enabled  # defaults to False
```

Pass to `run_single_model_verification()` instead of the hardcoded `abstention_enabled=True`.

### 4. Call Site Consolidation (Issues 114, 160, 165, 166, 168, 169)

The `run_single_model_verification()` call at task_eval.py:748 becomes:

```python
verification_result = run_single_model_verification(
    question_id=question_id,              # 165: pass as-is, no force-hashing
    question_text=question_text,          # 169: already passed correctly
    template_code=answer_template,
    answering_model=answering_model,      # 166: from evaluate() param or sentinel
    parsing_model=parsing_model,
    rubric=rubric,
    dynamic_rubric=dynamic_rubric,
    cached_answer_data=cached_answer_data,
    run_name=run_name,                    # 168: threaded from evaluate()
    replicate=replicate,                  # 114: threaded from loop index
    abstention_enabled=abstention_enabled,      # 160: from config
    sufficiency_enabled=sufficiency_enabled,    # 160: from config
    rubric_evaluation_strategy="batch",
    evaluation_mode=evaluation_mode,
)
```

**Question ID (165)**: Remove the `_is_valid_md5_hash()` check and forced MD5 re-hashing. Pass the question_id from `question_dict.get("id", "unknown")` directly. The `_is_valid_md5_hash()` helper method can be removed entirely if no other code references it.

**Mock model removal (166)**: Delete the `mock_answering_model` construction block (lines 727-733). Use the `answering_model` parameter instead.

### 5. Exception Narrowing (Issue 117)

Add `failed_questions` field to `StepEval` in `models.py`:

```python
failed_questions: dict[str, list[str]] = Field(
    default_factory=dict,
    description="Questions that failed evaluation, keyed by question_id with error messages",
)
```

In `_evaluate_and_store()`, narrow the catch:

```python
try:
    verification_result = self._evaluate(...)
    # ... store result ...
except (KareninaError, ValueError, RuntimeError) as e:
    logger.warning("Evaluation failed for %s: %s", error_context, e)
    if question_id not in step_eval.failed_questions:
        step_eval.failed_questions[question_id] = []
    step_eval.failed_questions[question_id].append(str(e))
```

Import `KareninaError` from `karenina.exceptions`. This catches all domain-specific errors (PortError, ParseError, AgentExecutionError, VerificationBatchError, etc.) while letting programming bugs (AttributeError, TypeError, KeyError) propagate.

### 6. `get_summary_stats()` Rubric-Only Fix (Issue 024)

In `models.py`, replace the `passed_traces` logic:

```python
for _trace_id, results in self.verification_results.items():
    total_results += len(results)

    template_passed = any(
        result.template and result.template.verify_result for result in results
    )
    template_performed = any(
        result.template and result.template.template_verification_performed
        for result in results
    )
    rubric_passed = any(
        result.rubric and result.rubric.rubric_evaluation_performed
        and all(
            score is True or (isinstance(score, int) and score > 0)
            for score in result.rubric.get_all_trait_scores().values()
        )
        for result in results
    )

    if template_passed or (not template_performed and rubric_passed):
        passed_traces += 1
```

### 7. One-Line Fixes

**Issue 115** (`_get_available_step_ids()`): Add after line 417:

```python
step_ids.update(self.step_dynamic_rubrics.keys())
```

**Issue 179** (FewShotConfig warning): Covered in Section 1 above.

## Implementation: Pass B (Cross-Path Normalization, Issue 167)

Normalize source code in `generate_template_id()` in `utils/checkpoint.py` before hashing:
- Strip leading whitespace per line
- Normalize line endings to `\n`
- Strip trailing whitespace

This ensures the same template class produces the same hash regardless of indentation context (module-level vs. string literal in TaskEval). Requires testing against both Benchmark and TaskEval paths.

Deferred to a separate pass because it touches shared infrastructure.

## Testing Strategy

### Unit Tests (Pass A)

For each fix, add or update tests in `tests/unit/benchmark/test_task_eval.py` and `test_task_eval_issues.py`:

- **024**: Verify `success_rate > 0` in rubric_only mode when all traits pass; verify `success_rate == 0` when traits fail
- **114**: Verify `replicate` parameter reaches `run_single_model_verification()` with correct index; verify `None` when `replicate_count == 1`
- **115**: Verify a step with only dynamic rubrics appears in `_get_available_step_ids()`
- **117**: Verify `TypeError`/`AttributeError` propagate; verify `KareninaError`/`ValueError` are caught and recorded in `failed_questions`
- **160**: Verify `abstention_enabled` and `sufficiency_enabled` from config reach `run_single_model_verification()`
- **165**: Verify non-MD5 question IDs pass through without hashing
- **166**: Verify sentinel `ModelConfig` constructs without validation error; verify sentinel is used when no `answering_model` provided; verify user-provided model passes through
- **168**: Verify `run_name` reaches runner (auto-generated and explicit)
- **169**: Already passing correctly (confirmed in code review)
- **179**: Verify debug log emitted when `is_few_shot_enabled()` is True

### Regression

Run full TaskEval test suite: `uv run pytest tests/ -x -q -k "task_eval"` from `karenina/`.

### Cross-Path (Pass B)

Run full test suite: `uv run pytest tests/ -x -q` to confirm `generate_template_id()` changes don't break the Benchmark path.

## Out of Scope

- Issue 170: Already fixed with regression tests
- Issue 167: Deferred to Pass B (separate commit)
