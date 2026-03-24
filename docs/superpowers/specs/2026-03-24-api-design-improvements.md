# API Design Improvements

**Date**: 2026-03-24
**Scope**: Core library (`karenina/`) and server (`karenina-server/`), ~10 files
**Effort**: ~3 hours

## Overview

Fourteen items covering API gaps, naming inconsistencies, type improvements, and design decisions. One item (131: inject_question_id) was dropped after investigation showed the current approach is correct. Grouped into five sections.

## Section 1: ResultsStore (replaces facade result storage)

### Decision

The `Benchmark` object defines evaluation items (questions, templates, rubrics), not results. Result storage methods are removed from the facade and replaced with a standalone `ResultsStore` class.

### Changes

**Remove from `Benchmark` facade:**
- `store_verification_results()`
- `get_verification_results()`
- `get_verification_history()`
- Remove `ResultsManager` as a Benchmark dependency (if it is only used for result storage)

**Create `ResultsStore`** in `karenina/benchmark/core/results_store.py`:

```python
class ResultsStore:
    """Container for gathering verification results across runs."""

    def add(self, result_set: VerificationResultSet) -> None:
        """Store a result set, keyed by run name."""

    def get_latest(self, question_id: str | None = None) -> dict[str, VerificationResult]:
        """Get latest results, optionally filtered by question."""

    def has_results(self, question_id: str | None = None, run_name: str | None = None) -> bool:
        """Check whether results exist."""

    def get_by_question(self, question_id: str) -> dict[str, VerificationResult]:
        """Get all results for a question across runs."""

    def get_by_run(self, run_name: str) -> dict[str, VerificationResult]:
        """Get all results for a specific run."""

    def get_all_runs(self) -> list[str]:
        """List all stored run names."""
```

**Usage pattern:**

```python
store = ResultsStore()
results = benchmark.run_verification(config)
store.add(results)

store.get_latest("q1")
store.has_results(run_name="run_2")
store.get_by_question("q1")
```

## Section 2: Facade API Gaps

### 019: Add `get_question_rubric()` to facade

Simple delegation:

```python
def get_question_rubric(self, question_id: str) -> Rubric | None:
    """Get the question-specific rubric for a question."""
    return self._rubric_manager.get_question_rubric(question_id)
```

### 144: `set_global_rubric()` and `set_question_rubric()` delegation

Both facade methods currently decompose a `Rubric` into traits and add them one-by-one. The manager should accept both a `Rubric` object and a raw traits list.

**RubricManager changes:**
- `set_global_rubric(rubric: Rubric | list[TraitUnion])`: accepts either form
- Add `set_question_rubric(question_id: str, rubric: Rubric | list[TraitUnion])`: new method

**Facade changes:** Both `set_global_rubric()` and `set_question_rubric()` become one-liner delegations.

### 145: Rename `RubricManager.get_questions_with_rubric()` to `get_question_ids_with_rubric()`

Internal-only rename. The facade's same-named method delegates to `QuestionManager` (returns `list[dict]`), which is unchanged.

## Section 3: Naming and Consistency

### 026: Document `summary()` vs `summary_compact()` scope

Add docstring clarification to both `TaskEvalResult` methods:
- `summary()`: "Aggregates across global evaluation and all per-step evaluations"
- `summary_compact()`: "Reports global evaluation statistics only"

### 047: Unify `validate_rubrics()` error shape

Change `validate_rubrics()` return from `tuple[bool, list[str]]` to `tuple[bool, list[dict[str, str]]]` with `{"source": trait_name_or_scope, "error": message}`. Matches `validate_templates()`.

### 064: Remove duplicate config router mount

Remove `app.include_router(config_router, prefix="/api/config")` from `server.py`. Keep only the `/api` mount. Verify the GUI does not reference `/api/config/v2/config/...` paths before removing.

## Section 4: Type Improvements

### 120: Simplify `VerificationResultSet.__repr__`

Replace the current `get_summary()`-calling repr with a lightweight version:

```python
def __repr__(self) -> str:
    return f"VerificationResultSet(results={len(self.results)}, run_name={self.run_name!r})"
```

### 136: Type `deep_judgment_rubric_config`

Change from `dict[str, Any] | None` to `dict[str, DeepJudgmentTraitConfig] | None`. `DeepJudgmentTraitConfig` already exists in the same file. Add a cross-field `model_validator` to require the field when `deep_judgment_rubric_mode="custom"`.

### 176: Add `by` parameter to view-level `group_by_model()`

Add `by: Literal["answering", "parsing", "both"] = "answering"` to:
- `TemplateResults.group_by_model()`
- `RubricResults.group_by_model()`
- `JudgmentResults.group_by_model()`

Use the same key-building logic as `VerificationResultSet.group_by_model()`.

### 121: Preserve `None` in `group_by_replicate()`

Change return type from `dict[int, VerificationResultSet]` to `dict[int | None, VerificationResultSet]`. Use `result.metadata.replicate` directly instead of `result.metadata.replicate or 0`.

## Section 5: `higher_is_better` on all trait types

### Decision

Add `higher_is_better: bool | None` to all trait types. `None` means directionality does not apply.

**Changes:**
- `LLMRubricTrait`: change from `bool` to `bool | None`, default stays `True`
- `RegexRubricTrait`: change from `bool` to `bool | None`, default stays `True`
- `CallableRubricTrait`: change from `bool` to `bool | None`, default stays `True`
- `AgenticRubricTrait`: change from `bool` to `bool | None`, default stays `True` (verify current state)
- `MetricRubricTrait`: add `higher_is_better: bool | None = None`
- `Rubric.get_trait_directionalities()`: include metric traits, return `dict[str, bool | None]`

## Dropped

### 131: `inject_question_id` subclass chain

Dropped after investigation. The function operates on classes (not instances) and the class is instantiated later during LLM response parsing. The subclassing approach is necessary; the proposed "set after construction" fix does not apply.

## Verification

- Full test suite passes in both `karenina/` and `karenina-server/`
- New facade methods (019) are callable and delegate correctly
- `set_global_rubric` and `set_question_rubric` accept both `Rubric` and `list` (144)
- Config router (064) produces exactly 11 config endpoints, not 22
- `__repr__` (120) is fast for large result sets
- `higher_is_better` (071) works as `bool | None` across all trait types
- `group_by_replicate` (121) preserves `None` keys
- `ResultsStore` accumulates results across runs
