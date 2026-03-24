# API Design Improvements

**Date**: 2026-03-24
**Scope**: Core library (`karenina/`) and server (`karenina-server/`), ~12 files
**Effort**: ~4 hours

## Overview

Fourteen items covering API gaps, naming inconsistencies, type improvements, and design decisions. One item (131: inject_question_id) was dropped after investigation showed the current approach is correct. Grouped into five sections.

## Section 1: ResultsStore (replaces facade result storage)

### Decision

The `Benchmark` object defines evaluation items (questions, templates, rubrics), not results. Result storage methods are moved from the facade to a standalone `ResultsStore` class.

### Changes

**Create `ResultsStore`** in `karenina/benchmark/core/results_store.py`:

```python
class ResultsStore:
    """Container for gathering verification results across runs.

    Stores VerificationResultSets keyed by run name. Provides query
    methods for filtering by question, run, or recency.
    """

    def add(self, result_set: VerificationResultSet, run_name: str | None = None) -> None:
        """Store a result set.

        Args:
            result_set: The results to store.
            run_name: Key for this run. If None, auto-generates a
                timestamped name (e.g. "run_2026-03-24T10:30:00").
        """

    def get_latest(self, question_id: str | None = None) -> dict[str, VerificationResult]:
        """Get latest results. Keys are question IDs.

        If question_id is given, returns only that question's latest result
        (dict with one entry). Otherwise returns the latest result per question
        across all runs.
        """

    def has_results(self, question_id: str | None = None, run_name: str | None = None) -> bool:
        """Check whether results exist, optionally filtered."""

    def get_by_question(self, question_id: str) -> dict[str, VerificationResult]:
        """Get all results for a question. Keys are run names."""

    def get_by_run(self, run_name: str) -> VerificationResultSet:
        """Get the full result set for a specific run."""

    def get_all_runs(self) -> list[str]:
        """List all stored run names, ordered by insertion."""

    def get_summary(self, run_name: str | None = None) -> dict[str, Any]:
        """Get summary statistics, optionally filtered by run."""

    def get_statistics_by_run(self) -> dict[str, dict[str, Any]]:
        """Get per-run statistics."""

    def clear(self, question_ids: list[str] | None = None, run_name: str | None = None) -> None:
        """Remove results, optionally filtered."""

    def export(self, question_ids: list[str] | None = None, run_name: str | None = None) -> dict:
        """Export results as serializable dict."""

    def export_to_file(self, file_path: Path, **kwargs) -> None:
        """Export results to a JSON file."""

    @classmethod
    def from_file(cls, file_path: Path) -> ResultsStore:
        """Load a ResultsStore from a previously exported file."""
```

**Note**: `VerificationResultSet` has no `run_name` field. The `add()` method accepts an explicit `run_name` parameter; if omitted, a timestamped name is generated.

**Deprecate on `Benchmark` facade** (with deprecation warnings pointing to `ResultsStore`):
- `store_verification_results()`
- `get_verification_results()`
- `get_verification_history()`
- `clear_verification_results()`
- `export_verification_results()`
- `export_verification_results_to_file()`
- `load_verification_results_from_file()`
- `get_verification_summary()`
- `get_all_run_names()`
- `get_results_statistics_by_run()`

These methods remain functional but emit `DeprecationWarning` pointing to the `ResultsStore` equivalent. They will be removed in a future release. The internal `_results_manager` stays for now to back the deprecated methods.

**Server impact**: Check `karenina-server/` for any usage of the deprecated methods and update to use `ResultsStore` where appropriate.

**Usage pattern:**

```python
store = ResultsStore()
results = benchmark.run_verification(config)
store.add(results, run_name="experiment_1")

store.get_latest("q1")          # {"q1": VerificationResult}
store.has_results(run_name="experiment_1")
store.get_by_question("q1")     # {"experiment_1": VerificationResult}
store.get_by_run("experiment_1") # VerificationResultSet
```

## Section 2: Facade API Gaps

### 019: Add `get_question_rubric()` to facade

`RubricManager.get_question_rubric()` returns `list[TraitUnion] | None` (a flat trait list), not a `Rubric`. The facade should wrap this into a `Rubric` for a clean public API:

```python
def get_question_rubric(self, question_id: str) -> Rubric | None:
    """Get the question-specific rubric for a question."""
    traits = self._rubric_manager.get_question_rubric(question_id)
    if traits is None:
        return None
    return Rubric.from_traits(traits)
```

If `Rubric.from_traits()` does not exist, add a factory method that categorizes a flat trait list into the typed trait lists (`llm_traits`, `regex_traits`, etc.). Reference the existing categorization pattern in `RubricManager.get_global_rubric()` (rubrics.py:82-95) which already separates traits by type using `isinstance` checks.

### 144: `set_global_rubric()` and `set_question_rubric()` delegation

Both facade methods currently decompose a `Rubric` into traits and add them one-by-one. The manager should accept both a `Rubric` object and a raw traits list.

**RubricManager changes:**
- `set_global_rubric(rubric: Rubric | list[TraitUnion])`: accepts either form. When given a `Rubric`, extract all traits via its typed trait lists. The current manager method already replaces atomically (not append), so clear-then-set semantics are preserved.
- Add `set_question_rubric(question_id: str, rubric: Rubric | list[TraitUnion])`: entirely new method on the manager (does not currently exist). Clears existing question rubric first, then sets the new traits.

**Facade changes:** Both `set_global_rubric()` and `set_question_rubric()` become one-liner delegations.

### 145: Rename `RubricManager.get_questions_with_rubric()` to `get_question_ids_with_rubric()`

Internal-only rename. The facade's same-named method delegates to `QuestionManager` (returns `list[dict]`), which is unchanged. Update all internal callers of the old name.

## Section 3: Naming and Consistency

### 026: Document `summary()` vs `summary_compact()` scope

Add docstring clarification to both `TaskEvalResult` methods:
- `summary()`: "Aggregates across global evaluation and all per-step evaluations"
- `summary_compact()`: "Reports global evaluation statistics only"

### 047: Unify `validate_rubrics()` error shape

Change `validate_rubrics()` return from `tuple[bool, list[str]]` to `tuple[bool, list[dict[str, str]]]` with `{"source": trait_name_or_scope, "error": message}`.

Note: `validate_templates()` uses `{"question_id": ..., "error": ...}` as its dict keys. The key names differ intentionally because templates are scoped to questions while rubric errors are scoped to traits/rubric scopes. The shapes are structurally aligned (`list[dict[str, str]]`) even if the key names are domain-appropriate.

### 064: Remove duplicate config router mount

Remove `app.include_router(config_router, prefix="/api/config")` from `server.py` (line 541). Keep only the `/api` mount (line 542). The `/api/config` mount creates redundant paths like `/api/config/v2/config/...` (doubled prefix).

Pre-implementation check: grep `karenina-gui/` for any references to `/api/config/v2/config/` paths. If found, update them to use `/api/v2/config/` instead.

Expected result: exactly 11 config endpoints (down from 22).

## Section 4: Type Improvements

### 120: Simplify `VerificationResultSet.__repr__`

Replace the current `get_summary()`-calling repr with a lightweight version:

```python
def __repr__(self) -> str:
    n = len(self.results)
    return f"VerificationResultSet({n} result{'s' if n != 1 else ''})"
```

### 136: Type `deep_judgment_rubric_config`

The current `dict[str, Any] | None` has a nested structure:

```python
{
    "global": {"TraitName": {"enabled": True, "excerpt_enabled": True, ...}},
    "question_specific": {"question-id": {"TraitName": {"enabled": True, ...}}}
}
```

Create a dedicated Pydantic model:

```python
class DeepJudgmentRubricCustomConfig(BaseModel):
    """Per-trait deep judgment configuration for custom mode."""
    model_config = ConfigDict(extra="forbid")

    global_traits: dict[str, DeepJudgmentTraitConfig] = Field(
        default_factory=dict, alias="global"
    )
    question_specific: dict[str, dict[str, DeepJudgmentTraitConfig]] = Field(
        default_factory=dict
    )
```

Change the field type on `VerificationConfig`:
```python
deep_judgment_rubric_config: DeepJudgmentRubricCustomConfig | None = None
```

Add a cross-field `model_validator` to require this field when `deep_judgment_rubric_mode="custom"`.

Also update the `create()` factory method parameter type to match.

### 176: Add `by` parameter to view-level `group_by_model()`

Add `by: Literal["answering", "parsing", "both"] = "answering"` to:
- `TemplateResults.group_by_model()`
- `RubricResults.group_by_model()`
- `JudgmentResults.group_by_model()`

Use the same key-building logic as `VerificationResultSet.group_by_model()`, including MCP server info in answering keys when present.

### 121: Preserve `None` in `group_by_replicate()`

Change return type from `dict[int, VerificationResultSet]` to `dict[int | None, VerificationResultSet]`. Use `result.metadata.replicate` directly instead of `result.metadata.replicate or 0`.

Currently `group_by_replicate()` has no external callers (only used internally within `VerificationResultSet`). The internal `_calculate_replicate_stats` method uses its own grouping logic and already filters `None` replicates, so it is unaffected.

## Section 5: `higher_is_better` on all trait types

### Decision

Add `higher_is_better: bool | None` to all trait types. `None` means directionality does not apply.

### Legacy validator changes

`LLMRubricTrait`, `RegexRubricTrait`, `CallableRubricTrait`, and `AgenticRubricTrait` each have a `set_legacy_defaults` validator that coerces `None` to `True`:

```python
if "higher_is_better" not in values or values.get("higher_is_better") is None:
    values["higher_is_better"] = True
```

This must be changed to only fill the default when the key is **absent**, not when it is explicitly `None`:

```python
if "higher_is_better" not in values:
    values["higher_is_better"] = True
```

This preserves backward compatibility for legacy data missing the field, while allowing explicit `None` to mean "directionality does not apply."

### Field declaration changes

`LLMRubricTrait`, `RegexRubricTrait`, and `CallableRubricTrait` declare `higher_is_better` as `Field(...)` (required, no default). Change to `Field(default=True)` so the field is optional with a sensible default. Combined with the type widening to `bool | None`, the full declaration becomes:

```python
higher_is_better: bool | None = Field(
    default=True,
    description="Whether higher values indicate better performance. "
    "None means directionality does not apply."
)
```

### Per-trait changes

- `LLMRubricTrait`: type `bool` -> `bool | None`, add `default=True`, fix legacy validator
- `RegexRubricTrait`: type `bool` -> `bool | None`, add `default=True`, fix legacy validator
- `CallableRubricTrait`: type `bool` -> `bool | None`, add `default=True`, fix legacy validator
- `AgenticRubricTrait`: already `bool | None`, but its `set_legacy_defaults` validator has the same coercion bug (coerces explicit `None` to `True` for non-template kinds). Apply the same fix: only fill default when key is absent.
- `MetricRubricTrait`: add `higher_is_better: bool | None = Field(default=None, ...)`. No `set_legacy_defaults` validator needed since the default is `None` and Pydantic handles missing fields automatically.

### Rubric.get_trait_directionalities()

Update to include metric traits. Remove the comment "MetricRubricTraits are excluded as metrics (precision/recall/F1) are inherently 'higher is better'." Update the docstring to reflect that `None` values indicate directionality does not apply.

Return type is already `dict[str, bool | None]` (AgenticRubricTrait template kinds already return `None`).

## Dropped

### 131: `inject_question_id` subclass chain

Dropped after investigation. The function operates on classes (not instances) and the class is instantiated later during LLM response parsing. The subclassing approach is necessary; the proposed "set after construction" fix does not apply.

## Verification

- Full test suite passes in both `karenina/` and `karenina-server/`
- New facade methods (019) are callable and delegate correctly
- `set_global_rubric` and `set_question_rubric` accept both `Rubric` and `list` (144)
- Config router (064) produces exactly 11 config endpoints, not 22
- `__repr__` (120) is fast for large result sets
- `higher_is_better` works as `bool | None` across all trait types; explicit `None` is preserved (not coerced to `True`)
- `group_by_replicate` (121) preserves `None` keys; callers handle `None` gracefully
- `ResultsStore` accumulates results across runs and exposes query/export methods
- Deprecated facade methods emit `DeprecationWarning`
- `deep_judgment_rubric_config` validates against `DeepJudgmentRubricCustomConfig` model
