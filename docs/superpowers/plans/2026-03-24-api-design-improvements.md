# API Design Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix API gaps, naming inconsistencies, type improvements, and extract result storage from the Benchmark facade into a standalone ResultsStore.

**Architecture:** Changes span the rubric schema layer (higher_is_better unification), the manager layer (RubricManager method improvements), the facade layer (deprecations and new delegations), a new ResultsStore class, and miscellaneous type/naming fixes. Each task is independently testable and committable.

**Tech Stack:** Python 3.13, Pydantic v2, pytest, uv

**Spec:** `docs/superpowers/specs/2026-03-24-api-design-improvements.md`

**Test runner:** `cd /Users/carli/Projects/karenina-salvage/karenina/.claude/worktrees/api-design && uv run pytest tests/ -x -q`

**Important:** Do not reference issue numbers in commit messages. They were an internal naming scheme.

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `src/karenina/schemas/entities/rubric.py` | higher_is_better on all traits, from_traits factory, get_trait_directionalities |
| Modify | `src/karenina/benchmark/core/rubrics.py` | set_global_rubric, set_question_rubric, rename get_questions_with_rubric, validate_rubrics |
| Modify | `src/karenina/benchmark/benchmark.py` | Facade delegations, deprecations |
| Create | `src/karenina/benchmark/core/results_store.py` | Standalone ResultsStore class |
| Modify | `src/karenina/benchmark/core/__init__.py` | Export ResultsStore |
| Modify | `src/karenina/benchmark/task_eval/models.py` | Docstring updates for summary methods |
| Modify | `src/karenina/schemas/verification/config.py` | DeepJudgmentRubricCustomConfig, typed config field |
| Modify | `src/karenina/benchmark/verification/stages/helpers/deep_judgment_helpers.py` | Update dict access to model access for rubric config |
| Modify | `src/karenina/benchmark/verification/stages/core/base.py:302` | Update type annotation for rubric config |
| Modify | `src/karenina/benchmark/verification/runner.py:53` | Update type annotation for rubric config |
| Modify | `src/karenina/benchmark/verification/utils/task_helpers.py:186` | Update rubric config access |
| Modify | `src/karenina/cli/verify_config.py:101-104` | Accept raw dict and convert to model |
| Modify | `src/karenina/cli/interactive.py:204` | Convert dict to model when assigning |
| Modify | `tests/unit/benchmark/core/test_rubrics.py:561-590` | Update after get_questions_with_rubric rename |
| Modify | `src/karenina/schemas/results/verification_result_set.py` | __repr__, group_by_replicate |
| Modify | `src/karenina/schemas/results/template.py` | group_by_model by param |
| Modify | `src/karenina/schemas/results/rubric.py` | group_by_model by param |
| Modify | `src/karenina/schemas/results/judgment.py` | group_by_model by param |
| Create | `tests/unit/schemas/entities/test_higher_is_better.py` | Tests for higher_is_better changes |
| Create | `tests/unit/benchmark/core/test_results_store.py` | Tests for ResultsStore |
| Create | `tests/unit/schemas/test_api_design_fixes.py` | Tests for type improvements (repr, replicate, group_by_model, config) |
| Create | `tests/unit/benchmark/core/test_rubric_manager_api.py` | Tests for RubricManager API changes |
| Create | `tests/unit/benchmark/test_facade_api.py` | Tests for facade changes |

---

### Task 1: Unify `higher_is_better` across all trait types

**Files:**
- Modify: `src/karenina/schemas/entities/rubric.py:91-97` (LLMRubricTrait), `:198-201` (Regex), `:303-307` (Callable), `:527` (Metric), `:673-680` (Agentic), `:124-130` (LLM legacy validator), `:205-211` (Regex legacy), `:323-342` (Callable legacy), `:727-743` (Agentic legacy), `:939-968` (get_trait_directionalities)
- Create: `tests/unit/schemas/entities/test_higher_is_better.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for higher_is_better unification across all trait types."""

import pytest

from karenina.schemas.entities.rubric import (
    AgenticRubricTrait,
    CallableRubricTrait,
    LLMRubricTrait,
    MetricRubricTrait,
    RegexRubricTrait,
    Rubric,
)


@pytest.mark.unit
class TestHigherIsBetterUnification:
    """Test that all trait types support higher_is_better: bool | None."""

    def test_llm_trait_accepts_none(self):
        """LLMRubricTrait should accept None for higher_is_better."""
        trait = LLMRubricTrait(
            name="test",
            description="test trait",
            higher_is_better=None,
        )
        assert trait.higher_is_better is None

    def test_llm_trait_defaults_to_true(self):
        """LLMRubricTrait should default to True when not specified."""
        trait = LLMRubricTrait(name="test", description="test trait")
        assert trait.higher_is_better is True

    def test_regex_trait_accepts_none(self):
        """RegexRubricTrait should accept None for higher_is_better."""
        trait = RegexRubricTrait(
            name="test",
            description="test trait",
            pattern=r"\d+",
            higher_is_better=None,
        )
        assert trait.higher_is_better is None

    def test_regex_trait_defaults_to_true(self):
        """RegexRubricTrait should default to True when not specified."""
        trait = RegexRubricTrait(
            name="test", description="test trait", pattern=r"\d+"
        )
        assert trait.higher_is_better is True

    def test_callable_trait_accepts_none(self):
        """CallableRubricTrait should accept None for higher_is_better."""
        trait = CallableRubricTrait(
            name="test",
            description="test trait",
            callable_name="my_func",
            kind="boolean",
            higher_is_better=None,
        )
        assert trait.higher_is_better is None

    def test_callable_trait_defaults_to_true(self):
        """CallableRubricTrait should default to True when not specified."""
        trait = CallableRubricTrait(
            name="test",
            description="test trait",
            callable_name="my_func",
            kind="boolean",
        )
        assert trait.higher_is_better is True

    def test_agentic_trait_preserves_none(self):
        """AgenticRubricTrait should preserve explicit None (not coerce to True)."""
        trait = AgenticRubricTrait(
            name="test",
            description="test trait",
            kind="boolean",
            higher_is_better=None,
        )
        assert trait.higher_is_better is None

    def test_metric_trait_has_higher_is_better(self):
        """MetricRubricTrait should have higher_is_better field defaulting to None."""
        trait = MetricRubricTrait(
            name="test",
            description="test trait",
            metrics=["precision"],
        )
        assert trait.higher_is_better is None

    def test_metric_trait_accepts_true(self):
        """MetricRubricTrait should accept True for higher_is_better."""
        trait = MetricRubricTrait(
            name="test",
            description="test trait",
            metrics=["precision"],
            higher_is_better=True,
        )
        assert trait.higher_is_better is True

    def test_legacy_data_without_field_defaults_to_true(self):
        """Legacy data missing higher_is_better should still get True."""
        trait = LLMRubricTrait.model_validate(
            {"name": "test", "description": "test"}
        )
        assert trait.higher_is_better is True

    def test_get_trait_directionalities_includes_metric(self):
        """Rubric.get_trait_directionalities() should include metric traits."""
        rubric = Rubric(
            llm_traits=[
                LLMRubricTrait(name="clarity", description="clear", higher_is_better=True)
            ],
            metric_traits=[
                MetricRubricTrait(name="precision", description="prec", metrics=["precision"])
            ],
        )
        dirs = rubric.get_trait_directionalities()
        assert "clarity" in dirs
        assert dirs["clarity"] is True
        assert "precision" in dirs
        assert dirs["precision"] is None

    def test_get_trait_directionalities_none_values(self):
        """Traits with higher_is_better=None should appear as None in directionalities."""
        rubric = Rubric(
            llm_traits=[
                LLMRubricTrait(name="word_count", description="wc", higher_is_better=None)
            ],
        )
        dirs = rubric.get_trait_directionalities()
        assert dirs["word_count"] is None

    def test_agentic_template_kind_still_none(self):
        """AgenticRubricTrait with template kind should still have higher_is_better=None."""
        from pydantic import BaseModel

        class MyOutput(BaseModel):
            result: str

        trait = AgenticRubricTrait(
            name="test",
            description="test",
            kind=MyOutput,
            higher_is_better=None,
        )
        assert trait.higher_is_better is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/schemas/entities/test_higher_is_better.py -v`
Expected: Multiple FAILs (None coerced to True, MetricRubricTrait has no higher_is_better, metric excluded from directionalities)

- [ ] **Step 3: Implement higher_is_better changes**

In `src/karenina/schemas/entities/rubric.py`:

**LLMRubricTrait (line 91):** Change field type and add default:
```python
higher_is_better: bool | None = Field(
    default=True,
    description="Whether higher values indicate better performance. "
    "For boolean: True means True is good. "
    "For score: True means higher score is better. "
    "None means directionality does not apply.",
)
```

**LLMRubricTrait legacy validator (line 128):** Only fill when key is absent:
```python
if isinstance(values, dict) and "higher_is_better" not in values:
    values["higher_is_better"] = True
```

**RegexRubricTrait (line 198):** Same field change to `bool | None = Field(default=True, ...)`.

**RegexRubricTrait legacy validator (line 209):** Same fix, only when key absent.

**CallableRubricTrait (line 303):** Same field change to `bool | None = Field(default=True, ...)`.

**CallableRubricTrait legacy validator (line 328):** Same fix, only when key absent.

**AgenticRubricTrait legacy validator (line 741):** Same fix, only when key absent:
```python
if "higher_is_better" not in values or values.get("higher_is_better") is None:
```
Change to:
```python
if "higher_is_better" not in values:
```
(But preserve the template-kind skip logic that already exists earlier in the validator.)

**MetricRubricTrait (after line 527 fields):** Add field:
```python
higher_is_better: bool | None = Field(
    default=None,
    description="Whether higher metric values indicate better performance. "
    "None means directionality does not apply (default for metrics).",
)
```

**get_trait_directionalities (line 939-968):** Add metric traits and update docstring:
```python
def get_trait_directionalities(self) -> dict[str, bool | None]:
    """Get higher_is_better for all trait types.

    Returns:
        Dict mapping trait name to higher_is_better value. None indicates
        directionality does not apply (e.g. informational metrics,
        agentic template kinds).
    """
    directionalities = {}

    llm_trait: LLMRubricTrait
    for llm_trait in self.llm_traits:
        directionalities[llm_trait.name] = llm_trait.higher_is_better

    regex_trait: RegexRubricTrait
    for regex_trait in self.regex_traits:
        directionalities[regex_trait.name] = regex_trait.higher_is_better

    callable_trait: CallableRubricTrait
    for callable_trait in self.callable_traits:
        directionalities[callable_trait.name] = callable_trait.higher_is_better

    for agentic_trait in self.agentic_traits:
        directionalities[agentic_trait.name] = agentic_trait.higher_is_better

    for metric_trait in self.metric_traits:
        directionalities[metric_trait.name] = metric_trait.higher_is_better

    return directionalities
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/schemas/entities/test_higher_is_better.py -v`
Expected: All PASS

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `uv run pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/karenina/schemas/entities/rubric.py tests/unit/schemas/entities/test_higher_is_better.py
git commit -m "feat: unify higher_is_better as bool | None across all trait types

Add higher_is_better field to MetricRubricTrait (default None).
Widen type to bool | None on LLM, Regex, Callable traits with
default True. Fix legacy validators to not coerce explicit None.
Include metric traits in get_trait_directionalities()."
```

---

### Task 2: Add `Rubric.from_traits()` factory method

**Files:**
- Modify: `src/karenina/schemas/entities/rubric.py` (Rubric class, near line 811)
- Modify: `tests/unit/schemas/entities/test_higher_is_better.py` (add tests)

- [ ] **Step 1: Write failing test**

Append to `tests/unit/schemas/entities/test_higher_is_better.py`:

```python
@pytest.mark.unit
class TestRubricFromTraits:
    """Test Rubric.from_traits() factory method."""

    def test_from_mixed_traits(self):
        """from_traits should categorize a flat list into typed trait lists."""
        llm = LLMRubricTrait(name="clarity", description="clear")
        regex = RegexRubricTrait(name="has_number", description="num", pattern=r"\d+")
        metric = MetricRubricTrait(name="precision", description="prec", metrics=["precision"])

        rubric = Rubric.from_traits([llm, regex, metric])

        assert len(rubric.llm_traits) == 1
        assert rubric.llm_traits[0].name == "clarity"
        assert len(rubric.regex_traits) == 1
        assert rubric.regex_traits[0].name == "has_number"
        assert len(rubric.metric_traits) == 1
        assert rubric.metric_traits[0].name == "precision"
        assert len(rubric.callable_traits) == 0
        assert len(rubric.agentic_traits) == 0

    def test_from_empty_traits(self):
        """from_traits with empty list should produce empty Rubric."""
        rubric = Rubric.from_traits([])
        assert len(rubric.llm_traits) == 0
        assert len(rubric.regex_traits) == 0

    def test_from_none_returns_none(self):
        """from_traits with None should return None."""
        assert Rubric.from_traits(None) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/schemas/entities/test_higher_is_better.py::TestRubricFromTraits -v`
Expected: FAIL with AttributeError (no from_traits method)

- [ ] **Step 3: Implement from_traits**

Add classmethod to `Rubric` class in `src/karenina/schemas/entities/rubric.py`:

```python
@classmethod
def from_traits(
    cls,
    traits: list[LLMRubricTrait | RegexRubricTrait | CallableRubricTrait | MetricRubricTrait | AgenticRubricTrait] | None,
) -> "Rubric | None":
    """Create a Rubric from a flat list of traits, categorizing by type.

    Args:
        traits: Flat list of trait objects. If None, returns None.

    Returns:
        Rubric with traits sorted into typed lists, or None if input is None.
    """
    if traits is None:
        return None

    llm_traits = []
    regex_traits = []
    callable_traits = []
    metric_traits = []
    agentic_traits = []

    for trait in traits:
        if isinstance(trait, LLMRubricTrait):
            llm_traits.append(trait)
        elif isinstance(trait, RegexRubricTrait):
            regex_traits.append(trait)
        elif isinstance(trait, CallableRubricTrait):
            callable_traits.append(trait)
        elif isinstance(trait, MetricRubricTrait):
            metric_traits.append(trait)
        elif isinstance(trait, AgenticRubricTrait):
            agentic_traits.append(trait)

    return cls(
        llm_traits=llm_traits,
        regex_traits=regex_traits,
        callable_traits=callable_traits,
        metric_traits=metric_traits,
        agentic_traits=agentic_traits,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/schemas/entities/test_higher_is_better.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/karenina/schemas/entities/rubric.py tests/unit/schemas/entities/test_higher_is_better.py
git commit -m "feat: add Rubric.from_traits() factory for flat trait lists

Categorizes a flat list of mixed trait types into the typed trait
lists (llm_traits, regex_traits, etc.). Returns None for None input."
```

---

### Task 3: RubricManager API improvements

**Files:**
- Modify: `src/karenina/benchmark/core/rubrics.py:323-325` (rename), `:327-337` (set_global_rubric), add set_question_rubric, `:223-288` (validate_rubrics)
- Modify: `src/karenina/benchmark/benchmark.py:654-680` (facade delegations), `:393-395` (get_question_rubric)
- Create: `tests/unit/benchmark/core/test_rubric_manager_api.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for RubricManager API improvements."""

import pytest

from karenina.benchmark import Benchmark
from karenina.schemas.entities.rubric import (
    LLMRubricTrait,
    MetricRubricTrait,
    RegexRubricTrait,
    Rubric,
)


def _create_benchmark():
    """Create a fresh Benchmark for testing."""
    return Benchmark.create(name="test_rubric_api")


def _create_benchmark_with_question():
    """Create a Benchmark with one question."""
    b = Benchmark.create(name="test_rubric_api")
    b.add_question(question_id="q1", question="What is 2+2?", expected_answer="4")
    return b, "q1"


@pytest.mark.unit
class TestRubricManagerSetGlobalRubric:
    """Test set_global_rubric accepts both Rubric and list."""

    def test_accepts_rubric_object(self):
        """set_global_rubric should accept a Rubric object."""
        b = _create_benchmark()
        rubric = Rubric(
            llm_traits=[LLMRubricTrait(name="t1", description="d1")],
            regex_traits=[RegexRubricTrait(name="t2", description="d2", pattern=r"\d+")],
        )
        b._rubric_manager.set_global_rubric(rubric)
        result = b._rubric_manager.get_global_rubric()
        assert result is not None
        assert len(result.llm_traits) == 1
        assert len(result.regex_traits) == 1

    def test_accepts_trait_list(self):
        """set_global_rubric should accept a flat list of traits."""
        b = _create_benchmark()
        traits = [
            LLMRubricTrait(name="t1", description="d1"),
            RegexRubricTrait(name="t2", description="d2", pattern=r"\d+"),
        ]
        b._rubric_manager.set_global_rubric(traits)
        result = b._rubric_manager.get_global_rubric()
        assert result is not None
        assert len(result.llm_traits) == 1
        assert len(result.regex_traits) == 1


@pytest.mark.unit
class TestRubricManagerSetQuestionRubric:
    """Test set_question_rubric on RubricManager."""

    def test_set_question_rubric_with_rubric(self):
        """set_question_rubric should accept a Rubric object."""
        b, q_id = _create_benchmark_with_question()
        rubric = Rubric(
            llm_traits=[LLMRubricTrait(name="t1", description="d1")],
        )
        b._rubric_manager.set_question_rubric(q_id, rubric)
        traits = b._rubric_manager.get_question_rubric(q_id)
        assert traits is not None
        assert len(traits) == 1

    def test_set_question_rubric_replaces_existing(self):
        """set_question_rubric should clear existing then set new."""
        b, q_id = _create_benchmark_with_question()
        rubric1 = Rubric(
            llm_traits=[LLMRubricTrait(name="t1", description="d1")],
        )
        rubric2 = Rubric(
            regex_traits=[RegexRubricTrait(name="t2", description="d2", pattern=r"\d+")],
        )
        b._rubric_manager.set_question_rubric(q_id, rubric1)
        b._rubric_manager.set_question_rubric(q_id, rubric2)
        traits = b._rubric_manager.get_question_rubric(q_id)
        assert len(traits) == 1
        assert traits[0].name == "t2"


@pytest.mark.unit
class TestRubricManagerRename:
    """Test get_questions_with_rubric rename."""

    def test_get_question_ids_with_rubric_exists(self):
        """Renamed method should exist and return list[str]."""
        b = _create_benchmark()
        result = b._rubric_manager.get_question_ids_with_rubric()
        assert isinstance(result, list)

    def test_old_name_does_not_exist(self):
        """Old method name should not exist."""
        b = _create_benchmark()
        assert not hasattr(b._rubric_manager, "get_questions_with_rubric")


@pytest.mark.unit
class TestValidateRubricsErrorShape:
    """Test validate_rubrics returns dict-based errors."""

    def test_returns_dict_errors(self):
        """validate_rubrics errors should be list[dict[str, str]]."""
        b = _create_benchmark()
        b._rubric_manager.add_global_rubric_trait(
            LLMRubricTrait(name="", description="test")
        )
        valid, errors = b._rubric_manager.validate_rubrics()
        assert valid is False
        assert len(errors) > 0
        assert isinstance(errors[0], dict)
        assert "source" in errors[0]
        assert "error" in errors[0]
```

Note: Tests use `Benchmark.create()` inline (matching the pattern in `tests/unit/benchmark/core/test_rubrics.py`). No fixtures needed.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/benchmark/core/test_rubric_manager_api.py -v`
Expected: Multiple FAILs

- [ ] **Step 3: Implement RubricManager changes**

In `src/karenina/benchmark/core/rubrics.py`:

**Rename** `get_questions_with_rubric` (line 323) to `get_question_ids_with_rubric`.

**Modify** `set_global_rubric` (line 327) to accept `Rubric | list`:
```python
def set_global_rubric(
    self,
    rubric: Rubric | list[LLMRubricTrait | RegexRubricTrait | CallableRubricTrait | MetricRubricTrait | AgenticRubricTrait],
) -> None:
    """Set the global rubric, replacing any existing global traits.

    Args:
        rubric: A Rubric object or a flat list of trait objects.
    """
    self.clear_global_rubric()
    if isinstance(rubric, Rubric):
        traits = (
            list(rubric.llm_traits)
            + list(rubric.regex_traits)
            + list(rubric.callable_traits)
            + list(rubric.metric_traits)
            + list(rubric.agentic_traits)
        )
    else:
        traits = rubric
    for trait in traits:
        self.add_global_rubric_trait(trait)
```

**Add** `set_question_rubric` method:
```python
def set_question_rubric(
    self,
    question_id: str,
    rubric: Rubric | list[LLMRubricTrait | RegexRubricTrait | CallableRubricTrait | MetricRubricTrait | AgenticRubricTrait],
) -> None:
    """Set question-specific rubric, replacing any existing question rubric.

    Args:
        question_id: The question ID.
        rubric: A Rubric object or a flat list of trait objects.
    """
    self.remove_question_rubric(question_id)
    if isinstance(rubric, Rubric):
        traits = (
            list(rubric.llm_traits)
            + list(rubric.regex_traits)
            + list(rubric.callable_traits)
            + list(rubric.metric_traits)
            + list(rubric.agentic_traits)
        )
    else:
        traits = rubric
    for trait in traits:
        self.add_question_rubric_trait(question_id, trait)
```

**Modify** `validate_rubrics` (line 223) return type and error format. Change return type to `tuple[bool, list[dict[str, str]]]`. Replace each `errors.append("message")` with `errors.append({"source": scope_identifier, "error": message})`. The `source` should be the trait name for trait-level errors, or `"global"` / `f"question:{q_id}"` for scope-level errors.

- [ ] **Step 4: Update callers of renamed method and changed return types**

The rename from `get_questions_with_rubric` to `get_question_ids_with_rubric` on `RubricManager` affects:
- `tests/unit/benchmark/core/test_rubrics.py:561-590`: Update test calls to use the new name.

The `validate_rubrics()` return type change affects:
- `src/karenina/benchmark/benchmark.py:698`: Update facade return type annotation from `tuple[bool, list[str]]` to `tuple[bool, list[dict[str, str]]]`.
- Any existing tests in `tests/unit/benchmark/core/test_rubrics.py` or `tests/unit/benchmark/test_rubrics_issues.py` that assert on error item types.

Grep for both `get_questions_with_rubric` and `validate_rubrics` across the test suite to find all callers.

- [ ] **Step 5: Simplify facade delegations**

In `src/karenina/benchmark/benchmark.py`:

**Replace** `set_global_rubric` (lines 654-666):
```python
def set_global_rubric(self, rubric: Rubric) -> None:
    """Set the complete global rubric (replaces existing)."""
    self._rubric_manager.set_global_rubric(rubric)
```

**Replace** `set_question_rubric` (lines 668-680):
```python
def set_question_rubric(self, question_id: str, rubric: Rubric) -> None:
    """Set the complete question-specific rubric (replaces existing)."""
    self._rubric_manager.set_question_rubric(question_id, rubric)
```

**Add** `get_question_rubric` facade method (near the other rubric methods):
```python
def get_question_rubric(self, question_id: str) -> Rubric | None:
    """Get the question-specific rubric for a question."""
    traits = self._rubric_manager.get_question_rubric(question_id)
    return Rubric.from_traits(traits)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/unit/benchmark/core/test_rubric_manager_api.py -v`
Expected: All PASS

- [ ] **Step 7: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All pass. Watch for any existing tests that called `RubricManager.get_questions_with_rubric()` by the old name, or that check `validate_rubrics()` error format as plain strings.

- [ ] **Step 8: Commit**

```bash
git add src/karenina/benchmark/core/rubrics.py src/karenina/benchmark/benchmark.py tests/unit/benchmark/core/test_rubric_manager_api.py
git commit -m "feat: improve RubricManager API and facade delegations

RubricManager.set_global_rubric accepts Rubric or list.
Add RubricManager.set_question_rubric with same flexibility.
Rename get_questions_with_rubric to get_question_ids_with_rubric.
Unify validate_rubrics error shape to list[dict[str, str]].
Add get_question_rubric to facade, simplify set_* delegations."
```

---

### Task 4: Create ResultsStore

**Files:**
- Create: `src/karenina/benchmark/core/results_store.py`
- Modify: `src/karenina/benchmark/core/__init__.py`
- Create: `tests/unit/benchmark/core/test_results_store.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for standalone ResultsStore."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from karenina.benchmark.core.results_store import ResultsStore
from karenina.schemas.results.verification_result_set import VerificationResultSet


def _make_result(question_id: str, passed: bool = True) -> MagicMock:
    """Create a mock VerificationResult."""
    result = MagicMock()
    result.metadata = MagicMock()
    result.metadata.question_id = question_id
    result.metadata.passed = passed
    result.model_dump = MagicMock(return_value={
        "metadata": {"question_id": question_id, "passed": passed},
    })
    return result


def _make_result_set(results: list) -> VerificationResultSet:
    """Create a VerificationResultSet from mock results."""
    return VerificationResultSet(results=results)


@pytest.mark.unit
class TestResultsStoreAdd:
    """Test adding results to the store."""

    def test_add_with_run_name(self):
        """add() should store results under the given run name."""
        store = ResultsStore()
        rs = _make_result_set([_make_result("q1")])
        store.add(rs, run_name="run_1")
        assert store.has_results(run_name="run_1")

    def test_add_auto_generates_run_name(self):
        """add() without run_name should auto-generate one."""
        store = ResultsStore()
        rs = _make_result_set([_make_result("q1")])
        store.add(rs)
        runs = store.get_all_runs()
        assert len(runs) == 1


@pytest.mark.unit
class TestResultsStoreQuery:
    """Test querying results."""

    def test_get_by_run(self):
        """get_by_run should return the stored VerificationResultSet."""
        store = ResultsStore()
        rs = _make_result_set([_make_result("q1"), _make_result("q2")])
        store.add(rs, run_name="run_1")
        result = store.get_by_run("run_1")
        assert isinstance(result, VerificationResultSet)
        assert len(result.results) == 2

    def test_get_by_question(self):
        """get_by_question should return results keyed by run name."""
        store = ResultsStore()
        store.add(_make_result_set([_make_result("q1")]), run_name="run_1")
        store.add(_make_result_set([_make_result("q1")]), run_name="run_2")
        result = store.get_by_question("q1")
        assert "run_1" in result
        assert "run_2" in result

    def test_get_latest(self):
        """get_latest should return most recent result per question."""
        store = ResultsStore()
        store.add(_make_result_set([_make_result("q1")]), run_name="run_1")
        store.add(_make_result_set([_make_result("q1")]), run_name="run_2")
        latest = store.get_latest("q1")
        assert "q1" in latest

    def test_has_results_empty(self):
        """has_results should return False when empty."""
        store = ResultsStore()
        assert store.has_results() is False

    def test_get_all_runs_ordered(self):
        """get_all_runs should return run names in insertion order."""
        store = ResultsStore()
        store.add(_make_result_set([_make_result("q1")]), run_name="b")
        store.add(_make_result_set([_make_result("q1")]), run_name="a")
        assert store.get_all_runs() == ["b", "a"]


@pytest.mark.unit
class TestResultsStoreClear:
    """Test clearing results."""

    def test_clear_all(self):
        """clear() with no args should remove everything."""
        store = ResultsStore()
        store.add(_make_result_set([_make_result("q1")]), run_name="run_1")
        store.clear()
        assert store.has_results() is False

    def test_clear_by_run(self):
        """clear(run_name=...) should remove only that run."""
        store = ResultsStore()
        store.add(_make_result_set([_make_result("q1")]), run_name="run_1")
        store.add(_make_result_set([_make_result("q1")]), run_name="run_2")
        store.clear(run_name="run_1")
        assert not store.has_results(run_name="run_1")
        assert store.has_results(run_name="run_2")


@pytest.mark.unit
class TestResultsStoreSummary:
    """Test summary and export methods."""

    def test_get_summary(self):
        """get_summary should return dict with result counts."""
        store = ResultsStore()
        store.add(_make_result_set([_make_result("q1"), _make_result("q2")]), run_name="r1")
        summary = store.get_summary()
        assert isinstance(summary, dict)
        assert summary["total_results"] >= 2

    def test_get_statistics_by_run(self):
        """get_statistics_by_run should return per-run stats."""
        store = ResultsStore()
        store.add(_make_result_set([_make_result("q1")]), run_name="r1")
        store.add(_make_result_set([_make_result("q1"), _make_result("q2")]), run_name="r2")
        stats = store.get_statistics_by_run()
        assert "r1" in stats
        assert "r2" in stats

    def test_export_returns_serializable(self):
        """export should return a JSON-serializable dict."""
        store = ResultsStore()
        store.add(_make_result_set([_make_result("q1")]), run_name="r1")
        exported = store.export()
        assert isinstance(exported, dict)
        json.dumps(exported)  # should not raise

    def test_export_to_file_and_load(self, tmp_path):
        """export_to_file and from_file should round-trip."""
        store = ResultsStore()
        store.add(_make_result_set([_make_result("q1")]), run_name="r1")
        path = tmp_path / "results.json"
        store.export_to_file(path)
        loaded = ResultsStore.from_file(path)
        assert loaded.has_results(run_name="r1")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/benchmark/core/test_results_store.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement ResultsStore**

Create `src/karenina/benchmark/core/results_store.py`. The implementation should be self-contained (no dependency on `BenchmarkBase`). Use `ResultsManager` (in `results.py`) as reference for method behavior, but the store holds its own `dict[str, VerificationResultSet]` keyed by run name.

Key implementation notes:
- Internal storage: `_runs: dict[str, VerificationResultSet]` (ordered dict by insertion)
- `add()`: generate timestamped run name if not provided
- `get_latest()`: iterate runs in reverse insertion order, collect first result per question
- `get_by_question()`: iterate all runs, collect results matching the question_id
- `get_by_run()`: direct dict lookup, raise `KeyError` if not found
- `has_results()`: check filters against stored data
- `clear()`: support filtering by run_name and question_ids
- `get_summary()`, `get_statistics_by_run()`, `export()`, `export_to_file()`, `from_file()`: implement based on ResultsManager patterns

- [ ] **Step 4: Update __init__.py**

Add `ResultsStore` to exports in `src/karenina/benchmark/core/__init__.py`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/benchmark/core/test_results_store.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/karenina/benchmark/core/results_store.py src/karenina/benchmark/core/__init__.py tests/unit/benchmark/core/test_results_store.py
git commit -m "feat: add standalone ResultsStore for cross-run result management

ResultsStore accumulates VerificationResultSets keyed by run name,
with query methods for filtering by question, run, or recency.
Independent of Benchmark facade."
```

---

### Task 5: Deprecate facade result methods

**Files:**
- Modify: `src/karenina/benchmark/benchmark.py:911-980`
- Create: `tests/unit/benchmark/test_facade_api.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for facade API changes (deprecations and new methods)."""

import warnings

import pytest

from karenina.benchmark import Benchmark
from karenina.schemas.entities.rubric import LLMRubricTrait, Rubric


def _create_benchmark():
    return Benchmark.create(name="test_facade_api")


def _create_benchmark_with_question():
    b = Benchmark.create(name="test_facade_api")
    b.add_question(question_id="q1", question="What is 2+2?", expected_answer="4")
    return b


@pytest.mark.unit
class TestFacadeResultDeprecations:
    """Test that facade result methods emit deprecation warnings."""

    def test_store_verification_results_warns(self):
        """store_verification_results should emit DeprecationWarning."""
        b = _create_benchmark()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            b.store_verification_results({})
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "ResultsStore" in str(w[0].message)

    def test_get_verification_results_warns(self):
        """get_verification_results should emit DeprecationWarning."""
        b = _create_benchmark()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            b.get_verification_results()
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)


@pytest.mark.unit
class TestFacadeGetQuestionRubric:
    """Test get_question_rubric facade method."""

    def test_returns_none_when_no_rubric(self):
        """get_question_rubric returns None when no rubric set."""
        b = _create_benchmark_with_question()
        assert b.get_question_rubric("q1") is None

    def test_returns_rubric_when_set(self):
        """get_question_rubric returns Rubric after setting one."""
        b = _create_benchmark_with_question()
        rubric = Rubric(llm_traits=[LLMRubricTrait(name="t", description="d")])
        b.set_question_rubric("q1", rubric)
        result = b.get_question_rubric("q1")
        assert isinstance(result, Rubric)
        assert len(result.llm_traits) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/benchmark/test_facade_api.py -v`
Expected: FAIL (no DeprecationWarning emitted, no get_question_rubric method)

- [ ] **Step 3: Add deprecation warnings to facade result methods**

In `src/karenina/benchmark/benchmark.py`, add `warnings.warn()` to each of the 10 result methods (lines 911-980). Example pattern:

```python
def store_verification_results(self, results, run_name=None):
    """Store verification results. Deprecated: use ResultsStore.add() instead."""
    warnings.warn(
        "store_verification_results is deprecated. Use ResultsStore.add() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return self._results_manager.store_verification_results(results, run_name)
```

Add `import warnings` at the top of the file if not already present.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/benchmark/test_facade_api.py -v`
Expected: All PASS

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All pass. Some existing tests may trigger deprecation warnings; this is expected. If any tests use `warnings.filterwarnings("error")`, they may need updating.

- [ ] **Step 6: Commit**

```bash
git add src/karenina/benchmark/benchmark.py tests/unit/benchmark/test_facade_api.py
git commit -m "feat: deprecate facade result methods in favor of ResultsStore

All 10 result storage/query methods on Benchmark now emit
DeprecationWarning pointing to ResultsStore equivalents.
Add get_question_rubric facade method."
```

---

### Task 6: Docstring and naming consistency fixes

**Files:**
- Modify: `src/karenina/benchmark/task_eval/models.py:530-531,571-572`
- Modify: `src/karenina/benchmark/core/rubrics.py` (validate_rubrics, already done in Task 3 if combined; if not, do here)

- [ ] **Step 1: Update TaskEvalResult docstrings**

In `src/karenina/benchmark/task_eval/models.py`:

**`summary()` (line 531):** Replace docstring:
```python
def summary(self) -> str:
    """Return a concise summary of the evaluation results.

    Aggregates across global evaluation and all per-step evaluations.
    """
```

**`summary_compact()` (line 572):** Replace docstring:
```python
def summary_compact(self) -> str:
    """Return a very compact one-line summary.

    Reports global evaluation statistics only (excludes per-step evaluations).
    """
```

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All pass (docstring-only changes)

- [ ] **Step 3: Commit**

```bash
git add src/karenina/benchmark/task_eval/models.py
git commit -m "docs: clarify scope of summary() vs summary_compact() in TaskEvalResult

summary() aggregates global + per-step evaluations.
summary_compact() reports global evaluation statistics only."
```

---

### Task 7: Type improvements (repr, config, group_by_model, group_by_replicate)

**Files:**
- Modify: `src/karenina/schemas/results/verification_result_set.py:404-425` (group_by_replicate), `:1199+` (__repr__)
- Modify: `src/karenina/schemas/results/template.py:648` (group_by_model)
- Modify: `src/karenina/schemas/results/rubric.py:633` (group_by_model)
- Modify: `src/karenina/schemas/results/judgment.py:585` (group_by_model)
- Modify: `src/karenina/schemas/verification/config.py:51,158` (DeepJudgmentRubricCustomConfig)
- Create: `tests/unit/schemas/test_api_design_fixes.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for type improvements: repr, config, group_by_model, group_by_replicate."""

from typing import Literal
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from karenina.schemas.results.verification_result_set import VerificationResultSet
from karenina.schemas.verification.config import (
    DeepJudgmentRubricCustomConfig,
    DeepJudgmentTraitConfig,
    VerificationConfig,
)


def _make_result(question_id="q1", answering_model="gpt-4", parsing_model="gpt-4", replicate=1):
    """Create a mock VerificationResult with metadata."""
    result = MagicMock()
    result.metadata = MagicMock()
    result.metadata.question_id = question_id
    result.metadata.answering_model = answering_model
    result.metadata.parsing_model = parsing_model
    result.metadata.replicate = replicate
    result.metadata.mcp_servers = None
    return result


@pytest.mark.unit
class TestVerificationResultSetRepr:
    """Test simplified __repr__."""

    def test_repr_empty(self):
        """Empty result set repr should be concise."""
        rs = VerificationResultSet(results=[])
        r = repr(rs)
        assert "VerificationResultSet" in r
        assert "0" in r

    def test_repr_with_results(self):
        """Result set repr should show count without calling get_summary."""
        rs = VerificationResultSet(results=[_make_result(), _make_result()])
        r = repr(rs)
        assert "2" in r
        assert "VerificationResultSet" in r


@pytest.mark.unit
class TestGroupByReplicate:
    """Test group_by_replicate preserves None."""

    def test_none_replicate_preserved(self):
        """Results with replicate=None should be grouped under None key."""
        rs = VerificationResultSet(results=[
            _make_result(replicate=None),
            _make_result(replicate=1),
        ])
        groups = rs.group_by_replicate()
        assert None in groups
        assert 1 in groups
        assert len(groups[None].results) == 1
        assert len(groups[1].results) == 1


@pytest.mark.unit
class TestDeepJudgmentRubricCustomConfig:
    """Test typed deep judgment rubric config."""

    def test_valid_config(self):
        """DeepJudgmentRubricCustomConfig should validate correctly."""
        config = DeepJudgmentRubricCustomConfig(
            **{
                "global": {"trait1": DeepJudgmentTraitConfig(enabled=True)},
                "question_specific": {},
            }
        )
        assert len(config.global_traits) == 1

    def test_custom_mode_requires_config(self):
        """VerificationConfig with mode=custom should require config field."""
        with pytest.raises(ValidationError, match="deep_judgment_rubric_config"):
            VerificationConfig(
                deep_judgment_rubric_mode="custom",
                deep_judgment_rubric_config=None,
                answering_models=[{"provider": "anthropic", "model": "test"}],
                parsing_models=[{"provider": "anthropic", "model": "test"}],
            )


@pytest.mark.unit
class TestViewGroupByModel:
    """Test view-level group_by_model accepts and uses 'by' parameter."""

    def test_template_results_by_answering(self):
        """TemplateResults.group_by_model(by='answering') groups by answering model."""
        from karenina.schemas.results.template import TemplateResults
        rs = VerificationResultSet(results=[
            _make_result(answering_model="gpt-4", parsing_model="claude-3"),
            _make_result(answering_model="claude-3", parsing_model="claude-3"),
        ])
        tr = rs.get_template_results()
        groups = tr.group_by_model(by="answering")
        assert "gpt-4" in groups
        assert "claude-3" in groups

    def test_template_results_by_parsing(self):
        """TemplateResults.group_by_model(by='parsing') groups by parsing model."""
        from karenina.schemas.results.template import TemplateResults
        rs = VerificationResultSet(results=[
            _make_result(answering_model="gpt-4", parsing_model="claude-3"),
            _make_result(answering_model="gpt-4", parsing_model="gpt-4"),
        ])
        tr = rs.get_template_results()
        groups = tr.group_by_model(by="parsing")
        assert "claude-3" in groups
        assert "gpt-4" in groups

    def test_rubric_results_by_param(self):
        """RubricResults.group_by_model should accept 'by' parameter."""
        import inspect
        from karenina.schemas.results.rubric import RubricResults
        sig = inspect.signature(RubricResults.group_by_model)
        assert "by" in sig.parameters

    def test_judgment_results_by_param(self):
        """JudgmentResults.group_by_model should accept 'by' parameter."""
        import inspect
        from karenina.schemas.results.judgment import JudgmentResults
        sig = inspect.signature(JudgmentResults.group_by_model)
        assert "by" in sig.parameters
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/schemas/test_api_design_fixes.py -v`
Expected: Multiple FAILs

- [ ] **Step 3: Implement __repr__ simplification**

In `src/karenina/schemas/results/verification_result_set.py`, replace the `__repr__` method (line 1199+) with:

```python
def __repr__(self) -> str:
    n = len(self.results)
    return f"VerificationResultSet({n} result{'s' if n != 1 else ''})"
```

- [ ] **Step 4: Implement group_by_replicate None preservation**

In `src/karenina/schemas/results/verification_result_set.py` (line 404-425), change:
- Return type from `dict[int, VerificationResultSet]` to `dict[int | None, VerificationResultSet]`
- Replace `result.metadata.replicate or 0` with `result.metadata.replicate`

- [ ] **Step 5: Implement DeepJudgmentRubricCustomConfig**

In `src/karenina/schemas/verification/config.py`:

Add new model after `DeepJudgmentTraitConfig` (after line 65):

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

Change `deep_judgment_rubric_config` field (line 158) to:
```python
deep_judgment_rubric_config: DeepJudgmentRubricCustomConfig | None = None
```

Add cross-field validator:
```python
@model_validator(mode="after")
def validate_custom_mode_has_config(self) -> "VerificationConfig":
    """Require deep_judgment_rubric_config when mode is custom."""
    if self.deep_judgment_rubric_mode == "custom" and self.deep_judgment_rubric_config is None:
        raise ValueError("deep_judgment_rubric_config is required when deep_judgment_rubric_mode is 'custom'")
    return self
```

Update the `from_overrides()` parameter type (line 681) to `DeepJudgmentRubricCustomConfig | dict[str, Any] | None`. In the method body, convert dict to model if needed.

**Update all callers that use dict-style access on `deep_judgment_rubric_config`:**

- `src/karenina/schemas/verification/config.py:535-536`: Change `.get("global", {})` and `.get("question_specific", {})` to `.global_traits` and `.question_specific`
- `src/karenina/benchmark/verification/stages/helpers/deep_judgment_helpers.py:88-97`: Change `config_dict["question_specific"][question_id]` to use model attributes
- `src/karenina/benchmark/verification/stages/core/base.py:302`: Change type annotation from `dict[str, Any] | None` to `DeepJudgmentRubricCustomConfig | None`
- `src/karenina/benchmark/verification/runner.py:53`: Same type annotation update
- `src/karenina/benchmark/verification/utils/task_helpers.py:186`: Update attribute access
- `src/karenina/cli/verify_config.py:101-104`: When loading from JSON, wrap raw dict in `DeepJudgmentRubricCustomConfig.model_validate()`
- `src/karenina/cli/interactive.py:204`: Same conversion when assigning

**Update existing tests:**
- `tests/unit/schemas/test_verification_config.py`: Tests that pass raw dicts as `deep_judgment_rubric_config` need to pass `DeepJudgmentRubricCustomConfig` or use `model_validate`

- [ ] **Step 6: Add `by` parameter to view-level group_by_model**

In each view class, add the `by` parameter and implement key-building logic matching `VerificationResultSet.group_by_model()`:

**`src/karenina/schemas/results/template.py:648`:**
```python
def group_by_model(self, by: Literal["answering", "parsing", "both"] = "answering") -> dict[str, "TemplateResults"]:
```

**`src/karenina/schemas/results/rubric.py:633`:**
```python
def group_by_model(self, by: Literal["answering", "parsing", "both"] = "answering") -> dict[str, "RubricResults"]:
```

**`src/karenina/schemas/results/judgment.py:585`:**
```python
def group_by_model(self, by: Literal["answering", "parsing", "both"] = "answering") -> dict[str, "JudgmentResults"]:
```

For each, implement the key-building logic from `VerificationResultSet.group_by_model()` (lines 370-402 of verification_result_set.py), including MCP server info for answering keys.

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/unit/schemas/test_api_design_fixes.py -v`
Expected: All PASS

- [ ] **Step 8: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All pass. Watch for:
- Tests that depend on `__repr__` output format
- Tests that check `group_by_replicate` returns `dict[int, ...]`
- Tests that use `deep_judgment_rubric_config` as a raw dict
- Tests that check existing `group_by_model` without `by` parameter (should still work due to default)

- [ ] **Step 9: Commit**

```bash
git add src/karenina/schemas/results/verification_result_set.py src/karenina/schemas/results/template.py src/karenina/schemas/results/rubric.py src/karenina/schemas/results/judgment.py src/karenina/schemas/verification/config.py tests/unit/schemas/test_api_design_fixes.py
git commit -m "feat: type improvements for result sets and verification config

Simplify VerificationResultSet.__repr__ (no more get_summary call).
Preserve None in group_by_replicate return type.
Add DeepJudgmentRubricCustomConfig typed model for custom mode.
Add 'by' parameter to view-level group_by_model methods."
```

---

### Task 8: Remove duplicate server config router mount

**Files:**
- Modify: `karenina-server/src/karenina_server/server.py:541`

**Note:** The server is at `/Users/carli/Projects/karenina-salvage/karenina-server/`, a sibling submodule outside this worktree. This task must be done in the main repo or a server worktree.

- [ ] **Step 1: Verify GUI uses correct paths**

Grep the GUI source for `/api/config/v2/config` to confirm no references to the redundant mount:

```bash
cd /Users/carli/Projects/karenina-salvage/karenina-gui && grep -r "api/config/v2/config" src/
```

Expected: No matches. The GUI uses `/api/v2/config/...` (correct paths via the `/api` mount).

- [ ] **Step 2: Remove duplicate mount**

In `/Users/carli/Projects/karenina-salvage/karenina-server/src/karenina_server/server.py`, remove line 541:
```python
# Remove this line:
app.include_router(config_router, prefix="/api/config")
```

Keep line 542:
```python
app.include_router(config_router, prefix="/api")  # V2 endpoints at /api/v2/config/...
```

- [ ] **Step 3: Run server tests**

```bash
cd /Users/carli/Projects/karenina-salvage/karenina-server && uv run pytest tests/ -x -q
```

Expected: All pass

- [ ] **Step 4: Commit**

```bash
cd /Users/carli/Projects/karenina-salvage/karenina-server
git add src/karenina_server/server.py
git commit -m "fix: remove duplicate config router mount creating redundant endpoints

Config router was mounted at both /api/config and /api, producing
22 endpoints instead of 11. The /api/config mount created redundant
/api/config/v2/config/... paths."
```

---

## Task Dependency Summary

```
Task 1 (higher_is_better) ─► Task 2 (from_traits) ─► Task 3 (RubricManager)
                                                          │
Task 6 (docstrings) ◄── no deps                          ▼
Task 7 (type improvements) ◄── no deps        Task 4 (ResultsStore) ─► Task 5 (deprecations)
Task 8 (server) ◄── no deps
```

Tasks 1-5 are sequential. Tasks 6, 7, 8 are independent and can be parallelized with each other (and with tasks 4-5 if desired).
