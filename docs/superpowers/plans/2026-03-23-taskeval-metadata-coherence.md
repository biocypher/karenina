# TaskEval Metadata Coherence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 11 metadata divergences and functional bugs in TaskEval so its results are structurally compatible with Benchmark results for cross-path analysis.

**Architecture:** All fixes center on threading missing parameters through TaskEval's evaluation chain (`evaluate()` -> `_evaluate_global()`/`_evaluate_step()` -> `_evaluate_step_internal()` -> `_run_evaluation_loop()` -> `_evaluate_and_store()` -> `_evaluate()` -> `run_single_model_verification()`). A new `taskeval` adapter interface is registered as a no-op sentinel. Exception handling is narrowed to domain exceptions with failure tracking.

**Tech Stack:** Python, Pydantic v2, pytest, monkeypatch

**Spec:** `docs/superpowers/specs/2026-03-23-taskeval-metadata-coherence-design.md`

**Note on question_id length:** Issue 165 removes MD5 force-hashing of question_ids, allowing human-readable IDs. The storage layer uses `VARCHAR(32)` for `question_id`. TaskEval results are used in-memory and not written to the database, so this is not a problem. If database persistence for TaskEval is added later, the column width will need to increase.

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `src/karenina/adapters/taskeval/__init__.py` | Empty package init |
| Create | `src/karenina/adapters/taskeval/registration.py` | Register no-op `taskeval` interface |
| Modify | `src/karenina/adapters/registry.py:348-371` | Add taskeval to `_load_builtins()` |
| Modify | `src/karenina/benchmark/task_eval/models.py:30-36` | Add `failed_questions` field to `StepEval` |
| Modify | `src/karenina/benchmark/task_eval/models.py:145-148` | Fix rubric-only `passed_traces` logic |
| Modify | `src/karenina/benchmark/task_eval/task_eval.py:306-335` | Expand `evaluate()` signature |
| Modify | `src/karenina/benchmark/task_eval/task_eval.py:337-402` | Thread params through `_evaluate_global()`, `_evaluate_step()`, `_evaluate_step_internal()` |
| Modify | `src/karenina/benchmark/task_eval/task_eval.py:408-418` | Add `step_dynamic_rubrics` to `_get_available_step_ids()` |
| Modify | `src/karenina/benchmark/task_eval/task_eval.py:523-615` | Thread params through `_run_evaluation_loop()`, track replicate index |
| Modify | `src/karenina/benchmark/task_eval/task_eval.py:617-665` | Thread params through `_evaluate_and_store()`, narrow exception catch |
| Modify | `src/karenina/benchmark/task_eval/task_eval.py:670-773` | Thread params through `_evaluate()`, remove mock model and MD5 hashing |
| Create | `tests/unit/benchmark/test_task_eval_metadata.py` | New test file for all metadata coherence fixes |

---

### Task 1: Register `taskeval` Adapter Interface

**Files:**
- Create: `src/karenina/adapters/taskeval/__init__.py`
- Create: `src/karenina/adapters/taskeval/registration.py`
- Modify: `src/karenina/adapters/registry.py:348-371`
- Test: `tests/unit/benchmark/test_task_eval_metadata.py`

- [ ] **Step 1: Write test that `taskeval` interface is registered and ModelConfig validates**

Create `tests/unit/benchmark/test_task_eval_metadata.py`:

```python
"""Tests for TaskEval metadata coherence fixes (issues 024, 114, 115, 117, 160, 165, 166, 168, 179)."""

import pytest

from karenina.schemas.config import ModelConfig


@pytest.mark.unit
class TestTaskEvalInterface:
    """Issue 166: taskeval interface registration."""

    def test_taskeval_interface_registered(self):
        """ModelConfig with interface='taskeval' should not raise."""
        config = ModelConfig(
            id="taskeval_user_provided",
            model_name="user-provided",
            model_provider="user-provided",
            interface="taskeval",
        )
        assert config.interface == "taskeval"

    def test_taskeval_sentinel_fields(self):
        """Sentinel model has expected field values."""
        config = ModelConfig(
            id="taskeval_user_provided",
            model_name="user-provided",
            model_provider="user-provided",
            interface="taskeval",
        )
        assert config.model_name == "user-provided"
        assert config.model_provider == "user-provided"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/carli/Projects/karenina-salvage/karenina/.claude/worktrees/taskeval-harmonize && uv run pytest tests/unit/benchmark/test_task_eval_metadata.py::TestTaskEvalInterface -v`
Expected: FAIL with `ValueError: Unknown interface 'taskeval'`

- [ ] **Step 3: Create the taskeval adapter package**

Create `src/karenina/adapters/taskeval/__init__.py`:

```python
"""TaskEval adapter: no-op interface for pre-collected output evaluation."""
```

Create `src/karenina/adapters/taskeval/registration.py`:

```python
"""Registration module for the taskeval interface.

Registers a no-op 'taskeval' interface with the AdapterRegistry. This interface
exists solely as a valid sentinel for ModelConfig when TaskEval evaluates
pre-collected outputs. No LLM invocation occurs; the model is never called.
"""

from karenina.adapters.registry import AdapterAvailability, AdapterRegistry, AdapterSpec


def _check_availability() -> AdapterAvailability:
    """TaskEval interface is always available."""
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

- [ ] **Step 4: Add taskeval to `_load_builtins()` in registry.py**

In `src/karenina/adapters/registry.py`, after the langchain_deep_agents block (line 371), add:

```python
        try:
            from karenina.adapters.taskeval import registration as _te  # noqa: F401
        except ImportError:
            logger.debug("TaskEval registration module not available")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/carli/Projects/karenina-salvage/karenina/.claude/worktrees/taskeval-harmonize && uv run pytest tests/unit/benchmark/test_task_eval_metadata.py::TestTaskEvalInterface -v`
Expected: PASS

- [ ] **Step 6: Run full test suite to check for regressions**

Run: `cd /Users/carli/Projects/karenina-salvage/karenina/.claude/worktrees/taskeval-harmonize && uv run pytest tests/ -x -q`
Expected: All tests pass

- [ ] **Step 7: Commit**

```bash
git add src/karenina/adapters/taskeval/ src/karenina/adapters/registry.py tests/unit/benchmark/test_task_eval_metadata.py
git commit -m "feat(adapters): register no-op taskeval interface for sentinel ModelConfig"
```

---

### Task 2: Add `failed_questions` to StepEval (Issue 117, Part 1)

**Files:**
- Modify: `src/karenina/benchmark/task_eval/models.py:30-36`
- Test: `tests/unit/benchmark/test_task_eval_metadata.py`

- [ ] **Step 1: Write test for `failed_questions` field**

Append to `tests/unit/benchmark/test_task_eval_metadata.py`:

```python
from karenina.benchmark.task_eval import StepEval


@pytest.mark.unit
class TestStepEvalFailedQuestions:
    """Issue 117: StepEval tracks failed questions."""

    def test_failed_questions_defaults_empty(self):
        """failed_questions defaults to empty dict."""
        step = StepEval()
        assert step.failed_questions == {}

    def test_failed_questions_records_errors(self):
        """failed_questions stores error messages by question_id."""
        step = StepEval()
        step.failed_questions["q1"] = ["Some error"]
        step.failed_questions["q1"].append("Another error")
        assert len(step.failed_questions["q1"]) == 2
        assert step.failed_questions["q1"][0] == "Some error"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/carli/Projects/karenina-salvage/karenina/.claude/worktrees/taskeval-harmonize && uv run pytest tests/unit/benchmark/test_task_eval_metadata.py::TestStepEvalFailedQuestions -v`
Expected: FAIL with `ValidationError` (no `failed_questions` field)

- [ ] **Step 3: Add `failed_questions` field to StepEval**

In `src/karenina/benchmark/task_eval/models.py`, after the `verification_results` field (line 36), add:

```python
    failed_questions: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Questions that failed evaluation, keyed by question_id with error messages",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/carli/Projects/karenina-salvage/karenina/.claude/worktrees/taskeval-harmonize && uv run pytest tests/unit/benchmark/test_task_eval_metadata.py::TestStepEvalFailedQuestions -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/karenina/benchmark/task_eval/models.py tests/unit/benchmark/test_task_eval_metadata.py
git commit -m "feat(task_eval): add failed_questions field to StepEval for error tracking"
```

---

### Task 3: Fix Rubric-Only `success_rate` (Issue 024)

**Files:**
- Modify: `src/karenina/benchmark/task_eval/models.py:145-148`
- Test: `tests/unit/benchmark/test_task_eval_metadata.py`

- [ ] **Step 1: Write tests for rubric-only pass logic**

Append to `tests/unit/benchmark/test_task_eval_metadata.py`:

```python
from karenina.schemas.verification import VerificationResult
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.schemas.verification.result_components import (
    VerificationResultMetadata,
    VerificationResultRubric,
    VerificationResultTemplate,
)


def _make_metadata(question_id: str = "test_q") -> VerificationResultMetadata:
    """Helper to create test metadata."""
    return VerificationResultMetadata(
        question_id=question_id,
        template_id="test_t",
        completed_without_errors=True,
        question_text="test question",
        answering=ModelIdentity(interface="mock", model_name="mock"),
        parsing=ModelIdentity(interface="mock", model_name="mock"),
        execution_time=0.1,
        timestamp="2026-01-01T00:00:00",
        result_id="abcd1234abcd1234",
    )


@pytest.mark.unit
class TestSuccessRateRubricOnly:
    """Issue 024: success_rate should not be zero in rubric_only mode when traits pass."""

    def test_rubric_only_all_pass(self):
        """success_rate > 0 when all rubric traits pass and no template verification."""
        rubric = VerificationResultRubric(
            rubric_evaluation_performed=True,
            verify_rubric={"trait_a": True, "trait_b": True},
        )
        template = VerificationResultTemplate(
            template_verification_performed=False,
            verify_result=None,
        )
        result = VerificationResult(
            metadata=_make_metadata(),
            template=template,
            rubric=rubric,
        )
        step = StepEval(verification_results={"q1": [result]})
        stats = step.get_summary_stats()
        assert stats["success_rate"] == 100.0
        assert stats["traces_passed"] == 1

    def test_rubric_only_some_fail(self):
        """success_rate == 0 when rubric traits fail."""
        rubric = VerificationResultRubric(
            rubric_evaluation_performed=True,
            verify_rubric={"trait_a": True, "trait_b": False},
        )
        template = VerificationResultTemplate(
            template_verification_performed=False,
            verify_result=None,
        )
        result = VerificationResult(
            metadata=_make_metadata(),
            template=template,
            rubric=rubric,
        )
        step = StepEval(verification_results={"q1": [result]})
        stats = step.get_summary_stats()
        assert stats["success_rate"] == 0.0
        assert stats["traces_passed"] == 0

    def test_template_mode_unchanged(self):
        """Template-verified traces still use verify_result for pass counting."""
        template = VerificationResultTemplate(
            template_verification_performed=True,
            verify_result=True,
        )
        result = VerificationResult(
            metadata=_make_metadata(),
            template=template,
        )
        step = StepEval(verification_results={"q1": [result]})
        stats = step.get_summary_stats()
        assert stats["success_rate"] == 100.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/carli/Projects/karenina-salvage/karenina/.claude/worktrees/taskeval-harmonize && uv run pytest tests/unit/benchmark/test_task_eval_metadata.py::TestSuccessRateRubricOnly -v`
Expected: `test_rubric_only_all_pass` FAILS (success_rate is 0.0)

- [ ] **Step 3: Fix `get_summary_stats()` passed_traces logic**

In `src/karenina/benchmark/task_eval/models.py`, replace lines 145-148:

```python
        for _trace_id, results in self.verification_results.items():
            total_results += len(results)
            if any(result.template and result.template.verify_result for result in results):
                passed_traces += 1
```

With:

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
                result.rubric
                and result.rubric.rubric_evaluation_performed
                and all(
                    score is True or (isinstance(score, int) and score > 0)
                    for score in result.rubric.get_all_trait_scores().values()
                )
                for result in results
            )

            if template_passed or (not template_performed and rubric_passed):
                passed_traces += 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/carli/Projects/karenina-salvage/karenina/.claude/worktrees/taskeval-harmonize && uv run pytest tests/unit/benchmark/test_task_eval_metadata.py::TestSuccessRateRubricOnly -v`
Expected: All PASS

- [ ] **Step 5: Run existing TaskEval tests for regression**

Run: `cd /Users/carli/Projects/karenina-salvage/karenina/.claude/worktrees/taskeval-harmonize && uv run pytest tests/unit/benchmark/test_task_eval.py -v -x`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/karenina/benchmark/task_eval/models.py tests/unit/benchmark/test_task_eval_metadata.py
git commit -m "fix(task_eval): count rubric-only traces as passed when all traits pass"
```

---

### Task 4: Fix `_get_available_step_ids()` (Issue 115)

**Files:**
- Modify: `src/karenina/benchmark/task_eval/task_eval.py:408-418`
- Test: `tests/unit/benchmark/test_task_eval_metadata.py`

- [ ] **Step 1: Write test for dynamic rubric step discovery**

Append to `tests/unit/benchmark/test_task_eval_metadata.py`:

```python
from karenina.benchmark.task_eval import TaskEval


@pytest.mark.unit
class TestAvailableStepIds:
    """Issue 115: _get_available_step_ids() includes dynamic rubric steps."""

    def test_dynamic_rubric_only_step_discovered(self):
        """A step with only a dynamic rubric should appear in available step IDs."""
        from karenina.schemas.entities.rubric import DynamicRubric, LLMRubricTrait

        task = TaskEval()
        task.add_dynamic_rubric(
            DynamicRubric(
                llm_traits=[
                    LLMRubricTrait(
                        name="dynamic_trait",
                        summary="check",
                        description="Dynamic check",
                        kind="boolean",
                    )
                ]
            ),
            step_id="dynamic_step",
        )
        step_ids = task._get_available_step_ids()
        assert "dynamic_step" in step_ids
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/carli/Projects/karenina-salvage/karenina/.claude/worktrees/taskeval-harmonize && uv run pytest tests/unit/benchmark/test_task_eval_metadata.py::TestAvailableStepIds -v`
Expected: FAIL, `"dynamic_step"` not in set

- [ ] **Step 3: Add `step_dynamic_rubrics` to `_get_available_step_ids()`**

In `src/karenina/benchmark/task_eval/task_eval.py`, after line 417 (`step_ids.update(self.step_rubrics.keys())`), add:

```python
        step_ids.update(self.step_dynamic_rubrics.keys())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/carli/Projects/karenina-salvage/karenina/.claude/worktrees/taskeval-harmonize && uv run pytest tests/unit/benchmark/test_task_eval_metadata.py::TestAvailableStepIds -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/karenina/benchmark/task_eval/task_eval.py tests/unit/benchmark/test_task_eval_metadata.py
git commit -m "fix(task_eval): include dynamic rubric steps in _get_available_step_ids()"
```

---

### Task 5: Thread Parameters Through Evaluation Chain (Issues 114, 160, 165, 166, 168, 169)

This is the core task. It modifies `evaluate()`, `_evaluate_global()`, `_evaluate_step()`, `_evaluate_step_internal()`, `_run_evaluation_loop()`, `_evaluate_and_store()`, and `_evaluate()` to thread `answering_model`, `run_name`, `replicate`, `abstention_enabled`, and `sufficiency_enabled` down to `run_single_model_verification()`.

**Files:**
- Modify: `src/karenina/benchmark/task_eval/task_eval.py:306-762`
- Test: `tests/unit/benchmark/test_task_eval_metadata.py`

- [ ] **Step 1: Write tests for parameter threading**

Append to `tests/unit/benchmark/test_task_eval_metadata.py`:

```python
from typing import Any

from karenina.ports.messages import Message
from karenina.schemas.verification import VerificationConfig


def _make_mock_result():
    """Create a minimal mock VerificationResult."""
    from karenina.schemas.verification.model_identity import ModelIdentity
    from karenina.schemas.verification.result_components import VerificationResultMetadata

    metadata = VerificationResultMetadata(
        question_id="test_q",
        template_id="test_t",
        completed_without_errors=True,
        question_text="test question",
        answering=ModelIdentity(interface="mock", model_name="mock"),
        parsing=ModelIdentity(interface="mock", model_name="mock"),
        execution_time=0.1,
        timestamp="2026-01-01T00:00:00",
        result_id="abcd1234abcd1234",
    )
    return VerificationResult(metadata=metadata)


def _make_config(**overrides) -> VerificationConfig:
    """Create a test VerificationConfig."""
    defaults = {
        "parsing_models": [
            ModelConfig(
                id="test_parser",
                model_provider="mock",
                model_name="mock",
                interface="langchain",
            )
        ],
        "parsing_only": True,
    }
    defaults.update(overrides)
    return VerificationConfig(**defaults)


@pytest.mark.unit
class TestParameterThreading:
    """Issues 114, 160, 165, 166, 168: parameters reach run_single_model_verification."""

    def _capture_calls(self, monkeypatch) -> list[dict[str, Any]]:
        """Monkeypatch runner and return captured kwargs list."""
        captured: list[dict[str, Any]] = []

        def mock_run(*args, **kwargs):
            captured.append(kwargs)
            return _make_mock_result()

        monkeypatch.setattr(
            "karenina.benchmark.verification.runner.run_single_model_verification",
            mock_run,
        )
        return captured

    def test_sentinel_model_used_by_default(self, monkeypatch):
        """Issue 166: sentinel answering_model with interface='taskeval' when none provided."""
        from karenina.schemas.entities.rubric import LLMRubricTrait, Rubric

        captured = self._capture_calls(monkeypatch)
        task = TaskEval(task_id="test")
        task.log_trace([Message.assistant("response")])
        task.add_rubric(
            Rubric(
                llm_traits=[
                    LLMRubricTrait(name="q", description="check", kind="boolean"),
                ]
            )
        )
        task.evaluate(_make_config())

        assert len(captured) > 0
        model = captured[0]["answering_model"]
        assert model.interface == "taskeval"
        assert model.model_name == "user-provided"
        assert model.model_provider == "user-provided"

    def test_custom_answering_model_passed_through(self, monkeypatch):
        """Issue 166: user-provided answering_model reaches the runner."""
        from karenina.schemas.entities.rubric import LLMRubricTrait, Rubric

        captured = self._capture_calls(monkeypatch)
        task = TaskEval(task_id="test")
        task.log_trace([Message.assistant("response")])
        task.add_rubric(
            Rubric(
                llm_traits=[
                    LLMRubricTrait(name="q", description="check", kind="boolean"),
                ]
            )
        )
        custom_model = ModelConfig(
            id="gpt4",
            model_provider="openai",
            model_name="gpt-4",
            interface="langchain",
        )
        task.evaluate(_make_config(), answering_model=custom_model)

        assert captured[0]["answering_model"].model_name == "gpt-4"

    def test_run_name_auto_generated(self, monkeypatch):
        """Issue 168: auto-generated run_name reaches the runner."""
        from karenina.schemas.entities.rubric import LLMRubricTrait, Rubric

        captured = self._capture_calls(monkeypatch)
        task = TaskEval(task_id="test")
        task.log_trace([Message.assistant("response")])
        task.add_rubric(
            Rubric(
                llm_traits=[
                    LLMRubricTrait(name="q", description="check", kind="boolean"),
                ]
            )
        )
        task.evaluate(_make_config())

        assert captured[0]["run_name"] is not None
        assert captured[0]["run_name"].startswith("taskeval_")

    def test_run_name_explicit(self, monkeypatch):
        """Issue 168: explicit run_name reaches the runner."""
        from karenina.schemas.entities.rubric import LLMRubricTrait, Rubric

        captured = self._capture_calls(monkeypatch)
        task = TaskEval(task_id="test")
        task.log_trace([Message.assistant("response")])
        task.add_rubric(
            Rubric(
                llm_traits=[
                    LLMRubricTrait(name="q", description="check", kind="boolean"),
                ]
            )
        )
        task.evaluate(_make_config(), run_name="my_run")

        assert captured[0]["run_name"] == "my_run"

    def test_replicate_index_threaded(self, monkeypatch):
        """Issue 114: replicate index reaches the runner when replicate_count > 1."""
        from karenina.schemas.entities.rubric import LLMRubricTrait, Rubric

        captured = self._capture_calls(monkeypatch)
        task = TaskEval(task_id="test")
        task.log_trace([Message.assistant("response")])
        task.add_rubric(
            Rubric(
                llm_traits=[
                    LLMRubricTrait(name="q", description="check", kind="boolean"),
                ]
            )
        )
        config = _make_config(replicate_count=3)
        task.evaluate(config)

        assert len(captured) == 3
        assert captured[0]["replicate"] == 1
        assert captured[1]["replicate"] == 2
        assert captured[2]["replicate"] == 3

    def test_replicate_none_when_single(self, monkeypatch):
        """Issue 114: replicate is None when replicate_count is 1."""
        from karenina.schemas.entities.rubric import LLMRubricTrait, Rubric

        captured = self._capture_calls(monkeypatch)
        task = TaskEval(task_id="test")
        task.log_trace([Message.assistant("response")])
        task.add_rubric(
            Rubric(
                llm_traits=[
                    LLMRubricTrait(name="q", description="check", kind="boolean"),
                ]
            )
        )
        task.evaluate(_make_config())

        assert captured[0]["replicate"] is None

    def test_guard_flags_from_config(self, monkeypatch):
        """Issue 160: abstention_enabled and sufficiency_enabled from config."""
        from karenina.schemas.entities.rubric import LLMRubricTrait, Rubric

        captured = self._capture_calls(monkeypatch)
        task = TaskEval(task_id="test")
        task.log_trace([Message.assistant("response")])
        task.add_rubric(
            Rubric(
                llm_traits=[
                    LLMRubricTrait(name="q", description="check", kind="boolean"),
                ]
            )
        )
        config = _make_config(abstention_enabled=True, sufficiency_enabled=True)
        task.evaluate(config)

        assert captured[0]["abstention_enabled"] is True
        assert captured[0]["sufficiency_enabled"] is True

    def test_guard_flags_default_false(self, monkeypatch):
        """Issue 160: guard flags default to False."""
        from karenina.schemas.entities.rubric import LLMRubricTrait, Rubric

        captured = self._capture_calls(monkeypatch)
        task = TaskEval(task_id="test")
        task.log_trace([Message.assistant("response")])
        task.add_rubric(
            Rubric(
                llm_traits=[
                    LLMRubricTrait(name="q", description="check", kind="boolean"),
                ]
            )
        )
        task.evaluate(_make_config())

        assert captured[0]["abstention_enabled"] is False
        assert captured[0]["sufficiency_enabled"] is False

    def test_question_id_not_force_hashed(self, monkeypatch):
        """Issue 165: non-MD5 question IDs pass through without hashing."""
        captured = self._capture_calls(monkeypatch)
        task = TaskEval(task_id="test")
        task.log("Some output")

        # Add a question with a human-readable ID
        task.add_question({"id": "my_readable_id", "question": "What?", "answer_template": '''
from karenina.schemas.entities import BaseAnswer
class Answer(BaseAnswer):
    value: str = ""
    def verify(self) -> bool:
        return True
'''})

        task.evaluate(_make_config())

        assert captured[0]["question_id"] == "my_readable_id"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/carli/Projects/karenina-salvage/karenina/.claude/worktrees/taskeval-harmonize && uv run pytest tests/unit/benchmark/test_task_eval_metadata.py::TestParameterThreading -v`
Expected: Multiple failures (parameters not threaded, sentinel not used, etc.)

- [ ] **Step 3: Modify `evaluate()` signature**

In `src/karenina/benchmark/task_eval/task_eval.py`, change `evaluate()` (lines 306-335):

```python
    def evaluate(
        self,
        config: VerificationConfig,
        step_id: str | None = None,
        merge_strategy: Literal["concatenate", "traces_only"] | None = None,
        answering_model: ModelConfig | None = None,
        run_name: str | None = None,
    ) -> TaskEvalResult:
        """Evaluate logged outputs against questions and rubrics.

        Args:
            config: Verification configuration (parsing models only)
            step_id: Optional step ID to evaluate specific step (otherwise global)
            merge_strategy: Optional override for the instance merge_strategy.
                If None, uses the instance default.
            answering_model: Optional model identity for the source of pre-collected
                outputs. If None, uses a sentinel with interface='taskeval'.
            run_name: Optional run name for result tracking. If None, auto-generated.

        Returns:
            TaskEvalResult with evaluation outcomes and failure characterization
        """
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

        if config.is_few_shot_enabled():
            logger.debug("FewShotConfig has no effect in TaskEval mode")

        effective_strategy = merge_strategy or self.merge_strategy

        if step_id:
            return self._evaluate_step(
                config, step_id, effective_strategy, answering_model, run_name
            )
        else:
            return self._evaluate_global(
                config, effective_strategy, answering_model, run_name
            )
```

- [ ] **Step 4: Thread through `_evaluate_global()`, `_evaluate_step()`, `_evaluate_step_internal()`**

Update all three methods to accept and forward `answering_model` and `run_name`:

`_evaluate_global()` (lines 337-365):
```python
    def _evaluate_global(
        self,
        config: VerificationConfig,
        merge_strategy: Literal["concatenate", "traces_only"],
        answering_model: ModelConfig,
        run_name: str,
    ) -> TaskEvalResult:
        """Evaluate all global logs against global questions and rubrics."""
        step_eval = self._run_evaluation_loop(
            config, step_id=None, merge_strategy=merge_strategy,
            answering_model=answering_model, run_name=run_name,
        )

        task_result = TaskEvalResult(
            task_id=self.task_id,
            metadata=self.metadata,
            global_eval=step_eval,
        )

        for sid in self._get_available_step_ids():
            step_result = self._evaluate_step_internal(
                config, sid, merge_strategy, answering_model, run_name
            )
            task_result.per_step[sid] = step_result

        return task_result
```

`_evaluate_step()` (lines 367-384):
```python
    def _evaluate_step(
        self,
        config: VerificationConfig,
        step_id: str,
        merge_strategy: Literal["concatenate", "traces_only"],
        answering_model: ModelConfig,
        run_name: str,
    ) -> TaskEvalResult:
        """Evaluate step-specific logs against step-specific questions and rubrics."""
        step_eval = self._evaluate_step_internal(
            config, step_id, merge_strategy, answering_model, run_name
        )
        return self._build_result(step_eval, step_id=step_id)
```

`_evaluate_step_internal()` (lines 386-402):
```python
    def _evaluate_step_internal(
        self,
        config: VerificationConfig,
        step_id: str,
        merge_strategy: Literal["concatenate", "traces_only"],
        answering_model: ModelConfig,
        run_name: str,
    ) -> StepEval:
        """Internal method to evaluate a single step and return StepEval."""
        return self._run_evaluation_loop(
            config, step_id=step_id, merge_strategy=merge_strategy,
            answering_model=answering_model, run_name=run_name,
        )
```

- [ ] **Step 5: Thread through `_run_evaluation_loop()`**

Update `_run_evaluation_loop()` signature (line 523) to accept `answering_model` and `run_name`. Add replicate tracking. Forward all to `_evaluate_and_store()`:

Add to signature:
```python
    def _run_evaluation_loop(
        self,
        config: VerificationConfig,
        step_id: str | None,
        merge_strategy: Literal["concatenate", "traces_only"],
        answering_model: ModelConfig,
        run_name: str,
    ) -> StepEval:
```

Change the replicate loop (line 564) from `for _ in range(replicate_count):` to:
```python
        for rep_idx in range(replicate_count):
            replicate = None if replicate_count == 1 else rep_idx + 1
```

Add `answering_model`, `run_name`, `replicate` to both `_evaluate_and_store()` calls (lines 574 and 602):
```python
                self._evaluate_and_store(
                    ...,  # existing params unchanged
                    answering_model=answering_model,
                    run_name=run_name,
                    replicate=replicate,
                )
```

- [ ] **Step 6: Thread through `_evaluate_and_store()`**

Update signature (line 617) to accept `answering_model`, `run_name`, `replicate`, and guard flags:

```python
    def _evaluate_and_store(
        self,
        step_eval: StepEval,
        question_dict: dict[str, Any],
        response_text: str,
        parsing_model: ModelConfig,
        rubric: "Rubric | None",
        dynamic_rubric: "DynamicRubric | None",
        evaluation_mode: str,
        error_context: str,
        trace_messages: "list[Message] | None" = None,
        agent_metrics: dict[str, Any] | None = None,
        answering_model: ModelConfig | None = None,
        run_name: str | None = None,
        replicate: int | None = None,
        abstention_enabled: bool = False,
        sufficiency_enabled: bool = False,
    ) -> None:
```

Forward all to `_evaluate()`:
```python
            verification_result = self._evaluate(
                question_dict=question_dict,
                response_text=response_text,
                parsing_model=parsing_model,
                rubric=rubric,
                dynamic_rubric=dynamic_rubric,
                evaluation_mode=evaluation_mode,
                trace_messages=trace_messages,
                agent_metrics=agent_metrics,
                answering_model=answering_model,
                run_name=run_name,
                replicate=replicate,
                abstention_enabled=abstention_enabled,
                sufficiency_enabled=sufficiency_enabled,
            )
```

- [ ] **Step 7: Rewrite `_evaluate()` to use threaded parameters**

Update `_evaluate()` signature (line 670) to accept all threaded parameters. In `_run_evaluation_loop()`, after extracting `replicate_count`, add:

```python
        abstention_enabled = config.abstention_enabled
        sufficiency_enabled = config.sufficiency_enabled
```

Pass `abstention_enabled` and `sufficiency_enabled` to both `_evaluate_and_store()` calls alongside the other new params.

In `_evaluate()`, remove the MD5 hashing block (lines 718-722), the mock model block (lines 726-733), and replace the `run_single_model_verification()` call with:

```python
        # question_id passes through as-is (no force-hashing)
        assert isinstance(answer_template, str), "answer_template must be a string"

        # Prepare cached answer data to inject logged output
        cached_answer_data: dict[str, Any] = {
            "raw_llm_response": response_text,
            "recursion_limit_reached": False,
            "answering_mcp_servers": None,
            "usage_metadata": None,
            "agent_metrics": agent_metrics,
        }

        if trace_messages:
            cached_answer_data["trace_messages"] = [m.to_dict() for m in trace_messages]

        verification_result = run_single_model_verification(
            question_id=question_id,
            question_text=question_text,
            template_code=answer_template,
            answering_model=answering_model,
            parsing_model=parsing_model,
            rubric=rubric,
            dynamic_rubric=dynamic_rubric,
            cached_answer_data=cached_answer_data,
            run_name=run_name,
            replicate=replicate,
            abstention_enabled=abstention_enabled,
            sufficiency_enabled=sufficiency_enabled,
            rubric_evaluation_strategy="batch",
            evaluation_mode=evaluation_mode,
        )

        return verification_result
```

Remove the `_is_valid_md5_hash()` method (lines 768-773) and the `import hashlib` (line 719) since they are no longer used.

- [ ] **Step 8: Run tests to verify they pass**

Run: `cd /Users/carli/Projects/karenina-salvage/karenina/.claude/worktrees/taskeval-harmonize && uv run pytest tests/unit/benchmark/test_task_eval_metadata.py::TestParameterThreading -v`
Expected: All PASS

- [ ] **Step 9: Run full TaskEval test suite**

Run: `cd /Users/carli/Projects/karenina-salvage/karenina/.claude/worktrees/taskeval-harmonize && uv run pytest tests/unit/benchmark/test_task_eval.py tests/unit/benchmark/test_task_eval_issues.py -v -x`
Expected: All pass (existing tests should not break; monkeypatched tests accept `*args, **kwargs`)

- [ ] **Step 10: Commit**

```bash
git add src/karenina/benchmark/task_eval/task_eval.py tests/unit/benchmark/test_task_eval_metadata.py
git commit -m "feat(task_eval): thread answering_model, run_name, replicate, and guard flags to runner

Fixes issues 114 (replicate tracking), 160 (guard flags from config),
165 (question_id passthrough), 166 (sentinel model), 168 (run_name)."
```

---

### Task 6: Narrow Exception Handling (Issue 117, Part 2)

**Depends on:** Task 2 (adds `failed_questions` field to `StepEval`)

**Files:**
- Modify: `src/karenina/benchmark/task_eval/task_eval.py:647-664`
- Test: `tests/unit/benchmark/test_task_eval_metadata.py`

- [ ] **Step 1: Write tests for exception narrowing**

Append to `tests/unit/benchmark/test_task_eval_metadata.py`:

```python
from karenina.exceptions import KareninaError


@pytest.mark.unit
class TestExceptionNarrowing:
    """Issue 117: _evaluate_and_store() narrows exception scope."""

    def test_programming_error_propagates(self, monkeypatch):
        """TypeError/AttributeError should NOT be caught."""
        from karenina.schemas.entities.rubric import LLMRubricTrait, Rubric

        def mock_run(*args, **kwargs):
            raise TypeError("This is a programming bug")

        monkeypatch.setattr(
            "karenina.benchmark.verification.runner.run_single_model_verification",
            mock_run,
        )

        task = TaskEval(task_id="test")
        task.log_trace([Message.assistant("response")])
        task.add_rubric(
            Rubric(
                llm_traits=[
                    LLMRubricTrait(name="q", description="check", kind="boolean"),
                ]
            )
        )

        with pytest.raises(TypeError, match="programming bug"):
            task.evaluate(_make_config())

    def test_domain_error_caught_and_recorded(self, monkeypatch):
        """KareninaError should be caught and recorded in failed_questions."""
        from karenina.schemas.entities.rubric import LLMRubricTrait, Rubric

        def mock_run(*args, **kwargs):
            raise KareninaError("Evaluation pipeline failed")

        monkeypatch.setattr(
            "karenina.benchmark.verification.runner.run_single_model_verification",
            mock_run,
        )

        task = TaskEval(task_id="test")
        task.log_trace([Message.assistant("response")])
        task.add_rubric(
            Rubric(
                llm_traits=[
                    LLMRubricTrait(name="q", description="check", kind="boolean"),
                ]
            )
        )

        result = task.evaluate(_make_config())
        step = result.global_eval
        assert step is not None
        assert len(step.failed_questions) > 0
        # The error message should be recorded
        failed = list(step.failed_questions.values())[0]
        assert "Evaluation pipeline failed" in failed[0]

    def test_value_error_caught_and_recorded(self, monkeypatch):
        """ValueError should be caught and recorded in failed_questions."""
        from karenina.schemas.entities.rubric import LLMRubricTrait, Rubric

        def mock_run(*args, **kwargs):
            raise ValueError("Invalid template")

        monkeypatch.setattr(
            "karenina.benchmark.verification.runner.run_single_model_verification",
            mock_run,
        )

        task = TaskEval(task_id="test")
        task.log_trace([Message.assistant("response")])
        task.add_rubric(
            Rubric(
                llm_traits=[
                    LLMRubricTrait(name="q", description="check", kind="boolean"),
                ]
            )
        )

        result = task.evaluate(_make_config())
        step = result.global_eval
        assert step is not None
        assert len(step.failed_questions) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/carli/Projects/karenina-salvage/karenina/.claude/worktrees/taskeval-harmonize && uv run pytest tests/unit/benchmark/test_task_eval_metadata.py::TestExceptionNarrowing -v`
Expected: `test_programming_error_propagates` FAILS (TypeError currently caught by broad `except Exception`)

- [ ] **Step 3: Narrow the exception catch in `_evaluate_and_store()`**

In `src/karenina/benchmark/task_eval/task_eval.py`, replace the except block (lines 663-664):

```python
        except Exception as e:
            logger.warning("Evaluation failed for %s: %s", error_context, e)
```

With:

```python
        except (KareninaError, ValueError, RuntimeError) as e:
            logger.warning("Evaluation failed for %s: %s", error_context, e)
            if question_id not in step_eval.failed_questions:
                step_eval.failed_questions[question_id] = []
            step_eval.failed_questions[question_id].append(str(e))
```

Add the import near the top of the method or at module level:

```python
from karenina.exceptions import KareninaError
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/carli/Projects/karenina-salvage/karenina/.claude/worktrees/taskeval-harmonize && uv run pytest tests/unit/benchmark/test_task_eval_metadata.py::TestExceptionNarrowing -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/karenina/benchmark/task_eval/task_eval.py tests/unit/benchmark/test_task_eval_metadata.py
git commit -m "fix(task_eval): narrow exception catch to domain errors and record failures"
```

---

### Task 7: Add FewShotConfig Warning (Issue 179)

**Files:**
- Modify: `src/karenina/benchmark/task_eval/task_eval.py` (already done in Task 5, Step 3)
- Test: `tests/unit/benchmark/test_task_eval_metadata.py`

- [ ] **Step 1: Write test for FewShotConfig warning**

Append to `tests/unit/benchmark/test_task_eval_metadata.py`:

```python
import logging


@pytest.mark.unit
class TestFewShotConfigWarning:
    """Issue 179: FewShotConfig warning in TaskEval mode."""

    def test_few_shot_warning_emitted(self, monkeypatch, caplog):
        """Debug log emitted when FewShotConfig is enabled."""
        from karenina.schemas.entities.rubric import LLMRubricTrait, Rubric

        def mock_run(*args, **kwargs):
            return _make_mock_result()

        monkeypatch.setattr(
            "karenina.benchmark.verification.runner.run_single_model_verification",
            mock_run,
        )

        task = TaskEval(task_id="test")
        task.log_trace([Message.assistant("response")])
        task.add_rubric(
            Rubric(
                llm_traits=[
                    LLMRubricTrait(name="q", description="check", kind="boolean"),
                ]
            )
        )

        config = _make_config()
        # Enable few-shot on the config
        from karenina.schemas.config import FewShotConfig

        config.few_shot = FewShotConfig(enabled=True, examples=[])

        with caplog.at_level(logging.DEBUG, logger="karenina.benchmark.task_eval.task_eval"):
            task.evaluate(config)

        assert any("FewShotConfig has no effect in TaskEval mode" in m for m in caplog.messages)
```

- [ ] **Step 2: Run test to verify it passes**

This test should already pass if Task 5 Step 3 was implemented correctly (the warning is in `evaluate()`).

Run: `cd /Users/carli/Projects/karenina-salvage/karenina/.claude/worktrees/taskeval-harmonize && uv run pytest tests/unit/benchmark/test_task_eval_metadata.py::TestFewShotConfigWarning -v`
Expected: PASS

If it fails, check that the `config.is_few_shot_enabled()` call and `logger.debug()` are in `evaluate()`. The `FewShotConfig` import path may need adjustment; check `karenina/schemas/config/` for the actual module.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/benchmark/test_task_eval_metadata.py
git commit -m "test(task_eval): add FewShotConfig warning test for issue 179"
```

---

### Task 8: Final Regression and Cleanup

**Files:**
- All modified files from previous tasks

- [ ] **Step 1: Run full TaskEval test suite**

Run: `cd /Users/carli/Projects/karenina-salvage/karenina/.claude/worktrees/taskeval-harmonize && uv run pytest tests/unit/benchmark/test_task_eval.py tests/unit/benchmark/test_task_eval_issues.py tests/unit/benchmark/test_task_eval_metadata.py -v`
Expected: All pass

- [ ] **Step 2: Run full project test suite**

Run: `cd /Users/carli/Projects/karenina-salvage/karenina/.claude/worktrees/taskeval-harmonize && uv run pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 3: Verify `_is_valid_md5_hash` removal didn't break anything**

Run: `cd /Users/carli/Projects/karenina-salvage/karenina/.claude/worktrees/taskeval-harmonize && uv run pytest tests/ -x -q -k "md5 or hash or valid"` (confirm no tests reference the removed method)

Also check no other code references it:
```bash
grep -r "_is_valid_md5_hash" src/karenina/benchmark/task_eval/
```
Expected: No matches (the method was only used in `_evaluate()`)

- [ ] **Step 4: Run type checking**

Run: `cd /Users/carli/Projects/karenina-salvage/karenina/.claude/worktrees/taskeval-harmonize && uv run mypy src/karenina/benchmark/task_eval/ --ignore-missing-imports`
Expected: No new errors

- [ ] **Step 5: Commit any cleanup**

If any cleanup was needed, commit it:
```bash
git add -A
git commit -m "chore(task_eval): cleanup after metadata coherence fixes"
```
