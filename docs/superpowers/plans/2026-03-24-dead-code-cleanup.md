# Dead Code Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean up dead code paths, wire in unused PromptConfig.generation, add instruction shortcuts, restructure FewShotConfig, and remove runner auto-upgrade.

**Architecture:** Four independent changes, ordered by complexity. Each is TDD: write failing test, implement, verify, commit. The FewShotConfig restructure (Task 4) has the widest ripple effects across models.py, config.py, task_helpers.py, interactive.py, and test files.

**Tech Stack:** Python 3.13, Pydantic v2, pytest

**Spec:** `docs/superpowers/specs/2026-03-24-dead-code-design-decisions.md`

---

## Task 1: Remove Runner Auto-Upgrade

**Files:**
- Modify: `src/karenina/benchmark/verification/runner.py:211-220`
- Test: `tests/unit/benchmark/verification/test_runner_auto_upgrade.py` (create)

- [ ] **Step 1: Write failing tests**

Create `tests/unit/benchmark/verification/test_runner_auto_upgrade.py`:

```python
"""Tests for runner evaluation_mode auto-upgrade removal."""

import logging
from unittest.mock import patch

import pytest

from karenina.schemas.entities.rubric import LLMRubricTrait, Rubric


@pytest.mark.unit
class TestRunnerAutoUpgrade:
    """Verify that runner no longer auto-upgrades template_only to template_and_rubric."""

    def test_template_only_with_rubric_warns_and_does_not_upgrade(self, caplog: pytest.LogCaptureFixture) -> None:
        """Rubric traits with template_only should warn, not upgrade."""
        from karenina.benchmark.verification.runner import run_single_model_verification
        from karenina.schemas.config import ModelConfig

        rubric = Rubric(llm_traits=[LLMRubricTrait(name="clarity", description="Is the response clear?")])
        answering = ModelConfig(id="test", model_name="test", model_provider="openai", system_prompt="You are helpful.")
        parsing = ModelConfig(id="test", model_name="test", model_provider="openai", system_prompt="Parse.")

        # Mock StageOrchestrator to avoid running the full pipeline
        with (
            patch("karenina.benchmark.verification.runner.StageOrchestrator") as mock_orch,
            caplog.at_level(logging.WARNING),
        ):
            mock_orch.from_config.return_value.execute.return_value = None
            run_single_model_verification(
                question_id="q1",
                question_text="What is 2+2?",
                template_code="class Answer(BaseAnswer): value: str",
                answering_model=answering,
                parsing_model=parsing,
                evaluation_mode="template_only",
                rubric=rubric,
            )

        assert any("template_only" in r.message and "Rubric" in r.message for r in caplog.records)
        # Verify evaluation_mode was NOT changed: from_config should receive "template_only"
        call_kwargs = mock_orch.from_config.call_args
        assert call_kwargs.kwargs.get("evaluation_mode", call_kwargs[1].get("evaluation_mode")) == "template_only"

    def test_template_only_without_rubric_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """No rubric traits should produce no warning."""
        from karenina.benchmark.verification.runner import run_single_model_verification
        from karenina.schemas.config import ModelConfig

        answering = ModelConfig(id="test", model_name="test", model_provider="openai", system_prompt="You are helpful.")
        parsing = ModelConfig(id="test", model_name="test", model_provider="openai", system_prompt="Parse.")

        with (
            patch("karenina.benchmark.verification.runner.StageOrchestrator") as mock_orch,
            caplog.at_level(logging.WARNING),
        ):
            mock_orch.from_config.return_value.execute.return_value = None
            run_single_model_verification(
                question_id="q1",
                question_text="What is 2+2?",
                template_code="class Answer(BaseAnswer): value: str",
                answering_model=answering,
                parsing_model=parsing,
                evaluation_mode="template_only",
            )

        assert not any("template_only" in r.message for r in caplog.records)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/carli/Projects/karenina-salvage/karenina/.claude/worktrees/dead-code && uv run pytest tests/unit/benchmark/verification/test_runner_auto_upgrade.py -v`

Expected: First test FAILS (no warning emitted because the auto-upgrade happens silently).

- [ ] **Step 3: Implement the change**

In `src/karenina/benchmark/verification/runner.py`, replace lines 211-220 (the auto-upgrade block):

```python
# OLD (lines 211-220):
_has_rubric_traits = rubric and (...)
_has_dynamic_rubric_traits = ...
if (...) and evaluation_mode == "template_only":
    evaluation_mode = "template_and_rubric"

# NEW:
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

The only change: replace `evaluation_mode = "template_and_rubric"` with `logger.warning(...)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/carli/Projects/karenina-salvage/karenina/.claude/worktrees/dead-code && uv run pytest tests/unit/benchmark/verification/test_runner_auto_upgrade.py -v`

Expected: PASS

- [ ] **Step 5: Run existing runner tests to check for regressions**

Run: `cd /Users/carli/Projects/karenina-salvage/karenina/.claude/worktrees/dead-code && uv run pytest tests/unit/benchmark/verification/ -v -x`

Expected: PASS (no existing tests depend on the auto-upgrade)

- [ ] **Step 6: Commit**

```bash
git add src/karenina/benchmark/verification/runner.py tests/unit/benchmark/verification/test_runner_auto_upgrade.py
git commit -m "fix: replace silent evaluation_mode auto-upgrade with warning"
```

---

## Task 2: Wire In PromptConfig.generation

**Files:**
- Modify: `src/karenina/benchmark/verification/stages/pipeline/generate_answer.py:268-272`
- Test: `tests/unit/benchmark/verification/stages/test_generate_answer_prompt_config.py` (create)

- [ ] **Step 1: Write failing tests**

Create `tests/unit/benchmark/verification/stages/test_generate_answer_prompt_config.py`:

```python
"""Tests for PromptConfig.generation wiring in GenerateAnswerStage."""

from unittest.mock import MagicMock, patch

import pytest

from karenina.benchmark.verification.stages.core.base import VerificationContext
from karenina.benchmark.verification.stages.pipeline.generate_answer import GenerateAnswerStage
from karenina.ports import Message
from karenina.schemas.config import ModelConfig
from karenina.schemas.verification.prompt_config import PromptConfig


def _make_context(
    system_prompt: str | None = "You are helpful.",
    prompt_config: PromptConfig | None = None,
) -> VerificationContext:
    """Create a minimal VerificationContext for testing."""
    model = ModelConfig(
        id="test",
        model_name="test",
        model_provider="openai",
        system_prompt=system_prompt,
    )
    ctx = VerificationContext(
        question_id="q1",
        template_id="t1",
        question_text="What is 2+2?",
        template_code="class Answer(BaseAnswer): value: str",
        answering_model=model,
        parsing_model=model,
        prompt_config=prompt_config,
    )
    return ctx


def _capture_adapter_messages(context: VerificationContext) -> list[Message]:
    """Run GenerateAnswerStage.execute() with mocked adapter, return captured messages."""
    stage = GenerateAnswerStage()
    captured: list[Message] = []

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "4"
    mock_response.usage = None
    mock_llm.invoke.side_effect = lambda msgs: (captured.extend(msgs), mock_response)[1]

    with patch("karenina.benchmark.verification.stages.pipeline.generate_answer.get_llm", return_value=mock_llm):
        stage.execute(context)

    return captured


@pytest.mark.unit
class TestPromptConfigGeneration:
    """Test that PromptConfig.generation is injected into the system message."""

    def test_generation_instructions_appended_to_system_prompt(self) -> None:
        """PromptConfig.generation should appear in the system message."""
        ctx = _make_context(
            system_prompt="You are helpful.",
            prompt_config=PromptConfig(generation="Focus on clinical accuracy."),
        )
        messages = _capture_adapter_messages(ctx)
        system_msgs = [m for m in messages if m.role == "system"]
        assert len(system_msgs) == 1
        assert "You are helpful." in system_msgs[0].content
        assert "Focus on clinical accuracy." in system_msgs[0].content

    def test_no_prompt_config_uses_system_prompt_only(self) -> None:
        """Without PromptConfig, only ModelConfig.system_prompt is used."""
        ctx = _make_context(system_prompt="You are helpful.", prompt_config=None)
        messages = _capture_adapter_messages(ctx)
        system_msgs = [m for m in messages if m.role == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0].content == "You are helpful."

    def test_generation_none_does_not_change_system_prompt(self) -> None:
        """PromptConfig with generation=None should not modify the system message."""
        ctx = _make_context(
            system_prompt="You are helpful.",
            prompt_config=PromptConfig(generation=None),
        )
        messages = _capture_adapter_messages(ctx)
        system_msgs = [m for m in messages if m.role == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0].content == "You are helpful."

    def test_no_system_prompt_generation_only(self) -> None:
        """When ModelConfig.system_prompt is None, only generation instructions appear."""
        ctx = _make_context(
            system_prompt=None,
            prompt_config=PromptConfig(generation="Be concise."),
        )
        messages = _capture_adapter_messages(ctx)
        system_msgs = [m for m in messages if m.role == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0].content == "Be concise."
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/carli/Projects/karenina-salvage/karenina/.claude/worktrees/dead-code && uv run pytest tests/unit/benchmark/verification/stages/test_generate_answer_prompt_config.py -v`

Expected: First test FAILS (generation instructions not in system message because the stage doesn't read PromptConfig yet).

- [ ] **Step 3: Implement the change**

In `src/karenina/benchmark/verification/stages/pipeline/generate_answer.py`, replace lines 270-272:

```python
# OLD:
            adapter_messages: list[Message] = []
            if answering_model.system_prompt:
                adapter_messages.append(Message.system(answering_model.system_prompt))

# NEW:
            adapter_messages: list[Message] = []
            system_parts: list[str] = []
            if answering_model.system_prompt:
                system_parts.append(answering_model.system_prompt)
            if context.prompt_config:
                gen_instructions = context.prompt_config.get_for_task("generation")
                if gen_instructions:
                    system_parts.append(gen_instructions)
            if system_parts:
                adapter_messages.append(Message.system("\n\n".join(system_parts)))
```

- [ ] **Step 4: Run existing generate_answer tests to check for regressions**

Run: `cd /Users/carli/Projects/karenina-salvage/karenina/.claude/worktrees/dead-code && uv run pytest tests/unit/benchmark/verification/stages/test_generate_answer_routing.py -v -x`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/karenina/benchmark/verification/stages/pipeline/generate_answer.py tests/unit/benchmark/verification/stages/test_generate_answer_prompt_config.py
git commit -m "feat: wire PromptConfig.generation into answer generation stage"
```

---

## Task 3: Remove Dead Parsing System Prompt + Add Instruction Shortcuts

**Files:**
- Modify: `src/karenina/schemas/verification/config.py:320-330,369-380,173`
- Test: `tests/unit/schemas/test_verification_config.py` (modify existing)
- Test: `tests/unit/schemas/test_instruction_shortcuts.py` (create)
- Test: `tests/unit/schemas/test_verification_config_issues.py` (modify existing)

### Step Group A: Remove Dead Auto-Assignment

- [ ] **Step 1: Write failing test for parsing model without system_prompt**

Add to `tests/unit/schemas/test_verification_config.py` (or create `tests/unit/schemas/test_parsing_system_prompt_removal.py`):

```python
"""Tests for parsing model system_prompt relaxation."""

import pytest

from karenina.schemas.config import ModelConfig
from karenina.schemas.verification.config import VerificationConfig


@pytest.mark.unit
class TestParsingSystemPromptRelaxation:
    """Verify parsing models no longer need system_prompt."""

    def test_parsing_model_none_system_prompt_valid(self) -> None:
        """Parsing model with system_prompt=None should not raise."""
        answering = ModelConfig(id="a", model_name="m", model_provider="openai", system_prompt="You are helpful.")
        parsing = ModelConfig(id="p", model_name="m", model_provider="openai", system_prompt=None)
        config = VerificationConfig(answering_models=[answering], parsing_models=[parsing])
        assert config.parsing_models[0].system_prompt is None

    def test_answering_model_still_requires_system_prompt(self) -> None:
        """Answering model with system_prompt=None should still raise."""
        answering = ModelConfig(id="a", model_name="m", model_provider="openai", system_prompt=None)
        parsing = ModelConfig(id="p", model_name="m", model_provider="openai", system_prompt=None)
        with pytest.raises(ValueError, match="[Ss]ystem prompt.*required.*answering"):
            VerificationConfig(answering_models=[answering], parsing_models=[parsing])

    def test_parsing_model_no_auto_assignment(self) -> None:
        """Parsing model should NOT receive DEFAULT_PARSING_SYSTEM_PROMPT."""
        answering = ModelConfig(id="a", model_name="m", model_provider="openai", system_prompt="You are helpful.")
        parsing = ModelConfig(id="p", model_name="m", model_provider="openai")
        config = VerificationConfig(answering_models=[answering], parsing_models=[parsing])
        # Should be None, not auto-assigned
        assert config.parsing_models[0].system_prompt is None
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd /Users/carli/Projects/karenina-salvage/karenina/.claude/worktrees/dead-code && uv run pytest tests/unit/schemas/test_parsing_system_prompt_removal.py -v`

Expected: FAIL (first test raises ValueError, third test gets auto-assigned prompt)

- [ ] **Step 3: Implement Part A**

In `src/karenina/schemas/verification/config.py`:

1. **Remove parsing auto-assignment** (lines 320-330): Delete the `if "parsing_models" in data:` block that copies models with DEFAULT_PARSING_SYSTEM_PROMPT.

2. **Split validation loop** (lines 369-390): Replace the single loop with two separate loops:

```python
# Validate answering model configurations
for model in self.answering_models:
    if not model.model_name:
        raise ValueError(f"Model name is required in model configuration (model: {model.id})")
    from karenina.adapters.registry import AdapterRegistry
    spec = AdapterRegistry.get_spec(model.interface)
    if spec is not None and spec.requires_provider and not model.model_provider:
        raise ValueError(f"Model provider is required for interface '{model.interface}'. (model: {model.id})")
    if not model.system_prompt:
        raise ValueError(f"System prompt is required for answering model {model.id}")

# Validate parsing model configurations (system_prompt not required)
for model in self.parsing_models:
    if not model.model_name:
        raise ValueError(f"Model name is required in model configuration (model: {model.id})")
    from karenina.adapters.registry import AdapterRegistry
    spec = AdapterRegistry.get_spec(model.interface)
    if spec is not None and spec.requires_provider and not model.model_provider:
        raise ValueError(f"Model provider is required for interface '{model.interface}'. (model: {model.id})")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/carli/Projects/karenina-salvage/karenina/.claude/worktrees/dead-code && uv run pytest tests/unit/schemas/test_parsing_system_prompt_removal.py -v`

Expected: PASS

- [ ] **Step 5: Fix existing tests that assert auto-assignment**

Search for assertions matching `== DEFAULT_PARSING_SYSTEM_PROMPT` in the test files:
- In `tests/unit/schemas/test_verification_config.py`: find `assert config.parsing_models[0].system_prompt == DEFAULT_PARSING_SYSTEM_PROMPT` and change to `assert config.parsing_models[0].system_prompt is None`.
- In `tests/unit/schemas/test_verification_config_issues.py`: same pattern, change to `assert ... is None`.

- [ ] **Step 6: Run full schema test suite**

Run: `cd /Users/carli/Projects/karenina-salvage/karenina/.claude/worktrees/dead-code && uv run pytest tests/unit/schemas/ -v -x`

Expected: PASS

- [ ] **Step 7: Commit Part A**

```bash
git add src/karenina/schemas/verification/config.py tests/unit/schemas/test_parsing_system_prompt_removal.py tests/unit/schemas/test_verification_config.py tests/unit/schemas/test_verification_config_issues.py
git commit -m "fix: remove dead DEFAULT_PARSING_SYSTEM_PROMPT auto-assignment for parsing models"
```

### Step Group B: Add Instruction Shortcuts

- [ ] **Step 8: Write failing tests for shortcuts**

Create `tests/unit/schemas/test_instruction_shortcuts.py`:

```python
"""Tests for VerificationConfig instruction shortcut fields."""

import pytest

from karenina.schemas.config import ModelConfig
from karenina.schemas.verification.config import VerificationConfig
from karenina.schemas.verification.prompt_config import PromptConfig


def _make_models() -> tuple[ModelConfig, ModelConfig]:
    answering = ModelConfig(id="a", model_name="m", model_provider="openai", system_prompt="You are helpful.")
    parsing = ModelConfig(id="p", model_name="m", model_provider="openai")
    return answering, parsing


@pytest.mark.unit
class TestInstructionShortcuts:
    """Test that *_instructions shortcuts wire into PromptConfig."""

    def test_parsing_instructions_creates_prompt_config(self) -> None:
        a, p = _make_models()
        config = VerificationConfig(
            answering_models=[a], parsing_models=[p],
            parsing_instructions="Be strict",
        )
        assert config.prompt_config is not None
        assert config.prompt_config.parsing == "Be strict"

    def test_generation_instructions_creates_prompt_config(self) -> None:
        a, p = _make_models()
        config = VerificationConfig(
            answering_models=[a], parsing_models=[p],
            generation_instructions="Focus on accuracy",
        )
        assert config.prompt_config is not None
        assert config.prompt_config.generation == "Focus on accuracy"

    def test_multiple_shortcuts_merge(self) -> None:
        a, p = _make_models()
        config = VerificationConfig(
            answering_models=[a], parsing_models=[p],
            parsing_instructions="Be strict",
            generation_instructions="Focus on accuracy",
        )
        assert config.prompt_config is not None
        assert config.prompt_config.parsing == "Be strict"
        assert config.prompt_config.generation == "Focus on accuracy"

    def test_explicit_prompt_config_takes_precedence(self) -> None:
        a, p = _make_models()
        config = VerificationConfig(
            answering_models=[a], parsing_models=[p],
            prompt_config=PromptConfig(parsing="Explicit"),
            parsing_instructions="Shortcut",
        )
        assert config.prompt_config.parsing == "Explicit"

    def test_shortcut_fills_unset_prompt_config_fields(self) -> None:
        a, p = _make_models()
        config = VerificationConfig(
            answering_models=[a], parsing_models=[p],
            prompt_config=PromptConfig(parsing="Explicit"),
            generation_instructions="From shortcut",
        )
        assert config.prompt_config.parsing == "Explicit"
        assert config.prompt_config.generation == "From shortcut"

    def test_all_shortcut_fields_exist(self) -> None:
        a, p = _make_models()
        config = VerificationConfig(
            answering_models=[a], parsing_models=[p],
            generation_instructions="g",
            parsing_instructions="p",
            abstention_detection_instructions="a",
            sufficiency_detection_instructions="s",
            rubric_evaluation_instructions="r",
            agentic_parsing_instructions="ap",
            deep_judgment_instructions="dj",
        )
        pc = config.prompt_config
        assert pc is not None
        assert pc.generation == "g"
        assert pc.parsing == "p"
        assert pc.abstention_detection == "a"
        assert pc.sufficiency_detection == "s"
        assert pc.rubric_evaluation == "r"
        assert pc.agentic_parsing == "ap"
        assert pc.deep_judgment == "dj"
```

- [ ] **Step 9: Run to verify they fail**

Run: `cd /Users/carli/Projects/karenina-salvage/karenina/.claude/worktrees/dead-code && uv run pytest tests/unit/schemas/test_instruction_shortcuts.py -v`

Expected: FAIL (fields don't exist on VerificationConfig yet)

- [ ] **Step 10: Implement Part B**

In `src/karenina/schemas/verification/config.py`, use the pop-before-init pattern (same as existing `rubric_enabled` and `deep_judgment_rubric_search_enabled` handling). Do NOT declare shortcut fields on the model; pop them from `data` in `__init__` before `super().__init__()`.

In `__init__`, add the shortcut wiring before `super().__init__(**data)`:

```python
        # Wire instruction shortcuts into prompt_config (pop before super().__init__)
        # These are NOT declared as model fields; they are consumed here and removed from data.
        shortcut_mapping = {
            "generation": data.pop("generation_instructions", None),
            "parsing": data.pop("parsing_instructions", None),
            "abstention_detection": data.pop("abstention_detection_instructions", None),
            "sufficiency_detection": data.pop("sufficiency_detection_instructions", None),
            "rubric_evaluation": data.pop("rubric_evaluation_instructions", None),
            "agentic_parsing": data.pop("agentic_parsing_instructions", None),
            "deep_judgment": data.pop("deep_judgment_instructions", None),
        }
        active_shortcuts = {k: v for k, v in shortcut_mapping.items() if v is not None}
        if active_shortcuts:
            pc = data.get("prompt_config") or PromptConfig()
            if isinstance(pc, dict):
                pc = PromptConfig(**pc)
            updates = {k: v for k, v in active_shortcuts.items() if getattr(pc, k) is None}
            if updates:
                pc = pc.model_copy(update=updates)
            data["prompt_config"] = pc
```

This follows the same pattern as `data.pop("rubric_enabled", None)` and `data.pop("deep_judgment_rubric_search_enabled", None)` already in this `__init__`.

- [ ] **Step 11: Run tests to verify they pass**

Run: `cd /Users/carli/Projects/karenina-salvage/karenina/.claude/worktrees/dead-code && uv run pytest tests/unit/schemas/test_instruction_shortcuts.py -v`

Expected: PASS

- [ ] **Step 12: Run full schema test suite**

Run: `cd /Users/carli/Projects/karenina-salvage/karenina/.claude/worktrees/dead-code && uv run pytest tests/unit/schemas/ -v -x`

Expected: PASS

- [ ] **Step 13: Commit Part B**

```bash
git add src/karenina/schemas/verification/config.py tests/unit/schemas/test_instruction_shortcuts.py
git commit -m "feat: add instruction shortcut fields on VerificationConfig"
```

---

## Task 4: FewShotConfig Restructure

**Files:**
- Modify: `src/karenina/schemas/config/models.py` (FewShotConfig, QuestionFewShotConfig)
- Modify: `src/karenina/schemas/verification/config.py:387-398,566-572,596-604`
- Modify: `src/karenina/benchmark/verification/utils/task_helpers.py:111`
- Modify: `src/karenina/cli/interactive.py:413-419`
- Test: `tests/unit/schemas/test_few_shot_restructure.py` (create)
- Modify: `tests/unit/schemas/test_kshot_reproducibility.py`
- Modify: `tests/unit/schemas/test_verification_config.py`
- Modify: `tests/unit/benchmark/test_task_eval_metadata.py`

### Step Group A: Core Model Restructure

- [ ] **Step 1: Write failing tests for new FewShotConfig structure**

Create `tests/unit/schemas/test_few_shot_restructure.py`:

```python
"""Tests for FewShotConfig restructure with source/pool_mode/pool_k semantics."""

import pytest

from karenina.schemas.config.models import FewShotConfig, QuestionFewShotConfig


@pytest.mark.unit
class TestFewShotConfigNewFields:
    """Test the new field names and semantics."""

    def test_default_source_is_both(self) -> None:
        config = FewShotConfig()
        assert config.source == "both"

    def test_default_pool_mode_is_all(self) -> None:
        config = FewShotConfig()
        assert config.pool_mode == "all"

    def test_default_pool_k_is_3(self) -> None:
        config = FewShotConfig()
        assert config.pool_k == 3

    def test_old_field_names_rejected(self) -> None:
        with pytest.raises(Exception):
            FewShotConfig(enabled=True)  # type: ignore[call-arg]
        with pytest.raises(Exception):
            FewShotConfig(global_mode="all")  # type: ignore[call-arg]
        with pytest.raises(Exception):
            FewShotConfig(global_k=5)  # type: ignore[call-arg]
        with pytest.raises(Exception):
            FewShotConfig(global_external_examples=[])  # type: ignore[call-arg]

    def test_question_config_no_external_examples(self) -> None:
        with pytest.raises(Exception):
            QuestionFewShotConfig(external_examples=[])  # type: ignore[call-arg]

    def test_question_config_no_mode_none(self) -> None:
        """mode='none' should not be valid."""
        with pytest.raises(Exception):
            QuestionFewShotConfig(mode="none")  # type: ignore[arg-type]


@pytest.mark.unit
class TestFewShotSourceBehavior:
    """Test resolve_examples_for_question with different source values."""

    def _pool_examples(self) -> list[dict[str, str]]:
        return [
            {"question": "What is X?", "answer": "X is A."},
            {"question": "What is Y?", "answer": "Y is B."},
        ]

    def test_disabled_returns_empty(self) -> None:
        config = FewShotConfig(source="disabled")
        result = config.resolve_examples_for_question("q1", self._pool_examples())
        assert result == []

    def test_question_pool_returns_pool_only(self) -> None:
        config = FewShotConfig(
            source="question_pool",
            pool_mode="all",
            global_examples=[{"question": "Global?", "answer": "Global."}],
        )
        result = config.resolve_examples_for_question("q1", self._pool_examples())
        assert len(result) == 2
        assert all(ex in self._pool_examples() for ex in result)

    def test_global_returns_global_only(self) -> None:
        global_ex = [{"question": "Global?", "answer": "Global."}]
        config = FewShotConfig(source="global", global_examples=global_ex)
        result = config.resolve_examples_for_question("q1", self._pool_examples())
        assert result == global_ex

    def test_both_returns_pool_and_global(self) -> None:
        global_ex = [{"question": "Global?", "answer": "Global."}]
        config = FewShotConfig(source="both", pool_mode="all", global_examples=global_ex)
        result = config.resolve_examples_for_question("q1", self._pool_examples())
        assert len(result) == 3
        assert result[:2] == self._pool_examples()
        assert result[2:] == global_ex

    def test_pool_mode_kshot(self) -> None:
        config = FewShotConfig(source="question_pool", pool_mode="k-shot", pool_k=1)
        result = config.resolve_examples_for_question("q1", self._pool_examples())
        assert len(result) == 1

    def test_pool_mode_custom(self) -> None:
        config = FewShotConfig(
            source="question_pool",
            pool_mode="custom",
            question_configs={"q1": QuestionFewShotConfig(mode="custom", selected_examples=[0])},
        )
        result = config.resolve_examples_for_question("q1", self._pool_examples())
        assert len(result) == 1
        assert result[0] == self._pool_examples()[0]

    def test_inherit_uses_pool_mode(self) -> None:
        config = FewShotConfig(source="question_pool", pool_mode="k-shot", pool_k=1)
        # Default question config uses mode="inherit", inherits pool_mode="k-shot"
        result = config.resolve_examples_for_question("q1", self._pool_examples())
        assert len(result) == 1  # 1 pool example, no globals (source="question_pool")


@pytest.mark.unit
class TestFewShotFactoryMethods:
    """Test that factory methods use new field names."""

    def test_from_index_selections(self) -> None:
        config = FewShotConfig.from_index_selections({"q1": [0, 1]})
        assert config.pool_mode == "custom"
        assert "q1" in config.question_configs

    def test_from_hash_selections(self) -> None:
        config = FewShotConfig.from_hash_selections({"q1": ["abc123"]})
        assert config.pool_mode == "custom"

    def test_k_shot_for_questions(self) -> None:
        config = FewShotConfig.k_shot_for_questions({"q1": 5}, pool_k=3)
        assert config.pool_mode == "k-shot"
        assert config.pool_k == 3
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd /Users/carli/Projects/karenina-salvage/karenina/.claude/worktrees/dead-code && uv run pytest tests/unit/schemas/test_few_shot_restructure.py -v`

Expected: FAIL (old field names still present)

- [ ] **Step 3: Implement core model changes in models.py**

In `src/karenina/schemas/config/models.py`, update `QuestionFewShotConfig`:

```python
class QuestionFewShotConfig(BaseModel):
    """Per-question few-shot configuration."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["all", "k-shot", "custom", "inherit"] = "inherit"
    k: int | None = None
    selected_examples: list[str | int] | None = None
    excluded_examples: list[str | int] | None = None
```

Update `FewShotConfig`:

```python
class FewShotConfig(BaseModel):
    """Flexible configuration for few-shot prompting with convenient bulk setup interface."""

    model_config = ConfigDict(extra="forbid")

    # Master source selector: what types of examples are active
    source: Literal["disabled", "question_pool", "global", "both"] = "both"

    # Pool selection settings
    pool_mode: Literal["all", "k-shot", "custom"] = "all"
    pool_k: int = 3

    # Per-question configurations
    question_configs: dict[str, QuestionFewShotConfig] = Field(default_factory=dict)

    # Global examples appended to ALL questions
    global_examples: list[dict[str, str]] = Field(default_factory=list)
```

Update factory methods: rename `global_mode` to `pool_mode`, `global_k` to `pool_k` in all of `from_index_selections`, `from_hash_selections`, `k_shot_for_questions`.

Update `get_effective_config`: replace `self.global_mode` with `self.pool_mode`, `self.global_k` with `self.pool_k`, remove `external_examples`.

Replace the body of `resolve_examples_for_question` with source-based gating:

```python
    def resolve_examples_for_question(
        self,
        question_id: str,
        available_examples: list[dict[str, str]] | None = None,
        question_text: str | None = None,
    ) -> list[dict[str, str]]:
        """Resolve the final list of examples to use for a specific question."""
        import hashlib
        import random

        if self.source == "disabled":
            return []

        include_pool = self.source in ("question_pool", "both")
        include_global = self.source in ("global", "both")

        resolved_examples: list[dict[str, str]] = []

        # Pool selection (only when source includes question_pool)
        if include_pool:
            if available_examples is None:
                available_examples = []

            effective_config = self.get_effective_config(question_id)

            # Create lookup for hash-to-example mapping if needed
            example_hash_map: dict[str, int] = {}
            if question_text is not None and available_examples:
                for i, example in enumerate(available_examples):
                    example_question = example.get("question", "")
                    example_hash = hashlib.md5(example_question.encode("utf-8")).hexdigest()
                    example_hash_map[example_hash] = i

            if effective_config.mode == "all":
                resolved_examples = available_examples.copy()
            elif effective_config.mode == "k-shot":
                k = effective_config.k if effective_config.k is not None else self.pool_k
                if len(available_examples) <= k:
                    resolved_examples = available_examples.copy()
                else:
                    random.seed(hash(question_id) & 0x7FFFFFFF)
                    resolved_examples = random.sample(available_examples, k)
            elif effective_config.mode == "custom" and effective_config.selected_examples:
                for selection in effective_config.selected_examples:
                    if isinstance(selection, int):
                        if 0 <= selection < len(available_examples):
                            resolved_examples.append(available_examples[selection])
                    elif isinstance(selection, str) and selection in example_hash_map:
                        idx = example_hash_map[selection]
                        resolved_examples.append(available_examples[idx])

            # Apply exclusions if specified
            if effective_config.excluded_examples:
                exclusion_indices: set[int] = set()
                for exclusion in effective_config.excluded_examples:
                    if isinstance(exclusion, int):
                        exclusion_indices.add(exclusion)
                    elif isinstance(exclusion, str) and exclusion in example_hash_map:
                        exclusion_indices.add(example_hash_map[exclusion])
                resolved_examples = [
                    ex for i, ex in enumerate(resolved_examples) if i not in exclusion_indices
                ]

        # Append global examples (only when source includes global)
        final_examples = resolved_examples.copy()
        if include_global and self.global_examples:
            final_examples.extend(self.global_examples)

        return final_examples
```

Update `validate_selections`: remove `external_examples` references from the validation loop.

Update mutation methods (`add_selections_by_index`, etc.): no field name changes needed (they only set question_configs).

- [ ] **Step 4: Run new tests to verify they pass**

Run: `cd /Users/carli/Projects/karenina-salvage/karenina/.claude/worktrees/dead-code && uv run pytest tests/unit/schemas/test_few_shot_restructure.py -v`

Expected: PASS

- [ ] **Step 5: Commit core changes**

```bash
git add src/karenina/schemas/config/models.py tests/unit/schemas/test_few_shot_restructure.py
git commit -m "refactor: restructure FewShotConfig with source/pool_mode/pool_k semantics"
```

### Step Group B: Update Callers

- [ ] **Step 6: Update config.py callers**

In `src/karenina/schemas/verification/config.py`:

1. **_validate_config (line 387-398)**: Replace `self.few_shot_config.enabled` with `self.few_shot_config.source != "disabled"`, `global_mode` with `pool_mode`, `global_k` with `pool_k`.

2. **__repr__ (lines 568-574)**: Replace `few_shot_config.enabled` with `source != "disabled"`, `global_mode` with `pool_mode`, `global_k` with `pool_k`.

3. **is_few_shot_enabled (line 604)**: Replace `config.enabled` with `config.source != "disabled"`.

- [ ] **Step 7: Update task_helpers.py**

In `src/karenina/benchmark/verification/utils/task_helpers.py:111`:

```python
# OLD:
if not few_shot_config or not few_shot_config.enabled:

# NEW:
if not few_shot_config or few_shot_config.source == "disabled":
```

- [ ] **Step 8: Update interactive.py**

In `src/karenina/cli/interactive.py:413-419`:

```python
# OLD:
few_shot_mode_str = Prompt.ask("Few-shot mode", choices=["all", "k-shot", "custom", "none"], default="all")
few_shot_mode = cast(Literal["all", "k-shot", "custom", "none"], few_shot_mode_str)
few_shot_k = _prompt_int_min("Few-shot k (number of examples)", min_val=1, default="3")
return FewShotConfig(enabled=True, global_mode=few_shot_mode, global_k=few_shot_k)

# NEW:
few_shot_mode_str = Prompt.ask("Few-shot mode", choices=["all", "k-shot", "custom"], default="all")
few_shot_mode = cast(Literal["all", "k-shot", "custom"], few_shot_mode_str)
few_shot_k = _prompt_int_min("Few-shot k (number of examples)", min_val=1, default="3")
return FewShotConfig(source="both", pool_mode=few_shot_mode, pool_k=few_shot_k)
```

- [ ] **Step 9: Update existing tests**

1. **`tests/unit/schemas/test_kshot_reproducibility.py`**: Replace `FewShotConfig(global_mode="k-shot", global_k=3)` with `FewShotConfig(source="question_pool", pool_mode="k-shot", pool_k=3)`.

2. **`tests/unit/schemas/test_verification_config.py`**:
   - Replace `FewShotConfig(enabled=True, global_mode="all", global_k=3)` with `FewShotConfig(source="both", pool_mode="all", pool_k=3)`.
   - Replace `assert result.enabled is True` with `assert result.source == "both"`.
   - Replace `assert result.global_mode == "all"` with `assert result.pool_mode == "all"`.

3. **`tests/unit/benchmark/test_task_eval_metadata.py`**: Replace `FewShotConfig(enabled=True)` with `FewShotConfig(source="both")`.

- [ ] **Step 10: Run full test suite**

Run: `cd /Users/carli/Projects/karenina-salvage/karenina/.claude/worktrees/dead-code && uv run pytest tests/unit/ -v -x`

Expected: PASS

- [ ] **Step 11: Commit caller updates**

```bash
git add src/karenina/schemas/verification/config.py src/karenina/benchmark/verification/utils/task_helpers.py src/karenina/cli/interactive.py tests/unit/schemas/test_kshot_reproducibility.py tests/unit/schemas/test_verification_config.py tests/unit/benchmark/test_task_eval_metadata.py
git commit -m "refactor: update all FewShotConfig callers to new field names"
```

---

## Task 5: Final Verification

- [ ] **Step 1: Run full test suite**

Run: `cd /Users/carli/Projects/karenina-salvage/karenina/.claude/worktrees/dead-code && uv run pytest tests/ -v -x --timeout=60`

Expected: PASS

- [ ] **Step 2: Run type checker**

Run: `cd /Users/carli/Projects/karenina-salvage/karenina/.claude/worktrees/dead-code && uv run mypy src/karenina/schemas/config/models.py src/karenina/schemas/verification/config.py src/karenina/benchmark/verification/runner.py src/karenina/benchmark/verification/stages/pipeline/generate_answer.py`

Expected: No errors

- [ ] **Step 3: Run dead code checker**

Run: `cd /Users/carli/Projects/karenina-salvage/karenina/.claude/worktrees/dead-code && uv run vulture src/karenina/ vulture_whitelist.py`

Expected: No new dead code introduced
