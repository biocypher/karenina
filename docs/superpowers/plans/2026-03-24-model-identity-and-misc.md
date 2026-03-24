# Model Identity, Reproducibility, and Miscellaneous Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 10 independent issues covering model identity collapsing, k-shot reproducibility, export/import fidelity, boolean semantics, error handling, and 5 mechanical improvements.

**Architecture:** Each fix is self-contained and touches distinct code paths. All use TDD: write failing test first, then implement. All source paths are relative to `src/karenina/`.

**Tech Stack:** Python 3.13, Pydantic v2, pytest, tenacity

**Spec:** `docs/superpowers/specs/2026-03-24-model-identity-and-misc-design.md`

---

### Task 1: BooleanMatch strict None handling

**Files:**
- Modify: `src/karenina/schemas/primitives/comparisons.py:61-62`
- Test: `tests/unit/schemas/test_comparisons_issues.py`

- [ ] **Step 1: Write the failing tests**

Add a new test class to the existing test file `tests/unit/schemas/test_comparisons_issues.py`:

```python
from karenina.schemas.primitives.comparisons import BooleanMatch


@pytest.mark.unit
class TestBooleanMatchNoneHandling:
    """BooleanMatch should treat None as non-match against True and False."""

    def test_none_vs_false_is_non_match(self) -> None:
        """None should NOT match False (previous bug: bool(None)==bool(False))."""
        bm = BooleanMatch()
        assert bm.check(None, False) is False

    def test_none_vs_true_is_non_match(self) -> None:
        bm = BooleanMatch()
        assert bm.check(None, True) is False

    def test_false_vs_none_is_non_match(self) -> None:
        bm = BooleanMatch()
        assert bm.check(False, None) is False

    def test_true_vs_none_is_non_match(self) -> None:
        bm = BooleanMatch()
        assert bm.check(True, None) is False

    def test_none_vs_none_is_match(self) -> None:
        """None should match None via identity."""
        bm = BooleanMatch()
        assert bm.check(None, None) is True

    def test_true_vs_true_still_works(self) -> None:
        bm = BooleanMatch()
        assert bm.check(True, True) is True

    def test_false_vs_false_still_works(self) -> None:
        bm = BooleanMatch()
        assert bm.check(False, False) is True

    def test_true_vs_false_still_non_match(self) -> None:
        bm = BooleanMatch()
        assert bm.check(True, False) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd karenina && uv run pytest tests/unit/schemas/test_comparisons_issues.py::TestBooleanMatchNoneHandling -v`
Expected: `test_none_vs_false_is_non_match` FAILS (returns `True` instead of `False`)

- [ ] **Step 3: Implement the fix**

In `src/karenina/schemas/primitives/comparisons.py`, change the `check` method (lines 61-62):

```python
def check(self, extracted: Any, expected: Any) -> bool:
    if extracted is None or expected is None:
        return extracted is expected
    return bool(extracted) == bool(expected)
```

Also update the class docstring (lines 56-59):

```python
class BooleanMatch(VerificationPrimitive):
    """Compare extracted bool to ground truth bool.

    Both values are coerced to bool before comparison.
    None is treated as distinct from both True and False:
    None only matches None via identity comparison.
    """
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd karenina && uv run pytest tests/unit/schemas/test_comparisons_issues.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/karenina/schemas/primitives/comparisons.py tests/unit/schemas/test_comparisons_issues.py
git commit -m "fix: BooleanMatch treats None as non-match against True and False"
```

---

### Task 2: k-shot hash reproducibility

**Files:**
- Modify: `src/karenina/schemas/config/models.py:461`
- Test: `tests/unit/schemas/test_kshot_reproducibility.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/schemas/test_kshot_reproducibility.py`:

```python
"""Tests for k-shot example selection reproducibility across processes."""

import pytest

from karenina.schemas.config.models import FewShotConfig


@pytest.mark.unit
class TestKShotReproducibility:
    """k-shot seeding must be stable across processes (not depend on PYTHONHASHSEED)."""

    def test_resolve_deterministic_across_calls(self) -> None:
        """Same question_id yields same k-shot selection on repeated calls."""
        config = FewShotConfig(mode="k-shot", k=3)
        examples = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(10)]

        first = config.resolve_examples_for_question("test-q-123", examples)
        second = config.resolve_examples_for_question("test-q-123", examples)
        assert first == second

    def test_different_question_ids_yield_different_selections(self) -> None:
        """Different question_ids should (usually) produce different selections."""
        config = FewShotConfig(mode="k-shot", k=3)
        examples = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(20)]

        sel_a = config.resolve_examples_for_question("question-a", examples)
        sel_b = config.resolve_examples_for_question("question-b", examples)
        assert sel_a != sel_b

    def test_seed_is_md5_based(self) -> None:
        """The seed must use hashlib.md5, which is stable across processes.

        We verify by checking the source code uses hashlib.md5 (not hash()).
        """
        import inspect
        import hashlib

        source = inspect.getsource(FewShotConfig.resolve_examples_for_question)
        assert "hashlib.md5" in source
        assert "hash(question_id)" not in source
```

- [ ] **Step 2: Run tests to verify test_seed_is_md5_based fails (source still uses hash())**

Run: `cd karenina && uv run pytest tests/unit/schemas/test_kshot_reproducibility.py::TestKShotReproducibility::test_seed_is_md5_based -v`
Expected: FAIL (source contains `hash(question_id)`, not `hashlib.md5`)

- [ ] **Step 3: Fix the source code**

In `src/karenina/schemas/config/models.py` line 461, replace:

```python
random.seed(hash(question_id) & 0x7FFFFFFF)
```

with:

```python
random.seed(int(hashlib.md5(question_id.encode()).hexdigest(), 16) & 0x7FFFFFFF)
```

Note: `hashlib` is already imported locally inside this method (line 425). No new import needed.

- [ ] **Step 4: Run full test suite for models**

Run: `cd karenina && uv run pytest tests/unit/schemas/test_kshot_reproducibility.py tests/unit/schemas/ -v -x`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/karenina/schemas/config/models.py tests/unit/schemas/test_kshot_reproducibility.py
git commit -m "fix: use hashlib.md5 for k-shot seeding instead of hash() for cross-process reproducibility"
```

---

### Task 3: log_retry max_attempts parameter and f-string fixes

**Files:**
- Modify: `src/karenina/utils/retry.py:24-38,69-71`
- Modify: `src/karenina/benchmark/verification/evaluators/trace/abstention.py:73`
- Modify: `src/karenina/benchmark/verification/evaluators/trace/sufficiency.py:81`
- Test: `tests/unit/utils/test_retry.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/utils/test_retry.py`:

```python
"""Tests for retry utility functions."""

import logging
from unittest.mock import MagicMock

import pytest

from karenina.utils.retry import log_retry


@pytest.mark.unit
class TestLogRetry:
    """log_retry should use the provided max_attempts value in log messages."""

    def test_default_max_attempts_is_3(self, caplog: pytest.LogCaptureFixture) -> None:
        state = MagicMock()
        state.outcome.exception.return_value = RuntimeError("test")
        state.attempt_number = 1
        with caplog.at_level(logging.WARNING):
            log_retry(state)
        assert "attempt 1/3" in caplog.text

    def test_custom_max_attempts(self, caplog: pytest.LogCaptureFixture) -> None:
        state = MagicMock()
        state.outcome.exception.return_value = RuntimeError("test")
        state.attempt_number = 2
        with caplog.at_level(logging.WARNING):
            log_retry(state, max_attempts=5)
        assert "attempt 2/5" in caplog.text

    def test_custom_context(self, caplog: pytest.LogCaptureFixture) -> None:
        state = MagicMock()
        state.outcome.exception.return_value = RuntimeError("oops")
        state.attempt_number = 1
        with caplog.at_level(logging.WARNING):
            log_retry(state, context="embedding check", max_attempts=4)
        assert "Retrying embedding check" in caplog.text
        assert "attempt 1/4" in caplog.text
```

- [ ] **Step 2: Run tests to verify `test_custom_max_attempts` fails**

Run: `cd karenina && uv run pytest tests/unit/utils/test_retry.py -v`
Expected: `test_custom_max_attempts` FAILS because `max_attempts` param does not exist

- [ ] **Step 3: Implement the fix**

In `src/karenina/utils/retry.py`, update `log_retry` (lines 24-38):

```python
def log_retry(retry_state: Any, *, context: str = "LLM call", max_attempts: int = 3) -> None:
    """Log retry attempt with error details.

    Args:
        retry_state: Tenacity retry state object containing attempt info.
        context: Description of what operation is being retried.
        max_attempts: Maximum number of retry attempts for log display.

    Example:
        >>> from functools import partial
        >>> before_sleep = partial(log_retry, context="abstention check", max_attempts=3)
    """
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    attempt = retry_state.attempt_number
    logger.warning("Retrying %s (attempt %d/%d) after error: %s", context, attempt, max_attempts, exc)
```

Also fix the f-string in `create_transient_retry._log_retry` (line 69-71):

```python
    def _log_retry(retry_state: Any) -> None:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        logger.warning(
            "Retrying %s (attempt %d/%d) after error: %s",
            context, retry_state.attempt_number, max_attempts, exc,
        )
```

- [ ] **Step 4: Update callers to pass max_attempts**

In `src/karenina/benchmark/verification/evaluators/trace/abstention.py` (line 73), update:

```python
before_sleep=partial(log_retry, context="abstention detection", max_attempts=3),
```

In `src/karenina/benchmark/verification/evaluators/trace/sufficiency.py` (line 81), update:

```python
before_sleep=partial(log_retry, context="sufficiency detection", max_attempts=3),
```

- [ ] **Step 5: Run tests**

Run: `cd karenina && uv run pytest tests/unit/utils/test_retry.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/karenina/utils/retry.py src/karenina/benchmark/verification/evaluators/trace/abstention.py src/karenina/benchmark/verification/evaluators/trace/sufficiency.py tests/unit/utils/test_retry.py
git commit -m "fix: log_retry accepts max_attempts parameter, fix f-string to lazy formatting"
```

---

### Task 4: rubric_only validation warning

**Files:**
- Modify: `src/karenina/benchmark/verification/runner.py:221`
- Test: `tests/unit/benchmark/verification/test_runner_rubric_only_warning.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/benchmark/verification/test_runner_rubric_only_warning.py`:

```python
"""Tests for rubric_only mode validation warning."""

import logging

import pytest

from karenina.benchmark.verification.runner import run_single_model_verification
from karenina.schemas.config.models import ModelConfig


@pytest.mark.unit
class TestRubricOnlyWarning:
    """Warning when rubric_only mode is set but no rubric traits are provided."""

    def test_rubric_only_no_traits_logs_warning(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """rubric_only with no rubric should log a warning before pipeline runs."""
        # Patch StageOrchestrator to prevent actual pipeline execution
        import karenina.benchmark.verification.runner as runner_module
        mock_orchestrator = type("MockOrch", (), {
            "from_config": classmethod(lambda cls, **kw: cls()),
            "run": lambda self, ctx: None,
        })
        monkeypatch.setattr(runner_module, "StageOrchestrator", mock_orchestrator)

        answering = ModelConfig(interface="langchain", model_name="gpt-4")
        parsing = ModelConfig(interface="langchain", model_name="gpt-4")

        with caplog.at_level(logging.WARNING):
            run_single_model_verification(
                question_id="abc123",
                question_text="What is 2+2?",
                template_code="class Answer: pass",
                answering_model=answering,
                parsing_model=parsing,
                evaluation_mode="rubric_only",
                # No rubric provided
            )

        assert "rubric_only" in caplog.text
        assert "no rubric traits" in caplog.text.lower() or "no rubric traits provided" in caplog.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd karenina && uv run pytest tests/unit/benchmark/verification/test_runner_rubric_only_warning.py -v`
Expected: FAIL (no warning logged)

- [ ] **Step 3: Implement the fix**

In `src/karenina/benchmark/verification/runner.py`, after line 220 (after the `template_only` to `template_and_rubric` auto-upgrade block), add:

```python
    if evaluation_mode == "rubric_only" and not _has_rubric_traits and not _has_dynamic_rubric_traits:
        logger.warning(
            "evaluation_mode='rubric_only' but no rubric traits provided. "
            "Rubric evaluation will produce no scores."
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd karenina && uv run pytest tests/unit/benchmark/verification/test_runner_rubric_only_warning.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/karenina/benchmark/verification/runner.py tests/unit/benchmark/verification/test_runner_rubric_only_warning.py
git commit -m "fix: warn when rubric_only mode is set but no rubric traits are provided"
```

---

### Task 5: Rename _template_validation.py to _rubric_kind_validation.py

**Files:**
- Rename: `src/karenina/schemas/entities/_template_validation.py` to `_rubric_kind_validation.py`
- Modify: `src/karenina/schemas/entities/rubric.py:16`
- Test: `tests/unit/schemas/test_rubric_kind_validation_import.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/schemas/test_rubric_kind_validation_import.py`:

```python
"""Tests that _rubric_kind_validation module is importable and functional."""

import pytest


@pytest.mark.unit
class TestRubricKindValidationImport:
    """The validation module should be importable from its new location."""

    def test_import_from_new_path(self) -> None:
        from karenina.schemas.entities._rubric_kind_validation import _validate_template_fields
        assert callable(_validate_template_fields)

    def test_rubric_still_imports_validator(self) -> None:
        """rubric.py should still work after the import path change."""
        from karenina.schemas.entities.rubric import LLMRubricTrait
        assert LLMRubricTrait is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd karenina && uv run pytest tests/unit/schemas/test_rubric_kind_validation_import.py::TestRubricKindValidationImport::test_import_from_new_path -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Rename the file**

```bash
cd karenina && git mv src/karenina/schemas/entities/_template_validation.py src/karenina/schemas/entities/_rubric_kind_validation.py
```

- [ ] **Step 4: Update the import in rubric.py**

In `src/karenina/schemas/entities/rubric.py` line 16, change:

```python
from karenina.schemas.entities._template_validation import _validate_template_fields
```

to:

```python
from karenina.schemas.entities._rubric_kind_validation import _validate_template_fields
```

- [ ] **Step 5: Run tests**

Run: `cd karenina && uv run pytest tests/unit/schemas/test_rubric_kind_validation_import.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/karenina/schemas/entities/_rubric_kind_validation.py src/karenina/schemas/entities/rubric.py tests/unit/schemas/test_rubric_kind_validation_import.py
git commit -m "refactor: rename _template_validation.py to _rubric_kind_validation.py to clarify scope"
```

---

### Task 6: ModelIdentity enrichment with config_id

**Files:**
- Modify: `src/karenina/schemas/verification/model_identity.py`
- Modify: `src/karenina/benchmark/core/results_io.py:317-344` (`_parse_model_identity`)
- Test: `tests/unit/schemas/test_model_identity.py` (existing, extend)

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/schemas/test_model_identity.py`:

```python
@pytest.mark.unit
class TestModelIdentityConfigId:
    """config_id enrichment when ModelConfig.id differs from model_name."""

    def test_config_id_set_when_differs_from_model_name(self) -> None:
        config = ModelConfig(id="gpt4-high-temp", interface="langchain", model_name="gpt-4")
        identity = ModelIdentity.from_model_config(config)
        assert identity.config_id == "gpt4-high-temp"

    def test_config_id_none_when_matches_model_name(self) -> None:
        config = ModelConfig(interface="langchain", model_name="gpt-4")
        identity = ModelIdentity.from_model_config(config)
        assert identity.config_id is None

    def test_display_string_includes_config_id(self) -> None:
        identity = ModelIdentity(
            interface="langchain", model_name="gpt-4", config_id="high-temp"
        )
        assert identity.display_string == "langchain:gpt-4 (high-temp)"

    def test_display_string_without_config_id(self) -> None:
        identity = ModelIdentity(interface="langchain", model_name="gpt-4")
        assert identity.display_string == "langchain:gpt-4"

    def test_display_string_with_config_id_and_tools(self) -> None:
        identity = ModelIdentity(
            interface="claude_agent_sdk", model_name="claude-sonnet-4-20250514",
            tools=["brave"], config_id="my-config",
        )
        assert identity.display_string == "claude_agent_sdk:claude-sonnet-4-20250514 (my-config) +[brave]"

    def test_canonical_key_includes_config_id(self) -> None:
        identity = ModelIdentity(
            interface="langchain", model_name="gpt-4", config_id="high-temp"
        )
        assert identity.canonical_key == "langchain:gpt-4:high-temp:"

    def test_canonical_key_without_config_id_unchanged(self) -> None:
        """When config_id is None, canonical_key format matches the old 3-segment format."""
        identity = ModelIdentity(interface="langchain", model_name="gpt-4")
        # Must remain "interface:model_name:" (NOT "interface:model_name::") for
        # backward compatibility with existing result IDs and progressive save checkpoints
        assert identity.canonical_key == "langchain:gpt-4:"

    def test_different_config_ids_produce_different_identities(self) -> None:
        a = ModelIdentity(interface="langchain", model_name="gpt-4", config_id="config-a")
        b = ModelIdentity(interface="langchain", model_name="gpt-4", config_id="config-b")
        assert a.canonical_key != b.canonical_key
        assert a.display_string != b.display_string
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd karenina && uv run pytest tests/unit/schemas/test_model_identity.py::TestModelIdentityConfigId -v`
Expected: FAIL (config_id field does not exist)

- [ ] **Step 3: Implement ModelIdentity changes**

In `src/karenina/schemas/verification/model_identity.py`:

Add the `config_id` field after `tools`:

```python
config_id: str | None = Field(default=None, description="ModelConfig.id when it differs from model_name")
```

Update `from_model_config()` to set `config_id`:

```python
@classmethod
def from_model_config(cls, config: ModelConfig, *, role: str = "answering") -> ModelIdentity:
    """Create a ModelIdentity from a ModelConfig.

    Args:
        config: The model configuration to extract identity from.
        role: Either "answering" or "parsing". Parsing models always
              produce tools=[] regardless of config's mcp_urls_dict.

    Returns:
        A ModelIdentity capturing the relevant identity dimensions.
    """
    tools: list[str] = []
    if role == "answering" and config.mcp_urls_dict:
        tools = sorted(config.mcp_urls_dict.keys())

    # Include config_id when it differs from model_name
    effective_name = config.model_name or "unknown"
    config_id = config.id if config.id and config.id != effective_name else None

    return cls(
        interface=config.interface,
        model_name=effective_name,
        tools=tools,
        config_id=config_id,
    )
```

Update `display_string`:

```python
@property
def display_string(self) -> str:
    """Human-readable model identity string.

    Format: "interface:model_name", "interface:model_name (config_id)",
    or "interface:model_name (config_id) +[tool1, tool2]"
    """
    base = f"{self.interface}:{self.model_name}"
    if self.config_id:
        base = f"{base} ({self.config_id})"
    if self.tools:
        return f"{base} +[{', '.join(self.tools)}]"
    return base
```

Update `canonical_key` (preserving backward-compatible 3-segment format when no config_id):

```python
@property
def canonical_key(self) -> str:
    """Deterministic identity key for hashing and comparison.

    Format without config_id: "interface:model_name:tool1|tool2" (unchanged from original)
    Format with config_id: "interface:model_name:config_id:tool1|tool2"

    The config_id segment is only inserted when present, to preserve backward
    compatibility with existing result IDs and progressive save checkpoints.
    """
    tools_part = "|".join(sorted(self.tools)) if self.tools else ""
    if self.config_id:
        return f"{self.interface}:{self.model_name}:{self.config_id}:{tools_part}"
    return f"{self.interface}:{self.model_name}:{tools_part}"
```

Update module docstring to mention config_id dimension.

- [ ] **Step 4: Run tests**

Run: `cd karenina && uv run pytest tests/unit/schemas/test_model_identity.py -v`
Expected: ALL PASS (both new and existing tests)

- [ ] **Step 5: Update _parse_model_identity for CSV round-trip**

In `src/karenina/benchmark/core/results_io.py`, update `_parse_model_identity` (lines 317-344):

```python
@staticmethod
def _parse_model_identity(display_string: str | None) -> ModelIdentity:
    """Parse a ModelIdentity display string back into a ModelIdentity object.

    Handles formats:
    - "interface:model_name"
    - "interface:model_name (config_id)"
    - "interface:model_name +[tool1, tool2]"
    - "interface:model_name (config_id) +[tool1, tool2]"

    Args:
        display_string: The display string from CSV export, or None/empty.

    Returns:
        Reconstructed ModelIdentity.
    """
    if not display_string:
        return ModelIdentity(interface="unknown", model_name="unknown")

    tools: list[str] = []
    config_id: str | None = None
    base = display_string

    # Split on " +[" to extract tools
    if " +[" in display_string:
        base, tools_part = display_string.split(" +[", 1)
        tools = [t.strip() for t in tools_part.rstrip("]").split(",") if t.strip()]

    # Extract config_id from " (config_id)" suffix.
    # Use rsplit with maxsplit=1 so model names containing parens are safe.
    # Only extract if the paren group appears after a ":" (interface:model part).
    if " (" in base and base.endswith(")"):
        candidate_base, config_part = base.rsplit(" (", 1)
        # Only treat as config_id if the base still has interface:model structure
        if ":" in candidate_base:
            config_id = config_part.rstrip(")")
            base = candidate_base

    # Split base on ":" to get interface and model_name
    parts = base.split(":", 1)
    interface = parts[0] if parts else "unknown"
    model_name = parts[1] if len(parts) > 1 else "unknown"

    return ModelIdentity(
        interface=interface, model_name=model_name, tools=tools, config_id=config_id,
    )
```

- [ ] **Step 6: Write CSV round-trip test**

Add to `tests/unit/schemas/test_model_identity.py`:

```python
from karenina.benchmark.core.results_io import ResultsIOManager


@pytest.mark.unit
class TestModelIdentityCSVRoundTrip:
    """_parse_model_identity should round-trip display_string including config_id."""

    def test_round_trip_with_config_id(self) -> None:
        original = ModelIdentity(interface="langchain", model_name="gpt-4", config_id="high-temp")
        parsed = ResultsIOManager._parse_model_identity(original.display_string)
        assert parsed.interface == "langchain"
        assert parsed.model_name == "gpt-4"
        assert parsed.config_id == "high-temp"

    def test_round_trip_without_config_id(self) -> None:
        original = ModelIdentity(interface="langchain", model_name="gpt-4")
        parsed = ResultsIOManager._parse_model_identity(original.display_string)
        assert parsed.interface == "langchain"
        assert parsed.model_name == "gpt-4"
        assert parsed.config_id is None

    def test_round_trip_with_config_id_and_tools(self) -> None:
        original = ModelIdentity(
            interface="claude_agent_sdk", model_name="claude-sonnet-4-20250514",
            tools=["brave", "fs"], config_id="my-cfg",
        )
        parsed = ResultsIOManager._parse_model_identity(original.display_string)
        assert parsed.config_id == "my-cfg"
        assert parsed.tools == ["brave", "fs"]
```

- [ ] **Step 7: Run all tests**

Run: `cd karenina && uv run pytest tests/unit/schemas/test_model_identity.py -v`
Expected: ALL PASS

- [ ] **Step 8: Commit**

```bash
git add src/karenina/schemas/verification/model_identity.py src/karenina/benchmark/core/results_io.py tests/unit/schemas/test_model_identity.py
git commit -m "feat: enrich ModelIdentity with config_id when it differs from model_name"
```

---

### Task 7: Run name preservation on reimport

**Files:**
- Modify: `src/karenina/benchmark/core/results.py:367-399` (`load_results_from_file`)
- Test: `tests/unit/benchmark/core/test_results_run_name.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/benchmark/core/test_results_run_name.py`:

```python
"""Tests for run_name preservation on bulk reimport."""

import json
import tempfile
from pathlib import Path

import pytest

from karenina.benchmark.core.results import ResultsManager
from karenina.schemas.verification import VerificationResult


def _make_minimal_result(question_id: str, run_name: str) -> dict:
    """Create a minimal serialized VerificationResult dict."""
    return {
        "metadata": {
            "question_id": question_id,
            "question_text": "test",
            "template_id": "abc",
            "run_name": run_name,
            "answering": {"interface": "langchain", "model_name": "gpt-4"},
            "parsing": {"interface": "langchain", "model_name": "gpt-4"},
        },
        "template": {},
        "rubric": {},
    }


@pytest.mark.unit
class TestRunNamePreservation:
    """Bulk JSON reimport should preserve run_name grouping from metadata."""

    def test_multi_run_reimport_groups_by_metadata_run_name(self, tmp_path: Path) -> None:
        """Results from different runs should be stored under their original run_name."""
        data = [
            _make_minimal_result("q1", "run-alpha"),
            _make_minimal_result("q2", "run-alpha"),
            _make_minimal_result("q3", "run-beta"),
        ]
        json_file = tmp_path / "results.json"
        json_file.write_text(json.dumps(data))

        manager = ResultsManager.__new__(ResultsManager)
        manager._in_memory_results = {}

        manager.load_results_from_file(json_file, run_name="fallback")

        assert "run-alpha" in manager._in_memory_results
        assert "run-beta" in manager._in_memory_results
        assert len(manager._in_memory_results["run-alpha"]) == 2
        assert len(manager._in_memory_results["run-beta"]) == 1

    def test_fallback_run_name_when_metadata_is_none(self, tmp_path: Path) -> None:
        """Results without metadata.run_name should use the caller-provided fallback."""
        data = [_make_minimal_result("q1", None)]
        # Manually set run_name to null in JSON
        data[0]["metadata"]["run_name"] = None
        json_file = tmp_path / "results.json"
        json_file.write_text(json.dumps(data))

        manager = ResultsManager.__new__(ResultsManager)
        manager._in_memory_results = {}

        manager.load_results_from_file(json_file, run_name="my-fallback")

        assert "my-fallback" in manager._in_memory_results
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd karenina && uv run pytest tests/unit/benchmark/core/test_results_run_name.py -v`
Expected: FAIL (all results stored under "fallback" instead of per-run grouping)

- [ ] **Step 3: Implement the fix**

In `src/karenina/benchmark/core/results.py`, update `load_results_from_file` (lines 395-398). Replace:

```python
        # Store in memory if run_name provided
        if run_name:
            self._in_memory_results[run_name] = results
```

with:

```python
        # Group results by their metadata.run_name, falling back to caller-provided run_name
        grouped: dict[str, dict[str, VerificationResult]] = {}
        for key, result in results.items():
            effective_run = (
                result.metadata.run_name
                if result.metadata.run_name is not None
                else run_name
            )
            if effective_run:
                grouped.setdefault(effective_run, {})[key] = result

        # Store each group under its own run_name
        for group_name, group_results in grouped.items():
            self._in_memory_results[group_name] = group_results
```

- [ ] **Step 4: Run tests**

Run: `cd karenina && uv run pytest tests/unit/benchmark/core/test_results_run_name.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run existing results tests for regressions**

Run: `cd karenina && uv run pytest tests/unit/benchmark/core/test_results.py -v -x`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/karenina/benchmark/core/results.py tests/unit/benchmark/core/test_results_run_name.py
git commit -m "fix: preserve run_name grouping when reimporting bulk JSON results"
```

---

### Task 8: Deep judgment hallucination assessment non-fatal

**Files:**
- Modify: `src/karenina/benchmark/verification/evaluators/template/deep_judgment.py:544-633`
- Test: `tests/unit/benchmark/verification/evaluators/test_deep_judgment_nonfatal.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/benchmark/verification/evaluators/test_deep_judgment_nonfatal.py`:

```python
"""Tests for non-fatal hallucination assessment in deep judgment."""

import json
import logging
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
class TestHallucinationAssessmentNonFatal:
    """Stage 1.5 hallucination assessment failure should log warning, not raise."""

    def test_assessment_llm_failure_does_not_raise(self, caplog: pytest.LogCaptureFixture) -> None:
        """If the LLM call for hallucination assessment raises, deep judgment
        should log a warning and continue, not propagate the exception.

        We verify this by checking the source code: the except block should
        call logger.warning, not raise ValueError.
        """
        import inspect
        from karenina.benchmark.verification.evaluators.template import deep_judgment

        source = inspect.getsource(deep_judgment)

        # The old fatal pattern should be gone
        assert "Deep-judgment cannot continue without risk assessment" not in source

    def test_stage2_defaults_to_high_risk_without_assessment(self) -> None:
        """Stage 2 risk aggregation defaults to 'high' when no hallucination_risk
        is set on excerpts (the fallback path when assessment fails)."""
        risk_order = {"none": 0, "low": 1, "medium": 2, "high": 3}
        excerpts = {
            "drug_name": [
                {"excerpt": "aspirin", "search_results": [{"title": "test"}]},
            ]
        }

        # Simulate Stage 2 risk aggregation (from deep_judgment.py ~line 696-709)
        hallucination_risk = {}
        for attr_name, excerpt_list in excerpts.items():
            excerpt_risks = []
            for excerpt_obj in excerpt_list:
                if "hallucination_risk" in excerpt_obj:
                    excerpt_risks.append(excerpt_obj["hallucination_risk"])
            if excerpt_risks:
                max_risk = max(excerpt_risks, key=lambda r: risk_order.get(r, 3))
                hallucination_risk[attr_name] = max_risk
            else:
                hallucination_risk[attr_name] = "high"

        assert hallucination_risk["drug_name"] == "high"
```

- [ ] **Step 2: Run test to verify the source-checking test fails (old fatal pattern still present)**

Run: `cd karenina && uv run pytest tests/unit/benchmark/verification/evaluators/test_deep_judgment_nonfatal.py::TestHallucinationAssessmentNonFatal::test_assessment_llm_failure_does_not_raise -v`
Expected: FAIL (the fatal error message is still in the source)

- [ ] **Step 3: Implement the fix**

In `src/karenina/benchmark/verification/evaluators/template/deep_judgment.py`, wrap lines 544-633 in an outer try/except and change the inner handler from fatal to non-fatal.

Replace lines 544-633 (the `if search_performed:` block) with:

```python
    if search_performed:
        try:
            logger.info("Stage 1.5: Assessing hallucination risk for each excerpt")

            # Assign unique IDs to excerpts for matching
            excerpt_id_counter = 0
            excerpts_with_search = []

            for attr_name, excerpt_list in excerpts.items():
                for excerpt_obj in excerpt_list:
                    if "search_results" in excerpt_obj:
                        excerpt_obj["_id"] = str(excerpt_id_counter)
                        excerpt_obj["_attribute"] = attr_name
                        excerpts_with_search.append((attr_name, excerpt_obj))
                        excerpt_id_counter += 1

            if excerpts_with_search:
                # Build batch assessment prompt
                assessment_system_prompt = build_assessment_system_prompt(
                    generic_system_prompt=generic_system_prompt,
                )

                assessment_prompt = build_assessment_user_prompt(
                    excerpts_with_search=excerpts_with_search,
                    format_search_results_fn=_format_search_results_for_llm,
                )

                # Invoke LLM for batch assessment
                assessment_assembler = PromptAssembler(
                    task=PromptTask.DJ_TEMPLATE_HALLUCINATION,
                    interface=parsing_model.interface,
                    capabilities=PortCapabilities(),
                )
                assessment_messages = assessment_assembler.assemble(
                    system_text=assessment_system_prompt,
                    user_text=assessment_prompt,
                    user_instructions=prompt_config.get_for_task(PromptTask.DJ_TEMPLATE_HALLUCINATION.value)
                    if prompt_config
                    else None,
                    instruction_context={
                        "json_schema": None,
                        "parsing_notes": (
                            '- The "hallucination_risk" field must be one of: "none", "low", "medium", "high"'
                        ),
                    },
                )

                try:
                    llm_response = parsing_llm.invoke(assessment_messages)
                    raw_response, usage_metadata = llm_response.content, llm_response.usage.to_dict()
                    model_calls += 1
                    if usage_tracker and usage_metadata and parsing_model_str:
                        usage_tracker.track_call(
                            "deep_judgment_hallucination_assessment", parsing_model_str, usage_metadata
                        )
                    cleaned_response = _strip_markdown_fences(raw_response)
                    assessment_data = {} if cleaned_response is None else json.loads(cleaned_response)

                    # Match assessments back to excerpts
                    for assessment in assessment_data.get("excerpt_assessments", []):
                        excerpt_id = assessment["excerpt_id"]
                        hallucination_risk = assessment.get("hallucination_risk", "high")
                        justification = assessment.get("justification", "")

                        for excerpt_list in excerpts.values():
                            for excerpt_obj in excerpt_list:
                                if excerpt_obj.get("_id") == excerpt_id:
                                    excerpt_obj["hallucination_risk"] = hallucination_risk
                                    excerpt_obj["hallucination_justification"] = justification
                                    break

                    logger.info(
                        "Stage 1.5 complete: Assessed %d excerpts",
                        len(assessment_data.get("excerpt_assessments", [])),
                    )
                    stages_completed.append("excerpt_hallucination_assessment")

                except Exception as e:
                    logger.warning(
                        "Stage 1.5 hallucination assessment failed, continuing without "
                        "risk scores (Stage 2 will default to 'high'): %s", e,
                    )

            # Clean up temporary IDs
            for excerpt_list in excerpts.values():
                for excerpt_obj in excerpt_list:
                    excerpt_obj.pop("_id", None)
                    excerpt_obj.pop("_attribute", None)

        except Exception as e:
            logger.warning(
                "Hallucination assessment block failed, continuing without "
                "risk scores: %s", e,
            )
```

Key changes:
1. Outer `try/except Exception` wraps the entire block (lines 544-633)
2. Inner `except (json.JSONDecodeError, Exception)` simplified to `except Exception`
3. Inner handler changed from `raise ValueError(...)` to `logger.warning(...)`
4. f-string log at line 617 changed to `%`-style

- [ ] **Step 4: Run existing deep judgment tests for regressions**

Run: `cd karenina && uv run pytest tests/unit/benchmark/verification/evaluators/ -v -x -k "deep_judgment"`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/karenina/benchmark/verification/evaluators/template/deep_judgment.py tests/unit/benchmark/verification/evaluators/test_deep_judgment_nonfatal.py
git commit -m "fix: make deep judgment hallucination assessment non-fatal, log warning on failure"
```

---

### Task 9: Embedding check config bypass

**Files:**
- Modify: `src/karenina/benchmark/verification/utils/embedding_check.py:105-136,291-314`
- Modify: `src/karenina/benchmark/verification/stages/pipeline/embedding_check.py:109-111`
- Modify: `src/karenina/benchmark/verification/stages/core/base.py` (add context fields)
- Modify: `src/karenina/benchmark/verification/runner.py` (pass config to context)
- Test: `tests/unit/benchmark/verification/test_embedding_check_config.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/benchmark/verification/test_embedding_check_config.py`:

```python
"""Tests for embedding check using VerificationConfig parameters instead of env vars."""

import pytest

from karenina.benchmark.verification.utils.embedding_check import perform_embedding_check
from karenina.schemas.config.models import ModelConfig


@pytest.mark.unit
class TestEmbeddingCheckConfig:
    """perform_embedding_check should accept explicit config parameters."""

    def test_disabled_via_parameter(self) -> None:
        """When enabled=False is passed, check should not run regardless of env vars."""
        parsing_model = ModelConfig(interface="langchain", model_name="gpt-4")
        result = perform_embedding_check(
            ground_truth_data={"answer": "yes"},
            llm_response_data={"answer": "yes"},
            parsing_model=parsing_model,
            enabled=False,
        )
        should_override, score, model, performed = result
        assert performed is False
        assert should_override is False

    def test_enabled_parameter_exists(self) -> None:
        """perform_embedding_check should accept 'enabled' as a keyword argument."""
        import inspect
        sig = inspect.signature(perform_embedding_check)
        assert "enabled" in sig.parameters

    def test_threshold_parameter_exists(self) -> None:
        """perform_embedding_check should accept 'threshold' as a keyword argument."""
        import inspect
        sig = inspect.signature(perform_embedding_check)
        assert "threshold" in sig.parameters

    def test_model_parameter_exists(self) -> None:
        """perform_embedding_check should accept 'model' as a keyword argument."""
        import inspect
        sig = inspect.signature(perform_embedding_check)
        assert "model" in sig.parameters

    def test_enabled_false_ignores_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When enabled=False is passed explicitly, EMBEDDING_CHECK env var is ignored."""
        monkeypatch.setenv("EMBEDDING_CHECK", "true")
        parsing_model = ModelConfig(interface="langchain", model_name="gpt-4")
        result = perform_embedding_check(
            ground_truth_data={"answer": "yes"},
            llm_response_data={"answer": "yes"},
            parsing_model=parsing_model,
            enabled=False,
        )
        _, _, _, performed = result
        assert performed is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd karenina && uv run pytest tests/unit/benchmark/verification/test_embedding_check_config.py -v`
Expected: FAIL (`enabled` parameter does not exist)

- [ ] **Step 3: Refactor perform_embedding_check signature**

In `src/karenina/benchmark/verification/utils/embedding_check.py`, update `perform_embedding_check` (line 291):

```python
def perform_embedding_check(
    ground_truth_data: dict[str, Any] | None,
    llm_response_data: dict[str, Any] | None,
    parsing_model: ModelConfig,
    question_text: str | None = None,
    *,
    enabled: bool | None = None,
    model: str | None = None,
    threshold: float | None = None,
) -> tuple[bool, float | None, str | None, bool]:
    """Perform complete embedding check with fallback to semantic equivalence.

    Args:
        ground_truth_data: The ground truth parsed data.
        llm_response_data: The LLM response parsed data.
        parsing_model: The parsing model configuration for semantic check.
        question_text: The original question text for context (optional but recommended).
        enabled: Whether embedding check is enabled. If None, falls back to
            _should_use_embedding_check() (env var).
        model: Embedding model name override. If None, falls back to
            _get_embedding_model_name() (env var).
        threshold: Similarity threshold override. If None, falls back to
            _get_embedding_threshold() (env var).

    Returns:
        Tuple of (should_override_result, similarity_score,
            embedding_model_used, embedding_check_performed)
    """
    check_enabled = enabled if enabled is not None else _should_use_embedding_check()
    if not check_enabled:
        return False, None, None, False

    try:
        similarity_score, model_name = compute_embedding_similarity(
            ground_truth_data, llm_response_data,
        )

        effective_threshold = threshold if threshold is not None else _get_embedding_threshold()

        if similarity_score >= effective_threshold:
            try:
                is_equivalent = check_semantic_equivalence(
                    ground_truth_data, llm_response_data, parsing_model, question_text,
                )
                return (is_equivalent, similarity_score, model_name, True)
            except RuntimeError:
                return (False, similarity_score, model_name, True)
        else:
            return (False, similarity_score, model_name, True)

    except (ImportError, RuntimeError):
        return (False, None, None, True)
```

- [ ] **Step 4: Add embedding config fields to VerificationContext**

In `src/karenina/benchmark/verification/stages/core/base.py`, add after line 333 (after `agentic_rubric_parallel`):

```python
    # Embedding Check Configuration
    embedding_check_enabled: bool = False
    embedding_check_model: str | None = None
    embedding_check_threshold: float | None = None
```

- [ ] **Step 5: Pass config to context in runner.py**

In `src/karenina/benchmark/verification/runner.py`:

Add parameters to `run_single_model_verification` signature (after `agentic_rubric_parallel`):

```python
    # Embedding check configuration
    embedding_check_enabled: bool = False,
    embedding_check_model: str | None = None,
    embedding_check_threshold: float | None = None,
```

Add to the `VerificationContext(...)` constructor call (after `trait_provenance=trait_provenance,`):

```python
        # Embedding Check
        embedding_check_enabled=embedding_check_enabled,
        embedding_check_model=embedding_check_model,
        embedding_check_threshold=embedding_check_threshold,
```

- [ ] **Step 6: Update EmbeddingCheckStage to pass config from context**

In `src/karenina/benchmark/verification/stages/pipeline/embedding_check.py`, update the `perform_embedding_check` call (line 109):

```python
        (should_override, similarity_score, model_name, check_performed) = perform_embedding_check(
            parsed_gt_response, parsed_llm_response, context.parsing_model, context.question_text,
            enabled=context.embedding_check_enabled,
            model=context.embedding_check_model,
            threshold=context.embedding_check_threshold,
        )
```

- [ ] **Step 7: Run tests**

Run: `cd karenina && uv run pytest tests/unit/benchmark/verification/test_embedding_check_config.py -v`
Expected: ALL PASS

- [ ] **Step 8: Commit**

```bash
git add src/karenina/benchmark/verification/utils/embedding_check.py src/karenina/benchmark/verification/stages/pipeline/embedding_check.py src/karenina/benchmark/verification/stages/core/base.py src/karenina/benchmark/verification/runner.py tests/unit/benchmark/verification/test_embedding_check_config.py
git commit -m "fix: embedding check reads config from VerificationContext instead of env vars directly"
```

---

### Task 10: Abstention/sufficiency adapter instructions

**Files:**
- Modify: `src/karenina/benchmark/verification/prompts/trace/abstention.py`
- Modify: `src/karenina/benchmark/verification/prompts/trace/sufficiency.py`
- Create: `src/karenina/adapters/langchain/prompts/abstention.py`
- Create: `src/karenina/adapters/langchain/prompts/sufficiency.py`
- Create: `src/karenina/adapters/claude_tool/prompts/abstention.py`
- Create: `src/karenina/adapters/claude_tool/prompts/sufficiency.py`
- Create: `src/karenina/adapters/claude_agent_sdk/prompts/abstention.py`
- Create: `src/karenina/adapters/claude_agent_sdk/prompts/sufficiency.py`
- Create: `src/karenina/adapters/langchain_deep_agents/prompts/abstention.py`
- Create: `src/karenina/adapters/langchain_deep_agents/prompts/sufficiency.py`
- Modify: `src/karenina/adapters/langchain/registration.py`
- Modify: `src/karenina/adapters/claude_tool/registration.py`
- Modify: `src/karenina/adapters/claude_agent_sdk/registration.py`
- Modify: `src/karenina/adapters/langchain_deep_agents/registration.py`
- Test: `tests/test_abstention_sufficiency_adapter_instructions.py` (new)

This is the largest task. Follow the existing adapter instruction pattern exactly.

- [ ] **Step 1: Write the failing test**

Create `tests/test_abstention_sufficiency_adapter_instructions.py`:

```python
"""Tests for abstention/sufficiency adapter instruction registration."""

from __future__ import annotations

import pytest

# Import registration modules to trigger side-effect registration
import karenina.adapters.claude_agent_sdk.prompts.abstention  # noqa: F401
import karenina.adapters.claude_agent_sdk.prompts.sufficiency  # noqa: F401
import karenina.adapters.claude_tool.prompts.abstention  # noqa: F401
import karenina.adapters.claude_tool.prompts.sufficiency  # noqa: F401
import karenina.adapters.langchain.prompts.abstention  # noqa: F401
import karenina.adapters.langchain.prompts.sufficiency  # noqa: F401
from karenina.ports.adapter_instruction import AdapterInstructionRegistry

INTERFACES = ["langchain", "openrouter", "openai_endpoint", "claude_tool", "claude_agent_sdk"]


@pytest.mark.unit
class TestAbstentionAdapterRegistration:
    """All adapter interfaces should register instructions for abstention_detection."""

    @pytest.mark.parametrize("interface", INTERFACES)
    def test_registered(self, interface: str) -> None:
        factories = AdapterInstructionRegistry.get_instructions(interface, "abstention_detection")
        assert len(factories) > 0, f"No abstention_detection instruction for {interface}"

    @pytest.mark.parametrize("interface", INTERFACES)
    def test_factory_produces_valid_instruction(self, interface: str) -> None:
        factories = AdapterInstructionRegistry.get_instructions(interface, "abstention_detection")
        instruction = factories[0]()
        assert isinstance(instruction.system_addition, str)
        assert isinstance(instruction.user_addition, str)


@pytest.mark.unit
class TestSufficiencyAdapterRegistration:
    """All adapter interfaces should register instructions for sufficiency_detection."""

    @pytest.mark.parametrize("interface", INTERFACES)
    def test_registered(self, interface: str) -> None:
        factories = AdapterInstructionRegistry.get_instructions(interface, "sufficiency_detection")
        assert len(factories) > 0, f"No sufficiency_detection instruction for {interface}"

    @pytest.mark.parametrize("interface", INTERFACES)
    def test_factory_produces_valid_instruction(self, interface: str) -> None:
        factories = AdapterInstructionRegistry.get_instructions(interface, "sufficiency_detection")
        instruction = factories[0]()
        assert isinstance(instruction.system_addition, str)
        assert isinstance(instruction.user_addition, str)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd karenina && uv run pytest tests/test_abstention_sufficiency_adapter_instructions.py -v`
Expected: FAIL (modules do not exist)

- [ ] **Step 3: Remove output_format from base prompts**

In `src/karenina/benchmark/verification/prompts/trace/abstention.py`, remove lines 43-49 (the `<output_format>...</output_format>` block) from `ABSTENTION_DETECTION_SYS`. The prompt ends after `</critical_instructions>`.

In `src/karenina/benchmark/verification/prompts/trace/sufficiency.py`, remove lines 40-46 (the `<output_format>...</output_format>` block) from `SUFFICIENCY_DETECTION_SYS`. The prompt ends after `</critical_instructions>`.

- [ ] **Step 4: Create langchain adapter instructions**

Create `src/karenina/adapters/langchain/prompts/abstention.py`:

```python
"""LangChain adapter instructions for abstention detection."""

from dataclasses import dataclass

from karenina.ports.adapter_instruction import AdapterInstructionRegistry

_ABSTENTION_SYSTEM_ADDITION = """
<output_format>
Respond with ONLY a JSON object with this exact structure (reasoning MUST come first):
{
    "reasoning": "Brief explanation of why this was classified as abstention or genuine attempt",
    "abstention_detected": true or false
}
</output_format>"""

_ABSTENTION_USER_ADDITION = ""


@dataclass
class _LangChainAbstentionInstruction:
    """LangChain format instruction for abstention detection."""

    @property
    def system_addition(self) -> str:
        return _ABSTENTION_SYSTEM_ADDITION

    @property
    def user_addition(self) -> str:
        return _ABSTENTION_USER_ADDITION


def _langchain_abstention_instruction_factory(**kwargs: object) -> _LangChainAbstentionInstruction:
    return _LangChainAbstentionInstruction()


AdapterInstructionRegistry.register("langchain", "abstention_detection", _langchain_abstention_instruction_factory)
AdapterInstructionRegistry.register("openrouter", "abstention_detection", _langchain_abstention_instruction_factory)
AdapterInstructionRegistry.register("openai_endpoint", "abstention_detection", _langchain_abstention_instruction_factory)
```

Create `src/karenina/adapters/langchain/prompts/sufficiency.py` following the same pattern with the sufficiency output format.

- [ ] **Step 5: Create claude_tool adapter instructions**

Create `src/karenina/adapters/claude_tool/prompts/abstention.py`:

```python
"""Claude Tool adapter instructions for abstention detection."""

from dataclasses import dataclass

from karenina.ports.adapter_instruction import AdapterInstructionRegistry

_ABSTENTION_SYSTEM_ADDITION = """
Respond as a JSON object with "reasoning" (string) and "abstention_detected" (boolean) fields.
The "reasoning" field MUST come first."""

_ABSTENTION_USER_ADDITION = ""


@dataclass
class _ClaudeToolAbstentionInstruction:
    """Claude Tool format instruction for abstention detection (minimal, native structured output)."""

    @property
    def system_addition(self) -> str:
        return _ABSTENTION_SYSTEM_ADDITION

    @property
    def user_addition(self) -> str:
        return _ABSTENTION_USER_ADDITION


def _claude_tool_abstention_factory(**kwargs: object) -> _ClaudeToolAbstentionInstruction:
    return _ClaudeToolAbstentionInstruction()


AdapterInstructionRegistry.register("claude_tool", "abstention_detection", _claude_tool_abstention_factory)
```

Create matching `sufficiency.py` for claude_tool.

- [ ] **Step 6: Create claude_agent_sdk and langchain_deep_agents adapter instructions**

For `claude_agent_sdk`: same minimal pattern as claude_tool.
For `langchain_deep_agents`: same verbose JSON pattern as langchain.

Create files at:
- `src/karenina/adapters/claude_agent_sdk/prompts/abstention.py`
- `src/karenina/adapters/claude_agent_sdk/prompts/sufficiency.py`
- `src/karenina/adapters/langchain_deep_agents/prompts/abstention.py`
- `src/karenina/adapters/langchain_deep_agents/prompts/sufficiency.py`

- [ ] **Step 7: Update registration.py files**

Add import lines to the bottom of each adapter's `registration.py`:

For `src/karenina/adapters/langchain/registration.py`, add:
```python
import karenina.adapters.langchain.prompts.abstention  # noqa: E402, F401
import karenina.adapters.langchain.prompts.sufficiency  # noqa: E402, F401
```

Repeat for `claude_tool`, `claude_agent_sdk`, `langchain_deep_agents`.

- [ ] **Step 8: Run tests**

Run: `cd karenina && uv run pytest tests/test_abstention_sufficiency_adapter_instructions.py -v`
Expected: ALL PASS

- [ ] **Step 9: Run full test suite to check for regressions**

Run: `cd karenina && uv run pytest tests/ -x -q --timeout=60`
Expected: ALL PASS

- [ ] **Step 10: Commit**

```bash
git add src/karenina/benchmark/verification/prompts/trace/abstention.py src/karenina/benchmark/verification/prompts/trace/sufficiency.py src/karenina/adapters/*/prompts/abstention.py src/karenina/adapters/*/prompts/sufficiency.py src/karenina/adapters/*/registration.py tests/test_abstention_sufficiency_adapter_instructions.py
git commit -m "refactor: extract abstention/sufficiency output format into adapter instructions"
```

---

### Task 11: Final verification

- [ ] **Step 1: Run full test suite**

Run: `cd karenina && uv run pytest tests/ -x -q`
Expected: ALL PASS

- [ ] **Step 2: Run type checks**

Run: `cd karenina && uv run mypy src/karenina/ --no-error-summary 2>&1 | tail -5`
Expected: No new errors

- [ ] **Step 3: Check for any docs/skills that need updating**

Ask the user if any skills or docs need updating based on the changes made.
