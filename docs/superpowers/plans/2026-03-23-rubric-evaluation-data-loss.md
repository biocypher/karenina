# Rubric Evaluation Data Loss: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three categories of data loss in rubric/template evaluation: missing DataFrame metadata columns (Issue 156), missing trait provenance after rubric merge (Issue 020), and missing reasoning-only mode for template deep judgment (Issue 154, reframed).

**Architecture:** Three independent workstreams. Issue 156 adds two passthrough columns to the DataFrame builder. Issue 020 threads provenance metadata from `merge_rubrics()` through the pipeline to the DataFrame. Issue 154 adds a reasoning-only path to template deep judgment (mirroring the existing rubric deep judgment without-excerpts flow), requiring a new config field, new prompt, modified `deep_judgment_parse()`, and updated auto-fail stage.

**Tech Stack:** Python, Pydantic v2, pytest, karenina verification pipeline

**Source spec:** `issues_plan/issue_chunks/22-rubric-evaluation-data-loss/TASK.md` (Issues 020, 156 as-written; Issue 154 reframed per design discussion)

---

## Stream A: Issue 156 (DataFrame Metadata Columns)

### Task 1: Add `rubric_evaluation_performed` and `rubric_evaluation_strategy` columns to RubricDataFrameBuilder

**Files:**
- Modify: `karenina/src/karenina/schemas/dataframes/rubric.py` (row builder methods + column ordering)
- Test: `karenina/tests/unit/schemas/dataframes/test_rubric_dataframe_metadata_columns.py`

- [ ] **Step 1: Write failing tests**

Create `karenina/tests/unit/schemas/dataframes/test_rubric_dataframe_metadata_columns.py`:

```python
"""Tests for rubric evaluation metadata columns in RubricDataFrameBuilder."""

import pytest

from karenina.schemas.dataframes.rubric import RubricDataFrameBuilder
from karenina.schemas.verification.result import VerificationResult
from karenina.schemas.verification.result_components import (
    VerificationResultMetadata,
    VerificationResultRubric,
)


def _make_result(
    rubric_evaluation_performed: bool = True,
    rubric_evaluation_strategy: str | None = "batch",
) -> VerificationResult:
    """Create a minimal VerificationResult with rubric data for testing."""
    metadata = VerificationResultMetadata(
        question_id="q1",
        template_id="t1",
        completed_without_errors=True,
    )
    rubric = VerificationResultRubric(
        rubric_evaluation_performed=rubric_evaluation_performed,
        rubric_evaluation_strategy=rubric_evaluation_strategy,
        llm_trait_scores={"safety": True},
    )
    return VerificationResult(metadata=metadata, rubric=rubric)


@pytest.mark.unit
class TestRubricDataFrameMetadataColumns:
    """Verify rubric_evaluation_performed and rubric_evaluation_strategy appear in DataFrame."""

    def test_rubric_evaluation_performed_column_present(self):
        result = _make_result(rubric_evaluation_performed=True)
        builder = RubricDataFrameBuilder(results=[result])
        df = builder.build()
        assert "rubric_evaluation_performed" in df.columns
        assert df["rubric_evaluation_performed"].iloc[0] is True

    def test_rubric_evaluation_strategy_column_present(self):
        result = _make_result(rubric_evaluation_strategy="sequential")
        builder = RubricDataFrameBuilder(results=[result])
        df = builder.build()
        assert "rubric_evaluation_strategy" in df.columns
        assert df["rubric_evaluation_strategy"].iloc[0] == "sequential"

    def test_rubric_evaluation_strategy_none_when_not_performed(self):
        result = _make_result(
            rubric_evaluation_performed=False,
            rubric_evaluation_strategy=None,
        )
        builder = RubricDataFrameBuilder(results=[result])
        df = builder.build()
        assert df["rubric_evaluation_strategy"].iloc[0] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd karenina && uv run pytest tests/unit/schemas/dataframes/test_rubric_dataframe_metadata_columns.py -v`
Expected: FAIL (columns not present in DataFrame)

- [ ] **Step 3: Add columns to each trait row builder method**

In `karenina/src/karenina/schemas/dataframes/rubric.py`, add two keys to every `_create_*_trait_row()` method (there are five: `_create_llm_trait_row`, `_create_regex_trait_row`, `_create_callable_trait_row`, `_create_metric_trait_row`, `_create_agentic_trait_row`). In each method, after the `"run_name"` entry, add:

```python
            # Rubric evaluation metadata
            "rubric_evaluation_performed": result.rubric.rubric_evaluation_performed if result.rubric else None,
            "rubric_evaluation_strategy": result.rubric.rubric_evaluation_strategy if result.rubric else None,
```

Also add both column names to the `desired_order` list (after `"run_name"` and before the deep judgment columns):

```python
    "rubric_evaluation_performed", "rubric_evaluation_strategy",
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd karenina && uv run pytest tests/unit/schemas/dataframes/test_rubric_dataframe_metadata_columns.py -v`
Expected: PASS

- [ ] **Step 5: Run existing rubric dataframe tests to check for regressions**

Run: `cd karenina && uv run pytest tests/ -x -q -k "dataframe" --timeout=60`
Expected: All existing tests pass

- [ ] **Step 6: Commit**

```bash
git add karenina/src/karenina/schemas/dataframes/rubric.py karenina/tests/unit/schemas/dataframes/test_rubric_dataframe_metadata_columns.py
git commit -m "feat(dataframe): add rubric_evaluation_performed and rubric_evaluation_strategy columns

Closes Issue 156: these VerificationResultRubric fields were not exported
to the rubric DataFrame, preventing users from filtering or comparing
evaluation strategies."
```

---

## Stream B: Issue 020 (Trait Provenance Tracking)

### Task 2: Make `merge_rubrics()` return provenance metadata

**Files:**
- Modify: `karenina/src/karenina/schemas/entities/rubric.py` (`merge_rubrics` function)
- Test: `karenina/tests/unit/schemas/entities/test_rubric_provenance.py`

- [ ] **Step 1: Write failing tests**

Create `karenina/tests/unit/schemas/entities/test_rubric_provenance.py`:

```python
"""Tests for trait provenance tracking in merge_rubrics()."""

import pytest

from karenina.schemas.entities.rubric import (
    LLMRubricTrait,
    RegexRubricTrait,
    Rubric,
    merge_rubrics,
)


def _llm_trait(name: str) -> LLMRubricTrait:
    return LLMRubricTrait(name=name, description=f"Test {name}", kind="boolean")


def _regex_trait(name: str) -> RegexRubricTrait:
    return RegexRubricTrait(name=name, pattern=r"\d+")


@pytest.mark.unit
class TestMergeRubricsProvenance:
    """Verify merge_rubrics returns provenance mapping for merged traits."""

    def test_global_only_provenance(self):
        global_rubric = Rubric(llm_traits=[_llm_trait("safety")])
        merged, provenance = merge_rubrics(global_rubric, None)
        assert provenance == {"safety": "global"}

    def test_question_only_provenance(self):
        question_rubric = Rubric(regex_traits=[_regex_trait("has_numbers")])
        merged, provenance = merge_rubrics(None, question_rubric)
        assert provenance == {"has_numbers": "question_specific"}

    def test_both_rubrics_provenance(self):
        global_rubric = Rubric(llm_traits=[_llm_trait("safety")])
        question_rubric = Rubric(
            llm_traits=[_llm_trait("relevance")],
            regex_traits=[_regex_trait("has_numbers")],
        )
        merged, provenance = merge_rubrics(global_rubric, question_rubric)
        assert provenance == {
            "safety": "global",
            "relevance": "question_specific",
            "has_numbers": "question_specific",
        }

    def test_none_none_returns_none_provenance(self):
        result = merge_rubrics(None, None)
        assert result == (None, None)

    def test_provenance_backwards_compat_rubric_unchanged(self):
        """Merged Rubric object itself is unchanged by provenance tracking."""
        global_rubric = Rubric(llm_traits=[_llm_trait("a")])
        question_rubric = Rubric(llm_traits=[_llm_trait("b")])
        merged, provenance = merge_rubrics(global_rubric, question_rubric)
        assert len(merged.llm_traits) == 2
        assert {t.name for t in merged.llm_traits} == {"a", "b"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd karenina && uv run pytest tests/unit/schemas/entities/test_rubric_provenance.py -v`
Expected: FAIL (merge_rubrics returns Rubric|None, not tuple)

- [ ] **Step 3: Modify `merge_rubrics()` to return `tuple[Rubric | None, dict[str, str] | None]`**

In `karenina/src/karenina/schemas/entities/rubric.py`, change `merge_rubrics` signature and implementation:

```python
def merge_rubrics(
    global_rubric: "Rubric | None",
    question_rubric: "Rubric | None",
) -> tuple["Rubric | None", dict[str, str] | None]:
    """Merge global and question-specific rubrics with provenance tracking.

    Returns:
        Tuple of (merged_rubric, provenance_map). provenance_map maps
        each trait name to its source: "global" or "question_specific".
    """
```

Build the provenance dict by iterating over each rubric's traits before merging:

```python
    if not global_rubric and not question_rubric:
        return None, None

    if not global_rubric:
        provenance = {}
        for traits in [question_rubric.llm_traits, question_rubric.regex_traits,
                       question_rubric.callable_traits, question_rubric.metric_traits,
                       question_rubric.agentic_traits]:
            for t in traits:
                provenance[t.name] = "question_specific"
        return question_rubric, provenance

    if not question_rubric:
        provenance = {}
        for traits in [global_rubric.llm_traits, global_rubric.regex_traits,
                       global_rubric.callable_traits, global_rubric.metric_traits,
                       global_rubric.agentic_traits]:
            for t in traits:
                provenance[t.name] = "global"
        return global_rubric, provenance
```

For the full merge case, build provenance from both before the collision check, then proceed as before. Return `(merged_rubric, provenance)`.

- [ ] **Step 4: Fix all callers of `merge_rubrics()` to unpack the tuple**

Search for all call sites:
```bash
cd karenina && grep -rn "merge_rubrics(" src/ --include="*.py"
```

Each call like `rubric = merge_rubrics(global_r, question_r)` becomes `rubric, provenance = merge_rubrics(global_r, question_r)`. Store or discard `provenance` depending on the call site. The key call site is in the rubric evaluation stage or wherever the merged rubric is passed into the pipeline context.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd karenina && uv run pytest tests/unit/schemas/entities/test_rubric_provenance.py -v`
Expected: PASS

- [ ] **Step 6: Run full rubric test suite to check for regressions**

Run: `cd karenina && uv run pytest tests/ -x -q -k "rubric" --timeout=120`
Expected: All existing tests pass (callers updated)

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(rubric): add provenance tracking to merge_rubrics()

merge_rubrics() now returns a tuple of (merged_rubric, provenance_map)
where provenance_map tracks which traits came from global vs
question-specific rubrics. Part of Issue 020."
```

### Task 3: Thread provenance through pipeline to VerificationResultRubric and DataFrame

**Important:** `merge_rubrics()` is NOT called inside any pipeline stage. It is called upstream in `benchmark/verification/utils/task_helpers.py` (via `merge_rubrics_for_task()`) and the already-merged rubric is passed into `VerificationContext` at construction time. The provenance threading path is:

1. `merge_rubrics()` returns `(rubric, provenance)` (done in Task 2)
2. `merge_rubrics_for_task()` unpacks and returns provenance
3. `batch_runner.py` / `run_single_verification()` passes provenance into `VerificationContext`
4. `VerificationContext` gets a `trait_provenance` field
5. Dynamic rubric resolution in `RubricEvaluationStage` annotates promoted traits with `"dynamic"`
6. `FinalizeResultStage` reads provenance from context and stores in `VerificationResultRubric`
7. `RubricDataFrameBuilder` adds a `trait_provenance` column

**Files:**
- Modify: `karenina/src/karenina/schemas/verification/result_components.py` (add `trait_provenance` field)
- Modify: `karenina/src/karenina/benchmark/verification/utils/task_helpers.py` (unpack provenance from `merge_rubrics`)
- Modify: `karenina/src/karenina/benchmark/verification/runner.py` or `batch_runner.py` (pass provenance to context)
- Modify: `karenina/src/karenina/benchmark/verification/stages/core/base.py` (add `trait_provenance` to VerificationContext)
- Modify: `karenina/src/karenina/benchmark/verification/stages/pipeline/rubric_evaluation.py` (annotate dynamic traits)
- Modify: `karenina/src/karenina/benchmark/verification/stages/pipeline/finalize_result.py` (map provenance to result)
- Modify: `karenina/src/karenina/schemas/dataframes/rubric.py` (add `trait_provenance` column)
- Test: `karenina/tests/unit/schemas/dataframes/test_rubric_provenance_column.py`

- [ ] **Step 1: Write failing test for provenance in DataFrame**

Create `karenina/tests/unit/schemas/dataframes/test_rubric_provenance_column.py`:

```python
"""Tests for trait_provenance column in RubricDataFrameBuilder."""

import pytest

from karenina.schemas.dataframes.rubric import RubricDataFrameBuilder
from karenina.schemas.verification.result import VerificationResult
from karenina.schemas.verification.result_components import (
    VerificationResultMetadata,
    VerificationResultRubric,
)


def _make_result_with_provenance() -> VerificationResult:
    metadata = VerificationResultMetadata(
        question_id="q1",
        template_id="t1",
        completed_without_errors=True,
    )
    rubric = VerificationResultRubric(
        rubric_evaluation_performed=True,
        llm_trait_scores={"safety": True, "relevance": 4},
        trait_provenance={"safety": "global", "relevance": "question_specific"},
    )
    return VerificationResult(metadata=metadata, rubric=rubric)


@pytest.mark.unit
class TestRubricProvenanceColumn:
    """Verify trait_provenance appears in DataFrame rows."""

    def test_provenance_column_present(self):
        result = _make_result_with_provenance()
        builder = RubricDataFrameBuilder(results=[result])
        df = builder.build()
        assert "trait_provenance" in df.columns

    def test_provenance_value_matches_trait(self):
        result = _make_result_with_provenance()
        builder = RubricDataFrameBuilder(results=[result])
        df = builder.build()
        safety_row = df[df["trait_name"] == "safety"].iloc[0]
        relevance_row = df[df["trait_name"] == "relevance"].iloc[0]
        assert safety_row["trait_provenance"] == "global"
        assert relevance_row["trait_provenance"] == "question_specific"

    def test_provenance_none_when_not_set(self):
        metadata = VerificationResultMetadata(
            question_id="q1", template_id="t1", completed_without_errors=True,
        )
        rubric = VerificationResultRubric(
            rubric_evaluation_performed=True,
            llm_trait_scores={"safety": True},
            trait_provenance=None,
        )
        result = VerificationResult(metadata=metadata, rubric=rubric)
        builder = RubricDataFrameBuilder(results=[result])
        df = builder.build()
        assert df["trait_provenance"].iloc[0] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd karenina && uv run pytest tests/unit/schemas/dataframes/test_rubric_provenance_column.py -v`
Expected: FAIL (field and column don't exist)

- [ ] **Step 3: Add `trait_provenance` field to `VerificationResultRubric`**

In `karenina/src/karenina/schemas/verification/result_components.py`, add to `VerificationResultRubric`:

```python
    # Provenance tracking: maps trait name to source ("global", "question_specific", "dynamic")
    trait_provenance: dict[str, str] | None = None
```

- [ ] **Step 4: Add `trait_provenance` field to `VerificationContext`**

In `karenina/src/karenina/benchmark/verification/stages/core/base.py`, add to `VerificationContext`:

```python
    trait_provenance: dict[str, str] | None = None
```

- [ ] **Step 5: Thread provenance from `merge_rubrics_for_task()` into `VerificationContext`**

First, find and update `merge_rubrics_for_task()` in `karenina/src/karenina/benchmark/verification/utils/task_helpers.py`:

```bash
cd karenina && grep -rn "merge_rubrics" src/ --include="*.py"
```

Update each caller of `merge_rubrics()` to unpack the tuple and pass provenance forward. The key path is:
1. `merge_rubrics_for_task()` calls `merge_rubrics()`, unpack provenance
2. The caller of `merge_rubrics_for_task()` (in runner/batch_runner) passes provenance to `VerificationContext` constructor

- [ ] **Step 5b: Annotate dynamic promoted traits in RubricEvaluationStage**

In `karenina/src/karenina/benchmark/verification/stages/pipeline/rubric_evaluation.py`, after dynamic rubric resolution (where promoted traits are added), update the provenance on the context:

```python
if promoted_trait_names and context.trait_provenance is not None:
    for name in promoted_trait_names:
        context.trait_provenance[name] = "dynamic"
```

- [ ] **Step 6: Map provenance in FinalizeResultStage**

In `karenina/src/karenina/benchmark/verification/stages/pipeline/finalize_result.py`, read provenance from the context field and pass to `VerificationResultRubric`:

```python
trait_provenance = context.trait_provenance

rubric_result = VerificationResultRubric(
    ...existing fields...,
    trait_provenance=trait_provenance,
)
```

- [ ] **Step 7: Add `trait_provenance` column to DataFrame builder**

In `karenina/src/karenina/schemas/dataframes/rubric.py`, in each of the five `_create_*_trait_row()` methods (llm, regex, callable, metric, agentic), add:

```python
            "trait_provenance": (
                result.rubric.trait_provenance.get(trait_name)
                if result.rubric and result.rubric.trait_provenance
                else None
            ),
```

Add `"trait_provenance"` to `desired_order` list (after `"rubric_evaluation_strategy"`).

- [ ] **Step 8: Run tests to verify they pass**

Run: `cd karenina && uv run pytest tests/unit/schemas/dataframes/test_rubric_provenance_column.py tests/unit/schemas/entities/test_rubric_provenance.py -v`
Expected: PASS

- [ ] **Step 9: Run full test suite to check for regressions**

Run: `cd karenina && uv run pytest tests/ -x -q -k "rubric or dataframe" --timeout=120`
Expected: All pass

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "feat(rubric): thread trait provenance from merge through pipeline to DataFrame

Adds trait_provenance field to VerificationResultRubric and a matching
column in RubricDataFrameBuilder. Provenance is populated during rubric
merge (global vs question_specific) and dynamic rubric resolution
(dynamic). Closes Issue 020."
```

---

## Stream C: Issue 154 (Template Deep Judgment Reasoning-Only Mode)

### Task 4: Add `deep_judgment_reasoning_only` config field

**Files:**
- Modify: `karenina/src/karenina/schemas/verification/config.py`
- Modify: `karenina/src/karenina/benchmark/verification/stages/core/base.py` (VerificationContext mirror field)
- Test: `karenina/tests/unit/schemas/verification/test_config_reasoning_only.py`

- [ ] **Step 1: Write failing test**

Create `karenina/tests/unit/schemas/verification/test_config_reasoning_only.py`:

```python
"""Tests for deep_judgment_reasoning_only config field."""

import pytest

from karenina.schemas.verification.config import VerificationConfig


@pytest.mark.unit
class TestDeepJudgmentReasoningOnlyConfig:
    """Verify reasoning_only config field behavior."""

    def test_default_is_false(self):
        config = VerificationConfig(answering_models=[], parsing_models=[])
        assert config.deep_judgment_reasoning_only is False

    def test_set_reasoning_only(self):
        config = VerificationConfig(
            answering_models=[],
            parsing_models=[],
            deep_judgment_enabled=True,
            deep_judgment_reasoning_only=True,
        )
        assert config.deep_judgment_reasoning_only is True
        assert config.deep_judgment_enabled is True

    def test_reasoning_only_without_deep_judgment_warns(self):
        """reasoning_only=True with deep_judgment_enabled=False is a no-op."""
        config = VerificationConfig(
            answering_models=[],
            parsing_models=[],
            deep_judgment_enabled=False,
            deep_judgment_reasoning_only=True,
        )
        # Should not raise; reasoning_only is ignored when DJ is disabled
        assert config.deep_judgment_reasoning_only is True
        assert config.deep_judgment_enabled is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd karenina && uv run pytest tests/unit/schemas/verification/test_config_reasoning_only.py -v`
Expected: FAIL (field doesn't exist)

- [ ] **Step 3: Add field to VerificationConfig**

In `karenina/src/karenina/schemas/verification/config.py`, add after `deep_judgment_enabled`:

```python
    deep_judgment_reasoning_only: bool = False  # Skip excerpt extraction, generate reasoning directly
```

Mirror in `VerificationContext` in `stages/core/base.py`:

```python
    deep_judgment_reasoning_only: bool = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd karenina && uv run pytest tests/unit/schemas/verification/test_config_reasoning_only.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(config): add deep_judgment_reasoning_only field

When deep_judgment_enabled=True and deep_judgment_reasoning_only=True,
template deep judgment skips excerpt extraction and generates per-attribute
reasoning directly from the response. Mirrors rubric DJ without-excerpts."
```

### Task 5: Build reasoning-only prompt

**Files:**
- Create: `karenina/src/karenina/benchmark/verification/prompts/deep_judgment/template/reasoning_only.py`
- Test: `karenina/tests/unit/prompts/deep_judgment/test_reasoning_only_prompt.py`

- [ ] **Step 1: Write failing test**

Create `karenina/tests/unit/prompts/deep_judgment/test_reasoning_only_prompt.py`:

```python
"""Tests for reasoning-only prompt builders."""

import pytest

from karenina.benchmark.verification.prompts.deep_judgment.template.reasoning_only import (
    build_reasoning_only_system_prompt,
    build_reasoning_only_user_prompt,
)


@pytest.mark.unit
class TestReasoningOnlyPrompt:
    """Verify reasoning-only prompts are well-formed."""

    def test_system_prompt_contains_attribute_guidance(self):
        prompt = build_reasoning_only_system_prompt(
            generic_system_prompt="You are a parser.",
            attr_guidance="- drug_target: The target protein\n- mechanism: How it works",
        )
        assert "drug_target" in prompt
        assert "mechanism" in prompt
        # Should NOT reference excerpts
        assert "excerpt" not in prompt.lower()

    def test_system_prompt_does_not_mention_excerpts(self):
        prompt = build_reasoning_only_system_prompt(
            generic_system_prompt="Base prompt.",
            attr_guidance="- x: description",
        )
        assert "excerpt" not in prompt.lower()
        assert "extracted_excerpts" not in prompt

    def test_user_prompt_contains_question_and_response(self):
        prompt = build_reasoning_only_user_prompt(
            question_text="What drug targets BCL-2?",
            raw_llm_response="Venetoclax targets BCL-2 for CLL treatment.",
        )
        assert "BCL-2" in prompt
        assert "Venetoclax" in prompt

    def test_user_prompt_uses_xml_tags(self):
        prompt = build_reasoning_only_user_prompt(
            question_text="Q",
            raw_llm_response="A",
        )
        assert "<original_question>" in prompt
        assert "<response>" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd karenina && uv run pytest tests/unit/prompts/deep_judgment/test_reasoning_only_prompt.py -v`
Expected: FAIL (module doesn't exist)

- [ ] **Step 3: Create reasoning-only prompt module**

Create `karenina/src/karenina/benchmark/verification/prompts/deep_judgment/template/reasoning_only.py`:

```python
"""Reasoning-only prompt builders for template deep judgment.

These builders produce system and user prompts for generating per-attribute
reasoning directly from the response text, without prior excerpt extraction.
This is the lightweight alternative to full deep judgment (no excerpts, no
fuzzy matching, no search validation).
"""


def build_reasoning_only_system_prompt(
    generic_system_prompt: str,
    attr_guidance: str,
) -> str:
    """Build system prompt for reasoning-only deep judgment.

    Args:
        generic_system_prompt: Base system instructions.
        attr_guidance: Formatted attribute descriptions, one per line.

    Returns:
        Complete system prompt for direct reasoning generation.
    """
    return f"""{generic_system_prompt}

You are an expert reasoning generator for deep-judgment template parsing. Your role is to explain how the response informs each attribute value.

# Task Overview

You will receive:
1. An original question in <original_question> tags
2. The full response text in <response> tags

Your task: Generate **reasoning** explaining how the response content should inform each attribute's value.

# Attribute Definitions

For each attribute below, the description indicates what value it expects:

{attr_guidance}

# Reasoning Protocol

For each attribute:
1. **Review the attribute's description** to understand what value it expects
2. **Analyze the response** to find relevant content for this attribute
3. **Generate reasoning** (2-3 sentences) explaining:
   - What parts of the response are relevant to this attribute
   - What value the attribute should have based on the response
   - Any ambiguities or confidence issues

When no relevant content exists: explain why and how this affects the attribute.

# Critical Requirements

**All Attributes**: Generate reasoning for EVERY attribute listed above.

**Evidence-Based**: Base reasoning on the actual response content.

# What NOT to Do

- Do NOT skip any attributes
- Do NOT fabricate information not present in the response"""


def build_reasoning_only_user_prompt(
    question_text: str,
    raw_llm_response: str,
) -> str:
    """Build user prompt for reasoning-only deep judgment.

    Args:
        question_text: Original question text.
        raw_llm_response: Full response text to reason about.

    Returns:
        User prompt with question and response in XML tags.
    """
    return f"""<original_question>
{question_text}
</original_question>

<response>
{raw_llm_response}
</response>"""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd karenina && uv run pytest tests/unit/prompts/deep_judgment/test_reasoning_only_prompt.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(prompts): add reasoning-only prompt for template deep judgment

New prompt builders that generate per-attribute reasoning directly from
the response, without requiring prior excerpt extraction."
```

### Task 6: Implement reasoning-only path in `deep_judgment_parse()`

**Files:**
- Modify: `karenina/src/karenina/benchmark/verification/evaluators/template/deep_judgment.py`
- Test: `karenina/tests/unit/evaluators/template/test_deep_judgment_reasoning_only.py`

- [ ] **Step 1: Write failing test**

Create `karenina/tests/unit/evaluators/template/test_deep_judgment_reasoning_only.py`:

```python
"""Tests for reasoning-only path in deep_judgment_parse."""

import json
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from karenina.benchmark.verification.evaluators.template.deep_judgment import (
    deep_judgment_parse,
)
from karenina.schemas.verification.config import VerificationConfig


class SimpleAnswer(BaseModel):
    """Minimal answer template for testing."""
    id: str = ""
    drug_target: str = Field(description="The protein target")
    mechanism: str = Field(description="Mechanism of action")


def _make_mock_llm(reasoning_response: str, parse_response: str):
    """Create mock LLMPort that returns canned responses in order."""
    llm = MagicMock()
    responses = [
        MagicMock(content=reasoning_response, usage=None),
        MagicMock(content=parse_response, usage=None),
    ]
    llm.invoke = MagicMock(side_effect=responses)
    return llm


def _make_mock_parser(parsed_answer):
    """Create mock ParserPort."""
    parser = MagicMock()
    parser.capabilities = MagicMock(supports_structured_output=False)
    parser.parse = MagicMock(return_value=parsed_answer)
    return parser


@pytest.mark.unit
class TestDeepJudgmentReasoningOnly:
    """Verify reasoning-only path skips excerpts and produces reasoning."""

    def test_reasoning_only_skips_excerpts(self):
        reasoning_json = json.dumps({
            "drug_target": "Response mentions BCL-2 as the target.",
            "mechanism": "Inhibition mechanism described.",
        })
        parsed = SimpleAnswer(drug_target="BCL-2", mechanism="inhibition")
        llm = _make_mock_llm(reasoning_json, "unused")
        parser = _make_mock_parser(parsed)

        config = VerificationConfig(
            answering_models=[],
            parsing_models=[],
            deep_judgment_enabled=True,
            deep_judgment_reasoning_only=True,
        )

        result_answer, excerpts, reasoning, metadata = deep_judgment_parse(
            raw_llm_response="Venetoclax targets BCL-2 via inhibition.",
            RawAnswer=SimpleAnswer,
            parsing_model=MagicMock(),
            parsing_llm=llm,
            parser=parser,
            question_text="What drug targets BCL-2?",
            config=config,
            format_instructions="",
            combined_system_prompt="You are a parser.",
        )

        # Excerpts should be empty (not extracted)
        assert excerpts == {} or excerpts is None
        # Reasoning should be populated
        assert "drug_target" in reasoning
        assert "mechanism" in reasoning
        # Metadata should reflect reasoning-only
        assert "reasoning" in metadata["stages_completed"]
        assert "excerpts" not in metadata["stages_completed"]

    def test_reasoning_only_metadata_no_excerpt_retries(self):
        reasoning_json = json.dumps({"drug_target": "Reasoning text."})
        parsed = SimpleAnswer(drug_target="BCL-2", mechanism="inhibition")
        llm = _make_mock_llm(reasoning_json, "unused")
        parser = _make_mock_parser(parsed)

        config = VerificationConfig(
            answering_models=[],
            parsing_models=[],
            deep_judgment_enabled=True,
            deep_judgment_reasoning_only=True,
        )

        _, _, _, metadata = deep_judgment_parse(
            raw_llm_response="Response text.",
            RawAnswer=SimpleAnswer,
            parsing_model=MagicMock(),
            parsing_llm=llm,
            parser=parser,
            question_text="Q?",
            config=config,
            format_instructions="",
            combined_system_prompt="System.",
        )

        assert metadata["excerpt_retry_count"] == 0
        assert metadata.get("attributes_without_excerpts") is None or metadata["attributes_without_excerpts"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd karenina && uv run pytest tests/unit/evaluators/template/test_deep_judgment_reasoning_only.py -v`
Expected: FAIL (no reasoning-only branch exists)

- [ ] **Step 3: Add reasoning-only branch to `deep_judgment_parse()`**

In `karenina/src/karenina/benchmark/verification/evaluators/template/deep_judgment.py`:

1. Import the new prompt builders at the top:
```python
from karenina.benchmark.verification.prompts.deep_judgment.template.reasoning_only import (
    build_reasoning_only_system_prompt,
    build_reasoning_only_user_prompt,
)
```

2. After the prompt preparation section (after `attr_guidance` is built, around line 142), add a branch:

```python
    # ==========================================
    # REASONING-ONLY MODE (skip excerpt extraction)
    # ==========================================
    if getattr(config, "deep_judgment_reasoning_only", False):
        return _reasoning_only_parse(
            raw_llm_response=raw_llm_response,
            RawAnswer=RawAnswer,
            parsing_llm=parsing_llm,
            parser=parser,
            question_text=question_text,
            config=config,
            generic_system_prompt=generic_system_prompt,
            attr_guidance=attr_guidance,
            attribute_names=attribute_names,
            usage_tracker=usage_tracker,
            parsing_model_str=parsing_model_str,
            prompt_config=prompt_config,
            parsing_model=parsing_model,
            combined_system_prompt=combined_system_prompt,
        )
```

3. Add the `_reasoning_only_parse()` helper function (before `deep_judgment_parse`):

```python
def _reasoning_only_parse(
    raw_llm_response: str,
    RawAnswer: type[BaseAnswer],
    parsing_llm: LLMPort,
    parser: ParserPort,
    question_text: str,
    config: VerificationConfig,
    generic_system_prompt: str,
    attr_guidance: str,
    attribute_names: list[str],
    usage_tracker: Any | None,
    parsing_model_str: str | None,
    prompt_config: PromptConfig | None,
    parsing_model: ModelConfig,
    combined_system_prompt: str,
) -> tuple[BaseAnswer, dict, dict[str, str], dict[str, Any]]:
    """Execute reasoning-only deep judgment: reasoning -> parameters (no excerpts).

    Generates per-attribute reasoning directly from the response, then
    uses that reasoning to guide parameter extraction via ParserPort.
    """
    stages_completed = []
    model_calls = 0

    logger.info(
        "Starting reasoning-only deep judgment for %d attributes: %s",
        len(attribute_names),
        ", ".join(attribute_names),
    )

    # Stage 1: Generate reasoning directly from response
    reasoning_system = build_reasoning_only_system_prompt(
        generic_system_prompt=generic_system_prompt,
        attr_guidance=attr_guidance,
    )
    reasoning_user = build_reasoning_only_user_prompt(
        question_text=question_text,
        raw_llm_response=raw_llm_response,
    )

    reasoning_assembler = PromptAssembler(
        task=PromptTask.DJ_TEMPLATE_REASONING,
        interface=parsing_model.interface,
        capabilities=PortCapabilities(),
    )
    reasoning_messages = reasoning_assembler.assemble(
        system_text=reasoning_system,
        user_text=reasoning_user,
        user_instructions=prompt_config.get_for_task(PromptTask.DJ_TEMPLATE_REASONING.value)
        if prompt_config
        else None,
    )

    llm_response = parsing_llm.invoke(reasoning_messages)
    model_calls += 1

    if usage_tracker and parsing_model_str:
        usage_metadata = getattr(llm_response, "usage", None)
        if usage_metadata:
            usage_tracker.track_call(
                "deep_judgment_reasoning_only", parsing_model_str, usage_metadata
            )

    # Parse reasoning JSON
    cleaned_response = _clean_json_response(llm_response.content)
    try:
        reasoning_raw = {} if cleaned_response is None else json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse reasoning-only JSON: %s", e)
        reasoning_raw = {}

    reasoning = {}
    for attr, value in reasoning_raw.items():
        reasoning[attr] = str(value) if not isinstance(value, dict) else value.get("reasoning", str(value))

    stages_completed.append("reasoning")
    logger.info(
        "Reasoning-only stage completed: generated reasoning for %d attributes",
        len(reasoning),
    )

    # Stage 2: Parse answer using reasoning as context
    reasoning_text = f"""Original Question: {question_text}

Reasoning Traces (explaining how the response informs each attribute value):

{format_reasoning_for_parsing(reasoning)}"""

    # Use ParserPort for final extraction
    parser_assembler = PromptAssembler(
        task=PromptTask.PARSING,
        interface=parsing_model.interface,
        capabilities=parser.capabilities,
    )
    parser_messages = parser_assembler.assemble(
        system_text=combined_system_prompt,
        user_text=reasoning_text,
        user_instructions=prompt_config.get_for_task(PromptTask.PARSING.value)
        if prompt_config
        else None,
        instruction_context={
            "json_schema": build_parsing_schema(RawAnswer),
            "format_instructions": "",
        },
    )

    parsed_answer = parser.parse(RawAnswer, parser_messages)
    model_calls += 1
    stages_completed.append("parameters")

    metadata = {
        "stages_completed": stages_completed,
        "model_calls": model_calls,
        "excerpt_retry_count": 0,
        "attributes_without_excerpts": [],
        "reasoning_only": True,
    }

    return parsed_answer, {}, reasoning, metadata
```

Note: `_clean_json_response`, `format_reasoning_for_parsing`, `build_parsing_schema`, `PromptAssembler`, `PromptTask`, `PortCapabilities` are already imported/available in the module. Check the existing imports at the top of the file and add any that are missing.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd karenina && uv run pytest tests/unit/evaluators/template/test_deep_judgment_reasoning_only.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(deep-judgment): implement reasoning-only path in deep_judgment_parse

When config.deep_judgment_reasoning_only=True, skips excerpt extraction
and generates per-attribute reasoning directly from the response.
Two LLM calls: reasoning generation + parameter extraction (vs 3-4 for
full deep judgment)."
```

### Task 7: Update auto-fail stage to skip for reasoning-only mode

**Files:**
- Modify: `karenina/src/karenina/benchmark/verification/stages/pipeline/deep_judgment_autofail.py`
- Test: `karenina/tests/unit/stages/test_deep_judgment_autofail_reasoning_only.py`

- [ ] **Step 1: Write failing test**

Create `karenina/tests/unit/stages/test_deep_judgment_autofail_reasoning_only.py`:

```python
"""Tests for auto-fail stage skipping in reasoning-only mode."""

import pytest
from unittest.mock import MagicMock

from karenina.benchmark.verification.stages.pipeline.deep_judgment_autofail import (
    DeepJudgmentAutoFailStage,
)


@pytest.mark.unit
class TestAutoFailReasoningOnly:
    """Auto-fail stage should not trigger in reasoning-only mode."""

    def test_should_not_run_when_reasoning_only(self):
        stage = DeepJudgmentAutoFailStage()
        context = MagicMock()
        context.has_prior_error = False
        context.get_artifact = MagicMock(side_effect=lambda key, default=None: {
            "deep_judgment_performed": True,
            "attributes_without_excerpts": [],  # empty, no excerpts attempted
            "deep_judgment_reasoning_only": True,
        }.get(key, default))

        assert stage.should_run(context) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd karenina && uv run pytest tests/unit/stages/test_deep_judgment_autofail_reasoning_only.py -v`
Expected: FAIL (auto-fail doesn't check reasoning_only flag)

- [ ] **Step 3: Update `should_run()` to check reasoning-only flag**

In `deep_judgment_autofail.py`, modify the `should_run` method to return `False` when reasoning-only mode was used. Add a check near the top of the condition:

```python
    # Skip auto-fail in reasoning-only mode (no excerpts to be missing)
    if context.get_artifact(ArtifactKeys.DEEP_JUDGMENT_REASONING_ONLY, False):
        return False
```

Add `DEEP_JUDGMENT_REASONING_ONLY = "deep_judgment_reasoning_only"` to `ArtifactKeys` in `base.py`.

In the finalize stage or wherever deep judgment metadata is stored, set this artifact when reasoning-only mode was used.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd karenina && uv run pytest tests/unit/stages/test_deep_judgment_autofail_reasoning_only.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "fix(deep-judgment): skip auto-fail stage in reasoning-only mode

Auto-fail checks for missing excerpts, which don't apply when excerpt
extraction was intentionally skipped."
```

### Task 8: Wire reasoning-only flag through the parse template stage

**Files:**
- Modify: `karenina/src/karenina/benchmark/verification/evaluators/template/evaluator.py` (pass flag to DJ config)
- Modify: `karenina/src/karenina/benchmark/verification/stages/pipeline/parse_template.py` (store reasoning-only artifact)

- [ ] **Step 1: Update TemplateEvaluator to pass `deep_judgment_reasoning_only` to DJ config**

In `evaluator.py`, where `dj_config = VerificationConfig(...)` is created (around line 400), add:

```python
    deep_judgment_reasoning_only=deep_judgment_config.get("reasoning_only", False),
```

- [ ] **Step 2: Update parse template stage to store reasoning-only artifact**

In the parse template stage, after deep judgment parse completes, store the flag:

```python
if dj_metadata.get("reasoning_only", False):
    context.set_artifact(ArtifactKeys.DEEP_JUDGMENT_REASONING_ONLY, True)
```

- [ ] **Step 3: Run full deep judgment test suite**

Run: `cd karenina && uv run pytest tests/ -x -q -k "deep_judgment" --timeout=120`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat(pipeline): wire reasoning-only flag through template evaluation stage"
```

### Task 9: LLM integration test for reasoning-only mode

**Files:**
- Create: `karenina/tests/integration/test_deep_judgment_reasoning_only_api.py`

- [ ] **Step 1: Write integration test**

Create `karenina/tests/integration/test_deep_judgment_reasoning_only_api.py`:

```python
"""Integration test: reasoning-only deep judgment with real LLM.

Requires ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable.
"""

import os

import pytest

from karenina.benchmark.verification.evaluators.template import (
    TemplateEvaluator,
    deep_judgment_parse,
)
from karenina.schemas.verification.config import VerificationConfig


# Skip if no API key available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.deep_judgment,
    pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"),
        reason="No LLM API key available",
    ),
]


# Use a simple answer template for testing
from pydantic import BaseModel, Field


class DrugAnswer(BaseModel):
    """Simple drug info template."""
    id: str = ""
    drug_name: str = Field(description="Name of the drug")
    target_protein: str = Field(description="The protein target of the drug")
    approved: bool = Field(description="Whether the drug is FDA approved")


class TestReasoningOnlyWithRealLLM:
    """Verify reasoning-only mode produces valid reasoning with a real LLM."""

    def test_produces_reasoning_without_excerpts(self):
        """Core test: reasoning-only mode generates reasoning, no excerpts."""
        from karenina.adapters import get_llm
        from karenina.schemas.config import ModelConfig

        model_config = ModelConfig(
            provider="anthropic",
            model_name="claude-haiku-4-5-20251001",
        )
        llm = get_llm(model_config)
        parser = get_llm(model_config)  # Use same for parsing

        config = VerificationConfig(
            answering_models=[],
            parsing_models=[model_config],
            deep_judgment_enabled=True,
            deep_judgment_reasoning_only=True,
        )

        response_text = (
            "Venetoclax (brand name Venclexta) is an FDA-approved drug that "
            "selectively targets the BCL-2 protein. It received FDA approval "
            "in 2016 for treatment of chronic lymphocytic leukemia."
        )

        parsed_answer, excerpts, reasoning, metadata = deep_judgment_parse(
            raw_llm_response=response_text,
            RawAnswer=DrugAnswer,
            parsing_model=model_config,
            parsing_llm=llm,
            parser=parser,
            question_text="What drug targets BCL-2?",
            config=config,
            format_instructions="",
            combined_system_prompt="You are an expert biomedical parser.",
        )

        # Excerpts should be empty
        assert excerpts == {} or not excerpts

        # Reasoning should exist for all attributes
        assert "drug_name" in reasoning
        assert "target_protein" in reasoning
        assert "approved" in reasoning

        # Each reasoning entry should be non-empty
        for attr, text in reasoning.items():
            assert len(text) > 10, f"Reasoning for {attr} too short: {text}"

        # Metadata should reflect reasoning-only
        assert "reasoning" in metadata["stages_completed"]
        assert "excerpts" not in metadata["stages_completed"]
        assert metadata.get("reasoning_only") is True
```

- [ ] **Step 2: Run integration test (requires API key)**

Run: `cd karenina && uv run pytest tests/integration/test_deep_judgment_reasoning_only_api.py -v --timeout=120`
Expected: PASS (with valid API key)

Note: This test will be skipped in CI without API keys. Verify locally before committing.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "test(integration): add real LLM test for reasoning-only deep judgment

Verifies end-to-end that reasoning-only mode produces per-attribute
reasoning without excerpt extraction using a real LLM call."
```

---

## Stream D: Documentation and Skills

### Task 10: Update deep judgment template docs

**Files:**
- Modify: `karenina/docs/advanced-pipeline/deep-judgment-templates.md`

- [ ] **Step 1: Add reasoning-only mode section**

In `deep-judgment-templates.md`, add a new section (after the main configuration section) documenting:

- What reasoning-only mode is (per-attribute reasoning without excerpt extraction)
- When to use it (want reasoning traces but not the cost of excerpt extraction + validation)
- Configuration: `deep_judgment_enabled=True, deep_judgment_reasoning_only=True`
- Cost comparison: 2 LLM calls (reasoning + parse) vs 3-4 for full DJ
- Result structure: `attribute_reasoning` populated, `extracted_excerpts` empty
- Auto-fail behavior: skipped (no excerpts to validate)

- [ ] **Step 2: Update the mode comparison table**

Add reasoning-only to any existing comparison of DJ modes:

| Mode | Excerpts | Reasoning | Search | LLM Calls |
|------|----------|-----------|--------|-----------|
| Disabled | No | No | No | 0 |
| Reasoning-only | No | Yes | No | 2 |
| Full | Yes | Yes | No | 3 |
| Full + Search | Yes | Yes | Yes | 4+ |

- [ ] **Step 3: Commit**

```bash
git add karenina/docs/advanced-pipeline/deep-judgment-templates.md
git commit -m "docs: add reasoning-only mode to deep judgment template documentation"
```

### Task 11: Update skills

**Files:**
- Modify: `.claude/skills/using-karenina/references/advanced-pipeline/deep-judgment-templates.md`
- Modify: `.claude/skills/karenina-verification/references/config-fields.md`

- [ ] **Step 1: Update using-karenina skill reference**

Mirror the doc changes from Task 10 into the skill reference file at `.claude/skills/using-karenina/references/advanced-pipeline/deep-judgment-templates.md`.

- [ ] **Step 2: Update config-fields reference**

In `.claude/skills/karenina-verification/references/config-fields.md`, add `deep_judgment_reasoning_only` to the config field listing.

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/
git commit -m "docs(skills): update skills with reasoning-only deep judgment config"
```

### Task 12: Final regression check

- [ ] **Step 1: Run full test suite**

```bash
cd karenina && uv run pytest tests/ -x -q --timeout=120
```

Expected: All tests pass.

- [ ] **Step 2: Run rubric-specific and dataframe-specific tests**

```bash
cd karenina && uv run pytest tests/ -x -q -k "rubric or dataframe or deep_judgment" --timeout=120
```

Expected: All pass.

---

## Parallelization Guide

These streams are independent and can be dispatched as parallel subagents:

| Stream | Tasks | Dependencies |
|--------|-------|-------------|
| **A** (Issue 156) | Task 1 | None |
| **B** (Issue 020) | Tasks 2-3 | Task 2 before Task 3 |
| **C** (Issue 154) | Tasks 4-9 | Sequential within stream |
| **D** (Docs) | Tasks 10-12 | After A, B, C complete |

**Recommended dispatch:** A+B+C in parallel, then D after all three complete.
