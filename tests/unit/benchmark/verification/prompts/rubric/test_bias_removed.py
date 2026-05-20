"""Tests asserting the rewritten rubric prompts no longer carry the
'lean toward false / true for positive / negative traits' conservative bias.

The bias used to live in:
- LLMTraitPromptBuilder.build_batch_system_prompt
- LLMTraitPromptBuilder._build_single_boolean_system_prompt
- LLMTraitPromptBuilder._build_single_score_system_prompt
- LLMTraitPromptBuilder.build_template_system_prompt
- DeepJudgmentPromptBuilder.build_score_extraction_system_prompt

These tests pin its removal so it does not creep back in.
"""

from __future__ import annotations

import re

import pytest

from karenina.benchmark.verification.prompts.deep_judgment.rubric.deep_judgment import (
    DeepJudgmentPromptBuilder,
)
from karenina.benchmark.verification.prompts.rubric.llm_trait import LLMTraitPromptBuilder
from karenina.schemas.entities import LLMRubricTrait

_BIAS_PATTERNS = [
    re.compile(r"lean\s+toward\s+`?false`?", re.IGNORECASE),
    re.compile(r"lean\s+toward\s+`?true`?", re.IGNORECASE),
    re.compile(r"lean\s+toward\s+(lower|higher)\s+scores", re.IGNORECASE),
]


def _has_bias(text: str) -> bool:
    return any(pattern.search(text) for pattern in _BIAS_PATTERNS)


@pytest.fixture
def boolean_trait() -> LLMRubricTrait:
    return LLMRubricTrait(
        name="GroundedAnswer",
        description="Answer is grounded in the retrieved evidence.",
        kind="boolean",
        higher_is_better=True,
    )


@pytest.fixture
def score_trait() -> LLMRubricTrait:
    return LLMRubricTrait(
        name="Clarity",
        description="Answer is clear and well-structured.",
        kind="score",
        min_score=1,
        max_score=5,
        higher_is_better=True,
    )


class TestLLMTraitSystemPromptsNoBias:
    def test_batch_system_prompt_has_no_bias(self) -> None:
        prompt = LLMTraitPromptBuilder().build_batch_system_prompt()
        assert not _has_bias(prompt), prompt

    def test_single_boolean_system_prompt_has_no_bias(self, boolean_trait: LLMRubricTrait) -> None:
        prompt = LLMTraitPromptBuilder()._build_single_boolean_system_prompt(boolean_trait)
        assert not _has_bias(prompt), prompt

    def test_single_score_system_prompt_has_no_bias(self, score_trait: LLMRubricTrait) -> None:
        prompt = LLMTraitPromptBuilder()._build_single_score_system_prompt(score_trait)
        assert not _has_bias(prompt), prompt

    def test_template_system_prompt_has_no_bias(self, boolean_trait: LLMRubricTrait) -> None:
        prompt = LLMTraitPromptBuilder().build_template_system_prompt(boolean_trait)
        assert not _has_bias(prompt), prompt


class TestDeepJudgmentSystemPromptsNoBias:
    def test_score_extraction_boolean_system_prompt_has_no_bias(self) -> None:
        prompt = DeepJudgmentPromptBuilder().build_score_extraction_system_prompt_boolean()
        assert not _has_bias(prompt), prompt

    def test_score_extraction_numeric_system_prompt_has_no_bias(self) -> None:
        prompt = DeepJudgmentPromptBuilder().build_score_extraction_system_prompt_numeric()
        assert not _has_bias(prompt), prompt

    def test_reasoning_system_prompt_has_no_bias(self) -> None:
        prompt = DeepJudgmentPromptBuilder().build_reasoning_system_prompt()
        assert not _has_bias(prompt), prompt
