"""Tests for the ``task_eval_mode`` flag on rubric prompt builders.

When ``task_eval_mode=True`` (set by TaskEval, where the rubric is evaluated on
a logged response without a real question), the ``**QUESTION:**`` block must be
omitted entirely from rendered user prompts: no header, no placeholder. When
``task_eval_mode=False`` (the verification path), the QUESTION block is rendered
as before.

System prompts must not branch on the flag; the divergence is purely the
QUESTION-slot rendering.
"""

from __future__ import annotations

import re

import pytest

from karenina.benchmark.verification.prompts.deep_judgment.rubric.deep_judgment import (
    DeepJudgmentPromptBuilder,
)
from karenina.benchmark.verification.prompts.rubric.literal_trait import LiteralTraitPromptBuilder
from karenina.benchmark.verification.prompts.rubric.llm_trait import LLMTraitPromptBuilder
from karenina.schemas.entities import LLMRubricTrait

# Case-insensitive match for any QUESTION header variant rendered by the
# basic + literal-trait builders: "**QUESTION:**" / "**Question**:".
_QUESTION_HEADER_RE = re.compile(r"\*\*Question[: *]+", re.IGNORECASE)


def _question_header_present(text: str) -> bool:
    return bool(_QUESTION_HEADER_RE.search(text))


# Deep-judgment prompts use H2 markdown sections; match "## Question" at
# the start of a line.
_DJ_QUESTION_HEADER_RE = re.compile(r"^## Question\b", re.MULTILINE)


def _dj_question_header_present(text: str) -> bool:
    return bool(_DJ_QUESTION_HEADER_RE.search(text))


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


@pytest.fixture
def literal_trait() -> LLMRubricTrait:
    return LLMRubricTrait(
        name="Sentiment",
        description="Overall sentiment of the answer.",
        kind="literal",
        classes={
            "positive": "Expresses approval or optimism.",
            "neutral": "Neither approval nor disapproval.",
            "negative": "Expresses disapproval or concern.",
        },
        higher_is_better=None,
    )


# ------------------------------------------------------------------
# LLMTraitPromptBuilder
# ------------------------------------------------------------------


class TestLLMTraitPromptBuilderTaskEvalMode:
    def test_batch_user_prompt_renders_question_when_disabled(
        self, boolean_trait: LLMRubricTrait, score_trait: LLMRubricTrait
    ) -> None:
        prompt = LLMTraitPromptBuilder().build_batch_user_prompt(
            question="What is the target?",
            answer="The target is BCL2.",
            traits=[boolean_trait, score_trait],
            task_eval_mode=False,
        )
        assert _question_header_present(prompt)
        assert "What is the target?" in prompt

    def test_batch_user_prompt_omits_question_when_enabled(
        self, boolean_trait: LLMRubricTrait, score_trait: LLMRubricTrait
    ) -> None:
        prompt = LLMTraitPromptBuilder().build_batch_user_prompt(
            question="What is the target?",
            answer="The target is BCL2.",
            traits=[boolean_trait, score_trait],
            task_eval_mode=True,
        )
        assert not _question_header_present(prompt)
        assert "What is the target?" not in prompt
        # Sanity: the answer and traits sections still appear.
        assert "The target is BCL2." in prompt
        assert "GroundedAnswer" in prompt
        assert "Clarity" in prompt

    def test_single_trait_user_prompt_renders_question_when_disabled(self, boolean_trait: LLMRubricTrait) -> None:
        prompt = LLMTraitPromptBuilder().build_single_trait_user_prompt(
            question="What is the target?",
            answer="The target is BCL2.",
            trait=boolean_trait,
            task_eval_mode=False,
        )
        assert _question_header_present(prompt)
        assert "What is the target?" in prompt

    def test_single_trait_user_prompt_omits_question_when_enabled(self, boolean_trait: LLMRubricTrait) -> None:
        prompt = LLMTraitPromptBuilder().build_single_trait_user_prompt(
            question="What is the target?",
            answer="The target is BCL2.",
            trait=boolean_trait,
            task_eval_mode=True,
        )
        assert not _question_header_present(prompt)
        assert "What is the target?" not in prompt
        assert "The target is BCL2." in prompt
        assert "GroundedAnswer" in prompt

    def test_template_user_prompt_renders_question_when_disabled(self, boolean_trait: LLMRubricTrait) -> None:
        # build_template_user_prompt accepts any LLMRubricTrait; the kind check
        # is enforced at the evaluator level. Reusing the boolean fixture is
        # safe since this test only inspects rendered text.
        prompt = LLMTraitPromptBuilder().build_template_user_prompt(
            question="What is the target?",
            answer="The target is BCL2.",
            trait=boolean_trait,
            task_eval_mode=False,
        )
        assert _question_header_present(prompt)
        assert "What is the target?" in prompt

    def test_template_user_prompt_omits_question_when_enabled(self, boolean_trait: LLMRubricTrait) -> None:
        prompt = LLMTraitPromptBuilder().build_template_user_prompt(
            question="What is the target?",
            answer="The target is BCL2.",
            trait=boolean_trait,
            task_eval_mode=True,
        )
        assert not _question_header_present(prompt)
        assert "What is the target?" not in prompt


# ------------------------------------------------------------------
# LiteralTraitPromptBuilder
# ------------------------------------------------------------------


class TestLiteralTraitPromptBuilderTaskEvalMode:
    def test_batch_user_prompt_renders_question_when_disabled(self, literal_trait: LLMRubricTrait) -> None:
        prompt = LiteralTraitPromptBuilder().build_batch_user_prompt(
            question="How does it feel?",
            answer="Mostly hopeful but cautious.",
            traits=[literal_trait],
            task_eval_mode=False,
        )
        assert _question_header_present(prompt)
        assert "How does it feel?" in prompt

    def test_batch_user_prompt_omits_question_when_enabled(self, literal_trait: LLMRubricTrait) -> None:
        prompt = LiteralTraitPromptBuilder().build_batch_user_prompt(
            question="How does it feel?",
            answer="Mostly hopeful but cautious.",
            traits=[literal_trait],
            task_eval_mode=True,
        )
        assert not _question_header_present(prompt)
        assert "How does it feel?" not in prompt
        assert "Mostly hopeful but cautious." in prompt
        assert "Sentiment" in prompt

    def test_single_trait_user_prompt_renders_question_when_disabled(self, literal_trait: LLMRubricTrait) -> None:
        prompt = LiteralTraitPromptBuilder().build_single_trait_user_prompt(
            question="How does it feel?",
            answer="Mostly hopeful but cautious.",
            trait=literal_trait,
            task_eval_mode=False,
        )
        assert _question_header_present(prompt)
        assert "How does it feel?" in prompt

    def test_single_trait_user_prompt_omits_question_when_enabled(self, literal_trait: LLMRubricTrait) -> None:
        prompt = LiteralTraitPromptBuilder().build_single_trait_user_prompt(
            question="How does it feel?",
            answer="Mostly hopeful but cautious.",
            trait=literal_trait,
            task_eval_mode=True,
        )
        assert not _question_header_present(prompt)
        assert "How does it feel?" not in prompt


# ------------------------------------------------------------------
# DeepJudgmentPromptBuilder
# ------------------------------------------------------------------


class TestDeepJudgmentPromptBuilderTaskEvalMode:
    def test_reasoning_with_excerpts_renders_question_when_disabled(self, boolean_trait: LLMRubricTrait) -> None:
        prompt = DeepJudgmentPromptBuilder().build_reasoning_user_prompt_with_excerpts(
            question="What is the target?",
            trait=boolean_trait,
            excerpts=[{"text": "BCL2 is targeted.", "confidence": "high"}],
            hallucination_risk=None,
            task_eval_mode=False,
        )
        assert _dj_question_header_present(prompt)
        assert "What is the target?" in prompt

    def test_reasoning_with_excerpts_omits_question_when_enabled(self, boolean_trait: LLMRubricTrait) -> None:
        prompt = DeepJudgmentPromptBuilder().build_reasoning_user_prompt_with_excerpts(
            question="What is the target?",
            trait=boolean_trait,
            excerpts=[{"text": "BCL2 is targeted.", "confidence": "high"}],
            hallucination_risk=None,
            task_eval_mode=True,
        )
        assert not _dj_question_header_present(prompt)
        assert "What is the target?" not in prompt
        assert "BCL2 is targeted." in prompt
        assert boolean_trait.name in prompt

    def test_reasoning_without_excerpts_renders_question_when_disabled(self, boolean_trait: LLMRubricTrait) -> None:
        prompt = DeepJudgmentPromptBuilder().build_reasoning_user_prompt_without_excerpts(
            question="What is the target?",
            answer="The target is BCL2.",
            trait=boolean_trait,
            task_eval_mode=False,
        )
        assert _dj_question_header_present(prompt)
        assert "What is the target?" in prompt

    def test_reasoning_without_excerpts_omits_question_when_enabled(self, boolean_trait: LLMRubricTrait) -> None:
        prompt = DeepJudgmentPromptBuilder().build_reasoning_user_prompt_without_excerpts(
            question="What is the target?",
            answer="The target is BCL2.",
            trait=boolean_trait,
            task_eval_mode=True,
        )
        assert not _dj_question_header_present(prompt)
        assert "What is the target?" not in prompt
        assert "The target is BCL2." in prompt
        assert boolean_trait.name in prompt
