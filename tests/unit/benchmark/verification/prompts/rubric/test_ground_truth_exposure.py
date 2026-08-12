"""Tests for REFERENCE ANSWER block rendering in rubric trait prompts.

An ``LLMRubricTrait`` may opt in to ground-truth exposure via
``include_ground_truth=True``. Opted-in traits receive the question's
reference answer in a **REFERENCE ANSWER:** block in the rendered user
prompt. The block is omitted for traits that keep the default (False) and
whenever the ground truth is empty or None. The block renders even in
``task_eval_mode``: question suppression is about redundancy with the
logged context, while ground truth is never redundant.
"""

from __future__ import annotations

import pytest

from karenina.benchmark.verification.prompts.rubric.literal_trait import LiteralTraitPromptBuilder
from karenina.benchmark.verification.prompts.rubric.llm_trait import LLMTraitPromptBuilder
from karenina.schemas.entities import LLMRubricTrait


@pytest.fixture
def grounded_trait() -> LLMRubricTrait:
    return LLMRubricTrait(
        name="grounded",
        description="Answer matches the reference.",
        kind="boolean",
        include_ground_truth=True,
    )


@pytest.fixture
def plain_trait() -> LLMRubricTrait:
    return LLMRubricTrait(
        name="plain",
        description="Answer is clear.",
        kind="boolean",
    )


@pytest.fixture
def grounded_literal_trait() -> LLMRubricTrait:
    return LLMRubricTrait(
        name="sentiment_grounded",
        description="Sentiment relative to the reference.",
        kind="literal",
        classes={
            "positive": "Expresses approval or optimism.",
            "negative": "Expresses disapproval or concern.",
        },
        higher_is_better=None,
        include_ground_truth=True,
    )


@pytest.fixture
def plain_literal_trait() -> LLMRubricTrait:
    return LLMRubricTrait(
        name="sentiment_plain",
        description="Overall sentiment of the answer.",
        kind="literal",
        classes={
            "positive": "Expresses approval or optimism.",
            "negative": "Expresses disapproval or concern.",
        },
        higher_is_better=None,
    )


@pytest.mark.unit
class TestIncludeGroundTruthField:
    def test_defaults_to_false(self) -> None:
        trait = LLMRubricTrait(name="t", description="d", kind="boolean")
        assert trait.include_ground_truth is False


# ------------------------------------------------------------------
# LLMTraitPromptBuilder: batch
# ------------------------------------------------------------------


@pytest.mark.unit
class TestBatchPromptGroundTruth:
    def test_renders_reference_for_opted_in_trait(self, grounded_trait: LLMRubricTrait) -> None:
        prompt = LLMTraitPromptBuilder().build_batch_user_prompt(
            question="q",
            answer="a",
            traits=[grounded_trait],
            ground_truth="the reference",
        )
        assert "REFERENCE ANSWER" in prompt
        assert "the reference" in prompt

    def test_omits_reference_by_default(self, plain_trait: LLMRubricTrait) -> None:
        prompt = LLMTraitPromptBuilder().build_batch_user_prompt(
            question="q",
            answer="a",
            traits=[plain_trait],
            ground_truth="the reference",
        )
        assert "REFERENCE ANSWER" not in prompt
        assert "the reference" not in prompt

    def test_omits_reference_when_ground_truth_none(self, grounded_trait: LLMRubricTrait) -> None:
        prompt = LLMTraitPromptBuilder().build_batch_user_prompt(
            question="q",
            answer="a",
            traits=[grounded_trait],
            ground_truth=None,
        )
        assert "REFERENCE ANSWER" not in prompt

    def test_omits_reference_when_ground_truth_empty(self, grounded_trait: LLMRubricTrait) -> None:
        prompt = LLMTraitPromptBuilder().build_batch_user_prompt(
            question="q",
            answer="a",
            traits=[grounded_trait],
            ground_truth="",
        )
        assert "REFERENCE ANSWER" not in prompt

    def test_renders_reference_in_task_eval_mode(self, grounded_trait: LLMRubricTrait) -> None:
        prompt = LLMTraitPromptBuilder().build_batch_user_prompt(
            question="q",
            answer="a",
            traits=[grounded_trait],
            task_eval_mode=True,
            ground_truth="the reference",
        )
        assert "REFERENCE ANSWER" in prompt
        assert "the reference" in prompt


# ------------------------------------------------------------------
# LLMTraitPromptBuilder: single trait
# ------------------------------------------------------------------


@pytest.mark.unit
class TestSingleTraitPromptGroundTruth:
    def test_single_trait_renders_reference_for_opted_in_trait(self, grounded_trait: LLMRubricTrait) -> None:
        prompt = LLMTraitPromptBuilder().build_single_trait_user_prompt(
            question="q",
            answer="a",
            trait=grounded_trait,
            ground_truth="the reference",
        )
        assert "REFERENCE ANSWER" in prompt
        assert "the reference" in prompt

    def test_single_trait_omits_reference_by_default(self, plain_trait: LLMRubricTrait) -> None:
        prompt = LLMTraitPromptBuilder().build_single_trait_user_prompt(
            question="q",
            answer="a",
            trait=plain_trait,
            ground_truth="the reference",
        )
        assert "REFERENCE ANSWER" not in prompt
        assert "the reference" not in prompt


# ------------------------------------------------------------------
# LLMTraitPromptBuilder: template kind
# ------------------------------------------------------------------


@pytest.mark.unit
class TestTemplatePromptGroundTruth:
    def test_template_renders_reference_for_opted_in_trait(self, grounded_trait: LLMRubricTrait) -> None:
        prompt = LLMTraitPromptBuilder().build_template_user_prompt(
            question="q",
            answer="a",
            trait=grounded_trait,
            ground_truth="the reference",
        )
        assert "REFERENCE ANSWER" in prompt
        assert "the reference" in prompt

    def test_template_omits_reference_by_default(self, plain_trait: LLMRubricTrait) -> None:
        prompt = LLMTraitPromptBuilder().build_template_user_prompt(
            question="q",
            answer="a",
            trait=plain_trait,
            ground_truth="the reference",
        )
        assert "REFERENCE ANSWER" not in prompt


# ------------------------------------------------------------------
# LiteralTraitPromptBuilder
# ------------------------------------------------------------------


@pytest.mark.unit
class TestLiteralPromptGroundTruth:
    def test_batch_renders_reference_for_opted_in_trait(self, grounded_literal_trait: LLMRubricTrait) -> None:
        prompt = LiteralTraitPromptBuilder().build_batch_user_prompt(
            question="q",
            answer="a",
            traits=[grounded_literal_trait],
            ground_truth="the reference",
        )
        assert "REFERENCE ANSWER" in prompt
        assert "the reference" in prompt

    def test_batch_omits_reference_by_default(self, plain_literal_trait: LLMRubricTrait) -> None:
        prompt = LiteralTraitPromptBuilder().build_batch_user_prompt(
            question="q",
            answer="a",
            traits=[plain_literal_trait],
            ground_truth="the reference",
        )
        assert "REFERENCE ANSWER" not in prompt

    def test_single_trait_renders_reference_for_opted_in_trait(self, grounded_literal_trait: LLMRubricTrait) -> None:
        prompt = LiteralTraitPromptBuilder().build_single_trait_user_prompt(
            question="q",
            answer="a",
            trait=grounded_literal_trait,
            ground_truth="the reference",
        )
        assert "REFERENCE ANSWER" in prompt
        assert "the reference" in prompt

    def test_single_trait_omits_reference_by_default(self, plain_literal_trait: LLMRubricTrait) -> None:
        prompt = LiteralTraitPromptBuilder().build_single_trait_user_prompt(
            question="q",
            answer="a",
            trait=plain_literal_trait,
            ground_truth="the reference",
        )
        assert "REFERENCE ANSWER" not in prompt
