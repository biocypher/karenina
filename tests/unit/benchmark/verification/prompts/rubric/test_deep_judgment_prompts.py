"""Structural tests for the redesigned DeepJudgmentPromptBuilder.

These tests pin the new prompt layout (Role / Principles / Anti-patterns /
Output handoff) and the integer-anchor scoring helper. They are independent of
the existing bias and task_eval_mode tests.
"""

from __future__ import annotations

import pytest

from karenina.benchmark.verification.prompts.deep_judgment.rubric.deep_judgment import (
    DeepJudgmentPromptBuilder,
    build_integer_score_labels,
)
from karenina.schemas.entities import LLMRubricTrait

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


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


# ----------------------------------------------------------------------
# build_integer_score_labels
# ----------------------------------------------------------------------


class TestBuildIntegerScoreLabels:
    def test_one_to_three_uses_lookup(self) -> None:
        result = build_integer_score_labels(1, 3)
        assert result == [
            (1, "Does not meet"),
            (2, "Partially meets"),
            (3, "Fully meets"),
        ]

    def test_one_to_five_uses_lookup(self) -> None:
        result = build_integer_score_labels(1, 5)
        assert result == [
            (1, "Poor"),
            (2, "Below average"),
            (3, "Adequate"),
            (4, "Strong"),
            (5, "Excellent"),
        ]

    def test_one_to_seven_uses_lookup(self) -> None:
        result = build_integer_score_labels(1, 7)
        assert [pair[0] for pair in result] == [1, 2, 3, 4, 5, 6, 7]
        assert result[0] == (1, "Very poor")
        assert result[-1] == (7, "Excellent")
        assert len(result) == 7

    def test_one_to_ten_uses_lookup(self) -> None:
        result = build_integer_score_labels(1, 10)
        assert [pair[0] for pair in result] == list(range(1, 11))
        assert result[0][1].lower().startswith("poor") or result[0][1].lower().startswith("very")
        assert result[-1] == (10, "Excellent")

    def test_zero_to_two_falls_back_to_ladder(self) -> None:
        # Atypical range: triggers the fallback ladder.
        result = build_integer_score_labels(0, 2)
        assert [pair[0] for pair in result] == [0, 1, 2]
        # First label is the lowest rung; last is the highest.
        assert "lowest" in result[0][1].lower() or result[0][1].lower() == "lowest"
        assert "highest" in result[-1][1].lower() or result[-1][1].lower() == "highest"

    def test_two_to_six_falls_back_to_ladder(self) -> None:
        # Range length 5 but not starting at 1 -> fallback.
        result = build_integer_score_labels(2, 6)
        assert [pair[0] for pair in result] == [2, 3, 4, 5, 6]
        # Five labels, all distinct.
        labels = [label for _, label in result]
        assert len(set(labels)) == 5

    def test_invalid_range_raises(self) -> None:
        with pytest.raises(ValueError):
            build_integer_score_labels(5, 1)


# ----------------------------------------------------------------------
# Stage 1: Excerpt extraction
# ----------------------------------------------------------------------


class TestExcerptExtractionPrompts:
    def test_system_prompt_has_four_sections(self) -> None:
        prompt = DeepJudgmentPromptBuilder().build_excerpt_extraction_system_prompt()
        assert "## Role" in prompt
        assert "## Principles" in prompt
        assert "## Anti-patterns" in prompt
        assert "## Output handoff" in prompt

    def test_system_prompt_states_verbatim_rule(self) -> None:
        prompt = DeepJudgmentPromptBuilder().build_excerpt_extraction_system_prompt()
        assert "Verbatim only" in prompt

    def test_system_prompt_defines_confidence_levels(self) -> None:
        prompt = DeepJudgmentPromptBuilder().build_excerpt_extraction_system_prompt()
        assert "high:" in prompt
        assert "medium:" in prompt
        assert "low:" in prompt

    def test_system_prompt_includes_anti_patterns(self) -> None:
        prompt = DeepJudgmentPromptBuilder().build_excerpt_extraction_system_prompt()
        assert "Paraphrasing" in prompt
        assert "Inventing quotes" in prompt

    def test_user_prompt_uses_h2_sections(self, boolean_trait: LLMRubricTrait) -> None:
        prompt = DeepJudgmentPromptBuilder().build_excerpt_extraction_user_prompt(
            trait=boolean_trait,
            max_excerpts=3,
            answer="The target is BCL2 based on the SPRINT trial.",
        )
        assert "## Trait" in prompt
        assert "## Criteria" in prompt
        assert "## Response" in prompt
        assert "## Maximum excerpts" in prompt
        assert "## Task" in prompt
        assert "**TRAIT:**" not in prompt
        assert "**ANSWER TO ANALYZE:**" not in prompt

    def test_user_prompt_omits_static_rules(self, boolean_trait: LLMRubricTrait) -> None:
        prompt = DeepJudgmentPromptBuilder().build_excerpt_extraction_user_prompt(
            trait=boolean_trait,
            max_excerpts=3,
            answer="The target is BCL2.",
        )
        # The old layout had "**CONFIDENCE LEVELS:**" and "**IMPORTANT RULES:**"
        # blocks in the user prompt; the new layout keeps those rules in the
        # system prompt only.
        assert "**CONFIDENCE LEVELS" not in prompt
        assert "**IMPORTANT RULES" not in prompt

    def test_user_prompt_includes_retry_feedback_when_provided(self, boolean_trait: LLMRubricTrait) -> None:
        prompt = DeepJudgmentPromptBuilder().build_excerpt_extraction_user_prompt(
            trait=boolean_trait,
            max_excerpts=3,
            answer="The target is BCL2.",
            feedback="Excerpt #1 not found in answer.",
        )
        assert "## Retry feedback" in prompt
        assert "Excerpt #1 not found in answer." in prompt

    def test_user_prompt_omits_retry_feedback_when_absent(self, boolean_trait: LLMRubricTrait) -> None:
        prompt = DeepJudgmentPromptBuilder().build_excerpt_extraction_user_prompt(
            trait=boolean_trait,
            max_excerpts=3,
            answer="The target is BCL2.",
        )
        assert "## Retry feedback" not in prompt

    def test_retry_feedback_contains_failed_excerpts(self) -> None:
        feedback = DeepJudgmentPromptBuilder().build_retry_feedback(
            failed_excerpts=[
                {"text": "fake quote", "similarity_score": 0.42},
            ],
            fuzzy_threshold=0.85,
        )
        assert "fake quote" in feedback
        assert "0.42" in feedback
        assert "0.85" in feedback
        assert "Please provide verbatim quotes" not in feedback


# ----------------------------------------------------------------------
# Stage 1.5: Hallucination assessment
# ----------------------------------------------------------------------


class TestHallucinationAssessmentPrompts:
    def test_system_prompt_has_four_sections(self) -> None:
        prompt = DeepJudgmentPromptBuilder().build_hallucination_assessment_system_prompt()
        assert "## Role" in prompt
        assert "## Principles" in prompt
        assert "## Anti-patterns" in prompt
        assert "## Output handoff" in prompt

    def test_system_prompt_defines_risk_levels(self) -> None:
        prompt = DeepJudgmentPromptBuilder().build_hallucination_assessment_system_prompt()
        assert "none:" in prompt
        assert "low:" in prompt
        assert "medium:" in prompt
        assert "high:" in prompt

    def test_system_prompt_includes_skepticism_anti_patterns(self) -> None:
        prompt = DeepJudgmentPromptBuilder().build_hallucination_assessment_system_prompt()
        assert "empty or irrelevant" in prompt.lower()
        assert "single matching snippet" in prompt.lower()

    def test_user_prompt_is_data_only(self) -> None:
        prompt = DeepJudgmentPromptBuilder().build_hallucination_assessment_user_prompt(
            excerpt_text="BCL2 was the target.",
            search_results="No matches found.",
        )
        assert "## Excerpt" in prompt
        assert "## Search results" in prompt
        assert "## Task" in prompt
        assert "**EXCERPT TO VERIFY:**" not in prompt
        assert "**RISK LEVELS" not in prompt
        assert "EVALUATION GUIDELINES" not in prompt.upper()

    def test_user_prompt_includes_input_data(self) -> None:
        prompt = DeepJudgmentPromptBuilder().build_hallucination_assessment_user_prompt(
            excerpt_text="BCL2 was the target.",
            search_results="Five sources confirm BCL2.",
        )
        assert "BCL2 was the target." in prompt
        assert "Five sources confirm BCL2." in prompt
