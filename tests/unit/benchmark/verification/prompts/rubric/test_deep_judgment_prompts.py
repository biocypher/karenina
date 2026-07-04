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


# ----------------------------------------------------------------------
# Stage 2: Reasoning generation
# ----------------------------------------------------------------------


class TestReasoningPrompts:
    def test_system_prompt_has_four_sections(self) -> None:
        prompt = DeepJudgmentPromptBuilder().build_reasoning_system_prompt()
        assert "## Role" in prompt
        assert "## Principles" in prompt
        assert "## Anti-patterns" in prompt
        assert "## Output handoff" in prompt

    def test_system_prompt_describes_evidence_interpretation_conclusion(self) -> None:
        prompt = DeepJudgmentPromptBuilder().build_reasoning_system_prompt()
        assert "Evidence" in prompt
        assert "Interpretation" in prompt
        assert "Conclusion" in prompt

    def test_system_prompt_warns_against_scoring(self) -> None:
        prompt = DeepJudgmentPromptBuilder().build_reasoning_system_prompt()
        assert "score" in prompt.lower()
        assert "next stage" in prompt.lower()

    def test_user_with_excerpts_uses_h2_sections(self, boolean_trait: LLMRubricTrait) -> None:
        prompt = DeepJudgmentPromptBuilder().build_reasoning_user_prompt_with_excerpts(
            question="What is the target?",
            trait=boolean_trait,
            excerpts=[{"text": "BCL2 is targeted.", "confidence": "high"}],
            hallucination_risk=None,
            task_eval_mode=False,
        )
        assert "## Trait" in prompt
        assert "## Criteria" in prompt
        assert "## Question" in prompt
        assert "## Excerpts" in prompt
        assert "## Task" in prompt
        assert "**Trait**:" not in prompt
        assert "**Question**:" not in prompt

    def test_user_with_excerpts_renders_overall_risk_when_provided(self, boolean_trait: LLMRubricTrait) -> None:
        prompt = DeepJudgmentPromptBuilder().build_reasoning_user_prompt_with_excerpts(
            question="What is the target?",
            trait=boolean_trait,
            excerpts=[{"text": "BCL2 is targeted.", "confidence": "high"}],
            hallucination_risk={"overall_risk": "low"},
            task_eval_mode=False,
        )
        assert "## Overall hallucination risk" in prompt
        assert "low" in prompt

    def test_user_with_excerpts_omits_overall_risk_when_absent(self, boolean_trait: LLMRubricTrait) -> None:
        prompt = DeepJudgmentPromptBuilder().build_reasoning_user_prompt_with_excerpts(
            question="What is the target?",
            trait=boolean_trait,
            excerpts=[{"text": "BCL2 is targeted.", "confidence": "high"}],
            hallucination_risk=None,
            task_eval_mode=False,
        )
        assert "## Overall hallucination risk" not in prompt

    def test_user_with_excerpts_renders_per_excerpt_risk_when_present(self, boolean_trait: LLMRubricTrait) -> None:
        prompt = DeepJudgmentPromptBuilder().build_reasoning_user_prompt_with_excerpts(
            question="What is the target?",
            trait=boolean_trait,
            excerpts=[
                {"text": "BCL2 is targeted.", "confidence": "high", "hallucination_risk": "medium"},
            ],
            hallucination_risk={"overall_risk": "medium"},
            task_eval_mode=False,
        )
        assert "hallucination risk: medium" in prompt

    def test_user_without_excerpts_uses_h2_sections(self, boolean_trait: LLMRubricTrait) -> None:
        prompt = DeepJudgmentPromptBuilder().build_reasoning_user_prompt_without_excerpts(
            question="What is the target?",
            answer="The target is BCL2.",
            trait=boolean_trait,
            task_eval_mode=False,
        )
        assert "## Trait" in prompt
        assert "## Criteria" in prompt
        assert "## Question" in prompt
        assert "## Response" in prompt
        assert "## Task" in prompt

    def test_user_with_excerpts_task_skeleton_is_present(self, boolean_trait: LLMRubricTrait) -> None:
        prompt = DeepJudgmentPromptBuilder().build_reasoning_user_prompt_with_excerpts(
            question="Q?",
            trait=boolean_trait,
            excerpts=[{"text": "x", "confidence": "low"}],
            hallucination_risk=None,
            task_eval_mode=False,
        )
        assert "Evidence:" in prompt
        assert "Interpretation:" in prompt
        assert "Conclusion:" in prompt


# ----------------------------------------------------------------------
# Stage 3: Score extraction
# ----------------------------------------------------------------------


class TestScoreExtractionPrompts:
    def test_boolean_system_prompt_has_four_sections(self) -> None:
        prompt = DeepJudgmentPromptBuilder().build_score_extraction_system_prompt_boolean()
        assert "## Role" in prompt
        assert "## Principles" in prompt
        assert "## Anti-patterns" in prompt
        assert "## Output handoff" in prompt

    def test_boolean_score_prompts_route_hedged_reasoning_by_conclusion(self, boolean_trait: LLMRubricTrait) -> None:
        """Regression: hedged reasoning with a negative conclusion must not default true.

        The score-extraction stage only sees the reasoning text plus these
        prompts. A realistic failure is a reasoning stage that acknowledges
        partial support but concludes the criteria are not met; boolean judges
        tend to over-reward the hedge unless the system prompt tells them to
        mirror the Conclusion line and avoid directional defaults.
        """
        builder = DeepJudgmentPromptBuilder()
        system_prompt = builder.build_score_extraction_system_prompt_boolean()
        user_prompt = builder.build_score_extraction_user_prompt(
            trait=boolean_trait,
            reasoning=(
                "Evidence: The answer cites one source but leaves the main claim unsupported.\n"
                "Interpretation: The evidence is mixed and the response sounds confident.\n"
                "Conclusion: the criteria are not met."
            ),
        )

        anti_patterns = system_prompt.split("## Anti-patterns", 1)[1].split("## Output handoff", 1)[0]
        assert "The verdict must follow logically from the reasoning's Conclusion." in system_prompt
        assert "Biasing toward true or false when the reasoning is hedged or balanced" in anti_patterns
        assert "verdict should mirror the Conclusion line" in anti_patterns
        assert "directional default" in anti_patterns
        assert "Conclusion: the criteria are not met." in user_prompt
        assert "otherwise false" in user_prompt

    def test_numeric_system_prompt_has_four_sections(self) -> None:
        prompt = DeepJudgmentPromptBuilder().build_score_extraction_system_prompt_numeric()
        assert "## Role" in prompt
        assert "## Principles" in prompt
        assert "## Anti-patterns" in prompt
        assert "## Output handoff" in prompt

    def test_numeric_system_prompt_warns_against_middle_clustering(self) -> None:
        prompt = DeepJudgmentPromptBuilder().build_score_extraction_system_prompt_numeric()
        assert "middle" in prompt.lower()
        assert "balanced" in prompt.lower()

    def test_boolean_user_prompt_uses_h2_sections(self, boolean_trait: LLMRubricTrait) -> None:
        prompt = DeepJudgmentPromptBuilder().build_score_extraction_user_prompt(
            trait=boolean_trait,
            reasoning="Evidence: ...\nInterpretation: ...\nConclusion: criteria met.",
        )
        assert "## Trait" in prompt
        assert "## Criteria" in prompt
        assert "## Reasoning" in prompt
        assert "## Task" in prompt

    def test_numeric_user_prompt_renders_all_integer_anchors(self, score_trait: LLMRubricTrait) -> None:
        prompt = DeepJudgmentPromptBuilder().build_score_extraction_user_prompt(
            trait=score_trait,
            reasoning="Evidence: ...\nInterpretation: ...\nConclusion: adequate.",
        )
        assert "## Scale" in prompt
        for n in range(1, 6):
            assert f"{n} = " in prompt
        assert "Poor" in prompt
        assert "Excellent" in prompt

    def test_numeric_user_prompt_handles_atypical_range(self) -> None:
        atypical_trait = LLMRubricTrait(
            name="OffScale",
            description="Test trait.",
            kind="score",
            min_score=2,
            max_score=4,
            higher_is_better=True,
        )
        prompt = DeepJudgmentPromptBuilder().build_score_extraction_user_prompt(
            trait=atypical_trait,
            reasoning="Evidence: ...\nInterpretation: ...\nConclusion: medium.",
        )
        for n in range(2, 5):
            assert f"{n} = " in prompt


# ----------------------------------------------------------------------
# Cross-cutting: layout invariants
# ----------------------------------------------------------------------


class TestLayoutInvariants:
    SYSTEM_PROMPT_FACTORIES = [
        "build_excerpt_extraction_system_prompt",
        "build_hallucination_assessment_system_prompt",
        "build_reasoning_system_prompt",
        "build_score_extraction_system_prompt_boolean",
        "build_score_extraction_system_prompt_numeric",
    ]

    @pytest.mark.parametrize("method_name", SYSTEM_PROMPT_FACTORIES)
    def test_every_system_prompt_has_four_sections(self, method_name: str) -> None:
        builder = DeepJudgmentPromptBuilder()
        prompt = getattr(builder, method_name)()
        for section in ("## Role", "## Principles", "## Anti-patterns", "## Output handoff"):
            assert section in prompt, f"{method_name} is missing {section}"

    @pytest.mark.parametrize("method_name", SYSTEM_PROMPT_FACTORIES)
    def test_no_old_uppercase_headers(self, method_name: str) -> None:
        builder = DeepJudgmentPromptBuilder()
        prompt = getattr(builder, method_name)()
        for header in ("**TRAIT:**", "**CRITICAL REQUIREMENTS:**", "**RISK LEVELS"):
            assert header not in prompt, f"{method_name} still uses old header {header}"

    def test_user_prompts_omit_principles_lists(
        self, boolean_trait: LLMRubricTrait, score_trait: LLMRubricTrait
    ) -> None:
        builder = DeepJudgmentPromptBuilder()
        prompts = [
            builder.build_excerpt_extraction_user_prompt(trait=boolean_trait, max_excerpts=3, answer="x"),
            builder.build_hallucination_assessment_user_prompt(excerpt_text="x", search_results="y"),
            builder.build_reasoning_user_prompt_with_excerpts(
                question="q",
                trait=boolean_trait,
                excerpts=[{"text": "x", "confidence": "low"}],
                hallucination_risk=None,
                task_eval_mode=False,
            ),
            builder.build_reasoning_user_prompt_without_excerpts(
                question="q", answer="a", trait=boolean_trait, task_eval_mode=False
            ),
            builder.build_score_extraction_user_prompt(trait=boolean_trait, reasoning="r"),
            builder.build_score_extraction_user_prompt(trait=score_trait, reasoning="r"),
        ]
        for prompt in prompts:
            # Role/Principles/Anti-patterns/Output handoff sections are
            # system-prompt-only; user prompts must not contain them.
            assert "## Role" not in prompt
            assert "## Principles" not in prompt
            assert "## Anti-patterns" not in prompt
            assert "## Output handoff" not in prompt
