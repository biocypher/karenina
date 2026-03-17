"""Integration tests for LLMRubricTrait evaluation.

These tests verify that LLMRubricTrait correctly evaluates qualitative
traits using LLM-as-judge, leveraging the fixture-backed LLM client.

Test scenarios:
- Boolean trait evaluation (pass/fail)
- Scored trait evaluation (1-5 scale)
- Multi-trait batch evaluation
- Trait configuration and validation
- Integration with Rubric structures

The tests use captured LLM fixtures from tests/fixtures/llm_responses/
to ensure deterministic and reproducible results.

Fixtures used:
- rubric_evaluation/f9dca28ae... (boolean clarity trait: true)
- rubric_evaluation/4f11c90e... (scored completeness trait: 1)
- rubric_evaluation/8b368df2... (multi-trait: accuracy, helpfulness, safety)
"""

import pytest

from karenina.schemas.entities import LLMRubricTrait, Rubric

# =============================================================================
# LLMRubricTrait Configuration Tests
# =============================================================================


@pytest.mark.integration
class TestLLMRubricTraitConfiguration:
    """Test LLMRubricTrait configuration and validation."""

    def test_boolean_trait_configuration(self):
        """Verify boolean trait has correct defaults."""
        trait = LLMRubricTrait(
            name="clarity",
            description="The response is clear and easy to understand",
            kind="boolean",
            higher_is_better=True,
        )

        assert trait.name == "clarity"
        assert trait.kind == "boolean"
        assert trait.higher_is_better is True
        # Score bounds should still have defaults
        assert trait.min_score == 1
        assert trait.max_score == 5

    def test_scored_trait_configuration(self):
        """Verify scored trait has correct configuration."""
        trait = LLMRubricTrait(
            name="completeness",
            description="The response thoroughly addresses all aspects",
            kind="score",
            min_score=1,
            max_score=5,
            higher_is_better=True,
        )

        assert trait.name == "completeness"
        assert trait.kind == "score"
        assert trait.min_score == 1
        assert trait.max_score == 5

    def test_custom_score_range(self):
        """Verify custom score range is accepted."""
        trait = LLMRubricTrait(
            name="quality",
            description="Overall quality rating",
            kind="score",
            min_score=0,
            max_score=10,
            higher_is_better=True,
        )

        assert trait.min_score == 0
        assert trait.max_score == 10

    def test_lower_is_better_configuration(self):
        """Verify lower_is_better configuration works."""
        trait = LLMRubricTrait(
            name="error_count",
            description="Number of errors in the response",
            kind="score",
            min_score=0,
            max_score=10,
            higher_is_better=False,  # Fewer errors is better
        )

        assert trait.higher_is_better is False

    def test_deep_judgment_disabled_by_default(self):
        """Verify deep judgment is disabled by default."""
        trait = LLMRubricTrait(
            name="test",
            kind="boolean",
            higher_is_better=True,
        )

        assert trait.deep_judgment_enabled is False
        assert trait.deep_judgment_excerpt_enabled is True  # But excerpt is enabled by default if DJ enabled

    def test_deep_judgment_configuration(self):
        """Verify deep judgment can be enabled with options."""
        trait = LLMRubricTrait(
            name="accuracy",
            kind="boolean",
            higher_is_better=True,
            deep_judgment_enabled=True,
            deep_judgment_excerpt_enabled=True,
            deep_judgment_max_excerpts=3,
            deep_judgment_fuzzy_match_threshold=0.8,
            deep_judgment_excerpt_retry_attempts=2,
        )

        assert trait.deep_judgment_enabled is True
        assert trait.deep_judgment_max_excerpts == 3
        assert trait.deep_judgment_fuzzy_match_threshold == 0.8


# =============================================================================
# LLMRubricTrait Evaluation Tests with Fixtures
# =============================================================================


@pytest.mark.integration
class TestLLMRubricTraitEvaluation:
    """Test LLMRubricTrait evaluation using fixture-backed LLM.

    These tests use the rubric_evaluator fixture which has a fixture-backed
    LLM client that returns captured responses.
    """

    def test_boolean_trait_returns_true(self, rubric_evaluator):
        """Test boolean trait evaluation returning True.

        Uses fixture: rubric_evaluation/f9dca28ae... (clarity: true)
        Question: "What is the capital of France?"
        Answer: "Paris is the capital of France. It is a major European city."
        """
        trait = LLMRubricTrait(
            name="clarity",
            description="The response is clear, unambiguous, and easy to understand",
            kind="boolean",
            higher_is_better=True,
        )

        rubric = Rubric(llm_traits=[trait])

        # This should match the fixture for boolean clarity trait
        scores, labels, usage = rubric_evaluator.evaluate_rubric(
            question="What is the capital of France?",
            answer="Paris is the capital of France. It is a major European city.",
            rubric=rubric,
        )

        assert "clarity" in scores
        # The fixture returns true for this clear answer
        assert scores["clarity"] is True

    def test_scored_trait_returns_low_score(self, rubric_evaluator):
        """Test scored trait evaluation returning low score.

        Uses fixture: rubric_evaluation/4f11c90e... (completeness: 1)
        Question: "Explain the process of photosynthesis."
        Answer: "Photosynthesis converts sunlight to energy."
        """
        trait = LLMRubricTrait(
            name="completeness",
            description="The response thoroughly addresses all aspects of the question",
            kind="score",
            min_score=1,
            max_score=5,
            higher_is_better=True,
        )

        rubric = Rubric(llm_traits=[trait])

        scores, labels, usage = rubric_evaluator.evaluate_rubric(
            question="Explain the process of photosynthesis.",
            answer="Photosynthesis converts sunlight to energy.",
            rubric=rubric,
        )

        assert "completeness" in scores
        # The fixture returns 1 (low score) for this incomplete answer
        assert scores["completeness"] == 1

    def test_multi_trait_evaluation(self, rubric_evaluator):
        """Test evaluating multiple traits in one call.

        Uses fixture: rubric_evaluation/8b368df2... (multi-trait batch)
        Question: "How do I treat a minor burn?"
        Answer: "For a minor burn, run cool water over it..."
        Expected: accuracy=True, helpfulness=5, safety=True
        """
        traits = [
            LLMRubricTrait(
                name="accuracy",
                description="The response contains factually correct information",
                kind="boolean",
                higher_is_better=True,
            ),
            LLMRubricTrait(
                name="helpfulness",
                description="The response is helpful and addresses the user's actual need",
                kind="score",
                min_score=1,
                max_score=5,
                higher_is_better=True,
            ),
            LLMRubricTrait(
                name="safety",
                description="The response does not contain harmful or dangerous advice",
                kind="boolean",
                higher_is_better=True,
            ),
        ]

        rubric = Rubric(llm_traits=traits)

        scores, labels, usage = rubric_evaluator.evaluate_rubric(
            question="How do I treat a minor burn?",
            answer="For a minor burn, run cool water over it for 10-20 minutes. Do not use ice directly. Apply aloe vera gel if available. Cover with a sterile bandage. Seek medical attention if the burn blisters or is larger than 3 inches.",
            rubric=rubric,
        )

        # Verify all traits were evaluated
        assert "accuracy" in scores
        assert "helpfulness" in scores
        assert "safety" in scores

        # Check expected values from fixture
        assert scores["accuracy"] is True
        assert scores["helpfulness"] == 5
        assert scores["safety"] is True


# =============================================================================
# Integration with Rubric Fixtures
# =============================================================================


@pytest.mark.integration
class TestLLMRubricTraitWithRubricFixtures:
    """Test LLMRubricTrait integration with conftest rubric fixtures."""

    def test_boolean_rubric_fixture(self, boolean_rubric):
        """Verify boolean_rubric fixture has correct structure."""
        assert len(boolean_rubric.llm_traits) == 1
        trait = boolean_rubric.llm_traits[0]
        assert trait.name == "clarity"
        assert trait.kind == "boolean"

    def test_scored_rubric_fixture(self, scored_rubric):
        """Verify scored_rubric fixture has correct structure."""
        assert len(scored_rubric.llm_traits) == 1
        trait = scored_rubric.llm_traits[0]
        assert trait.name == "completeness"
        assert trait.kind == "score"
        assert trait.min_score == 1
        assert trait.max_score == 5

    def test_multi_trait_rubric_fixture(self, multi_trait_rubric):
        """Verify multi_trait_rubric fixture has mixed traits."""
        # Should have LLM traits and regex traits
        assert len(multi_trait_rubric.llm_traits) == 2
        assert len(multi_trait_rubric.regex_traits) == 1

        # Check LLM trait types
        trait_kinds = {t.kind for t in multi_trait_rubric.llm_traits}
        assert "boolean" in trait_kinds
        assert "score" in trait_kinds


# =============================================================================
# LLMRubricTrait with Trace Fixtures
# =============================================================================


@pytest.mark.integration
class TestLLMRubricTraitWithTraceFixtures:
    """Test LLMRubricTrait configuration for trace-relevant scenarios."""

    def test_citation_quality_trait(self, trace_with_citations: str):
        """Verify trait can be configured for citation quality assessment."""
        trait = LLMRubricTrait(
            name="citation_quality",
            description="The response properly cites sources with numbered references",
            kind="score",
            min_score=1,
            max_score=5,
            higher_is_better=True,
        )

        assert trait.name == "citation_quality"
        # Trace has proper citations, would score well
        assert "[1]" in trace_with_citations

    def test_scientific_accuracy_trait(self, trace_with_citations: str):
        """Verify trait can be configured for scientific accuracy."""
        trait = LLMRubricTrait(
            name="scientific_accuracy",
            description="The response contains accurate scientific information",
            kind="boolean",
            higher_is_better=True,
        )

        assert trait.kind == "boolean"
        # Trace discusses BCL2 accurately
        assert "BCL2" in trace_with_citations

    def test_abstention_detection_trait(self, trace_with_abstention: str):
        """Verify trait can be configured for abstention detection."""
        trait = LLMRubricTrait(
            name="provides_answer",
            description="The response provides a substantive answer rather than refusing",
            kind="boolean",
            higher_is_better=True,
        )

        assert trait.kind == "boolean"
        # Abstention trace would score False on this trait
        assert "cannot" in trace_with_abstention.lower()

    def test_hedging_detection_trait(self, trace_with_hedging: str):
        """Verify trait can be configured for hedging detection."""
        trait = LLMRubricTrait(
            name="confidence_level",
            description="The response demonstrates appropriate confidence (not overconfident, not overly hedging)",
            kind="score",
            min_score=1,
            max_score=5,
            higher_is_better=True,
        )

        assert trait.kind == "score"
        # Hedging trace has uncertainty language
        assert "cannot be completely certain" in trace_with_hedging.lower()


# =============================================================================
# Rubric Serialization Tests
# =============================================================================


@pytest.mark.integration
class TestLLMRubricTraitSerialization:
    """Test LLMRubricTrait serialization and roundtrip."""

    def test_boolean_trait_roundtrip(self):
        """Verify boolean trait survives serialization."""
        trait = LLMRubricTrait(
            name="clarity",
            description="Clear and understandable",
            kind="boolean",
            higher_is_better=True,
        )

        rubric = Rubric(llm_traits=[trait])
        data = rubric.model_dump()
        restored = Rubric(**data)

        assert len(restored.llm_traits) == 1
        assert restored.llm_traits[0].name == "clarity"
        assert restored.llm_traits[0].kind == "boolean"

    def test_scored_trait_roundtrip(self):
        """Verify scored trait survives serialization."""
        trait = LLMRubricTrait(
            name="quality",
            description="Overall quality",
            kind="score",
            min_score=0,
            max_score=10,
            higher_is_better=True,
        )

        rubric = Rubric(llm_traits=[trait])
        data = rubric.model_dump()
        restored = Rubric(**data)

        assert len(restored.llm_traits) == 1
        assert restored.llm_traits[0].name == "quality"
        assert restored.llm_traits[0].min_score == 0
        assert restored.llm_traits[0].max_score == 10

    def test_deep_judgment_settings_roundtrip(self):
        """Verify deep judgment settings survive serialization."""
        trait = LLMRubricTrait(
            name="accuracy",
            kind="boolean",
            higher_is_better=True,
            deep_judgment_enabled=True,
            deep_judgment_max_excerpts=5,
            deep_judgment_fuzzy_match_threshold=0.9,
        )

        rubric = Rubric(llm_traits=[trait])
        data = rubric.model_dump()
        restored = Rubric(**data)

        restored_trait = restored.llm_traits[0]
        assert restored_trait.deep_judgment_enabled is True
        assert restored_trait.deep_judgment_max_excerpts == 5
        assert restored_trait.deep_judgment_fuzzy_match_threshold == 0.9


# =============================================================================
# Edge Cases
# =============================================================================


@pytest.mark.integration
class TestLLMRubricTraitEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_trait_with_long_description(self):
        """Verify trait handles long descriptions."""
        long_desc = "This is a very detailed description " * 20
        trait = LLMRubricTrait(
            name="detailed",
            description=long_desc,
            kind="boolean",
            higher_is_better=True,
        )

        assert len(trait.description) > 500

    def test_trait_with_special_characters(self):
        """Verify trait handles special characters in name/description."""
        trait = LLMRubricTrait(
            name="code_quality",
            description="Checks for proper use of: {}, [], (), and other syntax",
            kind="score",
            min_score=1,
            max_score=5,
            higher_is_better=True,
        )

        assert "{}" in trait.description

    def test_trait_with_unicode(self):
        """Verify trait handles unicode in description."""
        trait = LLMRubricTrait(
            name="multilingual",
            description="Évaluation de la qualité • 品质评估 • Оценка качества",
            kind="boolean",
            higher_is_better=True,
        )

        assert "品质评估" in trait.description

    def test_rubric_with_many_traits(self):
        """Verify rubric handles many LLM traits."""
        traits = [
            LLMRubricTrait(
                name=f"trait_{i}",
                description=f"Description for trait {i}",
                kind="boolean" if i % 2 == 0 else "score",
                higher_is_better=True,
            )
            for i in range(10)
        ]

        rubric = Rubric(llm_traits=traits)
        assert len(rubric.llm_traits) == 10
        assert len(rubric.get_llm_trait_names()) == 10

    def test_trait_directionality_preserved(self):
        """Verify higher_is_better is correctly preserved."""
        trait_higher = LLMRubricTrait(
            name="good",
            kind="score",
            higher_is_better=True,
        )
        trait_lower = LLMRubricTrait(
            name="bad",
            kind="score",
            higher_is_better=False,
        )

        rubric = Rubric(llm_traits=[trait_higher, trait_lower])
        directionalities = rubric.get_trait_directionalities()

        assert directionalities["good"] is True
        assert directionalities["bad"] is False
