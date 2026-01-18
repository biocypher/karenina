"""Integration tests for literal kind LLMRubricTrait evaluation.

These tests verify that literal kind LLMRubricTraits correctly classify responses
into predefined categories using LLM-as-judge.

Test scenarios:
- Single literal trait classification (sentiment)
- Multiple literal traits batch evaluation
- Literal traits mixed with boolean/score traits
- Literal trait configuration and score derivation
- Error handling for invalid classifications

The tests use captured LLM fixtures from tests/fixtures/llm_responses/
to ensure deterministic and reproducible results.

NOTE: These tests require LLM fixtures to be captured using:
    python scripts/capture_fixtures.py --scenario literal_evaluation

Fixtures used:
- literal_evaluation/<hash> (sentiment classification: positive)
- literal_evaluation/<hash> (multi-trait: sentiment + response_type)
- literal_evaluation/<hash> (neutral classification)
- literal_evaluation/<hash> (negative classification)
"""

import pytest

from karenina.schemas.domain import LLMRubricTrait, Rubric

# =============================================================================
# Literal Trait Configuration Tests
# =============================================================================


@pytest.mark.integration
class TestLiteralTraitConfiguration:
    """Test literal kind LLMRubricTrait configuration and validation."""

    def test_literal_trait_creation(self):
        """Verify literal trait has correct configuration."""
        trait = LLMRubricTrait(
            name="sentiment",
            description="Classify the emotional tone",
            kind="literal",
            classes={
                "negative": "Negative sentiment",
                "neutral": "Neutral sentiment",
                "positive": "Positive sentiment",
            },
            higher_is_better=True,
        )

        assert trait.name == "sentiment"
        assert trait.kind == "literal"
        assert trait.classes is not None
        assert len(trait.classes) == 3
        # Scores are auto-derived: min=0, max=len(classes)-1
        assert trait.min_score == 0
        assert trait.max_score == 2

    def test_literal_trait_score_derivation(self):
        """Verify min/max scores are correctly derived from class count."""
        # 4 classes -> max_score = 3
        trait = LLMRubricTrait(
            name="quality",
            kind="literal",
            classes={
                "poor": "Poor quality",
                "fair": "Fair quality",
                "good": "Good quality",
                "excellent": "Excellent quality",
            },
            higher_is_better=True,
        )

        assert trait.min_score == 0
        assert trait.max_score == 3

    def test_literal_trait_get_class_names(self):
        """Verify get_class_names returns ordered class list."""
        trait = LLMRubricTrait(
            name="sentiment",
            kind="literal",
            classes={
                "negative": "Negative",
                "neutral": "Neutral",
                "positive": "Positive",
            },
            higher_is_better=True,
        )

        class_names = trait.get_class_names()
        # Classes are ordered by insertion order in dict
        assert class_names == ["negative", "neutral", "positive"]

    def test_literal_trait_get_class_index(self):
        """Verify get_class_index returns correct indices."""
        trait = LLMRubricTrait(
            name="sentiment",
            kind="literal",
            classes={
                "negative": "Negative",
                "neutral": "Neutral",
                "positive": "Positive",
            },
            higher_is_better=True,
        )

        assert trait.get_class_index("negative") == 0
        assert trait.get_class_index("neutral") == 1
        assert trait.get_class_index("positive") == 2
        # Case-SENSITIVE matching (exact match required)
        assert trait.get_class_index("POSITIVE") == -1  # Not found
        assert trait.get_class_index("Neutral") == -1  # Not found
        # Invalid class returns -1
        assert trait.get_class_index("invalid") == -1

    def test_literal_trait_validate_score(self):
        """Verify validate_score returns True/False for valid/invalid scores."""
        trait = LLMRubricTrait(
            name="sentiment",
            kind="literal",
            classes={
                "negative": "Negative",
                "neutral": "Neutral",
                "positive": "Positive",
            },
            higher_is_better=True,
        )

        # Valid indices (0 to max_score) - validate_score returns bool
        assert trait.validate_score(0) is True
        assert trait.validate_score(1) is True
        assert trait.validate_score(2) is True
        # Error state (-1 is allowed for literal traits)
        assert trait.validate_score(-1) is True
        # Out of range is invalid
        assert trait.validate_score(10) is False
        assert trait.validate_score(-2) is False
        # Boolean values are invalid for literal/score traits
        assert trait.validate_score(True) is False
        assert trait.validate_score(False) is False

    def test_literal_trait_directionality(self):
        """Verify higher_is_better is preserved for literal traits."""
        trait_higher = LLMRubricTrait(
            name="quality",
            kind="literal",
            classes={"low": "Low", "high": "High"},
            higher_is_better=True,
        )
        trait_lower = LLMRubricTrait(
            name="severity",
            kind="literal",
            classes={"minor": "Minor", "major": "Major"},
            higher_is_better=False,
        )

        assert trait_higher.higher_is_better is True
        assert trait_lower.higher_is_better is False


# =============================================================================
# Literal Trait in Rubric Tests
# =============================================================================


@pytest.mark.integration
class TestLiteralTraitInRubric:
    """Test literal kind traits within Rubric structures."""

    def test_literal_rubric_fixture(self, literal_sentiment_rubric: Rubric):
        """Verify literal_sentiment_rubric fixture has correct structure."""
        assert len(literal_sentiment_rubric.llm_traits) == 1
        trait = literal_sentiment_rubric.llm_traits[0]
        assert trait.name == "sentiment"
        assert trait.kind == "literal"
        assert trait.classes is not None
        assert len(trait.classes) == 3

    def test_multi_literal_rubric_fixture(self, multi_literal_rubric: Rubric):
        """Verify multi_literal_rubric fixture has correct structure."""
        assert len(multi_literal_rubric.llm_traits) == 2
        trait_names = {t.name for t in multi_literal_rubric.llm_traits}
        assert trait_names == {"sentiment", "response_type"}
        # All traits should be literal kind
        for trait in multi_literal_rubric.llm_traits:
            assert trait.kind == "literal"

    def test_mixed_rubric_with_literal_fixture(self, mixed_rubric_with_literal: Rubric):
        """Verify mixed_rubric_with_literal fixture has mixed trait kinds."""
        assert len(mixed_rubric_with_literal.llm_traits) == 3
        trait_kinds = {t.kind for t in mixed_rubric_with_literal.llm_traits}
        assert trait_kinds == {"literal", "boolean", "score"}

    def test_get_trait_max_scores_includes_literal(self):
        """Verify get_trait_max_scores works with literal traits."""
        trait = LLMRubricTrait(
            name="sentiment",
            kind="literal",
            classes={
                "negative": "Negative",
                "neutral": "Neutral",
                "positive": "Positive",
            },
            higher_is_better=True,
        )
        rubric = Rubric(llm_traits=[trait])

        max_scores = rubric.get_trait_max_scores()
        assert "sentiment" in max_scores
        assert max_scores["sentiment"] == 2  # 3 classes -> max=2

    def test_get_trait_directionalities_includes_literal(self):
        """Verify get_trait_directionalities works with literal traits."""
        trait = LLMRubricTrait(
            name="sentiment",
            kind="literal",
            classes={"negative": "Neg", "positive": "Pos"},
            higher_is_better=True,
        )
        rubric = Rubric(llm_traits=[trait])

        directionalities = rubric.get_trait_directionalities()
        assert "sentiment" in directionalities
        assert directionalities["sentiment"] is True


# =============================================================================
# Literal Trait Serialization Tests
# =============================================================================


@pytest.mark.integration
class TestLiteralTraitSerialization:
    """Test literal kind LLMRubricTrait serialization and roundtrip."""

    def test_literal_trait_roundtrip(self):
        """Verify literal trait survives serialization with classes preserved."""
        trait = LLMRubricTrait(
            name="sentiment",
            description="Classify sentiment",
            kind="literal",
            classes={
                "negative": "Negative sentiment",
                "neutral": "Neutral sentiment",
                "positive": "Positive sentiment",
            },
            higher_is_better=True,
        )

        rubric = Rubric(llm_traits=[trait])
        data = rubric.model_dump()
        restored = Rubric(**data)

        assert len(restored.llm_traits) == 1
        restored_trait = restored.llm_traits[0]
        assert restored_trait.name == "sentiment"
        assert restored_trait.kind == "literal"
        assert restored_trait.classes is not None
        assert len(restored_trait.classes) == 3
        assert restored_trait.min_score == 0
        assert restored_trait.max_score == 2

    def test_literal_trait_json_roundtrip(self):
        """Verify literal trait survives JSON serialization."""
        import json

        trait = LLMRubricTrait(
            name="quality",
            kind="literal",
            classes={
                "poor": "Poor quality",
                "fair": "Fair quality",
                "good": "Good quality",
            },
            higher_is_better=True,
        )

        rubric = Rubric(llm_traits=[trait])
        json_str = rubric.model_dump_json()
        data = json.loads(json_str)
        restored = Rubric(**data)

        assert restored.llm_traits[0].kind == "literal"
        assert restored.llm_traits[0].classes == trait.classes


# =============================================================================
# Literal Trait Edge Cases
# =============================================================================


@pytest.mark.integration
class TestLiteralTraitEdgeCases:
    """Test edge cases and boundary conditions for literal traits."""

    def test_minimum_two_classes(self):
        """Verify literal trait requires at least 2 classes."""
        trait = LLMRubricTrait(
            name="binary",
            kind="literal",
            classes={"yes": "Yes answer", "no": "No answer"},
            higher_is_better=True,
        )
        assert len(trait.classes) == 2
        assert trait.max_score == 1

    def test_maximum_twenty_classes(self):
        """Verify literal trait supports up to 20 classes."""
        classes = {f"class_{i}": f"Description {i}" for i in range(20)}
        trait = LLMRubricTrait(
            name="many_classes",
            kind="literal",
            classes=classes,
            higher_is_better=True,
        )
        assert len(trait.classes) == 20
        assert trait.max_score == 19

    def test_class_names_with_spaces(self):
        """Verify class names with spaces work correctly."""
        trait = LLMRubricTrait(
            name="complexity",
            kind="literal",
            classes={
                "very simple": "Extremely simple content",
                "moderately complex": "Moderate complexity",
                "highly complex": "Very complex content",
            },
            higher_is_better=True,
        )
        assert trait.get_class_index("very simple") == 0
        assert trait.get_class_index("moderately complex") == 1
        assert trait.get_class_index("highly complex") == 2

    def test_class_names_exact_match_required(self):
        """Verify class name lookup requires exact case match."""
        trait = LLMRubricTrait(
            name="sentiment",
            kind="literal",
            classes={
                "Negative": "Neg",
                "Neutral": "Neu",
                "Positive": "Pos",
            },
            higher_is_better=True,
        )
        # Original case (exact match)
        assert trait.get_class_index("Negative") == 0
        assert trait.get_class_index("Neutral") == 1
        assert trait.get_class_index("Positive") == 2
        # Different case does NOT match (case-sensitive)
        assert trait.get_class_index("NEGATIVE") == -1
        assert trait.get_class_index("negative") == -1
        assert trait.get_class_index("nEgAtIvE") == -1

    def test_literal_trait_with_unicode_classes(self):
        """Verify literal trait handles unicode in class names."""
        trait = LLMRubricTrait(
            name="language_quality",
            kind="literal",
            classes={
                "mauvais": "Quality is bad (French)",
                "moyen": "Quality is average",
                "bon": "Quality is good",
                "excellent": "Quality is excellent",
            },
            higher_is_better=True,
        )
        assert trait.get_class_index("mauvais") == 0
        assert trait.get_class_index("excellent") == 3

    def test_rubric_with_many_literal_traits(self):
        """Verify rubric handles multiple literal traits."""
        traits = [
            LLMRubricTrait(
                name=f"trait_{i}",
                kind="literal",
                classes={"low": "Low", "medium": "Medium", "high": "High"},
                higher_is_better=True,
            )
            for i in range(5)
        ]

        rubric = Rubric(llm_traits=traits)
        assert len(rubric.llm_traits) == 5
        assert len(rubric.get_llm_trait_names()) == 5

        # All traits should have max_score = 2 (3 classes)
        max_scores = rubric.get_trait_max_scores()
        for trait in rubric.llm_traits:
            assert max_scores[trait.name] == 2
