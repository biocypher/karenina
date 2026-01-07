"""Tests for higher_is_better rubric trait directionality feature.

This module tests the higher_is_better field added to LLMRubricTrait, RegexTrait,
and CallableTrait. It verifies:
1. Legacy checkpoint loading defaults to higher_is_better=True
2. Rubric.get_trait_directionalities() returns correct mapping
3. GEPA score inversion for higher_is_better=False traits
4. Checkpoint round-trip preserves higher_is_better
"""

import pytest

from karenina.schemas.domain import (
    CallableTrait,
    LLMRubricTrait,
    MetricRubricTrait,
    RegexTrait,
    Rubric,
)


class TestHigherIsBetterField:
    """Test higher_is_better field on trait schemas."""

    def test_llm_trait_higher_is_better_true(self) -> None:
        """Test creating LLM trait with higher_is_better=True."""
        trait = LLMRubricTrait(
            name="clarity",
            description="Is the response clear?",
            kind="boolean",
            higher_is_better=True,
        )
        assert trait.higher_is_better is True

    def test_llm_trait_higher_is_better_false(self) -> None:
        """Test creating LLM trait with higher_is_better=False."""
        trait = LLMRubricTrait(
            name="verbosity",
            description="Does the response contain unnecessary words?",
            kind="boolean",
            higher_is_better=False,
        )
        assert trait.higher_is_better is False

    def test_llm_score_trait_higher_is_better(self) -> None:
        """Test score trait with higher_is_better field."""
        trait = LLMRubricTrait(
            name="error_count",
            description="Number of errors (1=many, 5=none)",
            kind="score",
            min_score=1,
            max_score=5,
            higher_is_better=False,  # Lower error count is better
        )
        assert trait.higher_is_better is False
        assert trait.kind == "score"

    def test_regex_trait_higher_is_better(self) -> None:
        """Test regex trait with higher_is_better field."""
        trait = RegexTrait(
            name="contains_error",
            description="Does response contain error keyword?",
            pattern=r"\berror\b",
            higher_is_better=False,  # Not containing error is better
        )
        assert trait.higher_is_better is False

    def test_callable_trait_higher_is_better(self) -> None:
        """Test callable trait with higher_is_better field."""

        def evaluate(trace: str) -> int:
            return len(trace.split())

        trait = CallableTrait.from_callable(
            name="word_count",
            func=evaluate,
            kind="score",
            min_score=0,
            max_score=100,
            description="Number of words in response",
            higher_is_better=False,  # Fewer words might be better for conciseness
        )
        assert trait.higher_is_better is False

    def test_metric_trait_no_higher_is_better(self) -> None:
        """Test that MetricRubricTrait does not have higher_is_better field."""
        trait = MetricRubricTrait(
            name="extraction_accuracy",
            description="Precision/recall for extractions",
            evaluation_mode="tp_only",
            metrics=["precision", "recall", "f1"],
            tp_instructions=["correct extraction"],
        )
        # MetricRubricTrait should not have higher_is_better
        # (precision/recall/F1 are inherently "higher is better")
        assert not hasattr(trait, "higher_is_better") or getattr(trait, "higher_is_better", None) is None


class TestLegacyDataLoading:
    """Test backward compatibility with legacy data missing higher_is_better."""

    def test_llm_trait_legacy_default(self) -> None:
        """Test that legacy LLM traits without higher_is_better default to True."""
        # Simulate loading legacy data without the field
        legacy_data = {
            "name": "clarity",
            "description": "Is the response clear?",
            "kind": "boolean",
            # No higher_is_better field
        }
        trait = LLMRubricTrait(**legacy_data)
        assert trait.higher_is_better is True

    def test_llm_trait_explicit_none_default(self) -> None:
        """Test that explicit None for higher_is_better defaults to True."""
        legacy_data = {
            "name": "clarity",
            "description": "Is the response clear?",
            "kind": "boolean",
            "higher_is_better": None,  # Explicitly None
        }
        trait = LLMRubricTrait(**legacy_data)
        assert trait.higher_is_better is True

    def test_regex_trait_legacy_default(self) -> None:
        """Test that legacy regex traits without higher_is_better default to True."""
        legacy_data = {
            "name": "has_citations",
            "description": "Contains citations",
            "pattern": r"\[\d+\]",
            # No higher_is_better field
        }
        trait = RegexTrait(**legacy_data)
        assert trait.higher_is_better is True

    def test_callable_trait_legacy_default(self) -> None:
        """Test that legacy callable traits without higher_is_better default to True."""

        def evaluate(trace: str) -> bool:
            return len(trace.split()) > 10

        # Create trait without higher_is_better - should default to True
        trait = CallableTrait.from_callable(
            name="word_count",
            func=evaluate,
            kind="boolean",
            description="Word count",
            # No higher_is_better specified
        )
        assert trait.higher_is_better is True


class TestRubricDirectionalities:
    """Test Rubric.get_trait_directionalities() method."""

    def test_empty_rubric(self) -> None:
        """Test get_trait_directionalities on empty rubric."""
        rubric = Rubric(llm_traits=[], regex_traits=[], callable_traits=[])
        directionalities = rubric.get_trait_directionalities()
        assert directionalities == {}

    def test_llm_traits_only(self) -> None:
        """Test get_trait_directionalities with only LLM traits."""
        rubric = Rubric(
            llm_traits=[
                LLMRubricTrait(name="clarity", kind="boolean", higher_is_better=True),
                LLMRubricTrait(name="verbosity", kind="boolean", higher_is_better=False),
            ]
        )
        directionalities = rubric.get_trait_directionalities()
        assert directionalities == {"clarity": True, "verbosity": False}

    def test_regex_traits_only(self) -> None:
        """Test get_trait_directionalities with only regex traits."""
        rubric = Rubric(
            regex_traits=[
                RegexTrait(name="has_citations", pattern=r"\[\d+\]", higher_is_better=True),
                RegexTrait(name="contains_error", pattern=r"\berror\b", higher_is_better=False),
            ]
        )
        directionalities = rubric.get_trait_directionalities()
        assert directionalities == {"has_citations": True, "contains_error": False}

    def test_callable_traits_only(self) -> None:
        """Test get_trait_directionalities with only callable traits."""

        def length_check(text: str) -> bool:
            return len(text) > 100

        rubric = Rubric(
            callable_traits=[
                CallableTrait.from_callable(
                    name="length_check",
                    func=length_check,
                    kind="boolean",
                    higher_is_better=True,
                ),
            ]
        )
        directionalities = rubric.get_trait_directionalities()
        assert directionalities == {"length_check": True}

    def test_mixed_traits(self) -> None:
        """Test get_trait_directionalities with mixed trait types."""

        def word_count(text: str) -> bool:
            return len(text.split()) < 50

        rubric = Rubric(
            llm_traits=[
                LLMRubricTrait(name="clarity", kind="boolean", higher_is_better=True),
            ],
            regex_traits=[
                RegexTrait(name="contains_error", pattern=r"\berror\b", higher_is_better=False),
            ],
            callable_traits=[
                CallableTrait.from_callable(
                    name="word_count",
                    func=word_count,
                    kind="boolean",
                    higher_is_better=False,
                ),
            ],
            metric_traits=[
                MetricRubricTrait(
                    name="extraction",
                    evaluation_mode="tp_only",
                    metrics=["precision"],
                    tp_instructions=["correct"],
                ),
            ],
        )
        directionalities = rubric.get_trait_directionalities()
        # Should include LLM, regex, and callable traits, but NOT metric traits
        assert directionalities == {
            "clarity": True,
            "contains_error": False,
            "word_count": False,
        }
        # Metric traits should not be included
        assert "extraction" not in directionalities


class TestGEPAScoreInversion:
    """Test GEPA scoring inversion for lower-is-better traits."""

    def test_compute_objective_scores_higher_is_better(self) -> None:
        """Test that higher_is_better=True keeps scores unchanged."""
        from unittest.mock import MagicMock

        from karenina.integrations.gepa.config import ObjectiveConfig
        from karenina.integrations.gepa.scoring import compute_objective_scores

        # Create properly structured mock result
        result = MagicMock()
        result.template = None  # No template

        # Mock rubric with get_all_trait_scores method
        result.rubric = MagicMock()
        result.rubric.rubric_evaluation_performed = True
        result.rubric.get_all_trait_scores.return_value = {"clarity": True, "accuracy": 4}

        config = ObjectiveConfig(
            rubric_traits=["clarity", "accuracy"],
            template_passing=False,
            embedding_similarity=False,
        )

        trait_max_scores = {"accuracy": 5}
        trait_directionalities = {"clarity": True, "accuracy": True}

        scores = compute_objective_scores(
            result=result,
            model_name="test_model",
            config=config,
            trait_max_scores=trait_max_scores,
            trait_directionalities=trait_directionalities,
        )

        # clarity=True should map to 1.0 (key is "test_model:clarity")
        assert scores["test_model:clarity"] == 1.0
        # accuracy=4 out of 5 should map to 0.8
        assert scores["test_model:accuracy"] == pytest.approx(0.8)

    def test_compute_objective_scores_lower_is_better_boolean(self) -> None:
        """Test that higher_is_better=False inverts boolean scores."""
        from unittest.mock import MagicMock

        from karenina.integrations.gepa.config import ObjectiveConfig
        from karenina.integrations.gepa.scoring import compute_objective_scores

        # Create properly structured mock result
        result = MagicMock()
        result.template = None

        result.rubric = MagicMock()
        result.rubric.rubric_evaluation_performed = True
        result.rubric.get_all_trait_scores.return_value = {"contains_error": True}  # Error found = bad

        config = ObjectiveConfig(
            rubric_traits=["contains_error"],
            template_passing=False,
            embedding_similarity=False,
        )

        trait_directionalities = {"contains_error": False}  # Lower is better

        scores = compute_objective_scores(
            result=result,
            model_name="test_model",
            config=config,
            trait_directionalities=trait_directionalities,
        )

        # contains_error=True with higher_is_better=False should invert to 0.0
        assert scores["test_model:contains_error"] == 0.0

    def test_compute_objective_scores_lower_is_better_score(self) -> None:
        """Test that higher_is_better=False inverts score-based traits."""
        from unittest.mock import MagicMock

        from karenina.integrations.gepa.config import ObjectiveConfig
        from karenina.integrations.gepa.scoring import compute_objective_scores

        # Create properly structured mock result
        result = MagicMock()
        result.template = None

        result.rubric = MagicMock()
        result.rubric.rubric_evaluation_performed = True
        result.rubric.get_all_trait_scores.return_value = {"error_count": 2}  # 2 out of 5 errors

        config = ObjectiveConfig(
            rubric_traits=["error_count"],
            template_passing=False,
            embedding_similarity=False,
        )

        trait_max_scores = {"error_count": 5}
        trait_directionalities = {"error_count": False}  # Lower error count is better

        scores = compute_objective_scores(
            result=result,
            model_name="test_model",
            config=config,
            trait_max_scores=trait_max_scores,
            trait_directionalities=trait_directionalities,
        )

        # error_count=2 out of 5 normalizes to 0.4
        # With higher_is_better=False, inverts to 1.0 - 0.4 = 0.6
        assert scores["test_model:error_count"] == pytest.approx(0.6)

    def test_compute_objective_scores_no_directionalities(self) -> None:
        """Test that missing directionalities default to higher_is_better=True."""
        from unittest.mock import MagicMock

        from karenina.integrations.gepa.config import ObjectiveConfig
        from karenina.integrations.gepa.scoring import compute_objective_scores

        result = MagicMock()
        result.template = None

        result.rubric = MagicMock()
        result.rubric.rubric_evaluation_performed = True
        result.rubric.get_all_trait_scores.return_value = {"clarity": True}

        config = ObjectiveConfig(
            rubric_traits=["clarity"],
            template_passing=False,
            embedding_similarity=False,
        )

        # No trait_directionalities provided - should default to higher=better
        scores = compute_objective_scores(
            result=result,
            model_name="test_model",
            config=config,
            trait_directionalities=None,
        )

        # Should use default (no inversion)
        assert scores["test_model:clarity"] == 1.0


class TestCheckpointRoundTrip:
    """Test that higher_is_better is preserved through checkpoint serialization."""

    def test_llm_trait_checkpoint_roundtrip(self) -> None:
        """Test LLM trait higher_is_better survives checkpoint round-trip."""
        from karenina.schemas.checkpoint import SchemaOrgRating
        from karenina.utils.checkpoint import convert_rating_to_rubric_trait, convert_rubric_trait_to_rating

        # Create trait with higher_is_better=False
        original_trait = LLMRubricTrait(
            name="verbosity",
            description="Does response have unnecessary words?",
            kind="boolean",
            higher_is_better=False,
        )

        # Convert to rating (checkpoint format)
        rating = convert_rubric_trait_to_rating(original_trait)
        assert isinstance(rating, SchemaOrgRating)

        # Verify higher_is_better is in additionalProperty
        higher_is_better_prop = next(
            (p for p in rating.additionalProperty or [] if p.name == "higher_is_better"),
            None,
        )
        assert higher_is_better_prop is not None
        assert higher_is_better_prop.value is False

        # Convert back to trait
        restored_trait = convert_rating_to_rubric_trait(rating)
        assert isinstance(restored_trait, LLMRubricTrait)
        assert restored_trait.higher_is_better is False

    def test_regex_trait_checkpoint_roundtrip(self) -> None:
        """Test regex trait higher_is_better survives checkpoint round-trip."""
        from karenina.schemas.checkpoint import SchemaOrgRating
        from karenina.utils.checkpoint import convert_rating_to_rubric_trait, convert_rubric_trait_to_rating

        original_trait = RegexTrait(
            name="contains_error",
            description="Contains error keyword",
            pattern=r"\berror\b",
            higher_is_better=False,
        )

        rating = convert_rubric_trait_to_rating(original_trait)
        assert isinstance(rating, SchemaOrgRating)

        # Verify higher_is_better is preserved
        higher_is_better_prop = next(
            (p for p in rating.additionalProperty or [] if p.name == "higher_is_better"),
            None,
        )
        assert higher_is_better_prop is not None
        assert higher_is_better_prop.value is False

        restored_trait = convert_rating_to_rubric_trait(rating)
        assert isinstance(restored_trait, RegexTrait)
        assert restored_trait.higher_is_better is False

    def test_callable_trait_checkpoint_roundtrip(self) -> None:
        """Test callable trait higher_is_better survives checkpoint round-trip."""
        from karenina.schemas.checkpoint import SchemaOrgRating
        from karenina.utils.checkpoint import convert_rating_to_rubric_trait, convert_rubric_trait_to_rating

        def evaluate(trace: str) -> bool:
            return len(trace.split()) > 10

        original_trait = CallableTrait.from_callable(
            name="word_count",
            func=evaluate,
            kind="boolean",
            description="Number of words",
            higher_is_better=False,
        )

        rating = convert_rubric_trait_to_rating(original_trait)
        assert isinstance(rating, SchemaOrgRating)

        higher_is_better_prop = next(
            (p for p in rating.additionalProperty or [] if p.name == "higher_is_better"),
            None,
        )
        assert higher_is_better_prop is not None
        assert higher_is_better_prop.value is False

        restored_trait = convert_rating_to_rubric_trait(rating)
        assert isinstance(restored_trait, CallableTrait)
        assert restored_trait.higher_is_better is False


class TestTraitSerialization:
    """Test trait serialization includes higher_is_better."""

    def test_llm_trait_model_dump(self) -> None:
        """Test LLM trait serialization includes higher_is_better."""
        trait = LLMRubricTrait(
            name="clarity",
            kind="boolean",
            higher_is_better=False,
        )
        data = trait.model_dump()
        assert "higher_is_better" in data
        assert data["higher_is_better"] is False

    def test_regex_trait_model_dump(self) -> None:
        """Test regex trait serialization includes higher_is_better."""
        trait = RegexTrait(
            name="has_error",
            pattern=r"\berror\b",
            higher_is_better=False,
        )
        data = trait.model_dump()
        assert "higher_is_better" in data
        assert data["higher_is_better"] is False

    def test_callable_trait_model_dump(self) -> None:
        """Test callable trait serialization includes higher_is_better."""

        def evaluate(trace: str) -> bool:
            return False

        trait = CallableTrait.from_callable(
            name="word_count",
            func=evaluate,
            kind="boolean",
            higher_is_better=False,
        )
        data = trait.model_dump()
        assert "higher_is_better" in data
        assert data["higher_is_better"] is False

    def test_rubric_model_dump_includes_directionalities(self) -> None:
        """Test rubric serialization includes higher_is_better for all trait types."""
        rubric = Rubric(
            llm_traits=[
                LLMRubricTrait(name="clarity", kind="boolean", higher_is_better=True),
                LLMRubricTrait(name="verbosity", kind="boolean", higher_is_better=False),
            ],
            regex_traits=[
                RegexTrait(name="has_error", pattern=r"\berror\b", higher_is_better=False),
            ],
        )
        data = rubric.model_dump()

        assert data["llm_traits"][0]["higher_is_better"] is True
        assert data["llm_traits"][1]["higher_is_better"] is False
        assert data["regex_traits"][0]["higher_is_better"] is False
