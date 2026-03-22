"""Unit tests for LLMTraitEvaluator surviving methods.

Tests cover _validate_score, _validate_batch_scores, and
_validate_literal_classification, which remain after removing
the dead fallback parser methods.
"""

import pytest

from karenina.schemas.config.models import ModelConfig
from karenina.schemas.entities.rubric import LLMRubricTrait


def _make_model_config() -> ModelConfig:
    """Create a test ModelConfig."""
    return ModelConfig(
        id="test-model",
        model_name="test-model",
        model_provider="anthropic",
        interface="claude_agent_sdk",
    )


def _make_evaluator():
    """Create an LLMTraitEvaluator with a mock LLM."""
    from unittest.mock import MagicMock

    from karenina.benchmark.verification.evaluators.rubric.llm_trait import LLMTraitEvaluator

    mock_llm = MagicMock()
    return LLMTraitEvaluator(mock_llm, model_config=_make_model_config())


def _make_boolean_trait(name: str = "clarity") -> LLMRubricTrait:
    """Create a boolean LLM rubric trait for testing."""
    return LLMRubricTrait(
        name=name,
        description="Is the response clear?",
        kind="boolean",
        higher_is_better=True,
    )


def _make_score_trait(name: str = "quality", min_score: int = 1, max_score: int = 5) -> LLMRubricTrait:
    """Create a score LLM rubric trait for testing."""
    return LLMRubricTrait(
        name=name,
        description="Rate the quality.",
        kind="score",
        min_score=min_score,
        max_score=max_score,
        higher_is_better=True,
    )


def _make_literal_trait(name: str = "sentiment") -> LLMRubricTrait:
    """Create a literal LLM rubric trait for testing."""
    return LLMRubricTrait(
        name=name,
        description="Classify the sentiment.",
        kind="literal",
        classes={"positive": "Favorable tone", "neutral": "No strong tone", "negative": "Unfavorable tone"},
        higher_is_better=False,
    )


@pytest.mark.unit
class TestValidateScore:
    """Tests for LLMTraitEvaluator._validate_score."""

    def test_boolean_true(self):
        evaluator = _make_evaluator()
        trait = _make_boolean_trait()
        assert evaluator._validate_score(True, trait) is True

    def test_boolean_false(self):
        evaluator = _make_evaluator()
        trait = _make_boolean_trait()
        assert evaluator._validate_score(False, trait) is False

    def test_boolean_from_int_one(self):
        evaluator = _make_evaluator()
        trait = _make_boolean_trait()
        assert evaluator._validate_score(1, trait) is True

    def test_boolean_from_string_false(self):
        evaluator = _make_evaluator()
        trait = _make_boolean_trait()
        assert evaluator._validate_score("false", trait) is False

    def test_boolean_from_string_zero(self):
        evaluator = _make_evaluator()
        trait = _make_boolean_trait()
        assert evaluator._validate_score("0", trait) is False

    def test_score_in_range(self):
        evaluator = _make_evaluator()
        trait = _make_score_trait(min_score=1, max_score=5)
        assert evaluator._validate_score(3, trait) == 3

    def test_score_clamped_above_max(self):
        evaluator = _make_evaluator()
        trait = _make_score_trait(min_score=1, max_score=5)
        assert evaluator._validate_score(10, trait) == 5

    def test_score_clamped_below_min(self):
        evaluator = _make_evaluator()
        trait = _make_score_trait(min_score=1, max_score=5)
        assert evaluator._validate_score(-1, trait) == 1

    def test_score_from_string(self):
        evaluator = _make_evaluator()
        trait = _make_score_trait(min_score=1, max_score=5)
        assert evaluator._validate_score("4", trait) == 4

    def test_score_invalid_type_raises(self):
        evaluator = _make_evaluator()
        trait = _make_score_trait()
        with pytest.raises(ValueError, match="Invalid score type"):
            evaluator._validate_score("not_a_number", trait)


@pytest.mark.unit
class TestValidateBatchScores:
    """Tests for LLMTraitEvaluator._validate_batch_scores."""

    def test_all_traits_present(self):
        evaluator = _make_evaluator()
        trait_a = _make_boolean_trait("clarity")
        trait_b = _make_score_trait("quality")
        scores = {"clarity": True, "quality": 4}
        result = evaluator._validate_batch_scores(scores, [trait_a, trait_b])
        assert result == {"clarity": True, "quality": 4}

    def test_missing_trait_becomes_none(self):
        evaluator = _make_evaluator()
        trait_a = _make_boolean_trait("clarity")
        trait_b = _make_score_trait("quality")
        scores = {"clarity": True}
        result = evaluator._validate_batch_scores(scores, [trait_a, trait_b])
        assert result["clarity"] is True
        assert result["quality"] is None

    def test_extra_keys_ignored(self):
        evaluator = _make_evaluator()
        trait = _make_boolean_trait("clarity")
        scores = {"clarity": False, "unknown_trait": 99}
        result = evaluator._validate_batch_scores(scores, [trait])
        assert "unknown_trait" not in result
        assert result["clarity"] is False


@pytest.mark.unit
class TestValidateLiteralClassification:
    """Tests for LLMTraitEvaluator._validate_literal_classification."""

    def test_valid_class_returns_index_and_label(self):
        evaluator = _make_evaluator()
        trait = _make_literal_trait()
        score, label = evaluator._validate_literal_classification(trait, "positive")
        assert score == 0
        assert label == "positive"

    def test_case_insensitive_matching(self):
        evaluator = _make_evaluator()
        trait = _make_literal_trait()
        score, label = evaluator._validate_literal_classification(trait, "NEUTRAL")
        assert score == 1
        assert label == "neutral"

    def test_invalid_class_returns_negative_one(self):
        evaluator = _make_evaluator()
        trait = _make_literal_trait()
        score, label = evaluator._validate_literal_classification(trait, "unknown_class")
        assert score == -1
        assert label == "unknown_class"

    def test_non_literal_trait_returns_error(self):
        evaluator = _make_evaluator()
        trait = _make_boolean_trait()
        score, label = evaluator._validate_literal_classification(trait, "anything")
        assert score == -1
        assert "NOT_LITERAL_TRAIT" in label
