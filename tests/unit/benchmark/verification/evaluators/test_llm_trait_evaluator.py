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


# =============================================================================
# Template kind evaluation
# =============================================================================

from pydantic import BaseModel, Field  # noqa: E402


class _CitationCheck(BaseModel):
    has_citations: bool = Field(description="Whether any citations appear")
    citation_count: int = Field(description="Number of citations")
    cited_sources: list[str] = Field(description="Source identifiers cited")


def _make_template_trait(name: str = "citations") -> LLMRubricTrait:
    return LLMRubricTrait(
        name=name,
        description="Assess citation usage.",
        kind=_CitationCheck,
        higher_is_better=None,
    )


@pytest.mark.unit
class TestTemplateEvaluation:
    """Tests for LLMTraitEvaluator.evaluate_template."""

    def _make_evaluator_with_mock_structured(self, parsed_instance):
        """Build an evaluator whose LLM returns ``parsed_instance`` via structured output."""
        from dataclasses import dataclass
        from unittest.mock import MagicMock

        from karenina.benchmark.verification.evaluators.rubric.llm_trait import LLMTraitEvaluator

        @dataclass
        class _FakeUsage:
            input_tokens: int = 10
            output_tokens: int = 5

        @dataclass
        class _FakeResponse:
            raw: object
            usage: object

        structured_llm = MagicMock()
        structured_llm.invoke.return_value = _FakeResponse(raw=parsed_instance, usage=_FakeUsage())

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = structured_llm

        evaluator = LLMTraitEvaluator(
            mock_llm,
            async_enabled=False,
            model_config=_make_model_config(),
        )
        return evaluator, mock_llm, structured_llm

    def test_empty_traits_list_returns_empty(self):
        evaluator = _make_evaluator()
        results, usage = evaluator.evaluate_template("q", "a", [])
        assert results == {}
        assert usage == []

    def test_rejects_non_template_trait(self):
        evaluator = _make_evaluator()
        trait = _make_boolean_trait()
        with pytest.raises(ValueError, match="template-kind"):
            evaluator.evaluate_template("q", "a", [trait])

    def test_template_evaluation_flattens_to_dotted_keys(self):
        parsed = _CitationCheck(
            has_citations=True,
            citation_count=3,
            cited_sources=["Smith et al. 2024", "Jones 2023", "Doe 2022"],
        )
        evaluator, _, _ = self._make_evaluator_with_mock_structured(parsed)
        trait = _make_template_trait()

        results, usage = evaluator.evaluate_template("q", "a", [trait])

        assert set(results.keys()) == {
            "citations.has_citations",
            "citations.citation_count",
            "citations.cited_sources",
        }
        assert results["citations.has_citations"] is True
        assert results["citations.citation_count"] == 3
        assert results["citations.cited_sources"] == [
            "Smith et al. 2024",
            "Jones 2023",
            "Doe 2022",
        ]
        assert len(usage) == 1

    def test_template_evaluation_failure_stores_none(self):
        """If structured_output.invoke raises, the bare trait name maps to None."""
        from unittest.mock import MagicMock

        from karenina.benchmark.verification.evaluators.rubric.llm_trait import LLMTraitEvaluator

        structured_llm = MagicMock()
        structured_llm.invoke.side_effect = RuntimeError("parse failure")
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = structured_llm

        evaluator = LLMTraitEvaluator(
            mock_llm,
            async_enabled=False,
            model_config=_make_model_config(),
        )
        results, usage = evaluator.evaluate_template("q", "a", [_make_template_trait()])

        assert results == {"citations": None}
        assert usage == [{}]

    def test_template_evaluation_calls_with_structured_output_on_kind(self):
        parsed = _CitationCheck(has_citations=False, citation_count=0, cited_sources=[])
        evaluator, mock_llm, _ = self._make_evaluator_with_mock_structured(parsed)

        evaluator.evaluate_template("q", "a", [_make_template_trait()])

        mock_llm.with_structured_output.assert_called_once_with(_CitationCheck)
