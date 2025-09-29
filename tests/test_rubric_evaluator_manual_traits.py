"""Tests for RubricEvaluator with ManualRubricTrait support."""

from unittest.mock import Mock, patch

import pytest

from karenina.benchmark.models import ModelConfig
from karenina.benchmark.verification.rubric_evaluator import RubricEvaluator
from karenina.schemas.rubric_class import ManualRubricTrait, Rubric, RubricTrait


class TestRubricEvaluatorManualTraits:
    """Test RubricEvaluator with manual trait functionality."""

    @pytest.fixture
    def mock_model_config(self) -> None:
        """Create a mock model configuration."""
        return ModelConfig(
            id="test-model",
            model_provider="openai",
            model_name="gpt-4.1-mini",
            temperature=0.1,
            interface="langchain",
            system_prompt="You are a helpful assistant.",
        )

    @pytest.fixture
    def sample_callable_registry(self) -> None:
        """Create sample callable functions for testing."""

        def is_numeric(text: str) -> bool:
            return text.strip().isdigit()

        def contains_error(text: str) -> bool:
            return "error" in text.lower()

        def word_count_over_10(text: str) -> bool:
            return len(text.split()) > 10

        return {
            "is_numeric": is_numeric,
            "contains_error": contains_error,
            "word_count_over_10": word_count_over_10,
        }

    @patch("karenina.benchmark.verification.rubric_evaluator.init_chat_model_unified")
    def test_evaluator_with_callable_registry(
        self, mock_init_model, mock_model_config, sample_callable_registry
    ) -> None:
        """Test RubricEvaluator initialization with callable registry."""
        mock_llm = Mock()
        mock_init_model.return_value = mock_llm

        evaluator = RubricEvaluator(mock_model_config, sample_callable_registry)

        assert evaluator.callable_registry == sample_callable_registry
        assert "is_numeric" in evaluator.callable_registry
        assert "contains_error" in evaluator.callable_registry

    @patch("karenina.benchmark.verification.rubric_evaluator.init_chat_model_unified")
    def test_register_callable(self, mock_init_model, mock_model_config) -> None:
        """Test registering callable functions."""
        mock_llm = Mock()
        mock_init_model.return_value = mock_llm

        evaluator = RubricEvaluator(mock_model_config)

        def test_func(text: str) -> bool:
            return "test" in text

        evaluator.register_callable("test_func", test_func)

        assert "test_func" in evaluator.callable_registry
        assert evaluator.callable_registry["test_func"] == test_func

    @patch("karenina.benchmark.verification.rubric_evaluator.init_chat_model_unified")
    def test_register_callable_invalid_signature(self, mock_init_model, mock_model_config) -> None:
        """Test error when registering callable with wrong signature."""
        mock_llm = Mock()
        mock_init_model.return_value = mock_llm

        evaluator = RubricEvaluator(mock_model_config)

        def bad_func(text: str, _extra: str) -> bool:
            return True

        with pytest.raises(ValueError, match="must have exactly one parameter"):
            evaluator.register_callable("bad_func", bad_func)

    @patch("karenina.benchmark.verification.rubric_evaluator.init_chat_model_unified")
    def test_evaluate_manual_traits_only(self, mock_init_model, mock_model_config, sample_callable_registry) -> None:
        """Test evaluation with only manual traits."""
        mock_llm = Mock()
        mock_init_model.return_value = mock_llm

        evaluator = RubricEvaluator(mock_model_config, sample_callable_registry)

        manual_traits = [
            ManualRubricTrait(name="has_number", pattern=r"\d+"),
            ManualRubricTrait(name="is_numeric", callable_name="is_numeric"),
            ManualRubricTrait(name="no_error", callable_name="contains_error", invert_result=True),
        ]

        rubric = Rubric(manual_traits=manual_traits)

        # Test with text that has numbers and is not purely numeric
        result = evaluator.evaluate_rubric("Test question", "The answer is 42 words", rubric)

        assert result["has_number"] is True  # Contains number
        assert result["is_numeric"] is False  # Not purely numeric
        assert result["no_error"] is True  # No error (inverted)

    @patch("karenina.benchmark.verification.rubric_evaluator.init_chat_model_unified")
    def test_evaluate_mixed_traits(self, mock_init_model, mock_model_config, sample_callable_registry) -> None:
        """Test evaluation with both LLM and manual traits."""
        mock_llm = Mock()
        mock_init_model.return_value = mock_llm

        # Mock LLM response for LLM traits
        mock_llm.invoke.return_value.content = '{"accuracy": true, "quality": 4}'

        evaluator = RubricEvaluator(mock_model_config, sample_callable_registry)

        llm_traits = [
            RubricTrait(name="accuracy", description="Is accurate?", kind="boolean"),
            RubricTrait(name="quality", description="Quality score", kind="score", min_score=1, max_score=5),
        ]

        manual_traits = [
            ManualRubricTrait(name="has_number", pattern=r"\d+"),
            ManualRubricTrait(name="long_response", callable_name="word_count_over_10"),
        ]

        rubric = Rubric(traits=llm_traits, manual_traits=manual_traits)

        result = evaluator.evaluate_rubric(
            "What is 2+2?", "The answer is 4 because addition is a mathematical operation.", rubric
        )

        # Manual traits evaluated first
        assert result["has_number"] is True
        assert result["long_response"] is False  # Less than 10 words

        # LLM traits evaluated second
        assert result["accuracy"] is True
        assert result["quality"] == 4

    @patch("karenina.benchmark.verification.rubric_evaluator.init_chat_model_unified")
    def test_evaluate_manual_trait_error_handling(self, mock_init_model, mock_model_config) -> None:
        """Test error handling in manual trait evaluation."""
        mock_llm = Mock()
        mock_init_model.return_value = mock_llm

        evaluator = RubricEvaluator(mock_model_config)

        # Manual trait with missing callable
        manual_traits = [
            ManualRubricTrait(name="missing_func", callable_name="nonexistent"),
        ]

        rubric = Rubric(manual_traits=manual_traits)

        result = evaluator.evaluate_rubric("Test", "Test response", rubric)

        # Should handle error gracefully and mark as None
        assert result["missing_func"] is None

    @patch("karenina.benchmark.verification.rubric_evaluator.init_chat_model_unified")
    def test_evaluate_regex_patterns(self, mock_init_model, mock_model_config) -> None:
        """Test various regex pattern evaluations."""
        mock_llm = Mock()
        mock_init_model.return_value = mock_llm

        evaluator = RubricEvaluator(mock_model_config)

        manual_traits = [
            ManualRubricTrait(name="has_email", pattern=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            ManualRubricTrait(name="has_phone", pattern=r"\d{3}-\d{3}-\d{4}"),
            ManualRubricTrait(name="starts_with_hello", pattern=r"^[Hh]ello"),
            ManualRubricTrait(name="contains_yes", pattern=r"\byes\b", case_sensitive=False),
        ]

        rubric = Rubric(manual_traits=manual_traits)

        test_text = "Hello! YES, my email is test@example.com and phone is 555-123-4567."

        result = evaluator.evaluate_rubric("Test", test_text, rubric)

        assert result["has_email"] is True
        assert result["has_phone"] is True
        assert result["starts_with_hello"] is True
        assert result["contains_yes"] is True

    @patch("karenina.benchmark.verification.rubric_evaluator.init_chat_model_unified")
    def test_case_sensitivity_in_patterns(self, mock_init_model, mock_model_config) -> None:
        """Test case sensitivity handling in regex patterns."""
        mock_llm = Mock()
        mock_init_model.return_value = mock_llm

        evaluator = RubricEvaluator(mock_model_config)

        manual_traits = [
            ManualRubricTrait(name="case_sensitive", pattern=r"Error", case_sensitive=True),
            ManualRubricTrait(name="case_insensitive", pattern=r"Error", case_sensitive=False),
        ]

        rubric = Rubric(manual_traits=manual_traits)

        # Test with lowercase "error"
        result1 = evaluator.evaluate_rubric("Test", "There was an error in processing.", rubric)
        assert result1["case_sensitive"] is False  # Should not match (case sensitive)
        assert result1["case_insensitive"] is True  # Should match (case insensitive)

        # Test with exact case "Error"
        result2 = evaluator.evaluate_rubric("Test", "There was an Error in processing.", rubric)
        assert result2["case_sensitive"] is True  # Should match (exact case)
        assert result2["case_insensitive"] is True  # Should match (case insensitive)

    @patch("karenina.benchmark.verification.rubric_evaluator.init_chat_model_unified")
    def test_invert_result_functionality(self, mock_init_model, mock_model_config) -> None:
        """Test result inversion functionality."""
        mock_llm = Mock()
        mock_init_model.return_value = mock_llm

        evaluator = RubricEvaluator(mock_model_config)

        manual_traits = [
            ManualRubricTrait(name="has_error", pattern=r"error", case_sensitive=False),
            ManualRubricTrait(name="no_error", pattern=r"error", case_sensitive=False, invert_result=True),
        ]

        rubric = Rubric(manual_traits=manual_traits)

        # Test with error
        result1 = evaluator.evaluate_rubric("Test", "There was an error", rubric)
        assert result1["has_error"] is True
        assert result1["no_error"] is False  # Inverted

        # Test without error
        result2 = evaluator.evaluate_rubric("Test", "Everything is fine", rubric)
        assert result2["has_error"] is False
        assert result2["no_error"] is True  # Inverted

    @patch("karenina.benchmark.verification.rubric_evaluator.init_chat_model_unified")
    def test_empty_rubric_handling(self, mock_init_model, mock_model_config) -> None:
        """Test evaluation with empty rubric."""
        mock_llm = Mock()
        mock_init_model.return_value = mock_llm

        evaluator = RubricEvaluator(mock_model_config)

        # Empty rubric (no traits at all)
        empty_rubric = Rubric()
        result = evaluator.evaluate_rubric("Test", "Test response", empty_rubric)
        assert result == {}

        # Rubric with empty manual traits list
        empty_manual_rubric = Rubric(manual_traits=[])
        result = evaluator.evaluate_rubric("Test", "Test response", empty_manual_rubric)
        assert result == {}
