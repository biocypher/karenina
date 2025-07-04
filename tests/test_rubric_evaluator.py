"""Tests for rubric evaluation logic."""

import pytest
from unittest.mock import Mock, patch

from karenina.benchmark.verification.rubric_evaluator import RubricEvaluator
from karenina.benchmark.models import ModelConfiguration
from karenina.schemas.rubric_class import Rubric, RubricTrait


class TestRubricEvaluator:
    """Test RubricEvaluator functionality."""

    @pytest.fixture
    def sample_rubric(self):
        """Create a sample rubric for testing."""
        return Rubric(
            title="Test Rubric",
            traits=[
                RubricTrait(
                    name="accuracy",
                    description="Is the response factually accurate?",
                    kind="boolean"
                ),
                RubricTrait(
                    name="completeness",
                    description="How complete is the response (1-5)?",
                    kind="score",
                    min_score=1,
                    max_score=5
                )
            ]
        )

    @pytest.fixture
    def mock_model_config(self):
        """Create a mock model configuration."""
        return ModelConfiguration(
            id="test-model",
            model_provider="openai",
            model_name="gpt-3.5-turbo",
            temperature=0.1,
            interface="langchain"
        )

    @patch("karenina.benchmark.verification.rubric_evaluator.init_chat_model_unified")
    def test_evaluator_initialization(self, mock_init_model, mock_model_config):
        """Test RubricEvaluator initialization."""
        mock_llm = Mock()
        mock_init_model.return_value = mock_llm
        
        evaluator = RubricEvaluator(mock_model_config)
        
        assert evaluator.model_config == mock_model_config
        assert evaluator.llm == mock_llm
        mock_init_model.assert_called_once_with(
            model="gpt-3.5-turbo",
            provider="openai",
            temperature=0.1,
            interface="langchain"
        )

    @patch("karenina.benchmark.verification.rubric_evaluator.init_chat_model_unified")
    def test_evaluate_empty_rubric(self, mock_init_model, mock_model_config):
        """Test evaluation with empty rubric."""
        mock_llm = Mock()
        mock_init_model.return_value = mock_llm
        
        evaluator = RubricEvaluator(mock_model_config)
        
        empty_rubric = Rubric(title="Empty", traits=[])
        result = evaluator.evaluate_rubric("Test question?", "Test answer.", empty_rubric)
        
        assert result == {}

    @patch("karenina.benchmark.verification.rubric_evaluator.init_chat_model_unified")
    def test_evaluate_rubric_success(self, mock_init_model, mock_model_config, sample_rubric):
        """Test successful rubric evaluation."""
        mock_llm = Mock()
        mock_init_model.return_value = mock_llm
        
        # Mock LLM response with valid JSON
        mock_llm.invoke.return_value.content = """
        {
            "accuracy": true,
            "completeness": 4
        }
        """
        
        evaluator = RubricEvaluator(mock_model_config)
        
        result = evaluator.evaluate_rubric(
            "What is the capital of France?",
            "The capital of France is Paris.",
            sample_rubric
        )
        
        assert result["accuracy"] is True
        assert result["completeness"] == 4

    @patch("karenina.benchmark.verification.rubric_evaluator.init_chat_model_unified")
    def test_evaluate_rubric_llm_failure(self, mock_init_model, mock_model_config, sample_rubric):
        """Test rubric evaluation when LLM fails."""
        mock_llm = Mock()
        mock_init_model.return_value = mock_llm
        
        # Mock LLM to raise exception
        mock_llm.invoke.side_effect = Exception("API error")
        
        evaluator = RubricEvaluator(mock_model_config)
        
        result = evaluator.evaluate_rubric(
            "Test question?",
            "Test answer.",
            sample_rubric
        )
        
        # Should return empty dict on complete failure
        assert result == {}

    @patch("karenina.benchmark.verification.rubric_evaluator.init_chat_model_unified")
    def test_evaluate_rubric_invalid_json(self, mock_init_model, mock_model_config, sample_rubric):
        """Test evaluation with invalid JSON response."""
        mock_llm = Mock()
        mock_init_model.return_value = mock_llm
        
        # Mock LLM response with invalid JSON
        mock_llm.invoke.return_value.content = "This is not valid JSON"
        
        evaluator = RubricEvaluator(mock_model_config)
        
        result = evaluator.evaluate_rubric(
            "Test question?",
            "Test answer.",
            sample_rubric
        )
        
        # Should handle gracefully and return empty dict
        assert result == {}

    @patch("karenina.benchmark.verification.rubric_evaluator.init_chat_model_unified")
    def test_evaluate_rubric_partial_response(self, mock_init_model, mock_model_config, sample_rubric):
        """Test evaluation with partial response from LLM."""
        mock_llm = Mock()
        mock_init_model.return_value = mock_llm
        
        # Mock LLM response with only some traits
        mock_llm.invoke.return_value.content = """
        {
            "accuracy": true
        }
        """
        
        evaluator = RubricEvaluator(mock_model_config)
        
        result = evaluator.evaluate_rubric(
            "Test question?",
            "Test answer.",
            sample_rubric
        )
        
        assert result["accuracy"] is True
        # Should only include traits that were returned
        assert "completeness" not in result or result["completeness"] is None

    @patch("karenina.benchmark.verification.rubric_evaluator.init_chat_model_unified")
    def test_evaluate_rubric_mixed_types(self, mock_init_model, mock_model_config):
        """Test evaluation with mixed boolean and score traits."""
        mock_llm = Mock()
        mock_init_model.return_value = mock_llm
        
        # Create rubric with mixed trait types
        mixed_rubric = Rubric(
            title="Mixed Types",
            traits=[
                RubricTrait(name="bool_trait", description="Boolean trait", kind="boolean"),
                RubricTrait(name="score_trait", description="Score trait", kind="score", min_score=1, max_score=3)
            ]
        )
        
        # Mock comprehensive response
        mock_llm.invoke.return_value.content = """
        {
            "bool_trait": false,
            "score_trait": 2
        }
        """
        
        evaluator = RubricEvaluator(mock_model_config)
        
        result = evaluator.evaluate_rubric(
            "Test question?",
            "Test answer.",
            mixed_rubric
        )
        
        assert result["bool_trait"] is False
        assert result["score_trait"] == 2

    @patch("karenina.benchmark.verification.rubric_evaluator.init_chat_model_unified")
    def test_evaluator_with_different_providers(self, mock_init_model):
        """Test evaluator with different model providers."""
        test_configs = [
            ("openai", "gpt-4", "langchain"),
            ("google_genai", "gemini-2.0-flash", "langchain"),
            ("anthropic", "claude-3-sonnet", "langchain")
        ]
        
        for provider, model, interface in test_configs:
            mock_llm = Mock()
            mock_init_model.return_value = mock_llm
            
            config = ModelConfiguration(
                id=f"test-{provider}",
                model_provider=provider,
                model_name=model,
                temperature=0.1,
                interface=interface
            )
            
            evaluator = RubricEvaluator(config)
            
            assert evaluator.model_config.model_provider == provider
            assert evaluator.model_config.model_name == model
            assert evaluator.llm == mock_llm

    @patch("karenina.benchmark.verification.rubric_evaluator.init_chat_model_unified")
    def test_evaluate_rubric_integration(self, mock_init_model, mock_model_config):
        """Integration test with comprehensive rubric evaluation."""
        mock_llm = Mock()
        mock_init_model.return_value = mock_llm
        
        # Create comprehensive rubric
        comprehensive_rubric = Rubric(
            title="Comprehensive Assessment",
            traits=[
                RubricTrait(name="factual_accuracy", description="Factually correct", kind="boolean"),
                RubricTrait(name="completeness", description="Complete response", kind="score", min_score=1, max_score=5),
                RubricTrait(name="clarity", description="Clear writing", kind="boolean"),
                RubricTrait(name="relevance", description="Relevant to question", kind="score", min_score=1, max_score=3)
            ]
        )
        
        # Mock comprehensive response
        mock_llm.invoke.return_value.content = """
        {
            "factual_accuracy": true,
            "completeness": 4,
            "clarity": true,
            "relevance": 3
        }
        """
        
        evaluator = RubricEvaluator(mock_model_config)
        
        result = evaluator.evaluate_rubric(
            "Explain the process of photosynthesis.",
            "Photosynthesis is the process by which plants convert light energy into chemical energy...",
            comprehensive_rubric
        )
        
        # Verify all traits are evaluated
        assert len(result) == 4
        assert result["factual_accuracy"] is True
        assert result["completeness"] == 4
        assert result["clarity"] is True
        assert result["relevance"] == 3
        
        # Verify LLM was called
        mock_llm.invoke.assert_called()