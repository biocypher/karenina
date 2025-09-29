"""Tests for rubric evaluation logic."""

from unittest.mock import Mock, patch

import pytest

from karenina.benchmark.models import INTERFACE_LANGCHAIN, INTERFACE_MANUAL, INTERFACE_OPENROUTER, ModelConfig
from karenina.benchmark.verification.rubric_evaluator import RubricEvaluator
from karenina.schemas.rubric_class import Rubric, RubricTrait


class TestRubricEvaluator:
    """Test RubricEvaluator functionality."""

    @pytest.fixture
    def sample_rubric(self):
        """Create a sample rubric for testing."""
        return Rubric(
            traits=[
                RubricTrait(name="accuracy", description="Is the response factually accurate?", kind="boolean"),
                RubricTrait(
                    name="completeness",
                    description="How complete is the response (1-5)?",
                    kind="score",
                    min_score=1,
                    max_score=5,
                ),
            ]
        )

    @pytest.fixture
    def mock_model_config(self):
        """Create a mock model configuration."""
        return ModelConfig(
            id="test-model",
            model_provider="openai",
            model_name="gpt-4.1-mini",
            temperature=0.1,
            interface="langchain",
            system_prompt="You are a helpful assistant.",
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
            model="gpt-4.1-mini", provider="openai", temperature=0.1, interface="langchain"
        )

    @patch("karenina.benchmark.verification.rubric_evaluator.init_chat_model_unified")
    def test_evaluate_empty_rubric(self, mock_init_model, mock_model_config):
        """Test evaluation with empty rubric."""
        mock_llm = Mock()
        mock_init_model.return_value = mock_llm

        evaluator = RubricEvaluator(mock_model_config)

        empty_rubric = Rubric(traits=[])
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
            "What is the capital of France?", "The capital of France is Paris.", sample_rubric
        )

        assert result["accuracy"] is True
        assert result["completeness"] == 4

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

        result = evaluator.evaluate_rubric("Test question?", "Test answer.", sample_rubric)

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
            traits=[
                RubricTrait(name="bool_trait", description="Boolean trait", kind="boolean"),
                RubricTrait(name="score_trait", description="Score trait", kind="score", min_score=1, max_score=3),
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

        result = evaluator.evaluate_rubric("Test question?", "Test answer.", mixed_rubric)

        assert result["bool_trait"] is False
        assert result["score_trait"] == 2

    @patch("karenina.benchmark.verification.rubric_evaluator.init_chat_model_unified")
    def test_evaluator_with_different_providers(self, mock_init_model):
        """Test evaluator with different model providers."""
        test_configs = [
            ("openai", "gpt-4.1-mini", "langchain"),
            ("google_genai", "gemini-2.0-flash", "langchain"),
            ("anthropic", "claude-3-sonnet", "langchain"),
        ]

        for provider, model, interface in test_configs:
            mock_llm = Mock()
            mock_init_model.return_value = mock_llm

            config = ModelConfig(
                id=f"test-{provider}",
                model_provider=provider,
                model_name=model,
                temperature=0.1,
                interface=interface,
                system_prompt="You are a helpful assistant.",
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
            traits=[
                RubricTrait(name="factual_accuracy", description="Factually correct", kind="boolean"),
                RubricTrait(
                    name="completeness", description="Complete response", kind="score", min_score=1, max_score=5
                ),
                RubricTrait(name="clarity", description="Clear writing", kind="boolean"),
                RubricTrait(
                    name="relevance", description="Relevant to question", kind="score", min_score=1, max_score=3
                ),
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
            comprehensive_rubric,
        )

        # Verify all traits are evaluated
        assert len(result) == 4
        assert result["factual_accuracy"] is True
        assert result["completeness"] == 4
        assert result["clarity"] is True
        assert result["relevance"] == 3

        # Verify LLM was called
        mock_llm.invoke.assert_called()


class TestRubricEvaluatorEdgeCases:
    """Test RubricEvaluator initialization edge cases and validation."""

    def test_evaluator_initialization_no_config(self):
        """Test RubricEvaluator initialization with no config."""
        with pytest.raises(ValueError, match="Model configuration is required"):
            RubricEvaluator(None)

    def test_evaluator_initialization_missing_model_name(self):
        """Test RubricEvaluator initialization with missing model name."""
        config = ModelConfig(
            id="test-model",
            model_provider="openai",
            model_name="",  # Empty model name
            temperature=0.1,
            interface=INTERFACE_LANGCHAIN,
            system_prompt="You are a helpful assistant.",
        )

        with pytest.raises(ValueError, match="Model name is required"):
            RubricEvaluator(config)

    def test_evaluator_initialization_missing_provider_langchain(self):
        """Test RubricEvaluator initialization with missing provider for langchain."""
        config = ModelConfig(
            id="test-model",
            model_provider="",  # Empty provider for langchain interface
            model_name="gpt-4.1-mini",
            temperature=0.1,
            interface=INTERFACE_LANGCHAIN,
            system_prompt="You are a helpful assistant.",
        )

        with pytest.raises(ValueError, match="Model provider is required.*interface: langchain"):
            RubricEvaluator(config)

    def test_evaluator_initialization_openrouter_no_provider(self):
        """Test RubricEvaluator initialization with OpenRouter interface (no provider required)."""
        config = ModelConfig(
            id="test-openrouter",
            model_provider="",  # Empty provider allowed for OpenRouter
            model_name="openrouter/model",
            temperature=0.1,
            interface=INTERFACE_OPENROUTER,
            system_prompt="You are a helpful assistant.",
        )

        # Should not raise errors during validation
        with patch("karenina.benchmark.verification.rubric_evaluator.init_chat_model_unified") as mock_init:
            mock_init.return_value = Mock()
            evaluator = RubricEvaluator(config)
            assert evaluator.model_config == config

    def test_evaluator_initialization_manual_no_provider(self):
        """Test RubricEvaluator initialization with manual interface (no provider required)."""
        config = ModelConfig(
            id="test-manual",
            model_provider="",  # Empty provider allowed for manual
            model_name="manual-model",
            temperature=0.1,
            interface=INTERFACE_MANUAL,
            system_prompt="You are a helpful assistant.",
        )

        # Should not raise errors during validation
        with patch("karenina.benchmark.verification.rubric_evaluator.init_chat_model_unified") as mock_init:
            mock_init.return_value = Mock()
            evaluator = RubricEvaluator(config)
            assert evaluator.model_config == config

    @patch("karenina.benchmark.verification.rubric_evaluator.init_chat_model_unified")
    def test_evaluator_initialization_llm_failure(self, mock_init_model):
        """Test RubricEvaluator initialization with LLM initialization failure."""
        config = ModelConfig(
            id="test-model",
            model_provider="openai",
            model_name="gpt-4.1-mini",
            temperature=0.1,
            interface=INTERFACE_LANGCHAIN,
            system_prompt="You are a helpful assistant.",
        )

        # Mock LLM initialization to fail
        mock_init_model.side_effect = Exception("API initialization failed")

        with pytest.raises(RuntimeError, match="Failed to initialize LLM for rubric evaluation"):
            RubricEvaluator(config)

    @patch("karenina.benchmark.verification.rubric_evaluator.init_chat_model_unified")
    def test_evaluator_initialization_provider_validation_error_message(self, _mock_init_model):
        """Test that provider validation error message includes interface information."""
        config = ModelConfig(
            id="test-model-123",
            model_provider="",  # Empty provider for langchain
            model_name="gpt-4.1-mini",
            temperature=0.1,
            interface=INTERFACE_LANGCHAIN,
            system_prompt="You are a helpful assistant.",
        )

        try:
            RubricEvaluator(config)
            pytest.fail("Should have raised ValueError")
        except ValueError as e:
            error_msg = str(e)
            assert "test-model-123" in error_msg
            assert "interface: langchain" in error_msg
            assert "openrouter" in error_msg
            assert "manual" in error_msg

    @patch("karenina.benchmark.verification.rubric_evaluator.init_chat_model_unified")
    def test_evaluator_handles_different_interface_types(self, mock_init_model):
        """Test that evaluator properly handles different interface types."""
        mock_init_model.return_value = Mock()

        configs = [
            ModelConfig(
                id="langchain-model",
                model_provider="openai",
                model_name="gpt-4.1-mini",
                temperature=0.1,
                interface=INTERFACE_LANGCHAIN,
                system_prompt="You are a helpful assistant.",
            ),
            ModelConfig(
                id="openrouter-model",
                model_provider="",
                model_name="openrouter/model",
                temperature=0.1,
                interface=INTERFACE_OPENROUTER,
                system_prompt="You are a helpful assistant.",
            ),
            ModelConfig(
                id="manual-model",
                model_provider="",
                model_name="manual-model",
                temperature=0.1,
                interface=INTERFACE_MANUAL,
                system_prompt="You are a helpful assistant.",
            ),
        ]

        for config in configs:
            evaluator = RubricEvaluator(config)
            assert evaluator.model_config.interface == config.interface
            assert evaluator.model_config.id == config.id
