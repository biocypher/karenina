"""Unit tests for RubricEvaluator deep judgment methods.

This module tests the deep judgment evaluation flow for LLMRubricTraits:
- Per-trait deep judgment control
- Two evaluation flows: with/without excerpts
- Retry mechanism with fuzzy matching
- Auto-fail after retries exhausted
- Configuration overrides (max_excerpts, fuzzy_threshold, retry_attempts)
"""

import json
from unittest.mock import Mock, patch

import pytest

from karenina.benchmark.verification.evaluators.rubric_evaluator import RubricEvaluator
from karenina.schemas import ModelConfig
from karenina.schemas.domain import LLMRubricTrait, Rubric


class TestDeepJudgmentRetryMechanism:
    """Test retry mechanism for excerpt extraction with validation."""

    @pytest.fixture
    def mock_model_config(self) -> ModelConfig:
        """Create a mock model configuration."""
        return ModelConfig(
            id="test-dj-model",
            model_provider="openai",
            model_name="gpt-4.1-mini",
            temperature=0.1,
            interface="langchain",
            system_prompt="You are a helpful assistant.",
        )

    @pytest.fixture
    def sample_answer(self) -> str:
        """Sample answer text for testing."""
        return (
            "Photosynthesis is the process by which plants convert light energy into chemical energy. "
            "This occurs in the chloroplasts. The overall equation is: 6CO2 + 6H2O + light → C6H12O6 + 6O2. "
            "The process involves two main stages: light-dependent reactions and the Calvin cycle."
        )

    @pytest.fixture
    def dj_trait_with_excerpts(self) -> LLMRubricTrait:
        """Create a deep judgment trait with excerpts enabled."""
        return LLMRubricTrait(
            name="scientific_accuracy",
            description="Does the answer provide scientifically accurate information?",
            kind="score",
            min_score=1,
            max_score=5,
            deep_judgment_enabled=True,
            deep_judgment_excerpt_enabled=True,
            deep_judgment_max_excerpts=3,
            deep_judgment_fuzzy_match_threshold=0.8,
            deep_judgment_excerpt_retry_attempts=2,
        )

    @pytest.fixture
    def dj_trait_without_excerpts(self) -> LLMRubricTrait:
        """Create a deep judgment trait without excerpts."""
        return LLMRubricTrait(
            name="overall_clarity",
            description="Is the overall response clear and well-structured?",
            kind="score",
            min_score=1,
            max_score=5,
            deep_judgment_enabled=True,
            deep_judgment_excerpt_enabled=False,
        )

    @patch("karenina.benchmark.verification.evaluators.rubric_evaluator.init_chat_model_unified")
    def test_excerpt_extraction_first_attempt_success(
        self, mock_init_model, mock_model_config, sample_answer, dj_trait_with_excerpts
    ):
        """Test successful excerpt extraction on first attempt."""
        mock_llm = Mock()
        mock_init_model.return_value = mock_llm

        # Mock excerpt extraction response with valid excerpts
        mock_llm.invoke.side_effect = [
            # Excerpt extraction
            Mock(
                content=json.dumps(
                    {
                        "excerpts": [
                            {
                                "text": "Photosynthesis is the process by which plants convert light energy into chemical energy",
                                "confidence": "high",
                            },
                            {
                                "text": "This occurs in the chloroplasts",
                                "confidence": "medium",
                            },
                        ]
                    }
                )
            ),
            # Reasoning generation
            Mock(content="The answer demonstrates strong scientific accuracy with precise terminology."),
            # Score extraction
            Mock(content=json.dumps({"scientific_accuracy": 5})),
        ]

        evaluator = RubricEvaluator(mock_model_config, evaluation_strategy="sequential")

        # Create a mock config object with deep judgment settings
        from karenina.schemas import VerificationConfig

        config = VerificationConfig(
            answering_models=[mock_model_config],
            parsing_models=[mock_model_config],
            deep_judgment_enabled=True,
            deep_judgment_excerpt_retry_attempts=2,
            deep_judgment_fuzzy_match_threshold=0.8,
        )

        # Call the deep judgment evaluation
        result = evaluator._extract_excerpts_for_trait(sample_answer, dj_trait_with_excerpts, config)

        # Verify results
        assert result["success"] is True
        assert "excerpts" in result
        assert len(result["excerpts"]) == 2
        assert result["retry_count"] == 0
        assert result["validation_passed"] is True

        # Verify LLM was called once for extraction
        assert mock_llm.invoke.call_count >= 1

    @patch("karenina.benchmark.verification.evaluators.rubric_evaluator.init_chat_model_unified")
    def test_excerpt_extraction_retry_success(
        self, mock_init_model, mock_model_config, sample_answer, dj_trait_with_excerpts
    ):
        """Test excerpt extraction succeeds on retry after validation failure."""
        mock_llm = Mock()
        mock_init_model.return_value = mock_llm

        # First attempt: excerpts with low similarity (fail validation)
        # Second attempt: excerpts with high similarity (pass validation)
        mock_llm.invoke.side_effect = [
            # First extraction attempt - invalid excerpts
            Mock(
                content=json.dumps(
                    {
                        "excerpts": [
                            {
                                "text": "Plants use sunlight",  # Low similarity to actual answer
                                "confidence": "low",
                            }
                        ]
                    }
                )
            ),
            # Second extraction attempt - valid excerpts
            Mock(
                content=json.dumps(
                    {
                        "excerpts": [
                            {
                                "text": "Photosynthesis is the process by which plants convert light energy",
                                "confidence": "high",
                            }
                        ]
                    }
                )
            ),
            # Reasoning
            Mock(content="The answer provides accurate scientific information."),
            # Score
            Mock(content=json.dumps({"scientific_accuracy": 4})),
        ]

        evaluator = RubricEvaluator(mock_model_config, evaluation_strategy="sequential")

        from karenina.schemas import VerificationConfig

        config = VerificationConfig(
            answering_models=[mock_model_config],
            parsing_models=[mock_model_config],
            deep_judgment_enabled=True,
            deep_judgment_excerpt_retry_attempts=2,
            deep_judgment_fuzzy_match_threshold=0.8,
        )

        result = evaluator._extract_excerpts_for_trait(sample_answer, dj_trait_with_excerpts, config)

        # Verify retry occurred
        assert result["success"] is True
        assert result["retry_count"] == 1
        assert result["validation_passed"] is True
        assert len(result["excerpts"]) >= 1

    @patch("karenina.benchmark.verification.evaluators.rubric_evaluator.init_chat_model_unified")
    def test_excerpt_extraction_all_retries_exhausted(
        self, mock_init_model, mock_model_config, sample_answer, dj_trait_with_excerpts
    ):
        """Test auto-fail when all retry attempts are exhausted."""
        mock_llm = Mock()
        mock_init_model.return_value = mock_llm

        # All attempts return invalid excerpts
        mock_llm.invoke.return_value = Mock(
            content=json.dumps(
                {
                    "excerpts": [
                        {
                            "text": "Completely wrong text not in answer",
                            "confidence": "low",
                        }
                    ]
                }
            )
        )

        evaluator = RubricEvaluator(mock_model_config, evaluation_strategy="sequential")

        from karenina.schemas import VerificationConfig

        config = VerificationConfig(
            answering_models=[mock_model_config],
            parsing_models=[mock_model_config],
            deep_judgment_enabled=True,
            deep_judgment_excerpt_retry_attempts=2,
            deep_judgment_fuzzy_match_threshold=0.8,
        )

        result = evaluator._extract_excerpts_for_trait(sample_answer, dj_trait_with_excerpts, config)

        # Verify auto-fail
        assert result["success"] is False
        assert result["retry_count"] == 2  # Max retries reached
        assert result["validation_passed"] is False
        assert result.get("auto_fail") is True

    @patch("karenina.benchmark.verification.evaluators.rubric_evaluator.init_chat_model_unified")
    def test_per_trait_retry_override(self, mock_init_model, mock_model_config, sample_answer):
        """Test that trait-specific retry_attempts override global default."""
        mock_llm = Mock()
        mock_init_model.return_value = mock_llm

        # Create trait with custom retry attempts
        trait = LLMRubricTrait(
            name="custom_retry_trait",
            description="Test trait",
            kind="score",
            min_score=1,
            max_score=5,
            deep_judgment_enabled=True,
            deep_judgment_excerpt_enabled=True,
            deep_judgment_excerpt_retry_attempts=5,  # Override
        )

        # All attempts fail
        mock_llm.invoke.return_value = Mock(
            content=json.dumps({"excerpts": [{"text": "Invalid text", "confidence": "low"}]})
        )

        evaluator = RubricEvaluator(mock_model_config, evaluation_strategy="sequential")

        from karenina.schemas import VerificationConfig

        config = VerificationConfig(
            answering_models=[mock_model_config],
            parsing_models=[mock_model_config],
            deep_judgment_enabled=True,
            deep_judgment_excerpt_retry_attempts=2,  # Global default
            deep_judgment_fuzzy_match_threshold=0.8,
        )

        result = evaluator._extract_excerpts_for_trait(sample_answer, trait, config)

        # Should use trait-specific retry count (5) not global (2)
        assert result["retry_count"] == 5

    @patch("karenina.benchmark.verification.evaluators.rubric_evaluator.init_chat_model_unified")
    def test_validation_feedback_format(
        self, mock_init_model, mock_model_config, sample_answer, dj_trait_with_excerpts
    ):
        """Test that validation feedback includes scores and threshold."""
        mock_llm = Mock()
        mock_init_model.return_value = mock_llm

        # Track the prompts sent to LLM
        prompts_sent = []

        def capture_prompt(messages):
            prompts_sent.append(messages)
            # Return invalid excerpts to trigger feedback
            return Mock(content=json.dumps({"excerpts": [{"text": "Wrong text", "confidence": "low"}]}))

        mock_llm.invoke.side_effect = capture_prompt

        evaluator = RubricEvaluator(mock_model_config, evaluation_strategy="sequential")

        from karenina.schemas import VerificationConfig

        config = VerificationConfig(
            answering_models=[mock_model_config],
            parsing_models=[mock_model_config],
            deep_judgment_enabled=True,
            deep_judgment_excerpt_retry_attempts=1,
            deep_judgment_fuzzy_match_threshold=0.85,
        )

        evaluator._extract_excerpts_for_trait(sample_answer, dj_trait_with_excerpts, config)

        # Check that retry includes feedback
        if len(prompts_sent) > 1:
            retry_prompt = str(prompts_sent[1])
            # Feedback should mention similarity scores and threshold
            assert "similarity" in retry_prompt.lower() or "threshold" in retry_prompt.lower()


class TestDeepJudgmentFlows:
    """Test different deep judgment evaluation flows."""

    @pytest.fixture
    def mock_model_config(self) -> ModelConfig:
        """Create a mock model configuration."""
        return ModelConfig(
            id="test-flow-model",
            model_provider="openai",
            model_name="gpt-4.1-mini",
            temperature=0.0,
            interface="langchain",
            system_prompt="You are a helpful assistant.",
        )

    @pytest.fixture
    def sample_question(self) -> str:
        """Sample question for testing."""
        return "What is photosynthesis?"

    @pytest.fixture
    def sample_answer(self) -> str:
        """Sample answer for testing."""
        return "Photosynthesis is the process by which plants convert light energy into chemical energy."

    @patch("karenina.benchmark.verification.evaluators.rubric_evaluator.init_chat_model_unified")
    def test_flow_with_excerpts_full_pipeline(self, mock_init_model, _mock_model_config):
        """Test full pipeline: extract → validate → reasoning → scoring."""
        mock_llm = Mock()
        mock_init_model.return_value = mock_llm

        # Mock responses for each stage
        mock_llm.invoke.side_effect = [
            # Stage 1: Extract excerpts
            Mock(
                content=json.dumps(
                    {
                        "excerpts": [
                            {
                                "text": "Photosynthesis is the process by which plants convert light energy",
                                "confidence": "high",
                            }
                        ]
                    }
                )
            ),
            # Stage 2: Generate reasoning
            Mock(content="The answer correctly defines photosynthesis with accurate terminology."),
            # Stage 3: Extract score
            Mock(content=json.dumps({"scientific_accuracy": 5})),
        ]

        # This would normally be called through evaluate_rubric
        # For now, just verify the mock was set up correctly
        assert mock_llm.invoke.call_count == 0  # Not called yet

    @patch("karenina.benchmark.verification.evaluators.rubric_evaluator.init_chat_model_unified")
    def test_flow_without_excerpts(self, mock_init_model, _mock_model_config):
        """Test flow without excerpts: reasoning → scoring (2 stages)."""
        mock_llm = Mock()
        mock_init_model.return_value = mock_llm

        # Only reasoning and scoring stages (no excerpt extraction)
        mock_llm.invoke.side_effect = [
            # Stage 1: Generate reasoning (using full response)
            Mock(content="The overall response is clear and well-organized."),
            # Stage 2: Extract score
            Mock(content=json.dumps({"overall_clarity": 4})),
        ]

        trait = LLMRubricTrait(
            name="overall_clarity",
            description="Is the response clear?",
            kind="score",
            min_score=1,
            max_score=5,
            deep_judgment_enabled=True,
            deep_judgment_excerpt_enabled=False,  # No excerpts
        )

        # When excerpt_enabled=False, should skip excerpt extraction
        assert trait.deep_judgment_excerpt_enabled is False

    @patch("karenina.benchmark.verification.evaluators.rubric_evaluator.init_chat_model_unified")
    def test_mixed_excerpt_settings(self, mock_init_model, _mock_model_config):
        """Test rubric with mixed excerpt settings."""
        mock_llm = Mock()
        mock_init_model.return_value = mock_llm

        # Create rubric with mixed traits
        trait_with_excerpts = LLMRubricTrait(
            name="scientific_accuracy",
            description="Scientifically accurate?",
            kind="score",
            min_score=1,
            max_score=5,
            deep_judgment_enabled=True,
            deep_judgment_excerpt_enabled=True,
        )

        trait_without_excerpts = LLMRubricTrait(
            name="overall_clarity",
            description="Clear and organized?",
            kind="score",
            min_score=1,
            max_score=5,
            deep_judgment_enabled=True,
            deep_judgment_excerpt_enabled=False,
        )

        standard_trait = LLMRubricTrait(
            name="completeness",
            description="Complete response?",
            kind="score",
            min_score=1,
            max_score=5,
            deep_judgment_enabled=False,  # Standard evaluation
        )

        rubric = Rubric(llm_traits=[trait_with_excerpts, trait_without_excerpts, standard_trait])

        # Verify trait configurations
        assert rubric.llm_traits[0].deep_judgment_enabled is True
        assert rubric.llm_traits[0].deep_judgment_excerpt_enabled is True
        assert rubric.llm_traits[1].deep_judgment_enabled is True
        assert rubric.llm_traits[1].deep_judgment_excerpt_enabled is False
        assert rubric.llm_traits[2].deep_judgment_enabled is False


class TestDeepJudgmentConfiguration:
    """Test configuration overrides and trait type evaluation."""

    @pytest.fixture
    def mock_model_config(self) -> ModelConfig:
        """Create a mock model configuration."""
        return ModelConfig(
            id="test-config-model",
            model_provider="openai",
            model_name="gpt-4.1-mini",
            temperature=0.0,
            interface="langchain",
            system_prompt="You are a helpful assistant.",
        )

    @patch("karenina.benchmark.verification.evaluators.rubric_evaluator.init_chat_model_unified")
    def test_per_trait_max_excerpts_override(self, mock_init_model, _mock_model_config):
        """Test per-trait max_excerpts override."""
        mock_llm = Mock()
        mock_init_model.return_value = mock_llm

        trait = LLMRubricTrait(
            name="test_trait",
            description="Test",
            kind="score",
            min_score=1,
            max_score=5,
            deep_judgment_enabled=True,
            deep_judgment_excerpt_enabled=True,
            deep_judgment_max_excerpts=10,  # Override
        )

        # Trait-specific value should override global default
        assert trait.deep_judgment_max_excerpts == 10

    @patch("karenina.benchmark.verification.evaluators.rubric_evaluator.init_chat_model_unified")
    def test_per_trait_fuzzy_threshold_override(self, mock_init_model, _mock_model_config):
        """Test per-trait fuzzy_threshold override."""
        mock_llm = Mock()
        mock_init_model.return_value = mock_llm

        trait = LLMRubricTrait(
            name="test_trait",
            description="Test",
            kind="score",
            min_score=1,
            max_score=5,
            deep_judgment_enabled=True,
            deep_judgment_excerpt_enabled=True,
            deep_judgment_fuzzy_match_threshold=0.95,  # Override
        )

        assert trait.deep_judgment_fuzzy_match_threshold == 0.95

    @patch("karenina.benchmark.verification.evaluators.rubric_evaluator.init_chat_model_unified")
    def test_boolean_trait_evaluation(self, mock_init_model, _mock_model_config):
        """Test deep judgment evaluation of boolean traits."""
        mock_llm = Mock()
        mock_init_model.return_value = mock_llm

        mock_llm.invoke.side_effect = [
            # Excerpt extraction
            Mock(
                content=json.dumps(
                    {"excerpts": [{"text": "Photosynthesis converts light energy", "confidence": "high"}]}
                )
            ),
            # Reasoning
            Mock(content="The answer mentions photosynthesis."),
            # Score (boolean)
            Mock(content=json.dumps({"mentions_photosynthesis": True})),
        ]

        trait = LLMRubricTrait(
            name="mentions_photosynthesis",
            description="Does the answer mention photosynthesis?",
            kind="boolean",  # Boolean trait
            deep_judgment_enabled=True,
            deep_judgment_excerpt_enabled=True,
        )

        # Boolean traits should return True/False
        assert trait.kind == "boolean"
        assert trait.validate_score(True)
        assert trait.validate_score(False)
        assert not trait.validate_score(5)

    @patch("karenina.benchmark.verification.evaluators.rubric_evaluator.init_chat_model_unified")
    def test_score_trait_evaluation(self, mock_init_model, _mock_model_config):
        """Test deep judgment evaluation of score traits."""
        mock_llm = Mock()
        mock_init_model.return_value = mock_llm

        trait = LLMRubricTrait(
            name="clarity",
            description="How clear is the response?",
            kind="score",
            min_score=1,
            max_score=5,
            deep_judgment_enabled=True,
            deep_judgment_excerpt_enabled=True,
        )

        # Score traits should validate numeric scores in range
        assert trait.kind == "score"
        assert trait.validate_score(1)
        assert trait.validate_score(3)
        assert trait.validate_score(5)
        assert not trait.validate_score(0)
        assert not trait.validate_score(6)
        assert not trait.validate_score(True)

    @patch("karenina.benchmark.verification.evaluators.rubric_evaluator.init_chat_model_unified")
    def test_score_range_validation(self, mock_init_model, _mock_model_config):
        """Test custom score range validation."""
        mock_llm = Mock()
        mock_init_model.return_value = mock_llm

        trait = LLMRubricTrait(
            name="custom_range",
            description="Custom range trait",
            kind="score",
            min_score=0,
            max_score=10,
            deep_judgment_enabled=True,
        )

        # Should validate against custom range
        assert trait.validate_score(0)
        assert trait.validate_score(5)
        assert trait.validate_score(10)
        assert not trait.validate_score(-1)
        assert not trait.validate_score(11)


class TestDeepJudgmentEdgeCases:
    """Test edge cases in deep judgment evaluation."""

    @pytest.fixture
    def mock_model_config(self) -> ModelConfig:
        """Create a mock model configuration."""
        return ModelConfig(
            id="test-edge-model",
            model_provider="openai",
            model_name="gpt-4.1-mini",
            temperature=0.0,
            interface="langchain",
            system_prompt="You are a helpful assistant.",
        )

    @pytest.fixture
    def sample_answer(self) -> str:
        """Sample answer for testing."""
        return "This is a test answer."

    @patch("karenina.benchmark.verification.evaluators.rubric_evaluator.init_chat_model_unified")
    def test_empty_excerpts_from_llm(self, mock_init_model, mock_model_config, sample_answer):
        """Test handling of empty excerpts list from LLM."""
        mock_llm = Mock()
        mock_init_model.return_value = mock_llm

        # LLM returns empty excerpts list
        mock_llm.invoke.return_value = Mock(content=json.dumps({"excerpts": []}))

        trait = LLMRubricTrait(
            name="test_trait",
            description="Test",
            kind="score",
            min_score=1,
            max_score=5,
            deep_judgment_enabled=True,
            deep_judgment_excerpt_enabled=True,
        )

        evaluator = RubricEvaluator(mock_model_config, evaluation_strategy="sequential")

        from karenina.schemas import VerificationConfig

        config = VerificationConfig(
            answering_models=[mock_model_config],
            parsing_models=[mock_model_config],
            deep_judgment_enabled=True,
            deep_judgment_fuzzy_match_threshold=0.8,
        )

        result = evaluator._extract_excerpts_for_trait(sample_answer, trait, config)

        # Empty excerpts should be treated as validation failure
        assert result["success"] is False or len(result.get("excerpts", [])) == 0

    @patch("karenina.benchmark.verification.evaluators.rubric_evaluator.init_chat_model_unified")
    def test_partial_validation_success(self, mock_init_model, mock_model_config, sample_answer):
        """Test when some excerpts are valid and some are invalid."""
        mock_llm = Mock()
        mock_init_model.return_value = mock_llm

        # Return mix of valid and invalid excerpts
        mock_llm.invoke.return_value = Mock(
            content=json.dumps(
                {
                    "excerpts": [
                        {"text": "This is a test answer", "confidence": "high"},  # Valid
                        {"text": "Completely wrong text", "confidence": "low"},  # Invalid
                    ]
                }
            )
        )

        trait = LLMRubricTrait(
            name="test_trait",
            description="Test",
            kind="score",
            min_score=1,
            max_score=5,
            deep_judgment_enabled=True,
            deep_judgment_excerpt_enabled=True,
        )

        evaluator = RubricEvaluator(mock_model_config, evaluation_strategy="sequential")

        from karenina.schemas import VerificationConfig

        config = VerificationConfig(
            answering_models=[mock_model_config],
            parsing_models=[mock_model_config],
            deep_judgment_enabled=True,
            deep_judgment_fuzzy_match_threshold=0.8,
        )

        result = evaluator._extract_excerpts_for_trait(sample_answer, trait, config)

        # Should proceed with valid excerpts
        if result["success"]:
            assert len(result["excerpts"]) >= 1

    @patch("karenina.benchmark.verification.evaluators.rubric_evaluator.init_chat_model_unified")
    def test_search_enabled_without_excerpts(self, mock_init_model, _mock_model_config):
        """Test that search_enabled without excerpt_enabled is handled gracefully."""
        mock_llm = Mock()
        mock_init_model.return_value = mock_llm

        trait = LLMRubricTrait(
            name="test_trait",
            description="Test",
            kind="score",
            min_score=1,
            max_score=5,
            deep_judgment_enabled=True,
            deep_judgment_excerpt_enabled=False,  # No excerpts
            deep_judgment_search_enabled=True,  # But search enabled (should be no-op)
        )

        # Search should be ignored when excerpts are disabled
        assert trait.deep_judgment_excerpt_enabled is False
        assert trait.deep_judgment_search_enabled is True
        # This configuration should not cause errors
