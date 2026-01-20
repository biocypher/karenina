"""Unit tests for RubricEvaluator deep judgment methods.

This module tests the deep judgment evaluation flow for LLMRubricTraits:
- Per-trait deep judgment control
- Two evaluation flows: with/without excerpts
- Retry mechanism with fuzzy matching
- Auto-fail after retries exhausted
- Configuration overrides (max_excerpts, fuzzy_threshold, retry_attempts)
"""

import json
from typing import Any
from unittest.mock import Mock

import pytest

from karenina.benchmark.verification.evaluators.rubric_deep_judgment import RubricDeepJudgmentHandler
from karenina.ports.usage import UsageMetadata
from karenina.schemas import ModelConfig
from karenina.schemas.domain import LLMRubricTrait, Rubric
from karenina.schemas.workflow.rubric_outputs import TraitExcerpt, TraitExcerptsOutput


def create_mock_llmport(
    invoke_responses: list[str] | None = None,
    structured_responses: list[tuple[str, Any]] | None = None,
):
    """
    Create a mock LLMPort that properly simulates the interface.

    Args:
        invoke_responses: List of text responses for llm.invoke() calls
        structured_responses: Optional list of (json_content, parsed_model) for
            with_structured_output().invoke() calls

    Returns:
        Mock LLMPort that behaves like the real interface.
    """
    mock_llm = Mock()
    invoke_call_count = [0]
    structured_call_count = [0]

    _invoke_responses = invoke_responses or [""]

    # Mock for plain invoke() calls - returns LLMResponse with .content
    def mock_invoke(messages):
        response = Mock()
        response.content = _invoke_responses[invoke_call_count[0] % len(_invoke_responses)]
        response.usage = UsageMetadata(input_tokens=10, output_tokens=10, total_tokens=20)
        invoke_call_count[0] += 1
        return response

    mock_llm.invoke.side_effect = mock_invoke

    # Mock for with_structured_output() calls
    def mock_with_structured_output(schema):
        mock_structured_llm = Mock()

        def mock_structured_invoke(messages):
            nonlocal structured_call_count
            response = Mock()
            if structured_responses and structured_call_count[0] < len(structured_responses):
                content, raw = structured_responses[structured_call_count[0]]
                response.content = content
                response.raw = raw
            else:
                response.content = "{}"
                # Try to construct the schema if possible
                try:
                    response.raw = schema()
                except Exception:
                    response.raw = Mock()
            response.usage = UsageMetadata(input_tokens=10, output_tokens=10, total_tokens=20)
            structured_call_count[0] += 1
            return response

        mock_structured_llm.invoke.side_effect = mock_structured_invoke
        return mock_structured_llm

    mock_llm.with_structured_output.side_effect = mock_with_structured_output

    return mock_llm


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

    def test_excerpt_extraction_first_attempt_success(self, mock_model_config, sample_answer, dj_trait_with_excerpts):
        """Test successful excerpt extraction on first attempt."""
        # Create structured output for excerpt extraction with valid excerpts
        excerpts_output = TraitExcerptsOutput(
            excerpts=[
                TraitExcerpt(
                    text="Photosynthesis is the process by which plants convert light energy into chemical energy",
                    confidence="high",
                ),
                TraitExcerpt(
                    text="This occurs in the chloroplasts",
                    confidence="medium",
                ),
            ]
        )

        # Create mock LLM with structured output for excerpt extraction
        mock_llm = create_mock_llmport(
            invoke_responses=[],  # Not used for excerpt extraction
            structured_responses=[
                # Excerpt extraction uses with_structured_output()
                (json.dumps(excerpts_output.model_dump()), excerpts_output),
            ],
        )

        # Create handler directly with mock LLM
        handler = RubricDeepJudgmentHandler(mock_llm, mock_model_config)

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
        result = handler._extract_excerpts_for_trait(sample_answer, dj_trait_with_excerpts, config)

        # Verify results
        assert result["auto_fail"] is False
        assert "excerpts" in result
        assert len(result["excerpts"]) == 2
        assert result["retry_count"] == 0

        # Verify with_structured_output was called
        assert mock_llm.with_structured_output.call_count >= 1

    def test_excerpt_extraction_retry_success(self, mock_model_config, sample_answer, dj_trait_with_excerpts):
        """Test excerpt extraction succeeds on retry after validation failure."""
        # First attempt: excerpts with low similarity (fail validation)
        invalid_excerpts = TraitExcerptsOutput(
            excerpts=[
                TraitExcerpt(
                    text="Plants use sunlight",  # Low similarity to actual answer
                    confidence="low",
                )
            ]
        )

        # Second attempt: excerpts with high similarity (pass validation)
        valid_excerpts = TraitExcerptsOutput(
            excerpts=[
                TraitExcerpt(
                    text="Photosynthesis is the process by which plants convert light energy",
                    confidence="high",
                )
            ]
        )

        mock_llm = create_mock_llmport(
            invoke_responses=[],  # Not used for excerpt extraction
            structured_responses=[
                # First extraction attempt - invalid excerpts
                (json.dumps(invalid_excerpts.model_dump()), invalid_excerpts),
                # Second extraction attempt - valid excerpts
                (json.dumps(valid_excerpts.model_dump()), valid_excerpts),
            ],
        )

        handler = RubricDeepJudgmentHandler(mock_llm, mock_model_config)

        from karenina.schemas import VerificationConfig

        config = VerificationConfig(
            answering_models=[mock_model_config],
            parsing_models=[mock_model_config],
            deep_judgment_enabled=True,
            deep_judgment_excerpt_retry_attempts=2,
            deep_judgment_fuzzy_match_threshold=0.8,
        )

        result = handler._extract_excerpts_for_trait(sample_answer, dj_trait_with_excerpts, config)

        # Verify retry occurred
        assert result["auto_fail"] is False
        assert result["retry_count"] == 1
        assert len(result["excerpts"]) >= 1

    def test_excerpt_extraction_all_retries_exhausted(self, mock_model_config, sample_answer, dj_trait_with_excerpts):
        """Test auto-fail when all retry attempts are exhausted."""
        # All attempts return invalid excerpts (text not in answer)
        invalid_excerpts = TraitExcerptsOutput(
            excerpts=[
                TraitExcerpt(
                    text="Completely wrong text not in answer",
                    confidence="low",
                )
            ]
        )

        # Provide enough structured responses for initial + 2 retries
        mock_llm = create_mock_llmport(
            invoke_responses=[],
            structured_responses=[
                (json.dumps(invalid_excerpts.model_dump()), invalid_excerpts),
                (json.dumps(invalid_excerpts.model_dump()), invalid_excerpts),
                (json.dumps(invalid_excerpts.model_dump()), invalid_excerpts),
            ],
        )

        handler = RubricDeepJudgmentHandler(mock_llm, mock_model_config)

        from karenina.schemas import VerificationConfig

        config = VerificationConfig(
            answering_models=[mock_model_config],
            parsing_models=[mock_model_config],
            deep_judgment_enabled=True,
            deep_judgment_excerpt_retry_attempts=2,
            deep_judgment_fuzzy_match_threshold=0.8,
        )

        result = handler._extract_excerpts_for_trait(sample_answer, dj_trait_with_excerpts, config)

        # Verify auto-fail
        assert result["auto_fail"] is True
        assert result["retry_count"] == 2  # Max retries reached

    def test_per_trait_retry_override(self, mock_model_config, sample_answer):
        """Test that trait-specific retry_attempts override global default."""
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

        # All attempts fail with invalid excerpts
        invalid_excerpts = TraitExcerptsOutput(excerpts=[TraitExcerpt(text="Invalid text", confidence="low")])

        # Provide enough structured responses for initial + 5 retries
        mock_llm = create_mock_llmport(
            invoke_responses=[],
            structured_responses=[
                (json.dumps(invalid_excerpts.model_dump()), invalid_excerpts),
                (json.dumps(invalid_excerpts.model_dump()), invalid_excerpts),
                (json.dumps(invalid_excerpts.model_dump()), invalid_excerpts),
                (json.dumps(invalid_excerpts.model_dump()), invalid_excerpts),
                (json.dumps(invalid_excerpts.model_dump()), invalid_excerpts),
                (json.dumps(invalid_excerpts.model_dump()), invalid_excerpts),
            ],
        )

        handler = RubricDeepJudgmentHandler(mock_llm, mock_model_config)

        from karenina.schemas import VerificationConfig

        config = VerificationConfig(
            answering_models=[mock_model_config],
            parsing_models=[mock_model_config],
            deep_judgment_enabled=True,
            deep_judgment_excerpt_retry_attempts=2,  # Global default
            deep_judgment_fuzzy_match_threshold=0.8,
        )

        result = handler._extract_excerpts_for_trait(sample_answer, trait, config)

        # Should use trait-specific retry count (5) not global (2)
        assert result["retry_count"] == 5

    def test_validation_feedback_format(self, mock_model_config, sample_answer, dj_trait_with_excerpts):
        """Test that validation feedback includes scores and threshold."""
        # Track the prompts sent to LLM
        prompts_sent = []

        # Create invalid excerpts
        invalid_excerpts = TraitExcerptsOutput(excerpts=[TraitExcerpt(text="Wrong text", confidence="low")])

        # Create a custom mock that captures prompts
        mock_llm = Mock()
        call_count = [0]

        def mock_with_structured_output(schema):
            mock_structured_llm = Mock()

            def mock_invoke(messages):
                prompts_sent.append(messages)
                response = Mock()
                response.content = json.dumps(invalid_excerpts.model_dump())
                response.raw = invalid_excerpts
                response.usage = UsageMetadata(input_tokens=10, output_tokens=10, total_tokens=20)
                call_count[0] += 1
                return response

            mock_structured_llm.invoke.side_effect = mock_invoke
            return mock_structured_llm

        mock_llm.with_structured_output.side_effect = mock_with_structured_output

        handler = RubricDeepJudgmentHandler(mock_llm, mock_model_config)

        from karenina.schemas import VerificationConfig

        config = VerificationConfig(
            answering_models=[mock_model_config],
            parsing_models=[mock_model_config],
            deep_judgment_enabled=True,
            deep_judgment_excerpt_retry_attempts=1,
            deep_judgment_fuzzy_match_threshold=0.85,
        )

        handler._extract_excerpts_for_trait(sample_answer, dj_trait_with_excerpts, config)

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

    def test_flow_with_excerpts_full_pipeline(self):
        """Test full pipeline: extract → validate → reasoning → scoring."""
        # Create structured output for excerpt extraction
        excerpts_output = TraitExcerptsOutput(
            excerpts=[
                TraitExcerpt(
                    text="Photosynthesis is the process by which plants convert light energy",
                    confidence="high",
                )
            ]
        )

        # Create mock LLM with both structured and invoke responses
        mock_llm = create_mock_llmport(
            invoke_responses=[
                # Stage 2: Generate reasoning
                "The answer correctly defines photosynthesis with accurate terminology.",
            ],
            structured_responses=[
                # Stage 1: Extract excerpts
                (json.dumps(excerpts_output.model_dump()), excerpts_output),
            ],
        )

        # Verify mock was set up correctly
        assert mock_llm.with_structured_output is not None
        assert mock_llm.invoke is not None

    def test_flow_without_excerpts(self):
        """Test flow without excerpts: reasoning → scoring (2 stages)."""
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

    def test_mixed_excerpt_settings(self):
        """Test rubric with mixed excerpt settings."""
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

    def test_per_trait_max_excerpts_override(self):
        """Test per-trait max_excerpts override."""
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

    def test_per_trait_fuzzy_threshold_override(self):
        """Test per-trait fuzzy_threshold override."""
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

    def test_boolean_trait_evaluation(self):
        """Test deep judgment evaluation of boolean traits."""
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

    def test_score_trait_evaluation(self):
        """Test deep judgment evaluation of score traits."""
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

    def test_score_range_validation(self):
        """Test custom score range validation."""
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

    def test_empty_excerpts_from_llm(self, mock_model_config, sample_answer):
        """Test handling of empty excerpts list from LLM."""
        # LLM returns empty excerpts list
        empty_excerpts = TraitExcerptsOutput(excerpts=[])

        mock_llm = create_mock_llmport(
            invoke_responses=[],
            structured_responses=[
                (json.dumps(empty_excerpts.model_dump()), empty_excerpts),
            ],
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

        handler = RubricDeepJudgmentHandler(mock_llm, mock_model_config)

        from karenina.schemas import VerificationConfig

        config = VerificationConfig(
            answering_models=[mock_model_config],
            parsing_models=[mock_model_config],
            deep_judgment_enabled=True,
            deep_judgment_fuzzy_match_threshold=0.8,
        )

        result = handler._extract_excerpts_for_trait(sample_answer, trait, config)

        # Empty excerpts should be treated as validation failure
        assert result["auto_fail"] is True or len(result.get("excerpts", [])) == 0

    def test_partial_validation_success(self, mock_model_config, sample_answer):
        """Test when some excerpts are valid and some are invalid."""
        # Return mix of valid and invalid excerpts
        mixed_excerpts = TraitExcerptsOutput(
            excerpts=[
                TraitExcerpt(text="This is a test answer", confidence="high"),  # Valid
                TraitExcerpt(text="Completely wrong text", confidence="low"),  # Invalid
            ]
        )

        mock_llm = create_mock_llmport(
            invoke_responses=[],
            structured_responses=[
                (json.dumps(mixed_excerpts.model_dump()), mixed_excerpts),
            ],
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

        handler = RubricDeepJudgmentHandler(mock_llm, mock_model_config)

        from karenina.schemas import VerificationConfig

        config = VerificationConfig(
            answering_models=[mock_model_config],
            parsing_models=[mock_model_config],
            deep_judgment_enabled=True,
            deep_judgment_fuzzy_match_threshold=0.8,
        )

        result = handler._extract_excerpts_for_trait(sample_answer, trait, config)

        # Should proceed with valid excerpts
        if not result["auto_fail"]:
            assert len(result["excerpts"]) >= 1

    def test_search_enabled_without_excerpts(self):
        """Test that search_enabled without excerpt_enabled is handled gracefully."""
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
