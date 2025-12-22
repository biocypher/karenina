"""Tests for abstention detection functionality."""

from unittest.mock import Mock, patch

import pytest

from karenina.benchmark.verification.evaluators.abstention_checker import (
    detect_abstention,
    is_retryable_error,
)
from karenina.benchmark.verification.utils.parsing import _strip_markdown_fences
from karenina.schemas import ModelConfig
from karenina.schemas.workflow.verification import (
    VerificationResult,
    VerificationResultMetadata,
    VerificationResultTemplate,
)


class TestStripMarkdownFences:
    """Test the markdown fence stripping utility."""

    def test_strip_json_fences(self):
        """Test stripping JSON markdown fences."""
        text = '```json\n{"key": "value"}\n```'
        result = _strip_markdown_fences(text)
        assert result == '{"key": "value"}'

    def test_strip_plain_fences(self):
        """Test stripping plain markdown fences."""
        text = '```\n{"key": "value"}\n```'
        result = _strip_markdown_fences(text)
        assert result == '{"key": "value"}'

    def test_no_fences(self):
        """Test text without fences is unchanged."""
        text = '{"key": "value"}'
        result = _strip_markdown_fences(text)
        assert result == '{"key": "value"}'

    def test_partial_fences(self):
        """Test text with only opening fence."""
        text = '```json\n{"key": "value"}'
        result = _strip_markdown_fences(text)
        assert result == '{"key": "value"}'

    def test_non_string_input(self):
        """Test non-string input is returned as-is."""
        result = _strip_markdown_fences(123)
        assert result == 123


class TestIsRetryableError:
    """Test retryable error detection."""

    def test_connection_error(self):
        """Test connection errors are retryable."""
        error = Exception("Connection error occurred")
        assert is_retryable_error(error) is True

    def test_timeout_error(self):
        """Test timeout errors are retryable."""
        error = Exception("Request timed out")
        assert is_retryable_error(error) is True

    def test_rate_limit_error(self):
        """Test rate limit errors are retryable."""
        error = Exception("Rate limit exceeded: 429")
        assert is_retryable_error(error) is True

    def test_500_error(self):
        """Test 500 errors are retryable."""
        error = Exception("Server error: 500")
        assert is_retryable_error(error) is True

    def test_non_retryable_error(self):
        """Test non-retryable errors."""
        error = Exception("Invalid JSON format")
        assert is_retryable_error(error) is False

    def test_validation_error(self):
        """Test validation errors are not retryable."""
        error = ValueError("Invalid input")
        assert is_retryable_error(error) is False


class TestDetectAbstention:
    """Test the detect_abstention function."""

    @pytest.fixture
    def parsing_model(self):
        """Create a mock parsing model config."""
        return ModelConfig(
            id="test-parser",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            interface="langchain",
            system_prompt="You are a validation assistant.",
        )

    def test_detect_abstention_with_refusal(self, parsing_model):
        """Test abstention detection with a clear refusal."""
        # Mock the LLM to return abstention detected
        with patch(
            "karenina.benchmark.verification.evaluators.abstention_checker.init_chat_model_unified"
        ) as mock_init:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.content = '{"abstention_detected": true, "reasoning": "Model refused to answer"}'
            mock_llm.invoke.return_value = mock_response
            mock_init.return_value = mock_llm

            response = "I cannot answer this question as I don't have access to that information."
            question = "What is the secret password?"

            abstention_detected, check_performed, reasoning, usage_metadata = detect_abstention(
                response, parsing_model, question
            )

            assert check_performed is True
            assert abstention_detected is True
            assert reasoning == "Model refused to answer"

    def test_detect_abstention_with_genuine_answer(self, parsing_model):
        """Test abstention detection with a genuine answer."""
        with patch(
            "karenina.benchmark.verification.evaluators.abstention_checker.init_chat_model_unified"
        ) as mock_init:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.content = '{"abstention_detected": false, "reasoning": "Model provided substantive answer"}'
            mock_llm.invoke.return_value = mock_response
            mock_init.return_value = mock_llm

            response = "The capital of France is Paris."
            question = "What is the capital of France?"

            abstention_detected, check_performed, reasoning, usage_metadata = detect_abstention(
                response, parsing_model, question
            )

            assert check_performed is True
            assert abstention_detected is False
            assert reasoning == "Model provided substantive answer"

    def test_detect_abstention_with_markdown_fences(self, parsing_model):
        """Test abstention detection with markdown-fenced JSON response."""
        with patch(
            "karenina.benchmark.verification.evaluators.abstention_checker.init_chat_model_unified"
        ) as mock_init:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.content = '```json\n{"abstention_detected": true, "reasoning": "Clear refusal"}\n```'
            mock_llm.invoke.return_value = mock_response
            mock_init.return_value = mock_llm

            response = "I'm not able to provide that information."
            question = "What is X?"

            abstention_detected, check_performed, reasoning, usage_metadata = detect_abstention(
                response, parsing_model, question
            )

            assert check_performed is True
            assert abstention_detected is True
            assert reasoning == "Clear refusal"

    def test_detect_abstention_with_json_parse_error(self, parsing_model):
        """Test abstention detection with invalid JSON response."""
        with patch(
            "karenina.benchmark.verification.evaluators.abstention_checker.init_chat_model_unified"
        ) as mock_init:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.content = "This is not valid JSON"
            mock_llm.invoke.return_value = mock_response
            mock_init.return_value = mock_llm

            response = "Some answer"
            question = "Some question?"

            abstention_detected, check_performed, reasoning, usage_metadata = detect_abstention(
                response, parsing_model, question
            )

            # When JSON parsing fails, check_performed should be False
            assert check_performed is False
            assert abstention_detected is False
            assert reasoning is None

    def test_detect_abstention_with_retryable_error(self, parsing_model):
        """Test abstention detection with retryable error (should retry)."""
        with patch(
            "karenina.benchmark.verification.evaluators.abstention_checker.init_chat_model_unified"
        ) as mock_init:
            mock_llm = Mock()
            # First call: retryable error, second call: success
            mock_llm.invoke.side_effect = [
                Exception("Connection timeout"),
                Mock(content='{"abstention_detected": false, "reasoning": "OK"}'),
            ]
            mock_init.return_value = mock_llm

            response = "Answer text"
            question = "Question text?"

            # Should retry and succeed on second attempt
            abstention_detected, check_performed, reasoning, usage_metadata = detect_abstention(
                response, parsing_model, question
            )

            assert check_performed is True
            assert abstention_detected is False
            assert reasoning == "OK"
            assert mock_llm.invoke.call_count == 2

    def test_detect_abstention_with_non_retryable_error(self, parsing_model):
        """Test abstention detection with non-retryable error (should not retry)."""
        with patch(
            "karenina.benchmark.verification.evaluators.abstention_checker.init_chat_model_unified"
        ) as mock_init:
            mock_llm = Mock()
            mock_llm.invoke.side_effect = ValueError("Invalid input format")
            mock_init.return_value = mock_llm

            response = "Answer text"
            question = "Question text?"

            # Should not retry, return False
            abstention_detected, check_performed, reasoning, usage_metadata = detect_abstention(
                response, parsing_model, question
            )

            assert check_performed is False
            assert abstention_detected is False
            assert reasoning is None
            assert mock_llm.invoke.call_count == 1


class TestAbstentionIntegrationWithOrchestrator:
    """Test abstention detection integration with orchestrator."""

    def test_abstention_parameter_propagation(self):
        """Test that abstention_enabled parameter propagates through the orchestrator."""
        from karenina.benchmark.verification.multi_model_orchestrator import _create_verification_task

        task = _create_verification_task(
            question_id="test-q-123",
            question_text="Test question?",
            template_code="test_code",
            answering_model=ModelConfig(
                id="a", model_provider="openai", model_name="gpt-4o-mini", temperature=0.1, system_prompt="Test"
            ),
            parsing_model=ModelConfig(
                id="p", model_provider="openai", model_name="gpt-4o-mini", temperature=0.0, system_prompt="Test"
            ),
            run_name="test",
            replicate=1,
            rubric=None,
            abstention_enabled=True,
        )

        assert task["abstention_enabled"] is True

    def test_abstention_parameter_default_false(self):
        """Test that abstention_enabled defaults to False."""
        from karenina.benchmark.verification.multi_model_orchestrator import _create_verification_task

        task = _create_verification_task(
            question_id="test-q-123",
            question_text="Test question?",
            template_code="test_code",
            answering_model=ModelConfig(
                id="a", model_provider="openai", model_name="gpt-4o-mini", temperature=0.1, system_prompt="Test"
            ),
            parsing_model=ModelConfig(
                id="p", model_provider="openai", model_name="gpt-4o-mini", temperature=0.0, system_prompt="Test"
            ),
            run_name="test",
            replicate=1,
            rubric=None,
            abstention_enabled=False,
        )

        assert task["abstention_enabled"] is False


class TestAbstentionPrompts:
    """Test abstention detection prompts."""

    def test_abstention_prompts_exist(self):
        """Test that abstention prompts are defined."""
        from karenina.benchmark.verification.utils.prompts import (
            ABSTENTION_DETECTION_SYS,
            ABSTENTION_DETECTION_USER,
        )

        assert ABSTENTION_DETECTION_SYS is not None
        assert len(ABSTENTION_DETECTION_SYS) > 0
        assert ABSTENTION_DETECTION_USER is not None
        assert len(ABSTENTION_DETECTION_USER) > 0

    def test_abstention_system_prompt_content(self):
        """Test that system prompt contains key instructions."""
        from karenina.benchmark.verification.utils.prompts import ABSTENTION_DETECTION_SYS

        # Check for key elements
        assert "<role>" in ABSTENTION_DETECTION_SYS
        assert "abstention detection" in ABSTENTION_DETECTION_SYS.lower()
        assert "JSON" in ABSTENTION_DETECTION_SYS or "json" in ABSTENTION_DETECTION_SYS
        assert "abstention_detected" in ABSTENTION_DETECTION_SYS

    def test_abstention_user_prompt_format(self):
        """Test that user prompt has correct format placeholders."""
        from karenina.benchmark.verification.utils.prompts import ABSTENTION_DETECTION_USER

        assert "{question}" in ABSTENTION_DETECTION_USER
        assert "{response}" in ABSTENTION_DETECTION_USER


class TestVerificationConfigAbstention:
    """Test VerificationConfig with abstention settings."""

    def test_verification_config_has_abstention_field(self):
        """Test that VerificationConfig has abstention_enabled field."""
        from karenina.schemas import ModelConfig, VerificationConfig

        parsing_model = ModelConfig(
            id="parser",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            interface="langchain",
            system_prompt="You are a validation assistant.",
        )
        config = VerificationConfig(parsing_models=[parsing_model], parsing_only=True)
        assert hasattr(config, "abstention_enabled")
        assert config.abstention_enabled is False  # Default should be False

    def test_verification_config_abstention_enabled(self):
        """Test setting abstention_enabled to True."""
        from karenina.schemas import ModelConfig, VerificationConfig

        parsing_model = ModelConfig(
            id="parser",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            interface="langchain",
            system_prompt="You are a validation assistant.",
        )
        config = VerificationConfig(parsing_models=[parsing_model], parsing_only=True, abstention_enabled=True)
        assert config.abstention_enabled is True


class TestVerificationResultAbstention:
    """Test VerificationResult with abstention metadata."""

    def test_verification_result_has_abstention_fields(self):
        """Test that VerificationResult has abstention metadata fields."""

        result = VerificationResult(
            metadata=VerificationResultMetadata(
                question_id="test-id",
                template_id="test-template-id",
                completed_without_errors=True,
                question_text="Test question?",
                answering_model="openai/gpt-4o-mini",
                parsing_model="openai/gpt-4o-mini",
                execution_time=1.0,
                timestamp="2025-01-01T00:00:00",
            ),
            template=VerificationResultTemplate(
                raw_llm_response="Test response",
            ),
        )

        assert hasattr(result.template, "abstention_check_performed")
        assert hasattr(result.template, "abstention_detected")
        assert hasattr(result.template, "abstention_override_applied")

    def test_verification_result_abstention_defaults(self):
        """Test default values for abstention fields."""

        result = VerificationResult(
            metadata=VerificationResultMetadata(
                question_id="test-id",
                template_id="test-template-id",
                completed_without_errors=True,
                question_text="Test question?",
                answering_model="openai/gpt-4o-mini",
                parsing_model="openai/gpt-4o-mini",
                execution_time=1.0,
                timestamp="2025-01-01T00:00:00",
            ),
            template=VerificationResultTemplate(
                raw_llm_response="Test response",
            ),
        )

        assert result.template.abstention_check_performed is False
        assert result.template.abstention_detected is None
        assert result.template.abstention_override_applied is False

    def test_verification_result_abstention_custom_values(self):
        """Test setting custom values for abstention fields."""

        result = VerificationResult(
            metadata=VerificationResultMetadata(
                question_id="test-id",
                template_id="test-template-id",
                completed_without_errors=True,
                question_text="Test question?",
                answering_model="openai/gpt-4o-mini",
                parsing_model="openai/gpt-4o-mini",
                execution_time=1.0,
                timestamp="2025-01-01T00:00:00",
            ),
            template=VerificationResultTemplate(
                raw_llm_response="Test response",
                abstention_check_performed=True,
                abstention_detected=True,
                abstention_override_applied=True,
            ),
        )

        assert result.template.abstention_check_performed is True
        assert result.template.abstention_detected is True
        assert result.template.abstention_override_applied is True
