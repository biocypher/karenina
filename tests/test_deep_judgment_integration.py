"""Integration tests for deep-judgment feature in verification runner.

This module tests the integration of deep-judgment parsing into the
run_single_model_verification function.
"""

from unittest.mock import MagicMock, patch

import pytest

from karenina.benchmark.models import ModelConfig
from karenina.benchmark.verification.runner import run_single_model_verification


@pytest.fixture
def test_template_code():
    """Minimal Answer template for testing."""
    return """
from karenina.schemas.answer_class import BaseAnswer

class Answer(BaseAnswer):
    id: str
    correct: str
    drug_target: str
    mechanism: str
"""


@pytest.fixture
def minimal_model_config():
    """Minimal ModelConfig for testing."""
    return ModelConfig(
        id="test-model",
        model_provider="openai",
        model_name="gpt-4.1-mini",
        temperature=0.0,
        system_prompt="You are a helpful assistant.",
    )


class TestDeepJudgmentIntegration:
    """Integration tests for deep-judgment in runner."""

    def test_deep_judgment_config_parameters_accepted(self):
        """Test that deep-judgment parameters are accepted by runner function."""
        # This test verifies the function signature accepts deep-judgment parameters
        # We're not actually running it (would need real LLM), just checking it doesn't error
        from inspect import signature

        sig = signature(run_single_model_verification)
        params = sig.parameters

        # Verify deep-judgment parameters exist in function signature
        assert "deep_judgment_enabled" in params
        assert "deep_judgment_max_excerpts_per_attribute" in params
        assert "deep_judgment_fuzzy_match_threshold" in params
        assert "deep_judgment_excerpt_retry_attempts" in params

        # Verify defaults match VerificationConfig defaults
        assert params["deep_judgment_enabled"].default is False
        assert params["deep_judgment_max_excerpts_per_attribute"].default == 3
        assert params["deep_judgment_fuzzy_match_threshold"].default == 0.80
        assert params["deep_judgment_excerpt_retry_attempts"].default == 2

    @patch("karenina.benchmark.verification.runner.init_chat_model_unified")
    @patch("karenina.benchmark.verification.runner.deep_judgment_parse")
    def test_deep_judgment_disabled_uses_standard_parsing(
        self, mock_deep_judgment_parse, mock_init_chat, test_template_code, minimal_model_config
    ):
        """Test that disabling deep_judgment uses standard parsing."""
        # Setup mocks
        mock_llm = MagicMock()
        mock_init_chat.return_value = mock_llm

        # Mock LLM responses
        mock_llm.invoke.side_effect = [
            MagicMock(content="The drug targets BCL-2 protein."),  # Answering
            MagicMock(content='{"drug_target": "BCL-2", "mechanism": "inhibition"}'),  # Parsing
        ]

        # Call with deep_judgment_enabled=False (default)
        result = run_single_model_verification(
            question_id="a" * 32,
            question_text="What is the drug target?",
            template_code=test_template_code,
            answering_model=minimal_model_config,
            parsing_model=minimal_model_config,
            deep_judgment_enabled=False,
        )

        # Verify deep_judgment_parse was NOT called
        assert not mock_deep_judgment_parse.called
        assert result.deep_judgment_enabled is False
        assert result.deep_judgment_performed is False

    @patch("karenina.benchmark.verification.runner.init_chat_model_unified")
    def test_deep_judgment_metadata_in_error_cases(self, mock_init_chat, test_template_code, minimal_model_config):
        """Test that deep-judgment metadata is present even in error cases."""
        # Setup mocks to cause an error
        mock_llm = MagicMock()
        mock_init_chat.return_value = mock_llm
        mock_llm.invoke.side_effect = Exception("Test error")

        # Call with deep_judgment_enabled=True
        result = run_single_model_verification(
            question_id="a" * 32,
            question_text="What is the drug target?",
            template_code=test_template_code,
            answering_model=minimal_model_config,
            parsing_model=minimal_model_config,
            deep_judgment_enabled=True,
        )

        # Verify error result includes deep-judgment metadata
        assert result.success is False
        assert result.deep_judgment_enabled is True
        assert result.deep_judgment_performed is False  # Failed before parsing
        assert result.extracted_excerpts is None
        assert result.attribute_reasoning is None
