"""Tests for agentic parsing validation in VerificationConfig."""

import pytest

from karenina.schemas.config.models import ModelConfig
from karenina.schemas.verification.config import VerificationConfig


def _sdk_model():
    return ModelConfig(
        id="test",
        model_name="test",
        interface="claude_agent_sdk",
    )


@pytest.mark.unit
class TestAgenticParsingValidation:
    def test_disabled_agentic_parsing_needs_no_special_config(self):
        config = VerificationConfig(
            parsing_models=[_sdk_model()],
            parsing_only=True,
            agentic_parsing=False,
        )
        assert config.agentic_parsing is False

    def test_agentic_parsing_accepted_without_workspace_root(self):
        """workspace_root now lives on Benchmark, not VerificationConfig."""
        config = VerificationConfig(
            parsing_models=[_sdk_model()],
            parsing_only=True,
            agentic_parsing=True,
        )
        assert config.agentic_parsing is True

    def test_agentic_parsing_rejects_tool_loop_interface(self):
        """agentic_parsing=True should reject interfaces with agent_tier='tool_loop'."""
        with pytest.raises(ValueError, match="deep_agent"):
            VerificationConfig(
                answering_models=[
                    ModelConfig(
                        id="a",
                        model_name="m",
                        model_provider="anthropic",
                    )
                ],
                parsing_models=[
                    ModelConfig(
                        id="p",
                        model_name="m",
                        model_provider="anthropic",
                        interface="langchain",  # agent_tier="tool_loop"
                    )
                ],
                agentic_parsing=True,
            )

    def test_agentic_parsing_not_supported_with_rubric_only(self):
        with pytest.raises(ValueError, match="rubric_only"):
            VerificationConfig(
                answering_models=[_sdk_model()],
                parsing_models=[_sdk_model()],
                evaluation_mode="rubric_only",
                agentic_parsing=True,
            )
