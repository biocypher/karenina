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

    def test_agentic_parsing_not_supported_with_rubric_only(self):
        with pytest.raises(ValueError, match="rubric_only"):
            VerificationConfig(
                answering_models=[_sdk_model()],
                parsing_models=[_sdk_model()],
                evaluation_mode="rubric_only",
                agentic_parsing=True,
            )
