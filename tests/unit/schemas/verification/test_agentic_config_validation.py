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
    def test_disabled_agentic_parsing_needs_no_workspace_root(self):
        config = VerificationConfig(
            parsing_models=[_sdk_model()],
            parsing_only=True,
            agentic_parsing=False,
        )
        assert config.agentic_parsing is False

    def test_agentic_parsing_requires_workspace_root(self):
        with pytest.raises(ValueError, match="workspace_root is required"):
            VerificationConfig(
                parsing_models=[_sdk_model()],
                parsing_only=True,
                agentic_parsing=True,
            )

    def test_agentic_parsing_requires_existing_workspace_root(self, tmp_path):
        missing = tmp_path / "nonexistent"
        with pytest.raises(ValueError, match="does not exist"):
            VerificationConfig(
                parsing_models=[_sdk_model()],
                parsing_only=True,
                agentic_parsing=True,
                workspace_root=missing,
            )

    def test_valid_config_with_existing_workspace_root(self, tmp_path):
        config = VerificationConfig(
            parsing_models=[_sdk_model()],
            parsing_only=True,
            agentic_parsing=True,
            workspace_root=tmp_path,
        )
        assert config.agentic_parsing is True
        assert config.workspace_root == tmp_path
