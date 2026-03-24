"""Tests for parsing model system_prompt relaxation."""

import pytest

from karenina.schemas.config import ModelConfig
from karenina.schemas.verification.config import VerificationConfig


@pytest.mark.unit
class TestParsingSystemPromptRelaxation:
    """Verify parsing models no longer need system_prompt."""

    def test_parsing_model_none_system_prompt_valid(self) -> None:
        """Parsing model with system_prompt=None should not raise."""
        answering = ModelConfig(id="a", model_name="m", model_provider="openai", system_prompt="You are helpful.")
        parsing = ModelConfig(id="p", model_name="m", model_provider="openai", system_prompt=None)
        config = VerificationConfig(answering_models=[answering], parsing_models=[parsing])
        assert config.parsing_models[0].system_prompt is None

    def test_answering_model_still_requires_system_prompt(self) -> None:
        """Answering model with no system_prompt gets auto-assigned default.

        The auto-assignment block fills in DEFAULT_ANSWERING_SYSTEM_PROMPT for
        answering models with empty/None system_prompt. The validation check
        acts as a safety net after auto-assignment. To verify that the
        validation targets only answering models, we call _validate_config
        directly on a config where the answering model's prompt was cleared
        after construction.
        """
        answering = ModelConfig(id="a", model_name="m", model_provider="openai", system_prompt="You are helpful.")
        parsing = ModelConfig(id="p", model_name="m", model_provider="openai", system_prompt=None)
        config = VerificationConfig(answering_models=[answering], parsing_models=[parsing])
        # Bypass auto-assignment by clearing the prompt after construction
        cleared = answering.model_copy(update={"system_prompt": None})
        object.__setattr__(config, "answering_models", [cleared])
        with pytest.raises(ValueError, match="[Ss]ystem prompt.*required.*answering"):
            config._validate_config()

    def test_parsing_model_no_auto_assignment(self) -> None:
        """Parsing model should NOT receive DEFAULT_PARSING_SYSTEM_PROMPT."""
        answering = ModelConfig(id="a", model_name="m", model_provider="openai", system_prompt="You are helpful.")
        parsing = ModelConfig(id="p", model_name="m", model_provider="openai")
        config = VerificationConfig(answering_models=[answering], parsing_models=[parsing])
        assert config.parsing_models[0].system_prompt is None
