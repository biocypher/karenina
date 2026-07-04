"""Tests for VerificationConfig instruction shortcut fields."""

import pytest

from karenina.schemas.config import ModelConfig
from karenina.schemas.verification.config import VerificationConfig
from karenina.schemas.verification.prompt_config import PromptConfig


def _make_models() -> tuple[ModelConfig, ModelConfig]:
    answering = ModelConfig(id="a", model_name="m", model_provider="openai", system_prompt="You are helpful.")
    parsing = ModelConfig(id="p", model_name="m", model_provider="openai")
    return answering, parsing


@pytest.mark.unit
class TestInstructionShortcuts:
    """Test that *_instructions shortcuts wire into PromptConfig."""

    def test_parsing_instructions_creates_prompt_config(self) -> None:
        a, p = _make_models()
        config = VerificationConfig(
            answering_models=[a],
            parsing_models=[p],
            parsing_instructions="Be strict",
        )
        assert config.prompt_config is not None
        assert config.prompt_config.parsing == "Be strict"

    def test_generation_instructions_creates_prompt_config(self) -> None:
        a, p = _make_models()
        config = VerificationConfig(
            answering_models=[a],
            parsing_models=[p],
            generation_instructions="Focus on accuracy",
        )
        assert config.prompt_config is not None
        assert config.prompt_config.generation == "Focus on accuracy"

    def test_multiple_shortcuts_merge(self) -> None:
        a, p = _make_models()
        config = VerificationConfig(
            answering_models=[a],
            parsing_models=[p],
            parsing_instructions="Be strict",
            generation_instructions="Focus on accuracy",
        )
        assert config.prompt_config is not None
        assert config.prompt_config.parsing == "Be strict"
        assert config.prompt_config.generation == "Focus on accuracy"

    def test_explicit_prompt_config_takes_precedence(self) -> None:
        a, p = _make_models()
        config = VerificationConfig(
            answering_models=[a],
            parsing_models=[p],
            prompt_config=PromptConfig(parsing="Explicit"),
            parsing_instructions="Shortcut",
        )
        assert config.prompt_config.parsing == "Explicit"

    def test_shortcut_fills_unset_prompt_config_fields(self) -> None:
        a, p = _make_models()
        config = VerificationConfig(
            answering_models=[a],
            parsing_models=[p],
            prompt_config=PromptConfig(parsing="Explicit"),
            generation_instructions="From shortcut",
        )
        assert config.prompt_config.parsing == "Explicit"
        assert config.prompt_config.generation == "From shortcut"

    def test_all_shortcut_fields_exist(self) -> None:
        a, p = _make_models()
        config = VerificationConfig(
            answering_models=[a],
            parsing_models=[p],
            generation_instructions="g",
            parsing_instructions="p",
            abstention_detection_instructions="a",
            sufficiency_detection_instructions="s",
            rubric_evaluation_instructions="r",
            agentic_parsing_instructions="ap",
            deep_judgment_instructions="dj",
        )
        pc = config.prompt_config
        assert pc is not None
        assert pc.generation == "g"
        assert pc.parsing == "p"
        assert pc.abstention_detection == "a"
        assert pc.sufficiency_detection == "s"
        assert pc.rubric_evaluation == "r"
        assert pc.agentic_parsing == "ap"
        assert pc.deep_judgment == "dj"
