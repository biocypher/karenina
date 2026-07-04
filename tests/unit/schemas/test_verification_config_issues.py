"""Tests for VerificationConfig bug fixes.

Covers:
- Issue 015: System prompt auto-assignment mutates shared ModelConfig in-place
- Issue 135: db_config typed as Any instead of DBConfig
"""

import pytest
from pydantic import ValidationError

from karenina.schemas.config import ModelConfig
from karenina.schemas.verification.config import (
    DEFAULT_ANSWERING_SYSTEM_PROMPT,
    VerificationConfig,
)
from karenina.storage.db_config import DBConfig


@pytest.mark.unit
class TestSharedModelConfigMutation:
    """Issue 015: Shared ModelConfig should not be mutated in-place."""

    def test_shared_model_gets_correct_prompts(self) -> None:
        """When the same ModelConfig is used for answering and parsing,
        each role should get its own system prompt without cross-contamination.
        """
        shared = ModelConfig(
            id="shared",
            model_name="gpt-4",
            model_provider="openai",
            interface="langchain",
        )
        config = VerificationConfig(
            answering_models=[shared],
            parsing_models=[shared],
        )
        # After construction, the answering copy should have the answering prompt
        assert config.answering_models[0].system_prompt == DEFAULT_ANSWERING_SYSTEM_PROMPT
        # Parsing models no longer receive auto-assigned system_prompt
        assert config.parsing_models[0].system_prompt is None

    def test_original_model_not_mutated(self) -> None:
        """The original ModelConfig instance must not be mutated by VerificationConfig.__init__."""
        original = ModelConfig(
            id="original",
            model_name="gpt-4",
            model_provider="openai",
            interface="langchain",
        )
        assert original.system_prompt is None

        VerificationConfig(
            answering_models=[original],
            parsing_models=[
                ModelConfig(
                    id="parser",
                    model_name="gpt-4",
                    model_provider="openai",
                    interface="langchain",
                )
            ],
        )
        # The original should still be None (not mutated)
        assert original.system_prompt is None

    def test_explicit_prompt_preserved(self) -> None:
        """Models with an explicit system_prompt should not be overwritten."""
        custom_prompt = "I am a custom prompt"
        model = ModelConfig(
            id="custom",
            model_name="gpt-4",
            model_provider="openai",
            interface="langchain",
            system_prompt=custom_prompt,
        )
        config = VerificationConfig(
            answering_models=[model],
            parsing_models=[
                ModelConfig(
                    id="parser",
                    model_name="gpt-4",
                    model_provider="openai",
                    interface="langchain",
                )
            ],
        )
        assert config.answering_models[0].system_prompt == custom_prompt


@pytest.mark.unit
class TestDBConfigTyping:
    """Issue 135: db_config should validate against DBConfig, not accept Any."""

    def test_valid_db_config_accepted(self) -> None:
        """A valid DBConfig instance should be accepted."""
        db = DBConfig(storage_url="sqlite:///test.db")
        config = VerificationConfig(
            answering_models=[
                ModelConfig(
                    id="a",
                    model_name="gpt-4",
                    model_provider="openai",
                    interface="langchain",
                )
            ],
            parsing_models=[
                ModelConfig(
                    id="p",
                    model_name="gpt-4",
                    model_provider="openai",
                    interface="langchain",
                )
            ],
            db_config=db,
        )
        assert config.db_config is not None
        assert config.db_config.storage_url == "sqlite:///test.db"

    def test_none_db_config_accepted(self) -> None:
        """None should still be accepted for db_config."""
        config = VerificationConfig(
            answering_models=[
                ModelConfig(
                    id="a",
                    model_name="gpt-4",
                    model_provider="openai",
                    interface="langchain",
                )
            ],
            parsing_models=[
                ModelConfig(
                    id="p",
                    model_name="gpt-4",
                    model_provider="openai",
                    interface="langchain",
                )
            ],
            db_config=None,
        )
        assert config.db_config is None

    def test_arbitrary_value_rejected(self) -> None:
        """An arbitrary non-DBConfig value should be rejected by validation."""
        with pytest.raises((ValidationError, TypeError)):
            VerificationConfig(
                answering_models=[
                    ModelConfig(
                        id="a",
                        model_name="gpt-4",
                        model_provider="openai",
                        interface="langchain",
                    )
                ],
                parsing_models=[
                    ModelConfig(
                        id="p",
                        model_name="gpt-4",
                        model_provider="openai",
                        interface="langchain",
                    )
                ],
                db_config="not-a-db-config",
            )
