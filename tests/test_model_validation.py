"""Tests for ModelConfiguration validation logic."""

import pytest

from karenina.benchmark.models import (
    INTERFACE_LANGCHAIN,
    INTERFACE_MANUAL,
    INTERFACE_OPENROUTER,
    INTERFACES_NO_PROVIDER_REQUIRED,
    ModelConfig,
    VerificationConfig,
)


class TestModelConfigurationValidation:
    """Test ModelConfiguration validation logic."""

    def test_valid_langchain_model_config(self) -> None:
        """Test valid configuration for langchain interface."""
        config = ModelConfig(
            id="test-model",
            model_provider="openai",
            model_name="gpt-3.5-turbo",
            temperature=0.1,
            interface=INTERFACE_LANGCHAIN,
            system_prompt="You are a helpful assistant.",
        )

        # Should not raise any errors
        assert config.id == "test-model"
        assert config.model_provider == "openai"
        assert config.interface == INTERFACE_LANGCHAIN

    def test_valid_openrouter_model_config(self) -> None:
        """Test valid configuration for openrouter interface (no provider required)."""
        config = ModelConfig(
            id="test-openrouter",
            model_provider="",  # Empty provider is allowed for openrouter
            model_name="openrouter/model",
            temperature=0.2,
            interface=INTERFACE_OPENROUTER,
            system_prompt="You are a helpful assistant.",
        )

        # Should not raise any errors
        assert config.id == "test-openrouter"
        assert config.model_provider == ""
        assert config.interface == INTERFACE_OPENROUTER

    def test_valid_manual_model_config(self) -> None:
        """Test valid configuration for manual interface (no provider required)."""
        config = ModelConfig(
            id="test-manual",
            model_provider="",  # Empty provider is allowed for manual
            model_name="manual-model",
            temperature=0.0,
            interface=INTERFACE_MANUAL,
            system_prompt="You are a helpful assistant.",
        )

        # Should not raise any errors
        assert config.id == "test-manual"
        assert config.model_provider == ""
        assert config.interface == INTERFACE_MANUAL

    def test_missing_model_provider_langchain_interface(self) -> None:
        """Test that missing provider fails for langchain interface."""
        with pytest.raises(ValueError, match="Model provider is required.*interface: langchain"):
            VerificationConfig(
                answering_models=[
                    ModelConfig(
                        id="test-model",
                        model_provider="",  # Empty provider should fail for langchain
                        model_name="gpt-3.5-turbo",
                        temperature=0.1,
                        interface=INTERFACE_LANGCHAIN,
                        system_prompt="You are a helpful assistant.",
                    )
                ],
                parsing_models=[
                    ModelConfig(
                        id="parsing-model",
                        model_provider="openai",
                        model_name="gpt-3.5-turbo",
                        temperature=0.1,
                        interface=INTERFACE_LANGCHAIN,
                        system_prompt="You are a validator.",
                    )
                ],
            )

    def test_missing_model_name(self) -> None:
        """Test that missing model name fails validation."""
        with pytest.raises(ValueError, match="Model name is required"):
            VerificationConfig(
                answering_models=[
                    ModelConfig(
                        id="test-model",
                        model_provider="openai",
                        model_name="",  # Empty model name should fail
                        temperature=0.1,
                        interface=INTERFACE_LANGCHAIN,
                        system_prompt="You are a helpful assistant.",
                    )
                ],
                parsing_models=[
                    ModelConfig(
                        id="parsing-model",
                        model_provider="openai",
                        model_name="gpt-3.5-turbo",
                        temperature=0.1,
                        interface=INTERFACE_LANGCHAIN,
                        system_prompt="You are a validator.",
                    )
                ],
            )

    def test_missing_system_prompt(self) -> None:
        """Test that missing system prompt fails validation."""
        with pytest.raises(ValueError, match="System prompt is required"):
            VerificationConfig(
                answering_models=[
                    ModelConfig(
                        id="test-model",
                        model_provider="openai",
                        model_name="gpt-3.5-turbo",
                        temperature=0.1,
                        interface=INTERFACE_LANGCHAIN,
                        system_prompt="",  # Empty system prompt should fail
                    )
                ],
                parsing_models=[
                    ModelConfig(
                        id="parsing-model",
                        model_provider="openai",
                        model_name="gpt-3.5-turbo",
                        temperature=0.1,
                        interface=INTERFACE_LANGCHAIN,
                        system_prompt="You are a validator.",
                    )
                ],
            )

    def test_no_answering_models(self) -> None:
        """Test that configuration without answering models fails."""
        with pytest.raises(ValueError, match="At least one answering model must be configured"):
            VerificationConfig(
                answering_models=[],  # Empty answering models should fail
                parsing_models=[
                    ModelConfig(
                        id="parsing-model",
                        model_provider="openai",
                        model_name="gpt-3.5-turbo",
                        temperature=0.1,
                        interface=INTERFACE_LANGCHAIN,
                        system_prompt="You are a validator.",
                    )
                ],
            )

    def test_no_parsing_models(self) -> None:
        """Test that configuration without parsing models fails."""
        with pytest.raises(ValueError, match="At least one parsing model must be configured"):
            VerificationConfig(
                answering_models=[
                    ModelConfig(
                        id="answering-model",
                        model_provider="openai",
                        model_name="gpt-3.5-turbo",
                        temperature=0.1,
                        interface=INTERFACE_LANGCHAIN,
                        system_prompt="You are a helpful assistant.",
                    )
                ],
                parsing_models=[],  # Empty parsing models should fail
            )

    def test_rubric_enabled_without_parsing_models(self) -> None:
        """Test that rubric-enabled config without parsing models fails."""
        with pytest.raises(ValueError, match="At least one parsing model must be configured"):
            VerificationConfig(
                answering_models=[
                    ModelConfig(
                        id="answering-model",
                        model_provider="openai",
                        model_name="gpt-3.5-turbo",
                        temperature=0.1,
                        interface=INTERFACE_LANGCHAIN,
                        system_prompt="You are a helpful assistant.",
                    )
                ],
                parsing_models=[],  # Empty parsing models with rubric enabled should fail
                rubric_enabled=True,
            )

    def test_rubric_enabled_with_invalid_replicate_count(self) -> None:
        """Test that rubric-enabled config with invalid replicate count fails."""
        with pytest.raises(ValueError, match="Replicate count must be at least 1"):
            VerificationConfig(
                answering_models=[
                    ModelConfig(
                        id="answering-model",
                        model_provider="openai",
                        model_name="gpt-3.5-turbo",
                        temperature=0.1,
                        interface=INTERFACE_LANGCHAIN,
                        system_prompt="You are a helpful assistant.",
                    )
                ],
                parsing_models=[
                    ModelConfig(
                        id="parsing-model",
                        model_provider="openai",
                        model_name="gpt-3.5-turbo",
                        temperature=0.1,
                        interface=INTERFACE_LANGCHAIN,
                        system_prompt="You are a validator.",
                    )
                ],
                rubric_enabled=True,
                replicate_count=0,  # Invalid replicate count
            )

    def test_valid_rubric_configuration(self) -> None:
        """Test valid rubric-enabled configuration."""
        config = VerificationConfig(
            answering_models=[
                ModelConfig(
                    id="answering-model",
                    model_provider="openai",
                    model_name="gpt-3.5-turbo",
                    temperature=0.1,
                    interface=INTERFACE_LANGCHAIN,
                    system_prompt="You are a helpful assistant.",
                )
            ],
            parsing_models=[
                ModelConfig(
                    id="parsing-model",
                    model_provider="openai",
                    model_name="gpt-3.5-turbo",
                    temperature=0.1,
                    interface=INTERFACE_LANGCHAIN,
                    system_prompt="You are a validator.",
                )
            ],
            rubric_enabled=True,
            replicate_count=3,
        )

        # Should not raise any errors
        assert config.rubric_enabled is True
        assert config.replicate_count == 3

    def test_mixed_interface_types(self) -> None:
        """Test configuration with mixed interface types."""
        config = VerificationConfig(
            answering_models=[
                ModelConfig(
                    id="langchain-model",
                    model_provider="openai",
                    model_name="gpt-3.5-turbo",
                    temperature=0.1,
                    interface=INTERFACE_LANGCHAIN,
                    system_prompt="You are a helpful assistant.",
                ),
                ModelConfig(
                    id="openrouter-model",
                    model_provider="",  # Empty provider for openrouter
                    model_name="openrouter/model",
                    temperature=0.2,
                    interface=INTERFACE_OPENROUTER,
                    system_prompt="You are a helpful assistant.",
                ),
            ],
            parsing_models=[
                ModelConfig(
                    id="manual-model",
                    model_provider="",  # Empty provider for manual
                    model_name="manual-model",
                    temperature=0.0,
                    interface=INTERFACE_MANUAL,
                    system_prompt="You are a validator.",
                )
            ],
        )

        # Should not raise any errors
        assert len(config.answering_models) == 2
        assert len(config.parsing_models) == 1
        assert config.answering_models[0].interface == INTERFACE_LANGCHAIN
        assert config.answering_models[1].interface == INTERFACE_OPENROUTER
        assert config.parsing_models[0].interface == INTERFACE_MANUAL

    def test_interfaces_no_provider_required_constant(self) -> None:
        """Test that the INTERFACES_NO_PROVIDER_REQUIRED constant is correct."""
        assert INTERFACES_NO_PROVIDER_REQUIRED == [INTERFACE_OPENROUTER, INTERFACE_MANUAL]
        assert INTERFACE_LANGCHAIN not in INTERFACES_NO_PROVIDER_REQUIRED


class TestLegacyConfigurationSupport:
    """Test backward compatibility with legacy single-model configurations."""

    def test_legacy_answering_model_conversion(self) -> None:
        """Test conversion from legacy answering model fields."""
        config = VerificationConfig(
            answering_model_provider="openai",
            answering_model_name="gpt-3.5-turbo",
            answering_temperature=0.2,
            answering_interface=INTERFACE_LANGCHAIN,
            answering_system_prompt="Custom answering prompt",
            parsing_models=[
                ModelConfig(
                    id="parsing-model",
                    model_provider="openai",
                    model_name="gpt-3.5-turbo",
                    temperature=0.1,
                    interface=INTERFACE_LANGCHAIN,
                    system_prompt="You are a validator.",
                )
            ],
        )

        # Should convert legacy fields to answering_models array
        assert len(config.answering_models) == 1
        assert config.answering_models[0].id == "answering-legacy"
        assert config.answering_models[0].model_provider == "openai"
        assert config.answering_models[0].model_name == "gpt-3.5-turbo"
        assert config.answering_models[0].temperature == 0.2
        assert config.answering_models[0].interface == INTERFACE_LANGCHAIN
        assert config.answering_models[0].system_prompt == "Custom answering prompt"

    def test_legacy_parsing_model_conversion(self) -> None:
        """Test conversion from legacy parsing model fields."""
        config = VerificationConfig(
            answering_models=[
                ModelConfig(
                    id="answering-model",
                    model_provider="openai",
                    model_name="gpt-3.5-turbo",
                    temperature=0.1,
                    interface=INTERFACE_LANGCHAIN,
                    system_prompt="You are a helpful assistant.",
                )
            ],
            parsing_model_provider="anthropic",
            parsing_model_name="claude-3-haiku",
            parsing_temperature=0.0,
            parsing_interface=INTERFACE_MANUAL,
            parsing_system_prompt="Custom parsing prompt",
        )

        # Should convert legacy fields to parsing_models array
        assert len(config.parsing_models) == 1
        assert config.parsing_models[0].id == "parsing-legacy"
        assert config.parsing_models[0].model_provider == "anthropic"
        assert config.parsing_models[0].model_name == "claude-3-haiku"
        assert config.parsing_models[0].temperature == 0.0
        assert config.parsing_models[0].interface == INTERFACE_MANUAL
        assert config.parsing_models[0].system_prompt == "Custom parsing prompt"

    def test_legacy_defaults(self) -> None:
        """Test that legacy conversion uses appropriate defaults."""
        config = VerificationConfig(
            answering_model_provider="openai",
            answering_model_name="gpt-3.5-turbo",
            # Other fields use defaults
            parsing_model_provider="openai",
            parsing_model_name="gpt-3.5-turbo",
        )

        answering_model = config.answering_models[0]
        parsing_model = config.parsing_models[0]

        # Check defaults
        assert answering_model.temperature == 0.1
        assert answering_model.interface == INTERFACE_LANGCHAIN
        assert "expert assistant" in answering_model.system_prompt.lower()

        assert parsing_model.temperature == 0.1
        assert parsing_model.interface == INTERFACE_LANGCHAIN
        assert "validation assistant" in parsing_model.system_prompt.lower()
