"""Validation tests for ModelConfig Anthropic connection settings."""

from __future__ import annotations

import pytest

from karenina.schemas.config import ModelConfig


@pytest.mark.unit
class TestAnthropicConnectionFieldValidation:
    @pytest.mark.parametrize("interface", ["openai_endpoint", "openrouter"])
    def test_rejects_anthropic_fields_for_unsupported_interface(self, interface: str) -> None:
        with pytest.raises(ValueError, match="only supported for interfaces"):
            ModelConfig(
                id="model",
                model_name="model",
                interface=interface,
                anthropic_base_url="https://gateway.example/anthropic",
            )

    @pytest.mark.parametrize("interface", ["langchain", "langchain_deep_agents"])
    def test_requires_anthropic_provider_for_langchain_interfaces(self, interface: str) -> None:
        with pytest.raises(ValueError, match="require model_provider='anthropic'"):
            ModelConfig(
                id="model",
                model_name="model",
                model_provider="openai",
                interface=interface,
                anthropic_api_key="gateway-key",
            )

    @pytest.mark.parametrize("field", ["base_url", "api_key"])
    def test_rejects_conflicting_extra_kwargs(self, field: str) -> None:
        kwargs = {"anthropic_base_url": "https://gateway.example/anthropic"}
        if field == "api_key":
            kwargs = {"anthropic_api_key": "gateway-key"}

        with pytest.raises(ValueError, match="conflict with extra_kwargs"):
            ModelConfig(
                id="model",
                model_name="model",
                model_provider="anthropic",
                interface="langchain",
                extra_kwargs={field: "different-value"},
                **kwargs,
            )

    @pytest.mark.parametrize("interface", ["claude_tool", "claude_agent_sdk", "langchain", "langchain_deep_agents"])
    def test_allows_anthropic_fields_for_supported_interface(self, interface: str) -> None:
        config = ModelConfig(
            id="model",
            model_name="model",
            model_provider="anthropic",
            interface=interface,
            anthropic_base_url="https://gateway.example/anthropic",
            anthropic_api_key="gateway-key",
        )

        assert config.anthropic_base_url == "https://gateway.example/anthropic"
