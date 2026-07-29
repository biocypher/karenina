"""Tests for unified LangChain model initialization."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from karenina.adapters.factory import build_llm_kwargs
from karenina.adapters.langchain.initialization import init_chat_model_unified
from karenina.schemas.config import ModelConfig


@pytest.mark.unit
class TestAnthropicConnectionFields:
    def test_factory_includes_dedicated_connection_settings(self) -> None:
        config = ModelConfig(
            id="gateway-model",
            model_name="gateway-model",
            model_provider="anthropic",
            interface="langchain",
            anthropic_base_url="https://gateway.example/anthropic",
            anthropic_api_key="gateway-key",
        )

        kwargs = build_llm_kwargs(config)

        assert kwargs["anthropic_base_url"] == "https://gateway.example/anthropic"
        assert kwargs["anthropic_api_key"].get_secret_value() == "gateway-key"

    def test_forwards_explicit_anthropic_connection_settings(self) -> None:
        with patch("karenina.adapters.langchain.initialization.init_chat_model") as mock_init:
            mock_init.return_value = MagicMock()

            init_chat_model_unified(
                model="gateway-model",
                provider="anthropic",
                anthropic_base_url="https://gateway.example/anthropic",
                anthropic_api_key=SecretStr("gateway-key"),
            )

        _, kwargs = mock_init.call_args
        assert kwargs["base_url"] == "https://gateway.example/anthropic"
        assert kwargs["api_key"].get_secret_value() == "gateway-key"

    def test_does_not_forward_anthropic_settings_to_another_provider(self) -> None:
        with patch("karenina.adapters.langchain.initialization.init_chat_model") as mock_init:
            mock_init.return_value = MagicMock()

            init_chat_model_unified(
                model="model",
                provider="openai",
                anthropic_base_url="https://gateway.example/anthropic",
                anthropic_api_key="gateway-key",
            )

        _, kwargs = mock_init.call_args
        assert "base_url" not in kwargs
        assert "api_key" not in kwargs
