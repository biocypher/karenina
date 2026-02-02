"""Tests for ChatOpenRouter and ChatOpenAIEndpoint custom model classes."""

from __future__ import annotations

from unittest.mock import patch

import pytest


class TestChatOpenRouter:
    """Tests for ChatOpenRouter custom model class."""

    def test_chat_openrouter_initialization(self) -> None:
        """Test ChatOpenRouter initialization."""
        from karenina.adapters.langchain.models import ChatOpenRouter

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            model = ChatOpenRouter(model="gpt-4", temperature=0.5)

            assert model.model_name == "gpt-4"
            assert model.temperature == 0.5

    def test_chat_openrouter_lc_secrets(self) -> None:
        """Test ChatOpenRouter lc_secrets property."""
        from karenina.adapters.langchain.models import ChatOpenRouter

        model = ChatOpenRouter(model="gpt-4", openai_api_key="test-key")
        secrets = model.lc_secrets
        assert secrets == {"openai_api_key": "OPENROUTER_API_KEY"}


class TestChatOpenAIEndpoint:
    """Tests for ChatOpenAIEndpoint custom model class."""

    def test_chat_openai_endpoint_requires_api_key(self) -> None:
        """Test that ChatOpenAIEndpoint requires an API key."""
        from karenina.adapters.langchain.models import ChatOpenAIEndpoint

        with pytest.raises(ValueError, match="API key is required"):
            ChatOpenAIEndpoint(base_url="http://localhost:8000")

    def test_chat_openai_endpoint_requires_explicit_api_key(self) -> None:
        """Test that ChatOpenAIEndpoint does NOT read from environment."""
        from karenina.adapters.langchain.models import ChatOpenAIEndpoint

        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}),
            pytest.raises(ValueError, match="API key is required"),
        ):
            ChatOpenAIEndpoint(base_url="http://localhost:8000")

    def test_chat_openai_endpoint_initialization_with_key(self) -> None:
        """Test ChatOpenAIEndpoint initialization with explicit API key."""
        from karenina.adapters.langchain.models import ChatOpenAIEndpoint

        model = ChatOpenAIEndpoint(
            base_url="http://localhost:8000",
            openai_api_key="explicit-key",
        )

        assert model is not None

    def test_chat_openai_endpoint_lc_secrets_empty(self) -> None:
        """Test ChatOpenAIEndpoint lc_secrets returns empty dict."""
        from karenina.adapters.langchain.models import ChatOpenAIEndpoint

        model = ChatOpenAIEndpoint(
            base_url="http://localhost:8000",
            openai_api_key="test-key",
        )
        secrets = model.lc_secrets
        assert secrets == {}
