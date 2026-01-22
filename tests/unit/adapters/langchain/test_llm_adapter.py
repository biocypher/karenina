"""Tests for LangChainLLMAdapter.

Tests LLM invocation with mocked model.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from karenina.adapters.langchain import LangChainLLMAdapter
from karenina.ports import Message


class TestLangChainLLMAdapter:
    """Tests for LangChainLLMAdapter."""

    @pytest.fixture
    def model_config(self) -> Any:
        """Create a mock ModelConfig."""
        from karenina.schemas.workflow.models import ModelConfig

        return ModelConfig(
            id="test-model",
            model_name="claude-sonnet-4-20250514",
            model_provider="anthropic",
            interface="langchain",
        )

    def test_adapter_initialization(self, model_config: Any) -> None:
        """Test adapter can be initialized with ModelConfig."""
        with patch("karenina.adapters.langchain.initialization.init_chat_model") as mock_init:
            mock_init.return_value = MagicMock()
            adapter = LangChainLLMAdapter(model_config)

            assert adapter._config == model_config
            assert adapter._converter is not None
            mock_init.assert_called_once()

    @pytest.mark.asyncio
    async def test_ainvoke_basic(self, model_config: Any) -> None:
        """Test basic async invocation."""
        with patch("karenina.adapters.langchain.initialization.init_chat_model") as mock_init:
            mock_response = MagicMock()
            mock_response.content = "Hello! I'm happy to help."
            mock_response.response_metadata = {"usage": {"input_tokens": 10, "output_tokens": 20}}

            mock_model = AsyncMock()
            mock_model.ainvoke = AsyncMock(return_value=mock_response)
            mock_init.return_value = mock_model

            adapter = LangChainLLMAdapter(model_config)
            result = await adapter.ainvoke([Message.user("Hello")])

            assert result.content == "Hello! I'm happy to help."
            assert result.usage.input_tokens == 10
            assert result.usage.output_tokens == 20
            assert result.raw == mock_response

    @pytest.mark.asyncio
    async def test_ainvoke_with_usage_metadata(self, model_config: Any) -> None:
        """Test usage metadata extraction from response."""
        with patch("karenina.adapters.langchain.initialization.init_chat_model") as mock_init:
            mock_response = MagicMock()
            mock_response.content = "Response text"
            mock_response.response_metadata = {
                "token_usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cache_read_input_tokens": 25,
                }
            }

            mock_model = AsyncMock()
            mock_model.ainvoke = AsyncMock(return_value=mock_response)
            mock_init.return_value = mock_model

            adapter = LangChainLLMAdapter(model_config)
            result = await adapter.ainvoke([Message.system("System"), Message.user("User")])

            assert result.usage.input_tokens == 100
            assert result.usage.output_tokens == 50
            assert result.usage.total_tokens == 150
            assert result.usage.cache_read_tokens == 25

    def test_with_structured_output(self, model_config: Any) -> None:
        """Test with_structured_output returns new adapter instance."""
        with patch("karenina.adapters.langchain.initialization.init_chat_model") as mock_init:
            mock_model = MagicMock()
            mock_structured_model = MagicMock()
            mock_model.with_structured_output = MagicMock(return_value=mock_structured_model)
            mock_init.return_value = mock_model

            class TestSchema(BaseModel):
                value: str

            adapter = LangChainLLMAdapter(model_config)
            structured_adapter = adapter.with_structured_output(TestSchema)

            assert structured_adapter is not adapter
            assert structured_adapter._structured_schema == TestSchema
            assert structured_adapter._model == mock_structured_model
