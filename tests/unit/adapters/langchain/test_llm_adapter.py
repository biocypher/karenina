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
from karenina.schemas.config import ModelConfig


@pytest.mark.unit
class TestSDKRetrySuppression:
    """Tests for SDK max_retries=0."""

    def test_initialize_model_passes_max_retries_zero(self) -> None:
        """Test that _initialize_model passes max_retries=0 to suppress SDK retries."""
        config = ModelConfig(
            id="test",
            model_name="test-model",
            model_provider="openai",
            interface="langchain",
        )
        with patch("karenina.adapters.langchain.initialization.init_chat_model") as mock_init:
            mock_init.return_value = MagicMock()
            LangChainLLMAdapter(config)
            _, kwargs = mock_init.call_args
            assert kwargs.get("max_retries") == 0


@pytest.mark.unit
class TestRetryExecutorIntegration:
    """Tests for RetryExecutor replacing tenacity retry."""

    def test_adapter_creates_retry_executor(self) -> None:
        """Test that adapter creates a RetryExecutor on init."""
        from karenina.utils.retry_policy import RetryExecutor

        config = ModelConfig(
            id="test",
            model_name="test-model",
            model_provider="openai",
            interface="langchain",
        )
        with patch("karenina.adapters.langchain.initialization.init_chat_model") as mock_init:
            mock_init.return_value = MagicMock()
            adapter = LangChainLLMAdapter(config)
            assert isinstance(adapter._retry_executor, RetryExecutor)

    def test_default_policy_creates_executor_with_default_retry_policy(self) -> None:
        """Test that None retry_policy creates executor with default RetryPolicy."""
        from karenina.utils.retry_policy import RetryExecutor

        config = ModelConfig(
            id="test",
            model_name="test-model",
            model_provider="openai",
            interface="langchain",
        )
        with patch("karenina.adapters.langchain.initialization.init_chat_model") as mock_init:
            mock_init.return_value = MagicMock()
            adapter = LangChainLLMAdapter(config)
            assert isinstance(adapter._retry_executor, RetryExecutor)

    def test_custom_policy_used_by_executor(self) -> None:
        """Test that custom retry_policy is passed to executor."""
        from karenina.utils.retry_policy import CategoryRetryConfig, RetryExecutor, RetryPolicy

        policy = RetryPolicy(
            connection=CategoryRetryConfig(max_attempts=5),
        )
        config = ModelConfig(
            id="test",
            model_name="test-model",
            model_provider="openai",
            interface="langchain",
            retry_policy=policy,
        )
        with patch("karenina.adapters.langchain.initialization.init_chat_model") as mock_init:
            mock_init.return_value = MagicMock()
            adapter = LangChainLLMAdapter(config)
            assert isinstance(adapter._retry_executor, RetryExecutor)
            assert adapter._retry_executor._policy.connection.max_attempts == 5

    @pytest.mark.asyncio
    async def test_ainvoke_retries_via_executor(self) -> None:
        """Test that ainvoke retries transient errors through RetryExecutor."""
        config = ModelConfig(
            id="test",
            model_name="test-model",
            model_provider="openai",
            interface="langchain",
        )
        with patch("karenina.adapters.langchain.initialization.init_chat_model") as mock_init:
            mock_response = MagicMock()
            mock_response.content = "Success"
            mock_response.response_metadata = {"usage": {"input_tokens": 5, "output_tokens": 5}}

            mock_model = AsyncMock()
            call_count = 0

            async def _fail_then_succeed(*args: Any, **kwargs: Any) -> Any:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise ConnectionError("transient failure")
                return mock_response

            mock_model.ainvoke = _fail_then_succeed
            mock_init.return_value = mock_model

            adapter = LangChainLLMAdapter(config)
            result = await adapter.ainvoke([Message.user("Hello")])

            assert result.content == "Success"
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_ainvoke_raises_permanent_errors(self) -> None:
        """Test that permanent errors are raised immediately without retry."""
        config = ModelConfig(
            id="test",
            model_name="test-model",
            model_provider="openai",
            interface="langchain",
        )
        with patch("karenina.adapters.langchain.initialization.init_chat_model") as mock_init:
            mock_model = AsyncMock()
            mock_model.ainvoke = AsyncMock(side_effect=ValueError("permanent"))
            mock_init.return_value = mock_model

            adapter = LangChainLLMAdapter(config)
            with pytest.raises(ValueError, match="permanent"):
                await adapter.ainvoke([Message.user("Hello")])

    def test_no_invoke_model_with_retry_method(self) -> None:
        """Test that the old _invoke_model_with_retry method is removed."""
        config = ModelConfig(
            id="test",
            model_name="test-model",
            model_provider="openai",
            interface="langchain",
        )
        with patch("karenina.adapters.langchain.initialization.init_chat_model") as mock_init:
            mock_init.return_value = MagicMock()
            adapter = LangChainLLMAdapter(config)
            assert not hasattr(adapter, "_invoke_model_with_retry")


class TestLangChainLLMAdapter:
    """Tests for LangChainLLMAdapter."""

    @pytest.fixture
    def model_config(self) -> Any:
        """Create a mock ModelConfig."""
        from karenina.schemas.config import ModelConfig

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
