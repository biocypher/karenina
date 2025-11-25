"""Tests for Native SDK interface feature.

This module tests the native_sdk interface that bypasses LangChain
and uses OpenAI/Anthropic SDKs directly.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from karenina.schemas import ModelConfig, VerificationConfig
from karenina.schemas.workflow.models import INTERFACE_LANGCHAIN, INTERFACE_NATIVE_SDK


class TestNativeSdkModelConfigValidation:
    """Test ModelConfig validation for native_sdk interface."""

    def test_valid_native_sdk_openai_config(self) -> None:
        """Test valid native_sdk configuration with OpenAI provider."""
        config = ModelConfig(
            id="native-openai",
            model_provider="openai",
            model_name="gpt-4.1-mini",
            temperature=0.1,
            interface=INTERFACE_NATIVE_SDK,
            system_prompt="You are a helpful assistant.",
        )

        assert config.id == "native-openai"
        assert config.model_provider == "openai"
        assert config.model_name == "gpt-4.1-mini"
        assert config.interface == INTERFACE_NATIVE_SDK

    def test_valid_native_sdk_anthropic_config(self) -> None:
        """Test valid native_sdk configuration with Anthropic provider."""
        config = ModelConfig(
            id="native-anthropic",
            model_provider="anthropic",
            model_name="claude-sonnet-4-20250514",
            temperature=0.0,
            interface=INTERFACE_NATIVE_SDK,
            system_prompt="You are a helpful assistant.",
        )

        assert config.id == "native-anthropic"
        assert config.model_provider == "anthropic"
        assert config.model_name == "claude-sonnet-4-20250514"
        assert config.interface == INTERFACE_NATIVE_SDK

    def test_native_sdk_requires_valid_provider(self) -> None:
        """Test that native_sdk fails with invalid provider."""
        with pytest.raises(ValueError, match="Native SDK interface requires model_provider"):
            ModelConfig(
                id="invalid-native",
                model_provider="google_genai",  # Invalid for native_sdk
                model_name="gemini-2.5-flash",
                temperature=0.1,
                interface=INTERFACE_NATIVE_SDK,
            )

    def test_native_sdk_requires_provider_not_none(self) -> None:
        """Test that native_sdk fails when provider is None."""
        with pytest.raises(ValueError, match="Native SDK interface requires model_provider"):
            ModelConfig(
                id="no-provider-native",
                model_provider=None,  # None not allowed for native_sdk
                model_name="gpt-4.1-mini",
                temperature=0.1,
                interface=INTERFACE_NATIVE_SDK,
            )

    def test_native_sdk_in_verification_config(self) -> None:
        """Test native_sdk models work in VerificationConfig."""
        config = VerificationConfig(
            answering_models=[
                ModelConfig(
                    id="native-answering",
                    model_provider="openai",
                    model_name="gpt-4.1-mini",
                    temperature=0.1,
                    interface=INTERFACE_NATIVE_SDK,
                    system_prompt="You are a helpful assistant.",
                )
            ],
            parsing_models=[
                ModelConfig(
                    id="native-parsing",
                    model_provider="anthropic",
                    model_name="claude-sonnet-4-20250514",
                    temperature=0.0,
                    interface=INTERFACE_NATIVE_SDK,
                    system_prompt="You are a validation assistant.",
                )
            ],
        )

        assert len(config.answering_models) == 1
        assert config.answering_models[0].interface == INTERFACE_NATIVE_SDK
        assert config.parsing_models[0].interface == INTERFACE_NATIVE_SDK

    def test_mixed_interfaces_with_native_sdk(self) -> None:
        """Test configuration mixing native_sdk with other interfaces."""
        config = VerificationConfig(
            answering_models=[
                ModelConfig(
                    id="langchain-model",
                    model_provider="openai",
                    model_name="gpt-4.1-mini",
                    temperature=0.1,
                    interface=INTERFACE_LANGCHAIN,
                    system_prompt="You are a helpful assistant.",
                ),
                ModelConfig(
                    id="native-model",
                    model_provider="anthropic",
                    model_name="claude-sonnet-4-20250514",
                    temperature=0.0,
                    interface=INTERFACE_NATIVE_SDK,
                    system_prompt="You are a helpful assistant.",
                ),
            ],
            parsing_models=[
                ModelConfig(
                    id="parsing-model",
                    model_provider="openai",
                    model_name="gpt-4.1-mini",
                    temperature=0.1,
                    interface=INTERFACE_LANGCHAIN,
                    system_prompt="You are a validator.",
                )
            ],
        )

        assert len(config.answering_models) == 2
        assert config.answering_models[0].interface == INTERFACE_LANGCHAIN
        assert config.answering_models[1].interface == INTERFACE_NATIVE_SDK


class TestNativeSimpleLLM:
    """Test NativeSimpleLLM wrapper class."""

    def test_import_native_simple_llm(self) -> None:
        """Test that NativeSimpleLLM can be imported."""
        from karenina.infrastructure.llm.native_agents import NativeSimpleLLM, NativeSimpleLLMResponse

        assert NativeSimpleLLM is not None
        assert NativeSimpleLLMResponse is not None

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_native_simple_llm_openai_init(self) -> None:
        """Test NativeSimpleLLM initialization with OpenAI."""
        from karenina.infrastructure.llm.native_agents import NativeSimpleLLM

        llm = NativeSimpleLLM(
            provider="openai",
            model="gpt-4.1-mini",
            temperature=0.1,
        )

        assert llm.provider == "openai"
        assert llm.model == "gpt-4.1-mini"
        assert llm.temperature == 0.1

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    def test_native_simple_llm_anthropic_init(self) -> None:
        """Test NativeSimpleLLM initialization with Anthropic."""
        from karenina.infrastructure.llm.native_agents import NativeSimpleLLM

        llm = NativeSimpleLLM(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            temperature=0.0,
            system_prompt="You are a helpful assistant.",
        )

        assert llm.provider == "anthropic"
        assert llm.model == "claude-sonnet-4-20250514"
        assert llm.temperature == 0.0
        assert llm.system_prompt == "You are a helpful assistant."

    def test_native_simple_llm_invalid_provider(self) -> None:
        """Test NativeSimpleLLM fails with invalid provider."""
        from karenina.infrastructure.llm.native_agents import NativeSimpleLLM

        with pytest.raises(ValueError, match="Unsupported provider"):
            NativeSimpleLLM(
                provider="invalid_provider",
                model="some-model",
            )

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_native_simple_llm_message_conversion(self) -> None:
        """Test message conversion from LangChain format to native."""
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        from karenina.infrastructure.llm.native_agents import NativeSimpleLLM

        llm = NativeSimpleLLM(provider="openai", model="gpt-4.1-mini")

        # Test with LangChain messages
        messages = [
            SystemMessage(content="You are helpful."),
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
        ]

        native_messages = llm._convert_messages(messages)

        assert len(native_messages) == 3
        assert native_messages[0]["role"] == "system"
        assert native_messages[0]["content"] == "You are helpful."
        assert native_messages[1]["role"] == "user"
        assert native_messages[1]["content"] == "Hello"
        assert native_messages[2]["role"] == "assistant"
        assert native_messages[2]["content"] == "Hi there!"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_native_simple_llm_dict_message_conversion(self) -> None:
        """Test message conversion from dict format."""
        from karenina.infrastructure.llm.native_agents import NativeSimpleLLM

        llm = NativeSimpleLLM(provider="openai", model="gpt-4.1-mini")

        # Test with dict messages (already native format)
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]

        native_messages = llm._convert_messages(messages)

        assert len(native_messages) == 2
        assert native_messages[0] == messages[0]
        assert native_messages[1] == messages[1]


class TestInitChatModelUnifiedNativeSdk:
    """Test init_chat_model_unified routing for native_sdk interface."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_native_sdk_returns_native_simple_llm(self) -> None:
        """Test that native_sdk interface without MCP returns NativeSimpleLLM."""
        from karenina.infrastructure.llm.interface import init_chat_model_unified
        from karenina.infrastructure.llm.native_agents import NativeSimpleLLM

        model = init_chat_model_unified(
            model="gpt-4.1-mini",
            provider="openai",
            interface="native_sdk",
            temperature=0.1,
        )

        assert isinstance(model, NativeSimpleLLM)
        assert model.provider == "openai"
        assert model.model == "gpt-4.1-mini"

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    def test_native_sdk_anthropic_returns_native_simple_llm(self) -> None:
        """Test that native_sdk with anthropic returns NativeSimpleLLM."""
        from karenina.infrastructure.llm.interface import init_chat_model_unified
        from karenina.infrastructure.llm.native_agents import NativeSimpleLLM

        model = init_chat_model_unified(
            model="claude-sonnet-4-20250514",
            provider="anthropic",
            interface="native_sdk",
            temperature=0.0,
            system_prompt="You are a helpful assistant.",
        )

        assert isinstance(model, NativeSimpleLLM)
        assert model.provider == "anthropic"
        assert model.system_prompt == "You are a helpful assistant."

    def test_native_sdk_invalid_provider_raises(self) -> None:
        """Test that native_sdk with invalid provider raises ValueError."""
        from karenina.infrastructure.llm.interface import init_chat_model_unified

        with pytest.raises(ValueError, match="Native SDK interface requires provider"):
            init_chat_model_unified(
                model="some-model",
                provider="google_genai",  # Invalid for native_sdk
                interface="native_sdk",
            )

    def test_native_sdk_none_provider_raises(self) -> None:
        """Test that native_sdk with None provider raises ValueError."""
        from karenina.infrastructure.llm.interface import init_chat_model_unified

        with pytest.raises(ValueError, match="Native SDK interface requires provider"):
            init_chat_model_unified(
                model="some-model",
                provider=None,
                interface="native_sdk",
            )


class TestNativeSimpleLLMInvoke:
    """Test NativeSimpleLLM invoke methods with mocked API calls."""

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    async def test_ainvoke_openai(self) -> None:
        """Test async invoke with mocked OpenAI API."""
        from karenina.infrastructure.llm.native_agents import NativeSimpleLLM

        llm = NativeSimpleLLM(provider="openai", model="gpt-4.1-mini")

        # Mock the OpenAI client response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Hello! I'm doing well."
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30

        llm.client.chat.completions.create = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": "Hello, how are you?"}]
        result = await llm.ainvoke(messages)

        assert result.content == "Hello! I'm doing well."
        assert result.response_metadata["usage"]["input_tokens"] == 10
        assert result.response_metadata["usage"]["output_tokens"] == 20

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    async def test_ainvoke_anthropic(self) -> None:
        """Test async invoke with mocked Anthropic API."""
        from karenina.infrastructure.llm.native_agents import NativeSimpleLLM

        llm = NativeSimpleLLM(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            system_prompt="You are helpful.",
        )

        # Mock the Anthropic client response
        mock_block = Mock()
        mock_block.type = "text"
        mock_block.text = "Hi there! How can I help?"

        mock_response = Mock()
        mock_response.content = [mock_block]
        mock_response.usage = Mock()
        mock_response.usage.input_tokens = 15
        mock_response.usage.output_tokens = 25

        llm.client.messages.create = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": "Hello!"}]
        result = await llm.ainvoke(messages)

        assert result.content == "Hi there! How can I help?"
        assert result.response_metadata["usage"]["input_tokens"] == 15
        assert result.response_metadata["usage"]["output_tokens"] == 25

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_sync_invoke(self) -> None:
        """Test synchronous invoke wraps async invoke."""
        from karenina.infrastructure.llm.native_agents import NativeSimpleLLM

        llm = NativeSimpleLLM(provider="openai", model="gpt-4.1-mini")

        # Mock the OpenAI client response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Sync response"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 10
        mock_response.usage.total_tokens = 15

        llm.client.chat.completions.create = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": "Test"}]
        result = llm.invoke(messages)

        assert result.content == "Sync response"


class TestNativeSimpleLLMUsageTracking:
    """Test usage tracking integration with verification pipeline."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_invoke_llm_with_retry_extracts_usage(self) -> None:
        """Test that _invoke_llm_with_retry extracts usage from NativeSimpleLLM."""
        from langchain_core.messages import HumanMessage

        from karenina.benchmark.verification.verification_utils import _invoke_llm_with_retry
        from karenina.infrastructure.llm.native_agents import NativeSimpleLLM

        llm = NativeSimpleLLM(provider="openai", model="gpt-4.1-mini")

        # Mock the OpenAI client response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 45
        mock_response.usage.completion_tokens = 34
        mock_response.usage.total_tokens = 79

        llm.client.chat.completions.create = AsyncMock(return_value=mock_response)

        messages = [HumanMessage(content="Test question")]
        response, recursion_limit, usage_metadata, agent_metrics = _invoke_llm_with_retry(llm, messages, is_agent=False)

        # Verify response
        assert response == "Test response"
        assert recursion_limit is False
        assert agent_metrics is None

        # Verify usage metadata is extracted correctly
        assert usage_metadata is not None
        assert "gpt-4.1-mini" in usage_metadata
        assert usage_metadata["gpt-4.1-mini"]["input_tokens"] == 45
        assert usage_metadata["gpt-4.1-mini"]["output_tokens"] == 34
        assert usage_metadata["gpt-4.1-mini"]["total_tokens"] == 79

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    def test_invoke_llm_with_retry_extracts_usage_anthropic(self) -> None:
        """Test usage extraction for Anthropic NativeSimpleLLM."""
        from langchain_core.messages import HumanMessage

        from karenina.benchmark.verification.verification_utils import _invoke_llm_with_retry
        from karenina.infrastructure.llm.native_agents import NativeSimpleLLM

        llm = NativeSimpleLLM(provider="anthropic", model="claude-sonnet-4-20250514")

        # Mock the Anthropic client response
        mock_block = Mock()
        mock_block.type = "text"
        mock_block.text = "Anthropic response"

        mock_response = Mock()
        mock_response.content = [mock_block]
        mock_response.usage = Mock()
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50

        llm.client.messages.create = AsyncMock(return_value=mock_response)

        messages = [HumanMessage(content="Test question")]
        response, recursion_limit, usage_metadata, agent_metrics = _invoke_llm_with_retry(llm, messages, is_agent=False)

        # Verify usage metadata is extracted correctly
        assert usage_metadata is not None
        assert "claude-sonnet-4-20250514" in usage_metadata
        assert usage_metadata["claude-sonnet-4-20250514"]["input_tokens"] == 100
        assert usage_metadata["claude-sonnet-4-20250514"]["output_tokens"] == 50
