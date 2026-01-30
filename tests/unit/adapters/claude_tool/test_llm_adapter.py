"""Tests for ClaudeToolLLMAdapter.

Tests LLM adapter functionality including text and structured output modes.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from karenina.adapters.claude_tool import ClaudeToolLLMAdapter
from karenina.ports import Message, ParseError


class MockUsage:
    """Mock usage object."""

    def __init__(self, input_tokens: int = 100, output_tokens: int = 50) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class MockTextBlock:
    """Mock text content block."""

    def __init__(self, text: str) -> None:
        self.text = text


class MockResponse:
    """Mock Anthropic API response."""

    def __init__(
        self,
        content: list[MockTextBlock] | None = None,
        usage: MockUsage | None = None,
    ) -> None:
        self.content = content or [MockTextBlock("Hello!")]
        self.usage = usage or MockUsage()


@pytest.fixture
def model_config() -> Any:
    """Create a mock ModelConfig for claude_tool interface."""
    from karenina.schemas.workflow.models import ModelConfig

    return ModelConfig(
        id="test-claude-tool",
        model_name="claude-haiku-4-5",
        model_provider="anthropic",
        interface="claude_tool",
        max_tokens=1024,
    )


class TestLLMAdapterTextMode:
    """Tests for LLM adapter text output mode."""

    @pytest.mark.asyncio
    async def test_ainvoke_calls_messages_create(self, model_config: Any) -> None:
        """Test ainvoke calls client.messages.create."""
        adapter = ClaudeToolLLMAdapter(model_config)

        mock_client = AsyncMock()
        mock_client.messages.create.return_value = MockResponse()

        with patch.object(adapter, "_get_async_client", return_value=mock_client):
            result = await adapter.ainvoke([Message.user("Hello")])

            mock_client.messages.create.assert_called_once()
            assert result.content == "Hello!"

    @pytest.mark.asyncio
    async def test_ainvoke_returns_llm_response(self, model_config: Any) -> None:
        """Test ainvoke returns LLMResponse object."""
        from karenina.ports import LLMResponse

        adapter = ClaudeToolLLMAdapter(model_config)

        mock_client = AsyncMock()
        mock_client.messages.create.return_value = MockResponse()

        with patch.object(adapter, "_get_async_client", return_value=mock_client):
            result = await adapter.ainvoke([Message.user("Hello")])

            assert isinstance(result, LLMResponse)
            assert result.content is not None
            assert result.usage is not None

    @pytest.mark.asyncio
    async def test_ainvoke_extracts_text_content(self, model_config: Any) -> None:
        """Test ainvoke extracts text content from response."""
        adapter = ClaudeToolLLMAdapter(model_config)

        mock_response = MockResponse(content=[MockTextBlock("First"), MockTextBlock("Second")])
        mock_client = AsyncMock()
        mock_client.messages.create.return_value = mock_response

        with patch.object(adapter, "_get_async_client", return_value=mock_client):
            result = await adapter.ainvoke([Message.user("Hello")])

            assert result.content == "First\nSecond"

    @pytest.mark.asyncio
    async def test_ainvoke_passes_model_name(self, model_config: Any) -> None:
        """Test ainvoke passes model name to API."""
        adapter = ClaudeToolLLMAdapter(model_config)

        mock_client = AsyncMock()
        mock_client.messages.create.return_value = MockResponse()

        with patch.object(adapter, "_get_async_client", return_value=mock_client):
            await adapter.ainvoke([Message.user("Hello")])

            call_kwargs = mock_client.messages.create.call_args.kwargs
            assert call_kwargs["model"] == "claude-haiku-4-5"

    @pytest.mark.asyncio
    async def test_ainvoke_passes_max_tokens(self, model_config: Any) -> None:
        """Test ainvoke passes max_tokens to API."""
        adapter = ClaudeToolLLMAdapter(model_config)

        mock_client = AsyncMock()
        mock_client.messages.create.return_value = MockResponse()

        with patch.object(adapter, "_get_async_client", return_value=mock_client):
            await adapter.ainvoke([Message.user("Hello")])

            call_kwargs = mock_client.messages.create.call_args.kwargs
            assert call_kwargs["max_tokens"] == 1024

    @pytest.mark.asyncio
    async def test_ainvoke_extracts_system_prompt(self, model_config: Any) -> None:
        """Test ainvoke extracts and passes system prompt."""
        adapter = ClaudeToolLLMAdapter(model_config)

        mock_client = AsyncMock()
        mock_client.messages.create.return_value = MockResponse()

        with patch.object(adapter, "_get_async_client", return_value=mock_client):
            await adapter.ainvoke(
                [
                    Message.system("You are helpful"),
                    Message.user("Hello"),
                ]
            )

            call_kwargs = mock_client.messages.create.call_args.kwargs
            assert "system" in call_kwargs
            # System should be a cached block
            assert call_kwargs["system"][0]["text"] == "You are helpful"

    def test_model_config_requires_model_name(self) -> None:
        """Test ModelConfig validates model_name is required for claude_tool.

        Note: This validation happens at the Pydantic level in ModelConfig,
        not in the adapter itself.
        """
        from pydantic import ValidationError

        from karenina.schemas.workflow.models import ModelConfig

        with pytest.raises(ValidationError, match="model_name is required"):
            ModelConfig(
                id="test",
                model_name=None,
                model_provider="anthropic",
                interface="claude_tool",
            )

    @pytest.mark.asyncio
    async def test_ainvoke_passes_temperature(self) -> None:
        """Test ainvoke passes temperature when specified."""
        from karenina.schemas.workflow.models import ModelConfig

        config = ModelConfig(
            id="test",
            model_name="claude-haiku-4-5",
            model_provider="anthropic",
            interface="claude_tool",
            temperature=0.7,
        )
        adapter = ClaudeToolLLMAdapter(config)

        mock_client = AsyncMock()
        mock_client.messages.create.return_value = MockResponse()

        with patch.object(adapter, "_get_async_client", return_value=mock_client):
            await adapter.ainvoke([Message.user("Hello")])

            call_kwargs = mock_client.messages.create.call_args.kwargs
            assert call_kwargs["temperature"] == 0.7


class TestLLMAdapterStructuredOutput:
    """Tests for LLM adapter structured output mode."""

    def test_with_structured_output_returns_new_adapter(self, model_config: Any) -> None:
        """Test with_structured_output returns new adapter instance."""

        class TestSchema(BaseModel):
            value: str

        adapter = ClaudeToolLLMAdapter(model_config)
        structured = adapter.with_structured_output(TestSchema)

        assert structured is not adapter
        assert structured._structured_schema == TestSchema

    def test_with_structured_output_default_max_retries(self, model_config: Any) -> None:
        """Test with_structured_output uses default max_retries of 3."""

        class TestSchema(BaseModel):
            value: str

        adapter = ClaudeToolLLMAdapter(model_config)
        structured = adapter.with_structured_output(TestSchema)

        assert structured._max_retries == 3

    def test_with_structured_output_custom_max_retries(self, model_config: Any) -> None:
        """Test with_structured_output accepts custom max_retries."""

        class TestSchema(BaseModel):
            value: str

        adapter = ClaudeToolLLMAdapter(model_config)
        structured = adapter.with_structured_output(TestSchema, max_retries=5)

        assert structured._max_retries == 5

    @pytest.mark.asyncio
    async def test_structured_ainvoke_uses_beta_parse(self, model_config: Any) -> None:
        """Test structured output uses beta.messages.parse."""

        class TestSchema(BaseModel):
            value: str

        adapter = ClaudeToolLLMAdapter(model_config)
        structured = adapter.with_structured_output(TestSchema)

        mock_parsed = TestSchema(value="test")
        mock_response = MagicMock()
        mock_response.parsed_output = mock_parsed
        mock_response.usage = MockUsage()

        mock_client = AsyncMock()
        mock_client.beta.messages.parse.return_value = mock_response

        with patch.object(structured, "_get_async_client", return_value=mock_client):
            result = await structured.ainvoke([Message.user("Hello")])

            mock_client.beta.messages.parse.assert_called_once()
            assert result.raw == mock_parsed

    @pytest.mark.asyncio
    async def test_structured_ainvoke_passes_output_format(self, model_config: Any) -> None:
        """Test structured output passes schema as output_format."""

        class TestSchema(BaseModel):
            answer: int

        adapter = ClaudeToolLLMAdapter(model_config)
        structured = adapter.with_structured_output(TestSchema)

        mock_parsed = TestSchema(answer=42)
        mock_response = MagicMock()
        mock_response.parsed_output = mock_parsed
        mock_response.usage = MockUsage()

        mock_client = AsyncMock()
        mock_client.beta.messages.parse.return_value = mock_response

        with patch.object(structured, "_get_async_client", return_value=mock_client):
            await structured.ainvoke([Message.user("What is 6*7?")])

            call_kwargs = mock_client.beta.messages.parse.call_args.kwargs
            assert call_kwargs["output_format"] == TestSchema

    @pytest.mark.asyncio
    async def test_structured_ainvoke_raises_on_missing_parsed(self, model_config: Any) -> None:
        """Test raises ParseError when parsed output is missing."""

        class TestSchema(BaseModel):
            value: str

        adapter = ClaudeToolLLMAdapter(model_config)
        structured = adapter.with_structured_output(TestSchema, max_retries=0)

        mock_response = MagicMock()
        mock_response.parsed_output = None
        mock_response.usage = MockUsage()

        mock_client = AsyncMock()
        mock_client.beta.messages.parse.return_value = mock_response

        with (
            patch.object(structured, "_get_async_client", return_value=mock_client),
            pytest.raises(ParseError, match="did not return parsed_output"),
        ):
            await structured.ainvoke([Message.user("Hello")])

    @pytest.mark.asyncio
    async def test_structured_ainvoke_raises_on_wrong_type(self, model_config: Any) -> None:
        """Test raises ParseError when parsed output is wrong type."""

        class TestSchema(BaseModel):
            value: str

        class OtherSchema(BaseModel):
            other: int

        adapter = ClaudeToolLLMAdapter(model_config)
        structured = adapter.with_structured_output(TestSchema, max_retries=0)

        mock_response = MagicMock()
        mock_response.parsed_output = OtherSchema(other=42)  # Wrong type
        mock_response.usage = MockUsage()

        mock_client = AsyncMock()
        mock_client.beta.messages.parse.return_value = mock_response

        with (
            patch.object(structured, "_get_async_client", return_value=mock_client),
            pytest.raises(ParseError, match="expected TestSchema"),
        ):
            await structured.ainvoke([Message.user("Hello")])


class TestLLMAdapterClientManagement:
    """Tests for LLM adapter client management."""

    def test_client_lazy_initialization(self, model_config: Any) -> None:
        """Test clients are lazily initialized."""
        adapter = ClaudeToolLLMAdapter(model_config)

        assert adapter._client is None
        assert adapter._async_client is None

    def test_get_client_creates_sync_client(self, model_config: Any) -> None:
        """Test _get_client creates sync Anthropic client."""
        adapter = ClaudeToolLLMAdapter(model_config)

        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_instance = MagicMock()
            mock_anthropic.return_value = mock_instance

            client = adapter._get_client()

            mock_anthropic.assert_called_once()
            assert client == mock_instance

    def test_get_client_returns_cached_client(self, model_config: Any) -> None:
        """Test _get_client returns cached client on subsequent calls."""
        adapter = ClaudeToolLLMAdapter(model_config)

        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_instance = MagicMock()
            mock_anthropic.return_value = mock_instance

            client1 = adapter._get_client()
            client2 = adapter._get_client()

            assert client1 is client2
            mock_anthropic.assert_called_once()  # Only one instantiation

    def test_get_async_client_creates_async_client(self, model_config: Any) -> None:
        """Test _get_async_client creates async Anthropic client."""
        adapter = ClaudeToolLLMAdapter(model_config)

        with patch("anthropic.AsyncAnthropic") as mock_async_anthropic:
            mock_instance = MagicMock()
            mock_async_anthropic.return_value = mock_instance

            client = adapter._get_async_client()

            mock_async_anthropic.assert_called_once()
            assert client == mock_instance


class TestLLMAdapterSyncInvoke:
    """Tests for LLM adapter sync invoke method."""

    def test_invoke_exists(self, model_config: Any) -> None:
        """Test invoke method exists."""
        adapter = ClaudeToolLLMAdapter(model_config)

        assert hasattr(adapter, "invoke")
        assert callable(adapter.invoke)
