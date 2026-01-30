"""Tests for ClaudeToolParserAdapter.

Tests parser adapter functionality for structured output extraction.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel, Field

from karenina.adapters.claude_tool import ClaudeToolParserAdapter
from karenina.ports import LLMResponse, Message, ParseError


class SampleParserSchema(BaseModel):
    """Sample schema for parser tests."""

    gene_name: str = Field(description="The gene mentioned")
    is_oncogene: bool = Field(description="Whether it's an oncogene")


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


class TestParserAdapterInitialization:
    """Tests for parser adapter initialization."""

    def test_initialization_creates_llm_adapter(self, model_config: Any) -> None:
        """Test initialization creates internal LLM adapter."""
        adapter = ClaudeToolParserAdapter(model_config)

        assert adapter._llm_adapter is not None

    def test_initialization_stores_config(self, model_config: Any) -> None:
        """Test initialization stores config."""
        adapter = ClaudeToolParserAdapter(model_config)

        assert adapter._config == model_config

    def test_default_max_retries(self, model_config: Any) -> None:
        """Test default max_retries is 2."""
        adapter = ClaudeToolParserAdapter(model_config)

        assert adapter._max_retries == 2

    def test_custom_max_retries(self, model_config: Any) -> None:
        """Test custom max_retries."""
        adapter = ClaudeToolParserAdapter(model_config, max_retries=5)

        assert adapter._max_retries == 5


class TestParserAdapterAsyncParsing:
    """Tests for async parsing functionality."""

    @pytest.mark.asyncio
    async def test_aparse_to_pydantic_returns_schema_instance(self, model_config: Any) -> None:
        """Test aparse_to_pydantic returns instance of schema."""
        adapter = ClaudeToolParserAdapter(model_config)

        # Mock the LLM adapter's structured output
        mock_parsed = SampleParserSchema(gene_name="BCL2", is_oncogene=False)
        mock_response = LLMResponse(content="", usage=None, raw=mock_parsed)

        with patch.object(adapter, "_llm_adapter") as mock_llm:
            mock_structured = AsyncMock()
            mock_structured.ainvoke.return_value = mock_response
            mock_llm.with_structured_output.return_value = mock_structured

            result = await adapter.aparse_to_pydantic(
                [Message.system("Extract"), Message.user("BCL2 is an anti-apoptotic gene")],
                SampleParserSchema,
            )

            assert isinstance(result, SampleParserSchema)
            assert result.gene_name == "BCL2"
            assert result.is_oncogene is False

    @pytest.mark.asyncio
    async def test_aparse_to_pydantic_passes_max_retries(self, model_config: Any) -> None:
        """Test aparse_to_pydantic passes max_retries to structured adapter."""
        adapter = ClaudeToolParserAdapter(model_config, max_retries=5)

        mock_parsed = SampleParserSchema(gene_name="BCL2", is_oncogene=False)
        mock_response = LLMResponse(content="", usage=None, raw=mock_parsed)

        with patch.object(adapter, "_llm_adapter") as mock_llm:
            mock_structured = AsyncMock()
            mock_structured.ainvoke.return_value = mock_response
            mock_llm.with_structured_output.return_value = mock_structured

            await adapter.aparse_to_pydantic([Message.user("response")], SampleParserSchema)

            mock_llm.with_structured_output.assert_called_once_with(
                SampleParserSchema,
                max_retries=5,
            )

    @pytest.mark.asyncio
    async def test_aparse_to_pydantic_raises_parse_error_on_wrong_type(self, model_config: Any) -> None:
        """Test aparse_to_pydantic raises ParseError when result is wrong type."""

        class OtherSchema(BaseModel):
            other: str

        adapter = ClaudeToolParserAdapter(model_config)

        # Return wrong type
        mock_response = LLMResponse(content="", usage=None, raw=OtherSchema(other="value"))

        with patch.object(adapter, "_llm_adapter") as mock_llm:
            mock_structured = AsyncMock()
            mock_structured.ainvoke.return_value = mock_response
            mock_llm.with_structured_output.return_value = mock_structured

            with pytest.raises(ParseError, match="did not return SampleParserSchema"):
                await adapter.aparse_to_pydantic([Message.user("response")], SampleParserSchema)

    @pytest.mark.asyncio
    async def test_aparse_to_pydantic_raises_parse_error_on_exception(self, model_config: Any) -> None:
        """Test aparse_to_pydantic raises ParseError on underlying exception."""
        adapter = ClaudeToolParserAdapter(model_config)

        with patch.object(adapter, "_llm_adapter") as mock_llm:
            mock_structured = AsyncMock()
            mock_structured.ainvoke.side_effect = Exception("API error")
            mock_llm.with_structured_output.return_value = mock_structured

            with pytest.raises(ParseError, match="Failed to parse response"):
                await adapter.aparse_to_pydantic([Message.user("response")], SampleParserSchema)

    @pytest.mark.asyncio
    async def test_aparse_to_pydantic_preserves_parse_error(self, model_config: Any) -> None:
        """Test aparse_to_pydantic preserves ParseError without wrapping."""
        adapter = ClaudeToolParserAdapter(model_config)

        original_error = ParseError("Original parse error")

        with patch.object(adapter, "_llm_adapter") as mock_llm:
            mock_structured = AsyncMock()
            mock_structured.ainvoke.side_effect = original_error
            mock_llm.with_structured_output.return_value = mock_structured

            with pytest.raises(ParseError, match="Original parse error"):
                await adapter.aparse_to_pydantic([Message.user("response")], SampleParserSchema)


class TestParserAdapterSyncParsing:
    """Tests for sync parsing functionality."""

    def test_parse_to_pydantic_exists(self, model_config: Any) -> None:
        """Test parse_to_pydantic method exists."""
        adapter = ClaudeToolParserAdapter(model_config)

        assert hasattr(adapter, "parse_to_pydantic")
        assert callable(adapter.parse_to_pydantic)


class TestParserAdapterWithDifferentSchemas:
    """Tests for parsing with different schema types."""

    @pytest.mark.asyncio
    async def test_parsing_with_optional_fields(self, model_config: Any) -> None:
        """Test parsing schema with optional fields."""

        class OptionalSchema(BaseModel):
            required_field: str
            optional_field: str | None = None

        adapter = ClaudeToolParserAdapter(model_config)

        mock_parsed = OptionalSchema(required_field="value", optional_field=None)
        mock_response = LLMResponse(content="", usage=None, raw=mock_parsed)

        with patch.object(adapter, "_llm_adapter") as mock_llm:
            mock_structured = AsyncMock()
            mock_structured.ainvoke.return_value = mock_response
            mock_llm.with_structured_output.return_value = mock_structured

            result = await adapter.aparse_to_pydantic([Message.user("response")], OptionalSchema)

            assert result.required_field == "value"
            assert result.optional_field is None

    @pytest.mark.asyncio
    async def test_parsing_with_numeric_fields(self, model_config: Any) -> None:
        """Test parsing schema with numeric fields."""

        class NumericSchema(BaseModel):
            count: int
            score: float

        adapter = ClaudeToolParserAdapter(model_config)

        mock_parsed = NumericSchema(count=42, score=0.95)
        mock_response = LLMResponse(content="", usage=None, raw=mock_parsed)

        with patch.object(adapter, "_llm_adapter") as mock_llm:
            mock_structured = AsyncMock()
            mock_structured.ainvoke.return_value = mock_response
            mock_llm.with_structured_output.return_value = mock_structured

            result = await adapter.aparse_to_pydantic([Message.user("response")], NumericSchema)

            assert result.count == 42
            assert result.score == 0.95

    @pytest.mark.asyncio
    async def test_parsing_with_list_fields(self, model_config: Any) -> None:
        """Test parsing schema with list fields."""

        class ListSchema(BaseModel):
            items: list[str]

        adapter = ClaudeToolParserAdapter(model_config)

        mock_parsed = ListSchema(items=["a", "b", "c"])
        mock_response = LLMResponse(content="", usage=None, raw=mock_parsed)

        with patch.object(adapter, "_llm_adapter") as mock_llm:
            mock_structured = AsyncMock()
            mock_structured.ainvoke.return_value = mock_response
            mock_llm.with_structured_output.return_value = mock_structured

            result = await adapter.aparse_to_pydantic([Message.user("response")], ListSchema)

            assert result.items == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_parsing_with_nested_schema(self, model_config: Any) -> None:
        """Test parsing schema with nested objects."""

        class InnerSchema(BaseModel):
            value: str

        class OuterSchema(BaseModel):
            inner: InnerSchema

        adapter = ClaudeToolParserAdapter(model_config)

        mock_parsed = OuterSchema(inner=InnerSchema(value="nested"))
        mock_response = LLMResponse(content="", usage=None, raw=mock_parsed)

        with patch.object(adapter, "_llm_adapter") as mock_llm:
            mock_structured = AsyncMock()
            mock_structured.ainvoke.return_value = mock_response
            mock_llm.with_structured_output.return_value = mock_structured

            result = await adapter.aparse_to_pydantic([Message.user("response")], OuterSchema)

            assert result.inner.value == "nested"
