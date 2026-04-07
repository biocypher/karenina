"""Tests for LangChainParserAdapter.

Tests structured output parsing and transient error propagation.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from karenina.adapters.langchain import LangChainParserAdapter
from karenina.ports import LLMResponse, Message, UsageMetadata


class SimpleSchema(BaseModel):
    """Simple schema used across test classes."""

    name: str
    value: int


class TestLangChainParserAdapter:
    """Tests for LangChainParserAdapter."""

    @pytest.fixture
    def model_config(self) -> Any:
        """Create a mock ModelConfig."""
        from karenina.schemas.config import ModelConfig

        return ModelConfig(
            id="test-parser",
            model_name="claude-sonnet-4-20250514",
            model_provider="anthropic",
            interface="langchain",
        )

    def test_parser_initialization(self, model_config: Any) -> None:
        """Test parser adapter can be initialized."""
        with patch("karenina.adapters.langchain.initialization.init_chat_model") as mock_init:
            mock_init.return_value = MagicMock()
            parser = LangChainParserAdapter(model_config)

            assert parser._config == model_config
            assert parser._llm_adapter is not None

    def test_json_extraction_with_markdown_fences(self) -> None:
        """Test JSON extraction from markdown fences via shared utility."""
        from karenina.utils.json_extraction import extract_json_from_response

        # Test with json fence
        text = '```json\n{"value": "test"}\n```'
        result = extract_json_from_response(text)
        assert result == '{"value": "test"}'

        # Test without fence (direct JSON)
        text = '{"value": "test"}'
        result = extract_json_from_response(text)
        assert result == '{"value": "test"}'

    def test_json_extraction_from_mixed_text(self) -> None:
        """Test JSON extraction from mixed text via shared utility."""
        from karenina.utils.json_extraction import extract_json_from_response

        # Test embedded JSON
        text = 'Here is the result: {"gene": "BCL2", "score": 0.95} as requested.'
        result = extract_json_from_response(text)
        assert '{"gene": "BCL2", "score": 0.95}' in result

        # Test no JSON
        text = "No JSON here"
        with pytest.raises(ValueError, match="Could not extract JSON"):
            extract_json_from_response(text)

    def test_parse_response_content_direct_json(self, model_config: Any) -> None:
        """Test direct JSON parsing."""
        with patch("karenina.adapters.langchain.initialization.init_chat_model") as mock_init:
            mock_init.return_value = MagicMock()
            parser = LangChainParserAdapter(model_config)

            class SimpleSchema(BaseModel):
                name: str
                value: int

            content = '{"name": "test", "value": 42}'
            result = parser._parse_response_content(content, SimpleSchema)

            assert result.name == "test"
            assert result.value == 42

    def test_parse_response_content_with_markdown_fences(self, model_config: Any) -> None:
        """Test parsing JSON wrapped in markdown fences."""
        with patch("karenina.adapters.langchain.initialization.init_chat_model") as mock_init:
            mock_init.return_value = MagicMock()
            parser = LangChainParserAdapter(model_config)

            class SimpleSchema(BaseModel):
                name: str

            content = '```json\n{"name": "test"}\n```'
            result = parser._parse_response_content(content, SimpleSchema)

            assert result.name == "test"


@pytest.mark.unit
class TestTransientErrorPropagation:
    """Tests for transient error propagation in parser strategies."""

    @pytest.fixture
    def model_config(self) -> Any:
        """Create a mock ModelConfig."""
        from karenina.schemas.config import ModelConfig

        return ModelConfig(
            id="test-parser",
            model_name="claude-sonnet-4-20250514",
            model_provider="anthropic",
            interface="langchain",
        )

    @pytest.fixture
    def parser(self, model_config: Any) -> LangChainParserAdapter:
        """Create a parser with mocked LLM initialization."""
        with patch("karenina.adapters.langchain.initialization.init_chat_model") as mock_init:
            mock_init.return_value = MagicMock()
            return LangChainParserAdapter(model_config)

    @pytest.fixture
    def messages(self) -> list[Message]:
        """Create sample messages for parsing."""
        return [
            Message.system("Parse the following into JSON."),
            Message.user('The name is "test" and the value is 42.'),
        ]

    async def test_strategy1_transient_error_propagates(
        self, parser: LangChainParserAdapter, messages: list[Message]
    ) -> None:
        """Transient error in Strategy 1 re-raises without trying Strategy 2."""
        # Make with_structured_output return an adapter whose ainvoke raises TimeoutError
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(side_effect=TimeoutError("connection timed out"))
        parser._llm_adapter.with_structured_output = MagicMock(return_value=mock_structured)

        # Strategy 2's ainvoke should NOT be called
        parser._llm_adapter.ainvoke = AsyncMock(side_effect=AssertionError("Strategy 2 should not be reached"))

        with pytest.raises(TimeoutError, match="connection timed out"):
            await parser.aparse_to_pydantic(messages, SimpleSchema)

        # Confirm Strategy 1 was attempted exactly once
        mock_structured.ainvoke.assert_called_once()
        # Confirm Strategy 2 was NOT attempted
        parser._llm_adapter.ainvoke.assert_not_called()

    async def test_strategy1_parse_error_falls_through_to_strategy2(
        self, parser: LangChainParserAdapter, messages: list[Message]
    ) -> None:
        """Parse error in Strategy 1 falls through to Strategy 2."""
        # Strategy 1: structured output raises a non-transient parse error
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(side_effect=ValueError("invalid JSON"))
        parser._llm_adapter.with_structured_output = MagicMock(return_value=mock_structured)

        # Strategy 2: regular ainvoke returns valid JSON
        strategy2_response = LLMResponse(
            content='{"name": "test", "value": 42}',
            raw='{"name": "test", "value": 42}',
            usage=UsageMetadata(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        parser._llm_adapter.ainvoke = AsyncMock(return_value=strategy2_response)

        result = await parser.aparse_to_pydantic(messages, SimpleSchema)

        # Both strategies were tried
        mock_structured.ainvoke.assert_called_once()
        parser._llm_adapter.ainvoke.assert_called_once()
        assert result.parsed.name == "test"
        assert result.parsed.value == 42

    async def test_retry_with_null_feedback_propagates_transient_error(
        self, parser: LangChainParserAdapter, messages: list[Message]
    ) -> None:
        """_retry_with_null_feedback re-raises transient errors."""
        # Mock ainvoke to raise a transient error
        parser._llm_adapter.ainvoke = AsyncMock(side_effect=ConnectionError("connection reset by peer"))

        # Craft an error whose string contains embedded JSON with null values,
        # so _extract_null_fields_from_error finds null fields and the method
        # proceeds to the LLM call (where our ConnectionError mock fires).
        fake_error = ValueError('Failed to parse from completion {"name": null, "value": 42}. Error details.')

        with pytest.raises(ConnectionError, match="connection reset by peer"):
            await parser._retry_with_null_feedback(
                original_messages=messages,
                failed_response='{"name": null, "value": 42}',
                error=fake_error,
                schema=SimpleSchema,
            )

    async def test_retry_with_format_feedback_propagates_transient_error(
        self, parser: LangChainParserAdapter, messages: list[Message]
    ) -> None:
        """_retry_with_format_feedback re-raises transient errors."""
        from karenina.ports import ParseError

        # Mock ainvoke to raise a transient error
        parser._llm_adapter.ainvoke = AsyncMock(side_effect=TimeoutError("read timed out"))

        # Create a JSON-format-related error so the method does not bail early
        fake_error = ParseError("Invalid JSON output: expecting value")

        with pytest.raises(TimeoutError, match="read timed out"):
            await parser._retry_with_format_feedback(
                original_messages=messages,
                failed_response="This is not JSON at all, just plain text",
                error=fake_error,
                schema=SimpleSchema,
            )

    def test_parser_forwards_retry_policy_to_llm_adapter(self) -> None:
        """Parser passes retry_policy from ModelConfig to its LLM adapter."""
        from karenina.schemas.config import ModelConfig
        from karenina.utils.retry_policy import RetryPolicy

        policy = RetryPolicy()
        config = ModelConfig(
            id="test-retry-config",
            model_name="claude-sonnet-4-20250514",
            model_provider="anthropic",
            interface="langchain",
            retry_policy=policy,
        )

        with patch("karenina.adapters.langchain.initialization.init_chat_model") as mock_init:
            mock_init.return_value = MagicMock()
            parser = LangChainParserAdapter(config)

            assert parser._llm_adapter._retry_policy is not None
