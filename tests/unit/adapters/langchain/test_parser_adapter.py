"""Tests for LangChainParserAdapter.

Tests structured output parsing.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from karenina.adapters.langchain import LangChainParserAdapter


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
