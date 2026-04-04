"""Tests for extract_usage_from_chunk from the LangChain usage module.

Tests the streaming-specific usage extraction path that processes
AIMessageChunk objects during LangChain streaming.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from karenina.adapters.langchain.usage import extract_usage_from_chunk


@pytest.mark.unit
class TestExtractUsageFromChunk:
    """Tests for extract_usage_from_chunk()."""

    def test_extracts_usage_from_dict_metadata(self) -> None:
        """Chunk with usage_metadata dict yields correct token counts."""
        chunk = MagicMock()
        chunk.usage_metadata = {
            "input_tokens": 150,
            "output_tokens": 75,
        }

        result = extract_usage_from_chunk(chunk, model_name="claude-sonnet-4-20250514")

        assert result.input_tokens == 150
        assert result.output_tokens == 75
        assert result.total_tokens == 225
        assert result.model == "claude-sonnet-4-20250514"

    def test_extracts_cache_tokens(self) -> None:
        """Chunk with cache token fields populates cache metadata."""
        chunk = MagicMock()
        chunk.usage_metadata = {
            "input_tokens": 200,
            "output_tokens": 100,
            "cache_read_input_tokens": 80,
            "cache_creation_input_tokens": 20,
        }

        result = extract_usage_from_chunk(chunk)

        assert result.cache_read_tokens == 80
        assert result.cache_creation_tokens == 20

    def test_returns_zero_usage_when_no_metadata(self) -> None:
        """Chunk without usage_metadata returns zero token counts."""
        chunk = MagicMock(spec=["content"])
        # spec=["content"] means usage_metadata attribute does not exist

        result = extract_usage_from_chunk(chunk, model_name="test-model")

        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.total_tokens == 0
        assert result.model == "test-model"

    def test_returns_zero_usage_when_metadata_is_none(self) -> None:
        """Chunk with None usage_metadata returns zero token counts."""
        chunk = MagicMock()
        chunk.usage_metadata = None

        result = extract_usage_from_chunk(chunk)

        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.total_tokens == 0

    def test_handles_object_style_usage_metadata(self) -> None:
        """Chunk with attribute-based usage_metadata (non-dict) extracts correctly."""
        # LangChain sometimes returns an object instead of a dict
        usage_obj = MagicMock()
        usage_obj.input_tokens = 300
        usage_obj.output_tokens = 150
        usage_obj.cache_read_input_tokens = None
        usage_obj.cache_creation_input_tokens = None

        # Make it fail isinstance(um, dict) check
        chunk = MagicMock()
        chunk.usage_metadata = usage_obj

        result = extract_usage_from_chunk(chunk, model_name="gpt-4o")

        assert result.input_tokens == 300
        assert result.output_tokens == 150
        assert result.model == "gpt-4o"
