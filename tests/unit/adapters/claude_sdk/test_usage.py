"""Tests for SDK usage metadata extraction.

Tests extract_sdk_usage function.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from karenina.adapters.claude_agent_sdk import extract_sdk_usage
from karenina.ports import UsageMetadata


class TestExtractSdkUsage:
    """Tests for extract_sdk_usage function."""

    def test_basic_usage_extraction(self) -> None:
        """Test basic usage extraction from mock ResultMessage."""
        # Create mock ResultMessage
        mock_result = MagicMock()
        mock_result.usage = {
            "input_tokens": 100,
            "output_tokens": 50,
        }
        mock_result.total_cost_usd = 0.0015

        usage = extract_sdk_usage(mock_result)

        assert isinstance(usage, UsageMetadata)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150
        assert usage.cost_usd == 0.0015

    def test_usage_with_cache_tokens(self) -> None:
        """Test usage extraction with cache tokens."""
        mock_result = MagicMock()
        mock_result.usage = {
            "input_tokens": 200,
            "output_tokens": 100,
            "cache_read_input_tokens": 150,
            "cache_creation_input_tokens": 25,
        }
        mock_result.total_cost_usd = 0.003

        usage = extract_sdk_usage(mock_result)

        assert usage.input_tokens == 200
        assert usage.output_tokens == 100
        assert usage.total_tokens == 300
        assert usage.cache_read_tokens == 150
        assert usage.cache_creation_tokens == 25
        assert usage.cost_usd == 0.003

    def test_usage_with_model(self) -> None:
        """Test usage extraction with model parameter."""
        mock_result = MagicMock()
        mock_result.usage = {"input_tokens": 50, "output_tokens": 25}
        mock_result.total_cost_usd = None

        usage = extract_sdk_usage(mock_result, model="claude-sonnet-4-20250514")

        assert usage.model == "claude-sonnet-4-20250514"

    def test_usage_with_missing_data(self) -> None:
        """Test usage extraction handles missing data gracefully."""
        mock_result = MagicMock()
        mock_result.usage = None  # No usage data
        mock_result.total_cost_usd = None

        usage = extract_sdk_usage(mock_result)

        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0
        assert usage.cost_usd is None
        assert usage.cache_read_tokens is None
        assert usage.cache_creation_tokens is None

    def test_usage_with_empty_usage_dict(self) -> None:
        """Test usage extraction with empty usage dict."""
        mock_result = MagicMock()
        mock_result.usage = {}
        mock_result.total_cost_usd = 0.001

        usage = extract_sdk_usage(mock_result)

        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0
        assert usage.cost_usd == 0.001
