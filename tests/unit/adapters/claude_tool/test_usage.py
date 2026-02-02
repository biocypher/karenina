"""Tests for usage metadata extraction from Anthropic SDK responses.

Tests extract_usage, extract_usage_from_response, aggregate_usage,
and aggregate_usage_from_response.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from karenina.adapters.claude_tool.usage import (
    aggregate_usage,
    aggregate_usage_from_response,
    extract_usage,
    extract_usage_from_response,
)
from karenina.ports import UsageMetadata


class TestExtractUsage:
    """Tests for extract_usage function."""

    def test_extracts_basic_usage(self) -> None:
        """Test extracting basic token counts."""
        mock_usage = MagicMock()
        mock_usage.input_tokens = 100
        mock_usage.output_tokens = 50

        result = extract_usage(mock_usage)

        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.total_tokens == 150

    def test_extracts_usage_with_model(self) -> None:
        """Test extracting usage with model parameter."""
        mock_usage = MagicMock()
        mock_usage.input_tokens = 100
        mock_usage.output_tokens = 50

        result = extract_usage(mock_usage, model="claude-haiku-4-5")

        assert result.model == "claude-haiku-4-5"

    def test_extracts_cache_tokens(self) -> None:
        """Test extracting cache token information."""
        mock_usage = MagicMock()
        mock_usage.input_tokens = 200
        mock_usage.output_tokens = 100
        mock_usage.cache_read_input_tokens = 150
        mock_usage.cache_creation_input_tokens = 50

        result = extract_usage(mock_usage)

        assert result.cache_read_tokens == 150
        assert result.cache_creation_tokens == 50

    def test_handles_none_usage(self) -> None:
        """Test handling None usage object."""
        result = extract_usage(None)

        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.total_tokens == 0

    def test_handles_none_usage_with_model(self) -> None:
        """Test handling None usage with model parameter."""
        result = extract_usage(None, model="claude-sonnet-4")

        assert result.input_tokens == 0
        assert result.model == "claude-sonnet-4"

    def test_handles_missing_cache_attributes(self) -> None:
        """Test handling usage object without cache attributes."""
        mock_usage = MagicMock(spec=["input_tokens", "output_tokens"])
        mock_usage.input_tokens = 100
        mock_usage.output_tokens = 50

        result = extract_usage(mock_usage)

        assert result.input_tokens == 100
        assert result.cache_read_tokens is None
        assert result.cache_creation_tokens is None

    def test_handles_none_token_values(self) -> None:
        """Test handling usage object with None token values."""
        mock_usage = MagicMock()
        mock_usage.input_tokens = None
        mock_usage.output_tokens = None

        result = extract_usage(mock_usage)

        assert result.input_tokens == 0
        assert result.output_tokens == 0


class TestExtractUsageFromResponse:
    """Tests for extract_usage_from_response function."""

    def test_extracts_usage_from_response(self) -> None:
        """Test extracting usage from a response object."""
        mock_response = MagicMock()
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50

        result = extract_usage_from_response(mock_response)

        assert result.input_tokens == 100
        assert result.output_tokens == 50

    def test_extracts_usage_with_model(self) -> None:
        """Test extracting usage from response with model parameter."""
        mock_response = MagicMock()
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50

        result = extract_usage_from_response(mock_response, model="claude-opus-4")

        assert result.model == "claude-opus-4"

    def test_handles_response_without_usage(self) -> None:
        """Test handling response without usage attribute."""
        mock_response = MagicMock(spec=[])

        result = extract_usage_from_response(mock_response)

        assert result.input_tokens == 0
        assert result.output_tokens == 0

    def test_handles_response_with_none_usage(self) -> None:
        """Test handling response with None usage."""
        mock_response = MagicMock()
        mock_response.usage = None

        result = extract_usage_from_response(mock_response)

        assert result.input_tokens == 0


class TestAggregateUsage:
    """Tests for aggregate_usage function."""

    def test_aggregates_basic_usage(self) -> None:
        """Test aggregating basic token counts."""
        base = UsageMetadata(input_tokens=100, output_tokens=50, total_tokens=150)
        new = UsageMetadata(input_tokens=50, output_tokens=25, total_tokens=75)

        result = aggregate_usage(base, new)

        assert result.input_tokens == 150
        assert result.output_tokens == 75
        assert result.total_tokens == 225

    def test_aggregates_cost(self) -> None:
        """Test aggregating cost values."""
        base = UsageMetadata(input_tokens=100, output_tokens=50, total_tokens=150, cost_usd=0.001)
        new = UsageMetadata(input_tokens=50, output_tokens=25, total_tokens=75, cost_usd=0.0005)

        result = aggregate_usage(base, new)

        assert result.cost_usd == 0.0015

    def test_aggregates_cache_tokens(self) -> None:
        """Test aggregating cache token counts."""
        base = UsageMetadata(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cache_read_tokens=75,
            cache_creation_tokens=25,
        )
        new = UsageMetadata(
            input_tokens=50,
            output_tokens=25,
            total_tokens=75,
            cache_read_tokens=50,
            cache_creation_tokens=10,
        )

        result = aggregate_usage(base, new)

        assert result.cache_read_tokens == 125
        assert result.cache_creation_tokens == 35

    def test_uses_new_model(self) -> None:
        """Test that aggregation uses model from new usage."""
        base = UsageMetadata(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            model="claude-haiku-4-5",
        )
        new = UsageMetadata(
            input_tokens=50,
            output_tokens=25,
            total_tokens=75,
            model="claude-sonnet-4",
        )

        result = aggregate_usage(base, new)

        assert result.model == "claude-sonnet-4"

    def test_preserves_base_model_when_new_is_none(self) -> None:
        """Test preserves base model when new doesn't have one."""
        base = UsageMetadata(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            model="claude-haiku-4-5",
        )
        new = UsageMetadata(input_tokens=50, output_tokens=25, total_tokens=75)

        result = aggregate_usage(base, new)

        assert result.model == "claude-haiku-4-5"

    def test_handles_none_cache_tokens(self) -> None:
        """Test aggregation with None cache tokens."""
        base = UsageMetadata(input_tokens=100, output_tokens=50, total_tokens=150)
        new = UsageMetadata(input_tokens=50, output_tokens=25, total_tokens=75, cache_read_tokens=50)

        result = aggregate_usage(base, new)

        assert result.cache_read_tokens == 50

    def test_handles_none_cost(self) -> None:
        """Test aggregation when costs are None."""
        base = UsageMetadata(input_tokens=100, output_tokens=50, total_tokens=150)
        new = UsageMetadata(input_tokens=50, output_tokens=25, total_tokens=75)

        result = aggregate_usage(base, new)

        assert result.cost_usd is None


class TestAggregateUsageFromResponse:
    """Tests for aggregate_usage_from_response function."""

    def test_aggregates_from_response(self) -> None:
        """Test aggregating usage from a response object."""
        base = UsageMetadata(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            model="claude-haiku-4-5",
        )

        mock_response = MagicMock()
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 50
        mock_response.usage.output_tokens = 25

        result = aggregate_usage_from_response(base, mock_response)

        assert result.input_tokens == 150
        assert result.output_tokens == 75
        assert result.total_tokens == 225
        assert result.model == "claude-haiku-4-5"

    def test_preserves_model_from_base(self) -> None:
        """Test that model is preserved from base in aggregation."""
        base = UsageMetadata(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            model="claude-sonnet-4",
        )

        mock_response = MagicMock()
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 50
        mock_response.usage.output_tokens = 25

        result = aggregate_usage_from_response(base, mock_response)

        assert result.model == "claude-sonnet-4"
