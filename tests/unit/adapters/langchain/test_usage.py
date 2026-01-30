"""Tests for usage metadata extraction patterns."""

from __future__ import annotations

from karenina.ports import UsageMetadata


class TestUsageMetadataExtraction:
    """Tests for usage metadata extraction patterns."""

    def test_usage_metadata_dataclass(self) -> None:
        """Test UsageMetadata creation and fields."""
        usage = UsageMetadata(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cache_read_tokens=25,
            cache_creation_tokens=10,
            model="claude-sonnet-4",
        )

        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150
        assert usage.cache_read_tokens == 25
        assert usage.cache_creation_tokens == 10
        assert usage.model == "claude-sonnet-4"

    def test_usage_metadata_optional_fields(self) -> None:
        """Test UsageMetadata with only required fields."""
        usage = UsageMetadata(
            input_tokens=50,
            output_tokens=25,
            total_tokens=75,
        )

        assert usage.input_tokens == 50
        assert usage.output_tokens == 25
        assert usage.total_tokens == 75
        assert usage.cache_read_tokens is None
        assert usage.cache_creation_tokens is None
        assert usage.model is None
