"""Test that tool error detection no longer produces false positives."""

import pytest

from karenina.adapters.langchain.trace import _detect_tool_error


@pytest.mark.unit
class TestToolErrorDetectionRemoved:
    def test_legitimate_tool_output_not_flagged(self):
        """Tool output containing error keywords should not be flagged."""
        assert _detect_tool_error("No errors found in the analysis") is False
        assert _detect_tool_error("timeout set to 30 seconds") is False
        assert _detect_tool_error("The exception handling looks correct") is False
        assert _detect_tool_error("Search failed to find any results") is False

    def test_actual_error_also_not_flagged(self):
        """Even actual errors return False (no structural signal available)."""
        assert _detect_tool_error("Error: connection refused") is False
        assert _detect_tool_error("Traceback (most recent call last):") is False

    def test_empty_content(self):
        """Empty content returns False."""
        assert _detect_tool_error("") is False
