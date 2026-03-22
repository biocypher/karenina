"""Tests for ManualTraceManager.set_trace validation."""

import pytest

from karenina.adapters.manual.exceptions import ManualTraceError
from karenina.adapters.manual.manager import ManualTraceManager

VALID_HASH = "d41d8cd98f00b204e9800998ecf8427e"


@pytest.mark.unit
class TestSetTraceRejectsEmptyStrings:
    """Issue 029: set_trace must reject empty and whitespace-only traces."""

    def test_empty_string_raises_manual_trace_error(self):
        """Empty string trace is rejected with ManualTraceError."""
        manager = ManualTraceManager()
        with pytest.raises(ManualTraceError, match="non-empty string"):
            manager.set_trace(VALID_HASH, "")

    def test_whitespace_only_string_raises_manual_trace_error(self):
        """Whitespace-only trace is rejected with ManualTraceError."""
        manager = ManualTraceManager()
        with pytest.raises(ManualTraceError, match="non-empty string"):
            manager.set_trace(VALID_HASH, "   ")

    def test_newline_only_string_raises_manual_trace_error(self):
        """Newline-only trace is rejected with ManualTraceError."""
        manager = ManualTraceManager()
        with pytest.raises(ManualTraceError, match="non-empty string"):
            manager.set_trace(VALID_HASH, "\n\t\n")

    def test_valid_trace_is_accepted(self):
        """Non-empty trace string is accepted without error."""
        manager = ManualTraceManager()
        manager.set_trace(VALID_HASH, "This is a valid trace.")
        assert manager.get_trace(VALID_HASH) == "This is a valid trace."
