"""Tests for comparison primitive bug fixes.

Covers:
- Issue 045: RegexMatch silently ignores unknown flag names
- BooleanMatch treats None as distinct from True and False
"""

import logging

import pytest

from karenina.schemas.primitives.comparisons import BooleanMatch, RegexMatch


@pytest.mark.unit
class TestRegexMatchUnknownFlags:
    """Issue 045: RegexMatch should warn on unknown flag names."""

    def test_known_flags_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Valid re flags like IGNORECASE should produce no warning."""
        rm = RegexMatch(pattern=r"hello", flags=["IGNORECASE"])
        with caplog.at_level(logging.WARNING):
            result = rm.check("Hello World", None)
        assert result is True
        assert "Unknown regex flag" not in caplog.text

    def test_unknown_flag_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """A typo like 'IGNORCASE' should log a warning."""
        rm = RegexMatch(pattern=r"hello", flags=["IGNORCASE"])
        with caplog.at_level(logging.WARNING):
            rm.check("hello world", None)
        assert "Unknown regex flag" in caplog.text
        assert "IGNORCASE" in caplog.text

    def test_unknown_flag_does_not_contribute_to_flags(self) -> None:
        """An unknown flag must not contribute bits (no silent fallback to 0)."""
        # With correct IGNORECASE, this should match
        rm_good = RegexMatch(pattern=r"HELLO", flags=["IGNORECASE"])
        assert rm_good.check("hello", None) is True

        # With typo IGNORCASE, the flag is skipped, so case-sensitive
        rm_bad = RegexMatch(pattern=r"HELLO", flags=["IGNORCASE"])
        assert rm_bad.check("hello", None) is False

    def test_multiple_flags_with_one_unknown(self, caplog: pytest.LogCaptureFixture) -> None:
        """When mixing valid and invalid flags, only the valid ones apply."""
        rm = RegexMatch(pattern=r"hello", flags=["IGNORECASE", "BADFLAG"])
        with caplog.at_level(logging.WARNING):
            result = rm.check("Hello", None)
        # IGNORECASE still applies, so match should succeed
        assert result is True
        # But a warning should be emitted for BADFLAG
        assert "BADFLAG" in caplog.text


@pytest.mark.unit
class TestBooleanMatchNoneHandling:
    """BooleanMatch should treat None as non-match against True and False."""

    def test_none_vs_false_is_non_match(self) -> None:
        """None should NOT match False (previous bug: bool(None)==bool(False))."""
        bm = BooleanMatch()
        assert bm.check(None, False) is False

    def test_none_vs_true_is_non_match(self) -> None:
        bm = BooleanMatch()
        assert bm.check(None, True) is False

    def test_false_vs_none_is_non_match(self) -> None:
        bm = BooleanMatch()
        assert bm.check(False, None) is False

    def test_true_vs_none_is_non_match(self) -> None:
        bm = BooleanMatch()
        assert bm.check(True, None) is False

    def test_none_vs_none_is_match(self) -> None:
        """None should match None via identity."""
        bm = BooleanMatch()
        assert bm.check(None, None) is True

    def test_true_vs_true_still_works(self) -> None:
        bm = BooleanMatch()
        assert bm.check(True, True) is True

    def test_false_vs_false_still_works(self) -> None:
        bm = BooleanMatch()
        assert bm.check(False, False) is True

    def test_true_vs_false_still_non_match(self) -> None:
        bm = BooleanMatch()
        assert bm.check(True, False) is False
