"""Tests for retry utility functions."""

import logging
from unittest.mock import MagicMock

import pytest

from karenina.utils.retry import log_retry


@pytest.mark.unit
class TestLogRetry:
    """log_retry should use the provided max_attempts value in log messages."""

    def test_default_max_attempts_is_3(self, caplog: pytest.LogCaptureFixture) -> None:
        state = MagicMock()
        state.outcome.exception.return_value = RuntimeError("test")
        state.attempt_number = 1
        with caplog.at_level(logging.WARNING):
            log_retry(state)
        assert "attempt 1/3" in caplog.text

    def test_custom_max_attempts(self, caplog: pytest.LogCaptureFixture) -> None:
        state = MagicMock()
        state.outcome.exception.return_value = RuntimeError("test")
        state.attempt_number = 2
        with caplog.at_level(logging.WARNING):
            log_retry(state, max_attempts=5)
        assert "attempt 2/5" in caplog.text

    def test_custom_context(self, caplog: pytest.LogCaptureFixture) -> None:
        state = MagicMock()
        state.outcome.exception.return_value = RuntimeError("oops")
        state.attempt_number = 1
        with caplog.at_level(logging.WARNING):
            log_retry(state, context="embedding check", max_attempts=4)
        assert "Retrying embedding check" in caplog.text
        assert "attempt 1/4" in caplog.text
