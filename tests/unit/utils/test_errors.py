"""Tests for error detection utilities."""

from __future__ import annotations

import pytest

from karenina.utils.errors import is_retryable_error


@pytest.mark.unit
class TestIsRetryableErrorEventLoop:
    """Event loop errors should be classified as retryable (transient)."""

    def test_event_loop_closed_is_retryable(self) -> None:
        assert is_retryable_error(RuntimeError("Event loop is closed")) is True

    def test_event_loop_different_message_retryable(self) -> None:
        assert is_retryable_error(RuntimeError("Event loop is running")) is True


@pytest.mark.unit
class TestIsRetryableErrorPortal:
    """Portal errors should be classified as retryable (transient)."""

    def test_portal_not_running_is_retryable(self) -> None:
        assert is_retryable_error(RuntimeError("This portal is not running")) is True


@pytest.mark.unit
class TestIsRetryableErrorExisting:
    """Existing behavior preserved."""

    def test_connection_error_is_retryable(self) -> None:
        assert is_retryable_error(ConnectionError("connection reset")) is True

    def test_value_error_is_not_retryable(self) -> None:
        assert is_retryable_error(ValueError("invalid input")) is False

    def test_timeout_error_is_retryable(self) -> None:
        assert is_retryable_error(TimeoutError("timed out")) is True

    def test_rate_limit_message_retryable(self) -> None:
        assert is_retryable_error(Exception("rate limit exceeded")) is True
