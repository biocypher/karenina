"""Tests for error detection and classification utilities."""

from __future__ import annotations

import pytest

from karenina.utils.errors import ErrorCategory, ErrorRegistry, is_retryable_error


@pytest.mark.unit
class TestErrorCategory:
    """ErrorCategory enum values and is_retryable method."""

    def test_connection_is_retryable(self) -> None:
        assert ErrorCategory.CONNECTION.is_retryable() is True

    def test_timeout_is_retryable(self) -> None:
        assert ErrorCategory.TIMEOUT.is_retryable() is True

    def test_rate_limit_is_retryable(self) -> None:
        assert ErrorCategory.RATE_LIMIT.is_retryable() is True

    def test_server_error_is_retryable(self) -> None:
        assert ErrorCategory.SERVER_ERROR.is_retryable() is True

    def test_permanent_is_not_retryable(self) -> None:
        assert ErrorCategory.PERMANENT.is_retryable() is False

    def test_all_categories_have_string_values(self) -> None:
        expected = {
            "connection",
            "timeout",
            "rate_limit",
            "server_error",
            "permanent",
        }
        actual = {c.value for c in ErrorCategory}
        assert actual == expected


@pytest.mark.unit
class TestErrorRegistryBuiltinTypes:
    """ErrorRegistry classifies built-in exception types correctly."""

    def test_connection_error_type(self) -> None:
        registry = ErrorRegistry()
        assert registry.classify(ConnectionError("fail")) == ErrorCategory.CONNECTION

    def test_timeout_error_type(self) -> None:
        registry = ErrorRegistry()
        assert registry.classify(TimeoutError("fail")) == ErrorCategory.TIMEOUT

    def test_value_error_is_permanent(self) -> None:
        registry = ErrorRegistry()
        assert registry.classify(ValueError("bad input")) == ErrorCategory.PERMANENT

    def test_key_error_is_permanent(self) -> None:
        registry = ErrorRegistry()
        assert registry.classify(KeyError("missing")) == ErrorCategory.PERMANENT

    def test_type_error_is_permanent(self) -> None:
        registry = ErrorRegistry()
        assert registry.classify(TypeError("wrong type")) == ErrorCategory.PERMANENT


@pytest.mark.unit
class TestErrorRegistryBuiltinTypeNames:
    """ErrorRegistry classifies by type name for third-party exception types."""

    def _make_exception(self, name: str, message: str = "fail") -> BaseException:
        """Create an exception with a specific class name."""
        exc_class = type(name, (Exception,), {})
        return exc_class(message)

    def test_api_connection_error(self) -> None:
        registry = ErrorRegistry()
        exc = self._make_exception("APIConnectionError")
        assert registry.classify(exc) == ErrorCategory.CONNECTION

    def test_api_timeout_error(self) -> None:
        registry = ErrorRegistry()
        exc = self._make_exception("APITimeoutError")
        assert registry.classify(exc) == ErrorCategory.TIMEOUT

    def test_read_timeout(self) -> None:
        registry = ErrorRegistry()
        exc = self._make_exception("ReadTimeout")
        assert registry.classify(exc) == ErrorCategory.TIMEOUT

    def test_connect_timeout(self) -> None:
        registry = ErrorRegistry()
        exc = self._make_exception("ConnectTimeout")
        assert registry.classify(exc) == ErrorCategory.TIMEOUT

    def test_streaming_timeout_error(self) -> None:
        registry = ErrorRegistry()
        exc = self._make_exception("StreamingTimeoutError")
        assert registry.classify(exc) == ErrorCategory.TIMEOUT

    def test_rate_limit_error(self) -> None:
        registry = ErrorRegistry()
        exc = self._make_exception("RateLimitError")
        assert registry.classify(exc) == ErrorCategory.RATE_LIMIT

    def test_overloaded_error(self) -> None:
        registry = ErrorRegistry()
        exc = self._make_exception("OverloadedError")
        assert registry.classify(exc) == ErrorCategory.RATE_LIMIT

    def test_internal_server_error(self) -> None:
        registry = ErrorRegistry()
        exc = self._make_exception("InternalServerError")
        assert registry.classify(exc) == ErrorCategory.SERVER_ERROR

    def test_http_error(self) -> None:
        registry = ErrorRegistry()
        exc = self._make_exception("HTTPError")
        assert registry.classify(exc) == ErrorCategory.SERVER_ERROR


@pytest.mark.unit
class TestErrorRegistryBuiltinMessages:
    """ErrorRegistry classifies by message content for generic exceptions."""

    def test_connection_message(self) -> None:
        registry = ErrorRegistry()
        assert registry.classify(Exception("connection refused")) == ErrorCategory.CONNECTION

    def test_network_message(self) -> None:
        registry = ErrorRegistry()
        assert registry.classify(Exception("network unreachable")) == ErrorCategory.CONNECTION

    def test_dns_message(self) -> None:
        registry = ErrorRegistry()
        assert registry.classify(Exception("dns resolution failed")) == ErrorCategory.CONNECTION

    def test_timeout_message(self) -> None:
        registry = ErrorRegistry()
        assert registry.classify(Exception("request timeout")) == ErrorCategory.TIMEOUT

    def test_timed_out_message(self) -> None:
        registry = ErrorRegistry()
        assert registry.classify(Exception("operation timed out")) == ErrorCategory.TIMEOUT

    def test_rate_limit_message(self) -> None:
        registry = ErrorRegistry()
        assert registry.classify(Exception("rate limit exceeded")) == ErrorCategory.RATE_LIMIT

    def test_429_message(self) -> None:
        registry = ErrorRegistry()
        assert registry.classify(Exception("HTTP 429 Too Many Requests")) == ErrorCategory.RATE_LIMIT

    def test_overloaded_message(self) -> None:
        registry = ErrorRegistry()
        assert registry.classify(Exception("service overloaded")) == ErrorCategory.RATE_LIMIT

    def test_500_message(self) -> None:
        registry = ErrorRegistry()
        assert registry.classify(Exception("HTTP 500 Internal Server Error")) == ErrorCategory.SERVER_ERROR

    def test_502_message(self) -> None:
        registry = ErrorRegistry()
        assert registry.classify(Exception("HTTP 502 Bad Gateway")) == ErrorCategory.SERVER_ERROR

    def test_503_message(self) -> None:
        registry = ErrorRegistry()
        assert registry.classify(Exception("HTTP 503 Service Unavailable")) == ErrorCategory.SERVER_ERROR

    def test_unrecognized_message_is_permanent(self) -> None:
        registry = ErrorRegistry()
        assert registry.classify(Exception("something else entirely")) == ErrorCategory.PERMANENT


@pytest.mark.unit
class TestErrorRegistryCustomPatterns:
    """ErrorRegistry allows user-registered patterns that take priority."""

    def test_register_by_exception_class(self) -> None:
        registry = ErrorRegistry()
        registry.register(ValueError, ErrorCategory.RATE_LIMIT)
        assert registry.classify(ValueError("doesn't matter")) == ErrorCategory.RATE_LIMIT

    def test_register_by_type_name_string(self) -> None:
        registry = ErrorRegistry()
        registry.register("MyCustomTimeout", ErrorCategory.TIMEOUT)
        exc_class = type("MyCustomTimeout", (Exception,), {})
        assert registry.classify(exc_class("fail")) == ErrorCategory.TIMEOUT

    def test_register_pattern_message_substring(self) -> None:
        registry = ErrorRegistry()
        registry.register_pattern("custom_sentinel", ErrorCategory.SERVER_ERROR)
        assert registry.classify(Exception("hit custom_sentinel in request")) == ErrorCategory.SERVER_ERROR

    def test_register_pattern_type_name_match(self) -> None:
        registry = ErrorRegistry()
        registry.register_pattern("SpecialError", ErrorCategory.CONNECTION, match_type="type_name")
        exc_class = type("SpecialError", (Exception,), {})
        assert registry.classify(exc_class("whatever")) == ErrorCategory.CONNECTION

    def test_custom_type_overrides_builtin_message(self) -> None:
        """Custom type registration takes priority over built-in message rules."""
        registry = ErrorRegistry()
        # ValueError with "connection" in message would match built-in message rule
        # as CONNECTION, but custom registration should override to PERMANENT
        registry.register(ValueError, ErrorCategory.PERMANENT)
        assert registry.classify(ValueError("connection error")) == ErrorCategory.PERMANENT

    def test_custom_message_overrides_builtin_message(self) -> None:
        """Custom message patterns take priority over built-in message patterns."""
        registry = ErrorRegistry()
        # "500" normally matches SERVER_ERROR, but register it as RATE_LIMIT
        registry.register_pattern("500", ErrorCategory.RATE_LIMIT)
        assert registry.classify(Exception("HTTP 500 error")) == ErrorCategory.RATE_LIMIT

    def test_custom_type_name_overrides_builtin_type_name(self) -> None:
        """Custom type name registration overrides built-in type name rules."""
        registry = ErrorRegistry()
        # RateLimitError normally maps to RATE_LIMIT, override to CONNECTION
        registry.register("RateLimitError", ErrorCategory.CONNECTION)
        exc_class = type("RateLimitError", (Exception,), {})
        assert registry.classify(exc_class("fail")) == ErrorCategory.CONNECTION


@pytest.mark.unit
class TestErrorRegistryPriorityOrder:
    """Classification follows the documented priority order."""

    def test_isinstance_beats_type_name_string(self) -> None:
        """isinstance check (class registration) beats type name string check."""
        registry = ErrorRegistry()
        # Register by class and by type name string with different categories
        registry.register(ConnectionError, ErrorCategory.TIMEOUT)
        registry.register("ConnectionError", ErrorCategory.RATE_LIMIT)
        # Class-based isinstance should win
        assert registry.classify(ConnectionError("fail")) == ErrorCategory.TIMEOUT

    def test_user_type_name_beats_user_message(self) -> None:
        """User type name match beats user message substring match."""
        registry = ErrorRegistry()
        exc_class = type("CustomError", (Exception,), {})
        registry.register_pattern("CustomError", ErrorCategory.TIMEOUT, match_type="type_name")
        registry.register_pattern("fail", ErrorCategory.RATE_LIMIT)
        assert registry.classify(exc_class("fail")) == ErrorCategory.TIMEOUT

    def test_user_patterns_beat_builtin_types(self) -> None:
        """User-registered message patterns beat built-in type name rules."""
        registry = ErrorRegistry()
        # Register a message pattern that will match
        registry.register_pattern("fail", ErrorCategory.RATE_LIMIT)
        # HTTPError normally classified as SERVER_ERROR by built-in type rules
        exc_class = type("HTTPError", (Exception,), {})
        exc = exc_class("fail")
        # User message pattern should take priority
        assert registry.classify(exc) == ErrorCategory.RATE_LIMIT


@pytest.mark.unit
class TestIsRetryableErrorBackwardCompat:
    """is_retryable_error compatibility wrapper preserves existing behavior."""

    def test_connection_error_is_retryable(self) -> None:
        assert is_retryable_error(ConnectionError("connection reset")) is True

    def test_value_error_is_not_retryable(self) -> None:
        assert is_retryable_error(ValueError("invalid input")) is False

    def test_timeout_error_is_retryable(self) -> None:
        assert is_retryable_error(TimeoutError("timed out")) is True

    def test_rate_limit_message_retryable(self) -> None:
        assert is_retryable_error(Exception("rate limit exceeded")) is True

    def test_event_loop_closed_is_retryable(self) -> None:
        assert is_retryable_error(RuntimeError("Event loop is closed")) is True

    def test_portal_not_running_is_retryable(self) -> None:
        assert is_retryable_error(RuntimeError("This portal is not running")) is True

    def test_generic_exception_is_not_retryable(self) -> None:
        assert is_retryable_error(Exception("something random")) is False


@pytest.mark.unit
class TestStreamingTimeoutErrorClassification:
    """StreamingTimeoutError is always classified as TIMEOUT."""

    def test_zero_content_timeout_is_timeout(self) -> None:
        """Zero content or not, a timeout is a timeout."""
        from karenina.exceptions import StreamingTimeoutError

        registry = ErrorRegistry()
        exc = StreamingTimeoutError("Streaming timed out after 120s", partial_content="")
        assert registry.classify(exc) == ErrorCategory.TIMEOUT

    def test_partial_content_timeout_is_timeout(self) -> None:
        """Partial content timeout is also a timeout."""
        from karenina.exceptions import StreamingTimeoutError

        registry = ErrorRegistry()
        exc = StreamingTimeoutError("Streaming timed out after 120s", partial_content="The answer is")
        assert registry.classify(exc) == ErrorCategory.TIMEOUT

    def test_user_override_takes_precedence(self) -> None:
        """User can override StreamingTimeoutError classification."""
        from karenina.exceptions import StreamingTimeoutError

        registry = ErrorRegistry()
        registry.register(StreamingTimeoutError, ErrorCategory.SERVER_ERROR)
        exc = StreamingTimeoutError("timed out", partial_content="")
        assert registry.classify(exc) == ErrorCategory.SERVER_ERROR


@pytest.mark.unit
class TestProviderSDKExceptionClassification:
    """Real Anthropic and OpenAI SDK exceptions classify to retryable categories.

    Audit for design decision D1 (T2): with SDK max_retries=0, RetryExecutor
    is the sole retry layer for claude_tool and the claude_agent_sdk parser,
    so every transient SDK exception must classify via the built-in MRO
    type-name rules instead of falling through to PERMANENT.
    """

    @staticmethod
    def _request():
        import httpx

        return httpx.Request("POST", "http://localhost:8000/v1/messages")

    @staticmethod
    def _response(status_code: int):
        import httpx

        return httpx.Response(status_code, request=httpx.Request("POST", "http://localhost:8000/v1/messages"))

    def test_anthropic_api_connection_error(self) -> None:
        anthropic = pytest.importorskip("anthropic")
        exc = anthropic.APIConnectionError(request=self._request())
        assert ErrorRegistry().classify(exc) == ErrorCategory.CONNECTION

    def test_anthropic_api_timeout_error(self) -> None:
        anthropic = pytest.importorskip("anthropic")
        exc = anthropic.APITimeoutError(request=self._request())
        assert ErrorRegistry().classify(exc) == ErrorCategory.TIMEOUT

    def test_anthropic_rate_limit_error(self) -> None:
        anthropic = pytest.importorskip("anthropic")
        exc = anthropic.RateLimitError("rate limited", response=self._response(429), body=None)
        assert ErrorRegistry().classify(exc) == ErrorCategory.RATE_LIMIT

    def test_anthropic_internal_server_error(self) -> None:
        anthropic = pytest.importorskip("anthropic")
        exc = anthropic.InternalServerError("server exploded", response=self._response(500), body=None)
        assert ErrorRegistry().classify(exc) == ErrorCategory.SERVER_ERROR

    def test_openai_api_connection_error(self) -> None:
        openai = pytest.importorskip("openai")
        exc = openai.APIConnectionError(request=self._request())
        assert ErrorRegistry().classify(exc) == ErrorCategory.CONNECTION

    def test_openai_api_timeout_error(self) -> None:
        openai = pytest.importorskip("openai")
        exc = openai.APITimeoutError(request=self._request())
        assert ErrorRegistry().classify(exc) == ErrorCategory.TIMEOUT

    def test_openai_rate_limit_error(self) -> None:
        openai = pytest.importorskip("openai")
        exc = openai.RateLimitError("rate limited", response=self._response(429), body=None)
        assert ErrorRegistry().classify(exc) == ErrorCategory.RATE_LIMIT

    def test_openai_internal_server_error(self) -> None:
        openai = pytest.importorskip("openai")
        exc = openai.InternalServerError("server exploded", response=self._response(500), body=None)
        assert ErrorRegistry().classify(exc) == ErrorCategory.SERVER_ERROR
