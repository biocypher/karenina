"""Unit tests for error types in the ports module.

Tests cover:
- Error inheritance hierarchy (all errors inherit from PortError)
- AdapterUnavailableError with reason and fallback_interface fields
- AgentExecutionError with stderr field
- AgentTimeoutError inheriting from AgentExecutionError
- AgentResponseError (simple error with just message)
- ParseError with raw_response field
- Error message formatting
"""

import pytest

from karenina.ports import (
    AdapterUnavailableError,
    AgentExecutionError,
    AgentResponseError,
    AgentTimeoutError,
    ParseError,
    PortError,
)

# =============================================================================
# PortError Base Class Tests
# =============================================================================


@pytest.mark.unit
class TestPortError:
    """Tests for the PortError base exception class."""

    def test_port_error_is_exception(self) -> None:
        """Test that PortError inherits from Exception."""
        assert issubclass(PortError, Exception)

    def test_port_error_creation_with_message(self) -> None:
        """Test PortError creation with message."""
        error = PortError("Something went wrong")
        assert error.message == "Something went wrong"

    def test_port_error_message_in_str(self) -> None:
        """Test that PortError message appears in string representation."""
        error = PortError("Test error message")
        assert str(error) == "Test error message"

    def test_port_error_can_be_raised_and_caught(self) -> None:
        """Test that PortError can be raised and caught."""
        with pytest.raises(PortError) as exc_info:
            raise PortError("Raised error")
        assert exc_info.value.message == "Raised error"


# =============================================================================
# AdapterUnavailableError Tests
# =============================================================================


@pytest.mark.unit
class TestAdapterUnavailableError:
    """Tests for AdapterUnavailableError."""

    def test_adapter_unavailable_inherits_from_port_error(self) -> None:
        """Test that AdapterUnavailableError inherits from PortError."""
        assert issubclass(AdapterUnavailableError, PortError)

    def test_adapter_unavailable_can_be_caught_as_port_error(self) -> None:
        """Test that AdapterUnavailableError can be caught as PortError."""
        with pytest.raises(PortError):
            raise AdapterUnavailableError("Adapter not found")

    def test_adapter_unavailable_with_message_only(self) -> None:
        """Test AdapterUnavailableError with just message."""
        error = AdapterUnavailableError("Claude CLI not installed")
        assert error.message == "Claude CLI not installed"
        assert error.reason == "Claude CLI not installed"  # defaults to message
        assert error.fallback_interface is None

    def test_adapter_unavailable_with_reason(self) -> None:
        """Test AdapterUnavailableError with explicit reason."""
        error = AdapterUnavailableError(
            message="Cannot use Claude SDK adapter",
            reason="claude executable not in PATH",
        )
        assert error.message == "Cannot use Claude SDK adapter"
        assert error.reason == "claude executable not in PATH"

    def test_adapter_unavailable_with_fallback_interface(self) -> None:
        """Test AdapterUnavailableError with fallback_interface."""
        error = AdapterUnavailableError(
            message="Claude CLI not found",
            reason="Missing executable",
            fallback_interface="langchain",
        )
        assert error.fallback_interface == "langchain"

    def test_adapter_unavailable_fallback_enables_auto_fallback(self) -> None:
        """Test that fallback_interface suggests alternative adapter."""
        error = AdapterUnavailableError(
            message="Primary adapter unavailable",
            fallback_interface="langchain",
        )
        # Simulate auto-fallback logic
        fallback_used = error.fallback_interface or None
        assert fallback_used == "langchain"


# =============================================================================
# AgentExecutionError Tests
# =============================================================================


@pytest.mark.unit
class TestAgentExecutionError:
    """Tests for AgentExecutionError."""

    def test_agent_execution_inherits_from_port_error(self) -> None:
        """Test that AgentExecutionError inherits from PortError."""
        assert issubclass(AgentExecutionError, PortError)

    def test_agent_execution_can_be_caught_as_port_error(self) -> None:
        """Test that AgentExecutionError can be caught as PortError."""
        with pytest.raises(PortError):
            raise AgentExecutionError("Agent crashed")

    def test_agent_execution_with_message_only(self) -> None:
        """Test AgentExecutionError with just message."""
        error = AgentExecutionError("CLI process exited with code 1")
        assert error.message == "CLI process exited with code 1"
        assert error.stderr is None

    def test_agent_execution_with_stderr(self) -> None:
        """Test AgentExecutionError with stderr output."""
        error = AgentExecutionError(
            message="Agent failed",
            stderr="Error: connection refused\nRetry failed",
        )
        assert error.stderr == "Error: connection refused\nRetry failed"

    def test_agent_execution_stderr_none_by_default(self) -> None:
        """Test that stderr defaults to None."""
        error = AgentExecutionError("Failure")
        assert error.stderr is None


# =============================================================================
# AgentTimeoutError Tests
# =============================================================================


@pytest.mark.unit
class TestAgentTimeoutError:
    """Tests for AgentTimeoutError."""

    def test_agent_timeout_inherits_from_agent_execution_error(self) -> None:
        """Test that AgentTimeoutError inherits from AgentExecutionError."""
        assert issubclass(AgentTimeoutError, AgentExecutionError)

    def test_agent_timeout_also_inherits_from_port_error(self) -> None:
        """Test that AgentTimeoutError transitively inherits from PortError."""
        assert issubclass(AgentTimeoutError, PortError)

    def test_agent_timeout_can_be_caught_as_agent_execution_error(self) -> None:
        """Test that AgentTimeoutError can be caught as AgentExecutionError."""
        with pytest.raises(AgentExecutionError):
            raise AgentTimeoutError("Max turns exceeded")

    def test_agent_timeout_can_be_caught_as_port_error(self) -> None:
        """Test that AgentTimeoutError can be caught as PortError."""
        with pytest.raises(PortError):
            raise AgentTimeoutError("Timeout")

    def test_agent_timeout_with_message(self) -> None:
        """Test AgentTimeoutError with message."""
        error = AgentTimeoutError("Agent exceeded max_turns=25")
        assert error.message == "Agent exceeded max_turns=25"

    def test_agent_timeout_with_stderr(self) -> None:
        """Test AgentTimeoutError with stderr (inherited from parent)."""
        error = AgentTimeoutError(
            message="Recursion limit reached",
            stderr="Warning: max iterations exceeded",
        )
        assert error.stderr == "Warning: max iterations exceeded"

    def test_agent_timeout_is_distinguishable_from_general_execution_error(
        self,
    ) -> None:
        """Test that AgentTimeoutError can be distinguished from AgentExecutionError."""
        timeout_error = AgentTimeoutError("Timeout")
        general_error = AgentExecutionError("General failure")

        assert isinstance(timeout_error, AgentTimeoutError)
        assert isinstance(general_error, AgentExecutionError)
        assert not isinstance(general_error, AgentTimeoutError)


# =============================================================================
# AgentResponseError Tests
# =============================================================================


@pytest.mark.unit
class TestAgentResponseError:
    """Tests for AgentResponseError."""

    def test_agent_response_inherits_from_port_error(self) -> None:
        """Test that AgentResponseError inherits from PortError."""
        assert issubclass(AgentResponseError, PortError)

    def test_agent_response_can_be_caught_as_port_error(self) -> None:
        """Test that AgentResponseError can be caught as PortError."""
        with pytest.raises(PortError):
            raise AgentResponseError("Malformed response")

    def test_agent_response_with_message(self) -> None:
        """Test AgentResponseError with message."""
        error = AgentResponseError("Expected JSON but received plain text")
        assert error.message == "Expected JSON but received plain text"

    def test_agent_response_message_in_str(self) -> None:
        """Test AgentResponseError message appears in string representation."""
        error = AgentResponseError("Invalid format")
        assert str(error) == "Invalid format"


# =============================================================================
# ParseError Tests
# =============================================================================


@pytest.mark.unit
class TestParseError:
    """Tests for ParseError."""

    def test_parse_error_inherits_from_port_error(self) -> None:
        """Test that ParseError inherits from PortError."""
        assert issubclass(ParseError, PortError)

    def test_parse_error_can_be_caught_as_port_error(self) -> None:
        """Test that ParseError can be caught as PortError."""
        with pytest.raises(PortError):
            raise ParseError("Could not parse response")

    def test_parse_error_with_message_only(self) -> None:
        """Test ParseError with just message."""
        error = ParseError("Failed to extract Answer schema")
        assert error.message == "Failed to extract Answer schema"
        assert error.raw_response is None

    def test_parse_error_with_raw_response(self) -> None:
        """Test ParseError with raw_response field."""
        raw = "The model said: I don't know the answer to that question."
        error = ParseError(
            message="Could not extract structured data",
            raw_response=raw,
        )
        assert error.raw_response == raw

    def test_parse_error_raw_response_for_debugging(self) -> None:
        """Test that raw_response provides debugging context."""
        raw_response = "Some malformed content that couldn't be parsed"
        error = ParseError(
            message="Parsing failed",
            raw_response=raw_response,
        )
        # In real usage, raw_response would help debug why parsing failed
        assert error.raw_response is not None
        assert "malformed" in error.raw_response


# =============================================================================
# Error Hierarchy Tests
# =============================================================================


@pytest.mark.unit
class TestErrorHierarchy:
    """Tests verifying the overall error hierarchy structure."""

    def test_all_errors_inherit_from_port_error(self) -> None:
        """Test that all port errors inherit from PortError."""
        error_classes = [
            AdapterUnavailableError,
            AgentExecutionError,
            AgentTimeoutError,
            AgentResponseError,
            ParseError,
        ]
        for error_class in error_classes:
            assert issubclass(error_class, PortError), f"{error_class.__name__} should inherit from PortError"

    def test_agent_timeout_is_subclass_of_agent_execution(self) -> None:
        """Test AgentTimeoutError is subclass of AgentExecutionError."""
        assert issubclass(AgentTimeoutError, AgentExecutionError)

    def test_agent_response_is_not_subclass_of_agent_execution(self) -> None:
        """Test AgentResponseError is NOT a subclass of AgentExecutionError."""
        assert not issubclass(AgentResponseError, AgentExecutionError)

    def test_catching_port_error_catches_all_subclasses(self) -> None:
        """Test that catching PortError catches all specific error types."""
        errors = [
            AdapterUnavailableError("adapter"),
            AgentExecutionError("execution"),
            AgentTimeoutError("timeout"),
            AgentResponseError("response"),
            ParseError("parse"),
        ]

        for error in errors:
            try:
                raise error
            except PortError as e:
                # Should catch all of them
                assert e is error


# =============================================================================
# Error Message Formatting Tests
# =============================================================================


@pytest.mark.unit
class TestErrorMessageFormatting:
    """Tests for error message formatting."""

    def test_error_messages_are_strings(self) -> None:
        """Test that error messages are properly stored as strings."""
        errors = [
            PortError("base error"),
            AdapterUnavailableError("adapter error"),
            AgentExecutionError("execution error"),
            AgentTimeoutError("timeout error"),
            AgentResponseError("response error"),
            ParseError("parse error"),
        ]
        for error in errors:
            assert isinstance(error.message, str)
            assert len(error.message) > 0

    def test_error_str_matches_message(self) -> None:
        """Test that str(error) returns the message."""
        message = "Test error message"
        error = PortError(message)
        assert str(error) == message

    def test_error_with_multiline_message(self) -> None:
        """Test error with multiline message."""
        message = "Line 1\nLine 2\nLine 3"
        error = PortError(message)
        assert error.message == message
        assert "\n" in str(error)
