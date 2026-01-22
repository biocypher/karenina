"""Tests for SDK error wrapping.

Tests wrap_sdk_error function that converts SDK exceptions to port errors.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from karenina.adapters.claude_agent_sdk import wrap_sdk_error
from karenina.ports.errors import (
    AdapterUnavailableError,
    AgentExecutionError,
    AgentResponseError,
    AgentTimeoutError,
)


class TestWrapSdkError:
    """Tests for wrap_sdk_error function."""

    def test_cli_not_found_error(self) -> None:
        """Test CLINotFoundError wrapping."""
        mock_error = MagicMock()
        mock_error.__class__.__name__ = "CLINotFoundError"
        mock_error.cli_path = "/usr/local/bin/claude"

        result = wrap_sdk_error(mock_error)

        assert isinstance(result, AdapterUnavailableError)
        assert "Claude Code CLI not found" in result.message
        assert result.fallback_interface == "langchain"

    def test_cli_connection_error(self) -> None:
        """Test CLIConnectionError wrapping."""
        mock_error = MagicMock()
        mock_error.__class__.__name__ = "CLIConnectionError"

        result = wrap_sdk_error(mock_error)

        assert isinstance(result, AgentExecutionError)
        assert "connect" in result.message.lower()

    def test_process_error_timeout_124(self) -> None:
        """Test ProcessError with exit code 124 (timeout command)."""
        mock_error = MagicMock()
        mock_error.__class__.__name__ = "ProcessError"
        mock_error.exit_code = 124
        mock_error.stderr = "Timeout exceeded"

        result = wrap_sdk_error(mock_error)

        assert isinstance(result, AgentTimeoutError)
        assert "timed out" in result.message.lower()

    def test_process_error_timeout_137(self) -> None:
        """Test ProcessError with exit code 137 (SIGKILL)."""
        mock_error = MagicMock()
        mock_error.__class__.__name__ = "ProcessError"
        mock_error.exit_code = 137
        mock_error.stderr = "Killed"

        result = wrap_sdk_error(mock_error)

        assert isinstance(result, AgentTimeoutError)

    def test_process_error_general(self) -> None:
        """Test ProcessError with other exit codes."""
        mock_error = MagicMock()
        mock_error.__class__.__name__ = "ProcessError"
        mock_error.exit_code = 1
        mock_error.stderr = "Something went wrong"

        result = wrap_sdk_error(mock_error)

        assert isinstance(result, AgentExecutionError)
        assert "exit code 1" in result.message

    def test_process_error_truncates_long_stderr(self) -> None:
        """Test ProcessError truncates very long stderr."""
        mock_error = MagicMock()
        mock_error.__class__.__name__ = "ProcessError"
        mock_error.exit_code = 1
        mock_error.stderr = "x" * 1000  # Very long stderr

        result = wrap_sdk_error(mock_error)

        assert isinstance(result, AgentExecutionError)
        assert len(result.message) < 1000  # Should be truncated

    def test_cli_json_decode_error(self) -> None:
        """Test CLIJSONDecodeError wrapping."""
        mock_error = MagicMock()
        mock_error.__class__.__name__ = "CLIJSONDecodeError"
        mock_error.line = "invalid json here"

        result = wrap_sdk_error(mock_error)

        assert isinstance(result, AgentResponseError)
        assert "JSON" in result.message

    def test_cli_json_decode_error_long_line(self) -> None:
        """Test CLIJSONDecodeError truncates long line info."""
        mock_error = MagicMock()
        mock_error.__class__.__name__ = "CLIJSONDecodeError"
        mock_error.line = "x" * 500  # Long line

        result = wrap_sdk_error(mock_error)

        assert isinstance(result, AgentResponseError)
        assert "..." in result.message  # Should be truncated

    def test_unknown_exception(self) -> None:
        """Test wrapping of unknown exception types."""
        mock_error = MagicMock()
        mock_error.__class__.__name__ = "SomeRandomError"

        result = wrap_sdk_error(mock_error)

        assert isinstance(result, AgentExecutionError)
        assert "SomeRandomError" in result.message

    def test_standard_exception(self) -> None:
        """Test wrapping of standard Python exceptions."""
        error = ValueError("Invalid value provided")

        result = wrap_sdk_error(error)

        assert isinstance(result, AgentExecutionError)
        assert "ValueError" in result.message
