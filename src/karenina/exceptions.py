"""Canonical exception hierarchy for the karenina package.

All karenina-specific exceptions inherit from KareninaError, enabling
unified exception handling while allowing fine-grained catching of
specific error types.

Exception hierarchy::

    KareninaError (base)
    ├── PortError                    # Port/adapter layer errors (see ports/errors.py)
    │   ├── AdapterUnavailableError
    │   ├── AgentExecutionError
    │   │   └── AgentTimeoutError
    │   ├── AgentResponseError
    │   └── ParseError
    ├── ManualTraceError             # Manual adapter errors (see adapters/manual/exceptions.py)
    │   └── ManualTraceNotFoundError
    ├── ManualInterfaceError
    ├── McpError                     # MCP-related errors
    │   ├── McpTimeoutError
    │   ├── McpClientError
    │   └── McpConfigValidationError
    └── BenchmarkConversionError     # Benchmark conversion errors

Domain-specific modules (ports/errors.py, adapters/manual/exceptions.py, etc.)
remain the canonical definition sites for their respective exceptions. This module
re-exports them for convenience and defines the shared base class.
"""

from __future__ import annotations


class KareninaError(Exception):
    """Base exception for all karenina-specific errors."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class McpError(KareninaError):
    """Base exception for MCP-related errors."""

    pass


class McpTimeoutError(McpError):
    """Raised when an MCP operation times out.

    Args:
        message: Human-readable description of the timeout.
        server_name: Name of the MCP server that timed out, if known.
        timeout_seconds: The timeout duration that was exceeded.
    """

    def __init__(
        self,
        message: str,
        server_name: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.server_name = server_name
        self.timeout_seconds = timeout_seconds


class McpClientError(McpError):
    """Raised when MCP client creation or tool fetching fails.

    Args:
        message: Human-readable description of the failure.
        server_name: Name of the MCP server involved, if known.
    """

    def __init__(
        self,
        message: str,
        server_name: str | None = None,
    ) -> None:
        super().__init__(message)
        self.server_name = server_name
