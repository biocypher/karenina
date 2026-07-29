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
    ├── StreamingTimeoutError        # LLM streaming timeout (also inherits TimeoutError)
    ├── GlobalLimiterTimeoutError    # GlobalLLMLimiter borrow acquisition timeout
    ├── VerificationBatchError       # Partial-failure batch verification errors
    ├── BenchmarkConversionError     # Benchmark conversion errors
    ├── ReplayError                  # Replay layer errors (see replay/exceptions.py)
    │   ├── ReplayMissError
    │   ├── ReplayHydrationError
    │   ├── ReplayPersistenceError
    │   └── ProjectionError
    └── ErrorAnalysisError           # Error-analysis errors (see benchmark/error_analysis/exceptions.py)
        ├── MaterializationError
        ├── LauncherNotFoundError
        ├── LauncherUnavailableError
        ├── LauncherExecutionError
        └── LauncherNoOutputError

Domain-specific modules (ports/errors.py, adapters/manual/exceptions.py, etc.)
remain the canonical definition sites for their respective exceptions. This module
re-exports them for convenience and defines the shared base class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from karenina.schemas.verification.result import VerificationResult


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


class StreamingTimeoutError(KareninaError, TimeoutError):
    """Raised when an LLM streaming operation times out.

    Inherits from both KareninaError (karenina exception hierarchy) and
    TimeoutError (for adapter retry classification as TIMEOUT category).

    Args:
        message: Human-readable description of the timeout.
        partial_content: Any content accumulated before the timeout occurred.
    """

    def __init__(self, message: str, partial_content: str = "") -> None:
        super().__init__(message)
        self.partial_content = partial_content


class GlobalLimiterTimeoutError(KareninaError, TimeoutError):
    """Raised when a GlobalLLMLimiter borrow cannot acquire a permit in time.

    Inherits from both KareninaError (karenina exception hierarchy) and
    TimeoutError (for adapter retry classification as TIMEOUT category).
    Raised only when an opt-in acquire timeout was set on the limiter
    constructor (the default acquire is unbounded). Surfacing this error
    means the global concurrency cap stayed saturated for the whole
    acquire timeout, usually because a wedged endpoint is holding permits.
    """


class VerificationBatchError(KareninaError):
    """Raised when a batch verification completes with partial failures.

    Stores both the successfully completed results and the per-question
    errors so callers can inspect partial progress without losing data.

    Args:
        message: Human-readable summary of the batch failure.
        partial_results: Mapping of question ID to its completed
            VerificationResult for questions that succeeded.
        errors: List of (identifier, exception) pairs for items
            that failed during verification.
    """

    def __init__(
        self,
        message: str,
        partial_results: dict[str, VerificationResult],
        errors: list[tuple[str, BaseException]],
    ) -> None:
        super().__init__(message)
        self.partial_results = partial_results
        self.errors = errors


# Re-export error-analysis exceptions for convenience. The canonical
# definition lives in karenina.benchmark.error_analysis.exceptions.
from karenina.benchmark.error_analysis.exceptions import (  # noqa: E402, F401
    ErrorAnalysisError,
    LauncherExecutionError,
    LauncherNoOutputError,
    LauncherNotFoundError,
    LauncherUnavailableError,
    MaterializationError,
)

# Re-export replay layer exceptions for convenience. The canonical
# definition lives in karenina.replay.exceptions.
from karenina.replay.exceptions import (  # noqa: E402, F401
    ProjectionError,
    ReplayError,
    ReplayHydrationError,
    ReplayMissError,
    ReplayPersistenceError,
)
