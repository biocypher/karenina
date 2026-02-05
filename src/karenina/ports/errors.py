"""Port-specific error hierarchy.

This module defines exception classes for errors that occur within the
ports layer. All port errors inherit from `PortError`, enabling unified
exception handling while still allowing fine-grained catching of specific
error types.

Exception hierarchy:

    PortError (base)
    ├── AdapterUnavailableError   # Backend not available (missing CLI, config error)
    ├── AgentExecutionError       # Runtime failure during agent execution
    │   └── AgentTimeoutError     # Hit max_turns or time limit
    ├── AgentResponseError        # Invalid/malformed response from agent
    └── ParseError                # Structured output parsing failed

Example usage:

    from karenina.ports import (
        PortError,
        AdapterUnavailableError,
        AgentTimeoutError,
        ParseError,
    )

    try:
        result = await agent.run(messages)
    except AdapterUnavailableError as e:
        # Fall back to alternative adapter
        logger.warning(f"Adapter unavailable: {e.reason}, trying {e.fallback_interface}")
        result = await fallback_agent.run(messages)
    except AgentTimeoutError as e:
        # Handle timeout specifically
        logger.error(f"Agent timed out: {e}")
    except PortError as e:
        # Catch-all for any port error
        logger.error(f"Port error: {e}")
"""

from __future__ import annotations

from karenina.exceptions import KareninaError


class PortError(KareninaError):
    """Base exception for all port-related errors.

    All port errors inherit from this class, allowing unified exception
    handling at the port layer.

    Args:
        message: Human-readable description of the error.

    Example:
        try:
            await agent.run(messages)
        except PortError as e:
            logger.error(f"Port error: {e}")
    """

    pass


class AdapterUnavailableError(PortError):
    """Raised when an adapter/backend is not available.

    This error indicates that a particular adapter cannot be used, typically
    because required infrastructure is missing (e.g., Claude CLI not installed,
    API key not configured, service unreachable).

    The `fallback_interface` field suggests an alternative adapter that may
    be used instead, enabling automatic fallback logic in the factory.

    Args:
        message: Human-readable description of the error.
        reason: Specific reason the adapter is unavailable.
        fallback_interface: Suggested alternative interface (e.g., "langchain").

    Example:
        raise AdapterUnavailableError(
            message="Claude Code CLI not found",
            reason="claude executable not in PATH",
            fallback_interface="langchain",
        )
    """

    def __init__(
        self,
        message: str,
        reason: str | None = None,
        fallback_interface: str | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason or message
        self.fallback_interface = fallback_interface


class AgentExecutionError(PortError):
    """Raised when an agent fails during execution.

    This error indicates a runtime failure during agent execution, such as
    a CLI process crash, internal error, or unexpected exception. The
    `stderr` field may contain diagnostic output from the failed process.

    Args:
        message: Human-readable description of the error.
        stderr: Standard error output from the failed process, if available.

    Example:
        raise AgentExecutionError(
            message="CLI process exited with code 1",
            stderr="Error: connection refused",
        )
    """

    def __init__(self, message: str, stderr: str | None = None) -> None:
        super().__init__(message)
        self.stderr = stderr


class AgentTimeoutError(AgentExecutionError):
    """Raised when an agent hits a turn limit or time limit.

    This is a subclass of `AgentExecutionError` for timeout-specific failures.
    It can be raised when:
    - The agent exceeds `max_turns` (recursion limit)
    - The agent exceeds a wall-clock time limit
    - An external timeout kills the process

    Note: Hitting a turn limit may still produce partial results. Check the
    `limit_reached` field in `AgentResult` for partial success scenarios.

    Args:
        message: Human-readable description of the timeout.
        stderr: Standard error output from the timed-out process, if available.

    Example:
        raise AgentTimeoutError(
            message="Agent exceeded max_turns=25",
            stderr="Recursion limit reached",
        )
    """

    pass


class AgentResponseError(PortError):
    """Raised when an agent produces an invalid or malformed response.

    This error indicates the agent completed execution but produced output
    that cannot be processed - for example, invalid JSON, unexpected message
    format, or missing required fields.

    Args:
        message: Human-readable description of the error.

    Example:
        raise AgentResponseError(
            message="Expected JSON response but received malformed data",
        )
    """

    pass


class ParseError(PortError):
    """Raised when structured output parsing fails.

    This error indicates that the LLM-based parser could not extract
    structured data from the response. This may happen when:
    - The response text doesn't contain the expected information
    - The LLM judge cannot map the text to the schema
    - The extracted data doesn't validate against the Pydantic model

    The `raw_response` field contains the original text that failed to parse,
    which can be useful for debugging.

    Args:
        message: Human-readable description of the parse failure.
        raw_response: The original response text that failed to parse.

    Example:
        raise ParseError(
            message="Could not extract Answer schema from response",
            raw_response="The model's response was: ...",
        )
    """

    def __init__(self, message: str, raw_response: str | None = None) -> None:
        super().__init__(message)
        self.raw_response = raw_response
