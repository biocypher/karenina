"""Claude Agent SDK error wrapping.

This module wraps Claude Agent SDK exceptions into karenina port errors,
providing consistent error handling across adapters.

SDK Exception Mapping:
    CLINotFoundError    -> AdapterUnavailableError (CLI not installed)
    CLIConnectionError  -> AgentExecutionError (connection failure)
    ProcessError        -> AgentExecutionError or AgentTimeoutError (exit codes 124, 137)
    CLIJSONDecodeError  -> AgentResponseError (malformed response)
    Other exceptions    -> AgentExecutionError (fallback)

Example:
    >>> from karenina.adapters.claude_agent_sdk.errors import wrap_sdk_error
    >>> from karenina.ports import AdapterUnavailableError, AgentTimeoutError
    >>>
    >>> try:
    ...     async for msg in query(prompt="hello"):
    ...         pass
    ... except Exception as e:
    ...     raise wrap_sdk_error(e) from e
"""

from __future__ import annotations

from karenina.ports.errors import (
    AdapterUnavailableError,
    AgentExecutionError,
    AgentResponseError,
    AgentTimeoutError,
    PortError,
)

# Exit codes that indicate timeout
_TIMEOUT_EXIT_CODES = frozenset(
    {
        124,  # timeout command
        137,  # SIGKILL (128 + 9)
    }
)


def wrap_sdk_error(e: Exception) -> PortError:
    """Wrap a Claude Agent SDK exception into a karenina port error.

    This function provides a consistent error translation layer between
    the Claude Agent SDK and karenina's port error hierarchy. It maps
    SDK-specific exceptions to appropriate port errors with informative
    messages.

    Args:
        e: The SDK exception to wrap.

    Returns:
        A PortError subclass appropriate for the exception type.

    Raises:
        The returned PortError (caller should raise it).

    Example:
        >>> try:
        ...     await run_sdk_query()
        ... except Exception as e:
        ...     raise wrap_sdk_error(e) from e

    Mapping:
        - CLINotFoundError -> AdapterUnavailableError with install instructions
        - CLIConnectionError -> AgentExecutionError
        - ProcessError with exit code 124/137 -> AgentTimeoutError
        - ProcessError with other exit codes -> AgentExecutionError with stderr
        - CLIJSONDecodeError -> AgentResponseError with line info
        - Other exceptions -> AgentExecutionError (generic)
    """
    # Handle SDK exceptions by checking type name (avoids import requirement)
    # This allows the module to work even when SDK is not installed
    exc_type_name = type(e).__name__

    if exc_type_name == "CLINotFoundError":
        # CLI not installed - provide helpful installation instructions
        cli_path = getattr(e, "cli_path", "claude")
        return AdapterUnavailableError(
            message=(
                f"Claude Code CLI not found at '{cli_path}'. "
                "Install with: npm install -g @anthropic-ai/claude-code "
                "or visit https://claude.ai/code for setup instructions."
            ),
            reason="Claude Code CLI not installed",
            fallback_interface="langchain",
        )

    if exc_type_name == "CLIConnectionError":
        # Connection failure - could be transient
        return AgentExecutionError(
            message=f"Failed to connect to Claude Code CLI: {e}",
            stderr=str(e),
        )

    if exc_type_name == "ProcessError":
        # CLI process failed - check for timeout signals
        exit_code = getattr(e, "exit_code", None)
        stderr = getattr(e, "stderr", None)
        stderr_str = stderr if isinstance(stderr, str) else str(stderr) if stderr else None

        if exit_code in _TIMEOUT_EXIT_CODES:
            return AgentTimeoutError(
                message=f"Claude Code CLI process timed out (exit code {exit_code})",
                stderr=stderr_str,
            )

        # General process failure
        msg_parts = ["Claude Code CLI process failed"]
        if exit_code is not None:
            msg_parts.append(f"(exit code {exit_code})")
        if stderr_str:
            # Truncate stderr if very long
            truncated_stderr = stderr_str[:500] + "..." if len(stderr_str) > 500 else stderr_str
            msg_parts.append(f": {truncated_stderr}")

        return AgentExecutionError(
            message=" ".join(msg_parts),
            stderr=stderr_str,
        )

    if exc_type_name == "CLIJSONDecodeError":
        # Malformed JSON response from CLI
        line = getattr(e, "line", None)
        line_info = f" at line: {line[:200]}..." if line and len(line) > 200 else f" at line: {line}" if line else ""
        return AgentResponseError(
            message=f"Invalid JSON response from Claude Code CLI{line_info}",
        )

    # Generic fallback for unknown SDK errors or non-SDK exceptions
    # Preserve the original exception type name for debugging
    return AgentExecutionError(
        message=f"Claude Agent SDK error ({exc_type_name}): {e}",
        stderr=str(e) if str(e) else None,
    )
