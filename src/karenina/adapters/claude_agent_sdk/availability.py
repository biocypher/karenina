"""CLI availability checking for Claude Agent SDK.

This module provides utilities to check if the Claude Code CLI is installed
and available for use by the Claude Agent SDK adapter.

The Claude Agent SDK requires the 'claude' CLI binary to be available in PATH.
This module checks for its presence and returns availability information that
can be used by the adapter factory for routing decisions.
"""

import shutil
from dataclasses import dataclass


@dataclass
class AdapterAvailability:
    """Result of checking adapter availability.

    Attributes:
        available: Whether the adapter can be used.
        reason: Human-readable explanation of availability status.
        fallback_interface: Suggested alternative interface if unavailable.

    Example:
        >>> availability = check_claude_cli_available()
        >>> if not availability.available:
        ...     print(f"Claude SDK unavailable: {availability.reason}")
        ...     print(f"Suggested fallback: {availability.fallback_interface}")
    """

    available: bool
    reason: str
    fallback_interface: str | None = None


def check_claude_cli_available() -> AdapterAvailability:
    """Check if Claude Code CLI is installed and available.

    The Claude Agent SDK requires the 'claude' CLI binary to be in PATH.
    This function uses shutil.which() to check for its presence.

    Returns:
        AdapterAvailability with:
        - available=True if 'claude' command is found in PATH
        - available=False with helpful installation message if not found

    Example:
        >>> availability = check_claude_cli_available()
        >>> if availability.available:
        ...     print("Claude CLI is ready to use")
        ... else:
        ...     print(f"Not available: {availability.reason}")

    Notes:
        - The CLI is checked using shutil.which("claude")
        - When unavailable, suggests "langchain" as fallback_interface
        - The installation URL points to https://claude.ai/code
    """
    claude_path = shutil.which("claude")

    if claude_path is not None:
        return AdapterAvailability(
            available=True,
            reason=f"Claude CLI found at: {claude_path}",
        )
    else:
        return AdapterAvailability(
            available=False,
            reason=(
                "Claude Code CLI not found in PATH. "
                "Install from: https://claude.ai/code "
                "or run: npm install -g @anthropic-ai/claude-code"
            ),
            fallback_interface="langchain",
        )
