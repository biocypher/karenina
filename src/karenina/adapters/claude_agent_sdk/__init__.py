"""Claude Agent SDK adapter - native Anthropic integration using Claude Agent SDK.

This adapter provides implementations of the port interfaces (LLMPort, AgentPort,
ParserPort) using the Claude Agent SDK. It enables native integration with
Claude Code CLI and the Claude Agent SDK for agent loops with MCP support.

IMPORTANT: This adapter requires the Claude Code CLI to be installed and
available in PATH. Check availability using check_claude_cli_available().

Adapter classes:
    - ClaudeSDKLLMAdapter: Simple LLM invocation via query()
    - ClaudeSDKAgentAdapter: Agent loops via ClaudeSDKClient with MCP support
    - ClaudeSDKParserAdapter: Structured output parsing via query() with output_format

Utilities:
    - ClaudeSDKMessageConverter: Convert between unified Message and SDK types
    - check_claude_cli_available: Check if Claude CLI is installed
    - convert_mcp_config: Convert karenina MCP config to SDK format
    - extract_sdk_usage: Extract UsageMetadata from SDK responses

Example:
    >>> from karenina.adapters.claude_agent_sdk import (
    ...     ClaudeSDKAgentAdapter,
    ...     check_claude_cli_available,
    ... )
    >>> from karenina.schemas.workflow.models import ModelConfig
    >>>
    >>> # Check if Claude SDK is available
    >>> availability = check_claude_cli_available()
    >>> if availability.available:
    ...     config = ModelConfig(
    ...         id="claude-sonnet",
    ...         model_name="claude-sonnet-4-20250514",
    ...         model_provider="anthropic",
    ...         interface="claude_agent_sdk"
    ...     )
    ...     adapter = ClaudeSDKAgentAdapter(config)
    ...     result = await adapter.run(messages=[Message.user("Hello!")])
"""

from typing import TYPE_CHECKING, Any

# Exports defined for when submodules are implemented
__all__ = [
    # Adapter classes (sdk-003, sdk-004, sdk-005)
    "ClaudeSDKLLMAdapter",
    "ClaudeSDKAgentAdapter",
    "ClaudeSDKParserAdapter",
    # Message conversion (sdk-002)
    "ClaudeSDKMessageConverter",
    # Availability checking (sdk-009)
    "check_claude_cli_available",
    # MCP configuration (sdk-006)
    "convert_mcp_config",
    # Usage extraction (sdk-007)
    "extract_sdk_usage",
    # Trace utilities (sdk-008)
    "sdk_messages_to_raw_trace",
    "sdk_messages_to_trace_messages",
    # Error wrapping (sdk-010)
    "wrap_sdk_error",
]

if TYPE_CHECKING:
    # Type hints for IDE support before implementation
    from karenina.adapters.claude_agent_sdk.agent import ClaudeSDKAgentAdapter
    from karenina.adapters.claude_agent_sdk.availability import check_claude_cli_available
    from karenina.adapters.claude_agent_sdk.errors import wrap_sdk_error
    from karenina.adapters.claude_agent_sdk.llm import ClaudeSDKLLMAdapter
    from karenina.adapters.claude_agent_sdk.mcp import convert_mcp_config
    from karenina.adapters.claude_agent_sdk.messages import ClaudeSDKMessageConverter
    from karenina.adapters.claude_agent_sdk.parser import ClaudeSDKParserAdapter
    from karenina.adapters.claude_agent_sdk.trace import (
        sdk_messages_to_raw_trace,
        sdk_messages_to_trace_messages,
    )
    from karenina.adapters.claude_agent_sdk.usage import extract_sdk_usage


def __getattr__(name: str) -> Any:
    """Lazy import adapter classes to avoid circular imports.

    This allows importing from karenina.adapters.claude_agent_sdk before all
    submodules are implemented. Each import is resolved on first access.
    """
    if name == "ClaudeSDKLLMAdapter":
        from karenina.adapters.claude_agent_sdk.llm import ClaudeSDKLLMAdapter

        return ClaudeSDKLLMAdapter

    if name == "ClaudeSDKAgentAdapter":
        from karenina.adapters.claude_agent_sdk.agent import ClaudeSDKAgentAdapter

        return ClaudeSDKAgentAdapter

    if name == "ClaudeSDKParserAdapter":
        from karenina.adapters.claude_agent_sdk.parser import ClaudeSDKParserAdapter

        return ClaudeSDKParserAdapter

    if name == "ClaudeSDKMessageConverter":
        from karenina.adapters.claude_agent_sdk.messages import ClaudeSDKMessageConverter

        return ClaudeSDKMessageConverter

    if name == "check_claude_cli_available":
        from karenina.adapters.claude_agent_sdk.availability import check_claude_cli_available

        return check_claude_cli_available

    if name == "convert_mcp_config":
        from karenina.adapters.claude_agent_sdk.mcp import convert_mcp_config

        return convert_mcp_config

    if name == "extract_sdk_usage":
        from karenina.adapters.claude_agent_sdk.usage import extract_sdk_usage

        return extract_sdk_usage

    if name == "sdk_messages_to_raw_trace":
        from karenina.adapters.claude_agent_sdk.trace import sdk_messages_to_raw_trace

        return sdk_messages_to_raw_trace

    if name == "sdk_messages_to_trace_messages":
        from karenina.adapters.claude_agent_sdk.trace import sdk_messages_to_trace_messages

        return sdk_messages_to_trace_messages

    if name == "wrap_sdk_error":
        from karenina.adapters.claude_agent_sdk.errors import wrap_sdk_error

        return wrap_sdk_error

    raise AttributeError(f"module 'karenina.adapters.claude_agent_sdk' has no attribute '{name}'")
