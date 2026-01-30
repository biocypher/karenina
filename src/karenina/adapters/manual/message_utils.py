"""Message processing utilities for manual traces.

This module provides utilities for processing and extracting content from
message lists, working with the port's Message types and providing
compatibility with LangChain message formats.

The utilities here are used by ManualTraces to convert message lists
to harmonized string traces and extract agent metrics.
"""

import re
from collections import defaultdict
from typing import Any

from karenina.ports.messages import (
    Message,
    Role,
    ToolResultContent,
    ToolUseContent,
)

# ============================================================================
# Tool Call Failure Detection Patterns
# ============================================================================

# Pre-compiled regex patterns for detecting suspected tool failures in agent traces.
TOOL_FAILURE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Error indicators
    re.compile(r"\berror\b", re.IGNORECASE),
    re.compile(r"\bfailed\b", re.IGNORECASE),
    re.compile(r"\bexception\b", re.IGNORECASE),
    re.compile(r"\btraceback\b", re.IGNORECASE),
    re.compile(r"\bstack\s+trace\b", re.IGNORECASE),
    # HTTP errors
    re.compile(r"\b404\b", re.IGNORECASE),
    re.compile(r"\b500\b", re.IGNORECASE),
    re.compile(r"\b502\b", re.IGNORECASE),
    re.compile(r"\b503\b", re.IGNORECASE),
    re.compile(r"\btimeout\b", re.IGNORECASE),
    # API failures
    re.compile(r"\binvalid\b", re.IGNORECASE),
    re.compile(r"\bunauthorized\b", re.IGNORECASE),
    re.compile(r"\bforbidden\b", re.IGNORECASE),
    re.compile(r"\bnot\s+found\b", re.IGNORECASE),
    re.compile(r"\bcannot\b", re.IGNORECASE),
    re.compile(r"\bunable\s+to\b", re.IGNORECASE),
)


# ============================================================================
# LangChain Message Conversion
# ============================================================================


def convert_langchain_messages(messages: list[Any]) -> list[Message]:
    """
    Convert LangChain messages to port Message format.

    Args:
        messages: List of LangChain message objects (AIMessage, HumanMessage, etc.)

    Returns:
        List of port Message objects

    Raises:
        ImportError: If langchain_core is not available
    """
    try:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    except ImportError as e:
        raise ImportError(
            "langchain_core is required for LangChain message conversion. Install with: uv add langchain-core"
        ) from e

    result: list[Message] = []

    for msg in messages:
        if isinstance(msg, SystemMessage):
            content = str(msg.content) if msg.content else ""
            result.append(Message.system(content))

        elif isinstance(msg, HumanMessage):
            content = str(msg.content) if msg.content else ""
            result.append(Message.user(content))

        elif isinstance(msg, AIMessage):
            # Extract text content
            text = str(msg.content) if msg.content else ""

            # Extract tool calls if present
            tool_calls: list[ToolUseContent] = []
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append(
                        ToolUseContent(
                            id=tc.get("id") or "unknown",
                            name=tc.get("name", "unknown"),
                            input=tc.get("args", {}),
                        )
                    )

            result.append(Message.assistant(text, tool_calls=tool_calls if tool_calls else None))

        elif isinstance(msg, ToolMessage):
            tool_use_id = getattr(msg, "tool_call_id", "unknown")
            content = str(msg.content) if msg.content else ""
            # Check for error status
            status = getattr(msg, "status", None)
            is_error = bool(status and isinstance(status, str) and status.lower() in ["error", "failed", "failure"])
            result.append(Message.tool_result(tool_use_id, content, is_error=is_error))

        else:
            # Unknown message type - try to extract content
            content = str(getattr(msg, "content", msg))
            # Assume assistant role for unknown types
            result.append(Message.assistant(content))

    return result


def is_langchain_message_list(messages: list[Any]) -> bool:
    """
    Check if a list contains LangChain message objects.

    Args:
        messages: List to check

    Returns:
        True if the list contains LangChain messages
    """
    if not messages:
        return False

    # Check the first message's class name
    first = messages[0]
    class_name = type(first).__name__

    return class_name in ("AIMessage", "HumanMessage", "SystemMessage", "ToolMessage", "BaseMessage")


def is_port_message_list(messages: list[Any]) -> bool:
    """
    Check if a list contains port Message objects.

    Args:
        messages: List to check

    Returns:
        True if the list contains port Message objects
    """
    if not messages:
        return False

    return isinstance(messages[0], Message)


# ============================================================================
# Message Harmonization
# ============================================================================


def harmonize_messages(messages: list[Message], original_question: str | None = None) -> str:
    """
    Harmonize port Message list into a single string trace.

    Filters out system messages and the first user message (initial question),
    but preserves summary messages from SummarizationMiddleware.

    Args:
        messages: List of port Message objects
        original_question: The original user question. If provided, enables
                          detection of summary messages.

    Returns:
        Formatted string trace of assistant and tool messages
    """
    trace_parts: list[str] = []
    first_user_found = False

    for msg in messages:
        # Skip system messages
        if msg.role == Role.SYSTEM:
            continue

        # Handle user messages
        if msg.role == Role.USER:
            # Check if this is a summary message (differs from original question)
            if original_question is not None:
                content = msg.text.strip()
                if content != original_question.strip():
                    # This is a summary - include it
                    formatted = _format_message(msg)
                    if formatted:
                        trace_parts.append(formatted)
                    first_user_found = True
                    continue

            # Skip first user message (initial question)
            if not first_user_found:
                first_user_found = True
                continue

        # Include assistant, tool, and subsequent user messages
        formatted = _format_message(msg)
        if formatted:
            trace_parts.append(formatted)

    return "\n\n".join(trace_parts)


def _format_message(msg: Message) -> str:
    """
    Format a single Message for trace output.

    Args:
        msg: A port Message object

    Returns:
        Formatted message string with header and content
    """
    if msg.role == Role.ASSISTANT:
        header = "--- AI Message ---"
        parts: list[str] = []

        # Add text content
        text = msg.text
        if text:
            parts.append(text)

        # Add tool calls
        tool_calls = msg.tool_calls
        if tool_calls:
            parts.append("\nTool Calls:")
            for tc in tool_calls:
                parts.append(f"  {tc.name} (call_{tc.id})")
                parts.append(f"   Call ID: {tc.id}")
                if tc.input:
                    parts.append(f"   Args: {tc.input}")

        content = "\n".join(parts) if parts else ""
        return f"{header}\n{content}" if content else header

    elif msg.role == Role.TOOL:
        # Get tool result content
        tool_results = [c for c in msg.content if isinstance(c, ToolResultContent)]
        if tool_results:
            tr = tool_results[0]
            header = f"--- Tool Message (call_id: {tr.tool_use_id}) ---"
            return f"{header}\n{tr.content}" if tr.content else header
        return ""

    elif msg.role == Role.USER:
        header = "--- Human Message ---"
        text = msg.text
        return f"{header}\n{text}" if text else header

    else:
        # System or unknown
        header = f"--- {msg.role.value.title()} Message ---"
        text = msg.text
        return f"{header}\n{text}" if text else header


# ============================================================================
# Agent Metrics Extraction
# ============================================================================


def extract_agent_metrics(messages: list[Message]) -> dict[str, Any]:
    """
    Extract agent execution metrics from port Message list.

    Tracks:
    - Iterations (assistant message cycles)
    - Tool calls (successful tool invocations)
    - Tools used (unique tool names)
    - Per-tool call counts
    - Suspected failed tool calls (tools with error-like output patterns)

    Args:
        messages: List of port Message objects

    Returns:
        Dict with agent metrics
    """
    iterations = 0
    tool_calls = 0
    tools_used: set[str] = set()
    tool_call_counts: dict[str, int] = defaultdict(int)
    suspect_failed_tool_calls = 0
    suspect_failed_tools: set[str] = set()

    for msg in messages:
        # Count assistant messages as iterations
        if msg.role == Role.ASSISTANT:
            iterations += 1

            # Track tool calls requested by assistant
            for tc in msg.tool_calls:
                tools_used.add(tc.name)
                tool_call_counts[tc.name] += 1

        # Count tool result messages
        elif msg.role == Role.TOOL:
            for content_block in msg.content:
                if isinstance(content_block, ToolResultContent):
                    tool_calls += 1

                    # Check for suspected failures
                    is_suspect = content_block.is_error
                    if not is_suspect and content_block.content:
                        for pattern in TOOL_FAILURE_PATTERNS:
                            if pattern.search(content_block.content):
                                is_suspect = True
                                break

                    if is_suspect:
                        suspect_failed_tool_calls += 1
                        # Try to find tool name from the tool_use_id
                        suspect_failed_tools.add(f"tool_{content_block.tool_use_id}")

    return {
        "iterations": iterations,
        "tool_calls": tool_calls,
        "tools_used": sorted(tools_used),
        "tool_call_counts": dict(tool_call_counts),
        "suspect_failed_tool_calls": suspect_failed_tool_calls,
        "suspect_failed_tools": sorted(suspect_failed_tools),
        "model_call_limit_reached": False,
        "tool_call_limit_reached": False,
        "summarization_triggered": False,
        "model_retries": 0,
        "tool_retries": 0,
    }


def extract_agent_metrics_from_langchain(messages: list[Any]) -> dict[str, Any]:
    """
    Extract agent metrics from LangChain message list.

    This is a convenience wrapper that converts LangChain messages
    to port format before extracting metrics.

    Args:
        messages: List of LangChain message objects

    Returns:
        Dict with agent metrics
    """
    port_messages = convert_langchain_messages(messages)
    return extract_agent_metrics(port_messages)


# ============================================================================
# Public API
# ============================================================================


def preprocess_message_list(
    messages: list[Any],
    original_question: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Preprocess a message list (LangChain or port format) into harmonized trace and metrics.

    This is the main entry point for processing message lists in ManualTraces.

    Args:
        messages: List of messages (LangChain or port Message format)
        original_question: The original user question (for summary detection)

    Returns:
        Tuple of (harmonized_trace_string, agent_metrics_dict)

    Raises:
        TypeError: If message format is unrecognized
    """
    # Determine message format and convert if needed
    if is_port_message_list(messages):
        port_messages = messages
    elif is_langchain_message_list(messages):
        port_messages = convert_langchain_messages(messages)
    else:
        raise TypeError(
            f"Unrecognized message format. Expected list of port Message or LangChain messages, "
            f"got list containing {type(messages[0]).__name__ if messages else 'empty'}"
        )

    # Extract metrics and harmonize
    agent_metrics = extract_agent_metrics(port_messages)
    harmonized_trace = harmonize_messages(port_messages, original_question)

    return harmonized_trace, agent_metrics


__all__ = [
    "convert_langchain_messages",
    "is_langchain_message_list",
    "is_port_message_list",
    "harmonize_messages",
    "extract_agent_metrics",
    "extract_agent_metrics_from_langchain",
    "preprocess_message_list",
    "TOOL_FAILURE_PATTERNS",
]
