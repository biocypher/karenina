"""Trace conversion and harmonization utilities for LangChain messages.

This module provides utilities for:
1. Converting LangChain message history to standardized trace formats
2. Harmonizing LangGraph agent responses into trace strings
3. Extracting final AI messages from agent responses

Two output formats:
    - raw_trace: String with "--- Message Type ---" delimiters (legacy/database)
    - trace_messages: List of dicts matching TypeScript TraceMessage interface

Example:
    >>> from langchain_core.messages import AIMessage, ToolMessage
    >>> messages = [
    ...     AIMessage(content="Let me search for that."),
    ...     ToolMessage(content="results...", tool_call_id="call_abc"),
    ...     AIMessage(content="The answer is 42."),
    ... ]
    >>> raw = langchain_messages_to_raw_trace(messages)
    >>> structured = langchain_messages_to_trace_messages(messages)
    >>> # Or harmonize agent response directly
    >>> trace = harmonize_agent_response({"messages": messages})
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


def langchain_messages_to_raw_trace(
    messages: list[BaseMessage],
    include_system: bool = False,
) -> str:
    """Convert LangChain messages to raw trace string format.

    This produces the legacy format with "--- Message Type ---" delimiters that
    is used for:
    - Database storage (raw_llm_response TEXT column)
    - Regex-based highlighting in the frontend
    - Rubric evaluation on raw text

    Args:
        messages: List of LangChain BaseMessage instances
        include_system: If True, include SystemMessage blocks in output

    Returns:
        Trace string with message blocks separated by double newlines

    Format:
        --- AI Message ---
        I'll search for that information.

        Tool Calls:
          search (call_abc123)
           Call ID: abc123
           Args: {"query": "..."}

        --- Tool Message (call_id: call_abc123) ---
        {"result": "search results..."}

        --- AI Message ---
        Based on the results, the answer is 42.
    """
    try:
        from langchain_core.messages import SystemMessage
    except ImportError:
        # Fallback if langchain_core not available
        return _fallback_to_raw_trace(messages, include_system)

    trace_parts: list[str] = []

    for msg in messages:
        # Skip system messages unless requested
        if isinstance(msg, SystemMessage) and not include_system:
            continue

        formatted = _format_langchain_message(msg)
        if formatted.strip():
            trace_parts.append(formatted)

    return "\n\n".join(trace_parts)


def langchain_messages_to_trace_messages(
    messages: list[BaseMessage],
    include_system: bool = False,
) -> list[dict[str, Any]]:
    """Convert LangChain messages to structured TraceMessage list.

    This produces the new structured format that matches the TypeScript
    TraceMessage interface used by the frontend structured view.

    Args:
        messages: List of LangChain BaseMessage instances
        include_system: If True, include system messages in output

    Returns:
        List of TraceMessage dicts with keys:
            - role: "system" | "user" | "assistant" | "tool"
            - content: str (message text)
            - block_index: int (for navigation)
            - tool_calls?: list[ToolCall] (on assistant messages with tools)
            - tool_result?: ToolResultMeta (on tool messages)

    Example output:
        [
            {
                "role": "assistant",
                "content": "Let me search for that.",
                "block_index": 0,
                "tool_calls": [
                    {"id": "call_abc", "name": "search", "input": {"query": "..."}}
                ]
            },
            {
                "role": "tool",
                "content": "results...",
                "block_index": 1,
                "tool_result": {"tool_use_id": "call_abc", "is_error": False}
            },
            {
                "role": "assistant",
                "content": "The answer is 42.",
                "block_index": 2
            }
        ]
    """
    try:
        from langchain_core.messages import (
            AIMessage,
            HumanMessage,
            SystemMessage,
            ToolMessage,
        )
    except ImportError:
        # Fallback if langchain_core not available
        return _fallback_to_trace_messages(messages, include_system)

    result: list[dict[str, Any]] = []
    block_index = 0

    for msg in messages:
        # Skip system messages unless requested
        if isinstance(msg, SystemMessage) and not include_system:
            continue

        if isinstance(msg, SystemMessage):
            result.append(
                {
                    "role": "system",
                    "content": str(msg.content) if msg.content else "",
                    "block_index": block_index,
                }
            )
            block_index += 1

        elif isinstance(msg, HumanMessage):
            result.append(
                {
                    "role": "user",
                    "content": str(msg.content) if msg.content else "",
                    "block_index": block_index,
                }
            )
            block_index += 1

        elif isinstance(msg, AIMessage):
            trace_msg: dict[str, Any] = {
                "role": "assistant",
                "content": str(msg.content) if msg.content else "",
                "block_index": block_index,
            }

            # Add tool_calls if present
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_calls = []
                for tc in msg.tool_calls:
                    if isinstance(tc, dict):
                        tool_calls.append(
                            {
                                "id": tc.get("id") or "",
                                "name": tc.get("name") or "",
                                "input": tc.get("args") or {},
                            }
                        )
                    else:
                        # Object-style tool call
                        tool_calls.append(
                            {
                                "id": getattr(tc, "id", "") or "",
                                "name": getattr(tc, "name", "") or "",
                                "input": getattr(tc, "args", {}) or {},
                            }
                        )
                if tool_calls:
                    trace_msg["tool_calls"] = tool_calls

            result.append(trace_msg)
            block_index += 1

        elif isinstance(msg, ToolMessage):
            content = str(msg.content) if msg.content else ""
            tool_call_id = getattr(msg, "tool_call_id", "") or ""

            result.append(
                {
                    "role": "tool",
                    "content": content,
                    "block_index": block_index,
                    "tool_result": {
                        "tool_use_id": tool_call_id,
                        "is_error": _detect_tool_error(content),
                    },
                }
            )
            block_index += 1

    return result


def _detect_tool_error(content: str) -> bool:
    """Heuristic for detecting tool errors.

    LangChain ToolMessage lacks an is_error field, so we use pattern matching
    to detect error responses. This is imperfect but matches the design spec.

    Args:
        content: The tool result content string

    Returns:
        True if the content appears to contain an error
    """
    if not content:
        return False

    error_patterns = [
        "error",
        "failed",
        "exception",
        "traceback",
        "permission denied",
        "not found",
        "timeout",
    ]
    content_lower = content.lower()
    return any(pattern in content_lower for pattern in error_patterns)


def _format_langchain_message(msg: Any) -> str:
    """Format a single LangChain message for raw trace output.

    Uses simple dashes (---) as delimiters to avoid Excel formula confusion.

    Args:
        msg: A LangChain message object (AIMessage, ToolMessage, HumanMessage, etc.)

    Returns:
        Formatted message string with header and content
    """
    try:
        from langchain_core.messages import (
            AIMessage,
            HumanMessage,
            SystemMessage,
            ToolMessage,
        )
    except ImportError:
        # Fallback for non-LangChain messages
        msg_type = type(msg).__name__
        content = str(getattr(msg, "content", msg))
        return f"--- {msg_type} ---\n{content}"

    # Format based on message type
    if isinstance(msg, SystemMessage):
        header = "--- System Message ---"
        content = str(msg.content) if msg.content else ""

    elif isinstance(msg, HumanMessage):
        header = "--- Human Message ---"
        content = str(msg.content) if msg.content else ""

    elif isinstance(msg, AIMessage):
        header = "--- AI Message ---"
        content_parts: list[str] = []

        # Add main content if present
        if msg.content:
            content_parts.append(str(msg.content))

        # Add tool calls if present
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            content_parts.append("\nTool Calls:")
            for tool_call in msg.tool_calls:
                if isinstance(tool_call, dict):
                    tool_name = tool_call.get("name", "unknown")
                    tool_id = tool_call.get("id", "unknown")
                    tool_args = tool_call.get("args", {})
                else:
                    # Object-style tool call
                    tool_name = getattr(tool_call, "name", "unknown")
                    tool_id = getattr(tool_call, "id", "unknown")
                    tool_args = getattr(tool_call, "args", {})

                content_parts.append(f"  {tool_name} (call_{tool_id})")
                content_parts.append(f"   Call ID: {tool_id}")
                if tool_args:
                    content_parts.append(f"   Args: {tool_args}")

        content = "\n".join(content_parts) if content_parts else ""

    elif isinstance(msg, ToolMessage):
        tool_call_id = getattr(msg, "tool_call_id", "unknown")
        header = f"--- Tool Message (call_id: {tool_call_id}) ---"
        content = str(msg.content) if msg.content else ""

    else:
        # Fallback for unknown message types
        msg_type = type(msg).__name__
        header = f"--- {msg_type} ---"
        content = str(getattr(msg, "content", msg))

    return f"{header}\n{content}" if content else header


def _fallback_to_raw_trace(messages: list[Any], include_system: bool) -> str:
    """Fallback trace formatting when langchain_core not available.

    Args:
        messages: List of message objects with content attribute
        include_system: Whether to include system messages

    Returns:
        Formatted trace string
    """
    trace_parts: list[str] = []

    for msg in messages:
        msg_type = type(msg).__name__.lower()
        role = getattr(msg, "role", "").lower() if hasattr(msg, "role") else ""

        # Skip system messages unless requested
        if ("system" in msg_type or role == "system") and not include_system:
            continue

        # Format based on type hints in name/role
        if "ai" in msg_type or role == "assistant":
            header = "--- AI Message ---"
        elif "tool" in msg_type or role == "tool":
            tool_call_id = getattr(msg, "tool_call_id", "unknown")
            header = f"--- Tool Message (call_id: {tool_call_id}) ---"
        elif "human" in msg_type or role in ("user", "human"):
            header = "--- Human Message ---"
        elif "system" in msg_type or role == "system":
            header = "--- System Message ---"
        else:
            header = f"--- {type(msg).__name__} ---"

        content = str(getattr(msg, "content", msg))
        formatted = f"{header}\n{content}" if content else header
        trace_parts.append(formatted)

    return "\n\n".join(trace_parts)


def _fallback_to_trace_messages(messages: list[Any], include_system: bool) -> list[dict[str, Any]]:
    """Fallback structured trace when langchain_core not available.

    Args:
        messages: List of message objects with content attribute
        include_system: Whether to include system messages

    Returns:
        List of TraceMessage dicts
    """
    result: list[dict[str, Any]] = []
    block_index = 0

    for msg in messages:
        msg_type = type(msg).__name__.lower()
        role = getattr(msg, "role", "").lower() if hasattr(msg, "role") else ""
        content = str(getattr(msg, "content", ""))

        # Skip system messages unless requested
        if ("system" in msg_type or role == "system") and not include_system:
            continue

        # Determine role from type name or role attribute
        if "ai" in msg_type or role == "assistant":
            trace_role = "assistant"
        elif "tool" in msg_type or role == "tool":
            trace_role = "tool"
        elif "human" in msg_type or role in ("user", "human"):
            trace_role = "user"
        elif "system" in msg_type or role == "system":
            trace_role = "system"
        else:
            trace_role = "assistant"  # Default fallback

        trace_msg: dict[str, Any] = {
            "role": trace_role,
            "content": content,
            "block_index": block_index,
        }

        # Add tool_result metadata for tool messages
        if trace_role == "tool":
            tool_call_id = getattr(msg, "tool_call_id", "") or ""
            trace_msg["tool_result"] = {
                "tool_use_id": tool_call_id,
                "is_error": _detect_tool_error(content),
            }

        result.append(trace_msg)
        block_index += 1

    return result


# ==============================================================================
# Agent Response Harmonization
# ==============================================================================


def harmonize_agent_response(response: Any, original_question: str | None = None) -> str:
    """Harmonize LangGraph agent response into a single trace string.

    LangGraph agents return multiple messages instead of a single response.
    This function extracts AI and Tool messages (excluding system prompts and
    the first human message) and returns the complete agent execution trace.

    Args:
        response: Response from a LangGraph agent, which may be:
            - A single message with content attribute
            - List of messages
            - Dict with 'messages' key (from ainvoke)
            - Nested dict {'agent': {'messages': [...]}} (from astream)
        original_question: The original user question. If provided, enables
            reliable detection of summary messages from SummarizationMiddleware.
            If the first HumanMessage differs from this, it's treated as a
            summary and included in the trace.

    Returns:
        A single string containing the complete agent trace with reasoning
        and tool usage in the "--- Message Type ---" format.

    Example:
        >>> from langchain_core.messages import AIMessage, ToolMessage
        >>> messages = [
        ...     AIMessage(content="I need to search for information."),
        ...     ToolMessage(content="Search results: ...", tool_call_id="call_123"),
        ...     AIMessage(content="Based on the search, the answer is 42.")
        ... ]
        >>> trace = harmonize_agent_response({"messages": messages})
        >>> print("Full agent trace with reasoning and tool usage")
    """
    if response is None:
        return ""

    # Handle single message with content attribute
    if hasattr(response, "content"):
        return str(response.content)

    # Handle nested agent state dict (from astream): {'agent': {'messages': [...]}}
    if isinstance(response, dict):
        if "agent" in response and isinstance(response["agent"], dict) and "messages" in response["agent"]:
            messages = response["agent"]["messages"]
            return _extract_agent_trace(messages, original_question)
        # Handle flat state dict with 'messages' key (from ainvoke)
        elif "messages" in response:
            messages = response["messages"]
            return _extract_agent_trace(messages, original_question)

    # Handle list of messages directly
    if isinstance(response, list):
        return _extract_agent_trace(response, original_question)

    # Fallback: convert to string
    return str(response)


def _is_summary_message(msg: Any, original_question: str | None = None) -> bool:
    """Check if a message is a summarization middleware summary message.

    Detection methods (in order of reliability):
    1. If original_question is provided, check if the message content differs from it
    2. Fall back to pattern matching on known summary prefixes

    Args:
        msg: A message object to check
        original_question: The original user question (if known). If the HumanMessage
            content doesn't match this, it's likely a summary.

    Returns:
        True if the message appears to be a summary message
    """
    if not hasattr(msg, "content") or not msg.content:
        return False

    content = str(msg.content).strip()

    # Method 1: Compare with original question (most reliable)
    if original_question is not None:
        # If content differs significantly from original question, it's a summary
        original_normalized = original_question.strip()
        return content != original_normalized

    # Method 2: Fall back to pattern matching (less reliable but works without context)
    content_lower = content.lower()
    summary_markers = [
        "here is a summary of the conversation",
        "summary of the conversation to date",
        "previous conversation was too long to summarize",
        "conversation summary:",
        "summary of previous conversation",
    ]
    return any(marker in content_lower for marker in summary_markers)


def _extract_agent_trace(messages: list[Any], original_question: str | None = None) -> str:
    """Extract the complete agent trace from a list of messages.

    Filters out system messages and the first human message (initial user question),
    but preserves:
    - Subsequent human messages that may contain agent reasoning steps
    - Summary messages from SummarizationMiddleware (detected by comparing with original_question)

    Args:
        messages: List of LangChain messages
        original_question: The original user question. If provided, enables reliable
            detection of summary messages (if first HumanMessage differs from this,
            it's treated as a summary and included in trace).

    Returns:
        The formatted trace of AI, Tool, and intermediate Human messages
        (excluding first unless it's a summary), empty string if none found
    """
    try:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    except ImportError:
        # Fallback: use type name checking
        return _extract_agent_trace_fallback(messages, original_question)

    trace_parts: list[str] = []
    first_human_found = False

    for msg in messages:
        # Skip system messages
        if isinstance(msg, SystemMessage):
            continue

        # Handle human messages
        if isinstance(msg, HumanMessage):
            # Check if this is a summary message (not the original question)
            if _is_summary_message(msg, original_question):
                formatted_msg = _format_langchain_message(msg)
                if formatted_msg.strip():
                    trace_parts.append(formatted_msg.strip())
                first_human_found = True
                continue

            # Skip first human message (initial user question)
            if not first_human_found:
                first_human_found = True
                continue
            # Continue processing subsequent human messages

        # Include AI, Tool, and subsequent Human messages
        if isinstance(msg, AIMessage | ToolMessage | HumanMessage):
            formatted_msg = _format_langchain_message(msg)
            if formatted_msg.strip():
                trace_parts.append(formatted_msg.strip())

    return "\n\n".join(trace_parts) if trace_parts else ""


def _extract_agent_trace_fallback(messages: list[Any], original_question: str | None = None) -> str:
    """Fallback trace extraction when langchain_core not available."""
    trace_parts: list[str] = []
    first_human_found = False

    for msg in messages:
        msg_type = type(msg).__name__.lower()
        role = getattr(msg, "role", "").lower() if hasattr(msg, "role") else ""

        # Skip system messages
        if "system" in msg_type or role == "system":
            continue

        # Handle human messages
        if "human" in msg_type or role in ("user", "human"):
            # Check if this is a summary message
            if _is_summary_message(msg, original_question):
                formatted_msg = _format_langchain_message(msg)
                if formatted_msg.strip():
                    trace_parts.append(formatted_msg.strip())
                first_human_found = True
                continue

            # Skip first human message
            if not first_human_found:
                first_human_found = True
                continue

        # Include AI, Tool, and subsequent Human messages
        if (
            "ai" in msg_type
            or role == "assistant"
            or "tool" in msg_type
            or role == "tool"
            or "human" in msg_type
            or role in ("user", "human")
        ):
            formatted_msg = _format_langchain_message(msg)
            if formatted_msg.strip():
                trace_parts.append(formatted_msg.strip())

    return "\n\n".join(trace_parts) if trace_parts else ""


def extract_final_ai_message_from_response(response: Any) -> tuple[str | None, str | None]:
    """Extract only the final AI text response from an agent response.

    This function works with the original agent response before harmonization,
    checking message types directly rather than parsing strings.

    Args:
        response: Response from a LangGraph agent (messages list, dict with
            'messages', or state dict)

    Returns:
        Tuple of (extracted_message, error_message) where:
        - extracted_message: The final AI text content, or None if extraction failed
        - error_message: Error description if extraction failed, or None if successful

    Error cases:
        - Empty or no messages
        - Last message is not an AIMessage
        - Final AIMessage has no text content (only tool calls)

    Example:
        >>> from langchain_core.messages import AIMessage
        >>> messages = [AIMessage(content="Final answer")]
        >>> message, error = extract_final_ai_message_from_response(messages)
        >>> print(message)  # "Final answer"
    """
    try:
        from langchain_core.messages import AIMessage
    except ImportError:
        return None, "langchain_core not available"

    # Extract messages list from various response formats
    messages = None

    if response is None:
        return None, "Empty response"

    # Handle nested agent state dict: {'agent': {'messages': [...]}}
    if isinstance(response, dict):
        if "agent" in response and isinstance(response["agent"], dict) and "messages" in response["agent"]:
            messages = response["agent"]["messages"]
        elif "messages" in response:
            messages = response["messages"]
    # Handle list of messages directly
    elif isinstance(response, list):
        messages = response
    # Handle single message
    elif isinstance(response, AIMessage):
        messages = [response]

    if not messages or len(messages) == 0:
        return None, "No messages found in response"

    # Get the last message
    last_message = messages[-1]

    # Check if it's an AIMessage
    if not isinstance(last_message, AIMessage):
        return None, "Last message is not an AIMessage"

    # Extract content
    content = last_message.content if last_message.content else ""

    # Handle list content (convert to string)
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                text_parts.append(str(item["text"]))
        content = " ".join(text_parts) if text_parts else ""

    # Check if content is empty
    if not content or not content.strip():
        # Check if there are tool calls but no text content
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return None, "Final AI message has no text content (only tool calls)"
        return None, "Final AI message has no content"

    return content.strip(), None
