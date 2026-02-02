"""Trace conversion utilities for Claude Agent SDK messages.

This module provides functions to convert SDK message streams to both the
legacy string format (raw_trace) and the new structured format (trace_messages).

The dual output supports the migration period where both formats are needed:
- raw_trace: Backward compatible string format for database storage and regex highlighting
- trace_messages: Structured list[dict] format for the new frontend component

Functions:
    sdk_messages_to_raw_trace: Convert SDK messages to delimited string format
    sdk_messages_to_trace_messages: Convert SDK messages to structured TraceMessage list

IMPORTANT: SDK places ToolResultBlocks within AssistantMessage.content, not as
separate messages like LangChain. The converters handle this difference.
"""

from __future__ import annotations

from typing import Any


def sdk_messages_to_raw_trace(
    messages: list[Any],
    include_user_messages: bool = False,
) -> str:
    """Convert SDK messages to raw_trace string format.

    Produces the delimited string format used by the legacy frontend and
    database storage. Uses "--- AI Message ---" and "--- Tool Message ---"
    delimiters for backward compatibility with existing pattern matching.

    Args:
        messages: List of SDK message objects (UserMessage, AssistantMessage,
            ResultMessage).
        include_user_messages: If True, include user messages in trace.
            Defaults to False to match LangChain adapter behavior.

    Returns:
        Formatted trace string with message delimiters.

    Example:
        >>> trace = sdk_messages_to_raw_trace(sdk_messages)
        >>> print(trace)
        --- AI Message ---
        I'll search for that information.

        --- Tool Message (call_id: call_abc123) ---
        {"result": "search results..."}

        --- AI Message ---
        Based on the results, the answer is 42.
    """
    try:
        from claude_agent_sdk import AssistantMessage, UserMessage
        from claude_agent_sdk.types import (
            TextBlock,
            ThinkingBlock,
            ToolResultBlock,
            ToolUseBlock,
        )
    except ImportError:
        return "[SDK types not available - cannot format trace]"

    trace_parts: list[str] = []

    for msg in messages:
        if isinstance(msg, UserMessage):
            if include_user_messages:
                content = _extract_user_content(msg)
                if content:
                    trace_parts.append(f"--- Human Message ---\n{content}")
            continue

        elif isinstance(msg, AssistantMessage):
            parts = _format_assistant_message_for_raw_trace(
                msg, TextBlock, ToolUseBlock, ToolResultBlock, ThinkingBlock
            )
            trace_parts.extend(parts)

    return "\n\n".join(trace_parts)


def sdk_messages_to_trace_messages(
    messages: list[Any],
    include_user_messages: bool = False,
) -> list[dict[str, Any]]:
    """Convert SDK messages to structured TraceMessage list.

    Produces the structured format for the new frontend component. Each message
    includes role, content, block_index, and optional metadata like tool_calls,
    tool_result, thinking, and model.

    Note: System messages in Claude SDK are not part of the message stream;
    they are passed via ClaudeAgentOptions.system_prompt. Therefore there is
    no include_system parameter - system prompts don't appear in SDK messages.

    Args:
        messages: List of SDK message objects.
        include_user_messages: If True, include user messages. Defaults to False.

    Returns:
        List of TraceMessage dicts conforming to the TypeScript interface:
        - role: 'system' | 'user' | 'assistant' | 'tool'
        - content: string
        - block_index: number
        - tool_calls?: ToolCall[]
        - tool_result?: ToolResultMeta
        - thinking?: ThinkingMeta
        - model?: string

    Example:
        >>> trace_msgs = sdk_messages_to_trace_messages(sdk_messages)
        >>> for msg in trace_msgs:
        ...     print(f"{msg['role']}: {msg['content'][:50]}...")
    """
    try:
        from claude_agent_sdk import AssistantMessage, UserMessage
        from claude_agent_sdk.types import (
            TextBlock,
            ThinkingBlock,
            ToolResultBlock,
            ToolUseBlock,
        )
    except ImportError:
        return []

    result: list[dict[str, Any]] = []
    block_index = 0

    for msg in messages:
        if isinstance(msg, UserMessage):
            if include_user_messages:
                content = _extract_user_content(msg)
                if content:
                    result.append(
                        {
                            "role": "user",
                            "content": content,
                            "block_index": block_index,
                        }
                    )
                    block_index += 1
            continue

        elif isinstance(msg, AssistantMessage):
            # Process AssistantMessage blocks
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            thinking_meta: dict[str, Any] | None = None

            for block in msg.content:
                if isinstance(block, TextBlock):
                    text_parts.append(block.text)

                elif isinstance(block, ThinkingBlock):
                    # Store thinking metadata
                    thinking_meta = {
                        "thinking": block.thinking,
                    }
                    # Include signature if available
                    if hasattr(block, "signature") and block.signature:
                        thinking_meta["signature"] = block.signature

                elif isinstance(block, ToolUseBlock):
                    tool_calls.append(
                        {
                            "id": block.id,
                            "name": block.name,
                            "input": block.input if isinstance(block.input, dict) else {},
                        }
                    )

                elif isinstance(block, ToolResultBlock):
                    # Tool results as separate messages (matches design spec)
                    content_str = _extract_tool_result_content(block)
                    is_error = getattr(block, "is_error", False)
                    if not isinstance(is_error, bool):
                        is_error = False

                    result.append(
                        {
                            "role": "tool",
                            "content": content_str,
                            "block_index": block_index,
                            "tool_result": {
                                "tool_use_id": block.tool_use_id,
                                "is_error": is_error,
                            },
                        }
                    )
                    block_index += 1

            # Build assistant message if we have content or tool calls
            if text_parts or tool_calls or thinking_meta:
                trace_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": "\n".join(text_parts),
                    "block_index": block_index,
                }

                # Add tool calls if present
                if tool_calls:
                    trace_msg["tool_calls"] = tool_calls

                # Add thinking metadata if present (SDK-only feature)
                if thinking_meta:
                    trace_msg["thinking"] = thinking_meta

                # Add model if available (SDK provides actual model)
                if hasattr(msg, "model") and msg.model:
                    trace_msg["model"] = msg.model

                result.append(trace_msg)
                block_index += 1

    return result


def _extract_user_content(msg: Any) -> str:
    """Extract text content from a UserMessage.

    Args:
        msg: SDK UserMessage object.

    Returns:
        Text content as string.
    """
    if hasattr(msg, "content"):
        content = msg.content
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            parts = []
            for block in content:
                if hasattr(block, "text"):
                    parts.append(block.text)
            return "\n".join(parts)
    return ""


def _extract_tool_result_content(block: Any) -> str:
    """Extract content from a ToolResultBlock.

    Handles various content formats (string, list of text blocks, etc.).

    Args:
        block: SDK ToolResultBlock object.

    Returns:
        Content as string.
    """
    if not hasattr(block, "content") or not block.content:
        return ""

    content = block.content
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for c in content:
            if hasattr(c, "text"):
                parts.append(c.text)
        return "\n".join(parts)
    else:
        return str(content)


def _format_assistant_message_for_raw_trace(
    msg: Any,
    TextBlock: type,
    ToolUseBlock: type,
    ToolResultBlock: type,
    ThinkingBlock: type,
) -> list[str]:
    """Format an AssistantMessage for raw trace output.

    Produces the legacy delimiter format used by existing frontend patterns.

    Args:
        msg: AssistantMessage from SDK.
        TextBlock: SDK TextBlock type.
        ToolUseBlock: SDK ToolUseBlock type.
        ToolResultBlock: SDK ToolResultBlock type.
        ThinkingBlock: SDK ThinkingBlock type.

    Returns:
        List of formatted trace sections.
    """
    result: list[str] = []
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    tool_results: list[tuple[str, str, bool]] = []  # (call_id, content, is_error)

    for block in msg.content:
        if isinstance(block, TextBlock):
            text_parts.append(block.text)  # type: ignore[attr-defined]

        elif isinstance(block, ThinkingBlock):
            # Include thinking in trace with its own header
            result.append(f"--- Thinking ---\n{block.thinking}")  # type: ignore[attr-defined]

        elif isinstance(block, ToolUseBlock):
            tool_calls.append(
                {
                    "name": block.name,  # type: ignore[attr-defined]
                    "id": block.id,  # type: ignore[attr-defined]
                    "args": block.input if isinstance(block.input, dict) else {},  # type: ignore[attr-defined]
                }
            )

        elif isinstance(block, ToolResultBlock):
            content_str = _extract_tool_result_content(block)
            is_error = getattr(block, "is_error", False)
            if not isinstance(is_error, bool):
                is_error = False

            tool_results.append((block.tool_use_id, content_str, is_error))  # type: ignore[attr-defined]

    # Build AI Message section (matches LangChain format)
    if text_parts or tool_calls:
        header = "--- AI Message ---"
        content_parts: list[str] = []

        if text_parts:
            content_parts.append("\n".join(text_parts))

        if tool_calls:
            content_parts.append("\nTool Calls:")
            for tc in tool_calls:
                content_parts.append(f"  {tc['name']} (call_{tc['id']})")
                content_parts.append(f"   Call ID: {tc['id']}")
                if tc["args"]:
                    content_parts.append(f"   Args: {tc['args']}")

        result.append(f"{header}\n{''.join(content_parts)}")

    # Add tool result sections
    for call_id, content, is_error in tool_results:
        header = f"--- Tool Message (call_id: {call_id}) ---"
        if is_error:
            content = f"[ERROR] {content}"
        result.append(f"{header}\n{content}")

    return result
