"""Trace conversion utilities for the Claude Tool adapter.

Converts port Message objects to both legacy raw_trace string format
and structured trace_messages list[dict] format.

Unlike the LangChain and Claude Agent SDK adapters which work with their
native message types, this adapter already works with port Message objects,
making conversion straightforward.
"""

from __future__ import annotations

from typing import Any

from karenina.ports.messages import (
    Message,
    Role,
    TextContent,
    ThinkingContent,
    ToolResultContent,
    ToolUseContent,
)


def claude_tool_messages_to_raw_trace(messages: list[Message]) -> str:
    """Build raw_trace string from collected messages.

    Produces a format compatible with existing infrastructure that uses
    delimiters like "--- AI Message ---".

    Args:
        messages: List of unified Message objects.

    Returns:
        Formatted trace string.
    """
    from karenina.benchmark.verification.utils.trace_formatting import messages_to_raw_trace

    return messages_to_raw_trace(messages)


def claude_tool_messages_to_trace_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert port Message objects to structured TraceMessage list.

    Produces the flat dict format matching the TypeScript TraceMessage
    interface, consistent with langchain and claude_agent_sdk adapters.

    Args:
        messages: List of unified Message objects.

    Returns:
        List of TraceMessage dicts with role, content, block_index,
        and optional tool_calls, tool_result, thinking.
    """
    result: list[dict[str, Any]] = []
    block_index = 0

    for msg in messages:
        if msg.role == Role.USER:
            # Skip user messages (consistent with other adapters)
            continue

        if msg.role == Role.ASSISTANT:
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            thinking_meta: dict[str, Any] | None = None

            for block in msg.content:
                if isinstance(block, TextContent):
                    text_parts.append(block.text)
                elif isinstance(block, ThinkingContent):
                    thinking_meta = {"thinking": block.thinking}
                    if block.signature:
                        thinking_meta["signature"] = block.signature
                elif isinstance(block, ToolUseContent):
                    tool_calls.append(
                        {
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )

            if text_parts or tool_calls or thinking_meta:
                trace_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": "\n".join(text_parts),
                    "block_index": block_index,
                }
                if tool_calls:
                    trace_msg["tool_calls"] = tool_calls
                if thinking_meta:
                    trace_msg["thinking"] = thinking_meta
                result.append(trace_msg)
                block_index += 1

        elif msg.role == Role.TOOL:
            for block in msg.content:
                if isinstance(block, ToolResultContent):
                    result.append(
                        {
                            "role": "tool",
                            "content": block.content,
                            "block_index": block_index,
                            "tool_result": {
                                "tool_use_id": block.tool_use_id,
                                "is_error": block.is_error,
                            },
                        }
                    )
                    block_index += 1

    return result
