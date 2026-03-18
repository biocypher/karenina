"""Trace extraction from LangGraph agent results.

Converts LangGraph BaseMessage lists into karenina's raw trace format:
a delimited string for backward compatibility with existing infrastructure
(regex highlighting, database storage, trace validation).
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def deep_agents_messages_to_raw_trace(
    messages: list[Any],
    include_user_messages: bool = False,
) -> str:
    """Convert LangGraph messages to raw trace string format.

    Produces delimited trace compatible with existing karenina infrastructure
    (regex highlighting, database storage, backward compatibility).

    Args:
        messages: List of LangGraph BaseMessage objects.
        include_user_messages: If True, include HumanMessage in trace.

    Returns:
        Formatted trace string with --- delimiters.
    """
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    if not messages:
        return ""

    parts: list[str] = []

    for msg in messages:
        if isinstance(msg, SystemMessage):
            continue

        if isinstance(msg, HumanMessage):
            if include_user_messages:
                parts.append(f"--- Human Message ---\n{msg.content}")
            continue

        if isinstance(msg, AIMessage):
            text = _extract_ai_text(msg)

            if text:
                parts.append(f"--- AI Message ---\n{text}")

            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_name = tc.get("name", "unknown")
                    tool_input = tc.get("args", {})
                    input_str = json.dumps(tool_input, indent=2) if tool_input else "{}"
                    parts.append(f"--- Tool Call ---\nTool: {tool_name}\nInput: {input_str}")

        elif isinstance(msg, ToolMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            parts.append(f"--- Tool Result ---\n{content}")

    return "\n\n".join(parts)


def _extract_ai_text(msg: Any) -> str:
    """Extract text content from an AIMessage.

    Handles both simple string content and list content (mixed text/tool_use blocks).

    Args:
        msg: A LangGraph AIMessage instance.

    Returns:
        Concatenated text content from the message.
    """
    if isinstance(msg.content, str):
        return msg.content

    if isinstance(msg.content, list):
        text_parts = []
        for block in msg.content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block["text"])
        return "\n".join(text_parts)

    return ""
