"""Unified trace formatting: Message objects → raw trace string.

Converts port Message objects to the configurable text format used for:
- Database storage (raw_llm_response)
- Regex-based highlighting in the frontend
- Rubric evaluation on raw text

The default configuration produces the same "--- AI Message ---" format
as the existing LangChain and Claude SDK adapters, preserving backward compat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from karenina.ports.messages import Message


@dataclass
class TraceFormatConfig:
    """Controls how trace messages are compiled into a raw trace string."""

    assistant_header: str = "--- AI Message ---"
    tool_header_template: str = "--- Tool Message (call_id: {call_id}) ---"
    user_header: str = "--- Human Message ---"
    system_header: str = "--- System Message ---"
    message_separator: str = "\n\n"
    include_tool_calls: bool = True
    include_user_messages: bool = False
    include_system_messages: bool = False
    tool_call_format: str = "inline"  # "inline" or "json"


# Singleton default config — matches legacy adapter output exactly
DEFAULT_TRACE_FORMAT = TraceFormatConfig()


def messages_to_raw_trace(
    messages: list[Message],
    config: TraceFormatConfig | None = None,
) -> str:
    """Convert port Message objects to a raw trace string.

    Adapter-agnostic: works with any list[Message] regardless of which
    adapter produced it. The default config matches the existing
    "--- AI Message ---" delimiter format.

    Args:
        messages: List of port Message objects.
        config: Optional formatting configuration. Uses DEFAULT_TRACE_FORMAT
            if not provided.

    Returns:
        Formatted trace string with message blocks separated by the
        configured separator.
    """
    from karenina.ports.messages import (
        Role,
        TextContent,
        ThinkingContent,
        ToolResultContent,
        ToolUseContent,
    )

    cfg = config or DEFAULT_TRACE_FORMAT
    parts: list[str] = []

    for msg in messages:
        if msg.role == Role.SYSTEM:
            if not cfg.include_system_messages:
                continue
            text = msg.text
            if text:
                parts.append(f"{cfg.system_header}\n{text}")

        elif msg.role == Role.USER:
            if not cfg.include_user_messages:
                continue
            text = msg.text
            if text:
                parts.append(f"{cfg.user_header}\n{text}")

        elif msg.role == Role.ASSISTANT:
            content_parts: list[str] = []

            # Thinking blocks (if any)
            for block in msg.content:
                if isinstance(block, ThinkingContent):
                    parts.append(f"--- Thinking ---\n{block.thinking}")

            # Text content
            text_parts = [b.text for b in msg.content if isinstance(b, TextContent)]
            if text_parts:
                content_parts.append("\n".join(text_parts))

            # Tool calls
            if cfg.include_tool_calls:
                tool_calls = [b for b in msg.content if isinstance(b, ToolUseContent)]
                if tool_calls:
                    if cfg.tool_call_format == "json":
                        import json

                        tc_lines = ["\nTool Calls:"]
                        for tc in tool_calls:
                            tc_lines.append(f"  {tc.name} (call_{tc.id})")
                            tc_lines.append(f"   Call ID: {tc.id}")
                            if tc.input:
                                tc_lines.append(f"   Args: {json.dumps(tc.input, indent=2)}")
                        content_parts.append("\n".join(tc_lines))
                    else:
                        # inline format — matches LangChain adapter output
                        tc_lines = ["\nTool Calls:"]
                        for tc in tool_calls:
                            tc_lines.append(f"  {tc.name} (call_{tc.id})")
                            tc_lines.append(f"   Call ID: {tc.id}")
                            if tc.input:
                                tc_lines.append(f"   Args: {tc.input}")
                        content_parts.append("\n".join(tc_lines))

            if content_parts:
                parts.append(f"{cfg.assistant_header}\n{''.join(content_parts)}")

        elif msg.role == Role.TOOL:
            for block in msg.content:
                if isinstance(block, ToolResultContent):
                    header = cfg.tool_header_template.format(call_id=block.tool_use_id)
                    content = block.content
                    if block.is_error:
                        content = f"[ERROR] {content}"
                    parts.append(f"{header}\n{content}")

    return cfg.message_separator.join(parts)
