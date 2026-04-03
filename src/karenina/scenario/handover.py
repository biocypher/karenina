"""Handover utilities for multi-agent scenario context routing.

Provides TaggedMessage for tracking agent identity on messages,
and format_transcript for rendering tagged message history
as a labeled text transcript.
"""

from __future__ import annotations

from dataclasses import dataclass

from karenina.ports.messages import (
    ContentType,
    Message,
    TextContent,
    ToolResultContent,
    ToolUseContent,
)


@dataclass
class TaggedMessage:
    """A Message paired with the agent identity that produced it."""

    message: Message
    agent_id: str


def format_transcript(tagged_messages: list[TaggedMessage]) -> str:
    """Format tagged messages into a labeled transcript.

    Each content block gets a label: [agent_id:content_type] for agent
    messages, [__user__] for scenario prompts. ThinkingContent blocks
    are excluded.

    Args:
        tagged_messages: The full tagged message history.

    Returns:
        A formatted transcript string. Empty string if no messages.
    """
    if not tagged_messages:
        return ""

    lines: list[str] = []
    for tm in tagged_messages:
        for block in tm.message.content:
            if block.type == ContentType.THINKING:
                continue

            label = "[__user__]" if tm.agent_id == "__user__" else f"[{tm.agent_id}:{block.type.value}]"

            if isinstance(block, TextContent):
                lines.append(f"{label} {block.text}")
            elif isinstance(block, ToolUseContent):
                args = ", ".join(f"{k}={v!r}" for k, v in block.input.items())
                lines.append(f"{label} {block.name}({args})")
            elif isinstance(block, ToolResultContent):
                lines.append(f"{label} {block.content}")

    return "\n".join(lines)
