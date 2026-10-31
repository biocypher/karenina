"""Prompt and trace conversion for the Oh My Pi ACP adapter."""

from __future__ import annotations

import json

from karenina.ports import Message, Role, ThinkingContent, ToolResultContent


def build_omp_prompt(messages: list[Message], system_prompt: str | None = None) -> str:
    """Serialize Karenina messages into the single text prompt ACP v1 accepts.

    ACP sessions maintain history after creation, but Karenina passes a complete
    message list into each independent adapter run. Role labels make that history
    explicit while keeping system instructions distinguishable from user content.
    """
    system_parts = [message.text for message in messages if message.role == Role.SYSTEM and message.text]
    if system_prompt:
        system_parts.append(system_prompt)

    conversation: list[str] = []
    for message in messages:
        if message.role == Role.SYSTEM:
            continue
        if message.role == Role.USER:
            conversation.append(f"User:\n{message.text}")
            continue
        if message.role == Role.ASSISTANT:
            parts: list[str] = []
            if message.text:
                parts.append(message.text)
            for call in message.tool_calls:
                parts.append(f"[Tool call {call.name} ({call.id}): {json.dumps(call.input, default=str)}]")
            if parts:
                conversation.append("Assistant:\n" + "\n".join(parts))
            continue
        if message.role == Role.TOOL:
            for block in message.content:
                if isinstance(block, ToolResultContent):
                    state = "error" if block.is_error else "result"
                    conversation.append(f"Tool {state} ({block.tool_use_id}):\n{block.content}")

    sections: list[str] = []
    if system_parts:
        sections.append("System instructions:\n" + "\n\n".join(system_parts))
    if conversation:
        sections.append("Conversation:\n\n" + "\n\n".join(conversation))
    return "\n\n".join(sections).strip()


def omp_messages_to_raw_trace(messages: list[Message]) -> str:
    """Render unified messages in Karenina's canonical raw-trace format."""
    from karenina.benchmark.verification.utils.trace_formatting import messages_to_raw_trace

    return str(messages_to_raw_trace(messages))


def extract_final_response(messages: list[Message]) -> str | None:
    """Return the last visible assistant text from a collected ACP trace."""
    for message in reversed(messages):
        if message.role != Role.ASSISTANT:
            continue
        text_value = message.text
        text = str(text_value).strip() if text_value else ""
        if text:
            return text
        thinking = [block.thinking for block in message.content if isinstance(block, ThinkingContent)]
        if thinking:
            continue
    return None
