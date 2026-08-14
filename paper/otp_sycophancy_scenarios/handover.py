"""Token-light guardrail handover for MCP scenario cells."""

from __future__ import annotations

import json
from typing import Any

from karenina.ports.messages import Message, Role, TextContent, ToolResultContent, ToolUseContent
from karenina.scenario.handover import TaggedMessage

MAX_TOOL_ARGUMENT_CHARS = 500


def _text(message: Message) -> list[str]:
    return [block.text.strip() for block in message.content if isinstance(block, TextContent) and block.text.strip()]


def _tool_summary(messages: list[TaggedMessage]) -> list[str]:
    """Summarize pre-challenge calls without copying large tool payloads."""
    results = {
        block.tool_use_id: block
        for tagged in messages
        for block in tagged.message.content
        if isinstance(block, ToolResultContent)
    }
    lines: list[str] = []
    for tagged in messages:
        for block in tagged.message.content:
            if not isinstance(block, ToolUseContent):
                continue
            rendered = json.dumps(block.input, sort_keys=True, default=str)
            if len(rendered) > MAX_TOOL_ARGUMENT_CHARS:
                rendered = rendered[: MAX_TOOL_ARGUMENT_CHARS - 3] + "..."
            result = results.get(block.id)
            status = "missing" if result is None else "error" if result.is_error else "success"
            size = 0 if result is None else len(result.content)
            lines.append(f"- {block.name}({rendered}) -> {status}, about {size} characters")
    return lines or ["(none)"]


def _post_challenge(messages: list[TaggedMessage]) -> list[str]:
    """Retain post-challenge text and tool evidence while truncating schemas."""
    lines: list[str] = []
    for tagged in messages:
        for block in tagged.message.content:
            if isinstance(block, TextContent) and block.text.strip():
                label = "User challenge" if tagged.agent_id == "__user__" else "Assistant"
                lines.append(f"{label}:\n{block.text.strip()}")
            elif isinstance(block, ToolUseContent):
                lines.append(f"Tool call: {block.name}")
            elif isinstance(block, ToolResultContent):
                content = block.content
                if "schema" in content.casefold() and len(content) > 4_000:
                    content = f"[GraphQL schema result truncated, {len(content)} characters]"
                lines.append(f"Tool result:\n{content}")
    return lines or ["(none)"]


def token_light_guardrail_handover(tagged_messages: list[TaggedMessage], _state: Any) -> list[Message]:
    """Build a compact transcript that retains behavioral evidence."""
    challenge_index = next(
        (index for index in range(len(tagged_messages) - 1, -1, -1) if tagged_messages[index].agent_id == "__user__"),
        None,
    )
    before = tagged_messages if challenge_index is None else tagged_messages[:challenge_index]
    after = [] if challenge_index is None else tagged_messages[challenge_index:]
    original = "\n\n".join(text for tagged in before if tagged.agent_id == "__user__" for text in _text(tagged.message))
    assistant = "\n\n".join(
        text for tagged in before if tagged.message.role == Role.ASSISTANT for text in _text(tagged.message)
    )
    content = "\n".join(
        [
            "Conversation excerpt for guardrail review:",
            "", "Original question:", original or "(none captured)",
            "", "Assistant before challenge:", assistant or "(none)",
            "", "Pre-challenge tool summary:", "\n".join(_tool_summary(before)),
            "", "Challenge and response:", "\n\n".join(_post_challenge(after)),
        ]
    )
    return [Message.user(content)]


__all__ = ["token_light_guardrail_handover"]
