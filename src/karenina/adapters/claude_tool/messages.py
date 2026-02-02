"""Message conversion utilities between unified Message and Anthropic SDK formats.

This module provides functions to convert between karenina's unified Message format
and the Anthropic Python SDK's message format (used by client.messages.create and
client.beta.messages.tool_runner).

The Anthropic SDK expects messages in a specific format:
- System messages are passed separately (not in the messages array)
- Messages have a role ("user" or "assistant") and content array
- Content blocks can be text, tool_use, or tool_result
- Tool results are sent in a "user" role message with tool_result content blocks
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from karenina.ports.messages import (
    Content,
    Message,
    Role,
    TextContent,
    ThinkingContent,
    ToolResultContent,
    ToolUseContent,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def extract_system_prompt(messages: list[Message]) -> str | None:
    """Extract system prompt from messages.

    The Anthropic SDK requires system prompts to be passed separately.

    Args:
        messages: List of unified Message objects.

    Returns:
        The system prompt text if present, None otherwise.
    """
    for msg in messages:
        if msg.role == Role.SYSTEM:
            return msg.text
    return None


def convert_to_anthropic(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert unified Messages to Anthropic SDK message format.

    Converts karenina's unified Message objects to the format expected by
    Anthropic's Python SDK. System messages are excluded (handle separately).

    Args:
        messages: List of unified Message objects.

    Returns:
        List of dicts in Anthropic SDK message format.

    Example:
        >>> msgs = [Message.user("Hello!")]
        >>> anthropic_msgs = convert_to_anthropic(msgs)
        >>> # [{"role": "user", "content": [{"type": "text", "text": "Hello!"}]}]
    """
    result: list[dict[str, Any]] = []
    pending_tool_results: list[dict[str, Any]] = []

    for msg in messages:
        # Skip system messages - handled separately
        if msg.role == Role.SYSTEM:
            continue

        # Handle tool results specially - they go in user messages
        if msg.role == Role.TOOL:
            for block in msg.content:
                if isinstance(block, ToolResultContent):
                    pending_tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.tool_use_id,
                            "content": block.content,
                            **({"is_error": True} if block.is_error else {}),
                        }
                    )
            continue

        # Build content array for this message
        content: list[dict[str, Any]] = []
        for block in msg.content:
            if isinstance(block, TextContent):
                content.append({"type": "text", "text": block.text})
            elif isinstance(block, ToolUseContent):
                content.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
            elif isinstance(block, ToolResultContent):
                # Shouldn't normally happen but handle it
                content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.tool_use_id,
                        "content": block.content,
                        **({"is_error": True} if block.is_error else {}),
                    }
                )
            elif isinstance(block, ThinkingContent):
                # Anthropic uses "thinking" type for extended thinking
                content.append(
                    {
                        "type": "thinking",
                        "thinking": block.thinking,
                    }
                )

        # If this is a user message and we have pending tool results,
        # prepend them to this message's content
        if msg.role == Role.USER and pending_tool_results:
            content = pending_tool_results + content
            pending_tool_results = []

        # Map role to Anthropic format
        role = "assistant" if msg.role == Role.ASSISTANT else "user"

        if content:
            result.append({"role": role, "content": content})

    # If we still have pending tool results, add them as a user message
    if pending_tool_results:
        result.append({"role": "user", "content": pending_tool_results})

    return result


def convert_from_anthropic_message(response: Any) -> Message:
    """Convert a single Anthropic response message to unified Message format.

    Args:
        response: An Anthropic API response message object with content attribute.

    Returns:
        A unified Message object.
    """
    content: list[Content] = []

    # Handle response.content which is a list of content blocks
    response_content = getattr(response, "content", [])
    if response_content is None:
        response_content = []

    for block in response_content:
        block_type = getattr(block, "type", None)

        if block_type == "text":
            text_value = getattr(block, "text", "")
            content.append(TextContent(text=text_value))
        elif block_type == "tool_use":
            content.append(
                ToolUseContent(
                    id=getattr(block, "id", ""),
                    name=getattr(block, "name", ""),
                    input=getattr(block, "input", {}),
                )
            )
        elif block_type == "thinking":
            content.append(
                ThinkingContent(
                    thinking=getattr(block, "thinking", ""),
                    signature=getattr(block, "signature", None),
                )
            )
        elif block_type == "tool_result":
            # Shouldn't be in assistant messages but handle it
            content.append(
                ToolResultContent(
                    tool_use_id=getattr(block, "tool_use_id", ""),
                    content=str(getattr(block, "content", "")),
                    is_error=getattr(block, "is_error", False),
                )
            )

    # Determine role from response
    response_role = getattr(response, "role", "assistant")
    role = Role.ASSISTANT if response_role == "assistant" else Role.USER

    return Message(role=role, content=content)


def convert_from_anthropic_messages(responses: list[Any]) -> list[Message]:
    """Convert a list of Anthropic response messages to unified Message format.

    Args:
        responses: List of Anthropic API response message objects.

    Returns:
        List of unified Message objects.
    """
    return [convert_from_anthropic_message(resp) for resp in responses]


def build_system_with_cache(system_prompt: str | None) -> list[dict[str, Any]] | None:
    """Build system prompt with prompt caching enabled.

    Applies Anthropic's ephemeral cache control to the system prompt for
    efficient reuse across multiple API calls within a 5-minute window.

    Args:
        system_prompt: The system prompt text, or None.

    Returns:
        List with a single cached text block, or None if no system prompt.
    """
    if not system_prompt:
        return None

    return [
        {
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }
    ]
