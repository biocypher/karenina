"""Message conversion between karenina's unified Message and LangGraph types.

Handles bidirectional conversion:
- To Deep Agents: karenina Message -> prompt string (system extracted separately)
- From Deep Agents: LangGraph BaseMessage -> karenina Message
"""

from __future__ import annotations

import logging
from typing import Any

from karenina.ports import Content, Message, Role, TextContent, ToolResultContent, ToolUseContent

logger = logging.getLogger(__name__)


class DeepAgentsMessageConverter:
    """Convert between karenina's unified Message and LangGraph message types.

    Deep Agents accepts messages as dicts with role/content keys for invocation,
    and returns LangGraph BaseMessage subclasses in results.
    """

    def to_prompt_string(self, messages: list[Message]) -> str:
        """Convert user/assistant messages to a prompt string.

        System messages are excluded (use extract_system_prompt instead).

        Args:
            messages: List of karenina Message objects.

        Returns:
            Concatenated prompt string from non-system messages.
        """
        parts = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                continue
            text = msg.text
            if text:
                parts.append(text)
        return "\n\n".join(parts)

    def extract_system_prompt(self, messages: list[Message]) -> str | None:
        """Extract system prompt from messages.

        Args:
            messages: List of karenina Message objects.

        Returns:
            Combined system prompt text, or None if no system messages.
        """
        system_parts = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                text = msg.text
                if text:
                    system_parts.append(text)
        return "\n\n".join(system_parts) if system_parts else None

    def to_langchain_messages(self, messages: list[Message]) -> list[dict[str, str]]:
        """Convert karenina messages to LangGraph-compatible dicts.

        Args:
            messages: List of karenina Message objects.

        Returns:
            List of message dicts with role and content keys.
        """
        result = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                continue
            role_map = {
                Role.USER: "user",
                Role.ASSISTANT: "assistant",
                Role.TOOL: "tool",
            }
            role = role_map.get(msg.role, "user")
            result.append({"role": role, "content": msg.text or ""})
        return result

    def from_provider(self, lc_messages: list[Any]) -> list[Message]:
        """Convert LangGraph BaseMessage list to karenina Messages.

        Args:
            lc_messages: List of LangGraph message objects.

        Returns:
            List of karenina Message objects.
        """
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

        result: list[Message] = []

        for msg in lc_messages:
            if isinstance(msg, SystemMessage):
                result.append(Message.system(msg.content if isinstance(msg.content, str) else str(msg.content)))

            elif isinstance(msg, HumanMessage):
                result.append(Message.user(msg.content if isinstance(msg.content, str) else str(msg.content)))

            elif isinstance(msg, AIMessage):
                result.append(self._convert_ai_message(msg))

            elif isinstance(msg, ToolMessage):
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                tool_call_id = getattr(msg, "tool_call_id", "") or ""
                result.append(
                    Message(
                        role=Role.TOOL,
                        content=[ToolResultContent(tool_use_id=tool_call_id, content=content)],
                    )
                )

            else:
                logger.debug("Skipping unknown message type: %s", type(msg).__name__)

        return result

    def _convert_ai_message(self, msg: Any) -> Message:
        """Convert a LangGraph AIMessage to a karenina Message.

        Handles text content, structured content blocks, and tool calls.

        Args:
            msg: LangGraph AIMessage instance.

        Returns:
            Karenina Message with Role.ASSISTANT.
        """
        content_blocks: list[Content] = []

        if isinstance(msg.content, str) and msg.content:
            content_blocks.append(TextContent(text=msg.content))
        elif isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, str):
                    content_blocks.append(TextContent(text=block))
                elif isinstance(block, dict):
                    if block.get("type") == "text":
                        content_blocks.append(TextContent(text=block["text"]))
                    elif block.get("type") == "tool_use":
                        content_blocks.append(
                            ToolUseContent(
                                id=block.get("id", ""),
                                name=block.get("name", ""),
                                input=block.get("input", {}),
                            )
                        )

        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                content_blocks.append(
                    ToolUseContent(
                        id=tc.get("id", ""),
                        name=tc.get("name", ""),
                        input=tc.get("args", {}),
                    )
                )

        if content_blocks:
            return Message(role=Role.ASSISTANT, content=content_blocks)
        return Message.assistant("")
