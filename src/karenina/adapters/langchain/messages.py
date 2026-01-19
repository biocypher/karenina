"""LangChain message converter.

This module provides conversion between unified Message types from karenina.ports
and LangChain message types from langchain_core.messages.
"""

from __future__ import annotations

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from karenina.ports import (
    Message,
    Role,
    ToolResultContent,
    ToolUseContent,
)


class LangChainMessageConverter:
    """Convert between unified Message and LangChain messages.

    This converter handles bidirectional translation between karenina's
    unified Message format and LangChain's BaseMessage types.

    Example:
        >>> converter = LangChainMessageConverter()
        >>> messages = [Message.user("Hello")]
        >>> lc_messages = converter.to_provider(messages)
        >>> isinstance(lc_messages[0], HumanMessage)
        True
        >>> roundtrip = converter.from_provider(lc_messages)
        >>> roundtrip[0].text == "Hello"
        True
    """

    def to_provider(self, messages: list[Message]) -> list[BaseMessage]:
        """Convert unified messages to LangChain format.

        Args:
            messages: List of unified Message objects.

        Returns:
            List of LangChain BaseMessage objects.

        Mapping:
            - Role.SYSTEM -> SystemMessage
            - Role.USER -> HumanMessage
            - Role.ASSISTANT -> AIMessage (with tool_calls if present)
            - Role.TOOL -> ToolMessage (one per ToolResultContent)
        """
        result: list[BaseMessage] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                result.append(SystemMessage(content=msg.text))

            elif msg.role == Role.USER:
                result.append(HumanMessage(content=msg.text))

            elif msg.role == Role.ASSISTANT:
                # Build tool_calls list from ToolUseContent blocks
                tool_calls = None
                if msg.tool_calls:
                    tool_calls = [{"id": tc.id, "name": tc.name, "args": tc.input} for tc in msg.tool_calls]

                result.append(
                    AIMessage(
                        content=msg.text,
                        tool_calls=tool_calls if tool_calls else [],
                    )
                )

            elif msg.role == Role.TOOL:
                # Create one ToolMessage per ToolResultContent
                for content in msg.content:
                    if isinstance(content, ToolResultContent):
                        result.append(
                            ToolMessage(
                                content=content.content,
                                tool_call_id=content.tool_use_id,
                            )
                        )

        return result

    def from_provider(self, messages: list[BaseMessage]) -> list[Message]:
        """Convert LangChain messages to unified format.

        Args:
            messages: List of LangChain BaseMessage objects.

        Returns:
            List of unified Message objects.

        Mapping:
            - SystemMessage -> Role.SYSTEM
            - HumanMessage -> Role.USER
            - AIMessage -> Role.ASSISTANT (extracts tool_calls if present)
            - ToolMessage -> Role.TOOL
        """
        result: list[Message] = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                result.append(Message.system(str(msg.content)))

            elif isinstance(msg, HumanMessage):
                result.append(Message.user(str(msg.content)))

            elif isinstance(msg, AIMessage):
                # Extract tool calls if present
                tool_calls: list[ToolUseContent] = []
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        # LangChain tool_calls can be dicts or objects
                        if isinstance(tc, dict):
                            tool_calls.append(
                                ToolUseContent(
                                    id=tc.get("id") or "",
                                    name=tc.get("name") or "",
                                    input=tc.get("args") or {},
                                )
                            )
                        else:
                            # Handle object-style tool calls
                            tool_calls.append(
                                ToolUseContent(
                                    id=getattr(tc, "id", ""),
                                    name=getattr(tc, "name", ""),
                                    input=getattr(tc, "args", {}),
                                )
                            )

                result.append(
                    Message.assistant(
                        text=str(msg.content) if msg.content else "",
                        tool_calls=tool_calls if tool_calls else None,
                    )
                )

            elif isinstance(msg, ToolMessage):
                result.append(
                    Message.tool_result(
                        tool_use_id=msg.tool_call_id,
                        content=str(msg.content),
                    )
                )

        return result
