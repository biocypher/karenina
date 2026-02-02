"""Claude Agent SDK message converter.

This module provides conversion between unified Message types from karenina.ports
and Claude Agent SDK message types.

CRITICAL DIFFERENCE FROM LANGCHAIN:
- Claude SDK uses string prompts, not message arrays
- System prompts go in ClaudeAgentOptions.system_prompt
- Multi-turn context is managed by ClaudeSDKClient automatically

This converter handles:
- to_prompt_string(): Convert messages to prompt string for query()
- extract_system_prompt(): Extract system prompt for ClaudeAgentOptions
- from_provider(): Convert SDK response messages to unified format
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from karenina.ports import (
    Message,
    Role,
    TextContent,
    ThinkingContent,
    ToolUseContent,
)

if TYPE_CHECKING:
    from claude_agent_sdk import AssistantMessage, UserMessage


class ClaudeSDKMessageConverter:
    """Convert between unified Message and Claude Agent SDK messages.

    IMPORTANT: Claude Agent SDK works differently than LangChain.

    Input conversion:
    - query() and ClaudeSDKClient.query() take STRING prompt, not messages
    - System prompts go in ClaudeAgentOptions.system_prompt
    - Multi-turn context is managed by ClaudeSDKClient automatically

    Output conversion:
    - Convert SDK response messages to unified format for traces

    Example:
        >>> converter = ClaudeSDKMessageConverter()
        >>> messages = [Message.system("You are helpful"), Message.user("Hello")]
        >>> prompt = converter.to_prompt_string(messages)
        >>> prompt
        'Hello'
        >>> system = converter.extract_system_prompt(messages)
        >>> system
        'You are helpful'
    """

    def to_prompt_string(self, messages: list[Message]) -> str:
        """Convert unified messages to prompt string for SDK query().

        Extracts only USER role messages and joins them with double newlines.
        System prompts should be passed separately via extract_system_prompt().

        Args:
            messages: List of unified Message objects.

        Returns:
            Prompt string suitable for query() or ClaudeSDKClient.query().

        Example:
            >>> converter = ClaudeSDKMessageConverter()
            >>> msgs = [Message.user("Question 1"), Message.user("Question 2")]
            >>> converter.to_prompt_string(msgs)
            'Question 1\\n\\nQuestion 2'
        """
        user_parts = [m.text for m in messages if m.role == Role.USER]
        return "\n\n".join(user_parts) if user_parts else ""

    def extract_system_prompt(self, messages: list[Message]) -> str | None:
        """Extract system prompt for ClaudeAgentOptions.system_prompt.

        Joins all SYSTEM role messages with double newlines.

        Args:
            messages: List of unified Message objects.

        Returns:
            System prompt string, or None if no system messages present.

        Example:
            >>> converter = ClaudeSDKMessageConverter()
            >>> msgs = [Message.system("Be helpful"), Message.system("Be concise")]
            >>> converter.extract_system_prompt(msgs)
            'Be helpful\\n\\nBe concise'
        """
        system_msgs = [m for m in messages if m.role == Role.SYSTEM]
        if system_msgs:
            return "\n\n".join(m.text for m in system_msgs)
        return None

    def from_provider(self, messages: list[Any]) -> list[Message]:
        """Convert SDK messages to unified format for trace storage.

        Handles:
        - UserMessage -> Message.user()
        - AssistantMessage content blocks:
          - TextBlock -> text part of assistant message
          - ToolUseBlock -> ToolUseContent
          - ToolResultBlock -> separate Message.tool_result()
          - ThinkingBlock -> ThinkingContent on assistant message

        Args:
            messages: List of SDK message objects (UserMessage, AssistantMessage, etc.)

        Returns:
            List of unified Message objects.

        Note:
            SDK places ToolResultBlocks within AssistantMessage.content,
            not as separate messages. This converter extracts them as
            separate tool_result messages for consistency with LangChain format.
        """
        # Import SDK types at runtime to avoid import errors when SDK not installed
        try:
            from claude_agent_sdk import (
                AssistantMessage,
                UserMessage,
            )
            from claude_agent_sdk.types import (
                TextBlock,
                ThinkingBlock,
                ToolResultBlock,
                ToolUseBlock,
            )
        except ImportError:
            # If SDK not installed, return empty list
            # This allows the converter to be imported even without SDK
            return []

        result: list[Message] = []

        for msg in messages:
            if isinstance(msg, UserMessage):
                result.append(self._convert_user_message(msg))

            elif isinstance(msg, AssistantMessage):
                converted = self._convert_assistant_message(
                    msg, TextBlock, ToolUseBlock, ToolResultBlock, ThinkingBlock
                )
                result.extend(converted)

        return result

    def _convert_user_message(self, msg: UserMessage) -> Message:
        """Convert SDK UserMessage to unified Message.

        Args:
            msg: SDK UserMessage instance.

        Returns:
            Unified Message with Role.USER.
        """
        if isinstance(msg.content, str):
            return Message.user(msg.content)

        # Content is a list of blocks
        text_parts: list[str] = []
        for block in msg.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
        return Message.user("\n".join(text_parts) if text_parts else "")

    def _convert_assistant_message(
        self,
        msg: AssistantMessage,
        TextBlock: type,
        ToolUseBlock: type,
        ToolResultBlock: type,
        ThinkingBlock: type,
    ) -> list[Message]:
        """Convert SDK AssistantMessage to unified Messages.

        Returns multiple messages because ToolResultBlocks within the
        AssistantMessage are extracted as separate tool_result messages.

        Args:
            msg: SDK AssistantMessage instance.
            TextBlock: SDK TextBlock type.
            ToolUseBlock: SDK ToolUseBlock type.
            ToolResultBlock: SDK ToolResultBlock type.
            ThinkingBlock: SDK ThinkingBlock type.

        Returns:
            List of unified Messages (assistant + any tool results).
        """
        result: list[Message] = []
        text_parts: list[str] = []
        tool_calls: list[ToolUseContent] = []
        thinking_content: list[ThinkingContent] = []

        for block in msg.content:
            if isinstance(block, TextBlock):
                text_parts.append(block.text)  # type: ignore[union-attr]

            elif isinstance(block, ToolUseBlock):
                tool_calls.append(
                    ToolUseContent(
                        id=block.id,  # type: ignore[union-attr]
                        name=block.name,  # type: ignore[union-attr]
                        input=block.input if isinstance(block.input, dict) else {},  # type: ignore[union-attr]
                    )
                )

            elif isinstance(block, ToolResultBlock):
                # Tool results in SDK are within AssistantMessage
                # Extract as separate message for consistency with LangChain format
                content_str = ""
                if block.content:  # type: ignore[union-attr]
                    if isinstance(block.content, str):  # type: ignore[union-attr]
                        content_str = block.content  # type: ignore[union-attr]
                    elif isinstance(block.content, list):  # type: ignore[union-attr]
                        # Content is list of content blocks
                        content_parts = []
                        for c in block.content:  # type: ignore[union-attr]
                            if hasattr(c, "text"):
                                content_parts.append(c.text)
                        content_str = "\n".join(content_parts)
                    else:
                        content_str = str(block.content)  # type: ignore[union-attr]

                # Get is_error, ensuring it's a bool
                is_error_value = getattr(block, "is_error", False)
                if not isinstance(is_error_value, bool):
                    is_error_value = False

                result.append(
                    Message.tool_result(
                        tool_use_id=block.tool_use_id,  # type: ignore[union-attr]
                        content=content_str,
                        is_error=is_error_value,
                    )
                )

            elif isinstance(block, ThinkingBlock):
                # Extended thinking - create ThinkingContent
                thinking_content.append(
                    ThinkingContent(
                        thinking=block.thinking,  # type: ignore[union-attr]
                        signature=getattr(block, "signature", None),
                    )
                )

        # Build assistant message if there's content
        if text_parts or tool_calls or thinking_content:
            content: list[TextContent | ToolUseContent | ThinkingContent] = []

            # Add thinking first (matches SDK order)
            content.extend(thinking_content)

            # Add text
            if text_parts:
                content.append(TextContent(text="\n".join(text_parts)))

            # Add tool calls
            content.extend(tool_calls)

            result.insert(
                0,
                Message(role=Role.ASSISTANT, content=content),  # type: ignore[arg-type]
            )

        return result
