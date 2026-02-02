"""Message types for the ports abstraction layer.

This module defines the core message types used across all LLM backends,
providing a unified representation that can be converted to/from
LangChain and Claude Agent SDK formats.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    """Message role indicating the sender.

    Inherits from str for JSON serialization compatibility and
    string comparison convenience.
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ContentType(str, Enum):
    """Content block type for structured message content.

    Inherits from str for JSON serialization compatibility and
    string comparison convenience.
    """

    TEXT = "text"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"


# =============================================================================
# Content Blocks
# =============================================================================


@dataclass(frozen=True)
class TextContent:
    """Plain text content block.

    Attributes:
        text: The text content of this block.
        type: Always ContentType.TEXT, identifies this as a text block.
    """

    text: str
    type: ContentType = field(default=ContentType.TEXT, repr=False)


@dataclass(frozen=True)
class ToolUseContent:
    """Tool invocation content block.

    Represents a request from the assistant to use a tool.

    Attributes:
        id: Unique identifier for this tool use (used to match with result).
        name: Name of the tool being invoked.
        input: Arguments/parameters passed to the tool.
        type: Always ContentType.TOOL_USE, identifies this as a tool use block.
    """

    id: str
    name: str
    input: dict[str, Any]
    type: ContentType = field(default=ContentType.TOOL_USE, repr=False)


@dataclass(frozen=True)
class ToolResultContent:
    """Tool execution result content block.

    Represents the result of a tool invocation.

    Attributes:
        tool_use_id: ID of the ToolUseContent this is a result for.
        content: The result content (typically string or serialized data).
        is_error: Whether the tool execution resulted in an error.
        type: Always ContentType.TOOL_RESULT, identifies this as a tool result block.
    """

    tool_use_id: str
    content: str
    is_error: bool = False
    type: ContentType = field(default=ContentType.TOOL_RESULT, repr=False)


@dataclass(frozen=True)
class ThinkingContent:
    """Extended thinking content block.

    Used for Claude's extended thinking feature where the model
    shows its reasoning process.

    Attributes:
        thinking: The thinking/reasoning text.
        signature: Optional cryptographic signature for verification.
        type: Always ContentType.THINKING, identifies this as a thinking block.
    """

    thinking: str
    signature: str | None = None
    type: ContentType = field(default=ContentType.THINKING, repr=False)


# Type alias for union of all content types
Content = TextContent | ToolUseContent | ToolResultContent | ThinkingContent


# =============================================================================
# Message Class
# =============================================================================


@dataclass
class Message:
    """Unified message representation for LLM conversations.

    This class provides a backend-agnostic way to represent messages
    that can be converted to/from LangChain and Claude Agent SDK formats.

    Attributes:
        role: The role of the message sender (system, user, assistant, tool).
        content: List of content blocks that make up the message.

    Example:
        >>> msg = Message.user("Hello, world!")
        >>> msg.text
        'Hello, world!'
        >>> msg = Message.assistant("Here's a tool call", tool_calls=[
        ...     ToolUseContent(id="1", name="search", input={"q": "test"})
        ... ])
        >>> len(msg.tool_calls)
        1
    """

    role: Role
    content: list[Content]

    @property
    def text(self) -> str:
        """Extract all text content as a single string.

        Joins all TextContent blocks with newlines. Returns empty string
        if no text content is present.

        Returns:
            Concatenated text from all TextContent blocks.
        """
        return "\n".join(c.text for c in self.content if isinstance(c, TextContent))

    @property
    def tool_calls(self) -> list[ToolUseContent]:
        """Extract all tool use content blocks.

        Returns:
            List of ToolUseContent items from this message.
        """
        return [c for c in self.content if isinstance(c, ToolUseContent)]

    @classmethod
    def system(cls, text: str) -> "Message":
        """Create a system message.

        Args:
            text: The system prompt text.

        Returns:
            A new Message with Role.SYSTEM and the given text.
        """
        return cls(role=Role.SYSTEM, content=[TextContent(text=text)])

    @classmethod
    def user(cls, text: str) -> "Message":
        """Create a user message.

        Args:
            text: The user's message text.

        Returns:
            A new Message with Role.USER and the given text.
        """
        return cls(role=Role.USER, content=[TextContent(text=text)])

    @classmethod
    def assistant(cls, text: str = "", tool_calls: list[ToolUseContent] | None = None) -> "Message":
        """Create an assistant message.

        Args:
            text: The assistant's response text (optional if only tool calls).
            tool_calls: List of tool invocations (optional).

        Returns:
            A new Message with Role.ASSISTANT containing text and/or tool calls.
        """
        content: list[Content] = []
        if text:
            content.append(TextContent(text=text))
        if tool_calls:
            content.extend(tool_calls)
        return cls(role=Role.ASSISTANT, content=content)

    @classmethod
    def tool_result(cls, tool_use_id: str, content: str, is_error: bool = False) -> "Message":
        """Create a tool result message.

        Args:
            tool_use_id: ID of the ToolUseContent this is a result for.
            content: The result content (typically string or serialized data).
            is_error: Whether the tool execution resulted in an error.

        Returns:
            A new Message with Role.TOOL containing the tool result.
        """
        return cls(
            role=Role.TOOL,
            content=[ToolResultContent(tool_use_id=tool_use_id, content=content, is_error=is_error)],
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this Message to a flat dict matching the TraceMessage format.

        Produces the structured format stored in VerificationResultTemplate.trace_messages
        and consumed by the frontend TraceMessage TypeScript interface.

        Returns:
            Dict with keys: role, content, block_index (always 0; caller should set),
            and optional tool_calls, tool_result, thinking.
        """
        # For tool messages, content is in ToolResultContent, not TextContent
        if self.role == Role.TOOL:
            tool_content = ""
            for block in self.content:
                if isinstance(block, ToolResultContent):
                    tool_content = block.content
                    break
            content = tool_content
        else:
            content = self.text

        result: dict[str, Any] = {
            "role": self.role.value,
            "content": content,
            "block_index": 0,  # Caller sets the actual index
        }

        # Add tool_calls for assistant messages
        tool_calls = []
        for block in self.content:
            if isinstance(block, ToolUseContent):
                tool_calls.append(
                    {
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
        if tool_calls:
            result["tool_calls"] = tool_calls

        # Add tool_result for tool messages
        for block in self.content:
            if isinstance(block, ToolResultContent):
                result["tool_result"] = {
                    "tool_use_id": block.tool_use_id,
                    "is_error": block.is_error,
                }
                break

        # Add thinking for messages with thinking blocks
        for block in self.content:
            if isinstance(block, ThinkingContent):
                thinking: dict[str, Any] = {"thinking": block.thinking}
                if block.signature:
                    thinking["signature"] = block.signature
                result["thinking"] = thinking
                break

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        """Deserialize a flat TraceMessage dict back to a Message.

        Args:
            data: Dict with role, content, and optional tool_calls/tool_result/thinking.

        Returns:
            Reconstructed Message object.
        """
        role = Role(data["role"])
        content_blocks: list[Content] = []

        # Add thinking block if present
        if "thinking" in data:
            thinking_data = data["thinking"]
            content_blocks.append(
                ThinkingContent(
                    thinking=thinking_data["thinking"],
                    signature=thinking_data.get("signature"),
                )
            )

        # Add text content if present
        text = data.get("content", "")
        if text:
            content_blocks.append(TextContent(text=text))

        # Add tool calls if present
        if "tool_calls" in data:
            for tc in data["tool_calls"]:
                content_blocks.append(
                    ToolUseContent(
                        id=tc["id"],
                        name=tc["name"],
                        input=tc.get("input", {}),
                    )
                )

        # Add tool result if present
        if "tool_result" in data:
            tr = data["tool_result"]
            content_blocks.append(
                ToolResultContent(
                    tool_use_id=tr["tool_use_id"],
                    content=text,  # Tool result content is in the main content field
                    is_error=tr.get("is_error", False),
                )
            )

        return cls(role=role, content=content_blocks)
