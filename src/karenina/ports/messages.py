"""Message types for the ports abstraction layer.

This module defines the core message types used across all LLM backends,
providing a unified representation that can be converted to/from
LangChain and Claude Agent SDK formats.
"""

from enum import Enum


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
