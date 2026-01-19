"""Ports module - abstractions for LLM backends.

This module defines Protocol classes (ports) that provide a unified interface
for interacting with different LLM backends (LangChain, Claude Agent SDK, etc.).

The ports follow the Ports and Adapters (hexagonal) architecture pattern,
allowing the core application logic to remain independent of specific
LLM provider implementations.
"""

from karenina.ports.messages import (
    Content,
    ContentType,
    Message,
    Role,
    TextContent,
    ThinkingContent,
    ToolResultContent,
    ToolUseContent,
)
from karenina.ports.usage import UsageMetadata

__all__ = [
    # Enums
    "Role",
    "ContentType",
    # Content types
    "Content",
    "TextContent",
    "ToolUseContent",
    "ToolResultContent",
    "ThinkingContent",
    # Message class
    "Message",
    # Usage tracking
    "UsageMetadata",
]
