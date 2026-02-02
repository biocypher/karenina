"""Ports module - abstractions for LLM backends.

This module defines Protocol classes (ports) that provide a unified interface
for interacting with different LLM backends (LangChain, Claude Agent SDK, etc.).

The ports follow the Ports and Adapters (hexagonal) architecture pattern,
allowing the core application logic to remain independent of specific
LLM provider implementations.
"""

from karenina.ports.adapter_instruction import (
    AdapterInstruction,
    AdapterInstructionRegistry,
    InstructionFactory,
)
from karenina.ports.agent import (
    AgentConfig,
    AgentPort,
    AgentResult,
    MCPHttpServerConfig,
    MCPServerConfig,
    MCPStdioServerConfig,
    Tool,
)
from karenina.ports.capabilities import PortCapabilities
from karenina.ports.errors import (
    AdapterUnavailableError,
    AgentExecutionError,
    AgentResponseError,
    AgentTimeoutError,
    ParseError,
    PortError,
)
from karenina.ports.llm import LLMPort, LLMResponse
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
from karenina.ports.parser import ParsePortResult, ParserPort
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
    # LLM Port
    "LLMPort",
    "LLMResponse",
    # Parser Port
    "ParsePortResult",
    "ParserPort",
    # Agent configuration types
    "Tool",
    "MCPServerConfig",
    "MCPStdioServerConfig",
    "MCPHttpServerConfig",
    "AgentConfig",
    # Agent port and result
    "AgentPort",
    "AgentResult",
    # Port capabilities
    "PortCapabilities",
    # Adapter instructions
    "AdapterInstruction",
    "AdapterInstructionRegistry",
    "InstructionFactory",
    # Error types
    "PortError",
    "AdapterUnavailableError",
    "AgentExecutionError",
    "AgentTimeoutError",
    "AgentResponseError",
    "ParseError",
]
