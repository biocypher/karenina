"""Claude Tool adapter using the Anthropic Python SDK directly.

This adapter provides implementations of all three ports (LLMPort, AgentPort, ParserPort)
using the Anthropic Python SDK without LangChain:

- LLMPort: Uses client.messages.create() for simple invocations
- AgentPort: Uses client.beta.messages.tool_runner() for multi-turn agent loops
- ParserPort: Uses client.beta.messages.parse() for structured output extraction

Key features:
- Native Anthropic SDK for efficient API calls
- MCP server support via HTTP/SSE transport
- Prompt caching for cost efficiency
- Structured output with Pydantic models

Example:
    >>> from karenina.schemas.config import ModelConfig
    >>> from karenina.adapters.factory import get_agent
    >>>
    >>> config = ModelConfig(
    ...     id="claude-haiku",
    ...     model_name="claude-haiku-4-5",
    ...     model_provider="anthropic",
    ...     interface="claude_tool",
    ... )
    >>> agent = get_agent(config)
    >>> result = await agent.arun(
    ...     messages=[Message.user("What genes are associated with breast cancer?")],
    ...     mcp_servers={
    ...         "open-targets": {
    ...             "type": "http",
    ...             "url": "https://mcp.platform.opentargets.org/mcp",
    ...         }
    ...     }
    ... )
"""

from karenina.adapters.claude_tool.agent import ClaudeToolAgentAdapter
from karenina.adapters.claude_tool.llm import ClaudeToolLLMAdapter
from karenina.adapters.claude_tool.parser import ClaudeToolParserAdapter

__all__ = [
    "ClaudeToolAgentAdapter",
    "ClaudeToolLLMAdapter",
    "ClaudeToolParserAdapter",
]
